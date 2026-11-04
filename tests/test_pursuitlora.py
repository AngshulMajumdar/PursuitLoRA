"""pytest tests for PursuitLoRA (CPU, tiny random models)."""
import functools
import math
import os

import pytest
import torch
import torch.nn.functional as F
import transformers as T

from pursuitlora import PursuitLoRAConfig, PursuitLoRAModel, get_pursuitlora_model
from pursuitlora.model import greedy_ols, greedy_sigma

V = 97


def tiny(arch):
    common = dict(vocab_size=V, max_position_embeddings=64)
    if arch == "gpt2":
        return T.GPT2LMHeadModel(T.GPT2Config(n_layer=3, n_embd=32, n_head=4, vocab_size=V, n_positions=64))
    if arch == "llama":
        return T.LlamaForCausalLM(T.LlamaConfig(num_hidden_layers=3, hidden_size=32, intermediate_size=48,
                                                num_attention_heads=4, num_key_value_heads=2, **common))
    if arch == "qwen2":
        return T.Qwen2ForCausalLM(T.Qwen2Config(num_hidden_layers=3, hidden_size=32, intermediate_size=48,
                                                num_attention_heads=4, num_key_value_heads=2, **common))
    if arch == "mistral":
        return T.MistralForCausalLM(T.MistralConfig(num_hidden_layers=3, hidden_size=32, intermediate_size=48,
                                                    num_attention_heads=4, num_key_value_heads=2, **common))
    if arch == "phi":
        return T.PhiForCausalLM(T.PhiConfig(num_hidden_layers=3, hidden_size=32, intermediate_size=48,
                                            num_attention_heads=4, **common))
    if arch == "gemma":
        return T.GemmaForCausalLM(T.GemmaConfig(num_hidden_layers=3, hidden_size=32, intermediate_size=48,
                                                num_attention_heads=4, num_key_value_heads=2, head_dim=8, **common))
    raise ValueError(arch)


ARCHS = ["gpt2", "llama", "qwen2", "mistral", "phi", "gemma"]


def batches(n=2, b=3, t=17, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, V, (b, t), generator=g) for _ in range(n)]


def lm_loss(model, x):
    logits = model(x[:, :-1]).logits
    return F.cross_entropy(logits.float().flatten(0, 1), x[:, 1:].flatten())


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def f64(fn):
    @functools.wraps(fn)
    def wrapped(*a, **k):
        torch.set_default_dtype(torch.float64)
        try:
            return fn(*a, **k)
        finally:
            torch.set_default_dtype(torch.float32)
    return wrapped


# ---------------------------------------------------------------- selection rules
def test_greedy_sigma_matches_brute_force():
    g = torch.Generator().manual_seed(1)
    for _ in range(50):
        n = 7
        sig = torch.rand(n, generator=g).tolist()
        cost = torch.randint(1, 6, (n,), generator=g).tolist()
        budget = int(torch.randint(3, 15, (1,), generator=g))
        chosen, used = [], 0                       # reference: walk in sigma order, skip what does not fit
        for j in sorted(range(n), key=lambda j: -sig[j]):
            if used + cost[j] <= budget:
                chosen.append(j)
                used += cost[j]
        assert greedy_sigma(sig, cost, budget) == chosen
        assert sum(cost[j] for j in chosen) <= budget


def test_greedy_ols_matches_brute_force_least_squares():
    g = torch.Generator().manual_seed(2)
    for _ in range(30):
        P = torch.randn(40, 8, generator=g, dtype=torch.float64)
        cost = torch.randint(1, 4, (8,), generator=g).tolist()
        budget = 7
        y = torch.ones(40, dtype=torch.float64)
        got = greedy_ols(P, cost, budget)
        S, used = [], 0
        for j in got:                              # each pick must be the best least-squares reduction
            best, bj = None, None
            for c in range(8):
                if c in S or used + cost[c] > budget:
                    continue
                A = P[:, S + [c]]
                res = float(((y - A @ torch.linalg.lstsq(A, y).solution) ** 2).sum())
                if best is None or res < best - 1e-9:
                    best, bj = res, c
            assert j == bj
            S.append(j)
            used += cost[j]
        assert all(used + cost[c] > budget for c in range(8) if c not in S)   # stops only when nothing fits


# ---------------------------------------------------------------- model wrapping
@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("rule", ["sigma", "ols"])
def test_unchanged_before_and_after_init(arch, rule):
    base = tiny(arch).eval()
    x = batches(1)[0]
    ref = base(x).logits.detach().clone()
    model = get_pursuitlora_model(base, PursuitLoRAConfig(budget=600, rule=rule))
    assert torch.equal(model(x).logits, ref)
    model.init_atoms(batches(2), verbose=False)
    assert model.is_initialized
    assert float((model(x).logits.detach() - ref).abs().max()) < 1e-6        # coefficients are 0 at init
    t, _ = model.get_nb_trainable_parameters()
    assert 0 < t <= 600
    assert t == sum(model.atom_cost(model.slots[int(k)]) * a.alpha_raw.numel() for k, a in model.atoms.items())


@pytest.mark.parametrize("arch", ARCHS)
@f64
def test_initial_gradient_is_minus_scale_sigma(arch):
    """dL/d alpha_raw at init = -alpha_scale * sigma for every atom (atoms are steepest rank-one descents)."""
    base = tiny(arch).eval()
    model = get_pursuitlora_model(base, PursuitLoRAConfig(budget=800))
    xs = batches(2)
    rep = model.init_atoms(xs, verbose=False)
    loss = sum(lm_loss(model, x) for x in xs) / len(xs)
    keys = list(model.atoms)
    grads = torch.autograd.grad(loss, [model.atoms[k].alpha_raw for k in keys])
    for k, gr in zip(keys, grads):
        sig = torch.tensor([r["sigma"] for r in rep if r["slot"] == int(k)], dtype=gr.dtype)
        assert torch.allclose(gr, -model.peft_config.alpha_scale * sig, rtol=1e-6, atol=1e-10)


@f64
def test_sigma_rule_picks_globally_largest_singular_values():
    base = tiny("llama").eval()
    model = get_pursuitlora_model(base, PursuitLoRAConfig(budget=1000))
    xs = batches(2)
    rep = model.init_atoms(xs, verbose=False)
    # recompute every matrix's singular values independently
    negG, _ = model._probe(xs, None)
    allsv = sorted(((float(s), i) for i, G in enumerate(negG) for s in torch.linalg.svdvals(G)[:8]), reverse=True)
    got = [r["sigma"] for r in rep]
    assert got == sorted(got, reverse=True)
    assert abs(got[0] - allsv[0][0]) < 1e-9
    assert sum(model.atom_cost(model.slots[r["slot"]]) for r in rep) <= 1000


@f64
def test_ols_columns_sum_to_descent_rate():
    """sum_t psi_t(atom) = u^T G v, so the columns are a per-position split of -sigma."""
    base = tiny("gpt2").eval()
    cfg = PursuitLoRAConfig(budget=500, rule="ols", ols_batches=2)
    model = get_pursuitlora_model(base, cfg)
    xs = batches(2)
    negG, acts = model._probe(xs, None, capture_batches=2)
    cands = model._candidates(negG)
    P = model._ols_columns(cands, acts)
    sums = P.sum(0)
    sig = torch.tensor([c[0] for c in cands], dtype=sums.dtype)
    assert torch.allclose(sums, -len(xs) * sig, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("arch", ARCHS)
@f64
def test_merge_equals_hooks_and_save_load_roundtrip(arch, tmp_path):
    base = tiny(arch).eval()
    state = {k: v.clone() for k, v in base.state_dict().items()}
    x = batches(1, seed=5)[0]
    ref = base(x).logits.detach().clone()
    model = get_pursuitlora_model(base, PursuitLoRAConfig(budget=900))
    model.init_atoms(batches(2), verbose=False)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.3 * torch.randn_like(p))
    hooked = model(x).logits.detach().clone()
    assert float((hooked - ref).abs().max()) > 1e-3
    model.save_pretrained(tmp_path)
    base2 = tiny(arch).eval()
    base2.load_state_dict(state)
    re = PursuitLoRAModel.from_pretrained(base2, str(tmp_path))
    assert re.describe() == model.describe()
    assert float((re(x).logits - hooked).abs().max()) < 1e-10
    merged = model.merge_and_unload()
    assert float((merged(x).logits - hooked).abs().max()) < 1e-10


def test_gpt2_fused_qkv_is_split():
    m = get_pursuitlora_model(tiny("gpt2"), PursuitLoRAConfig())
    roles = [s.role for s in m.slots if s.layer == 0]
    assert roles[:3] == ["attn.c_attn.q", "attn.c_attn.k", "attn.c_attn.v"]
    assert all(s.shape == (32, 32) for s in m.slots if s.role.startswith("attn.c_attn"))
    assert all(not p.requires_grad for p in m.base_model.parameters())


def test_target_modules_subset():
    m = get_pursuitlora_model(tiny("llama"), PursuitLoRAConfig(target_modules=["q_proj", "v_proj"], budget=300))
    assert sorted({s.role for s in m.slots}) == ["self_attn.q_proj", "self_attn.v_proj"]
    m.init_atoms(batches(1), verbose=False)
    assert all(m.slots[int(k)].role in ("self_attn.q_proj", "self_attn.v_proj") for k in m.atoms)


def test_budget_too_small_and_bad_rule():
    m = get_pursuitlora_model(tiny("llama"), PursuitLoRAConfig(budget=10))
    with pytest.raises(ValueError):
        m.init_atoms(batches(1), verbose=False)
    with pytest.raises(ValueError):
        PursuitLoRAConfig(rule="nope")


def test_wrong_architecture_rejected(tmp_path):
    m = get_pursuitlora_model(tiny("llama"), PursuitLoRAConfig(budget=300))
    m.init_atoms(batches(1), verbose=False)
    m.save_pretrained(tmp_path)
    with pytest.raises(ValueError):
        PursuitLoRAModel.from_pretrained(tiny("gpt2"), str(tmp_path))


def test_dict_batches_with_attention_mask_and_labels():
    m = get_pursuitlora_model(tiny("llama").eval(), PursuitLoRAConfig(budget=300))
    x = batches(1)[0]
    m.init_atoms([{"input_ids": x, "attention_mask": torch.ones_like(x), "labels": x}], verbose=False)
    assert m.is_initialized


def test_bf16_base():
    base = tiny("llama").to(torch.bfloat16).eval()
    m = get_pursuitlora_model(base, PursuitLoRAConfig(budget=600))
    m.init_atoms(batches(2), verbose=False)
    with torch.no_grad():
        for a in m.atoms.values():
            a.alpha_raw.add_(0.01)
    out = m(batches(1)[0]).logits
    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()
    assert all(p.dtype == torch.float32 for p in m.parameters() if p.requires_grad)


def test_one_adamw_lr_trains():
    m = get_pursuitlora_model(tiny("gpt2").eval(), PursuitLoRAConfig(budget=900))
    xs = batches(2, b=4, t=33)
    m.init_atoms(xs, verbose=False)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=3e-3, weight_decay=0.0)
    l0 = lm_loss(m, xs[0]).item()
    for _ in range(10):
        opt.zero_grad()
        lm_loss(m, xs[0]).backward()
        opt.step()
    assert lm_loss(m, xs[0]).item() < l0


def test_generate_works():
    m = get_pursuitlora_model(tiny("llama").eval(), PursuitLoRAConfig(budget=300))
    m.init_atoms(batches(1), verbose=False)
    out = m.generate(batches(1)[0][:1, :5], max_new_tokens=4, do_sample=False)
    assert out.shape == (1, 9)


def test_hf_trainer_integration(tmp_path):
    from pursuitlora import PursuitLoRATrainer

    m = get_pursuitlora_model(tiny("llama"), PursuitLoRAConfig(budget=600))
    data = [{"input_ids": x, "attention_mask": torch.ones_like(x), "labels": x.clone()}
            for x in torch.randint(0, V, (16, 24)).unbind(0)]
    m.init_atoms(torch.utils.data.DataLoader(data, batch_size=4), verbose=False)

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(data)

        def __getitem__(self, i):
            return data[i]

    args = T.TrainingArguments(output_dir=str(tmp_path / "out"), per_device_train_batch_size=4, max_steps=6,
                               learning_rate=3e-3, warmup_steps=1, logging_steps=1, save_steps=3,
                               report_to=[], use_cpu=True)
    tr = PursuitLoRATrainer(model=m, args=args, train_dataset=DS())
    hist = tr.train()
    assert math.isfinite(hist.training_loss)
    tr.save_model(str(tmp_path / "final"))
    re = PursuitLoRAModel.from_pretrained(tiny("llama"), str(tmp_path / "final"))
    assert re.is_initialized
    assert os.path.exists(tmp_path / "out" / "checkpoint-3" / "adapter_model.safetensors")
