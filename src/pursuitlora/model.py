# Copyright 2026 Angshul Majumdar.
"""PursuitLoRA: LoRA-shaped adapters whose rank-one atoms are chosen by gradient pursuit.

``init_atoms`` runs one gradient probe. For every target matrix the candidate atoms are the leading
singular pairs of its negative gradient (the descent rate of an atom is its singular value). Atoms
are then chosen across ALL target matrices under a parameter budget, so the rank of each matrix's
update is decided by the gradient instead of being fixed. Coefficients start at zero, so the model
is unchanged until training starts; afterwards all coefficients and vectors are trained normally.
"""
import functools
import os
from collections import defaultdict
from contextlib import contextmanager

import torch
import torch.nn as nn

from . import __version__
from .config import CONFIG_NAME, WEIGHTS_NAME, PursuitLoRAConfig, _resolve
from .layers import attach_hooks, find_slots


@contextmanager
def _capture(slots):
    """Record each target module's input and output-gradient (flattened to (N, d)) during the block."""
    mods = list({id(s.module): s.module for s in slots}.values())
    store = {id(m): {"x": [], "d": []} for m in mods}
    handles = []
    for m in mods:
        def fhook(mod, inp, out):
            store[id(mod)]["x"].append(inp[0].detach().reshape(-1, inp[0].shape[-1]))
            out.register_hook(lambda g, mod=mod: store[id(mod)]["d"].append(g.detach().reshape(-1, g.shape[-1])))
        handles.append(m.register_forward_hook(fhook))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


class SlotAtoms(nn.Module):
    """All atoms living in one matrix: coefficients (k,), input vectors (k, d_in), output vectors (k, d_out)."""

    def __init__(self, k, d_in, d_out, device, dtype=torch.float32):
        super().__init__()
        self.alpha_raw = nn.Parameter(torch.zeros(k, device=device, dtype=dtype))
        self.u = nn.Parameter(torch.zeros(k, d_in, device=device, dtype=dtype))
        self.v = nn.Parameter(torch.zeros(k, d_out, device=device, dtype=dtype))


class PursuitLoRAModel(nn.Module):
    """Wraps a Hugging Face causal LM. Use :func:`get_pursuitlora_model` to create one."""

    def __init__(self, base_model, config: PursuitLoRAConfig):
        super().__init__()
        self.base_model = base_model
        self.peft_config = config
        config.pursuitlora_version = __version__
        if config.base_model_name_or_path is None:
            config.base_model_name_or_path = getattr(getattr(base_model, "config", None), "_name_or_path", None)
        for p in base_model.parameters():
            p.requires_grad_(False)
        self.slots = find_slots(base_model, config.target_modules, config.exclude_modules, config.split_fused_qkv)
        self.atoms = nn.ModuleDict()          # str(slot index) -> SlotAtoms
        self._idx = {id(s): i for i, s in enumerate(self.slots)}
        self._cache = {}
        self._handles = attach_hooks(self.slots, self._delta)
        self._handles.append(base_model.register_forward_pre_hook(lambda m, a: self._prepare()))
        self._modules_w = list({id(s.module): s.module for s in self.slots}.values())

        def forward(*args, **kwargs):          # keeps the base model's signature (HF Trainer columns)
            return self.base_model(*args, **kwargs)
        functools.update_wrapper(forward, base_model.forward)
        self.forward = forward

    # ------------------------------------------------------------------ plumbing
    def _prepare(self):
        sc = self.peft_config.alpha_scale
        self._cache = {int(k): (a.alpha_raw * sc, a.u, a.v) for k, a in self.atoms.items()}

    def _delta(self, s, x):
        c = self._cache.get(self._idx[id(s)])
        if c is None:
            return None
        a, U, V = c
        return ((x.to(U.dtype) @ U.T) * a) @ V

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("base_model"), name)

    def get_base_model(self):
        return self.base_model

    @property
    def is_initialized(self):
        return len(self.atoms) > 0

    @staticmethod
    def atom_cost(slot):
        return slot.shape[0] + slot.shape[1] + 1

    def get_nb_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.base_model.parameters()) + trainable
        return trainable, total

    def print_trainable_parameters(self):
        t, a = self.get_nb_trainable_parameters()
        print(f"trainable params: {t:,} || all params: {a:,} || trainable%: {100 * t / a:.4f}")

    def describe(self):
        """Atoms per matrix, e.g. {'layer 0 attn.c_proj': 2, ...}."""
        return {f"layer {self.slots[int(k)].layer} {self.slots[int(k)].role}": a.alpha_raw.numel()
                for k, a in self.atoms.items()}

    # ------------------------------------------------------------------ initialisation
    def _batch_loss(self, b, loss_fn, dev):
        if loss_fn is not None:
            return loss_fn(self.base_model, b)
        if torch.is_tensor(b):
            b = {"input_ids": b}
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        if "labels" not in b:
            b["labels"] = b["input_ids"]
        return self.base_model(**b).loss

    def _probe(self, use, loss_fn, capture_batches=0):
        """Mean negative gradient of every target matrix over ``use``; activations on the first batches."""
        dev = self._modules_w[0].weight.device
        was_training = self.base_model.training
        self.base_model.eval()
        for m in self._modules_w:
            m.weight.requires_grad_(True)
            m.weight.grad = None
        acts = None
        try:
            k = min(capture_batches, len(use))
            if k:
                with _capture(self.slots) as st:
                    for b in use[:k]:
                        self._batch_loss(b, loss_fn, dev).backward()
                acts = {key: (torch.cat(v["x"]), torch.cat(v["d"])) for key, v in st.items()}
                del st
            for b in use[k:]:
                self._batch_loss(b, loss_fn, dev).backward()
            negG = [(-s.grad()).to(torch.float64 if s.grad().dtype == torch.float64 else torch.float32) / len(use) for s in self.slots]
        finally:
            for m in self._modules_w:
                m.weight.grad = None
                m.weight.requires_grad_(False)
            self.base_model.train(was_training)
        return negG, acts

    def _candidates(self, negG):
        """(descent rate, slot index, u, v, cost) for the leading singular pairs of every matrix's -G."""
        cands = []
        for i, (s, G) in enumerate(zip(self.slots, negG)):
            U, S, Vh = torch.linalg.svd(G.double(), full_matrices=False)
            for j in range(min(self.peft_config.candidates_per_matrix, len(S))):
                cands.append((float(S[j]), i, U[:, j].clone(), Vh[j].clone(), self.atom_cost(s)))
        return cands

    @torch.no_grad()
    def init_atoms(self, batches, n_batches=None, loss_fn=None, verbose=True):
        """Choose all atoms from one gradient probe.

        batches: iterable of dicts (``input_ids``, optional ``attention_mask`` / ``labels``) or input-id
            tensors (causal-LM labels = inputs). A DataLoader works.
        n_batches: number of batches to use (default ``config.init_batches``).
        loss_fn: optional ``loss_fn(base_model, batch) -> scalar``.
        Returns the selection in the order chosen: list of dicts (matrix, slot, sigma) where sigma is the
        atom's initial descent rate (singular value of the matrix's negative gradient).
        """
        if self.is_initialized:
            raise RuntimeError("atoms are already initialised")
        cfg = self.peft_config
        n_batches = n_batches or cfg.init_batches
        use = []
        for b in batches:
            if len(use) >= n_batches:
                break
            use.append(b)
        if not use:
            raise ValueError("init_atoms: no batches given")
        with torch.enable_grad():
            negG, acts = self._probe(use, loss_fn, cfg.ols_batches if cfg.rule == "ols" else 0)
        cands = self._candidates(negG)
        del negG
        if cfg.rule == "sigma":
            chosen = [cands[j] for j in greedy_sigma([c[0] for c in cands], [c[4] for c in cands], cfg.budget)]
        else:
            P = self._ols_columns(cands, acts)
            del acts
            chosen = [cands[j] for j in greedy_ols(P, [c[4] for c in cands], cfg.budget)]
            del P
        if not chosen:
            raise ValueError(f"budget {cfg.budget} is smaller than the cheapest atom "
                             f"({min(c[4] for c in cands)} parameters)")
        by_slot = defaultdict(list)
        for c in chosen:
            by_slot[c[1]].append(c)
        for i, cs in sorted(by_slot.items()):
            s = self.slots[i]
            w = s.module.weight
            dt = w.dtype if w.dtype in (torch.float32, torch.float64) else torch.float32   # fp32 for half bases
            sa = SlotAtoms(len(cs), s.shape[0], s.shape[1], w.device, dt)
            sa.u.copy_(torch.stack([c[2] for c in cs]))
            sa.v.copy_(torch.stack([c[3] for c in cs]))
            self.atoms[str(i)] = sa
        self.atoms = nn.ModuleDict({k: self.atoms[k] for k in sorted(self.atoms, key=int)})
        report = [{"matrix": f"layer {self.slots[c[1]].layer} {self.slots[c[1]].role}", "slot": c[1], "sigma": c[0]}
                  for c in chosen]
        if verbose:
            t, _ = self.get_nb_trainable_parameters()
            print(f"PursuitLoRA: {len(chosen)} atoms in {len(by_slot)} of {len(self.slots)} matrices, "
                  f"{t:,} trainable parameters (budget {cfg.budget:,}, rule '{cfg.rule}')")
        return report

    def _ols_columns(self, cands, acts):
        """Per-position contribution of each candidate to the linearised loss decrease: (x_t.u)(delta_t.v)."""
        by = defaultdict(list)
        for k, c in enumerate(cands):
            by[c[1]].append(k)
        P = None
        for i, ks in by.items():
            s = self.slots[i]
            x, d = acts[id(s.module)]
            U = torch.stack([cands[k][2] for k in ks], 1)
            V = torch.stack([cands[k][3] for k in ks], 1)
            cols = (x.double() @ U) * (d[:, s.c0:s.c1].double() @ V)
            if P is None:
                P = torch.empty(cols.shape[0], len(cands), dtype=torch.float64)
            P[:, ks] = cols
        return P

    # ------------------------------------------------------------------ save / load
    def save_pretrained(self, save_directory, **kwargs):
        if not self.is_initialized:
            raise RuntimeError("call init_atoms(...) before saving")
        from safetensors.torch import save_file
        os.makedirs(save_directory, exist_ok=True)
        layout = [{"slot": int(k), "layer": self.slots[int(k)].layer, "role": self.slots[int(k)].role,
                   "shape": list(self.slots[int(k)].shape), "atoms": a.alpha_raw.numel()} for k, a in self.atoms.items()]
        self.peft_config.save_pretrained(save_directory, extra={"layout": layout})
        sd = {}
        for k, a in self.atoms.items():
            for n in ("alpha_raw", "u", "v"):
                sd[f"slot{k}.{n}"] = getattr(a, n).detach().cpu().contiguous()
        save_file(sd, os.path.join(save_directory, WEIGHTS_NAME), metadata={"format": "pt", "library": "pursuitlora"})
        readme = os.path.join(save_directory, "README.md")
        if not os.path.exists(readme):
            with open(readme, "w") as f:
                f.write(_model_card(self))

    @classmethod
    def from_pretrained(cls, base_model, model_id, is_trainable=False, **hub_kwargs):
        config, raw = PursuitLoRAConfig.from_pretrained(model_id, **hub_kwargs)
        model = cls(base_model, config)
        from safetensors.torch import load_file
        sd = load_file(_resolve(model_id, WEIGHTS_NAME, **hub_kwargs))
        with torch.no_grad():
            for e in raw["layout"]:
                i = e["slot"]
                if i >= len(model.slots) or model.slots[i].role != e["role"] or model.slots[i].layer != e["layer"] \
                        or list(model.slots[i].shape) != e["shape"]:
                    raise ValueError("adapter does not match this base model's architecture")
                w = model.slots[i].module.weight
                sa = SlotAtoms(e["atoms"], e["shape"][0], e["shape"][1], w.device, sd[f"slot{i}.u"].dtype)
                for n in ("alpha_raw", "u", "v"):
                    getattr(sa, n).copy_(sd[f"slot{i}.{n}"])
                model.atoms[str(i)] = sa
        if not is_trainable:
            for p in model.parameters():
                p.requires_grad_(False)
        return model

    def push_to_hub(self, repo_id, private=None, token=None, commit_message="Upload PursuitLoRA adapter"):
        import tempfile
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id, private=private, exist_ok=True)
        with tempfile.TemporaryDirectory() as d:
            self.save_pretrained(d)
            return api.upload_folder(repo_id=repo_id, folder_path=d, commit_message=commit_message)

    # ------------------------------------------------------------------ merge
    @torch.no_grad()
    def merge_and_unload(self):
        """Write dW into the base weights and return the plain base model (no inference overhead)."""
        self._prepare()
        for i, s in enumerate(self.slots):
            c = self._cache.get(i)
            if c is not None:
                a, U, V = c
                w = s.weight()
                w.add_(((U.T * a) @ V).to(w.dtype))
        return self.unload()

    def unload(self):
        """Remove the adapter hooks without merging and return the base model."""
        for h in self._handles:
            h.remove()
        self._handles = []
        return self.base_model


def greedy_sigma(sigmas, costs, budget):
    """Take atoms in order of descent rate while they fit in the budget."""
    chosen, used = [], 0
    for j in sorted(range(len(sigmas)), key=lambda j: -sigmas[j]):
        if used + costs[j] <= budget:
            chosen.append(j)
            used += costs[j]
    return chosen


def greedy_ols(P, costs, budget):
    """Orthogonal least squares: repeatedly add the column of P that most reduces ||1 - P_S beta||^2."""
    P = P.clone()
    r = torch.ones(P.shape[0], dtype=P.dtype)
    costs = torch.tensor(costs, dtype=P.dtype)
    avail = torch.ones(P.shape[1], dtype=torch.bool)
    chosen, used = [], 0.0
    while True:
        fits = avail & (costs <= budget - used)
        if not fits.any():
            break
        den = (P * P).sum(0)
        score = torch.where(den > 1e-24, (r @ P) ** 2 / den.clamp_min(1e-30), torch.zeros_like(den))
        score[~fits] = -1.0
        j = int(score.argmax())
        if score[j] <= 0:
            break
        chosen.append(j)
        used += float(costs[j])
        avail[j] = False
        q = P[:, j] / den[j].sqrt()
        r = r - (q @ r) * q
        P = P - torch.outer(q, q @ P)
    return chosen


def get_pursuitlora_model(model, config: PursuitLoRAConfig = None):
    """Wrap a Hugging Face model with PursuitLoRA. Call ``.init_atoms(batches)`` before training."""
    return PursuitLoRAModel(model, config or PursuitLoRAConfig())


def _model_card(m):
    c = m.peft_config
    t, _ = m.get_nb_trainable_parameters()
    return f"""---
library_name: pursuitlora
base_model: {c.base_model_name_or_path or 'unknown'}
tags:
- pursuitlora
- peft
---

# PursuitLoRA adapter

Base model: `{c.base_model_name_or_path}`; {len(m.atoms)} matrices adapted; {t:,} adapter parameters
(budget {c.budget:,}, selection rule `{c.rule}`).

```python
from transformers import AutoModelForCausalLM
from pursuitlora import PursuitLoRAModel
base = AutoModelForCausalLM.from_pretrained("{c.base_model_name_or_path}")
model = PursuitLoRAModel.from_pretrained(base, "<this repo id>")
model = model.merge_and_unload()   # optional
```
"""
