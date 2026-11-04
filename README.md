# PursuitLoRA

LoRA-shaped adapters whose rank-one atoms are chosen by gradient pursuit, for Hugging Face causal LMs.

The update to every target matrix is a sum of rank-one atoms `α · u vᵀ`, as in LoRA. Unlike LoRA,
the rank of each matrix is not fixed in advance. One gradient probe before training gives every
matrix's negative gradient `−G`. Its leading singular pairs are the candidate atoms, and an atom's
singular value is its initial descent rate. Atoms are then chosen greedily across **all** matrices
under a parameter budget. The gradient therefore decides which matrices are adapted and with what
rank; many matrices get none. Coefficients start at zero, so the model is unchanged at step 0.
Coefficients and vectors are then trained with AdamW. A merged adapter has zero inference overhead.

- Explanation with figures: [`docs/index.md`](docs/index.md) (GitHub Pages: https://angshulmajumdar.github.io/PursuitLoRA/)
- Sibling method, CP atoms over stacked layers: https://github.com/AngshulMajumdar/TensorPEFT

Two selection rules are available:

- `sigma`: largest singular value first (default).
- `ols`: orthogonal least squares on the linearised loss, which avoids picking atoms whose effects overlap.

## Install

```
pip install git+https://github.com/AngshulMajumdar/PursuitLoRA
pip install .                      # from a clone
```

Requires `torch>=2.1`, `transformers>=4.40`, `safetensors` and `huggingface_hub`.

## Usage

```python
from transformers import AutoModelForCausalLM
from pursuitlora import PursuitLoRAConfig, get_pursuitlora_model, PursuitLoRAModel

model = get_pursuitlora_model(AutoModelForCausalLM.from_pretrained("gpt2"),
                              PursuitLoRAConfig(budget=9216))
model.init_atoms(train_dataloader, n_batches=4)   # gradient probe -> atoms, chosen under the budget
model.print_trainable_parameters()                # 8,452
print(model.describe())                           # atoms per matrix

# train: AdamW, lr 3e-3, ~10 warm-up steps, weight_decay 0 (or pursuitlora.PursuitLoRATrainer)

model.save_pretrained("adapter")                  # adapter_config.json + adapter_model.safetensors
model = PursuitLoRAModel.from_pretrained(AutoModelForCausalLM.from_pretrained("gpt2"), "adapter")
merged = model.merge_and_unload()
```

`model.push_to_hub("user/repo")` uploads the adapter. `PursuitLoRATrainer` is a drop-in
`transformers.Trainer` whose checkpoints contain only the adapter.

| `PursuitLoRAConfig` | default | |
|---|---|---|
| `budget` | 9216 | maximum trainable parameters (one atom costs `d_in + d_out + 1`) |
| `rule` | `"sigma"` | `"sigma"` or `"ols"` |
| `target_modules` | `"all-linear"` | or name suffixes, e.g. `["q_proj", "v_proj"]` |
| `exclude_modules` | lm_head, embeddings | |
| `split_fused_qkv` | True | GPT-2 `c_attn` → q, k, v |
| `candidates_per_matrix` | 8 | singular pairs considered per matrix |
| `init_batches` | 4 | probe batches |
| `ols_batches` | 2 | probe batches whose activations are kept for `ols` |
| `alpha_scale` | 33.3 | coefficient scaling so that one learning rate fits all parameters |

## Results (GPT-2 124M → Bengali news, 300 steps, validation loss, lower is better)

Seed 0, identical batches for all methods. All numbers, training curves and selections are in
`benchmarks/results.json`.

| budget | PursuitLoRA `sigma` | PursuitLoRA `ols` | best baseline | LoRA |
|---|---:|---:|---|---:|
| ≈1.5k | 1.4842 (1,537) | **1.4812** (1,537) | prompt tuning, 1 token: 1.4928 (768) | – |
| ≈9k | 1.3693 (8,452) | **1.3684** (8,452) | prompt tuning, 12 tokens: 1.3704 (9,216) | r=1, q,v, last 3 layers: 1.4299 |
| ≈37k | 1.2732 (36,112) | 1.2633 (35,345) | **LayerNorm tuning: 1.2530** (38,400) | r=1, q,v: 1.3181 |
| ≈55k | **1.2536** (53,783) | 1.2594 (54,556) | (IA)³, k,v,MLP: 1.2894 (55,296) | – |

Parameter counts are in brackets. Base model: 1.7079. Learning rates: PursuitLoRA 3e-3,
LoRA 1e-3, prompt tuning 3e-2, LayerNorm 1e-3, (IA)³ 3e-3.

Three seeds against the best baseline (paired; seed changes probe and training batches):

| budget | PursuitLoRA `sigma` | best baseline | mean Δ | p (paired t) |
|---|---|---|---:|---:|
| ≈9k | 1.3693 / 1.3606 / 1.3530 | prompt tuning 1.3704 / 1.3773 / 1.3576 | −0.0075 | 0.26 |
| ≈37k | 1.2732 / 1.2840 / 1.2652 | LayerNorm tuning 1.2530 / 1.2516 / 1.2466 | +0.0237 | 0.03 |
| ≈55k | 1.2536 / 1.2671 / 1.2314 | (IA)³ 1.2894 / 1.2899 / 1.2845 | −0.0372 | 0.05 |

- PursuitLoRA is ahead of every baseline at ≈1.5k (single seed) and ≈55k (3 of 3 seeds).
- At ≈9k it is ahead of prompt tuning on 3 of 3 seeds, but by too little to separate from a tie.
- At ≈37k LayerNorm tuning is better.
- It beats LoRA at every budget where LoRA fits.
- At ≥120k parameters, LoRA r=1 on most or all matrices reaches lower loss (1.19 and 1.18).

The default rule is `sigma`: the two rules are within 0.01 of each other, and `ols` keeps per-token
activations of every target matrix for `ols_batches` batches (peak process memory about 5.5 GB for
GPT-2 with 2 × 8 × 128 tokens).

## Repository

```
src/pursuitlora/   package (config, layer discovery, probe and selection, save/load, merge, Trainer)
tests/             pytest: GPT-2, Llama, Qwen2, Mistral, Phi, Gemma (CPU, tiny random models)
examples/          prepare_bengali_data.py, reproduce_gpt2_bengali.py, train_with_trainer.py
benchmarks/        results.json, descent_spectrum_gpt2_bengali.json
scripts/           make_figures.py (renders docs/figures from benchmarks/)
docs/              GitHub Pages article (index.md + figures)
```

Reproduce a row:

```
python examples/prepare_bengali_data.py --out data
python examples/reproduce_gpt2_bengali.py --model gpt2 --data data --budget 9216 --rule sigma
```

Tests: `pip install .[test] && pytest tests`

## Notes

- Probe memory is about 2× the size of the targeted weights, briefly. For large models, restrict `target_modules`.
- Quantised (bitsandbytes) bases are not supported in this version.
