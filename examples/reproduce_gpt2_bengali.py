"""Reproduce the GPT-2 / Bengali small-budget benchmark with the package API (plain PyTorch loop).

  python reproduce_gpt2_bengali.py --model <gpt2 dir or 'gpt2'> --data <dir with train.bin, val.bin> \
      --budget 9216 --rule sigma   # budgets used in the benchmark: 1536, 9216, 36864, 55296

train.bin / val.bin are uint16 GPT-2 token ids, produced by prepare_bengali_data.py.  Reported number: mean cross-entropy (nats/token)
on the first 64 windows of 128+1 tokens of val.bin.
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from pursuitlora import PursuitLoRAConfig, get_pursuitlora_model

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--data", required=True)
p.add_argument("--budget", type=int, default=9216)
p.add_argument("--rule", choices=["sigma", "ols"], default="sigma")
p.add_argument("--steps", type=int, default=300)
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--seq", type=int, default=128)
p.add_argument("--init-batches", type=int, default=4)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--save", default=None)
a = p.parse_args()
torch.manual_seed(a.seed)

train = np.memmap(f"{a.data}/train.bin", dtype=np.uint16, mode="r")
val = np.memmap(f"{a.data}/val.bin", dtype=np.uint16, mode="r")


def sampler(seed):
    rng = np.random.default_rng(seed)
    while True:
        ix = rng.integers(0, len(train) - a.seq - 1, a.batch)
        yield torch.from_numpy(np.stack([train[i:i + a.seq + 1] for i in ix]).astype(np.int64))


V = torch.from_numpy(np.asarray(val[:64 * (a.seq + 1)]).astype(np.int64).reshape(64, a.seq + 1))


def loss_on(model, x):
    return F.cross_entropy(model(x[:, :-1]).logits.flatten(0, 1), x[:, 1:].flatten())


@torch.no_grad()
def evaluate(model):
    return sum(float(loss_on(model, V[i:i + 16])) * len(V[i:i + 16]) for i in range(0, len(V), 16)) / len(V)


base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
model = get_pursuitlora_model(base, PursuitLoRAConfig(budget=a.budget, rule=a.rule, init_batches=a.init_batches))
print("base val", round(evaluate(model), 4))
t0 = time.time()
model.init_atoms(sampler(a.seed + 12345))            # one gradient probe -> all atoms
model.print_trainable_parameters()
print(model.describe())

opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))
batches = sampler(a.seed)
for step in range(1, a.steps + 1):
    loss = loss_on(model, next(batches))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    if step % 60 == 0 or step == a.steps:
        print(f"step {step}  train {loss.item():.4f}  val {evaluate(model):.4f}  {time.time() - t0:.0f}s", flush=True)
v = evaluate(model)
print(f"final val {v:.4f}  (ppl {math.exp(v):.2f})")
if a.save:
    model.save_pretrained(a.save)
