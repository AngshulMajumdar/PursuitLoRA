# Copyright 2026 Angshul Majumdar.
"""Discovery of adaptable weight matrices ('slots') and the forward hooks that apply dW."""
import re
from collections import defaultdict

import torch
import torch.nn as nn

try:
    from transformers.pytorch_utils import Conv1D
except Exception:  # pragma: no cover
    class Conv1D(nn.Module):
        pass

# GPT-2 family: Conv1D c_attn holds [q | k | v] as three equal column blocks
FUSED_EQUAL_SPLIT = {"c_attn": ("q", "k", "v")}
_LAYER_RE = re.compile(r"\.(\d+)\.")


class Slot:
    """One adapted matrix, always viewed as (d_in, d_out) so that the adapter adds x @ dW."""

    def __init__(self, module_name, module, layer, role, c0, c1, d_in):
        self.module_name, self.module = module_name, module
        self.layer, self.role = layer, role
        self.c0, self.c1, self.d_in = c0, c1, d_in
        self.conv1d = isinstance(module, Conv1D)

    @property
    def shape(self):
        return (self.d_in, self.c1 - self.c0)

    def _view(self, w):
        return w[:, self.c0:self.c1] if self.conv1d else w[self.c0:self.c1, :].T

    def weight(self):
        return self._view(self.module.weight)

    def grad(self):
        return self._view(self.module.weight.grad)

    def __repr__(self):
        return f"Slot(layer {self.layer}: {self.role} {list(self.shape)})"


def find_slots(model, target_modules="all-linear", exclude_modules=(), split_fused_qkv=True):
    slots = []
    for name, mod in model.named_modules():
        if not isinstance(mod, (nn.Linear, Conv1D)):
            continue
        if any(e in name for e in exclude_modules):
            continue
        m = _LAYER_RE.search(name)
        if m is None:
            continue
        role = name[m.end():]
        w = mod.weight
        d_in, d_out = (w.shape[0], w.shape[1]) if isinstance(mod, Conv1D) else (w.shape[1], w.shape[0])
        parts = FUSED_EQUAL_SPLIT.get(role.split(".")[-1]) if (split_fused_qkv and isinstance(mod, Conv1D)) else None
        if parts and d_out % len(parts) == 0:
            wd = d_out // len(parts)
            for j, p in enumerate(parts):
                slots.append(Slot(name, mod, int(m.group(1)), f"{role}.{p}", j * wd, (j + 1) * wd, d_in))
        else:
            slots.append(Slot(name, mod, int(m.group(1)), role, 0, d_out, d_in))
    if target_modules != "all-linear":
        if isinstance(target_modules, str):
            target_modules = [target_modules]
        pats = [re.compile(r"(^|\.)" + re.escape(t) + r"$") for t in target_modules]
        slots = [s for s in slots if any(p.search(s.role) for p in pats)]
    if not slots:
        raise ValueError(f"PursuitLoRA: no target modules matched {target_modules!r}")
    return slots


def attach_hooks(slots, delta_fn):
    """Forward hooks adding delta_fn(slot, x) to each slot's output columns (cast to the output dtype)."""
    by_mod = defaultdict(list)
    for s in slots:
        by_mod[id(s.module)].append(s)
    handles = []
    for ss in by_mod.values():
        ss = sorted(ss, key=lambda s: s.c0)
        mod = ss[0].module
        width = mod.weight.shape[1] if ss[0].conv1d else mod.weight.shape[0]

        def hook(module, inputs, out, ss=ss, width=width):
            x = inputs[0]
            deltas = [delta_fn(s, x) for s in ss]
            if all(d is None for d in deltas):
                return out
            pieces, pos = [], 0
            for s, d in zip(ss, deltas):
                if s.c0 > pos:
                    pieces.append(out.new_zeros(*out.shape[:-1], s.c0 - pos))
                pieces.append(out.new_zeros(*out.shape[:-1], s.c1 - s.c0) if d is None else d.to(out.dtype))
                pos = s.c1
            if pos < width:
                pieces.append(out.new_zeros(*out.shape[:-1], width - pos))
            return out + (pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1))

        handles.append(mod.register_forward_hook(hook))
    return handles
