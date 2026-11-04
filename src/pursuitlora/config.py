# Copyright 2026 Angshul Majumdar.
"""PursuitLoRAConfig."""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Union

CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAME = "adapter_model.safetensors"


@dataclass
class PursuitLoRAConfig:
    """Configuration of a PursuitLoRA adapter.

    Args:
        budget: maximum number of trainable parameters. Each atom in a d_in x d_out matrix costs
            d_in + d_out + 1; atoms are chosen across all target matrices until the budget is used.
        rule: atom selection rule. ``"sigma"``: largest descent rate (singular value of the matrix's
            negative gradient) first. ``"ols"``: orthogonal least squares on the linearised loss, which
            avoids picking atoms that explain the same gradient signal (needs per-token activations of
            all target modules on ``ols_batches`` batches, so it uses more memory).
        target_modules: ``"all-linear"`` or a list of module-name suffixes, e.g. ``["q_proj", "v_proj"]``.
        exclude_modules: substrings of module names that are never adapted.
        split_fused_qkv: split GPT-2-style fused ``c_attn`` = [q|k|v] into q, k and v.
        candidates_per_matrix: leading singular pairs considered per matrix.
        alpha_scale: coefficients are parametrised as ``alpha_scale * alpha_raw`` so one optimiser
            learning rate (about 3e-3 for AdamW) suits coefficients and vectors.
        init_batches: default number of batches for the gradient probe.
        ols_batches: batches (out of the probe batches) whose activations are kept for ``rule="ols"``.
    """
    budget: int = 9216
    rule: str = "sigma"
    target_modules: Union[str, List[str]] = "all-linear"
    exclude_modules: List[str] = field(default_factory=lambda: ["lm_head", "embed", "wte", "wpe"])
    split_fused_qkv: bool = True
    candidates_per_matrix: int = 8
    alpha_scale: float = 100.0 / 3.0
    init_batches: int = 4
    ols_batches: int = 2
    base_model_name_or_path: Optional[str] = None
    peft_type: str = "PURSUITLORA"
    task_type: str = "CAUSAL_LM"
    pursuitlora_version: Optional[str] = None

    def __post_init__(self):
        if self.rule not in ("sigma", "ols"):
            raise ValueError("rule must be 'sigma' or 'ols'")
        if self.budget <= 0:
            raise ValueError("budget must be positive")

    def to_dict(self):
        return asdict(self)

    def save_pretrained(self, save_directory, extra=None):
        os.makedirs(save_directory, exist_ok=True)
        d = self.to_dict()
        if extra:
            d.update(extra)
        with open(os.path.join(save_directory, CONFIG_NAME), "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def from_dict(cls, d):
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in names})

    @classmethod
    def from_pretrained(cls, model_id, **hub_kwargs):
        with open(_resolve(model_id, CONFIG_NAME, **hub_kwargs)) as f:
            d = json.load(f)
        return cls.from_dict(d), d


def _resolve(model_id, filename, **hub_kwargs):
    local = os.path.join(model_id, filename)
    if os.path.isfile(local):
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=model_id, filename=filename, **hub_kwargs)
