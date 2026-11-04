# Copyright 2026 Angshul Majumdar.
"""PursuitLoRA: LoRA-shaped adapters with rank-one atoms chosen by gradient pursuit."""
__version__ = "0.1.0"

from .config import PursuitLoRAConfig  # noqa: E402
from .model import PursuitLoRAModel, get_pursuitlora_model  # noqa: E402


def __getattr__(name):  # lazy: importing the Trainer pulls in accelerate
    if name == "PursuitLoRATrainer":
        from .trainer import PursuitLoRATrainer
        return PursuitLoRATrainer
    raise AttributeError(name)


__all__ = ["PursuitLoRAConfig", "PursuitLoRAModel", "PursuitLoRATrainer", "get_pursuitlora_model", "__version__"]
