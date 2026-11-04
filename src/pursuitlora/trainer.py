# Copyright 2026 Angshul Majumdar.
"""Hugging Face Trainer that saves PursuitLoRA checkpoints as adapters (not the full model)."""
import os

from transformers import Trainer


class PursuitLoRATrainer(Trainer):
    """Drop-in ``transformers.Trainer``; checkpoints and ``save_model`` write only the adapter."""

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        if self.args.should_save:
            model.save_pretrained(output_dir)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
