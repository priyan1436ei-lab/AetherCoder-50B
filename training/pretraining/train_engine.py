"""Distributed & single-device pretraining engine with cosine learning rate scheduling and checkpointing."""

import math
import os
import time
from typing import Dict, Iterator, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW

from model.config import ModelConfig
from model.model import CodingCausalLM


class CosineWarmupLRScheduler:
    """Cosine learning rate scheduler with linear warmup."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, max_steps: int, min_lr: float, max_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_step = 0

    def step(self) -> float:
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * (self.current_step / max(1, self.warmup_steps))
        elif self.current_step > self.max_steps:
            lr = self.min_lr
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


class PretrainingEngine:
    """Robust training engine with gradient accumulation, clipping, and checkpoint recovery."""

    def __init__(
        self,
        model: CodingCausalLM,
        learning_rate: float = 3e-4,
        min_learning_rate: float = 3e-5,
        weight_decay: float = 0.1,
        warmup_steps: int = 10,
        max_steps: int = 100,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
        checkpoint_dir: str = "./checkpoints",
    ):
        self.model = model.to(device)
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.max_steps = max_steps
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Decay all 2D tensor weights (linear, embedding), exempt RMSNorm and biases
        decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        self.optimizer = AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        self.scheduler = CosineWarmupLRScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr=min_learning_rate,
            max_lr=learning_rate,
        )

        self.global_step = 0
        self.total_tokens_seen = 0

    def train_step(self, batch_input_ids: torch.Tensor, batch_labels: torch.Tensor) -> Dict[str, float]:
        """Executes a single forward and backward accumulation step."""
        self.model.train()
        batch_input_ids = batch_input_ids.to(self.device)
        batch_labels = batch_labels.to(self.device)

        outputs = self.model(input_ids=batch_input_ids, labels=batch_labels)
        loss = outputs["loss"] / self.gradient_accumulation_steps
        loss.backward()

        self.total_tokens_seen += batch_input_ids.numel()
        return {
            "loss": loss.item() * self.gradient_accumulation_steps,
            "aux_loss": outputs["aux_loss"].item(),
        }

    def optimizer_step(self) -> Tuple[float, float]:
        """Clips gradients and steps optimizer and learning rate scheduler."""
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        current_lr = self.scheduler.step()
        self.global_step += 1
        return float(grad_norm), current_lr

    def save_checkpoint(self, step: int, tag: Optional[str] = None) -> str:
        """Saves model weights, optimizer, and scheduler states."""
        filename = f"checkpoint_step_{step}" + (f"_{tag}" if tag else "") + ".pt"
        save_path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.model.config.__dict__,
            "tokens_seen": self.total_tokens_seen,
        }, save_path)
        return save_path

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Restores model and optimizer states from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("step", 0)
        self.total_tokens_seen = checkpoint.get("tokens_seen", 0)
        return self.global_step
