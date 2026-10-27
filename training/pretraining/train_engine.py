"""Distributed & single-device pretraining engine with cosine learning rate scheduling and atomic checkpointing."""

import math
import os
import random
import time
from typing import Any, Dict, Iterator, Optional, Tuple
import numpy as np
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
        if self.current_step <= self.warmup_steps:
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

    def state_dict(self) -> Dict[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
            "current_step": self.current_step,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.warmup_steps = state.get("warmup_steps", self.warmup_steps)
        self.max_steps = state.get("max_steps", self.max_steps)
        self.min_lr = state.get("min_lr", self.min_lr)
        self.max_lr = state.get("max_lr", self.max_lr)
        self.current_step = state.get("current_step", 0)


class PretrainingEngine:
    """Robust training engine with gradient accumulation, clipping, atomic checkpointing, and mixed precision."""

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
        mixed_precision: Optional[str] = None,  # "bf16", "fp16", or None
        device: str = "cpu",
        checkpoint_dir: str = "./checkpoints",
    ):
        self.model = model.to(device)
        self.device = device
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.max_steps = max_steps
        self.mixed_precision = mixed_precision
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Decay 2D weight tensors (linear, embedding), exempt RMSNorm and biases
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

        # GradScaler for FP16 training stability
        self.scaler = None
        if self.mixed_precision == "fp16" and "cuda" in self.device:
            self.scaler = torch.amp.GradScaler("cuda")

        self.global_step = 0
        self.total_tokens_seen = 0
        self.accumulated_loss = 0.0

    def train_step(self, batch_input_ids: torch.Tensor, batch_labels: torch.Tensor) -> Dict[str, float]:
        """Executes a single forward and backward accumulation step with non-finite loss protection."""
        self.model.train()
        batch_input_ids = batch_input_ids.to(self.device)
        batch_labels = batch_labels.to(self.device)

        amp_dtype = torch.bfloat16 if self.mixed_precision == "bf16" else (torch.float16 if self.mixed_precision == "fp16" else None)
        use_amp = amp_dtype is not None and ("cuda" in self.device or "cpu" in self.device and self.mixed_precision == "bf16")

        if use_amp:
            device_type = "cuda" if "cuda" in self.device else "cpu"
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):
                outputs = self.model(input_ids=batch_input_ids, labels=batch_labels)
        else:
            outputs = self.model(input_ids=batch_input_ids, labels=batch_labels)

        loss = outputs["loss"]
        if torch.isnan(loss) or torch.isinf(loss):
            return {"loss": float("nan"), "aux_loss": 0.0, "skipped": 1.0}

        scaled_loss = loss / self.gradient_accumulation_steps
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        self.total_tokens_seen += batch_input_ids.numel()
        raw_loss_item = loss.item()
        self.accumulated_loss += raw_loss_item

        return {
            "loss": raw_loss_item,
            "aux_loss": outputs["aux_loss"].item() if "aux_loss" in outputs and outputs["aux_loss"] is not None else 0.0,
        }

    def optimizer_step(self) -> Tuple[float, float]:
        """Clips gradients, applies optimizer step, and updates learning rate scheduler."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # Handle non-finite gradients safely
        if math.isnan(float(grad_norm)) or math.isinf(float(grad_norm)):
            self.optimizer.zero_grad(set_to_none=True)
            return float("nan"), self.scheduler.step()

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        current_lr = self.scheduler.step()
        self.global_step += 1
        self.accumulated_loss = 0.0
        return float(grad_norm), current_lr

    def save_checkpoint(self, step: int, tag: Optional[str] = None) -> str:
        """Atomically saves model weights, optimizer, scheduler, and RNG state to prevent corruption."""
        filename = f"checkpoint_step_{step}" + (f"_{tag}" if tag else "") + ".pt"
        save_path = os.path.join(self.checkpoint_dir, filename)
        tmp_path = save_path + ".tmp"

        rng_state = {
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        }
        if torch.cuda.is_available():
            rng_state["cuda_rng"] = torch.cuda.get_rng_state_all()

        checkpoint_data = {
            "step": step,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.model.config.__dict__,
            "tokens_seen": self.total_tokens_seen,
            "rng_state": rng_state,
        }

        torch.save(checkpoint_data, tmp_path)
        os.replace(tmp_path, save_path)  # Atomic on POSIX and modern Windows
        return save_path

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Restores model, optimizer, scheduler, and RNG states safely from a checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint.get("global_step", checkpoint.get("step", 0))
        self.total_tokens_seen = checkpoint.get("tokens_seen", 0)

        # Restore RNG if present
        if "rng_state" in checkpoint:
            rng = checkpoint["rng_state"]
            if "torch_rng" in rng:
                torch.set_rng_state(rng["torch_rng"])
            if "python_rng" in rng:
                random.setstate(rng["python_rng"])
            if "numpy_rng" in rng:
                np.random.set_state(rng["numpy_rng"])

        return self.global_step
