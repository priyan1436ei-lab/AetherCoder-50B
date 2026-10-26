"""Supervised Fine-Tuning (SFT) pipeline with prompt masking for coding instructions."""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from model.model import CodingCausalLM
from tokenizer.code_tokenizer import CodeTokenizer


class SFTDatasetPreparer:
    """Prepares instruction-tuning samples with prompt loss masking (label = -100)."""

    def __init__(self, tokenizer: CodeTokenizer, max_seq_len: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def format_and_tokenize(self, prompt: str, completion: str) -> Dict[str, torch.Tensor]:
        """Encodes sample and masks prompt tokens with -100 in labels."""
        prompt_ids = self.tokenizer.encode(f"<|user|>\n{prompt}\n<|assistant|>\n")
        completion_ids = self.tokenizer.encode(f"{completion}\n<|endoftext|>")

        input_ids = prompt_ids + completion_ids
        # Mask prompt in labels
        labels = ([-100] * len(prompt_ids)) + completion_ids

        # Truncate if exceeds max length
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class SFTTrainer:
    """Supervised fine-tuning trainer for instruction following and code generation."""

    def __init__(self, model: CodingCausalLM, learning_rate: float = 1e-4, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    def train_step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> float:
        self.model.train()
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()
