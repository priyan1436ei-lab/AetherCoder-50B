"""Runnable script for Supervised Fine-Tuning (SFT) on instruction-following datasets."""

import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from model.config import get_prototype_config
from model.model import CodingCausalLM
from tokenizer.code_tokenizer import create_default_code_tokenizer
from training.sft.train_sft import SFTTrainer, SFTDatasetPreparer


def run_sft_training(sft_file: str = "./data/sft/sft_instructions.jsonl", num_epochs: int = 2):
    print("=" * 65)
    print("CODEX-150M SUPERVISED FINE-TUNING (SFT) PIPELINE")
    print("=" * 65)

    if not os.path.exists(sft_file):
        raise FileNotFoundError(f"SFT dataset not found: {sft_file}. Run scripts/prepare_dataset.py first.")

    tokenizer = create_default_code_tokenizer()
    preparer = SFTDatasetPreparer(tokenizer=tokenizer, max_seq_len=256)

    # Load SFT samples
    samples = []
    with open(sft_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"[1/3] Loaded {len(samples)} instruction-following samples from {sft_file}")

    # Initialize model
    config = get_prototype_config()
    model = CodingCausalLM(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[2/3] Initializing SFT Trainer on {device.upper()}...")
    trainer = SFTTrainer(model=model, learning_rate=2e-4, device=device)

    # Training Loop
    print(f"[3/3] Training across {num_epochs} epoch(s)...")
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for i, sample in enumerate(samples):
            tokenized = preparer.format_and_tokenize(sample["prompt"], sample["completion"])
            # Batch dimension
            input_ids = tokenized["input_ids"].unsqueeze(0)
            labels = tokenized["labels"].unsqueeze(0)
            loss = trainer.train_step(input_ids, labels)
            total_loss += loss

        avg_loss = total_loss / max(1, len(samples))
        print(f"  Epoch {epoch:02d}/{num_epochs:02d} | Average SFT Loss: {avg_loss:.4f}")

    os.makedirs("./checkpoints/prototype_150m", exist_ok=True)
    out_path = "./checkpoints/prototype_150m/sft_aligned_model.pt"
    torch.save({
        "config": config.__dict__,
        "model_state_dict": model.state_dict(),
    }, out_path)

    print("=" * 65)
    print(f"[SUCCESS] SFT instruction tuning complete! Checkpoint saved to: {out_path}")
    print("=" * 65)


if __name__ == "__main__":
    run_sft_training(num_epochs=2)
