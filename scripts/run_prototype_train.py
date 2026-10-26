"""Runnable training script for the Codex-150M prototype on curated multilingual code shards."""

import glob
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from model.config import get_prototype_config
from model.model import CodingCausalLM
from tokenizer.code_tokenizer import create_default_code_tokenizer
from training.pretraining.train_engine import PretrainingEngine
from data_pipeline.dataset_sharder import BinaryDatasetReader
from scripts.prepare_dataset import prepare_training_dataset


def run_training_demo(num_steps: int = 10, batch_size: int = 2, seq_len: int = 64):
    print("=" * 65)
    print("CODEX-150M MULTILINGUAL PRETRAINING ON MULTI-DOMAIN SHARDS")
    print("=" * 65)

    # 1. Ensure Shards Exist
    shards_dir = "./data/shards"
    shard_files = glob.glob(os.path.join(shards_dir, "*.bin"))
    if not shard_files:
        print("[Notice] Shards not found. Generating fresh multilingual shards...")
        prepare_training_dataset()
        shard_files = glob.glob(os.path.join(shards_dir, "*.bin"))

    print(f"[1/4] Found {len(shard_files)} binary shard(s) in {shards_dir}")

    # 2. Initialize Model
    print("[2/4] Constructing Codex-150M Sparse MoE Model...")
    config = get_prototype_config()
    model = CodingCausalLM(config)
    param_info = model.count_parameters()
    print(f"      Total Parameters:  {param_info['total_parameters']:,}")
    print(f"      Active Parameters: {param_info['active_parameters']:,} (Top-2 routed + 1 shared)")

    # 3. Setup Pretraining Engine
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[3/4] Initializing Pretraining Engine on {device.upper()}...")

    engine = PretrainingEngine(
        model=model,
        learning_rate=5e-4,
        min_learning_rate=5e-5,
        warmup_steps=3,
        max_steps=num_steps,
        gradient_accumulation_steps=2,
        device=device,
        checkpoint_dir="./checkpoints/prototype_150m",
    )

    # 4. Stream Tokens from Shards and Train
    print(f"[4/4] Streaming batches from binary memory-mapped shards (seq_len={seq_len})...")
    reader = BinaryDatasetReader(shard_paths=shard_files)
    block_gen = reader.iterate_blocks(block_size=seq_len)

    step = 0
    while step < num_steps:
        batch_blocks = []
        for _ in range(batch_size):
            try:
                block = next(block_gen)
                batch_blocks.append(block)
            except StopIteration:
                # Reset reader generator
                block_gen = reader.iterate_blocks(block_size=seq_len)
                block = next(block_gen)
                batch_blocks.append(block)

        batch_arr = np.array(batch_blocks, dtype=np.int64)
        input_tensor = torch.from_numpy(batch_arr)
        labels_tensor = input_tensor.clone()

        step_metrics = engine.train_step(input_tensor, labels_tensor)
        grad_norm, lr = engine.optimizer_step()
        step += 1

        print(f"  Step {step:02d}/{num_steps:02d} | Loss: {step_metrics['loss']:.4f} | Aux Loss: {step_metrics['aux_loss']:.5f} | Grad Norm: {grad_norm:.2f} | LR: {lr:.6f}")

    # Save final verified checkpoint
    ckpt_path = engine.save_checkpoint(step=num_steps, tag="multilingual")
    print("=" * 65)
    print(f"[SUCCESS] Multilingual training complete! Checkpoint saved to: {ckpt_path}")
    print("=" * 65)


if __name__ == "__main__":
    run_training_demo(num_steps=5)
