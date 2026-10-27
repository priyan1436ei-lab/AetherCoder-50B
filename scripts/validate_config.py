"""Configuration validation utility for AetherCoder models.

Validates architectural hyperparameters, divisibility constraints, parameter counts,
and memory requirements without allocating large model tensors in RAM/VRAM.
"""

import argparse
import os
import sys
import yaml

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from scripts.check_hardware import calculate_moe_parameters, estimate_memory_requirements


def validate_config_file(config_path: str) -> bool:
    """Loads a YAML configuration file and validates its architecture and parameters."""
    print(f"\n======================================================================")
    print(f"Validating Model Configuration: {config_path}")
    print(f"======================================================================")

    if not os.path.exists(config_path):
        print(f"ERROR: Configuration file not found: {config_path}")
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Instantiate ModelConfig to trigger mathematical validation in __post_init__
    try:
        cfg = ModelConfig(**data)
        print("  [PASS] Mathematical and architectural constraints valid.")
    except Exception as exc:
        print(f"  [FAIL] Configuration validation error: {exc}")
        return False

    # Calculate exact theoretical parameter counts
    counts = calculate_moe_parameters(cfg)

    print(f"  Architecture:           {cfg.model_name} ({cfg.model_type})")
    print(f"  Hidden Size:            {cfg.hidden_size} (Heads: {cfg.num_attention_heads}, KV-Heads: {cfg.num_key_value_heads})")
    print(f"  Layers:                 {cfg.num_hidden_layers}")
    print(f"  Context Window:         {cfg.max_position_embeddings} tokens")
    print(f"  MoE Configuration:      {cfg.num_experts} total experts, top-{cfg.num_experts_per_tok} routed, {cfg.num_shared_experts} shared")
    print(f"  Total Parameters:       {counts['total_parameters']:,} ({counts['total_parameters'] / 1e9:.2f}B)")
    print(f"  Active Params / Token:  {counts['active_parameters']:,} ({counts['active_parameters'] / 1e9:.2f}B)")

    mem = estimate_memory_requirements(counts["total_parameters"], counts["active_parameters"])
    print(f"\n  Estimated Memory Requirements:")
    print(f"    - Model Weights (FP32):       {mem['weights_fp32_gb']:.2f} GB")
    print(f"    - Model Weights (BF16/FP16):  {mem['weights_bf16_gb']:.2f} GB")
    print(f"    - Model Weights (4-Bit QLoRA): {mem['weights_int4_gb']:.2f} GB")
    print(f"    - Pretrain AdamW Minimum:     {mem['pretraining_adamw_min_gb']:.2f} GB")

    return True


def main():
    parser = argparse.ArgumentParser(description="Validate model architecture configurations.")
    parser.add_argument("config_paths", nargs="*", help="Paths to YAML configuration files.")
    args = parser.parse_args()

    paths = args.config_paths
    if not paths:
        # Default to validating standard configs
        default_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "model")
        paths = [
            os.path.join(default_dir, "prototype_150m.yaml"),
            os.path.join(default_dir, "dense_50b.yaml"),
            os.path.join(default_dir, "moe_50b.yaml"),
        ]

    success = True
    for p in paths:
        if not validate_config_file(p):
            success = False

    print("\n" + "=" * 70)
    if success:
        print("ALL CONFIGURATIONS VALIDATED SUCCESSFULLY.")
        sys.exit(0)
    else:
        print("CONFIG VALIDATION FAILED FOR ONE OR MORE FILES.")
        sys.exit(1)


if __name__ == "__main__":
    main()
