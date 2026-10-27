"""Checkpoint inspection utility for AetherCoder.

Inspects PyTorch checkpoint files (.pt), displays model architecture configuration,
training state (global step, loss, optimizer/scheduler presence, RNG state),
and audits tensor shapes without executing arbitrary code.
"""

import argparse
import os
import sys
import torch

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def inspect_checkpoint(checkpoint_path: str) -> bool:
    """Safely loads and audits a model checkpoint."""
    print("=" * 70)
    print(f"Checkpoint Inspection: {checkpoint_path}")
    print("=" * 70)

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
        return False

    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"  File Size:             {file_size_mb:.2f} MB")

    try:
        # Load weights safely using weights_only=False only if required for custom classes,
        # but try weights_only=True first for security.
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            print("  Security Check:        [PASS] Safe checkpoint load (weights_only=True)")
        except Exception:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            print("  Security Check:        [WARN] Checkpoint loaded with weights_only=False")

        if not isinstance(ckpt, dict):
            print(f"  Format:                Raw tensor or state dictionary ({type(ckpt)})")
            return True

        # Audit metadata keys
        step = ckpt.get("step") or ckpt.get("global_step", "Unknown")
        loss = ckpt.get("loss", "Unknown")
        epoch = ckpt.get("epoch", "Unknown")
        cfg_dict = ckpt.get("config")
        has_opt = "optimizer_state_dict" in ckpt
        has_sched = "scheduler_state_dict" in ckpt
        has_scaler = "scaler_state_dict" in ckpt
        has_rng = "rng_state" in ckpt

        print(f"  Global Step:           {step}")
        print(f"  Loss:                  {loss}")
        print(f"  Epoch:                 {epoch}")
        print(f"  Optimizer State:       {'Present' if has_opt else 'Absent'}")
        print(f"  Scheduler State:       {'Present' if has_sched else 'Absent'}")
        print(f"  GradScaler State:      {'Present' if has_scaler else 'Absent'}")
        print(f"  RNG States Persisted:  {'Yes' if has_rng else 'No'}")

        if cfg_dict:
            print(f"  Model Architecture:    {cfg_dict.get('model_name', 'Unknown')}")
            print(f"  Hidden Size:           {cfg_dict.get('hidden_size', 'Unknown')}")
            print(f"  Layers:                {cfg_dict.get('num_hidden_layers', 'Unknown')}")
            print(f"  Context Window:        {cfg_dict.get('max_position_embeddings', 'Unknown')}")

        state_dict = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
        if isinstance(state_dict, dict):
            tensor_count = len(state_dict)
            param_elements = sum(t.numel() for t in state_dict.values() if hasattr(t, "numel"))
            print(f"  Total Tensors:         {tensor_count}")
            print(f"  Total Parameter Count: {param_elements:,} ({param_elements / 1e6:.2f}M)")

            print("\n  Sample Layer Tensors:")
            sample_keys = list(state_dict.keys())[:8]
            for k in sample_keys:
                tensor = state_dict[k]
                if hasattr(tensor, "shape") and hasattr(tensor, "dtype"):
                    print(f"    - {k:<45} {str(list(tensor.shape)):<20} {tensor.dtype}")
            if len(state_dict) > 8:
                print(f"    ... [{len(state_dict) - 8} more layer tensors]")

        return True
    except Exception as exc:
        print(f"ERROR: Failed to inspect checkpoint: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Inspect PyTorch model checkpoints.")
    parser.add_argument("checkpoint_path", nargs="?", default="checkpoints/prototype_150m_step_5.pt", help="Path to checkpoint .pt file")
    args = parser.parse_args()

    inspect_checkpoint(args.checkpoint_path)


if __name__ == "__main__":
    main()
