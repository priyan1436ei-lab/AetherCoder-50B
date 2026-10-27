# AetherCoder-50B Training & Fine-Tuning Guide

This guide details the training workflows for AetherCoder, covering local prototype validation, Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and distributed multi-node execution.

---

## 1. Prototype Pretraining (Constrained Hardware)

For local development on consumer hardware (16 GB RAM, 6-8 GB VRAM or CPU), use the 150M prototype:

```bash
# Pretrain 150M prototype for 5 steps on local binary shards
python scripts/run_prototype_train.py \
  --data_dir data/shards \
  --output_dir checkpoints/prototype_150m \
  --steps 5 \
  --batch_size 2 \
  --accum_steps 2 \
  --lr 3e-4 \
  --device cpu
```

### Key Training Engine Features
- **Atomic Checkpointing**: Checkpoints write to `.tmp` files first and commit using `os.replace` to prevent corrupted checkpoints if interrupted.
- **Complete RNG Serialization**: Checkpoints serialize PyTorch (CPU & CUDA), Python `random`, and NumPy RNG states to ensure deterministic resumption.
- **Non-Finite Loss & Gradient Handling**: Steps resulting in `NaN` or `Inf` loss or gradients are automatically skipped without corrupting model weights.
- **Hardware-Aware Precision**: Selects `bfloat16` or `float16` with dynamic `GradScaler` when CUDA is available.

---

## 2. Supervised Fine-Tuning (SFT)

To align the prototype or pretrained base checkpoint on instruction and repair data:

```bash
python scripts/run_sft_train.py \
  --base_checkpoint checkpoints/prototype_150m/checkpoint_step_5_final.pt \
  --sft_data data/sft/sft_examples.jsonl \
  --output_dir checkpoints/sft \
  --epochs 3 \
  --lr 5e-5
```

Target masking is applied so loss is only calculated on assistant completion tokens, ignoring prompt tokens.

---

## 3. Direct Preference Optimization (DPO)

DPO optimizes paired preferences (chosen vs. rejected code solutions) without requiring a separate reward model:

```bash
python -m training.preference_optimization.train_dpo \
  --policy_model checkpoints/sft/checkpoint_final.pt \
  --reference_model checkpoints/prototype_150m/checkpoint_step_5_final.pt \
  --preference_data data/dpo/preference_pairs.jsonl \
  --beta 0.1 \
  --lr 1e-6
```

---

## 4. Scaling Path to 50B Distributed MoE

Training a full 50B-class MoE (265.96B total params, 63.02B active params per token) from scratch requires enterprise HPC infrastructure.

### Resource Requirements
- **Cluster Size**: Minimum 32x NVIDIA H100 (80GB) or 64x A100 (80GB).
- **Interconnect**: InfiniBand 400 Gb/s (NDR) or 200 Gb/s (HDR).
- **Sharding Strategy**:
  - **PyTorch FSDP (Full Shard)**: Shards weights, gradients, and optimizer states across GPUs.
  - **Expert Parallelism**: Distributes routed experts across ranks, keeping shared experts replicated.
  - **Activation Checkpointing**: Mandatory to fit 32,768-token sequences into VRAM.

### Pre-flight Hardware Verification
Before launching distributed jobs, run:
```bash
python scripts/check_hardware.py
python scripts/validate_config.py configs/model/moe_50b.yaml
```
If GPU counts or VRAM are insufficient, the distributed training script will abort early with clear diagnostics rather than crashing with out-of-memory errors.
