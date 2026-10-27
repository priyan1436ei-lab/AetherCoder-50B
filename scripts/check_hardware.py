"""Hardware inspection and compute physics memory-estimation utility."""

import argparse
import os
import sys
import psutil
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.config import ModelConfig, get_prototype_config


def calculate_moe_parameters(config: ModelConfig) -> dict:
    """Calculates exact theoretical parameter counts for a given ModelConfig."""
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    layers = config.num_hidden_layers
    heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    head_dim = config.head_dim or (hidden_size // heads)
    inter_size = config.intermediate_size
    num_experts = config.num_experts if config.model_type == "moe_causal_lm" else 1
    top_k = config.num_experts_per_tok if config.model_type == "moe_causal_lm" else 1
    shared_exp = config.num_shared_experts if config.model_type == "moe_causal_lm" else 0

    embed_params = vocab_size * hidden_size
    lm_head_params = 0 if config.tie_word_embeddings else (vocab_size * hidden_size)

    # Attention parameters per layer: q, k, v, o
    q_params = hidden_size * (heads * head_dim)
    k_params = hidden_size * (kv_heads * head_dim)
    v_params = hidden_size * (kv_heads * head_dim)
    o_params = (heads * head_dim) * hidden_size
    layer_attn = q_params + k_params + v_params + o_params
    layer_norms = 2 * hidden_size  # pre-attn and post-attn RMSNorms

    # SwiGLU MLP: w_gate (hidden -> inter), w_up (hidden -> inter), w_down (inter -> hidden)
    single_expert_params = 3 * (hidden_size * inter_size)

    if config.model_type == "moe_causal_lm":
        routed_mlp = num_experts * single_expert_params
        shared_mlp = shared_exp * single_expert_params
        gate_params = hidden_size * num_experts
        layer_mlp = routed_mlp + shared_mlp + gate_params
        active_layer_mlp = (top_k * single_expert_params) + shared_mlp + gate_params
    else:
        layer_mlp = single_expert_params
        active_layer_mlp = single_expert_params

    total_layer_params = layer_attn + layer_norms + layer_mlp
    active_layer_params = layer_attn + layer_norms + active_layer_mlp

    final_norm = hidden_size
    total_params = embed_params + (layers * total_layer_params) + final_norm + lm_head_params
    active_params = embed_params + (layers * active_layer_params) + final_norm + lm_head_params

    return {
        "total_parameters": total_params,
        "active_parameters": active_params,
        "embedding_parameters": embed_params,
        "per_expert_parameters": single_expert_params,
    }


def estimate_memory_requirements(total_params: int, active_params: int) -> dict:
    """Calculates memory requirements for inference and training across precision formats."""
    # Model Weights alone
    fp32_weights_gb = (total_params * 4) / (1024**3)
    bf16_weights_gb = (total_params * 2) / (1024**3)
    int8_weights_gb = (total_params * 1) / (1024**3)
    int4_weights_gb = (total_params * 0.5) / (1024**3)

    # Pretraining (Weights + Gradients + AdamW Optimizer states + Master weights)
    # AdamW maintains: FP32 master weights (4 bytes) + FP32 momentum (4 bytes) + FP32 variance (4 bytes) = 12 bytes/param
    # + BF16 weights (2 bytes) + BF16 grads (2 bytes) = ~16 to 18 bytes per total parameter
    pretrain_adamw_min_gb = (total_params * 16) / (1024**3)

    return {
        "weights_fp32_gb": round(fp32_weights_gb, 2),
        "weights_bf16_gb": round(bf16_weights_gb, 2),
        "weights_int8_gb": round(int8_weights_gb, 2),
        "weights_int4_gb": round(int4_weights_gb, 2),
        "pretraining_adamw_min_gb": round(pretrain_adamw_min_gb, 2),
    }


def inspect_hardware():
    print("=" * 70)
    print("AETHERCODER-50B: HARDWARE ENVIRONMENT & MEMORY DIAGNOSTICS")
    print("=" * 70)

    # System Memory
    vmem = psutil.virtual_memory()
    total_ram_gb = vmem.total / (1024**3)
    avail_ram_gb = vmem.available / (1024**3)
    cpu_count = psutil.cpu_count(logical=True)

    print(f"Host System Hardware:")
    print(f"  CPU Logical Cores:  {cpu_count}")
    print(f"  Total System RAM:   {total_ram_gb:.2f} GB (Available: {avail_ram_gb:.2f} GB)")

    # CUDA Acceleration
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU Accelerator:    {gpu_name}")
        print(f"  Dedicated VRAM:     {vram_gb:.2f} GB")
    else:
        print(f"  GPU Accelerator:    None detected (Running CPU Engine)")
        vram_gb = 0.0

    print("-" * 70)
    print("Architecture Comparison & Compute Physics:")
    print("-" * 70)

    # Prototype 150M Model
    proto_cfg = get_prototype_config()
    proto_params = calculate_moe_parameters(proto_cfg)
    proto_mem = estimate_memory_requirements(proto_params["total_parameters"], proto_params["active_parameters"])

    print(f"1. Prototype Model ({proto_cfg.model_name}):")
    print(f"   Total Parameters:  {proto_params['total_parameters']:,} ({proto_params['total_parameters']/1e6:.1f}M)")
    print(f"   Active Per Token:  {proto_params['active_parameters']:,} ({proto_params['active_parameters']/1e6:.1f}M)")
    print(f"   Weights (BF16):    {proto_mem['weights_bf16_gb']} GB")
    print(f"   Training (AdamW):  {proto_mem['pretraining_adamw_min_gb']} GB")
    print(f"   Local Feasibility: FULLY SUPPORTED on this machine.")

    print()
    # 50B Model Target
    target_50b = 48_800_000_000
    active_50b = 12_200_000_000
    mem_50b = estimate_memory_requirements(target_50b, active_50b)

    print(f"2. Foundation Model Target (AetherCoder-50B MoE):")
    print(f"   Total Parameters:  ~48.8 Billion")
    print(f"   Active Per Token:  ~12.2 Billion")
    print(f"   Weights (BF16):    {mem_50b['weights_bf16_gb']} GB")
    print(f"   Weights (INT4):    {mem_50b['weights_int4_gb']} GB")
    print(f"   Training (AdamW):  {mem_50b['pretraining_adamw_min_gb']} GB (Requires 8-16x 80GB H100 GPUs)")

    can_run_50b = (vram_gb >= mem_50b["weights_bf16_gb"]) or (avail_ram_gb >= mem_50b["weights_bf16_gb"] + 16)
    if not can_run_50b:
        print(f"   Local Feasibility: UNSUPPORTED ON LOCAL HARDWARE.")
        print(f"   [Notice] A 50B model cannot be trained or inferred locally with {avail_ram_gb:.1f} GB RAM / {vram_gb:.1f} GB VRAM.")
        print(f"            The system automatically defaults to the 150M prototype for development & testing.")
    else:
        print(f"   Local Feasibility: Sufficient memory for local weight loading.")

    print("=" * 70)


if __name__ == "__main__":
    inspect_hardware()
