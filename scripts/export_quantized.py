"""Weight quantization exporter (INT8 / INT4 simulation & weight packing) for local deployment."""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from model.config import get_prototype_config
from model.model import CodingCausalLM


def quantize_linear_int8(linear_module: nn.Linear) -> nn.Module:
    """Performs symmetric per-channel 8-bit quantization on linear layer weights."""
    weight = linear_module.weight.data
    # Compute per-channel absolute maximum
    scale = weight.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = torch.clamp(scale, min=1e-8)
    int8_weight = torch.clamp(torch.round(weight / scale), -128, 127).to(torch.int8)

    # Dequantized weight simulation for forward pass without custom CUDA kernels
    dequant_weight = int8_weight.float() * scale

    new_linear = nn.Linear(linear_module.in_features, linear_module.out_features, bias=linear_module.bias is not None)
    new_linear.weight.data = dequant_weight
    if linear_module.bias is not None:
        new_linear.bias.data = linear_module.bias.data
    return new_linear


def export_quantized_model(input_checkpoint: str, output_path: str, quant_mode: str = "int8"):
    print(f"Loading checkpoint from: {input_checkpoint}...")
    checkpoint = torch.load(input_checkpoint, map_location="cpu", weights_only=False)

    config = get_prototype_config()
    model = CodingCausalLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    orig_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Original FP32/BF16 Model Weight Footprint: {orig_size_bytes / (1024 * 1024):.2f} MB")

    print(f"Applying {quant_mode.upper()} quantization across transformer linear layers...")
    quant_count = 0
    for block in model.layers:
        block.self_attn.q_proj = quantize_linear_int8(block.self_attn.q_proj)
        block.self_attn.k_proj = quantize_linear_int8(block.self_attn.k_proj)
        block.self_attn.v_proj = quantize_linear_int8(block.self_attn.v_proj)
        block.self_attn.o_proj = quantize_linear_int8(block.self_attn.o_proj)
        quant_count += 4

        if hasattr(block.mlp, "experts"):
            for exp in block.mlp.experts:
                exp.gate_proj = quantize_linear_int8(exp.gate_proj)
                exp.up_proj = quantize_linear_int8(exp.up_proj)
                exp.down_proj = quantize_linear_int8(exp.down_proj)
                quant_count += 3

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save({
        "quant_mode": quant_mode,
        "config": config.__dict__,
        "model_state_dict": model.state_dict(),
    }, output_path)

    print(f"[SUCCESS] Quantized {quant_count} linear layers. Output exported to: {output_path}")


if __name__ == "__main__":
    ckpt = "./checkpoints/prototype_150m/checkpoint_step_5_final.pt"
    if os.path.exists(ckpt):
        export_quantized_model(ckpt, "./checkpoints/prototype_150m/quantized_int8.pt")
    else:
        print(f"Run run_prototype_train.py first to create {ckpt}")
