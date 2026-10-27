"""Rotary Positional Embeddings (RoPE) with YaRN / Linear Scaling support."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Precomputes and applies Rotary Position Embeddings (RoPE) to Q and K tensors."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 500000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        self.max_seq_len_cached = seq_len
        dev = device if device is not None else self.inv_freq.device
        dt = dtype if dtype is not None else self.inv_freq.dtype
        t = torch.arange(self.max_seq_len_cached, device=dev, dtype=dt)
        if self.scaling_factor != 1.0:
            t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq.to(device=dev, dtype=dt))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached or x.device != self.cos_cached.device:
            self._set_cos_sin_cache(seq_len=max(seq_len, self.max_seq_len_cached), device=x.device, dtype=self.inv_freq.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embeddings to query and key states with arbitrary position IDs.
    
    Args:
        q: [batch_size, num_heads, seq_len, head_dim]
        k: [batch_size, num_kv_heads, seq_len, head_dim]
        cos: [max_seq_len, head_dim]
        sin: [max_seq_len, head_dim]
        position_ids: [batch_size, seq_len]
    """
    # Guard against position_ids exceeding cached length
    max_pos = int(position_ids.max().item()) if position_ids.numel() > 0 else 0
    if max_pos >= cos.shape[0]:
        dev = cos.device
        dt = cos.dtype
        dim = cos.shape[-1]
        inv_freq = 1.0 / (500000.0 ** (torch.arange(0, dim, 2, device=dev, dtype=torch.float32) / dim))
        t = torch.arange(max_pos + 1, device=dev, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dt)
        sin = emb.sin().to(dtype=dt)

    # Index cos, sin by position_ids: [batch_size, 1, seq_len, head_dim]
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
