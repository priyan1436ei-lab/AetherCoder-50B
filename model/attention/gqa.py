"""Grouped-Query Attention (GQA) with KV-cache support and SDPA kernel integration."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention.rope import apply_rotary_pos_emb


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expands KV heads to match query heads in Grouped-Query Attention."""
    if n_rep == 1:
        return hidden_states
    batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention with scalable KV caching and FlashAttention/SDPA."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Update and concatenate KV cache if provided
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            key_states = torch.cat([prev_k, key_states], dim=2)
            value_states = torch.cat([prev_v, value_states], dim=2)

        current_kv_cache = (key_states, value_states) if use_cache else None

        # Repeat KV heads to match query heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        total_kv_seq_len = key_states.shape[2]

        # Determine causal masking vs custom attention mask
        if attention_mask is not None:
            # Process custom mask to match [batch_size, 1, seq_len, total_kv_seq_len]
            if attention_mask.dim() == 2:
                # [batch, kv_seq_len] -> [batch, 1, 1, kv_seq_len]
                attn_mask = attention_mask[:, None, None, :]
                if attn_mask.dtype in (torch.int64, torch.int32, torch.bool):
                    attn_mask = (1.0 - attn_mask.float()) * -1e9
            else:
                attn_mask = attention_mask
            is_causal = False
        else:
            attn_mask = None
            # Only use SDPA is_causal when seq_len == total_kv_seq_len (square prefill matrix)
            is_causal = (seq_len > 1 and seq_len == total_kv_seq_len)

        try:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=is_causal,
            )
        except Exception:
            # Numerically stable manual fallback for systems/masks where SDPA is unsupported
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale
            if is_causal:
                causal_mask = torch.triu(
                    torch.full((seq_len, total_kv_seq_len), float("-inf"), device=query_states.device),
                    diagonal=1,
                )
                attn_weights = attn_weights + causal_mask[None, None, :, :]
            elif attn_mask is not None:
                attn_weights = attn_weights + attn_mask
            attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_probs, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)
        return output, current_kv_cache
