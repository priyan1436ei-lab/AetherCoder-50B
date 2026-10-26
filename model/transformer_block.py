"""Transformer Decoder Block with Pre-RMSNorm and Residual Connections."""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from model.config import ModelConfig
from model.norm.rmsnorm import RMSNorm
from model.attention.gqa import GroupedQueryAttention
from model.feedforward.swiglu import SwiGLUMLP
from model.feedforward.moe_layer import MoELayer


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with GQA attention and SwiGLU / MoE feedforward."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        )

        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.model_type == "moe_causal_lm":
            self.mlp = MoELayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                num_shared_experts=config.num_shared_experts,
                aux_loss_coef=config.moe_router_aux_loss_coef,
                z_loss_coef=config.moe_z_loss_coef,
            )
            self.is_moe = True
        else:
            self.mlp = SwiGLUMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
            )
            self.is_moe = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # 1. Attention sub-layer with Pre-RMSNorm
        residual = hidden_states
        normed_states = self.input_layernorm(hidden_states)
        attn_out, new_kv_cache = self.self_attn(
            hidden_states=normed_states,
            position_ids=position_ids,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_out

        # 2. Feed-Forward / MoE sub-layer with Pre-RMSNorm
        residual = hidden_states
        normed_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            mlp_out, aux_loss = self.mlp(normed_states)
        else:
            mlp_out = self.mlp(normed_states)
            aux_loss = torch.tensor(0.0, device=hidden_states.device)

        hidden_states = residual + mlp_out
        return hidden_states, aux_loss, new_kv_cache
