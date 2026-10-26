"""Complete Causal Language Model container supporting Dense and Sparse MoE architectures."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig
from model.norm.rmsnorm import RMSNorm
from model.attention.rope import RotaryEmbedding
from model.transformer_block import TransformerBlock


class CodingCausalLM(nn.Module):
    """Production-grade Coding Causal Language Model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Rotary Positional Embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ])

        # Final RMSNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward pass with causal masking, loss computation, and KV cache updates.
        
        Args:
            input_ids: [batch_size, seq_len]
            labels: [batch_size, seq_len] optional target token IDs
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden_states, seq_len=self.config.max_position_embeddings)

        total_aux_loss = torch.tensor(0.0, device=device)
        new_kv_caches = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            hidden_states, aux_loss, next_cache = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
                kv_cache=layer_cache,
                use_cache=use_cache,
            )
            total_aux_loss = total_aux_loss + aux_loss
            if use_cache:
                new_kv_caches.append(next_cache)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            loss = ce_loss + total_aux_loss

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss,
            "kv_caches": new_kv_caches,
        }

    def count_parameters(self) -> Dict[str, int]:
        """Calculates total, active, and embedding parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        embed_params = sum(p.numel() for p in self.embed_tokens.parameters())
        
        # Calculate active parameters for MoE
        if self.config.model_type == "moe_causal_lm":
            # In MoE: router selects num_experts_per_tok out of num_experts
            single_expert_params = sum(p.numel() for p in self.layers[0].mlp.experts[0].parameters())
            shared_experts_params = sum(p.numel() for p in self.layers[0].mlp.shared_experts.parameters()) if self.layers[0].mlp.shared_experts else 0
            inactive_expert_params_per_layer = single_expert_params * (self.config.num_experts - self.config.num_experts_per_tok)
            total_inactive_params = inactive_expert_params_per_layer * self.config.num_hidden_layers
            active_params = total_params - total_inactive_params
        else:
            active_params = total_params

        return {
            "total_parameters": total_params,
            "active_parameters": active_params,
            "embedding_parameters": embed_params,
        }
