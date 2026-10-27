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
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward pass with causal masking, loss computation, and KV cache updates.
        
        Args:
            input_ids: [batch_size, seq_len]
            position_ids: [batch_size, seq_len] optional position indices
            attention_mask: [batch_size, seq_len] or [batch_size, 1, seq_len, kv_seq_len] optional mask
            labels: [batch_size, seq_len] optional target token IDs (-100 ignored in loss)
            kv_caches: list of cached (K, V) tuples per layer
            use_cache: whether to output updated KV caches
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # If position_ids is None, determine from KV cache past length to ensure cached decoding aligns
        if position_ids is None:
            past_len = 0
            if kv_caches is not None and len(kv_caches) > 0 and kv_caches[0] is not None:
                past_len = kv_caches[0][0].shape[2]
            position_ids = (past_len + torch.arange(seq_len, dtype=torch.long, device=device)).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_tokens(input_ids)

        max_pos_needed = int(position_ids.max().item()) + 1 if position_ids.numel() > 0 else seq_len
        cos, sin = self.rotary_emb(hidden_states, seq_len=max_pos_needed)

        total_aux_loss = torch.tensor(0.0, device=device)
        new_kv_caches = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            hidden_states, aux_loss, next_cache = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
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
            if shift_labels.numel() == 0:
                loss = total_aux_loss
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                ce_loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
                loss = ce_loss + total_aux_loss


        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss,
            "kv_caches": new_kv_caches,
        }

    def count_parameters(self) -> Dict[str, int]:
        """Calculates total, active, routed, shared, and embedding parameters directly from modules."""
        total_params = sum(p.numel() for p in self.parameters())
        embed_params = sum(p.numel() for p in self.embed_tokens.parameters())
        
        if self.config.model_type == "moe_causal_lm" and len(self.layers) > 0 and hasattr(self.layers[0], "mlp") and hasattr(self.layers[0].mlp, "experts"):
            single_expert_params = sum(p.numel() for p in self.layers[0].mlp.experts[0].parameters())
            total_routed_expert_params = single_expert_params * self.config.num_experts * self.config.num_hidden_layers
            active_routed_expert_params = single_expert_params * self.config.num_experts_per_tok * self.config.num_hidden_layers
            
            shared_expert_params = 0
            if self.layers[0].mlp.shared_experts is not None:
                shared_expert_params = sum(p.numel() for p in self.layers[0].mlp.shared_experts.parameters()) * self.config.num_hidden_layers
                
            inactive_params = total_routed_expert_params - active_routed_expert_params
            active_params = total_params - inactive_params
        else:
            total_routed_expert_params = 0
            shared_expert_params = 0
            inactive_params = 0
            active_params = total_params

        return {
            "total_parameters": total_params,
            "active_parameters": active_params,
            "inactive_parameters": inactive_params,
            "routed_expert_parameters": total_routed_expert_params,
            "shared_expert_parameters": shared_expert_params,
            "embedding_parameters": embed_params,
        }
