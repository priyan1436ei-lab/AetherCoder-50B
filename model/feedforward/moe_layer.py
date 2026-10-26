"""Mixture of Experts (MoE) Layer with Top-K Gating, Load Balancing, and Shared Experts."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.feedforward.swiglu import SwiGLUMLP
from model.losses.load_balancing import compute_load_balancing_loss


class MoELayer(nn.Module):
    """Sparse Mixture-of-Experts Layer with Top-K Gating and Shared Experts."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 16,
        num_experts_per_tok: int = 2,
        num_shared_experts: int = 2,
        aux_loss_coef: float = 0.01,
        z_loss_coef: float = 0.001,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.aux_loss_coef = aux_loss_coef
        self.z_loss_coef = z_loss_coef

        # Gating router network
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Routed Experts pool
        self.experts = nn.ModuleList([
            SwiGLUMLP(hidden_size, intermediate_size) for _ in range(num_experts)
        ])

        # Shared Experts (always active for general syntax and foundational code reasoning)
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                SwiGLUMLP(hidden_size, intermediate_size) for _ in range(num_shared_experts)
            ])
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through router and dispatched experts.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            output: [batch_size, seq_len, hidden_size]
            aux_loss: Scalar auxiliary load balancing loss
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_dim)  # [N, hidden_dim]

        # 1. Compute router logits and Top-K expert indices
        router_logits = self.gate(flat_hidden)  # [N, num_experts]
        router_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(router_weights, self.num_experts_per_tok, dim=-1)

        # Normalize topk weights so they sum to 1.0 per token
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # 2. Compute auxiliary load balancing loss
        if self.training:
            aux_loss = compute_load_balancing_loss(
                router_logits=router_logits,
                selected_experts=topk_indices,
                num_experts=self.num_experts,
                aux_loss_coef=self.aux_loss_coef,
                z_loss_coef=self.z_loss_coef,
            )
        else:
            aux_loss = torch.tensor(0.0, device=hidden_states.device)

        # 3. Dispatch tokens to selected experts
        final_output = torch.zeros_like(flat_hidden)

        for k in range(self.num_experts_per_tok):
            expert_idx_k = topk_indices[:, k]
            weight_k = topk_weights[:, k].unsqueeze(-1)

            for e_idx in range(self.num_experts):
                token_mask = (expert_idx_k == e_idx)
                if token_mask.any():
                    selected_tokens = flat_hidden[token_mask]
                    expert_out = self.experts[e_idx](selected_tokens)
                    final_output[token_mask] += weight_k[token_mask] * expert_out

        # 4. Add Shared Experts output if present
        if self.shared_experts is not None:
            for shared_exp in self.shared_experts:
                final_output += shared_exp(flat_hidden)

        return final_output.view(batch_size, seq_len, hidden_dim), aux_loss
