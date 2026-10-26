"""Load balancing auxiliary loss for Mixture-of-Experts routing stability."""

import torch


def compute_load_balancing_loss(
    router_logits: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    aux_loss_coef: float = 0.01,
    z_loss_coef: float = 0.001,
) -> torch.Tensor:
    """Computes Switch-Transformer style load-balancing auxiliary loss and router z-loss.
    
    Args:
        router_logits: Tensor of shape [num_tokens, num_experts]
        selected_experts: Tensor of shape [num_tokens, top_k]
        num_experts: Total number of routed experts
        aux_loss_coef: Weight multiplier for load balancing loss
        z_loss_coef: Weight multiplier for router logit z-loss
    """
    num_tokens = router_logits.shape[0]
    if num_tokens == 0:
        return torch.tensor(0.0, device=router_logits.device)

    # 1. Router probabilities across all experts
    probs = torch.softmax(router_logits, dim=-1)

    # 2. Compute fraction of tokens dispatched to each expert (f_i)
    # selected_experts shape: [num_tokens, top_k]
    expert_mask = torch.zeros(num_tokens, num_experts, device=router_logits.device)
    expert_mask.scatter_(1, selected_experts, 1.0)
    density = expert_mask.mean(dim=0)  # [num_experts]

    # 3. Compute average routing probability per expert (P_i)
    prob_mean = probs.mean(dim=0)  # [num_experts]

    # Aux loss: num_experts * sum(density_i * prob_mean_i)
    aux_loss = aux_loss_coef * num_experts * torch.sum(density * prob_mean)

    # 4. Router z-loss: prevents router logits from drifting to large values
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = z_loss_coef * torch.mean(log_z ** 2)

    return aux_loss + z_loss
