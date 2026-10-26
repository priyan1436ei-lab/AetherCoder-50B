"""Direct Preference Optimization (DPO) pipeline for code preference alignment."""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model import CodingCausalLM


class DPOTrainer:
    """Direct Preference Optimization trainer comparing chosen vs rejected code snippets."""

    def __init__(
        self,
        policy_model: CodingCausalLM,
        reference_model: CodingCausalLM,
        beta: float = 0.1,
        learning_rate: float = 5e-5,
        device: str = "cpu",
    ):
        self.policy_model = policy_model.to(device)
        self.reference_model = reference_model.to(device)
        self.reference_model.eval()
        for p in self.reference_model.parameters():
            p.requires_grad = False

        self.beta = beta
        self.device = device
        self.optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)

    @staticmethod
    def compute_log_probs(model: CodingCausalLM, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Computes per-sequence sum of log probabilities for label tokens."""
        outputs = model(input_ids=input_ids)
        logits = outputs["logits"]

        # Shift logits and labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_mask = shift_labels != -100
        # Replace -100 with 0 for gather
        shift_labels_safe = shift_labels.clone()
        shift_labels_safe[shift_labels == -100] = 0

        per_token_logps = torch.gather(
            F.log_softmax(shift_logits, dim=-1),
            dim=2,
            index=shift_labels_safe.unsqueeze(2),
        ).squeeze(2)

        return (per_token_logps * loss_mask).sum(dim=-1)

    def train_step(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """Calculates DPO loss, policy rewards, and takes an optimizer step."""
        self.policy_model.train()
        chosen_input_ids = chosen_input_ids.to(self.device)
        chosen_labels = chosen_labels.to(self.device)
        rejected_input_ids = rejected_input_ids.to(self.device)
        rejected_labels = rejected_labels.to(self.device)

        # Policy log probabilities
        pi_chosen = self.compute_log_probs(self.policy_model, chosen_input_ids, chosen_labels)
        pi_rejected = self.compute_log_probs(self.policy_model, rejected_input_ids, rejected_labels)

        # Reference log probabilities (no gradients)
        with torch.no_grad():
            ref_chosen = self.compute_log_probs(self.reference_model, chosen_input_ids, chosen_labels)
            ref_rejected = self.compute_log_probs(self.reference_model, rejected_input_ids, rejected_labels)

        # Implicit rewards
        chosen_rewards = self.beta * (pi_chosen - ref_chosen)
        rejected_rewards = self.beta * (pi_rejected - ref_rejected)

        # DPO loss
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item(), chosen_rewards.mean().item(), rejected_rewards.mean().item()
