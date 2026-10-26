"""Streaming autoregressive token generation engine with temperature, top-p, and penalty controls."""

import time
from typing import AsyncIterator, Iterator, List, Optional, Set
import torch
import torch.nn.functional as F

from model.model import CodingCausalLM
from tokenizer.code_tokenizer import CodeTokenizer


class CodeGenerator:
    """Production inference generator supporting streaming and full completions."""

    def __init__(self, model: CodingCausalLM, tokenizer: CodeTokenizer, device: str = "cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        generated_tokens: Optional[List[int]] = None,
    ) -> int:
        """Applies repetition penalty, temperature scaling, top-k, and nucleus top-p filtering."""
        logits = logits.squeeze(0)  # [vocab_size]

        # Apply repetition penalty
        if repetition_penalty != 1.0 and generated_tokens:
            for token_id in set(generated_tokens):
                if logits[token_id] < 0:
                    logits[token_id] *= repetition_penalty
                else:
                    logits[token_id] /= repetition_penalty

        if temperature <= 0.0:
            return int(torch.argmax(logits).item())

        logits = logits / temperature

        # Top-K filtering
        if top_k > 0:
            top_k_val = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k_val)[0][..., -1, None]
            logits[indices_to_remove] = -float("Inf")

        # Top-P (Nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        return int(next_token)

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        stop_tokens: Optional[Set[int]] = None,
    ) -> Iterator[str]:
        """Synchronous generator yielding text chunks as tokens are generated."""
        input_ids = self.tokenizer.encode(prompt)
        if not input_ids:
            input_ids = [self.tokenizer.bos_token_id or 1]

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        generated_ids: List[int] = []

        stop_token_ids = stop_tokens or set()
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.add(self.tokenizer.eos_token_id)

        # Initial forward pass with prompt
        with torch.no_grad():
            outputs = self.model(input_ids=input_tensor, use_cache=True)
            kv_caches = outputs["kv_caches"]
            next_logits = outputs["logits"][:, -1, :]

        for _ in range(max_new_tokens):
            next_token = self._sample_next_token(
                next_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                generated_tokens=generated_ids,
            )

            if next_token in stop_token_ids:
                break

            generated_ids.append(next_token)
            chunk_text = self.tokenizer.decode([next_token])
            yield chunk_text

            # Single-token step forward using KV cache
            next_input = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            with torch.no_grad():
                step_out = self.model(input_ids=next_input, kv_caches=kv_caches, use_cache=True)
                kv_caches = step_out["kv_caches"]
                next_logits = step_out["logits"][:, -1, :]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
    ) -> str:
        """Convenience method returning the complete generated string."""
        return "".join(
            self.generate_stream(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        )
