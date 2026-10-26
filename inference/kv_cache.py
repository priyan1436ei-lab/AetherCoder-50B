"""Dynamic Key-Value cache manager for continuous autoregressive decoding."""

from typing import List, Optional, Tuple
import torch


class DynamicKVCache:
    """Stores key and value states per layer across generation steps."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * num_layers

    def get_cache(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self.caches[layer_idx]

    def update(self, layer_idx: int, key_state: torch.Tensor, value_state: torch.Tensor) -> None:
        self.caches[layer_idx] = (key_state, value_state)

    def reset(self) -> None:
        self.caches = [None] * self.num_layers

    @property
    def seq_length(self) -> int:
        if self.caches[0] is not None:
            return self.caches[0][0].shape[2]
        return 0
