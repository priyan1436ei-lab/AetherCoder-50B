"""Strongly typed model configurations for Dense and Sparse MoE Coding Models."""

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    """Configuration for Causal Language Models with GQA, RoPE, SwiGLU, and MoE."""

    model_type: str = "moe_causal_lm"  # "moe_causal_lm" or "dense_causal_lm"
    model_name: str = "codex-moe"
    vocab_size: int = 65536
    hidden_size: int = 6144
    num_hidden_layers: int = 48
    intermediate_size: int = 16384
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: Optional[int] = None
    max_position_embeddings: int = 32768
    rope_theta: float = 500000.0
    rope_scaling_type: Optional[str] = "yarn"
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02

    # MoE specific parameters
    num_experts: int = 16
    num_experts_per_tok: int = 2
    num_shared_experts: int = 2
    moe_router_aux_loss_coef: float = 0.01
    moe_z_loss_coef: float = 0.001

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        return cls(**cfg_dict)

    def to_yaml(self, yaml_path: str) -> None:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.__dict__, f, indent=2)


def get_prototype_config() -> ModelConfig:
    """Returns the lightweight 150M parameter config suitable for local validation."""
    return ModelConfig(
        model_type="moe_causal_lm",
        model_name="codex-150m-prototype",
        vocab_size=65536,
        hidden_size=512,
        num_hidden_layers=8,
        intermediate_size=1376,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=4096,
        rope_theta=100000.0,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
    )
