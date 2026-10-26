"""Unit tests for Model Architecture, Forward/Backward pass, and MoE routing."""

import pytest
import torch
from model.config import ModelConfig
from model.model import CodingCausalLM


@pytest.fixture
def mini_config():
    """Tiny config for fast unit testing."""
    return ModelConfig(
        model_type="moe_causal_lm",
        model_name="test-mini-moe",
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        rope_theta=10000.0,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
    )


def test_model_forward_pass_and_loss(mini_config):
    model = CodingCausalLM(mini_config)
    batch_size = 2
    seq_len = 16

    input_ids = torch.randint(0, mini_config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    outputs = model(input_ids=input_ids, labels=labels)

    assert "logits" in outputs
    assert "loss" in outputs
    assert outputs["logits"].shape == (batch_size, seq_len, mini_config.vocab_size)
    assert outputs["loss"] is not None
    assert not torch.isnan(outputs["loss"]).any()
    assert outputs["loss"].item() > 0.0


def test_model_backward_pass_gradients(mini_config):
    model = CodingCausalLM(mini_config)
    input_ids = torch.randint(0, mini_config.vocab_size, (2, 8))
    labels = input_ids.clone()

    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs["loss"]
    loss.backward()

    # Verify gradients exist and are non-zero for active layers
    embed_grad = model.embed_tokens.weight.grad
    assert embed_grad is not None
    assert not torch.isnan(embed_grad).any()

    router_grad = model.layers[0].mlp.gate.weight.grad
    assert router_grad is not None
    assert not torch.isnan(router_grad).any()


def test_model_kv_caching(mini_config):
    model = CodingCausalLM(mini_config)
    model.eval()

    # Step 1: Prompt
    prompt_ids = torch.randint(0, mini_config.vocab_size, (1, 4))
    with torch.no_grad():
        out1 = model(input_ids=prompt_ids, use_cache=True)
        kv_cache = out1["kv_caches"]

    assert kv_cache is not None
    assert len(kv_cache) == mini_config.num_hidden_layers

    # Step 2: Next single token with cache
    next_token = torch.tensor([[42]])
    with torch.no_grad():
        out2 = model(input_ids=next_token, kv_caches=kv_cache, use_cache=True)
        logits_step = out2["logits"]

    assert logits_step.shape == (1, 1, mini_config.vocab_size)


def test_parameter_counts(mini_config):
    model = CodingCausalLM(mini_config)
    param_dict = model.count_parameters()

    assert param_dict["total_parameters"] > 0
    assert param_dict["active_parameters"] > 0
    # In MoE with 4 experts and top-2, active parameters must be less than total parameters
    assert param_dict["active_parameters"] < param_dict["total_parameters"]
