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
    assert param_dict["routed_expert_parameters"] > 0
    assert param_dict["shared_expert_parameters"] > 0
    assert param_dict["active_parameters"] < param_dict["total_parameters"]


def test_config_validation_rules():
    # Test divisibility failure
    with pytest.raises(ValueError, match="divisible by num_attention_heads"):
        ModelConfig(hidden_size=100, num_attention_heads=8)

    # Test KV heads divisibility failure
    with pytest.raises(ValueError, match="divisible by num_key_value_heads"):
        ModelConfig(hidden_size=128, num_attention_heads=8, num_key_value_heads=3)

    # Test top-k > num_experts
    with pytest.raises(ValueError, match="num_experts_per_tok"):
        ModelConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=2,
            num_experts_per_tok=4,
        )


def test_causal_masking_isolation(mini_config):
    """Verifies causal attention: future tokens cannot influence earlier-token logits."""
    model = CodingCausalLM(mini_config)
    model.eval()

    seq1 = torch.tensor([[10, 20, 30, 40, 50]])
    seq2 = torch.tensor([[10, 20, 30, 99, 88]])  # Same prefix, different suffix

    with torch.no_grad():
        out1 = model(input_ids=seq1)["logits"]
        out2 = model(input_ids=seq2)["logits"]

    # Positions 0, 1, 2 must produce numerically identical logits
    diff = torch.abs(out1[:, :3, :] - out2[:, :3, :]).max().item()
    assert diff < 1e-4, f"Causal leakage detected! Max difference at prefix: {diff}"


def test_cached_vs_full_sequence_decoding_equivalence(mini_config):
    """Verifies that token-by-token cached decoding matches full-sequence decoding."""
    model = CodingCausalLM(mini_config)
    model.eval()

    input_ids = torch.tensor([[15, 25, 35, 45, 55]])

    # 1. Full-sequence decoding
    with torch.no_grad():
        full_logits = model(input_ids=input_ids)["logits"]

    # 2. Incremental cached decoding
    with torch.no_grad():
        # Prefill first 2 tokens
        prefill_out = model(input_ids=input_ids[:, :2], use_cache=True)
        kv_cache = prefill_out["kv_caches"]
        cached_logits = [prefill_out["logits"][:, i:i+1, :] for i in range(2)]

        # Decode tokens 2, 3, 4 one-by-one
        for t in range(2, 5):
            tok = input_ids[:, t:t+1]
            step_out = model(input_ids=tok, kv_caches=kv_cache, use_cache=True)
            kv_cache = step_out["kv_caches"]
            cached_logits.append(step_out["logits"])

        cached_logits = torch.cat(cached_logits, dim=1)

    max_diff = torch.abs(full_logits - cached_logits).max().item()
    assert max_diff < 1e-3, f"Cached decoding diverged from full sequence! Max diff: {max_diff}"


def test_moe_zero_token_expert_safety(mini_config):
    """Verifies that MoE layer handles experts receiving 0 tokens gracefully."""
    model = CodingCausalLM(mini_config)
    # A single token with top-2 leaves at least 2 experts with 0 tokens
    single_token = torch.tensor([[10]])
    out = model(input_ids=single_token, labels=single_token)
    loss = out["loss"]
    loss.backward()
    assert not torch.isnan(loss).any()
