"""Unit and integration tests for the OpenAI-compatible FastAPI Inference Server."""

import pytest
from starlette.testclient import TestClient
from api.server import create_app
from model.config import ModelConfig
from model.model import CodingCausalLM
from tokenizer.code_tokenizer import create_default_code_tokenizer


@pytest.fixture
def test_client():
    cfg = ModelConfig(
        model_type="moe_causal_lm",
        vocab_size=500,
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        max_position_embeddings=256,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
    )
    model = CodingCausalLM(cfg)
    tok = create_default_code_tokenizer()
    app = create_app(model=model, tokenizer=tok, api_key="test-secret-key")
    return TestClient(app)


def test_health_endpoint(test_client):
    res = test_client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "system_memory_used_mb" in data


def test_models_endpoint(test_client):
    res = test_client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    model_ids = [m["id"] for m in data["data"]]
    assert "codex-50b-moe" in model_ids


def test_auth_rejection(test_client):
    payload = {
        "model": "codex-50b-moe",
        "messages": [{"role": "user", "content": "print('hello')"}],
    }
    # No Auth Header
    res = test_client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 401

    # Invalid Auth Header
    res_bad = test_client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert res_bad.status_code == 403


def test_chat_completions_json(test_client):
    payload = {
        "model": "codex-50b-moe",
        "messages": [{"role": "user", "content": "Write a python function."}],
        "max_tokens": 4,
        "stream": False,
    }
    res = test_client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer test-secret-key"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "usage" in data


def test_chat_completions_streaming(test_client):
    payload = {
        "model": "codex-50b-moe",
        "messages": [{"role": "user", "content": "Write a python function."}],
        "max_tokens": 4,
        "stream": True,
    }
    res = test_client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer test-secret-key"},
    )
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    assert "data: [DONE]" in res.text
