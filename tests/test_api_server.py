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
    app = create_app(
        model=model,
        tokenizer=tok,
        api_key="test-secret-key",
        model_id="codex-150m-prototype",
    )
    return TestClient(app)


def test_health_endpoint(test_client):
    res = test_client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["loaded_model_id"] == "codex-150m-prototype"
    assert data["context_length"] == 256
    assert "system_memory_used_mb" in data


def test_models_endpoint(test_client):
    res = test_client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    model_ids = [m["id"] for m in data["data"]]
    assert "codex-150m-prototype" in model_ids
    assert data["data"][0]["status"] == "ready"


def test_auth_rejection(test_client):
    payload = {
        "model": "codex-150m-prototype",
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


def test_anonymous_access_gating():
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

    # App with no key and anonymous disallowed
    app_locked = create_app(
        model=model,
        tokenizer=tok,
        api_key=None,
        allow_anonymous=False,
    )
    client_locked = TestClient(app_locked)
    payload = {
        "model": "codex-150m-prototype",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
    }
    res_locked = client_locked.post("/v1/chat/completions", json=payload)
    assert res_locked.status_code == 401

    # App with no key and anonymous explicitly allowed
    app_open = create_app(
        model=model,
        tokenizer=tok,
        api_key=None,
        allow_anonymous=True,
    )
    client_open = TestClient(app_open)
    res_open = client_open.post("/v1/chat/completions", json=payload)
    assert res_open.status_code == 200


def test_unsupported_model_rejection(test_client):
    payload = {
        "model": "codex-50b-moe",  # Not loaded!
        "messages": [{"role": "user", "content": "Write code"}],
    }
    res = test_client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer test-secret-key"},
    )
    assert res.status_code == 400
    assert "Requested model 'codex-50b-moe' is not loaded" in res.json()["detail"]


def test_context_window_overflow(test_client):
    payload = {
        "model": "codex-150m-prototype",
        "messages": [{"role": "user", "content": "Write code"}],
        "max_tokens": 300,  # Context length is only 256!
    }
    res = test_client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer test-secret-key"},
    )
    assert res.status_code == 400
    assert "exceeds maximum context length" in res.json()["detail"]


def test_chat_completions_json(test_client):
    payload = {
        "model": "codex-150m-prototype",
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
    assert data["model"] == "codex-150m-prototype"


def test_chat_completions_streaming(test_client):
    payload = {
        "model": "codex-150m-prototype",
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

