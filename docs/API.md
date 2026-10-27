# AetherCoder-50B Inference & OpenAI-Compatible API

AetherCoder provides a high-throughput, OpenAI-compatible FastAPI inference server supporting streaming completions via Server-Sent Events (SSE).

---

## 1. Starting the Server

```bash
# Start server with the local prototype checkpoint
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

### Environment Variables
Configure the server via environment variables or a `.env` file (see `.env.example`):
- `AETHERCODER_API_KEY`: Master Bearer token required for protected endpoints.
- `AETHERCODER_ALLOW_ANONYMOUS`: Set to `1` to enable unauthenticated local development.
- `AETHERCODER_CORS_ORIGINS`: Comma-separated list of allowed origins (default: safe local origins).
- `AETHERCODER_CHECKPOINT`: Optional path to `.pt` checkpoint file to load at startup.

---

## 2. API Endpoints

### `GET /health`
Returns live system health, active device, and memory telemetry.

**Response:**
```json
{
  "status": "healthy",
  "model_name": "codex-150m-prototype",
  "loaded_model_id": "codex-150m-prototype",
  "checkpoint_path": "checkpoints/prototype_150m/checkpoint_step_5_final.pt",
  "device": "cpu",
  "context_length": 4096,
  "vram_used_mb": 0.0,
  "system_memory_used_mb": 8412.35
}
```

---

### `GET /v1/models`
Returns list of available models and their context limits.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "codex-150m-prototype",
      "object": "model",
      "created": 1729000000,
      "owned_by": "coding-llm",
      "status": "ready",
      "context_length": 4096
    }
  ]
}
```

---

### `POST /v1/chat/completions`
Generates chat completions using OpenAI standard schema.

**Headers:**
- `Content-Type: application/json`
- `Authorization: Bearer <API_KEY>` (if configured)

**Request Body:**
```json
{
  "model": "codex-150m-prototype",
  "messages": [
    { "role": "system", "content": "You are an expert coding assistant." },
    { "role": "user", "content": "Write a Python function to compute Fibonacci numbers." }
  ],
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 50,
  "max_tokens": 256,
  "stream": false,
  "repetition_penalty": 1.05
}
```

**Model Validation Policy:**
- If the requested `model` does not match the loaded model ID or an approved alias (`"default"`, `"prototype"`), the server responds with **HTTP 400 Bad Request**.
- If `prompt_tokens + max_tokens` exceeds the model's context window, the server rejects the request with **HTTP 400 Bad Request**.

---

### Streaming (`stream: true`)
When `stream: true` is passed, the server streams tokens formatted as SSE:
```text
data: {"id":"chatcmpl-12345","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"def"},"finish_reason":null}]}

data: {"id":"chatcmpl-12345","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" fib"},"finish_reason":null}]}

data: {"id":"chatcmpl-12345","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```
