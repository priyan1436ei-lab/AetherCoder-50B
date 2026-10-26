"""OpenAI-compatible FastAPI Inference Server with Server-Sent Events (SSE) streaming."""

import json
import os
import uuid
from typing import AsyncIterator, Optional
import psutil
import torch
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    HealthStatus,
    ModelCard,
    ModelList,
    StreamDelta,
    UsageInfo,
)
from api.security import SecurityManager
from inference.generator import CodeGenerator
from model.config import ModelConfig, get_prototype_config
from model.model import CodingCausalLM
from tokenizer.code_tokenizer import CodeTokenizer, create_default_code_tokenizer


def create_app(
    model: Optional[CodingCausalLM] = None,
    tokenizer: Optional[CodeTokenizer] = None,
    api_key: Optional[str] = None,
) -> FastAPI:
    """Builds and configures the FastAPI application instance."""
    app = FastAPI(
        title="Coding LLM 50B Inference Server",
        description="Production OpenAI-compatible API serving the 50B-scale Coding Large Language Model.",
        version="1.0.0",
    )

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    security = SecurityManager(master_key=api_key)

    # Initialize model and generator
    if model is None:
        config = get_prototype_config()
        model = CodingCausalLM(config)
    if tokenizer is None:
        tokenizer = create_default_code_tokenizer()

    generator = CodeGenerator(model=model, tokenizer=tokenizer, device="cpu")

    # Serve static frontend if available
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
    if os.path.exists(frontend_dir):
        app.mount("/ui", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    @app.get("/health", response_model=HealthStatus)
    def health_check():
        mem = psutil.virtual_memory()
        vram_mb = 0.0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)

        return HealthStatus(
            status="healthy",
            model_name=model.config.model_name,
            device=generator.device,
            vram_used_mb=round(vram_mb, 2),
            system_memory_used_mb=round((mem.total - mem.available) / (1024 * 1024), 2),
        )

    @app.get("/v1/models", response_model=ModelList)
    def list_models():
        return ModelList(
            data=[
                ModelCard(id="codex-50b-moe"),
                ModelCard(id="codex-150m-prototype"),
            ]
        )

    def _format_chat_prompt(messages: list) -> str:
        prompt_parts = []
        for msg in messages:
            role = msg.role
            content = msg.content
            prompt_parts.append(f"<|{role}|>\n{content}\n")
        prompt_parts.append("<|assistant|>\n")
        return "".join(prompt_parts)

    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: ChatCompletionRequest,
        raw_req: Request,
        token: str = Depends(security.verify_auth_header),
    ):
        security.check_rate_limit(raw_req.client.host if raw_req.client else "unknown")
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        prompt_text = _format_chat_prompt(req.messages)

        if req.stream:
            async def event_generator():
                for chunk_text in generator.generate_stream(
                    prompt=prompt_text,
                    max_new_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    repetition_penalty=req.repetition_penalty,
                ):
                    chunk = ChatCompletionChunk(
                        id=req_id,
                        model=req.model,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=StreamDelta(content=chunk_text),
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                # Send final stop chunk
                final_chunk = ChatCompletionChunk(
                    id=req_id,
                    model=req.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=StreamDelta(),
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # Synchronous completion
        completed_text = generator.generate(
            prompt=prompt_text,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
        )

        prompt_tokens = len(tokenizer.encode(prompt_text))
        completion_tokens = len(tokenizer.encode(completed_text))

        return ChatCompletionResponse(
            id=req_id,
            model=req.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=completed_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    return app


# Default app entrypoint for uvicorn
app = create_app()
