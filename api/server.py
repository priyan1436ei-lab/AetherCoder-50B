"""OpenAI-compatible FastAPI Inference Server with Server-Sent Events (SSE) streaming."""

import json
import logging
import os
import uuid
from typing import AsyncIterator, List, Optional
import psutil
import torch
from fastapi import Depends, FastAPI, HTTPException, Request, status
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

logger = logging.getLogger("coding_llm.api")


def create_app(
    model: Optional[CodingCausalLM] = None,
    tokenizer: Optional[CodeTokenizer] = None,
    api_key: Optional[str] = None,
    allow_anonymous: Optional[bool] = None,
    checkpoint_path: Optional[str] = None,
    model_id: str = "codex-150m-prototype",
) -> FastAPI:
    """Builds and configures the FastAPI application instance."""
    app = FastAPI(
        title="AetherCoder Inference Server",
        description="OpenAI-compatible API serving coding model checkpoints with SSE streaming.",
        version="1.0.0",
    )

    # CORS configuration
    cors_env = os.environ.get(
        "AETHERCODER_CORS_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://127.0.0.1:3000",
    )
    cors_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
    if "*" in cors_origins:
        cors_origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    security = SecurityManager(master_key=api_key, allow_anonymous=allow_anonymous)

    # Resolve checkpoint
    ckpt_to_load = checkpoint_path or os.environ.get("AETHERCODER_CHECKPOINT")

    # Initialize model and generator
    if model is None:
        config = get_prototype_config()
        model = CodingCausalLM(config)
        if ckpt_to_load and os.path.exists(ckpt_to_load):
            logger.info("Loading model weights from checkpoint: %s", ckpt_to_load)
            state = torch.load(ckpt_to_load, map_location="cpu", weights_only=True)
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)

    if tokenizer is None:
        tokenizer = create_default_code_tokenizer()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = CodeGenerator(model=model, tokenizer=tokenizer, device=device)
    current_model_id = model_id

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
            loaded_model_id=current_model_id,
            checkpoint_path=ckpt_to_load,
            device=generator.device,
            context_length=model.config.max_position_embeddings,
            vram_used_mb=round(vram_mb, 2),
            system_memory_used_mb=round((mem.total - mem.available) / (1024 * 1024), 2),
        )

    @app.get("/v1/models", response_model=ModelList)
    def list_models():
        return ModelList(
            data=[
                ModelCard(
                    id=current_model_id,
                    status="ready",
                    context_length=model.config.max_position_embeddings,
                )
            ]
        )

    def _format_chat_prompt(messages: List[ChatMessage]) -> str:
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
        client_host = raw_req.client.host if raw_req.client else "unknown"
        security.check_rate_limit(client_host)

        # Validate requested model ID
        allowed_aliases = {current_model_id, "default", "prototype", model.config.model_name}
        if req.model not in allowed_aliases:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Requested model '{req.model}' is not loaded. "
                    f"Currently available model: '{current_model_id}'."
                ),
            )

        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        prompt_text = _format_chat_prompt(req.messages)
        prompt_tokens = len(tokenizer.encode(prompt_text))

        # Context window enforcement
        max_ctx = model.config.max_position_embeddings
        if prompt_tokens + req.max_tokens > max_ctx:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Requested sequence length ({prompt_tokens} prompt + {req.max_tokens} max_tokens = "
                    f"{prompt_tokens + req.max_tokens}) exceeds maximum context length ({max_ctx})."
                ),
            )

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
                        model=current_model_id,
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
                    model=current_model_id,
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

        completion_tokens = len(tokenizer.encode(completed_text))

        return ChatCompletionResponse(
            id=req_id,
            model=current_model_id,
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

