"""OpenAI-compatible Pydantic API schemas for completions and streaming."""

import time
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


RoleType = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    role: RoleType
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    model: str = "codex-150m-prototype"
    messages: List[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=1)
    max_tokens: int = Field(512, ge=1, le=8192)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    repetition_penalty: float = Field(1.05, ge=0.1, le=3.0)


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class StreamDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: StreamDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChunkChoice]


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "coding-llm"
    status: str = "ready"
    context_length: Optional[int] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]


class HealthStatus(BaseModel):
    status: str
    model_name: str
    loaded_model_id: str
    checkpoint_path: Optional[str] = None
    device: str
    context_length: int
    vram_used_mb: float
    system_memory_used_mb: float

