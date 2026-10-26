"""OpenAI-compatible Pydantic API schemas for completions and streaming."""

import time
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    model: str = "codex-50b-moe"
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    max_tokens: int = 512
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    repetition_penalty: float = 1.05


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


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]


class HealthStatus(BaseModel):
    status: str
    model_name: str
    device: str
    vram_used_mb: float
    system_memory_used_mb: float
