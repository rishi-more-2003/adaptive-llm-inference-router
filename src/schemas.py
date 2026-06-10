from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] | str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.7, ge=0.0)
    stream: bool = False
    latency_target_ms: int | None = None
    batch_size: int = Field(default=1, ge=1)


class CompletionRequest(BaseModel):
    model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    prompt: str | list[str]
    max_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.7, ge=0.0)
    stream: bool = False
    latency_target_ms: int | None = None
    batch_size: int | None = Field(default=None, ge=1)


class RouteInspectRequest(BaseModel):
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    max_tokens: int = Field(default=128, ge=1)
    latency_target_ms: int | None = None
    batch_size: int = Field(default=1, ge=1)
    update_prefix_cache: bool = False


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str = "stop"


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[CompletionChoice | ChatCompletionChoice]
    usage: Usage
    routing: dict[str, Any]
    metrics: dict[str, Any]


def chat_messages_to_prompt(messages: list[ChatMessage]) -> str:
    rendered = []
    for message in messages:
        role = message.role if message.role in {"system", "user", "assistant"} else "user"
        rendered.append(f"<|{role}|>\n{message.content}</s>")
    rendered.append("<|assistant|>\n")
    return "\n".join(rendered)
