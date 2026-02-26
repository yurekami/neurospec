"""FastAPI-based serving layer for NeuroSpec.

Provides an HTTP API for generating text with behavioral enforcement.
Compatible with OpenAI-style chat completion format.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


def start_server(engine: Any, host: str = "0.0.0.0", port: int = 8420) -> None:
    """Start the NeuroSpec serving API.

    Args:
        engine: A loaded NeuroSpecEngine instance.
        host: Host to bind to.
        port: Port to listen on.
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI and uvicorn are required for serving. "
            "Install with: pip install neurospec[runtime]"
        )

    import uvicorn

    app = _create_app(engine)
    logger.info("Starting NeuroSpec server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


def _create_app(engine: Any) -> Any:
    """Create the FastAPI application."""
    app = FastAPI(
        title="NeuroSpec API",
        description="Behavioral enforcement for neural network inference",
        version="0.1.0",
    )

    # Request/response models (defined here to avoid import issues when FastAPI unavailable)
    class GenerateRequest(BaseModel):
        prompt: str
        max_new_tokens: int = 256
        temperature: float = 0.7
        top_p: float = 0.9

    class GenerateResponse(BaseModel):
        text: str
        spec_name: str = ""
        monitors_triggered: list[str] = []

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        messages: list[ChatMessage]
        max_tokens: int = 256
        temperature: float = 0.7
        top_p: float = 0.9

    class ChatCompletionChoice(BaseModel):
        index: int = 0
        message: ChatMessage
        finish_reason: str = "stop"

    class ChatCompletionResponse(BaseModel):
        id: str = "neurospec-0"
        object: str = "chat.completion"
        choices: list[ChatCompletionChoice]

    class HealthResponse(BaseModel):
        status: str
        model: str
        spec: str

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok" if engine.is_loaded else "loading",
            model=engine._model_id,
            spec=engine.spec.spec_name if engine.spec else "none",
        )

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        try:
            text = engine.generate(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            return GenerateResponse(
                text=text,
                spec_name=engine.spec.spec_name if engine.spec else "",
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
        """OpenAI-compatible chat completion endpoint."""
        # Concatenate messages into a prompt
        prompt_parts = []
        for msg in request.messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        prompt = "\n".join(prompt_parts) + "\nAssistant:"

        try:
            text = engine.generate(
                prompt=prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            return ChatCompletionResponse(
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessage(role="assistant", content=text),
                    )
                ],
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return app
