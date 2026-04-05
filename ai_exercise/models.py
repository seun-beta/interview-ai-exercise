"""Types for the API."""

from pydantic import BaseModel


class HealthRouteOutput(BaseModel):
    """Model for the health route output."""

    status: str


class LoadDocumentsOutput(BaseModel):
    """Model for the load documents route output."""

    status: str


class ChatQuery(BaseModel):
    """Model for the chat input."""

    query: str


class TokenUsage(BaseModel):
    """Token usage for a single LLM call."""

    prompt: int
    completion: int


class ChatOutput(BaseModel):
    """Model for the chat route output."""

    message: str
    token_usage: dict[str, TokenUsage] | None = None
