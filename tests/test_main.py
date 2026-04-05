"""Tests for `ai_exercise/main.py`."""

import httpx
import pytest

from ai_exercise.main import app


@pytest.mark.asyncio
async def test_health_check_route() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_route() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json={"query": "What is the capital of France?"})
        assert response.status_code == 200
