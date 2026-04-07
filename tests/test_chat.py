from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_chat_endpoint(client):
    mock_result = {"role": "assistant", "content": "Hello!", "model": "llama3"}

    with patch("app.services.ollama.OllamaClient.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = mock_result
        resp = await client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "mode": "general",
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["message"]["content"] == "Hello!"
    assert data["mode"] == "general"


@pytest.mark.asyncio
async def test_chat_financial_mode(client):
    mock_result = {"role": "assistant", "content": "Let me analyze...", "model": "llama3"}

    with patch("app.services.ollama.OllamaClient.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = mock_result
        resp = await client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "What are my expenses?"}],
                "mode": "financial",
            },
        )

    assert resp.status_code == 200
    assert resp.json()["mode"] == "financial"


@pytest.mark.asyncio
async def test_chat_unknown_mode(client):
    resp = await client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "Hi"}],
            "mode": "unknown_mode",
        },
    )
    assert resp.status_code == 400
