from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_chat_endpoint(client):
    mock_result = {"role": "assistant", "content": "Hello!", "model": "llama3"}

    with patch("app.services.ollama.OllamaClient.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = mock_result
        resp = await client.post(
            "/chat",
            json={"message": "Hi", "mode": "general"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["message"]["content"] == "Hello!"
    assert data["mode"] == "general"
    assert "conversation_id" in data


@pytest.mark.asyncio
async def test_chat_financial_mode(client):
    mock_result = {"role": "assistant", "content": "Let me analyze...", "model": "llama3"}

    with patch("app.services.ollama.OllamaClient.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = mock_result
        resp = await client.post(
            "/chat",
            json={"message": "What are my expenses?", "mode": "financial"},
        )

    assert resp.status_code == 200
    assert resp.json()["mode"] == "financial"


@pytest.mark.asyncio
async def test_chat_unknown_mode(client):
    resp = await client.post(
        "/chat",
        json={"message": "Hi", "mode": "unknown_mode"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_chat_conversation_continues(client):
    """Server should keep history — second message sees the first."""
    mock_result = {"role": "assistant", "content": "Hi there!", "model": "llama3"}

    with patch("app.services.ollama.OllamaClient.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = mock_result

        # First message — no conversation_id
        resp1 = await client.post("/chat", json={"message": "Hello"})
        conv_id = resp1.json()["conversation_id"]

        # Second message — same conversation
        mock_chat.return_value = {"role": "assistant", "content": "I'm an AI", "model": "llama3"}
        resp2 = await client.post(
            "/chat",
            json={"message": "Who are you?", "conversation_id": conv_id},
        )

        # The second call should include history (system + hello + hi there + who are you)
        call_args = mock_chat.call_args[0][0]  # messages list
        assert len(call_args) == 4  # system + user + assistant + user
        assert call_args[1]["content"] == "Hello"
        assert call_args[3]["content"] == "Who are you?"

    assert resp2.json()["conversation_id"] == conv_id
