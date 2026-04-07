import json
from collections.abc import AsyncGenerator

import httpx

from app.config import Settings


class OllamaClient:
    """Async client for the Ollama REST API."""

    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.ollama_base_url
        self.default_model = settings.ollama_model
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
    ) -> dict:
        """Send a chat request and return the full response."""
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": False,
        }
        resp = await self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return {
            "role": data["message"]["role"],
            "content": data["message"]["content"],
            "model": data["model"],
        }

    async def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat response, yielding content chunks as they arrive."""
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": True,
        }
        async with self._client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
                if data.get("done"):
                    break

    async def is_healthy(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            resp = await self._client.get("/api/tags")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
