import json
import logging
from collections.abc import AsyncGenerator

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


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
        options: dict | None = None,
    ) -> dict:
        """Send a chat request and return the full response."""
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": False,
        }
        if options:
            payload["options"] = options
        logger.info("POST /api/chat model=%s messages=%d", payload["model"], len(messages))
        logger.debug("Request payload: %s", json.dumps(payload, ensure_ascii=False, indent=2))
        resp = await self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        logger.info("Response: model=%s content_length=%d", data["model"], len(data["message"]["content"]))
        return {
            "role": data["message"]["role"],
            "content": data["message"]["content"],
            "model": data["model"],
        }

    async def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        options: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat response, yielding content chunks as they arrive."""
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": True,
        }
        if options:
            payload["options"] = options
        logger.info("POST /api/chat [stream] model=%s messages=%d", payload["model"], len(messages))
        logger.debug("Request payload: %s", json.dumps(payload, ensure_ascii=False, indent=2))
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

    async def list_models(self) -> list[str]:
        """Return list of available model names."""
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except httpx.HTTPError:
            return []

    async def is_healthy(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            resp = await self._client.get("/api/tags")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
