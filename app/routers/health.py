from fastapi import APIRouter, Depends

from app.dependencies import get_ollama_client
from app.services.ollama import OllamaClient

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(ollama: OllamaClient = Depends(get_ollama_client)):
    ollama_ok = await ollama.is_healthy()
    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama": "connected" if ollama_ok else "unavailable",
    }
