from pathlib import Path

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from app.dependencies import get_ollama_client
from app.services.agent_settings import agent_settings_store
from app.services.log_collector import log_collector
from app.services.ollama import OllamaClient

router = APIRouter(prefix="/admin", tags=["admin"])
STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


@router.get("", include_in_schema=False)
async def admin_page():
    return FileResponse(STATIC_DIR / "admin.html")


@router.get("/logs")
async def get_logs(
    ollama_only: bool = Query(False, description="Show only Ollama request/response logs"),
    limit: int = Query(200, ge=1, le=1000),
):
    return {"logs": log_collector.get_entries(ollama_only=ollama_only, limit=limit)}


@router.get("/models")
async def get_models(ollama: OllamaClient = Depends(get_ollama_client)):
    models = await ollama.list_models()
    return {"models": models}


@router.get("/settings")
async def get_settings():
    return {"settings": agent_settings_store.get_all()}


@router.patch("/settings/{agent_name}")
async def update_settings(agent_name: str, patch: dict):
    updated = agent_settings_store.update(agent_name, patch)
    return {"agent": agent_name, "settings": updated}
