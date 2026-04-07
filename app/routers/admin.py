from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from app.services.log_collector import log_collector

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
