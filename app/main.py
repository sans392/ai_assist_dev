import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.services.log_collector import log_collector

log_format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
logging.basicConfig(
    level=settings.log_level.upper(),
    format=log_format,
    datefmt="%H:%M:%S",
)
# Attach in-memory collector to root logger
log_collector.setFormatter(logging.Formatter(log_format, datefmt="%H:%M:%S"))
log_collector.setLevel(logging.DEBUG)
logging.getLogger().addHandler(log_collector)

from app.routers import admin, chat, health
from app.services.conversation import ConversationStore
from app.services.ollama import OllamaClient

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create shared resources
    app.state.ollama_client = OllamaClient(settings)
    app.state.conversation_store = ConversationStore()
    yield
    # Shutdown: cleanup
    await app.state.ollama_client.close()


app = FastAPI(
    title="AI Assistant",
    description="Backend for AI assistant — learning project",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(health.router)
app.include_router(admin.router)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")
