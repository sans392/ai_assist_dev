import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import app
from app.services.conversation import ConversationStore
from app.services.ollama import OllamaClient


@pytest.fixture
async def client():
    # Manually set up app state (lifespan doesn't run with ASGITransport)
    app.state.ollama_client = OllamaClient(settings)
    app.state.conversation_store = ConversationStore()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    await app.state.ollama_client.close()
