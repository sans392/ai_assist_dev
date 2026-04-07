from fastapi import Request

from app.agents import get_agent_by_mode
from app.agents.base import BaseAgent
from app.services.conversation import ConversationStore
from app.services.ollama import OllamaClient


def get_ollama_client(request: Request) -> OllamaClient:
    return request.app.state.ollama_client


def get_conversation_store(request: Request) -> ConversationStore:
    return request.app.state.conversation_store


def get_agent(mode: str) -> BaseAgent:
    return get_agent_by_mode(mode)
