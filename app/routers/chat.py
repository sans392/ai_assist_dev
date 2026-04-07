import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.agents.base import BaseAgent
from app.dependencies import get_ollama_client
from app.models.schemas import ChatRequest, ChatResponse, Message
from app.services.ollama import OllamaClient

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    ollama: OllamaClient = Depends(get_ollama_client),
):
    from app.dependencies import get_agent

    try:
        agent: BaseAgent = get_agent(request.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    raw_messages = [m.model_dump() for m in request.messages]
    prepared = agent.prepare_messages(raw_messages)

    if request.stream:
        return _stream_response(ollama, prepared, agent.name)

    result = await ollama.chat(prepared)

    return ChatResponse(
        message=Message(role=result["role"], content=result["content"]),
        model=result["model"],
        mode=agent.name,
    )


def _stream_response(
    ollama: OllamaClient,
    prepared: list[dict],
    mode: str,
) -> StreamingResponse:
    async def event_generator():
        async for chunk in ollama.chat_stream(prepared):
            event = json.dumps({"content": chunk, "done": False})
            yield f"data: {event}\n\n"

        final = json.dumps({"content": "", "done": True, "mode": mode})
        yield f"data: {final}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
