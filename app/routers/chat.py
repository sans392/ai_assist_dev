import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.agents.base import BaseAgent
from app.dependencies import get_conversation_store, get_ollama_client
from app.models.schemas import ChatRequest, ChatResponse, Message
from app.services.conversation import ConversationStore
from app.services.ollama import OllamaClient

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    ollama: OllamaClient = Depends(get_ollama_client),
    store: ConversationStore = Depends(get_conversation_store),
):
    from app.dependencies import get_agent

    try:
        agent: BaseAgent = get_agent(request.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Build message list from conversation history + new messages
    raw_messages = [m.model_dump() for m in request.messages]
    conversation_id = request.conversation_id or str(uuid.uuid4())

    history = store.get(conversation_id)
    all_messages = history + raw_messages
    prepared = agent.prepare_messages(all_messages)

    if request.stream:
        return _stream_response(ollama, prepared, agent.name, conversation_id, store, raw_messages)

    result = await ollama.chat(prepared)

    # Save user message(s) and assistant reply to conversation store
    for msg in raw_messages:
        store.append(conversation_id, msg)
    store.append(conversation_id, {"role": result["role"], "content": result["content"]})

    return ChatResponse(
        message=Message(role=result["role"], content=result["content"]),
        model=result["model"],
        mode=agent.name,
    )


def _stream_response(
    ollama: OllamaClient,
    prepared: list[dict],
    mode: str,
    conversation_id: str,
    store: ConversationStore,
    raw_messages: list[dict],
) -> StreamingResponse:
    async def event_generator():
        full_content = ""
        async for chunk in ollama.chat_stream(prepared):
            full_content += chunk
            event = json.dumps({"content": chunk, "done": False})
            yield f"data: {event}\n\n"

        # Save to conversation history
        for msg in raw_messages:
            store.append(conversation_id, msg)
        store.append(conversation_id, {"role": "assistant", "content": full_content})

        final = json.dumps({"content": "", "done": True, "mode": mode})
        yield f"data: {final}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
