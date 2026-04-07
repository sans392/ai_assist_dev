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

    conversation_id = request.conversation_id or str(uuid.uuid4())
    user_msg = {"role": "user", "content": request.message}

    # Save user message to server-side history
    store.append(conversation_id, user_msg)

    # Build full context: history (including the new message) -> agent prepares
    history = store.get(conversation_id)
    prepared = agent.prepare_messages(history)

    if request.stream:
        return _stream_response(ollama, prepared, agent.name, conversation_id, store)

    result = await ollama.chat(prepared)

    # Save assistant reply
    assistant_msg = {"role": result["role"], "content": result["content"]}
    store.append(conversation_id, assistant_msg)

    return ChatResponse(
        message=Message(role=result["role"], content=result["content"]),
        model=result["model"],
        mode=agent.name,
        conversation_id=conversation_id,
    )


def _stream_response(
    ollama: OllamaClient,
    prepared: list[dict],
    mode: str,
    conversation_id: str,
    store: ConversationStore,
) -> StreamingResponse:
    async def event_generator():
        full_content = ""
        async for chunk in ollama.chat_stream(prepared):
            full_content += chunk
            event = json.dumps({"content": chunk, "done": False})
            yield f"data: {event}\n\n"

        # Save assistant reply to server-side history
        store.append(conversation_id, {"role": "assistant", "content": full_content})

        final = json.dumps({
            "content": "", "done": True, "mode": mode, "conversation_id": conversation_id,
        })
        yield f"data: {final}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
