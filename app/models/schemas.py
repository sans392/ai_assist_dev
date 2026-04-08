from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(description="User message text")
    mode: str = Field(default="general", description="Agent mode: general, financial, etc.")
    stream: bool = False
    conversation_id: str | None = Field(
        default=None,
        description="Conversation ID. Omit to start a new conversation.",
    )


class ChatResponse(BaseModel):
    message: Message
    model: str
    mode: str
    conversation_id: str
