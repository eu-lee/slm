from datetime import datetime
from pydantic import BaseModel


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationOut(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ConversationDetail(ConversationOut):
    messages: list[MessageOut] = []


class ConversationCreate(BaseModel):
    title: str = "New Conversation"


class ConversationRename(BaseModel):
    title: str
