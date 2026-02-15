import json
from typing import Optional

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.database import get_db
from backend.models import Conversation, Message, User
from backend.auth import get_current_user

router = APIRouter(prefix="/api")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    conversation_id: int


class GuestChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


@router.post("/chat")
async def chat(req: ChatRequest, request: Request, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    engine = request.app.state.engine

    # Load conversation with messages (scoped to user)
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == req.conversation_id, Conversation.user_id == user.id)
        .options(selectinload(Conversation.messages))
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Build history from DB messages
    history = [
        {"role": m.role, "content": m.content}
        for m in conv.messages
        if m.role in ("user", "assistant")
    ]

    # Save user message to DB
    user_msg = Message(conversation_id=conv.id, role="user", content=req.message)
    db.add(user_msg)
    await db.commit()

    async def event_stream():
        full_text = ""
        async for item in engine.generate_stream(req.message, history):
            if isinstance(item, dict) and item.get("context_cleared"):
                yield f"event: context_cleared\ndata: {json.dumps({'message': 'Context window exceeded. History cleared — starting fresh.'})}\n\n"
            else:
                full_text += item
                yield f"event: token\ndata: {json.dumps({'token': item})}\n\n"

        # Save assistant message to DB
        assistant_msg = Message(conversation_id=conv.id, role="assistant", content=full_text.strip())
        db.add(assistant_msg)
        await db.commit()

        yield f"event: done\ndata: {json.dumps({'full_text': full_text.strip()})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/chat/guest")
async def guest_chat(req: GuestChatRequest, request: Request):
    """Stateless chat for guest users. No auth, no persistence."""
    engine = request.app.state.engine
    history = [{"role": m.role, "content": m.content} for m in req.history]

    async def event_stream():
        full_text = ""
        async for item in engine.generate_stream(req.message, history):
            if isinstance(item, dict) and item.get("context_cleared"):
                yield f"event: context_cleared\ndata: {json.dumps({'message': 'Context window exceeded. History cleared — starting fresh.'})}\n\n"
            else:
                full_text += item
                yield f"event: token\ndata: {json.dumps({'token': item})}\n\n"
        yield f"event: done\ndata: {json.dumps({'full_text': full_text.strip()})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
