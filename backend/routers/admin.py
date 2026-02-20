from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import ADMIN_API_KEY
from backend.database import get_db
from backend.models import User, Conversation, Message

router = APIRouter(prefix="/api/admin", tags=["admin"])


async def verify_admin_key(x_admin_key: str = Header(None)):
    if not ADMIN_API_KEY or x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.get("/stats", dependencies=[Depends(verify_admin_key)])
async def admin_stats(db: AsyncSession = Depends(get_db)):
    total_users = (await db.execute(select(func.count(User.id)))).scalar()
    total_conversations = (await db.execute(select(func.count(Conversation.id)))).scalar()
    total_messages = (await db.execute(select(func.count(Message.id)))).scalar()
    total_generations = (await db.execute(
        select(func.count(Message.id)).where(Message.role == "assistant")
    )).scalar()

    # Top 10 users by generation count
    top_users_q = (
        select(User.username, func.count(Message.id).label("generations"))
        .join(Conversation, Conversation.user_id == User.id)
        .join(Message, Message.conversation_id == Conversation.id)
        .where(Message.role == "assistant")
        .group_by(User.id)
        .order_by(func.count(Message.id).desc())
        .limit(10)
    )
    top_users = [
        {"username": row.username, "generations": row.generations}
        for row in (await db.execute(top_users_q)).all()
    ]

    # Last 10 registered users
    recent_users_q = (
        select(User.username, User.created_at)
        .order_by(User.created_at.desc())
        .limit(10)
    )
    recent_users = [
        {"username": row.username, "created_at": row.created_at.isoformat()}
        for row in (await db.execute(recent_users_q)).all()
    ]

    return {
        "total_users": total_users,
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "total_generations": total_generations,
        "top_users": top_users,
        "recent_users": recent_users,
    }


@router.get("/users", dependencies=[Depends(verify_admin_key)])
async def admin_users(db: AsyncSession = Depends(get_db)):
    """All users with per-user stats: conversations count, messages count, last active."""
    users_q = (
        select(
            User.id,
            User.username,
            User.created_at,
            func.count(func.distinct(Conversation.id)).label("conversations_count"),
            func.count(Message.id).label("messages_count"),
            func.max(Message.created_at).label("last_active"),
        )
        .outerjoin(Conversation, Conversation.user_id == User.id)
        .outerjoin(Message, Message.conversation_id == Conversation.id)
        .group_by(User.id)
        .order_by(User.created_at)
    )
    rows = (await db.execute(users_q)).all()
    return [
        {
            "id": r.id,
            "username": r.username,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "conversations_count": r.conversations_count,
            "messages_count": r.messages_count,
            "last_active": r.last_active.isoformat() if r.last_active else None,
        }
        for r in rows
    ]


@router.get("/users/{user_id}/conversations", dependencies=[Depends(verify_admin_key)])
async def admin_user_conversations(user_id: int, db: AsyncSession = Depends(get_db)):
    """Conversations for a specific user."""
    convos_q = (
        select(
            Conversation.id,
            Conversation.title,
            func.count(Message.id).label("message_count"),
            Conversation.created_at,
            Conversation.updated_at,
        )
        .where(Conversation.user_id == user_id)
        .outerjoin(Message, Message.conversation_id == Conversation.id)
        .group_by(Conversation.id)
        .order_by(Conversation.updated_at.desc())
    )
    rows = (await db.execute(convos_q)).all()
    return [
        {
            "id": r.id,
            "title": r.title,
            "message_count": r.message_count,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
        for r in rows
    ]


@router.get("/conversations/{conversation_id}/messages", dependencies=[Depends(verify_admin_key)])
async def admin_conversation_messages(conversation_id: int, db: AsyncSession = Depends(get_db)):
    """Messages for a specific conversation."""
    msgs_q = (
        select(Message.id, Message.role, Message.content, Message.created_at)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    rows = (await db.execute(msgs_q)).all()
    return [
        {
            "id": r.id,
            "role": r.role,
            "content": r.content,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@router.get("/conversations", dependencies=[Depends(verify_admin_key)])
async def admin_conversations(
    limit: int = 50, db: AsyncSession = Depends(get_db)
):
    """Recent conversations with username, title, message count, timestamps."""
    convos_q = (
        select(
            Conversation.id,
            User.username,
            Conversation.title,
            func.count(Message.id).label("message_count"),
            Conversation.created_at,
            Conversation.updated_at,
        )
        .join(User, User.id == Conversation.user_id)
        .outerjoin(Message, Message.conversation_id == Conversation.id)
        .group_by(Conversation.id, User.username)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
    )
    rows = (await db.execute(convos_q)).all()
    return [
        {
            "id": r.id,
            "username": r.username,
            "title": r.title,
            "message_count": r.message_count,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
        for r in rows
    ]
