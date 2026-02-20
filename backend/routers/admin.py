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
