from fastapi import APIRouter, Depends
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from models import User, OutreachLog, Lead, Campaign
from database import get_session
from auth_utils import get_current_user

router = APIRouter(prefix="/api", tags=["activity"])

@router.get("/activity")
async def get_activity(session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    # Get user's outreach logs with lead information
    result = await session.execute(
        select(OutreachLog, Lead)
        .join(Lead)
        .join(Campaign)
        .where(Campaign.user_id == user.id)
        .order_by(OutreachLog.timestamp.desc())
        .limit(50)
    )
    
    activities = []
    for log, lead in result.all():
        activities.append({
            "id": str(log.id),
            "action": f"Message {log.status}",
            "lead": f"{lead.first_name} {lead.last_name}",
            "timestamp": log.timestamp.isoformat(),
            "status": log.status,
            "message": log.message
        })
    
    return activities

@router.get("/usage")
async def get_usage(session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    # Get user's actual usage statistics
    leads_result = await session.exec(
        select(Lead)
        .join(Campaign)
        .where(Campaign.user_id == user.id)
    )
    leads = leads_result.all()
    
    messages_sent = len([lead for lead in leads if lead.status in ["contacted", "sent", "replied", "connected"]])
    
    # Subscription limits
    limits = {
        "free": {"leads": 10, "messages": 50},
        "pro": {"leads": 100, "messages": 500},
        "enterprise": {"leads": 1000, "messages": 5000}
    }
    
    user_limits = limits.get(user.subscription_tier, limits["free"])
    
    return {
        "subscription_tier": user.subscription_tier,
        "leads_used": len(leads),
        "leads_limit": user_limits["leads"],
        "messages_sent": messages_sent,
        "messages_limit": user_limits["messages"],
        "remaining": user_limits["leads"] - len(leads)
    }
