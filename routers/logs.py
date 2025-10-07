from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from models import OutreachLog, Lead, User, Campaign
from database import get_session
from auth_utils import get_current_user
from schemas import OutreachLogCreate
from datetime import datetime

router = APIRouter(prefix="/api/logs", tags=["logs"])

@router.post("/outreach")
async def log_outreach(log: OutreachLogCreate, session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    # Verify the lead belongs to the user
    result = await session.execute(
        select(Lead)
        .join(Campaign)
        .where(Lead.id == log.lead_id, Campaign.user_id == user.id)
    )
    lead = result.scalar_one_or_none()
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    outreach_log = OutreachLog(
        lead_id=log.lead_id,
        status=log.status,
        message=log.message,
        timestamp=datetime.utcnow()
    )
    session.add(outreach_log)
    await session.commit()
    return {"message": "Outreach logged successfully"}
