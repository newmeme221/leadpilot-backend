from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from models import User 
import uuid
from auth_utils import get_session, get_current_user
from typing import List

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/", response_model=List[User])
async def list_users(session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    # Only allow admins to list all users
    if user.role != "admin":
        return [user]
    result = await session.execute(select(User))
    users = result.scalars().all()
    return users

@router.post("/{user_id}/update-status")
async def admin_update_user_status(user_id: uuid.UUID, new_status: str, session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    target_user = await session.get(User, user_id)
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
    target_user.status = new_status
    session.add(target_user)
    await session.commit()
    return {"message": f"User {user_id} status updated to {new_status}"}

@router.post("/{user_id}/update-last-login")
async def admin_update_last_login(user_id: uuid.UUID, session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    target_user = await session.get(User, user_id)
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
    from datetime import datetime
    target_user.last_login = datetime.utcnow()
    session.add(target_user)
    await session.commit()
    return {"message": f"User {user_id} last login updated"}


@router.put("/me/api-keys")
async def update_api_keys(
    request: dict,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """
    Update user's API keys (Apollo.io, MailerSend, OpenAI)
    """
    # Update Apollo.io API key if provided (accept correct spelling, map to DB field)
    if "apollo_api_key" in request:
        user.appollo_api_key = request["apollo_api_key"] or None
    
    # Update MailerSend API key if provided
    if "mailersend_api_key" in request:
        user.mailersend_api_key = request["mailersend_api_key"] or None
    
    # Update OpenAI API key if provided
    if "openai_api_key" in request:
        user.openai_api_key = request["openai_api_key"] or None
    
    # Update sender details if provided
    if "sender_email" in request:
        user.sender_email = request["sender_email"] or None
    if "sender_name" in request:
        user.sender_name = request["sender_name"] or None
    if "reply_to_email" in request:
        user.reply_to_email = request["reply_to_email"] or None
    
    session.add(user)
    await session.commit()
    await session.refresh(user)
    
    return {
        "message": "API keys updated successfully",
        "has_apollo_key": bool(user.appollo_api_key),
        "has_mailersend_key": bool(user.mailersend_api_key),
        "has_openai_key": bool(user.openai_api_key)
    }


@router.get("/me")
async def get_current_user_info(user: User = Depends(get_current_user)):
    """
    Get current user's information (without sensitive data like password)
    """
    return {
        "id": str(user.id),
        "email": user.email,
        "role": user.role,
        "subscription_tier": user.subscription_tier,
        "subscription_status": user.subscription_status,
        "has_apollo_key": bool(user.appollo_api_key),
        "has_mailersend_key": bool(user.mailersend_api_key),
        "has_openai_key": bool(user.openai_api_key),
        "sender_email": user.sender_email,
        "sender_name": user.sender_name,
        "reply_to_email": user.reply_to_email
    }
