
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select
from models import Notification, User
from database import get_session
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional 
import uuid
from datetime import datetime
import json
from auth_utils import get_current_user

router = APIRouter(prefix="/api/notifications", tags=["notifications"])


# Enhanced: filtering, pagination
@router.get("/", response_model=List[Notification])
async def get_notifications(
    read: Optional[bool] = None,
    type: Optional[str] = None,
    priority: Optional[str] = None,
    skip: int = 0,
    limit: int = Query(20, le=100),
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user)
):
    query = select(Notification).where(Notification.user_id == user.id)
    if read is not None:
        query = query.where(Notification.read == read)
    if type:
        query = query.where(Notification.type == type)
    if priority:
        query = query.where(Notification.priority == priority)
    query = query.order_by(Notification.created_at.desc()).offset(skip).limit(limit)
    result = await session.execute(query)
    return result.scalars().all()


# Enhanced: support new fields (priority, expires_at, notification_metadata, delivery_status)
@router.post("/", response_model=Notification)
def create_notification(
    notification: Notification,
    session: Session = Depends(get_session)
):
    # Optionally validate notification_metadata as JSON
    if notification.notification_metadata:
        try:
            json.loads(notification.notification_metadata)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON")
    session.add(notification)
    session.commit()
    session.refresh(notification)
    return notification


# Mark single notification as read
@router.post("/mark-read/{notification_id}")
def mark_notification_read(notification_id: uuid.UUID, session: Session = Depends(get_session)):
    notification = session.get(Notification, notification_id)
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    notification.read = True
    session.add(notification)
    session.commit()
    return {"success": True}

# Batch mark all as read for a user
@router.post("/mark-all-read")
async def mark_all_notifications_read(session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    result = await session.execute(select(Notification).where(Notification.user_id == user.id, Notification.read == False))
    notifications = result.scalars().all()
    for n in notifications:
        n.read = True
        session.add(n)
    await session.commit()
    return {"success": True, "count": len(notifications)}


@router.patch("/{notification_id}", response_model=Notification)
def update_notification(notification_id: uuid.UUID, notification_update: dict, session: Session = Depends(get_session)):
    notification = session.get(Notification, notification_id)
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    for key, value in notification_update.items():
        if hasattr(notification, key):
            setattr(notification, key, value)
    # Optionally validate notification_metadata as JSON
    if getattr(notification, "notification_metadata", None):
        try:
            json.loads(notification.notification_metadata)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON")
    session.add(notification)
    session.commit()
    session.refresh(notification)
    return notification

# Auto-delete expired notifications (utility endpoint)
@router.delete("/expired/cleanup")
def cleanup_expired_notifications(session: Session = Depends(get_session)):
    now = datetime.utcnow()
    expired = session.exec(select(Notification).where(Notification.expires_at != None, Notification.expires_at < now)).all()
    count = 0
    for n in expired:
        session.delete(n)
        count += 1
    session.commit()
    return {"success": True, "deleted": count}

@router.delete("/{notification_id}")
def delete_notification(notification_id: uuid.UUID, session: Session = Depends(get_session)):
    notification = session.get(Notification, notification_id)
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    session.delete(notification)
    session.commit()
    return {"success": True}
