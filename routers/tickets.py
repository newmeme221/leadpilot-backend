from fastapi import APIRouter, Depends, HTTPException, Body
router = APIRouter(prefix="/api/tickets", tags=["tickets"])
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from models import Ticket, TicketComment, User, Notification
from pydantic import BaseModel
from database import get_session
from auth_utils import get_current_user
from typing import List 
import uuid
from datetime import datetime


# Pydantic schema for ticket creation
class TicketCreate(BaseModel):
    subject: str
    description: str
    status: str = "open"
    assigned_to: int | None = None

@router.post("/", response_model=Ticket)
async def create_ticket(ticket: TicketCreate, session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    # Validate subject is present (Pydantic enforces this)
    new_ticket = Ticket(
        subject=ticket.subject,
        description=ticket.description,
        status=ticket.status,
        assigned_to=ticket.assigned_to,
        user_id=user.id,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    session.add(new_ticket)
    await session.commit()
    await session.refresh(new_ticket)
    return new_ticket

@router.get("/", response_model=List[Ticket])
async def list_tickets(session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    from sqlalchemy.orm import selectinload
    if user.role == "admin":
        result = await session.execute(
            select(Ticket).options(selectinload(Ticket.user))
        )
        return result.scalars().all()
    else:
        result = await session.execute(
            select(Ticket).options(selectinload(Ticket.user)).where(Ticket.user_id == user.id)
        )
        return result.scalars().all()

@router.get("/{ticket_id}", response_model=Ticket)
async def get_ticket(ticket_id: uuid.UUID, session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    from sqlalchemy.orm import selectinload
    result = await session.execute(
        select(Ticket).options(selectinload(Ticket.user)).where(Ticket.id == ticket_id)
    )
    ticket = result.scalars().first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    if user.role == "admin" or ticket.user_id == user.id:
        return ticket
    raise HTTPException(status_code=403, detail="Not authorized")

@router.put("/{ticket_id}", response_model=Ticket)
async def update_ticket(ticket_id: uuid.UUID, update: dict = Body(...), session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    ticket = await session.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    if user.role == "admin" or ticket.user_id == user.id:
        old_status = ticket.status
        for k, v in update.items():
            setattr(ticket, k, v)
        ticket.updated_at = datetime.utcnow()
        session.add(ticket)
        await session.commit()
        await session.refresh(ticket)
        # Notify user if status changed
        if "status" in update and update["status"] != old_status:
            notif = Notification(
                user_id=ticket.user_id,
                type="ticket_status",
                message=f"Your ticket '{ticket.subject}' status changed to '{ticket.status}'.",
                priority="info"
            )
            session.add(notif)
            await session.commit()
        return ticket
    raise HTTPException(status_code=403, detail="Not authorized")

@router.post("/{ticket_id}/comment", response_model=TicketComment)
async def add_comment(ticket_id: uuid.UUID, comment: TicketComment, session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    ticket = await session.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    comment.ticket_id = ticket_id
    comment.user_id = user.id
    comment.created_at = datetime.utcnow()
    session.add(comment)
    await session.commit()
    await session.refresh(comment)
    return comment

@router.put("/{ticket_id}/assign", response_model=Ticket)
async def assign_ticket(ticket_id: uuid.UUID, assignee_id: int = Body(...), session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    ticket = await session.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    ticket.assigned_to = assignee_id
    ticket.status = "assigned"
    ticket.updated_at = datetime.utcnow()
    session.add(ticket)
    await session.commit()
    await session.refresh(ticket)
    # Notify user of assignment
    notif = Notification(
        user_id=ticket.user_id,
        type="ticket_assigned",
        message=f"Your ticket '{ticket.subject}' has been assigned to an admin.",
        priority="info"
    )
    session.add(notif)
    await session.commit()
    return ticket

@router.put("/{ticket_id}/close", response_model=Ticket)
async def close_ticket(ticket_id: uuid.UUID, session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    ticket = await session.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    if user.role == "admin" or ticket.user_id == user.id:
        ticket.status = "closed"
        ticket.updated_at = datetime.utcnow()
        session.add(ticket)
        await session.commit()
        await session.refresh(ticket)
        # Notify user of closure
        notif = Notification(
            user_id=ticket.user_id,
            type="ticket_closed",
            message=f"Your ticket '{ticket.subject}' has been closed.",
            priority="info"
        )
        session.add(notif)
        await session.commit()
        return ticket
    raise HTTPException(status_code=403, detail="Not authorized")
