from fastapi import APIRouter, HTTPException, Request, Depends, Header
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import desc
from database import get_session
from models import LeadReply, Lead, Notification
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid
import hmac
import hashlib
import json
import logging

router = APIRouter(prefix="/api", tags=["inbound-email"])
logger = logging.getLogger(__name__)


class MailerSendFrom(BaseModel):
    """Sender information from MailerSend webhook"""
    email: str
    name: Optional[str] = None
    raw: Optional[str] = None


class MailerSendRecipient(BaseModel):
    """Recipient information"""
    email: str
    name: Optional[str] = None


class MailerSendRecipients(BaseModel):
    """Recipients object"""
    to: Optional[Dict[str, Any]] = None
    cc: Optional[Dict[str, Any]] = None
    bcc: Optional[Dict[str, Any]] = None


class MailerSendInboundData(BaseModel):
    """Nested data object from MailerSend inbound webhook"""
    object: str = "message"
    id: Optional[str] = None
    from_: Optional[MailerSendFrom] = None
    recipients: Optional[MailerSendRecipients] = None
    subject: str
    text: Optional[str] = None
    html: Optional[str] = None
    date: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    attachments: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        fields = {'from_': 'from'}


class MailerSendInboundPayload(BaseModel):
    """Complete MailerSend inbound webhook payload structure"""
    type: str  # Should be "inbound.message"
    inbound_id: Optional[str] = None
    url: Optional[str] = None
    created_at: Optional[str] = None
    data: MailerSendInboundData


def verify_mailersend_signature(body: bytes, signature: str, secret: str) -> bool:
    """
    Verify MailerSend webhook signature using HMAC-SHA256
    
    Args:
        body: Raw request body as bytes
        signature: Signature from the Signature header
        secret: Webhook secret from MailerSend
        
    Returns:
        True if signature is valid, False otherwise
    """
    if not secret or not signature:
        return False
    
    try:
        computed = hmac.new(
            secret.encode('utf-8'),
            body,
            hashlib.sha256
        ).hexdigest()
        
        # Constant-time comparison to prevent timing attacks
        return hmac.compare_digest(computed, signature)
    except Exception as e:
        logger.error(f"Signature verification error: {e}")
        return False


@router.post("/mailersend/inbound")
async def receive_mailersend_inbound(
    request: Request,
    session: AsyncSession = Depends(get_session),
    signature: Optional[str] = Header(None)
):
    """
    Receive inbound emails from MailerSend webhook.
    
    This endpoint handles MailerSend's inbound routing webhooks:
    1. Verifies webhook signature (if secret is configured)
    2. Extracts email data from MailerSend's nested payload structure
    3. Finds the corresponding lead by sender email address
    4. Creates a LeadReply record
    5. Updates the lead's status to "replied"
    6. Creates a notification for the user
    
    MailerSend payload structure:
    {
      "type": "inbound.message",
      "data": {
        "from": {"email": "...", "name": "..."},
        "subject": "...",
        "text": "...",
        "html": "..."
      }
    }
    """
    try:
        # Get raw body for signature verification
        body = await request.body()
        
        # Optional: Verify webhook signature for security
        # You can set MAILERSEND_WEBHOOK_SECRET in environment variables
        import os
        webhook_secret = os.getenv("MAILERSEND_WEBHOOK_SECRET")
        if webhook_secret and signature:
            if not verify_mailersend_signature(body, signature, webhook_secret):
                logger.warning("Invalid MailerSend webhook signature")
                return {
                    "status": "error",
                    "message": "Invalid signature"
                }
        
        # Parse JSON payload
        payload_dict = json.loads(body.decode('utf-8'))
        payload = MailerSendInboundPayload(**payload_dict)
        
        # Validate it's an inbound message
        if payload.type != "inbound.message":
            logger.info(f"Received non-inbound webhook type: {payload.type}")
            return {
                "status": "success",
                "message": f"Webhook type {payload.type} acknowledged"
            }
        
        # Extract sender email
        if not payload.data.from_:
            return {
                "status": "error",
                "message": "Missing sender information in webhook payload"
            }
        sender_email = payload.data.from_.email
        sender_name = payload.data.from_.name or ""
        
        # Extract body (prefer text over html)
        body_text = payload.data.text or payload.data.html or ""
        
        # Parse received_at timestamp
        received_at = datetime.utcnow()
        if payload.data.date:
            try:
                from email.utils import parsedate_to_datetime
                received_at = parsedate_to_datetime(payload.data.date)
            except Exception as e:
                logger.warning(f"Failed to parse date: {e}")
        
        # Find the lead by email address
        statement = select(Lead).where(Lead.email == sender_email)
        result = await session.execute(statement)
        lead = result.scalar_one_or_none()
        
        if not lead:
            logger.info(f"No matching lead found for email: {sender_email}")
            # Return 200 so MailerSend doesn't retry
            return {
                "status": "success",
                "message": "Email received but no matching lead found",
                "sender": sender_email
            }
        
        # Create LeadReply record
        lead_reply = LeadReply(
            id=uuid.uuid4(),
            lead_id=lead.id,
            sender=sender_email,
            subject=payload.data.subject,
            body=body_text,
            received_at=received_at
        )
        session.add(lead_reply)
        
        # Update lead status to "replied"
        old_status = lead.status
        lead.status = "replied"
        session.add(lead)
        
        # Create notification for the user (if lead is part of a campaign or batch)
        try:
            # Get user_id from campaign or batch
            user_id = None
            if lead.campaign_id:
                from models import Campaign
                campaign_result = await session.execute(
                    select(Campaign).where(Campaign.id == lead.campaign_id)
                )
                campaign = campaign_result.scalar_one_or_none()
                if campaign:
                    user_id = campaign.user_id
            elif lead.batch_id:
                from models import LeadBatch
                batch_result = await session.execute(
                    select(LeadBatch).where(LeadBatch.id == lead.batch_id)
                )
                batch = batch_result.scalar_one_or_none()
                if batch:
                    user_id = batch.user_id
            
            if user_id:
                notification = Notification(
                    id=uuid.uuid4(),
                    user_id=user_id,
                    type="lead_replied",
                    message=f"New reply from {sender_name or sender_email}: {payload.data.subject}",
                    created_at=datetime.utcnow(),
                    priority="high",
                    notification_metadata=json.dumps({
                        "lead_id": str(lead.id),
                        "sender": sender_email,
                        "subject": payload.data.subject
                    })
                )
                session.add(notification)
        except Exception as e:
            logger.error(f"Failed to create notification: {e}")
        
        # Commit all changes
        await session.commit()
        
        logger.info(f"Successfully processed inbound email from {sender_email} for lead {lead.id}")
        
        return {
            "status": "success",
            "message": "Email processed successfully",
            "lead_id": str(lead.id),
            "sender": sender_email,
            "status_change": f"{old_status} -> replied"
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in webhook payload: {e}")
        return {
            "status": "error",
            "message": "Invalid JSON payload"
        }
    except Exception as e:
        logger.error(f"Error processing inbound email: {e}", exc_info=True)
        await session.rollback()
        # Return 200 so MailerSend doesn't retry endlessly
        return {
            "status": "error",
            "message": f"Error processing email: {str(e)}"
        }



@router.post("/mailersend/events")
async def receive_mailersend_events(
    request: Request,
    session: AsyncSession = Depends(get_session),
    signature: Optional[str] = Header(None)
):
    """
    Handle MailerSend event webhooks (delivered, opened, clicked, bounced, unsubscribed, complaint)

    Expected payload example (MailerSend):
    {
      "type": "event",
      "data": {
         "event": "delivered",
         "message": {"id": "<message-id>", ...},
         "recipient": {"email": "to@example.com"},
         "tags": ["campaign:Name","campaign_id:<id>","lead_id:<id>"]
      }
    }
    """
    try:
        body = await request.body()
        import os
        webhook_secret = os.getenv("MAILERSEND_WEBHOOK_SECRET")
        if webhook_secret and signature:
            if not verify_mailersend_signature(body, signature, webhook_secret):
                logger.warning("Invalid MailerSend event webhook signature")
                return {"status": "error", "message": "Invalid signature"}

        payload_dict = json.loads(body.decode("utf-8"))
        # Minimal validation
        event_type = payload_dict.get("type")
        data = payload_dict.get("data", {})
        if not data:
            return {"status": "error", "message": "No data in payload"}

        # Extract event name and message id
        event_name = data.get("event") or data.get("name") or data.get("event_name")
        message = data.get("message") or {}
        message_id = message.get("id") or data.get("message_id") or None
        recipient = data.get("recipient") or {}
        recipient_email = recipient.get("email") if isinstance(recipient, dict) else None

        # Try to find log by external_message_id first
        log = None
        if message_id:
            result = await session.execute(select(LeadReply).where(LeadReply.id == None))  # dummy to ensure imports are ready
            from models import EmailLog
            result = await session.execute(select(EmailLog).where(EmailLog.external_message_id == message_id))
            log = result.scalar_one_or_none()

        # Fallback: try to find by tags if present
        if not log:
            tags = data.get("tags") or []
            campaign_id = None
            lead_id = None
            for t in tags:
                if isinstance(t, str) and t.startswith("campaign_id:"):
                    try:
                        campaign_id = uuid.UUID(t.split(":", 1)[1])
                    except Exception:
                        pass
                if isinstance(t, str) and t.startswith("lead_id:"):
                    try:
                        lead_id = uuid.UUID(t.split(":", 1)[1])
                    except Exception:
                        pass
            if lead_id:
                from models import EmailLog
                result = await session.execute(select(EmailLog).where(EmailLog.lead_id == lead_id).order_by(EmailLog.sent_at.desc()))
                log = result.scalars().first()

        # If still not found, try match by recipient email and recent sent logs
        if not log and recipient_email:
            from models import EmailLog
            result = await session.execute(select(EmailLog).where(EmailLog.to_email == recipient_email).order_by(EmailLog.sent_at.desc()))
            log = result.scalars().first()

        if not log:
            logger.info(f"MailerSend event received but no matching EmailLog found (message_id={message_id}, recipient={recipient_email})")
            return {"status": "success", "message": "No matching log found"}

        # Update log based on event
        updated = False
        evt = (event_name or "").lower()
        now = datetime.utcnow()
        if evt in ("delivered", "delivery"):
            log.status = "delivered"
            if not log.sent_at:
                log.sent_at = now
            updated = True
        elif evt in ("opened", "open"):
            log.status = "opened"
            log.opened_at = now
            updated = True
        elif evt in ("clicked", "click"):
            log.status = "clicked"
            log.clicked_at = now
            updated = True
        elif evt in ("bounced", "bounce"):
            log.status = "bounced"
            log.error = (data.get("reason") or data.get("error") or "bounced")
            updated = True
        elif evt in ("complained", "complaint"):
            log.status = "complained"
            updated = True
        elif evt in ("unsubscribed", "unsubscribe"):
            log.status = "unsubscribed"
            updated = True

        if updated:
            session.add(log)
            await session.commit()
            logger.info(f"Updated EmailLog {getattr(log, 'id', None)} status to {log.status} from MailerSend event {evt}")

        return {"status": "success", "message": "Event processed"}

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in MailerSend event payload: {e}")
        return {"status": "error", "message": "Invalid JSON"}
    except Exception as e:
        logger.error(f"Error processing MailerSend event: {e}", exc_info=True)
        try:
            await session.rollback()
        except Exception:
            pass
        return {"status": "error", "message": f"Error: {str(e)}"}


@router.get("/leads/{lead_id}/replies")
async def get_lead_replies(
    lead_id: str,
    session: AsyncSession = Depends(get_session)
):
    """
    Get all replies for a specific lead.
    
    Returns a list of replies with sender, subject, body, and received_at.
    """
    try:
        # Convert string UUID to UUID object
        lead_uuid = uuid.UUID(lead_id)
        
        # Fetch all replies for this lead
        from sqlmodel import col
        statement = select(LeadReply).where(LeadReply.lead_id == lead_uuid).order_by(col(LeadReply.received_at).desc())
        result = await session.execute(statement)
        replies = result.scalars().all()
        
        # Convert to list of dicts
        replies_data = [
            {
                "id": str(reply.id),
                "sender": reply.sender,
                "subject": reply.subject,
                "body": reply.body,
                "received_at": reply.received_at.isoformat()
            }
            for reply in replies
        ]
        
        return {
            "replies": replies_data,
            "count": len(replies_data)
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid lead ID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching replies: {str(e)}")
