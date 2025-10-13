from fastapi import APIRouter, Depends, HTTPException, Body, Path, BackgroundTasks
from typing import List 
import random 
import httpx 
import re 
import uuid
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from models import Campaign, Lead, User, EmailCampaign, EmailLog, FollowUpEmail, FollowUpMessage
from database import get_session, async_session
from auth_utils import get_current_user
from schemas import CampaignStats, EmailCampaignCreate, FollowUpEmailCreate, FollowUpEmailOut
from datetime import datetime, timedelta 
import os
import logging
import asyncio
from schemas import CampaignCreate, FollowUpMessageCreate

from sqlalchemy.exc import NoResultFound

router = APIRouter(prefix="/api", tags=["campaigns"])

# MailerSend Configuration
MAILERSEND_API_URL = "https://api.mailersend.com/v1"

# Concurrency limiter for AI calls (minimal protection)
AI_CONCURRENCY_LIMIT = int(os.getenv("AI_CONCURRENCY_LIMIT", "5"))
ai_semaphore = asyncio.Semaphore(AI_CONCURRENCY_LIMIT)

@router.post("/email-campaigns/{campaign_id}/followups", response_model=FollowUpEmailOut)
async def create_followup_email(
    campaign_id: uuid.UUID,
    req: FollowUpEmailCreate = Body(...),
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user)
):
    # Ensure campaign exists and belongs to user
    result = await session.execute(select(EmailCampaign).where(EmailCampaign.id == campaign_id, EmailCampaign.user_id == user.id))
    campaign = result.scalar_one_or_none()
    if not campaign:
        raise HTTPException(status_code=404, detail="Email campaign not found")
    followup = FollowUpEmail(
        email_campaign_id=campaign_id,
        subject=req.subject,
        body=req.body,
        delay_days=req.delay_days,
        status="pending",
        sender_email=getattr(user, 'sender_email', None) or os.getenv('FROM_EMAIL'),
        sender_name=getattr(user, 'sender_name', None) or (user.email.split('@')[0] if user and getattr(user, 'email', None) else 'LeadPilot'),
        reply_to_email=getattr(user, 'reply_to_email', None) or os.getenv('REPLY_TO_EMAIL') or os.getenv('FROM_EMAIL')
    )
    session.add(followup)
    await session.commit()
    await session.refresh(followup)
    return followup

@router.get("/email-campaigns/{campaign_id}/followups", response_model=List[FollowUpEmailOut])
async def list_followup_emails(
    campaign_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user)
):
    # Ensure campaign exists and belongs to user
    result = await session.execute(select(EmailCampaign).where(EmailCampaign.id == campaign_id, EmailCampaign.user_id == user.id))
    campaign = result.scalar_one_or_none()
    if not campaign:
        raise HTTPException(status_code=404, detail="Email campaign not found")
    result = await session.execute(select(FollowUpEmail).where(FollowUpEmail.email_campaign_id == campaign_id))
    followups = result.scalars().all()
    return followups

@router.put("/email-campaigns/{campaign_id}/followups/{followup_id}", response_model=FollowUpEmailOut)
async def update_followup_email(
    campaign_id: uuid.UUID,
    followup_id: uuid.UUID,
    req: FollowUpEmailCreate = Body(...),
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user)
):
    # Ensure campaign exists and belongs to user
    result = await session.execute(select(EmailCampaign).where(EmailCampaign.id == campaign_id, EmailCampaign.user_id == user.id))
    campaign = result.scalar_one_or_none()
    if not campaign:
        raise HTTPException(status_code=404, detail="Email campaign not found")
    result = await session.execute(select(FollowUpEmail).where(FollowUpEmail.id == followup_id, FollowUpEmail.email_campaign_id == campaign_id))
    followup = result.scalar_one_or_none()
    if not followup:
        raise HTTPException(status_code=404, detail="Follow-up email not found")
    followup.subject = req.subject
    followup.body = req.body
    followup.delay_days = req.delay_days
    session.add(followup)
    await session.commit()
    await session.refresh(followup)
    return followup

@router.delete("/email-campaigns/{campaign_id}/followups/{followup_id}")
async def delete_followup_email(
    campaign_id: uuid.UUID,
    followup_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user)
):
    # Ensure campaign exists and belongs to user
    result = await session.execute(select(EmailCampaign).where(EmailCampaign.id == campaign_id, EmailCampaign.user_id == user.id))
    campaign = result.scalar_one_or_none()
    if not campaign:
        raise HTTPException(status_code=404, detail="Email campaign not found")
    result = await session.execute(select(FollowUpEmail).where(FollowUpEmail.id == followup_id, FollowUpEmail.email_campaign_id == campaign_id))
    followup = result.scalar_one_or_none()
    if not followup:
        raise HTTPException(status_code=404, detail="Follow-up email not found")
    await session.delete(followup)
    await session.commit()
    return {"message": "Follow-up email deleted"}



@router.get("/email-campaigns/performance")
async def email_campaign_performance(session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    # Aggregate sent and responses per day for user's campaigns
    result = await session.execute(
        select(Lead)
        .join(Campaign)
        .where(Campaign.user_id == user.id)
    )
    leads = result.scalars().all()
    # Group by date
    performance = {}
    for lead in leads:
        # Use sent_date or created_at as the date
        date = None
        if hasattr(lead, "sent_date") and lead.sent_date:
            date = lead.sent_date.date().isoformat()
        elif hasattr(lead, "created_at") and lead.created_at:
            date = lead.created_at.date().isoformat()
        else:
            continue
        if date not in performance:
            performance[date] = {"sent": 0, "responses": 0}
        if lead.status in ["contacted", "sent"]:
            performance[date]["sent"] += 1
        if lead.status == "replied":
            performance[date]["responses"] += 1
    # Convert to sorted list
    chart_data = [
        {"date": date, "sent": perf["sent"], "responses": perf["responses"]}
        for date, perf in sorted(performance.items())
    ]
    return chart_data






@router.post("/email-campaigns")
async def create_email_campaign(
    req: EmailCampaignCreate = Body(...),
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user)
):
    campaign = EmailCampaign(
        user_id=user.id,
        name=req.name,
        subject=req.subject,
        body=req.body,
        status="draft"
    )
    session.add(campaign)
    await session.commit()
    await session.refresh(campaign)

    # Store follow-up emails if provided
    followups = []
    if req.follow_ups:
        for fu in req.follow_ups:
            followup = FollowUpEmail(
                email_campaign_id=campaign.id,
                subject=fu.subject,
                body=fu.body,
                delay_days=fu.delay_days,
                    status="pending",
                    sender_email=getattr(user, 'sender_email', None) or os.getenv('FROM_EMAIL'),
                    sender_name=getattr(user, 'sender_name', None) or (user.email.split('@')[0] if user and getattr(user, 'email', None) else 'LeadPilot'),
                    reply_to_email=getattr(user, 'reply_to_email', None) or os.getenv('REPLY_TO_EMAIL') or os.getenv('FROM_EMAIL')
            )
            session.add(followup)
            followups.append(followup)
        await session.commit()

    return {"id": campaign.id, "message": "Email campaign created", "followups": [f.id for f in followups]}

@router.get("/email-campaigns")
async def list_email_campaigns(session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    result = await session.execute(select(EmailCampaign).where(EmailCampaign.user_id == user.id))
    campaigns = result.scalars().all()
    return [
        {
            "id": c.id,
            "name": c.name,
            "subject": c.subject,
            "body": c.body,
            "status": c.status,
            "scheduled_at": c.scheduled_at,
            "created_at": c.created_at
        } for c in campaigns
    ]

# Spam trigger words with alternatives
SPAM_TRIGGERS = {
    "buy now": ["check it out", "take a look", "see more", "learn more"],
    "100% free": ["complimentary", "no cost", "at no charge", "free of charge"],
    "free": ["complimentary", "no cost", "included"],
    "guaranteed": ["confident", "expected", "committed"],
    "risk-free": ["secure", "trusted", "reliable"],
    "act now": ["when ready", "at your convenience", "reach out"],
    "limited time": ["current opportunity", "special offer", "exclusive"],
    "urgent": ["important", "timely", "relevant"],
    "click here": ["visit", "check out", "see"],
    "make money": ["increase revenue", "grow income", "boost earnings"],
    "no obligation": ["flexible", "no pressure", "casual"],
}

def sanitize_spam_words(text: str) -> str:
    """Replace common spam trigger words with alternatives"""
    result = text
    text_lower = text.lower()
    
    for trigger, alternatives in SPAM_TRIGGERS.items():
        if trigger in text_lower:
            replacement = random.choice(alternatives)
            # Case-insensitive replacement
            pattern = re.compile(re.escape(trigger), re.IGNORECASE)
            result = pattern.sub(replacement, result)
    
    return result

async def ai_vary_content(subject: str, body: str, lead_data: dict, ai_api_key: str = None) -> dict:
    """
    Use AI to create unique variations of email content
    This example uses OpenAI - adapt to your AI provider
    """
    # No API key: deterministic fallback
    if not ai_api_key:
        return {
            "subject": sanitize_spam_words(subject),
            "body": sanitize_spam_words(body)
        }

    # Build prompt
    prompt = f"""Rewrite this cold email to be more unique and personalized while keeping the core message.
Make it conversational and natural. Avoid spam trigger words like "buy now", "free", "guaranteed", "act now".

Original Subject: {subject}
Original Body: {body}

Lead Info:
- Name: {lead_data.get('name', '')}
- Company: {lead_data.get('company', '')}
- Title: {lead_data.get('title', '')}

Provide:
1. A unique subject line (max 60 chars)
2. A personalized body that feels human-written

Return as JSON: {{"subject": "...", "body": "..."}}"""

    headers = {
        "Authorization": f"Bearer {ai_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 500
    }

    # Bound concurrency to avoid bursting AI provider
    try:
        async with ai_semaphore:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=payload,
                    headers=headers
                )

    except Exception as e:
        logging.error(f"AI request failed: {e}")
        return {"subject": sanitize_spam_words(subject), "body": sanitize_spam_words(body)}

    # Handle non-200 responses gracefully
    if response.status_code != 200:
        logging.warning(f"AI API returned {response.status_code}: {getattr(response, 'text', '')}")
        return {"subject": sanitize_spam_words(subject), "body": sanitize_spam_words(body)}

    try:
        result = response.json()
        ai_text = result.get("choices", [])[0].get("message", {}).get("content", "")
    except Exception as e:
        logging.warning(f"Failed to parse AI response JSON: {e}")
        ai_text = getattr(response, 'text', '') or ''

    # Try robust JSON extraction
    import json
    varied = None
    try:
        varied = json.loads(ai_text)
    except Exception:
        # attempt to extract JSON substring
        try:
            start = ai_text.find('{')
            end = ai_text.rfind('}')
            if start != -1 and end != -1 and end > start:
                snippet = ai_text[start:end+1]
                varied = json.loads(snippet)
        except Exception:
            varied = None

    # If JSON not found, try simple heuristics (Subject: ... then body)
    if not varied:
        try:
            lines = [l.strip() for l in ai_text.splitlines() if l.strip()]
            subj_lines = [l for l in lines if l.lower().startswith('subject:')]
            if subj_lines:
                subj = subj_lines[0].split(':', 1)[1].strip()
                # body is remainder
                body_lines = [l for l in lines if not l.lower().startswith('subject:')]
                bod = '\n'.join(body_lines).strip()
                varied = {"subject": subj or subject, "body": bod or body}
        except Exception:
            varied = None

    # Final fallback
    if not varied:
        logging.warning("AI variation not parsable; falling back to sanitized originals")
        return {"subject": sanitize_spam_words(subject), "body": sanitize_spam_words(body)}

    # Normalize returned values
    final_subject = sanitize_spam_words(str(varied.get("subject", subject)))
    final_body = sanitize_spam_words(str(varied.get("body", body)))
    return {"subject": final_subject, "body": final_body}

async def send_email_via_mailersend(
    to_email: str,
    to_name: str,
    subject: str,
    body: str,
    from_email: str,
    from_name: str,
    mailersend_api_key: str,
    reply_to_email: str = None,
    tags: List[str] = None,
    template_variables: dict = None
) -> dict:
    """
    Send an email via MailerSend API
    
    Args:
        to_email: Recipient email address
        to_name: Recipient name
        subject: Email subject
        body: Email body (HTML or plain text)
        from_email: Sender email (must be verified domain in MailerSend)
        from_name: Sender name
        mailersend_api_key: MailerSend API key
        reply_to_email: Optional reply-to email
        tags: Optional list of tags for tracking
        template_variables: Optional variables for template substitution
    
    Returns:
        dict with status, message_id, and any errors
    """
    
    # Prepare email payload for MailerSend
    payload = {
        "from": {
            "email": from_email,
            "name": from_name
        },
        "to": [
            {
                "email": to_email,
                "name": to_name
            }
        ],
        "subject": subject,
        "html": body,  # MailerSend accepts HTML
        "text": body   # Also include plain text version
    }
    
    # Add optional fields
    if reply_to_email:
        payload["reply_to"] = {
            "email": reply_to_email,
            "name": from_name
        }
    
    if tags:
        payload["tags"] = tags
    
    if template_variables:
        payload["variables"] = [
            {
                "email": to_email,
                "substitutions": [
                    {"var": key, "value": value}
                    for key, value in template_variables.items()
                ]
            }
        ]
    
    # Set up headers
    headers = {
        "Authorization": f"Bearer {mailersend_api_key}",
        "Content-Type": "application/json",
        "X-Requested-With": "XMLHttpRequest"
    }
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{MAILERSEND_API_URL}/email",
                json=payload,
                headers=headers
            )

        # Try to parse JSON safely
        resp_text = response.text if hasattr(response, 'text') else None
        resp_json = None
        if resp_text:
            try:
                resp_json = response.json()
            except Exception:
                resp_json = None

        # MailerSend may return 202 Accepted (or 200/201 depending on endpoint); treat 2xx as success
        if 200 <= response.status_code < 300:
            # attempt to extract a message id from common places
            message_id = None
            if resp_json:
                # Some MailerSend responses include an 'data' object or message id keys
                if isinstance(resp_json, dict):
                    # try common shapes
                    if 'data' in resp_json:
                        # data may be dict or list
                        data = resp_json['data']
                        if isinstance(data, dict):
                            message_id = data.get('id') or data.get('message_id')
                        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                            message_id = data[0].get('id') or data[0].get('message_id')
                    message_id = message_id or resp_json.get('message_id') or resp_json.get('id')

            # fallback to headers
            message_id = message_id or response.headers.get('X-Message-Id') or response.headers.get('Message-Id')

            return {
                "success": True,
                "message_id": message_id,
                "status_code": response.status_code,
                "response": resp_json or resp_text,
            }

        # Non-2xx: return diagnostic info
        err_body = None
        if resp_json:
            # try to find a message
            if isinstance(resp_json, dict):
                err_body = resp_json.get('message') or resp_json.get('error') or resp_json
            else:
                err_body = resp_json
        else:
            err_body = resp_text

        logging.warning(f"MailerSend returned non-2xx status: {response.status_code} - {err_body}")
        return {
            "success": False,
            "error": err_body,
            "status_code": response.status_code,
            "response": resp_json or resp_text,
        }

    except httpx.RequestError as e:
        logging.error(f"MailerSend request error: {e}")
        return {
            "success": False,
            "error": str(e),
            "status_code": None
        }
    except Exception as e:
        logging.error(f"Unexpected error sending via MailerSend: {e}")
        return {
            "success": False,
            "error": str(e),
            "status_code": None
        }
    
@router.post("/email-campaigns/{campaign_id}/send")
async def send_email_campaign(
    campaign_id: uuid.UUID,
    req: dict = Body(...),
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """
    Send an email campaign with AI-powered personalization and anti-spam measures via MailerSend.
    Features:
    - Throttled sending with randomized delays
    - AI-generated content variations per lead
    - Spam word detection and replacement
    - Deep personalization
    - Email tracking and analytics via MailerSend
    """
    
    # Get API keys: prefer per-user keys, fall back to environment variables
    MAILERSEND_API_KEY = getattr(user, 'mailersend_api_key', None) or os.getenv("MAILERSEND_API_KEY")
    AI_API_KEY = getattr(user, 'openai_api_key', None) or os.getenv("AI_API_KEY")

    # Get sender details (prefer user-configured sender, fall back to env)
    FROM_EMAIL = getattr(user, 'sender_email', None) or os.getenv("FROM_EMAIL")
    FROM_NAME = getattr(user, 'sender_name', None) or user.email.split('@')[0]
    REPLY_TO_EMAIL = getattr(user, 'reply_to_email', None) or FROM_EMAIL
    
    if not MAILERSEND_API_KEY:
        raise HTTPException(status_code=500, detail="MailerSend API key missing. Please add it in your settings.")
    
    if not FROM_EMAIL:
        raise HTTPException(status_code=500, detail="Sender email not configured. Please add a verified sender email in your settings.")
    
    # Fetch campaign
    result = await session.execute(
        select(EmailCampaign).where(
            EmailCampaign.id == campaign_id, 
            EmailCampaign.user_id == user.id
        )
    )
    campaign = result.scalar_one_or_none()
    if not campaign:
        raise HTTPException(status_code=404, detail="Email campaign not found")
    
    # Fetch leads
    leads_result = await session.execute(
        select(Lead).where(Lead.id.in_(req["leadIds"]))
    )
    leads = leads_result.scalars().all()
    
    # Configuration for throttling
    BATCH_SIZE = req.get("batchSize", 10)  # Emails per batch
    MIN_DELAY = req.get("minDelay", 5)     # Min seconds between emails
    MAX_DELAY = req.get("maxDelay", 30)    # Max seconds between emails
    BATCH_DELAY = req.get("batchDelay", 300)  # Seconds between batches (5 min)
    
    logs = []
    failed_count = 0
    sent_count = 0
    
    # Process leads in batches
    for batch_idx, i in enumerate(range(0, len(leads), BATCH_SIZE)):
        batch = leads[i:i + BATCH_SIZE]
        logging.info(f"Processing batch {batch_idx + 1}, {len(batch)} leads")
        
        for lead in batch:
            # Get personalization vars
            vars_dict = req.get("personalization", {}).get(str(lead.id), {})
            
            # Validate email
            to_email = getattr(lead, "email", None)
            if not to_email or "@" not in to_email:
                failed_count += 1
                log = EmailLog(
                    campaign_id=campaign.id,
                    lead_id=lead.id,
                    to_email=to_email or "",
                    status="failed",
                    sent_at=datetime.utcnow(),
                    error="Missing or invalid email address"
                )
                session.add(log)
                logs.append(log)
                continue
            
            # Prepare lead context for AI
            lead_data = {
                "name": f"{lead.first_name} {lead.last_name}",
                "first_name": lead.first_name,
                "company": getattr(lead, "company", ""),
                "title": getattr(lead, "title", ""),
                "industry": getattr(lead, "industry", ""),
                **vars_dict
            }
            
            # Apply basic template formatting first
            try:
                base_subject = campaign.subject.format(**vars_dict)
                base_body = campaign.body.format(**vars_dict)
            except KeyError as e:
                logging.warning(f"Template formatting error for lead {lead.id}: {e}")
                base_subject = campaign.subject
                base_body = campaign.body
            
            # AI-powered variation and personalization
            varied_content = await ai_vary_content(
                subject=base_subject,
                body=base_body,
                lead_data=lead_data,
                ai_api_key=AI_API_KEY
            )
            
            final_subject = varied_content["subject"]
            final_body = varied_content["body"]
            
            # Send via MailerSend with retry logic
            max_retries = 3
            backoff = 2
            status = "failed"
            error = None
            message_id = None
            
            # Prepare tags for tracking
            tags = [
                f"campaign:{campaign.name}",
                f"campaign_id:{campaign_id}",
                f"lead_id:{lead.id}"
            ]
            
            for attempt in range(max_retries):
                try:
                    result = await send_email_via_mailersend(
                        to_email=to_email,
                        to_name=f"{lead.first_name} {lead.last_name}",
                        subject=final_subject,
                        body=final_body,
                        from_email=FROM_EMAIL,
                        from_name=FROM_NAME,
                        mailersend_api_key=MAILERSEND_API_KEY,
                        reply_to_email=REPLY_TO_EMAIL,
                        tags=tags,
                        template_variables=vars_dict
                    )
                    
                    if result["success"]:
                        status = "sent"
                        message_id = result.get("message_id")
                        sent_count += 1
                        break
                    else:
                        # Check for rate limiting
                        if result.get("status_code") == 429:
                            logging.warning(f"Rate limit hit for lead {lead.id}, attempt {attempt+1}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(backoff ** attempt)
                                continue
                        
                        status = "failed"
                        error = result.get("error", "Unknown error")
                        
                        if attempt == max_retries - 1:
                            failed_count += 1
                        else:
                            await asyncio.sleep(backoff ** attempt)
                        
                except Exception as e:
                    error = str(e)
                    logging.error(f"Unexpected error for lead {lead.id}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(backoff ** attempt)
                    else:
                        failed_count += 1
            
            # Log the result
            log = EmailLog(
                campaign_id=campaign.id,
                lead_id=lead.id,
                to_email=to_email,
                status=status,
                sent_at=datetime.utcnow(),
                error=error
            )

            # Store MailerSend message_id for tracking (persist if available)
            if message_id:
                try:
                    log.external_message_id = message_id
                except Exception:
                    logging.warning(f"Failed to set external_message_id on EmailLog for lead {lead.id}")

            # If we didn't get a message id and the result had an error payload, log it for diagnostics
            if not message_id and result and not result.get("success"):
                logging.debug(f"MailerSend send result for lead {lead.id}: {result}")
            
            session.add(log)
            logs.append(log)
            
            # Update lead status
            if status == "sent":
                lead.status = "sent"
                lead.sent_date = datetime.utcnow()
                session.add(lead)
            
            # CRITICAL: Randomized delay between emails (anti-spam)
            delay = random.uniform(MIN_DELAY, MAX_DELAY)
            logging.info(f"Sent to {to_email}, waiting {delay:.1f}s before next email")
            await asyncio.sleep(delay)
        
        # Commit batch
        await session.commit()
        
        # Longer delay between batches (except for last batch)
        if i + BATCH_SIZE < len(leads):
            batch_wait = random.uniform(BATCH_DELAY * 0.8, BATCH_DELAY * 1.2)
            logging.info(f"Batch complete, waiting {batch_wait/60:.1f} minutes before next batch")
            await asyncio.sleep(batch_wait)
    
    # Schedule follow-ups
    followups_result = await session.execute(
        select(FollowUpEmail).where(FollowUpEmail.email_campaign_id == campaign.id)
    )
    followups = followups_result.scalars().all()
    
    now = datetime.utcnow()
    for lead in leads:
        # Only schedule follow-ups for successfully sent emails
        if lead.status != "sent":
            continue
            
        last_scheduled = now
        for fu in followups:
            # Add randomization to follow-up timing (±2 hours)
            random_offset = timedelta(hours=random.uniform(-2, 2))
            scheduled_time = last_scheduled + timedelta(days=fu.delay_days) + random_offset
            
            fu_instance = FollowUpEmail(
                email_campaign_id=campaign.id,
                subject=fu.subject,
                body=fu.body,
                delay_days=fu.delay_days,
                scheduled_at=scheduled_time,
                status="scheduled",
                sender_email=getattr(user, 'sender_email', None) or os.getenv('FROM_EMAIL'),
                sender_name=getattr(user, 'sender_name', None) or (user.email.split('@')[0] if user and getattr(user, 'email', None) else 'LeadPilot'),
                reply_to_email=getattr(user, 'reply_to_email', None) or os.getenv('REPLY_TO_EMAIL') or os.getenv('FROM_EMAIL')
            )
            session.add(fu_instance)
            last_scheduled = scheduled_time
    
    await session.commit()
    
    # Notify on failures
    if failed_count > 0:
        from models import Notification
        notification = Notification(
            user_id=user.id,
            type="email_failed",
            message=f"{failed_count} of {len(leads)} emails failed in campaign '{campaign.name}'.",
            created_at=datetime.utcnow()
        )
        session.add(notification)
        await session.commit()
    
    return {
        "message": f"Campaign complete: {sent_count} sent, {failed_count} failed",
        "sent": sent_count,
        "failed": failed_count,
        "total": len(leads),
        "ai_enabled": AI_API_KEY is not None,
        "provider": "MailerSend",
        "results": [
            {
                "lead_id": l.lead_id, 
                "status": l.status, 
                "error": l.error,
                "message_id": getattr(l, 'external_message_id', None)
            } for l in logs
        ]
    }

@router.get("/email-campaigns/{campaign_id}/logs")
async def get_email_campaign_logs(campaign_id: uuid.UUID, session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    result = await session.execute(select(EmailCampaign).where(EmailCampaign.id == campaign_id, EmailCampaign.user_id == user.id))
    campaign = result.scalar_one_or_none()
    if not campaign:
        raise HTTPException(status_code=404, detail="Email campaign not found")
    logs_result = await session.execute(select(EmailLog).where(EmailLog.campaign_id == campaign_id))
    logs = logs_result.scalars().all()
    # Fetch lead info for each log
    lead_ids = [log.lead_id for log in logs]
    leads_result = await session.execute(select(Lead).where(Lead.id.in_(lead_ids)))
    leads = {lead.id: lead for lead in leads_result.scalars().all()}
    return [
        {
            "lead_id": log.lead_id,
            "lead_name": f"{leads[log.lead_id].first_name} {leads[log.lead_id].last_name}" if log.lead_id in leads else None,
            "to_email": log.to_email,
            "status": log.status,
            "sent_at": log.sent_at,
            "opened_at": log.opened_at,
            "error": log.error,
            "message_id": getattr(log, 'external_message_id', None)
        }
        for log in logs
    ]


# PUT endpoint for editing drafted email campaigns
@router.put("/email-campaigns/{campaign_id}")
async def update_email_campaign(
    campaign_id: uuid.UUID = Path(...),
    req: dict = Body(...),
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user)
):
    result = await session.execute(select(EmailCampaign).where(EmailCampaign.id == campaign_id, EmailCampaign.user_id == user.id))
    campaign = result.scalar_one_or_none()
    if not campaign:
        raise HTTPException(status_code=404, detail="Email campaign not found")
    # Allow status change and field updates
    if "name" in req:
        campaign.name = req["name"]
    if "subject" in req:
        campaign.subject = req["subject"]
    if "body" in req:
        campaign.body = req["body"]
    if "status" in req:
        allowed_statuses = ["draft", "active", "paused", "completed"]
        new_status = req["status"]
        if new_status not in allowed_statuses:
            raise HTTPException(status_code=400, detail="Invalid status")
        # Only allow certain transitions for MVP
        if campaign.status == "draft" and new_status == "active":
            campaign.status = new_status
        elif campaign.status in ["active", "paused"] and new_status in ["paused", "active", "completed"]:
            campaign.status = new_status
        elif campaign.status == new_status:
            pass  # No change
        else:
            raise HTTPException(status_code=400, detail="Invalid status transition")
    session.add(campaign)
    await session.commit()
    await session.refresh(campaign)
    return {"id": campaign.id, "message": "Email campaign updated", "status": campaign.status}

# DELETE endpoint for deleting drafted email campaigns
@router.delete("/email-campaigns/{campaign_id}")
async def delete_email_campaign(
    campaign_id: uuid.UUID = Path(...),
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user)
):
    result = await session.execute(select(EmailCampaign).where(EmailCampaign.id == campaign_id, EmailCampaign.user_id == user.id))
    campaign = result.scalar_one_or_none()
    if not campaign:
        raise HTTPException(status_code=404, detail="Email campaign not found")
    await session.delete(campaign)
    await session.commit()
    return {"message": "Email campaign deleted"}

@router.get("/campaigns/stats")
async def get_campaign_stats(session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    from models import Lead, Campaign, LeadBatch
    # Aggregate lead stats for user's campaigns and batches
    campaign_leads_result = await session.execute(
        select(Lead)
        .join(Campaign)
        .where(Campaign.user_id == user.id)
    )
    campaign_leads = campaign_leads_result.scalars().all()

    batch_leads_result = await session.execute(
        select(Lead)
        .join(LeadBatch)
        .where(LeadBatch.user_id == user.id)
    )
    batch_leads = batch_leads_result.scalars().all()

    # Combine and deduplicate leads
    all_leads = {lead.id: lead for lead in campaign_leads}
    for lead in batch_leads:
        all_leads[lead.id] = lead
    leads = list(all_leads.values())

    sent = sum(1 for l in leads if l.status in ["sent", "contacted"])
    accepted = sum(1 for l in leads if l.status in ["accepted", "connected"])
    replied = sum(1 for l in leads if l.status == "replied")
    failed = sum(1 for l in leads if l.status in ["failed", "bounced"])
    return {
        "sent": sent,
        "accepted": accepted,
        "replied": replied,
        "failed": failed
    }