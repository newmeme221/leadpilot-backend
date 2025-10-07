import os
import logging
from datetime import datetime
from typing import Optional
import httpx
import random

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.interval import IntervalTrigger

from sqlmodel import select

from database import async_session
from models import FollowUpEmail, Lead, EmailLog

logger = logging.getLogger(__name__)


async def send_via_mailersend_raw(
    to_email: str,
    to_name: str,
    subject: str,
    body: str,
    from_email: str,
    from_name: str,
    mailersend_api_key: str,
    reply_to_email: Optional[str] = None,
    tags: Optional[list] = None,
    template_variables: Optional[dict] = None,
):
    MAILERSEND_API_URL = "https://api.mailersend.com/v1"
    payload = {
        "from": {"email": from_email, "name": from_name},
        "to": [{"email": to_email, "name": to_name}],
        "subject": subject,
        "html": body,
        "text": body,
    }
    if reply_to_email:
        payload["reply_to"] = {"email": reply_to_email, "name": from_name}
    if tags:
        payload["tags"] = tags
    if template_variables:
        payload["variables"] = [
            {"email": to_email, "substitutions": [{"var": k, "value": v} for k, v in template_variables.items()]}]

    headers = {
        "Authorization": f"Bearer {mailersend_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(f"{MAILERSEND_API_URL}/email", json=payload, headers=headers)
        if r.status_code == 202:
            data = r.json() if r.text else {}
            return {"success": True, "message_id": data.get("message_id") or r.headers.get("X-Message-Id")}
        else:
            return {"success": False, "error": r.text, "status_code": r.status_code}
    except Exception as e:
        logger.error(f"Followup send error: {e}")
        return {"success": False, "error": str(e)}


async def process_due_followups():
    """One-shot job: send any followups scheduled at or before now."""
    now = datetime.utcnow()
    try:
        async with async_session() as session:
            stmt = select(FollowUpEmail).where(FollowUpEmail.status == "scheduled", FollowUpEmail.scheduled_at <= now).limit(100)
            result = await session.execute(stmt)
            due = result.scalars().all()

            for fu in due:
                try:
                    # find leads for the campaign
                    lead_rows = await session.exec(select(Lead).where(Lead.campaign_id == fu.email_campaign_id))
                    leads = lead_rows.all()
                    for lead in leads:
                        if not getattr(lead, 'email', None):
                            continue
                        MAILERSEND_API_KEY = os.getenv("MAILERSEND_API_KEY")
                        # Prefer per-followup sender metadata when present
                        FROM_EMAIL = fu.sender_email or os.getenv("FROM_EMAIL")
                        FROM_NAME = fu.sender_name or os.getenv("FROM_NAME") or "LeadPilot"
                        REPLY_TO = fu.reply_to_email or os.getenv("REPLY_TO_EMAIL") or FROM_EMAIL

                        vars_dict = {}
                        final_subject = fu.subject
                        final_body = fu.body

                        tags = [f"followup", f"campaign_id:{fu.email_campaign_id}", f"lead_id:{lead.id}"]

                        res = await send_via_mailersend_raw(
                            to_email=lead.email,
                            to_name=f"{lead.first_name or ''} {lead.last_name or ''}".strip(),
                            subject=final_subject,
                            body=final_body,
                            from_email=FROM_EMAIL,
                            from_name=FROM_NAME,
                            mailersend_api_key=MAILERSEND_API_KEY,
                            reply_to_email=REPLY_TO,
                            tags=tags,
                            template_variables=vars_dict
                        )

                        log = EmailLog(
                            campaign_id=fu.email_campaign_id,
                            lead_id=lead.id,
                            to_email=lead.email,
                            status="sent" if res.get("success") else "failed",
                            sent_at=datetime.utcnow(),
                            error=res.get("error") if not res.get("success") else None,
                            external_message_id=res.get("message_id")
                        )
                        session.add(log)

                    fu.status = "completed"
                    session.add(fu)
                except Exception as e:
                    logger.error(f"Error processing scheduled followup {getattr(fu, 'id', None)}: {e}")

            await session.commit()
    except Exception as e:
        logger.error(f"Error processing due followups: {e}")


_scheduler: Optional[AsyncIOScheduler] = None


def start_scheduler(interval_seconds: int = 60):
    global _scheduler
    if _scheduler is not None:
        logger.info("FollowUp APScheduler already running")
        return

    jobstores = None
    # Optionally use SQLAlchemyJobStore if DATABASE_URL provided and env var set
    use_sql = os.getenv("USE_SQLALCHEMY_JOBSTORE", "false").lower() in ("1", "true", "yes")
    if use_sql:
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            jobstores = {'default': SQLAlchemyJobStore(url=db_url)}
        else:
            logger.warning("USE_SQLALCHEMY_JOBSTORE set but DATABASE_URL not provided — falling back to memory jobstore")

    _scheduler = AsyncIOScheduler(jobstores=jobstores) if jobstores else AsyncIOScheduler()
    trigger = IntervalTrigger(seconds=interval_seconds)
    # max_instances=1 prevents overlapping runs
    _scheduler.add_job(process_due_followups, trigger, id="process_due_followups", max_instances=1, coalesce=True)
    _scheduler.start()
    logger.info(f"FollowUp APScheduler started (interval={interval_seconds}s, use_sql={use_sql})")


def stop_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("FollowUp APScheduler stopped")
