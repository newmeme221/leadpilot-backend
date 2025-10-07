
from fastapi import APIRouter, Depends, HTTPException, Body, BackgroundTasks
from sqlmodel import select
from sqlalchemy import or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from models import Lead, User, Campaign, LeadHistory, LeadBatch
from database import get_session, async_session
from auth_utils import get_current_user
from schemas import LeadOut, LeadListResponse
from typing import List

import csv
from io import StringIO
from collections import defaultdict, deque
import os
import time
import logging
import random
from fastapi.responses import StreamingResponse
from datetime import datetime 
from email_verification import (
    verify_email_zerobounce,
    should_accept_email,
    check_zerobounce_rate_limit,
     bulk_verify_emails
) 
import re
import uuid
import socket
import unicodedata 




try:    
    from difflib import SequenceMatcher
except ImportError:
    SequenceMatcher = None

def normalize_str(s):
    if not s:
        return ""
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'[^a-z0-9]', '', s.lower())

def fuzzy_similar(a, b):
    if not a or not b:
        return False
    if SequenceMatcher:
        ratio = SequenceMatcher(None, a, b).ratio()
        return ratio > 0.85
    return a == b

router = APIRouter(prefix="/api/leads", tags=["leads"])

# Import subscription plans
from routers.subscriptions import SUBSCRIPTION_PLANS

ENRICH_RATE_LIMIT = 10  # requests
ENRICH_RATE_PERIOD = 60  # seconds
user_enrich_timestamps = defaultdict(lambda: deque(maxlen=ENRICH_RATE_LIMIT))

def check_enrich_rate_limit(user_id: uuid.UUID):
    now = time.time()
    timestamps = user_enrich_timestamps[user_id]
    # Remove timestamps older than ENRICH_RATE_PERIOD
    while timestamps and now - timestamps[0] > ENRICH_RATE_PERIOD:
        timestamps.popleft()
    if len(timestamps) >= ENRICH_RATE_LIMIT:
        return False
    timestamps.append(now)
    return True

def is_valid_email(email: str) -> bool:
    """Basic email format validation"""
    if not email:
        return False
    # Basic regex check
    if not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email):
        return False
    # MX record check for domain
    try:
        domain = email.split('@')[1]
        socket.getaddrinfo(domain, None)
        return True
    except Exception:
        return False
# Enhanced upload: deduplication, validation, Hunter.io enrichment
@router.post("/upload")
async def upload_leads(req: dict, session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    plan = SUBSCRIPTION_PLANS.get(user.subscription_tier, SUBSCRIPTION_PLANS["free"])
    limit = plan["leads_limit"]

    batch_name = req.get("batch_name")
    if not batch_name:
        raise HTTPException(status_code=400, detail="batch_name is required")
    
    # Get ZeroBounce settings from request
    enable_zerobounce = req.get("enable_zerobounce", True)  # Enable by default
    strict_mode = req.get("strict_mode", False)  # Strict validation mode
    
    # Check if batch exists for this user and name
    batch_result = await session.execute(select(LeadBatch).where(LeadBatch.name == batch_name, LeadBatch.user_id == user.id))
    batch = batch_result.scalar_one_or_none()
    if not batch:
        batch = LeadBatch(name=batch_name, user_id=user.id)
        session.add(batch) 
        await session.flush()
        await session.commit()
        await session.refresh(batch)
    
    # Count and dedupe against all leads owned by user via campaign OR batch
    leads_result = await session.execute(
        select(Lead)
        .join(Campaign, isouter=True)
        .join(LeadBatch, Lead.batch_id == LeadBatch.id, isouter=True)
        .where(or_(Campaign.user_id == user.id, LeadBatch.user_id == user.id))
    )
    existing_leads = leads_result.scalars().all()
    leads_count = len(existing_leads)
    warning_threshold = limit * 0.8
    is_near_limit = leads_count >= warning_threshold
    is_over_limit = leads_count >= limit
    
    if is_over_limit:
        return {
            "message": f"Upload completed, but you've reached your {user.subscription_tier} plan limit of {limit} leads. Consider upgrading for more capacity.",
            "leads_created": 0,
            "warning": "limit_reached",
            "current_usage": leads_count,
            "limit": limit,
            "upgrade_required": True
        }
    
    csv_data = req.get("csvData", "")
    if not csv_data:
        raise HTTPException(status_code=400, detail="No CSV data provided")
    
    leads_created = 0
    reader = csv.DictReader(StringIO(csv_data))
    required_columns = {"first_name", "last_name", "company", "profile_url", "job_title"}
    missing_columns = required_columns - set(reader.fieldnames or [])
    
    if missing_columns:
        return {
            "message": f"CSV missing required columns: {', '.join(missing_columns)}",
            "leads_created": 0,
            "current_usage": leads_count,
            "limit": limit,
            "warning": "csv_invalid",
            "row_errors": []
        }
    
    # Prepare normalized existing leads for fuzzy matching
    deduped = [
        (normalize_str(l.first_name), normalize_str(l.last_name), normalize_str(l.company), normalize_str(l.profile_url))
        for l in existing_leads
    ]
    existing_emails = set(getattr(l, "email", None) for l in existing_leads if getattr(l, "email", None))
    
    row_errors = []
    zerobounce_api_key = getattr(user, "zerobounce_api_key", None) or os.getenv("ZEROBOUNCE_API_KEY")
    
    # Track verification stats
    verification_stats = {
        "total_verified": 0,
        "valid": 0,
        "invalid": 0,
        "catch_all": 0,
        "unknown": 0,
        "skipped": 0
    }
    
    for idx, row in enumerate(reader, start=2):
        key = (
            normalize_str(row.get("first_name", "")),
            normalize_str(row.get("last_name", "")),
            normalize_str(row.get("company", "")),
            normalize_str(row.get("profile_url", ""))
        )
        email = row.get("email", "").strip()
        
        # Basic email format validation
        if email and not is_valid_email(email):
            row_errors.append({"row": idx, "reason": "Invalid email format", "row_data": row})
            continue
        
        # ZeroBounce verification (if enabled and email present)
        email_verification_status = None
        email_verification_score = None 
        
        if email and enable_zerobounce and zerobounce_api_key:
            # Check per-user ZeroBounce rate limit before attempting verification
            try:
                if not check_zerobounce_rate_limit(user.id):
                    row_errors.append({"row": idx, "reason": "ZeroBounce rate limit exceeded", "row_data": row})
                    continue
            except Exception:
                # If rate limit check fails for any reason, proceed cautiously
                pass

            try:
                verification = await verify_email_zerobounce(email=email, api_key=zerobounce_api_key)
                verification_stats["total_verified"] += 1
                status = verification.get("status", "unknown")
                verification_stats[status] = verification_stats.get(status, 0) + 1

                # Check if email should be accepted
                accepted, reason = should_accept_email(verification, strict_mode=strict_mode)
                if not accepted:
                    row_errors.append({
                        "row": idx,
                        "reason": reason,
                        "row_data": row,
                        "verification_details": {
                            "status": verification.get("status"),
                            "score": verification.get("score"),
                        }
                    })
                    continue

                # Normalize status to "valid" or "invalid" for display
                raw_status = verification.get("status", "unknown")
                email_verification_status = "valid" if raw_status == "valid" else "invalid"
                email_verification_score = verification.get("score")
            except Exception as e:
                logging.error(f"ZeroBounce verification error for {email}: {e}")
                if strict_mode:
                    row_errors.append({
                        "row": idx,
                        "reason": f"Email verification failed: {str(e)}",
                        "row_data": row
                    })
                    continue
                else:
                    # In lenient mode, mark verification as unknown and proceed
                    email_verification_status = "unknown"
                    email_verification_score = None
        else:
            # No verification performed for this row
            email_verification_status = None
            email_verification_score = None
        
        # Fuzzy duplicate check
        is_duplicate = False
        for exist in deduped:
            if (
                fuzzy_similar(key[0], exist[0]) and
                fuzzy_similar(key[1], exist[1]) and
                fuzzy_similar(key[2], exist[2]) and
                (not key[3] or not exist[3] or fuzzy_similar(key[3], exist[3]))
            ):
                is_duplicate = True
                break
        
        if is_duplicate or (email and email in existing_emails):
            row_errors.append({"row": idx, "reason": "Duplicate lead (fuzzy)", "row_data": row})
            continue
        
        if leads_created + leads_count >= limit:
            row_errors.append({"row": idx, "reason": "Plan lead limit reached", "row_data": row})
            break
        
        try:
            lead = Lead(
                first_name=row.get("first_name", ""),
                last_name=row.get("last_name", ""),
                job_title=row.get("job_title", ""),
                company=row.get("company", ""),
                profile_url=row.get("profile_url", ""),
                status="pending",
                batch_id=batch.id,
                email=email if email else None,
                email_verification_status=email_verification_status,
                email_verification_score=email_verification_score
            )
            session.add(lead)
            session.add(LeadHistory(
                lead_id=lead.id,
                user_id=user.id,
                action="created",
                field=None,
                old_value=None,
                new_value=None
            ))
            leads_created += 1
            deduped.append(key)
            if email:
                existing_emails.add(email)
        except Exception as e:
            row_errors.append({"row": idx, "reason": f"Exception: {str(e)}", "row_data": row})
    
    try:
        await session.commit()
        logger.info(f"User {user.id} uploaded {leads_created} leads to batch '{batch_name}'")
    except Exception as e:
        logger.error(f"User {user.id} failed to upload leads: {str(e)}")
        raise
    
    response = {
        "message": f"Successfully uploaded {leads_created} leads",
        "leads_created": leads_created,
        "current_usage": leads_count + leads_created,
        "limit": limit,
        "row_errors": row_errors,
        "verification_stats": verification_stats if enable_zerobounce else None
    }
    # include batch id for frontend convenience
    try:
        response["batch_id"] = str(batch.id)
    except Exception:
        pass
    
    if is_near_limit and not is_over_limit:
        response["warning"] = "approaching_limit"
        response["message"] += f". You're approaching your {user.subscription_tier} plan limit ({leads_count + leads_created}/{limit} leads)."
    
    return response




task_status = {}
# Track cancelled tasks
cancelled_tasks = set()
@router.post("/cancel-task/{task_id}")
async def cancel_task(task_id: str):
    cancelled_tasks.add(task_id)
    # Optionally update status immediately
    if task_id in task_status:
        task_status[task_id]["status"] = "cancelled"
        task_status[task_id]["step"] = "Task cancelled by user"
    return {"message": f"Task {task_id} cancellation requested."}

def set_task_status(task_id, status, result=None):
    # Allow progress reporting
    if isinstance(result, dict) and "progress" in result:
        task_status[task_id] = {"status": status, **result}
    else:
        task_status[task_id] = {"status": status, "result": result}

def get_task_status(task_id):
    return task_status.get(task_id, {"status": "unknown"})


async def _background_scrape_apollo(task_id: str, user_id: uuid.UUID, request: dict):
    """Background worker that performs paged Apollo scraping and updates task_status."""
    set_task_status(task_id, "pending", {"progress": 0, "step": "queued"})
    try:
        async with async_session() as session:
            # Refresh user from DB to get API key
            user = await session.get(User, user_id)
            if not user or not user.appollo_api_key:
                set_task_status(task_id, "error", {"result": "Apollo API key not configured"})
                return

            # Extract params
            person_titles = request.get("person_titles", [])
            person_locations = request.get("person_locations", [])
            q_keywords = request.get("q_keywords", "")
            organization_domains = request.get("organization_domains", "")
            per_page = min(request.get("per_page", 25), 100)
            max_pages = int(request.get("max_pages", 1))
            max_pages = max(1, min(max_pages, 10))
            batch_name = request.get("batch_name", f"Apollo Import {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")

            plan = SUBSCRIPTION_PLANS.get(user.subscription_tier, SUBSCRIPTION_PLANS["free"])
            limit = plan["leads_limit"]

            # Count existing leads for this user
            leads_result = await session.execute(
                select(Lead)
                .join(Campaign, isouter=True)
                .join(LeadBatch, Lead.batch_id == LeadBatch.id, isouter=True)
                .where(or_(Campaign.user_id == user.id, LeadBatch.user_id == user.id))
            )
            existing_leads = leads_result.scalars().all()
            leads_count = len(existing_leads)

            if leads_count >= limit:
                set_task_status(task_id, "error", {"result": f"Lead limit reached ({leads_count}/{limit})"})
                return

            # Prepare request
            apollo_url = "https://api.apollo.io/api/v1/mixed_people/search"
            headers = {
                "Content-Type": "application/json",
                "Cache-Control": "no-cache",
                "X-Api-Key": user.appollo_api_key
            }

            # Build JSON payload for Apollo API
            payload = {"page": 1, "per_page": per_page}
            if person_titles:
                payload["person_titles"] = person_titles if isinstance(person_titles, list) else [person_titles]
            if person_locations:
                payload["person_locations"] = person_locations if isinstance(person_locations, list) else [person_locations]
            if q_keywords:
                payload["q_keywords"] = q_keywords
            if organization_domains:
                payload["q_organization_domains"] = organization_domains

            all_people = []
            leads_created = 0

            async with httpx.AsyncClient(timeout=30.0) as client:
                for page in range(1, max_pages + 1):
                    if task_id in cancelled_tasks:
                        set_task_status(task_id, "cancelled", {"result": "Task cancelled by user"})
                        return
                    payload["page"] = page
                    set_task_status(task_id, "in_progress", {"progress": int((page-1)/max_pages*50), "step": f"fetching page {page}"})
                    # Retry on 429 with exponential backoff and jitter
                    max_retries = 4
                    attempt = 0
                    resp = None
                    while attempt <= max_retries:
                        try:
                            resp = await client.post(apollo_url, headers=headers, json=payload)
                        except Exception as e:
                            set_task_status(task_id, "error", {"result": f"Request failed: {str(e)}"})
                            return

                        if resp.status_code == 401:
                            set_task_status(task_id, "error", {"result": "Invalid Apollo.io API key"})
                            return
                        if resp.status_code == 429:
                            # Backoff and retry
                            attempt += 1
                            backoff = (2 ** attempt) + (random.random() * 0.5)
                            set_task_status(task_id, "in_progress", {"progress": int((page-1)/max_pages*50), "step": f"rate limited, retry {attempt}/{max_retries}", "retry": attempt})
                            if attempt > max_retries:
                                set_task_status(task_id, "error", {"result": "Apollo.io rate limit exceeded after retries"})
                                return
                            await asyncio.sleep(backoff)
                            continue
                        if resp.status_code >= 400:
                            set_task_status(task_id, "error", {"result": f"Apollo.io API error: {resp.text}"})
                            return
                        break

                    data = resp.json()
                    apollo_people = data.get("people", [])
                    if not apollo_people:
                        break

                    # Create or fetch batch (ensure it exists)
                    batch_result = await session.execute(
                        select(LeadBatch).where(LeadBatch.name == batch_name, LeadBatch.user_id == user.id)
                    )
                    batch = batch_result.scalar_one_or_none()
                    if not batch:
                        batch = LeadBatch(name=batch_name, user_id=user.id)
                        session.add(batch)
                        await session.flush()
                        await session.refresh(batch)

                    existing_profile_urls = {normalize_str(l.profile_url) for l in existing_leads if l.profile_url}

                    for person in apollo_people:
                        if task_id in cancelled_tasks:
                            set_task_status(task_id, "cancelled", {"result": "Task cancelled by user"})
                            return
                        first_name = person.get("first_name", "")
                        last_name = person.get("last_name", "")
                        title = person.get("title", "")
                        linkedin_url = person.get("linkedin_url", "")
                        email = person.get("email")
                        organization = person.get("organization") or {}
                        company_name = organization.get("name", "")

                        if not linkedin_url:
                            continue
                        if normalize_str(linkedin_url) in existing_profile_urls:
                            continue
                        if leads_count + leads_created >= limit:
                            set_task_status(task_id, "in_progress", {"progress": 90, "step": "plan limit reached, stopping"})
                            break

                        lead = Lead(
                            first_name=first_name or "Unknown",
                            last_name=last_name or "Unknown",
                            job_title=title or "Unknown",
                            company=company_name or "Unknown",
                            profile_url=linkedin_url,
                            status="pending",
                            batch_id=batch.id,
                            email=email
                        )
                        session.add(lead)
                        existing_profile_urls.add(normalize_str(linkedin_url))
                        leads_created += 1

                    # Commit after each page to persist progress
                    await session.commit()
                    set_task_status(task_id, "in_progress", {"progress": int(page/max_pages*80), "step": f"imported page {page}", "result": {"leads_created": leads_created}})

                    # If plan limit reached, stop
                    if leads_count + leads_created >= limit:
                        break

            # Finalize notification and status
            notification = Notification(
                user_id=user.id,
                type="apollo_import",
                message=f"Imported {leads_created} leads from Apollo.io into batch '{batch_name}'",
                created_at=datetime.utcnow()
            )
            async with async_session() as s2:
                s2.add(notification)
                await s2.commit()

            # include batch id in final result for frontend linking
            try:
                set_task_status(task_id, "done", {"result": {"scraped": leads_created, "batch_id": str(batch.id)}})
            except Exception:
                set_task_status(task_id, "done", {"result": {"scraped": leads_created}})
    except Exception as e:
        logger.error(f"Background Apollo scrape failed: {e}")
        set_task_status(task_id, "error", {"result": str(e)})


@router.get("/task-status/{task_id}")
async def task_status_endpoint(task_id: str):
    return get_task_status(task_id)





@router.get("/list", response_model=LeadListResponse)
async def list_leads(
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
    unassigned: bool = False,
    offset: int = 0,
    limit: int = 20,
    company: str = None,
    job_title: str = None,
    status: str = None,
    enriched: bool = None,
    search: str = None,
    email: str = None,
    profile_url: str = None,
    batch_id: uuid.UUID = None
):
    plan = SUBSCRIPTION_PLANS.get(user.subscription_tier, SUBSCRIPTION_PLANS["free"])
    max_limit = plan["leads_limit"]
    # Include leads owned by user either via campaign ownership or batch ownership
    base_query = (
        select(Lead)
        .join(Campaign, isouter=True)
        .join(LeadBatch, Lead.batch_id == LeadBatch.id, isouter=True)
        .where(or_(Campaign.user_id == user.id, LeadBatch.user_id == user.id))
    )
    if batch_id:
        base_query = base_query.where(Lead.batch_id == batch_id)
    # Filtering
    if unassigned:
        base_query = base_query.where(Lead.campaign_id == None)
    if company:
        base_query = base_query.where(Lead.company.ilike(f"%{company}%"))
    if job_title:
        base_query = base_query.where(Lead.job_title.ilike(f"%{job_title}%"))
    if status:
        base_query = base_query.where(Lead.status == status)
    if enriched is not None:
        if enriched:
            base_query = base_query.where(Lead.email != None)
        else:
            base_query = base_query.where(Lead.email == None)
    if email:
        base_query = base_query.where(Lead.email.ilike(f"%{email}%"))
    if profile_url:
        base_query = base_query.where(Lead.profile_url.ilike(f"%{profile_url}%"))
    # Advanced search (across more fields)
    if search:
        search_pattern = f"%{search}%"
        base_query = base_query.where(
            (Lead.first_name.ilike(search_pattern)) |
            (Lead.last_name.ilike(search_pattern)) |
            (Lead.company.ilike(search_pattern)) |
            (Lead.job_title.ilike(search_pattern)) |
            (Lead.email.ilike(search_pattern)) |
            (Lead.profile_url.ilike(search_pattern))
        )
    # Get total count for pagination (COUNT(*))
    count_query = base_query.with_only_columns(func.count(Lead.id)).order_by(None)
    total_result = await session.execute(count_query)
    total_count = total_result.scalar() or 0
    # Pagination
    query = base_query.offset(offset).limit(min(limit, max_limit))
    result = await session.execute(query)
    leads = result.scalars().all()
    return {"leads": leads, "total": total_count, "offset": offset, "limit": limit}

@router.put("/{lead_id}")
async def update_lead(lead_id: uuid.UUID, req: dict, session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    # Get lead and verify it belongs to user via campaign or batch
    result = await session.execute(
        select(Lead)
        .join(Campaign, isouter=True)
        .join(LeadBatch, Lead.batch_id == LeadBatch.id, isouter=True)
        .where(Lead.id == lead_id)
        .where(or_(Campaign.user_id == user.id, LeadBatch.user_id == user.id))
    )
    lead = result.scalar_one_or_none()
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    status_before = lead.status
    # Allow updating select core fields
    updatable_fields = [
        "status", "first_name", "last_name", "job_title", "company", "profile_url", "message_text"
    ]
    for field_name in updatable_fields:
        if field_name in req:
            old_value = getattr(lead, field_name)
            new_value = req[field_name]
            setattr(lead, field_name, new_value)
            if field_name == "status" or old_value != new_value:
                session.add(LeadHistory(
                    lead_id=lead.id,
                    user_id=user.id,
                    action="updated",
                    field=field_name,
                    old_value=str(old_value) if old_value is not None else None,
                    new_value=str(new_value) if new_value is not None else None
                ))
    session.add(lead)
    try:
        await session.commit()
        logger.info(f"User {user.id} updated lead {lead_id} (status: {lead.status})")
    except Exception as e:
        logger.error(f"User {user.id} failed to update lead {lead_id}: {str(e)}")
        raise
    # Notification trigger for lead update
    if status_before != lead.status:
        from models import Notification
        notification = Notification(
            user_id=user.id,
            type="lead_updated",
            message=f"Lead '{lead.first_name} {lead.last_name}' status updated to '{lead.status}'.",
            created_at=datetime.utcnow()
        )
        session.add(notification)
        await session.commit()
    return {"message": "Lead updated"}

@router.delete("/{lead_id}")
async def delete_lead(lead_id: uuid.UUID, session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    # Get lead and verify it belongs to user via campaign or batch
    result = await session.execute(
        select(Lead)
        .join(Campaign, isouter=True)
        .join(LeadBatch, Lead.batch_id == LeadBatch.id, isouter=True)
        .where(Lead.id == lead_id)
        .where(or_(Campaign.user_id == user.id, LeadBatch.user_id == user.id))
    )
    lead = result.scalar_one_or_none()
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    session.add(LeadHistory(
        lead_id=lead.id,
        user_id=user.id,
        action="deleted",
        field=None,
        old_value=None,
        new_value=None
    ))
    try:
        await session.delete(lead)
        await session.commit()
        logger.info(f"User {user.id} deleted lead {lead_id}")
    except Exception as e:
        logger.error(f"User {user.id} failed to delete lead {lead_id}: {str(e)}")
        raise
    return {"message": "Lead deleted"}

import requests
import asyncio
import httpx
from models import User, Notification
from database import get_session


@router.post("/assign-to-campaign")
async def assign_leads_to_campaign(req: dict = Body(...), session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    lead_ids = req.get("leadIds", [])
    campaign_id = req.get("campaignId")
    if not lead_ids or not campaign_id:
        raise HTTPException(status_code=400, detail="leadIds and campaignId are required")
    # Check campaign ownership
    campaign_result = await session.execute(select(Campaign).where(Campaign.id == campaign_id, Campaign.user_id == user.id))
    campaign = campaign_result.scalar_one_or_none()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    # Assign leads
    leads_result = await session.execute(select(Lead).where(Lead.id.in_(lead_ids), Lead.campaign_id == None))
    leads = leads_result.scalars().all()
    for lead in leads:
        old_campaign_id = lead.campaign_id
        lead.campaign_id = campaign_id
        session.add(LeadHistory(
            lead_id=lead.id,
            user_id=user.id,
            action="assigned",
            field="campaign_id",
            old_value=str(old_campaign_id) if old_campaign_id else None,
            new_value=str(campaign_id)
        ))
    try:
        await session.commit()
        logger.info(f"User {user.id} assigned {len(leads)} leads to campaign {campaign_id}")
    except Exception as e:
        logger.error(f"User {user.id} failed to assign leads to campaign {campaign_id}: {str(e)}")
        raise
    # Notification trigger for lead assignment
    if leads:
        from models import Notification
        notification = Notification(
            user_id=user.id,
            type="lead_assigned",
            message=f"{len(leads)} leads assigned to campaign '{campaign.name}'.",
            created_at=datetime.utcnow()
        )
        session.add(notification)
        await session.commit()
    return {"message": f"Assigned {len(leads)} leads to campaign"}

@router.post("/bulk-update")
async def bulk_update_leads(
    req: dict = Body(...),
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user)
):
    lead_ids = req.get("leadIds", [])
    updates = req.get("updates", {})
    if not lead_ids or not updates:
        raise HTTPException(status_code=400, detail="leadIds and updates are required")
    # allow updating leads owned via campaign OR batch
    result = await session.execute(
        select(Lead)
        .join(Campaign, isouter=True)
        .join(LeadBatch, Lead.batch_id == LeadBatch.id, isouter=True)
        .where(Lead.id.in_(lead_ids))
        .where(or_(Campaign.user_id == user.id, LeadBatch.user_id == user.id))
    )
    leads = result.scalars().all()
    for lead in leads:
        for k, v in updates.items():
            if hasattr(lead, k):
                setattr(lead, k, v)
        session.add(lead)
    try:
        await session.commit()
        logger.info(f"User {user.id} performed bulk update on leads: {lead_ids}")
    except Exception as e:
        logger.error(f"User {user.id} failed bulk update on leads: {lead_ids}, error: {str(e)}")
        raise
    return {"message": f"Updated {len(leads)} leads"}

@router.post("/bulk-delete")
async def bulk_delete_leads(
    req: dict = Body(...),
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user)
):
    lead_ids = req.get("leadIds", [])
    if not lead_ids:
        raise HTTPException(status_code=400, detail="leadIds are required")
    # allow deleting leads owned via campaign OR batch
    result = await session.execute(
        select(Lead)
        .join(Campaign, isouter=True)
        .join(LeadBatch, Lead.batch_id == LeadBatch.id, isouter=True)
        .where(Lead.id.in_(lead_ids))
        .where(or_(Campaign.user_id == user.id, LeadBatch.user_id == user.id))
    )
    leads = result.scalars().all()
    for lead in leads:
        await session.delete(lead)
    try:
        await session.commit()
        logger.info(f"User {user.id} performed bulk delete on leads: {lead_ids}")
    except Exception as e:
        logger.error(f"User {user.id} failed bulk delete on leads: {lead_ids}, error: {str(e)}")
        raise
    return {"message": f"Deleted {len(leads)} leads"}



@router.get("/export")
async def export_leads(
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
    unassigned: bool = False,
    batch_id: uuid.UUID | None = None,
):
    query = (
        select(Lead)
        .join(Campaign, isouter=True)
        .join(LeadBatch, Lead.batch_id == LeadBatch.id, isouter=True)
        .where(or_(Campaign.user_id == user.id, LeadBatch.user_id == user.id))
    )
    if unassigned:
        query = query.where(Lead.campaign_id == None)
    if batch_id:
        query = query.where(Lead.batch_id == batch_id)
    result = await session.execute(query)
    leads = result.scalars().all()
    # Prepare CSV
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["First Name", "Last Name", "Job Title", "Company", "Profile URL", "Status", "Email", "Email Confidence"])
    for lead in leads:
        writer.writerow([
            lead.first_name,
            lead.last_name,
            lead.job_title,
            lead.company,
            lead.profile_url,
            lead.status,
            getattr(lead, "email", ""),
            getattr(lead, "email_confidence", "")
        ])
    output.seek(0)
    return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=leads.csv"})

@router.get("/{lead_id}/history")
async def get_lead_history(lead_id: uuid.UUID, session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    # Only allow access to leads owned by user (campaign or batch)
    result = await session.execute(
        select(Lead)
        .join(Campaign, isouter=True)
        .join(LeadBatch, Lead.batch_id == LeadBatch.id, isouter=True)
        .where(Lead.id == lead_id)
        .where(or_(Campaign.user_id == user.id, LeadBatch.user_id == user.id))
    )
    lead = result.scalar_one_or_none()
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    history_result = await session.execute(select(LeadHistory).where(LeadHistory.lead_id == lead_id).order_by(LeadHistory.timestamp.desc()))
    history = history_result.scalars().all()
    return [{
        "action": h.action,
        "field": h.field,
        "old_value": h.old_value,
        "new_value": h.new_value,
        "timestamp": h.timestamp
    } for h in history]

@router.get("/{lead_id}/replies")
async def get_lead_replies(lead_id: uuid.UUID, session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    from models import LeadReply
    # Only allow access to leads owned by user (campaign or batch)
    result = await session.execute(
        select(Lead)
        .join(Campaign, isouter=True)
        .join(LeadBatch, Lead.batch_id == LeadBatch.id, isouter=True)
        .where(Lead.id == lead_id)
        .where(or_(Campaign.user_id == user.id, LeadBatch.user_id == user.id))
    )
    lead = result.scalar_one_or_none()
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    # Fetch replies for this lead
    replies_result = await session.execute(
        select(LeadReply)
        .where(LeadReply.lead_id == lead_id)
        .order_by(LeadReply.received_at.desc())
    )
    replies = replies_result.scalars().all()
    
    return {
        "replies": [{
            "id": str(reply.id),
            "sender": reply.sender,
            "subject": reply.subject,
            "body": reply.body,
            "received_at": reply.received_at.isoformat() if reply.received_at else None
        } for reply in replies]
    }

logger = logging.getLogger("lead_actions") 

@router.get("/batches")
async def list_batches(session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    from models import LeadBatch
    result = await session.execute(select(LeadBatch).where(LeadBatch.user_id == user.id))
    batches = result.scalars().all()
    return {"batches": [{"id": b.id, "name": b.name, "created_at": b.created_at.isoformat()} for b in batches]}

@router.get("/batches-stats")
async def list_batches_stats(session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    # Return batches with lead counts for this user
    from models import LeadBatch
    batch_rows = await session.execute(select(LeadBatch.id, LeadBatch.name, LeadBatch.created_at).where(LeadBatch.user_id == user.id))
    batches = batch_rows.all()
    if not batches:
        return {"batches": []}
    # Sort by created_at desc
    batches_sorted = sorted(batches, key=lambda r: r[2], reverse=True)
    batch_ids = [b[0] for b in batches_sorted]
    counts_rows = await session.execute(
        select(Lead.batch_id, func.count(Lead.id))
        .where(Lead.batch_id.in_(batch_ids))
        .group_by(Lead.batch_id)
    )
    counts_map = {bid: cnt for bid, cnt in counts_rows.all()}
    return {
        "batches": [
            {"id": bid, "name": name, "count": int(counts_map.get(bid, 0)), "created_at": created_at.isoformat()}
            for bid, name, created_at in batches_sorted
        ]
    }
@router.delete("/batches/{batch_id}")
async def delete_batch(
    batch_id: uuid.UUID, 
    session: AsyncSession = Depends(get_session), 
    user: User = Depends(get_current_user)
):
    # Verify batch exists and belongs to user
    batch_result = await session.execute(
        select(LeadBatch).where(LeadBatch.id == batch_id, LeadBatch.user_id == user.id)
    )
    batch = batch_result.scalar_one_or_none()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    # Get all leads in this batch for history logging
    leads_result = await session.execute(
        select(Lead).where(Lead.batch_id == batch_id)
    )
    leads = leads_result.scalars().all()
    
    # Log deletion in lead history for each lead before deleting
    for lead in leads:
        session.add(LeadHistory(
            lead_id=lead.id,
            user_id=user.id,
            action="batch_deleted",
            field="batch_id",
            old_value=str(batch_id),
            new_value=None
        ))
    
    try:
        # Delete all leads in the batch first (due to foreign key constraints)
        for lead in leads:
            await session.delete(lead)
        
        # Then delete the batch
        await session.delete(batch)
        await session.commit()
        
        logger.info(f"User {user.id} deleted batch {batch_id} with {len(leads)} leads")
        
        # Add notification
        from models import Notification
        notification = Notification(
            user_id=user.id,
            type="batch_deleted",
            message=f"Batch '{batch.name}' and {len(leads)} associated leads have been deleted.",
            created_at=datetime.utcnow()
        )
        session.add(notification)
        await session.commit()
        
    except Exception as e:
        logger.error(f"User {user.id} failed to delete batch {batch_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete batch: {str(e)}")
    
    return {
        "message": f"Batch '{batch.name}' and {len(leads)} associated leads deleted successfully",
        "batch_name": batch.name,
        "leads_deleted": len(leads)
    }


@router.post("/scrape-apollo")
async def scrape_apollo_leads(
    background_tasks: BackgroundTasks,
    request: dict = Body(...),
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """
    Scrape leads from Apollo.io using user's API key
    """
    # Check if user has Apollo API key
    if not user.appollo_api_key:
        raise HTTPException(
            status_code=400,
            detail="Apollo.io API key not configured. Please add your Apollo.io API key in settings."
        )
    
    # Get search parameters
    person_titles = request.get("person_titles", [])
    person_locations = request.get("person_locations", [])
    q_keywords = request.get("q_keywords", "")
    organization_domains = request.get("organization_domains", "")
    per_page = min(request.get("per_page", 25), 100)
    # Allow controlling how many pages to fetch in one request (default 1)
    max_pages = int(request.get("max_pages", 1))
    max_pages = max(1, min(max_pages, 10))  # safety cap to avoid huge imports
    batch_name = request.get("batch_name", f"Apollo Import {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")
    
    # Validate at least one search parameter
    if not any([person_titles, person_locations, q_keywords, organization_domains]):
        raise HTTPException(
            status_code=400,
            detail="At least one search parameter is required (person_titles, person_locations, q_keywords, or organization_domains)"
        )
    
    # Check subscription limit
    plan = SUBSCRIPTION_PLANS.get(user.subscription_tier, SUBSCRIPTION_PLANS["free"])
    limit = plan["leads_limit"]
    
    # Count existing leads
    leads_result = await session.execute(
        select(Lead)
        .join(Campaign, isouter=True)
        .join(LeadBatch, Lead.batch_id == LeadBatch.id, isouter=True)
        .where(or_(Campaign.user_id == user.id, LeadBatch.user_id == user.id))
    )
    existing_leads = leads_result.scalars().all()
    leads_count = len(existing_leads)
    
    if leads_count >= limit:
        raise HTTPException(
            status_code=400,
            detail=f"Lead limit reached ({leads_count}/{limit}). Please upgrade your plan or delete existing leads."
        )
    
    # Prepare Apollo.io API request
    apollo_url = "https://api.apollo.io/api/v1/mixed_people/search"
    headers = {
        "Content-Type": "application/json",
        "Cache-Control": "no-cache",
        "X-Api-Key": user.appollo_api_key
    }
    
    # Payload and paging handled by the background worker; request is forwarded as-is
    
    # Enqueue background scraping task and return task id immediately
    task_id = str(uuid.uuid4())
    set_task_status(task_id, "queued", {"step": "scheduled"})
    # Schedule background task
    background_tasks.add_task(_background_scrape_apollo, task_id, user.id, request)
    return {"message": "Scrape scheduled", "task_id": task_id}