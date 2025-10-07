from fastapi import APIRouter, Depends, HTTPException, Body
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from models import User 
from models import Lead
from database import get_session
from auth_utils import get_current_user

import os
import stripe
from datetime import datetime, timedelta
from typing import Dict, Any

router = APIRouter(prefix="/api/subscriptions", tags=["subscriptions"])



# Stripe initialization helper
def get_stripe_api_key():
    test_mode = os.getenv("STRIPE_TEST_MODE", "false").lower() == "true"
    if test_mode:
        key = os.getenv("STRIPE_TEST_SECRET_KEY")
        if not key:
            raise HTTPException(status_code=500, detail="STRIPE_TEST_SECRET_KEY environment variable is required for test mode")
    else:
        key = os.getenv("STRIPE_SECRET_KEY")
        if not key:
            raise HTTPException(status_code=500, detail="STRIPE_SECRET_KEY environment variable is required")
    stripe.api_key = key
    return key

# Subscription plans
SUBSCRIPTION_PLANS = {
    "free": {
        "name": "Free",
        "price": 0,
        "currency": "NGN",
        "leads_limit": 10,
        "description": "Perfect for getting started"
    },
    "pro": {
        "name": "Pro",
        "price": 5000,
        "currency": "NGN",
        "leads_limit": 100,
        "description": "For growing businesses"
    },
    "enterprise": {
        "name": "Enterprise",
        "price": 15000,
        "currency": "NGN",
        "leads_limit": 1000,
        "description": "For large-scale operations"
    }
}

@router.get("/plans")
async def get_subscription_plans():
    """Get available subscription plans"""
    return list(SUBSCRIPTION_PLANS.values())

@router.get("/current")
async def get_current_subscription(session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    """Get user's current subscription details"""
    plan = SUBSCRIPTION_PLANS.get(user.subscription_tier, SUBSCRIPTION_PLANS["free"])
    
    return {
        "tier": user.subscription_tier,
        "status": user.subscription_status,
        "expires_at": user.subscription_expires_at,
        "plan": plan,
        "is_active": user.subscription_status == "active" and (
            user.subscription_expires_at is None or user.subscription_expires_at > datetime.utcnow()
        )
    }


@router.post("/create-payment")
async def create_payment(
    plan_tier: str = Body(..., embed=True),
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """Create a Stripe payment for subscription"""
    if plan_tier not in SUBSCRIPTION_PLANS:
        raise HTTPException(status_code=400, detail="Invalid subscription plan")
    plan = SUBSCRIPTION_PLANS[plan_tier]
    get_stripe_api_key()
    try:
        # Create Stripe Checkout Session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            customer_email=user.email,
            line_items=[{
                "price_data": {
                    "currency": plan["currency"].lower(),
                    "product_data": {
                        "name": f"{plan['name']} Subscription"
                    },
                    "unit_amount": plan["price"] * 100,
                },
                "quantity": 1,
            }],
            mode="payment",
            success_url=os.getenv("FRONTEND_URL", "http://localhost:3000") + "/pricing?success=true",
            cancel_url=os.getenv("FRONTEND_URL", "http://localhost:3000") + "/pricing?cancel=true",
            notification_metadata={
                "user_id": user.id,
                "plan_tier": plan_tier,
                "type": "subscription"
            }
        )
        return {"checkout_url": checkout_session.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create Stripe payment: {str(e)}")


# Stripe webhook endpoint
from fastapi import Request
import hmac
import hashlib

@router.post("/webhook")
async def handle_stripe_webhook(request: Request, session: AsyncSession = Depends(get_session)):
    get_stripe_api_key()
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    event = None
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Webhook error: {str(e)}")

    if event["type"] == "checkout.session.completed":
        session_obj = event["data"]["object"]
        notification_metadata = session_obj.get("notification_metadata", {})
        if notification_metadata.get("type") == "subscription":
            user_id = notification_metadata.get("user_id")
            plan_tier = notification_metadata.get("plan_tier")
            result = await session.execute(select(User).where(User.id == int(user_id)))
            user = result.scalar_one_or_none()
            if user:
                user.subscription_tier = plan_tier
                user.subscription_status = "active"
                user.subscription_expires_at = datetime.utcnow() + timedelta(days=30)
                session.add(user)
                try:
                    await session.commit()
                except Exception as db_exc:
                    # Optionally log db_exc here
                    raise HTTPException(status_code=500, detail=f"Database error during webhook: {str(db_exc)}")
    return {"status": "success"}

@router.post("/cancel")
async def cancel_subscription(session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    """Cancel user's subscription"""
    user.subscription_status = "cancelled"
    session.add(user)
    try:
        await session.commit()
    except Exception as db_exc:
        # Optionally log db_exc here
        raise HTTPException(status_code=500, detail=f"Database error during cancellation: {str(db_exc)}")
    return {"message": "Subscription cancelled successfully"}

@router.get("/usage")
async def get_usage_stats(session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    """Get user's current usage statistics"""
    from models import Lead, Campaign, Notification
    # Count leads linked to user's campaigns
    campaign_leads_result = await session.execute(
        select(Lead)
        .join(Campaign)
        .where(Campaign.user_id == user.id)
    )
    campaign_leads = campaign_leads_result.scalars().all()

    # Count leads in batches owned by the user
    from models import LeadBatch
    batch_result = await session.execute(
        select(Lead)
        .join(LeadBatch)
        .where(LeadBatch.user_id == user.id)
    )
    batch_leads = batch_result.scalars().all()

    # Combine and deduplicate leads
    all_leads = {lead.id for lead in campaign_leads} | {lead.id for lead in batch_leads}
    current_usage = len(all_leads)
    plan = SUBSCRIPTION_PLANS.get(user.subscription_tier, SUBSCRIPTION_PLANS["free"])
    limit = plan["leads_limit"]
    warning_threshold = limit * 0.8

    # Update user fields
    user.current_usage = current_usage
    user.remaining_limit = max(0, limit - current_usage)
    session.add(user)
    await session.commit()

    # Notification trigger for usage limit approaching
    if current_usage >= warning_threshold and current_usage < limit:
        # Throttle: Only send a notification if none sent in the last 24 hours
        from sqlalchemy import and_
        twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
        notification_exists = await session.execute(
            select(Notification).where(
                Notification.user_id == user.id,
                Notification.type == "usage_limit",
                Notification.created_at >= twenty_four_hours_ago
            )
        )
        exists = notification_exists.scalar_one_or_none()
        if not exists:
            notification = Notification(
                user_id=user.id,
                type="usage_limit",
                message=f"You are approaching your usage limit ({current_usage}/{limit} leads).",
                created_at=datetime.utcnow()
            )
            session.add(notification)
            await session.commit()
    return {
        "current_usage": current_usage,
        "limit": limit,
        "remaining": max(0, limit - current_usage),
        "tier": user.subscription_tier
    }
