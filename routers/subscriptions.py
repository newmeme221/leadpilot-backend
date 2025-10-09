from fastapi import APIRouter, Depends, HTTPException, Body, Request
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from models import User, Lead, Campaign, LeadBatch, Notification
from database import get_session
from auth_utils import get_current_user

import os
import httpx
from datetime import datetime, timedelta
from typing import Dict, Any
import hashlib
import hmac

router = APIRouter(prefix="/api/subscriptions", tags=["subscriptions"])


# Flutterwave configuration helper
def get_flutterwave_config():
    """Get Flutterwave API configuration"""
    test_mode = os.getenv("FLUTTERWAVE_TEST_MODE", "false").lower() == "true"
    
    if test_mode:
        secret_key = os.getenv("FLUTTERWAVE_TEST_SECRET_KEY")
        public_key = os.getenv("FLUTTERWAVE_TEST_PUBLIC_KEY")
        if not secret_key or not public_key:
            raise HTTPException(
                status_code=500, 
                detail="FLUTTERWAVE_TEST_SECRET_KEY and FLUTTERWAVE_TEST_PUBLIC_KEY are required for test mode"
            )
    else:
        secret_key = os.getenv("FLUTTERWAVE_SECRET_KEY")
        public_key = os.getenv("FLUTTERWAVE_PUBLIC_KEY")
        if not secret_key or not public_key:
            raise HTTPException(
                status_code=500,
                detail="FLUTTERWAVE_SECRET_KEY and FLUTTERWAVE_PUBLIC_KEY are required"
            )
    
    return {
        "secret_key": secret_key,
        "public_key": public_key,
        "base_url": "https://api.flutterwave.com/v3"
    }


# Subscription plans
SUBSCRIPTION_PLANS = {
    "free": {
        "name": "Free",
        "price": 0,
        "currency": "USD",
        "leads_limit": 200,
        "description": "Perfect for getting started"
    },
    "pro": {
        "name": "Pro",
        "price": 39.99,
        "currency": "USD",
        "leads_limit": 5000,
        "description": "Ideal for growing team"
    },
    "enterprise": {
        "name": "Enterprise",
        "price": 129.99,
        "currency": "USD",
        "leads_limit": 30000,
        "description": "For large-scale teams that need power and flexibility"
    }
}


@router.get("/plans")
async def get_subscription_plans():
    """Get available subscription plans"""
    return list(SUBSCRIPTION_PLANS.values())


@router.get("/current")
async def get_current_subscription(
    session: AsyncSession = Depends(get_session), 
    user: User = Depends(get_current_user)
):
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
    """Create a Flutterwave payment for subscription"""
    if plan_tier not in SUBSCRIPTION_PLANS:
        raise HTTPException(status_code=400, detail="Invalid subscription plan")
    
    plan = SUBSCRIPTION_PLANS[plan_tier]
    
    if plan["price"] == 0:
        raise HTTPException(status_code=400, detail="Cannot create payment for free plan")
    
    config = get_flutterwave_config()
    
    try:
        # Generate unique transaction reference
        tx_ref = f"SUB-{user.id}-{int(datetime.utcnow().timestamp())}"
        
        # Prepare payment payload
        payload = {
            "tx_ref": tx_ref,
            "amount": plan["price"],
            "currency": plan["currency"],
            "redirect_url": os.getenv("FRONTEND_URL", "http://localhost:3000") + "/pricing?redirect=true",
            "customer": {
                "email": user.email,
                "name": user.email.split("@")[0]  # Use email prefix as name
            },
            "customizations": {
                "title": f"{plan['name']} Subscription",
                "description": plan["description"],
                "logo": os.getenv("LOGO_URL", "")
            },
            "meta": {
                "user_id": str(user.id),
                "plan_tier": plan_tier,
                "type": "subscription"
            }
        }
        
        # Initialize payment with Flutterwave
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{config['base_url']}/payments",
                json=payload,
                headers={
                    "Authorization": f"Bearer {config['secret_key']}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Flutterwave API error: {response.text}"
                )
            
            data = response.json()
            
            if data.get("status") != "success":
                raise HTTPException(
                    status_code=500,
                    detail=f"Payment initialization failed: {data.get('message', 'Unknown error')}"
                )
            
            return {
                "checkout_url": data["data"]["link"],
                "tx_ref": tx_ref
            }
    
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create Flutterwave payment: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )


@router.get("/verify-payment/{tx_ref}")
async def verify_payment(
    tx_ref: str,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """Verify payment status after redirect"""
    config = get_flutterwave_config()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{config['base_url']}/transactions/verify_by_reference",
                params={"tx_ref": tx_ref},
                headers={
                    "Authorization": f"Bearer {config['secret_key']}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to verify payment"
                )
            
            data = response.json()
            
            if data.get("status") != "success":
                return {
                    "status": "failed",
                    "message": "Payment verification failed"
                }
            
            transaction = data["data"]
            
            # Check if payment was successful
            if transaction["status"] == "successful" and transaction["currency"] == "NGN":
                # Extract metadata
                meta = transaction.get("meta", {})
                plan_tier = meta.get("plan_tier")
                
                if plan_tier and plan_tier in SUBSCRIPTION_PLANS:
                    # Update user subscription
                    user.subscription_tier = plan_tier
                    user.subscription_status = "active"
                    user.subscription_expires_at = datetime.utcnow() + timedelta(days=30)
                    
                    session.add(user)
                    await session.commit()
                    
                    return {
                        "status": "success",
                        "message": "Subscription activated successfully",
                        "plan": SUBSCRIPTION_PLANS[plan_tier]
                    }
            
            return {
                "status": "pending",
                "message": "Payment is being processed"
            }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Verification error: {str(e)}"
        )


@router.post("/webhook")
async def handle_flutterwave_webhook(
    request: Request,
    session: AsyncSession = Depends(get_session)
):
    """Handle Flutterwave webhook notifications"""
    config = get_flutterwave_config()
    
    # Get webhook secret hash
    webhook_secret = os.getenv("FLUTTERWAVE_WEBHOOK_SECRET_HASH", "")
    
    # Verify webhook signature
    signature = request.headers.get("verif-hash")
    
    if not signature or signature != webhook_secret:
        raise HTTPException(
            status_code=401,
            detail="Invalid webhook signature"
        )
    
    try:
        payload = await request.json()
        
        # Handle successful payment
        if payload.get("event") == "charge.completed":
            data = payload.get("data", {})
            
            # Only process successful payments
            if data.get("status") != "successful":
                return {"status": "ignored"}
            
            # Extract metadata
            meta = data.get("meta", {})
            if meta.get("type") != "subscription":
                return {"status": "ignored"}
            
            user_id = meta.get("user_id")
            plan_tier = meta.get("plan_tier")
            
            if not user_id or not plan_tier:
                return {"status": "error", "message": "Missing metadata"}
            
            # Get user and update subscription
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if user:
                user.subscription_tier = plan_tier
                user.subscription_status = "active"
                user.subscription_expires_at = datetime.utcnow() + timedelta(days=30)
                
                session.add(user)
                
                try:
                    await session.commit()
                    
                    # Create notification for user
                    notification = Notification(
                        user_id=user.id,
                        type="subscription",
                        message=f"Your {SUBSCRIPTION_PLANS[plan_tier]['name']} subscription has been activated!",
                        priority="info",
                        created_at=datetime.utcnow()
                    )
                    session.add(notification)
                    await session.commit()
                    
                except Exception as db_exc:
                    await session.rollback()
                    raise HTTPException(
                        status_code=500,
                        detail=f"Database error: {str(db_exc)}"
                    )
        
        return {"status": "success"}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Webhook processing error: {str(e)}"
        )


@router.post("/cancel")
async def cancel_subscription(
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """Cancel user's subscription"""
    user.subscription_status = "cancelled"
    session.add(user)
    
    try:
        await session.commit()
        
        # Create notification
        notification = Notification(
            user_id=user.id,
            type="subscription",
            message="Your subscription has been cancelled. You can continue using your current plan until it expires.",
            priority="info",
            created_at=datetime.utcnow()
        )
        session.add(notification)
        await session.commit()
        
    except Exception as db_exc:
        await session.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(db_exc)}"
        )
    
    return {"message": "Subscription cancelled successfully"}


@router.get("/usage")
async def get_usage_stats(
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """Get user's current usage statistics"""
    # Count leads linked to user's campaigns
    campaign_leads_result = await session.execute(
        select(Lead)
        .join(Campaign)
        .where(Campaign.user_id == user.id)
    )
    campaign_leads = campaign_leads_result.scalars().all()

    # Count leads in batches owned by the user
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
                priority="warning",
                created_at=datetime.utcnow()
            )
            session.add(notification)
            await session.commit()
    
    return {
        "current_usage": current_usage,
        "limit": limit,
        "remaining": max(0, limit - current_usage),
        "tier": user.subscription_tier,
        "percentage_used": round((current_usage / limit * 100), 2) if limit > 0 else 0
    }