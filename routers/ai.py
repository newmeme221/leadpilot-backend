from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from models import Lead, User
from database import get_session
from auth_utils import get_current_user
from schemas import MessageGenRequest, MessageGenResponse
import openai
import os

router = APIRouter(prefix="/api", tags=["ai"])

openai.api_key = os.environ.get("OPENAI_API_KEY", "sk-xxx")

PROMPT_TEMPLATE = """
Write a personalized LinkedIn outreach message for the following lead:
First Name: {first_name}
Last Name: {last_name}
Job Title: {job_title}
Company: {company}
Profile URL: {profile_url}
"""



@router.post("/generate-message", response_model=MessageGenResponse)
async def generate_message(req: dict, session: AsyncSession = Depends(get_session)):
    lead_info = req.get("leadInfo", "").strip()
    tone = req.get("tone", "professional")
    goal = req.get("goal", "connect")  # e.g., connect, pitch, invite
    length = req.get("length", 300)  # preferred max length in characters
    cta = req.get("cta", "")  # optional call-to-action
    personalization = req.get("personalization", "")  # shared interests, tags
    msg_type = req.get("type", "linkedin")  # 'linkedin' or 'email'

    if not lead_info:
        raise HTTPException(status_code=400, detail="Lead information is required")

    if not openai.api_key or openai.api_key == "sk-xxx":
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    try:
        if msg_type == "email":
            persona = "You are an expert email marketer with a proven track record in crafting high-converting cold email campaigns for B2B clients."
            prompt = (
                f"{persona} "
                f"Write a {tone} email outreach message for the following campaign: {lead_info}. "
                f"Goal: {goal}. "
                f"Personalization: {personalization}. "
                f"Include this call-to-action if provided: {cta}. "
                f"Keep it professional, personalized, and under {length} characters."
            )
        else:
            persona = "You are a seasoned LinkedIn outreach specialist, skilled in building professional relationships and starting meaningful conversations through LinkedIn messaging."
            prompt = (
                f"{persona} "
                f"Write a {tone} LinkedIn outreach message for the following campaign: {lead_info}. "
                f"Goal: {goal}. "
                f"Personalization: {personalization}. "
                f"Include this call-to-action if provided: {cta}. "
                f"Keep it professional, personalized, and under {length} characters."
            )

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=round(length * 0.7),
            temperature=0.7
        )
        message = response.choices[0].message.content.strip()
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate message: {str(e)}")
