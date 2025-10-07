from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime 
import uuid 
from pydantic  import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime
# Web scraping request/response schemas for SerpApi
class WebScrapeLeadsRequest(BaseModel):
    company: Optional[str] = None
    occupation: Optional[str] = None
    location: Optional[str] = None
    keywords: Optional[str] = None
    num_results: int = 10

class WebScrapeLeadOut(BaseModel):
    name: str
    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    profile_url: Optional[str] = None
    source: Optional[str] = None

# Follow-up message schemas for LinkedIn campaigns
class FollowUpMessageCreate(BaseModel):
    body: str
    delay_days: int

class FollowUpMessageOut(BaseModel):
    id: uuid.UUID
    body: str
    delay_days: int
    scheduled_at: Optional[datetime]
    status: str
    created_at: datetime

class CampaignCreate(BaseModel):
    name: str
    description: Optional[str] = None
    follow_ups: Optional[List[FollowUpMessageCreate]] = []


class UserRegister(BaseModel):
    email: EmailStr
    password: str
    subscription_tier: str = "free"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class LeadUpload(BaseModel):
    file: bytes  # Will use UploadFile in endpoint

class LeadOut(BaseModel):
    id: uuid.UUID
    first_name: str
    last_name: str
    job_title: str
    company: str
    profile_url: str
    status: str
    message_text: Optional[str]
    email: Optional[str]
    email_confidence: Optional[int]
    campaign_id: Optional[int]

class LeadListResponse(BaseModel):
    leads: List[LeadOut]

class CampaignStats(BaseModel):
    sent: int
    accepted: int
    replied: int
    failed: int

class OutreachLogCreate(BaseModel):
    lead_id: uuid.UUID
    status: str
    message: str

class MessageGenRequest(BaseModel):
    lead_id: uuid.UUID

class MessageGenResponse(BaseModel):
    message: str


class FollowUpEmailCreate(BaseModel):
    subject: str
    body: str
    delay_days: int

class FollowUpEmailOut(BaseModel):
    id: uuid.UUID
    subject: str
    body: str
    delay_days: int
    scheduled_at: Optional[datetime]
    status: str
    created_at: datetime

class EmailCampaignCreate(BaseModel):
    name: str
    subject: str
    body: str
    follow_ups: Optional[List[FollowUpEmailCreate]] = []

class EmailCampaignOut(BaseModel):
    id: uuid.UUID
    name: str
    subject: str
    body: str
    status: str
    scheduled_at: Optional[datetime]
    created_at: datetime

class EmailLogOut(BaseModel):
    lead_id: uuid.UUID
    lead_name: Optional[str]
    to_email: str
    status: str
    sent_at: Optional[datetime]
    opened_at: Optional[datetime]
    error: Optional[str]