import uuid
from datetime import datetime
from typing import Optional, List
import sqlalchemy.dialects.postgresql as pg
from sqlmodel import SQLModel, Field, Relationship, Column


# -------------------------------
# Follow-up Message (LinkedIn)
# -------------------------------
class FollowUpMessage(SQLModel, table=True):
    __tablename__ = "followupmessage"

    id: uuid.UUID = Field(
        sa_column=Column(pg.UUID, primary_key=True, nullable=False, default=uuid.uuid4)
    )
    campaign_id: Optional[uuid.UUID] = Field(
        foreign_key="campaign.id", default=None, nullable=True
    )
    body: str
    delay_days: int
    scheduled_at: Optional[datetime] = None
    status: str = Field(default="pending")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    campaign: Optional["Campaign"] = Relationship(back_populates="followupmessage")


# -------------------------------
# User
# -------------------------------
class User(SQLModel, table=True):
    __tablename__ = "user"

    id: uuid.UUID = Field(
        sa_column=Column(pg.UUID, primary_key=True, nullable=False, default=uuid.uuid4)
    )
    email: str = Field(index=True, unique=True)
    password_hash: str
    role: str = Field(default="user")
    subscription_tier: str = Field(default="free")
    appollo_api_key: Optional[str] = None
    paystack_customer_id: Optional[str] = None
    subscription_status: str = Field(default="active")
    subscription_expires_at: Optional[datetime] = None
    current_usage: int = Field(default=0)
    remaining_limit: int = Field(default=0)
    status: str = Field(default="active")
    last_login: Optional[datetime] = None
    
    # MailerSend Configuration Fields
    mailersend_api_key: Optional[str] = Field(default=None, index=False)
    sender_email: Optional[str] = Field(default=None)
    sender_name: Optional[str] = Field(default=None)
    reply_to_email: Optional[str] = Field(default=None)
    
    # AI Configuration (Optional)
    openai_api_key: Optional[str] = Field(default=None, index=False)

    campaign: List["Campaign"] = Relationship(back_populates="user")
    tickets: List["Ticket"] = Relationship(
        back_populates="user", sa_relationship_kwargs={"foreign_keys": "[Ticket.user_id]"}
    )
    assigned_tickets: List["Ticket"] = Relationship(
        back_populates="assignee", sa_relationship_kwargs={"foreign_keys": "[Ticket.assigned_to]"}
    )
    notification: List["Notification"] = Relationship(back_populates="user")
    leadbatch: List["LeadBatch"] = Relationship(back_populates="user")


# -------------------------------
# Campaign
# -------------------------------
class Campaign(SQLModel, table=True):
    __tablename__ = "campaign"

    id: uuid.UUID = Field(
        sa_column=Column(pg.UUID, primary_key=True, nullable=False, default=uuid.uuid4)
    )
    user_id: Optional[uuid.UUID] = Field(
        foreign_key="user.id", default=None, nullable=True
    )
    name: str
    description: Optional[str] = None
    status: str = Field(default="draft")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    user: Optional[User] = Relationship(back_populates="campaign")
    lead: List["Lead"] = Relationship(back_populates="campaign")
    followupmessage: List["FollowUpMessage"] = Relationship(back_populates="campaign")


# -------------------------------
# Lead
# -------------------------------
class Lead(SQLModel, table=True):
    __tablename__ = "lead"

    id: uuid.UUID = Field(
        sa_column=Column(pg.UUID, primary_key=True, nullable=False, default=uuid.uuid4)
    )
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    job_title: Optional[str] = None
    company: Optional[str] = None
    profile_url: Optional[str] = None
    status: str
    message_text: Optional[str] = None
    email: Optional[str] = Field(default=None, index=True)
    email_confidence: Optional[int] = Field(default=None)
    sent_date: Optional[datetime] = None
    campaign_id: Optional[uuid.UUID] = Field(
        foreign_key="campaign.id", default=None, nullable=True
    )
    batch_id: Optional[uuid.UUID] = Field(
        foreign_key="leadbatch.id", default=None, nullable=True
    ) 
    email_verification_status: Optional[str] = Field(default=None)
    email_verification_score: Optional[int] = Field(default=None)
    campaign: Optional["Campaign"] = Relationship(back_populates="lead")
    leadbatch: Optional["LeadBatch"] = Relationship(back_populates="lead")
    outreach_log: List["OutreachLog"] = Relationship(back_populates="lead")
    leadhistory: List["LeadHistory"] = Relationship(
        back_populates="lead", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    leadreply: Optional["LeadReply"] = Relationship(back_populates="lead")

# -------------------------------
# Outreach Log
# -------------------------------
class OutreachLog(SQLModel, table=True):
    __tablename__ = "outreach_log"

    id: uuid.UUID = Field(
        sa_column=Column(pg.UUID, primary_key=True, nullable=False, default=uuid.uuid4)
    )
    lead_id: Optional[uuid.UUID] = Field(
        foreign_key="lead.id", default=None, nullable=True
    )
    status: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    lead: Optional[Lead] = Relationship(back_populates="outreach_log")


# -------------------------------
# Email Campaign
# -------------------------------
class EmailCampaign(SQLModel, table=True):
    __tablename__ = "email_campaign"

    id: uuid.UUID = Field(
        sa_column=Column(pg.UUID, primary_key=True, nullable=False, default=uuid.uuid4)
    )
    user_id: Optional[uuid.UUID] = Field(
        foreign_key="user.id", default=None, nullable=True
    )
    name: str
    subject: str
    body: str
    status: str = Field(default="draft")
    scheduled_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    user: Optional[User] = Relationship()
    followupemail: List["FollowUpEmail"] = Relationship(back_populates="email_campaign")
    email_log: List["EmailLog"] = Relationship(back_populates="email_campaign")


# -------------------------------
# Email Log
# -------------------------------
class EmailLog(SQLModel, table=True):
    __tablename__ = "email_log"

    id: uuid.UUID = Field(
        sa_column=Column(pg.UUID, primary_key=True, nullable=False, default=uuid.uuid4)
    )
    campaign_id: Optional[uuid.UUID] = Field(
        foreign_key="email_campaign.id", default=None, nullable=True
    )
    lead_id: Optional[uuid.UUID] = Field(
        foreign_key="lead.id", default=None, nullable=True
    )
    to_email: str
    status: str  # sent, delivered, opened, clicked, bounced, failed, complained, unsubscribed
    sent_at: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    clicked_at: Optional[datetime] = None  # NEW - Track when links are clicked
    error: Optional[str] = None
    external_message_id: Optional[str] = None  # NEW - MailerSend message ID for tracking

    email_campaign: Optional[EmailCampaign] = Relationship(back_populates="email_log")
    lead: Optional[Lead] = Relationship()


# -------------------------------
# Follow-up Email
# -------------------------------
class FollowUpEmail(SQLModel, table=True):
    __tablename__ = "followupemail"

    id: uuid.UUID = Field(
        sa_column=Column(pg.UUID, primary_key=True, nullable=False, default=uuid.uuid4)
    )
    email_campaign_id: Optional[uuid.UUID] = Field(
        foreign_key="email_campaign.id", default=None, nullable=True
    )
    subject: str
    body: str
    delay_days: int
    scheduled_at: Optional[datetime] = None
    status: str = Field(default="pending")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    # Per-followup sender metadata (persisted so follow-ups can be sent after restarts)
    sender_email: Optional[str] = Field(default=None)
    sender_name: Optional[str] = Field(default=None)
    reply_to_email: Optional[str] = Field(default=None)

    email_campaign: Optional[EmailCampaign] = Relationship(back_populates="followupemail")


# -------------------------------
# Ticket System
# -------------------------------
class TicketComment(SQLModel, table=True):
    __tablename__ = "ticketcomment"

    id: uuid.UUID = Field(
        sa_column=Column(pg.UUID, primary_key=True, nullable=False, default=uuid.uuid4)
    )
    ticket_id: Optional[uuid.UUID] = Field(
        foreign_key="ticket.id", default=None, nullable=True
    )
    user_id: Optional[uuid.UUID] = Field(
        foreign_key="user.id", default=None, nullable=True
    )
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    ticket: Optional["Ticket"] = Relationship(back_populates="ticketcomment")


class Ticket(SQLModel, table=True):
    __tablename__ = "ticket"

    id: uuid.UUID = Field(
        sa_column=Column(pg.UUID, primary_key=True, nullable=False, default=uuid.uuid4)
    )
    subject: str
    description: str
    status: str = Field(default="open")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[uuid.UUID] = Field(
        foreign_key="user.id", default=None, nullable=True
    )
    assigned_to: Optional[uuid.UUID] = Field(
        foreign_key="user.id", default=None, nullable=True
    )

    ticketcomment: List[TicketComment] = Relationship(back_populates="ticket")
    user: Optional[User] = Relationship(
        back_populates="tickets", sa_relationship_kwargs={"foreign_keys": "[Ticket.user_id]"}
    )
    assignee: Optional[User] = Relationship(
        back_populates="assigned_tickets", sa_relationship_kwargs={"foreign_keys": "[Ticket.assigned_to]"}
    )


# -------------------------------
# Notification
# -------------------------------
class Notification(SQLModel, table=True):
    __tablename__ = "notification"

    id: uuid.UUID = Field(
        sa_column=Column(pg.UUID, primary_key=True, nullable=False, default=uuid.uuid4)
    )
    user_id: Optional[uuid.UUID] = Field(
        foreign_key="user.id", default=None, nullable=True
    )
    type: str
    message: str
    read: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    priority: str = Field(default="info")
    expires_at: Optional[datetime] = None
    notification_metadata: Optional[str] = None
    delivery_status: str = Field(default="delivered")

    user: Optional[User] = Relationship(back_populates="notification")


# -------------------------------
# Lead History
# -------------------------------
class LeadHistory(SQLModel, table=True):
    __tablename__ = "leadhistory"

    id: uuid.UUID = Field(
        sa_column=Column(pg.UUID, primary_key=True, nullable=False, default=uuid.uuid4)
    )
    lead_id: Optional[uuid.UUID] = Field(
        foreign_key="lead.id", default=None, nullable=True
    )
    user_id: Optional[uuid.UUID] = Field(
        foreign_key="user.id", default=None, nullable=True
    )
    action: str
    field: Optional[str] = None
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    lead: Optional[Lead] = Relationship(back_populates="leadhistory")


# -------------------------------
# Lead Batch
# -------------------------------
class LeadBatch(SQLModel, table=True):
    __tablename__ = "leadbatch"

    id: uuid.UUID = Field(
        sa_column=Column(pg.UUID, primary_key=True, nullable=False, default=uuid.uuid4)
    )
    name: str
    user_id: Optional[uuid.UUID] = Field(
        foreign_key="user.id", default=None, nullable=True
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    user: Optional[User] = Relationship(back_populates="leadbatch")
    lead: List[Lead] = Relationship(back_populates="leadbatch")


# -------------------------------
# Lead Reply (Email Responses)
# -------------------------------
class LeadReply(SQLModel, table=True):
    __tablename__ = "leadreply"

    id: uuid.UUID = Field(
        sa_column=Column(pg.UUID, primary_key=True, nullable=False, default=uuid.uuid4)
    )
    lead_id: Optional[uuid.UUID] = Field(
        foreign_key="lead.id", default=None, nullable=True
    )
    sender: str
    subject: str
    body: str
    received_at: datetime = Field(default_factory=datetime.utcnow)

    lead: Optional[Lead] = Relationship(back_populates="leadreply")