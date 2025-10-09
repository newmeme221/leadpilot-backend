from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from auth import router as auth_router 
# Static files are served separately when frontend is hosted on a static site
from routers import leads_router, campaigns_router, logs_router, ai_router, activity_router, subscriptions_router, tickets_router,users_router
from routers.notifications import router as notifications_router
from routers.inbound_email import router as inbound_email_router
from database import engine
from models import SQLModel
from utils.followup_scheduler import start_scheduler, stop_scheduler
import os 


app = FastAPI() 



# CORS: read allowed origins from FRONTEND_ORIGINS (comma-separated) for production.
# If not set, default to permissive ['*'] to preserve local dev behavior.
frontend_origins = os.getenv("FRONTEND_ORIGINS", "").strip()
if frontend_origins:
    allow_origins = [o.strip() for o in frontend_origins.split(",") if o.strip()]
else:
    allow_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables
@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    # Start follow-up scheduler (APScheduler)
    start_scheduler(interval_seconds=int(os.getenv("FOLLOWUP_POLL_INTERVAL", "60")))

@app.on_event("shutdown")
async def shutdown():
    stop_scheduler()

# Include routers with /api prefix to match frontend expectations
app.include_router(auth_router, prefix="") 
app.include_router(users_router, prefix="")
app.include_router(leads_router, prefix="")
app.include_router(campaigns_router, prefix="")
app.include_router(logs_router, prefix="")
app.include_router(ai_router, prefix="")
app.include_router(activity_router, prefix="")
app.include_router(tickets_router, prefix="")

app.include_router(subscriptions_router, prefix="")
app.include_router(notifications_router)
app.include_router(inbound_email_router, prefix="")

@app.get("/")
def root():
    return {"message": "LeadPilot Backend API"}
