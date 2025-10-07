# LeadPilot Backend

## Setup

1. Create and activate a virtual environment:
   ```
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Mac/Linux
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Database Setup (PostgreSQL)

1. Install PostgreSQL and create a database:
   ```sql
   CREATE DATABASE leadpilot;
   CREATE USER leadpilot_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE leadpilot TO leadpilot_user;
   ```

2. Update your `.env` file with PostgreSQL connection details:
   ```
   DATABASE_URL=postgresql+asyncpg://leadpilot_user:your_password@localhost:5432/leadpilot
   ```

## Database Migrations

1. Initialize Alembic (first time only):
   ```
   alembic init alembic
   ```
2. Generate migration:
   ```
   alembic revision --autogenerate -m "init"
   ```
3. Apply migration:
   ```
   alembic upgrade head
   ```

## Run the API

```
uvicorn main:app --reload
```

## Environment Variables
- `SECRET_KEY` (for JWT)
- `OPENAI_API_KEY` (for GPT-4o-mini)
- `DATABASE_URL` (PostgreSQL connection string)

## MailerSend integration and webhooks

This project integrates with MailerSend for outbound email delivery and inbound/event webhooks.

- Configure your MailerSend API key in the environment or per-user in the `User` model:
   - Env fallback: `MAILERSEND_API_KEY`
   - Per-user fields in the DB: `User.mailersend_api_key`, `User.sender_email`, `User.sender_name`, `User.reply_to_email`

- Webhook endpoints (FastAPI):
   - Inbound email replies: `POST /api/mailersend/inbound` — handles inbound messages and stores `LeadReply`.
   - Events (delivery, open, click, bounce): `POST /api/mailersend/events` — updates `EmailLog` entries.

- Security: set `MAILERSEND_WEBHOOK_SECRET` in your environment and MailerSend webhook settings. The app will verify HMAC-SHA256 signatures when present.

Example MailerSend event payload (simplified):
```json
{
   "type": "event",
   "data": {
      "event": "delivered",
      "message": {"id": "<message-id>"},
      "recipient": {"email": "recipient@example.com"},
      "tags": ["campaign:Name","campaign_id:<id>","lead_id:<id>"]
   }
}
```

The events endpoint will try to match an `EmailLog` using, in order:
1. `EmailLog.external_message_id` (matched to MailerSend message id)
2. `tags` containing `lead_id:<id>` — falls back to the most recent `EmailLog` for that lead
3. Recipient email and recent `EmailLog` entries

## Follow-up scheduler (APScheduler)

Follow-up emails are scheduled per-lead and persisted in `FollowUpEmail` DB records. To ensure follow-ups are sent with the correct sender, the system stores the following fields on `FollowUpEmail`:

- `sender_email`
- `sender_name`
- `reply_to_email`

Important: The application does NOT persist per-followup MailerSend API keys by design. The scheduler will use the environment `MAILERSEND_API_KEY` at send time (or you may extend the code to use per-user keys).

APScheduler is used (in-process) with an optional SQLAlchemy jobstore for persistence. To enable the jobstore set:

- `USE_SQLALCHEMY_JOBSTORE=true`
- `DATABASE_URL` (already required for app DB)

Notes:
- In-process APScheduler is fine for single-instance deployments. For multiple app instances, consider either a dedicated scheduler instance or migrating to Celery with a broker (Redis/RabbitMQ).

## Quick verification steps (smoke tests)

1. Start the backend:
```powershell
uvicorn backend.main:app --reload
```
2. Create a campaign and a lead (via API or UI) and ensure the user's `sender_email` is set or `FROM_EMAIL` is configured.
3. Trigger a campaign send for a single lead.
4. Confirm an `EmailLog` row is created with `status = sent` and `external_message_id` populated.
5. POST a MailerSend event payload to `/api/mailersend/events` (use the real `message_id`) and confirm the `EmailLog` status/timestamps update.
6. Confirm follow-up rows in `followupemail` have `sender_email`/`sender_name`/`reply_to_email` filled and that APScheduler sends follow-ups at the scheduled time.

## Migrations

We added sender fields to `FollowUpEmail` in models; if you use Alembic please add a revision and run migrations:

```powershell
alembic revision --autogenerate -m "add followup sender fields"
alembic upgrade head
```

If you'd like, I can generate a migration file for you.
