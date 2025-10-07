"""
ZeroBounce Email Verification Integration
Add this to a new file: email_verification.py
"""

import httpx
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio

logger = logging.getLogger(__name__)

# Rate limiting for ZeroBounce API 

ZEROBOUNCE_RATE_LIMIT = 100  # requests per minute (adjust based on your plan)
ZEROBOUNCE_RATE_PERIOD = 60  # seconds
zerobounce_timestamps = defaultdict(lambda: deque(maxlen=ZEROBOUNCE_RATE_LIMIT))

def check_zerobounce_rate_limit(user_id: int) -> bool:
    """Check if user has exceeded ZeroBounce rate limit"""
    now = datetime.now().timestamp()
    timestamps = zerobounce_timestamps[user_id]
    
    # Remove old timestamps
    while timestamps and now - timestamps[0] > ZEROBOUNCE_RATE_PERIOD:
        timestamps.popleft()
    
    if len(timestamps) >= ZEROBOUNCE_RATE_LIMIT:
        return False
    
    timestamps.append(now)
    return True


async def verify_email_zerobounce(
    email: str, 
    api_key: str,
    ip_address: Optional[str] = None
) -> Dict:
    """
    Verify email using ZeroBounce API
    
    Args:
        email: Email address to verify
        api_key: ZeroBounce API key
        ip_address: Optional IP address for additional validation
    
    Returns:
        Dict with verification results:
        {
            "status": "valid" | "invalid" | "catch-all" | "unknown" | "spamtrap" | "abuse" | "do_not_mail",
            "sub_status": str,
            "account": str,
            "domain": str,
            "did_you_mean": str | None,
            "free_email": bool,
            "mx_found": bool,
            "mx_record": str,
            "smtp_provider": str,
            "score": int (0-10, 10 being best),
            "error": str | None
        }
    """
    if not api_key:
        return {
            "status": "unknown",
            "error": "ZeroBounce API key not configured",
            "email": email
        }
    
    url = "https://api.zerobounce.net/v2/validate"
    params = {
        "api_key":api_key,
        "email": email
    }
    
    if ip_address:
        params["ip_address"] = ip_address
    
    max_retries = 3
    backoff = 2
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Map ZeroBounce status to our simplified format
                status = data.get("status", "unknown").lower()
                
                return {
                    "status": status,
                    "sub_status": data.get("sub_status", ""),
                    "account": data.get("account", ""),
                    "domain": data.get("domain", ""),
                    "did_you_mean": data.get("did_you_mean"),
                    "free_email": data.get("free_email", False),
                    "mx_found": data.get("mx_found", False),
                    "mx_record": data.get("mx_record", ""),
                    "smtp_provider": data.get("smtp_provider", ""),
                    "score": calculate_email_score(data),
                    "error": None,
                    "email": email
                }
            
            elif response.status_code == 429:
                logger.warning(f"ZeroBounce rate limit hit, attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(backoff ** attempt)
                    continue
                return {
                    "status": "unknown",
                    "error": "Rate limit exceeded",
                    "email": email
                }
            
            else:
                error_msg = response.text
                logger.error(f"ZeroBounce API error {response.status_code}: {error_msg}")
                return {
                    "status": "unknown",
                    "error": f"API error: {response.status_code}",
                    "email": email
                }
        
        except httpx.RequestError as e:
            logger.error(f"ZeroBounce request error: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(backoff ** attempt)
                continue
            return {
                "status": "unknown",
                "error": f"Request error: {str(e)}",
                "email": email
            }
        
        except Exception as e:
            logger.error(f"Unexpected ZeroBounce error: {e}")
            return {
                "status": "unknown",
                "error": f"Unexpected error: {str(e)}",
                "email": email
            }
    
    return {
        "status": "unknown",
        "error": "Max retries exceeded",
        "email": email
    }


def calculate_email_score(zerobounce_data: Dict) -> int:
    """
    Calculate email quality score (0-10) based on ZeroBounce data
    10 = excellent, 0 = very poor
    """
    score = 10
    
    status = zerobounce_data.get("status", "").lower()
    
    # Status penalties
    if status == "invalid":
        score = 0
    elif status == "spamtrap":
        score = 0
    elif status == "abuse":
        score = 0
    elif status == "do_not_mail":
        score = 0
    elif status == "unknown":
        score = 3
    elif status == "catch-all":
        score = 6
    elif status == "valid":
        score = 10
    
    # Additional factors for valid emails
    if status == "valid":
        if not zerobounce_data.get("mx_found", False):
            score -= 2
        
        if zerobounce_data.get("free_email", False):
            score -= 1  # Free emails are slightly lower quality for B2B
        
        sub_status = zerobounce_data.get("sub_status", "").lower()
        if "role" in sub_status:  # Role-based emails (info@, sales@)
            score -= 1
        if "disposable" in sub_status:
            score = 0
    
    return max(0, min(10, score))


def should_accept_email(verification_result: Dict, strict_mode: bool = False) -> tuple[bool, str]:
    """
    Determine if an email should be accepted based on verification results
    
    Args:
        verification_result: Result from verify_email_zerobounce()
        strict_mode: If True, only accept "valid" emails. If False, accept "catch-all" too.
    
    Returns:
        Tuple of (should_accept: bool, reason: str)
    """
    status = verification_result.get("status", "unknown")
    score = verification_result.get("score", 0)
    
    # Always reject these
    if status in ["invalid", "spamtrap", "abuse", "do_not_mail"]:
        return False, f"Email rejected: {status}"
    
    # Handle catch-all domains
    if status == "catch-all":
        if strict_mode:
            return False, "Email rejected: catch-all domain (strict mode)"
        else:
            return True, "Email accepted: catch-all domain (lower confidence)"
    
    # Handle unknown status
    if status == "unknown":
        if strict_mode:
            return False, "Email rejected: verification failed (strict mode)"
        else:
            # In lenient mode, accept unknown with warning
            return True, "Email accepted: verification incomplete (use caution)"
    
    # Valid emails
    if status == "valid":
        if score >= 7:
            return True, "Email accepted: high quality"
        elif score >= 5:
            return True, "Email accepted: moderate quality"
        else:
            return False, "Email rejected: low quality score"
    
    return False, f"Email rejected: unexpected status '{status}'"


async def bulk_verify_emails(
    emails: list[str],
    api_key: str,
    batch_size: int = 50,
    delay_between_batches: float = 1.0
) -> Dict[str, Dict]:
    """
    Verify multiple emails in batches
    
    Returns:
        Dict mapping email -> verification_result
    """
    results = {}
    
    for i in range(0, len(emails), batch_size):
        batch = emails[i:i + batch_size]
        
        # Process batch concurrently
        tasks = [verify_email_zerobounce(email, api_key) for email in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for email, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                results[email] = {
                    "status": "unknown",
                    "error": str(result),
                    "email": email
                }
            else:
                results[email] = result
        
        # Delay between batches to avoid rate limits
        if i + batch_size < len(emails):
            await asyncio.sleep(delay_between_batches)
    
    return results


# Example usage in your routes:
"""
from email_verification import (
    verify_email_zerobounce, 
    should_accept_email,
    check_zerobounce_rate_limit
)

# In leads.py upload route:
if email:
    # Check rate limit
    if not check_zerobounce_rate_limit(user.id):
        row_errors.append({
            "row": idx, 
            "reason": "ZeroBounce rate limit exceeded", 
            "row_data": row
        })
        continue
    
    # Verify email
    verification = await verify_email_zerobounce(
        email=email,
        api_key=user.zerobounce_api_key or os.getenv("ZEROBOUNCE_API_KEY")
    )
    
    # Check if acceptable
    accepted, reason = should_accept_email(verification, strict_mode=False)
    if not accepted:
        row_errors.append({
            "row": idx,
            "reason": reason,
            "row_data": row,
            "verification": verification
        })
        continue
    
    # Store verification score with lead
    lead.email_verification_score = verification.get("score")
    lead.email_verification_status = verification.get("status")
"""