
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_session
from models import User
import os


SECRET_KEY = os.environ.get("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
security = HTTPBearer()

def create_access_token(data: dict, expires_delta: int = 3600) -> str:
    """
    Create a JWT access token for the given data.
    Args:
        data (dict): Data to encode in the token (e.g., {"sub": user.email})
        expires_delta (int): Expiry time in seconds (default: 1 hour)
    Returns:
        str: Encoded JWT token
    """
    from datetime import datetime, timedelta
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(seconds=expires_delta)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: AsyncSession = Depends(get_session)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    result = await session.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if user is None:
        raise credentials_exception
    
    return user

async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: AsyncSession = Depends(get_session)
) -> User | None:
    try:
        return await get_current_user(credentials, session)
    except HTTPException:
        return None
