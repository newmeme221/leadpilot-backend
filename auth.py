from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
from models import User
from database import get_session
from schemas import UserRegister, UserLogin, Token
from auth_utils import get_current_user
import os

SECRET_KEY = os.environ.get("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
router = APIRouter(tags=["auth"])

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@router.post("/register", response_model=Token)
async def register(user: UserRegister, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(User).where(User.email == user.email))
    db_user = result.scalar_one_or_none()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_pw = get_password_hash(user.password)
    subscription_tier = user.subscription_tier if hasattr(user, 'subscription_tier') and user.subscription_tier else "free"
    new_user = User(email=user.email, password_hash=hashed_pw, subscription_tier=subscription_tier)
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)
    access_token = create_access_token({"sub": new_user.email, "role": new_user.role})
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/login", response_model=Token)
async def login(user: UserLogin, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(User).where(User.email == user.email))
    db_user = result.scalar_one_or_none()
    if not db_user or not verify_password(user.password, db_user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    # Update last_login on successful login
    db_user.last_login = datetime.utcnow()
    session.add(db_user)
    await session.commit()
    access_token = create_access_token({"sub": db_user.email, "role": db_user.role})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/user")
async def get_current_user_endpoint(user: User = Depends(get_current_user)):
    return {
        "id": user.id,
        "email": user.email,
        "subscription_tier": user.subscription_tier,
        "role": user.role
    }

@router.post("/logout")
async def logout():
    return {"message": "Logged out successfully"}

@router.post("/update-status")
async def update_user_status(new_status: str, session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    user.status = new_status
    await session.commit()
    return {"message": f"User status updated to {new_status}"}

@router.post("/update-last-login")
async def update_last_login(session: AsyncSession = Depends(get_session), user: User = Depends(get_current_user)):
    user.last_login = datetime.utcnow()
    await session.commit()
    return {"message": "Last login time updated"}
