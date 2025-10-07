
from dotenv import load_dotenv
load_dotenv()
from sqlmodel import SQLModel, create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from typing import AsyncGenerator
import os

DATABASE_URL = os.getenv("DATABASE_URL")
# Convert postgresql:// to postgresql+asyncpg:// for async support
if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
# Remove sslmode parameter as asyncpg handles SSL differently
if DATABASE_URL and "?sslmode=" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.split("?sslmode=")[0]
engine = create_async_engine(DATABASE_URL,
    pool_pre_ping=True,     
    pool_recycle=1800, 
    echo=True)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session
