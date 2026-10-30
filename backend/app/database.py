import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.engine import URL


# Load environment variables from backend/.env
BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")


# Database configuration
DATABASE_URL = URL.create(
    drivername="postgresql+psycopg",
    username=os.getenv("DB_USER", "dr_ai_user"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    database=os.getenv("DB_NAME", "dr_ai_platform"),
)

if not DATABASE_URL.password:
    raise RuntimeError(
        "DB_PASSWORD is missing. Add it to your backend/.env file."
    )


# SQLAlchemy engine and database sessions
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


# Base class for future database models
class Base(DeclarativeBase):
    pass


def get_db():
    """Provide a database session to an API request."""
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()