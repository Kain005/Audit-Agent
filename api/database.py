"""Database setup and ORM models for audit API."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Generator

from sqlalchemy import DateTime, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    """Base declarative class for ORM models."""


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "audit.db"
DB_URL = f"sqlite:///{DB_PATH.resolve().as_posix()}"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class AuditJob(Base):
    """Stores upload, processing state, and final report payload for one audit run."""

    __tablename__ = "audit_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    file_paths: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    report_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    def get_file_paths(self) -> list[str]:
        """Return parsed file path list from serialized JSON field."""
        try:
            parsed = json.loads(self.file_paths)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:
            return []
        return []


def create_tables() -> None:
    """Create all configured ORM tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Yield a request-scoped SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
