import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


def new_id() -> str:
    return str(uuid.uuid4())


class SupportCase(Base):
    __tablename__ = "support_cases"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    case_type: Mapped[str] = mapped_column(String(16), index=True)  # chat or audio
    source_filename: Mapped[str] = mapped_column(String(255), unique=True)
    agent_id: Mapped[str] = mapped_column(String(80))
    issue_hint: Mapped[str] = mapped_column(String(160))
    status: Mapped[str] = mapped_column(String(24), default="ready")
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    messages: Mapped[list["ChatMessage"]] = relationship(cascade="all, delete-orphan")
    audio: Mapped["AudioAsset | None"] = relationship(cascade="all, delete-orphan", uselist=False)


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("support_cases.id"), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    speaker: Mapped[str] = mapped_column(String(30))
    text: Mapped[str] = mapped_column(Text)


class AudioAsset(Base):
    __tablename__ = "audio_assets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("support_cases.id"), unique=True, index=True)
    file_path: Mapped[str] = mapped_column(String(500))
    duration_seconds: Mapped[float] = mapped_column(Float)
    language: Mapped[str] = mapped_column(String(20), default="en")
    transcription: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    batch_id: Mapped[str] = mapped_column(String(36), index=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("support_cases.id"), index=True)
    status: Mapped[str] = mapped_column(String(24), default="queued")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Assessment(Base):
    __tablename__ = "assessments"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    case_id: Mapped[str] = mapped_column(ForeignKey("support_cases.id"), unique=True, index=True)
    customer_issue: Mapped[str] = mapped_column(Text)
    customer_sentiment: Mapped[str] = mapped_column(String(80))
    sentiment_score: Mapped[float] = mapped_column(Float)
    agent_score: Mapped[int] = mapped_column(Integer)
    escalation_risk: Mapped[str] = mapped_column(String(24), index=True)
    policy_status: Mapped[str] = mapped_column(String(40), index=True)
    evidence_quote: Mapped[str] = mapped_column(Text)
    policy_source: Mapped[str] = mapped_column(Text)
    coaching_suggestion: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AnalysisEvent(Base):
    __tablename__ = "analysis_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[str] = mapped_column(String(36), index=True)
    job_id: Mapped[str] = mapped_column(String(36), index=True)
    level: Mapped[str] = mapped_column(String(16))
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
