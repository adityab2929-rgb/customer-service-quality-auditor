import asyncio
import json
import uuid
from collections import Counter, defaultdict

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from backend.config import settings
from backend.database import SessionLocal, init_db
from backend.models import AnalysisEvent, AnalysisJob, Assessment, AudioAsset, ChatMessage, SupportCase
from backend.seed_data import seed_data
from backend.tasks import process_case

from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Support Lens API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/admin/seed")
def seed() -> dict:
    return seed_data()


@app.post("/batches")
def create_batch() -> dict:
    batch_id = str(uuid.uuid4())
    with SessionLocal() as session:
        cases = session.scalars(select(SupportCase).order_by(SupportCase.case_type, SupportCase.source_filename)).all()
        if not cases:
            raise HTTPException(400, "No cases found. Call POST /admin/seed first.")
        jobs = []
        for case in cases:
            job = AnalysisJob(batch_id=batch_id, case_id=case.id)
            session.add(job)
            jobs.append(job)
        session.commit()
        for job in jobs:
            process_case.delay(job.id)
    return {"batch_id": batch_id, "total_jobs": len(jobs)}


@app.get("/batches/{batch_id}")
def batch_status(batch_id: str) -> dict:
    with SessionLocal() as session:
        jobs = session.scalars(select(AnalysisJob).where(AnalysisJob.batch_id == batch_id)).all()
        if not jobs:
            raise HTTPException(404, "Batch not found")
        counts = Counter(job.status for job in jobs)
        return {"batch_id": batch_id, "total": len(jobs), "completed": counts["completed"], "failed": counts["failed"], "processing": counts["processing"], "queued": counts["queued"]}


@app.get("/batches/{batch_id}/events")
def batch_events(batch_id: str, limit: int = 100) -> list[dict]:
    with SessionLocal() as session:
        events = session.scalars(select(AnalysisEvent).where(AnalysisEvent.batch_id == batch_id).order_by(AnalysisEvent.id.desc()).limit(limit)).all()
        return [{"level": event.level, "message": event.message, "created_at": event.created_at.isoformat()} for event in reversed(events)]


@app.get("/dashboard/summary")
def dashboard_summary() -> dict:
    with SessionLocal() as session:
        assessments = session.scalars(select(Assessment)).all()
        risks = Counter(item.escalation_risk for item in assessments)
        policies = Counter(item.policy_status for item in assessments)
        issues = Counter(item.customer_issue for item in assessments)
        scores: dict[str, list[int]] = defaultdict(list)
        for assessment in assessments:
            case = session.get(SupportCase, assessment.case_id)
            scores[case.agent_id].append(assessment.agent_score)
        return {
            "reviewed": len(assessments),
            "high_risk": risks["high"] + risks["critical"],
            "medium_risk": risks["medium"],
            "low_risk": risks["low"],
            "possible_violations": policies["possible_violation"] + policies["violation"],
            "risk_distribution": dict(risks),
            "policy_distribution": dict(policies),
            "sentiment_distribution": dict(Counter(item.customer_sentiment for item in assessments)),
            "agent_quality": {agent: round(sum(values) / len(values), 2) for agent, values in scores.items()},
            "common_issue": issues.most_common(1)[0][0] if issues else "—",
        }


@app.get("/cases")
def cases() -> list[dict]:
    with SessionLocal() as session:
        records = []
        for assessment in session.scalars(select(Assessment)).all():
            case = session.get(SupportCase, assessment.case_id)
            records.append({"case_id": case.id, "file": case.source_filename, "type": case.case_type, "agent": case.agent_id, "issue": assessment.customer_issue, "risk": assessment.escalation_risk, "agent_score": assessment.agent_score, "policy_status": assessment.policy_status})
        return records


@app.get("/cases/{case_id}")
def case_detail(case_id: str) -> dict:
    with SessionLocal() as session:
        case = session.get(SupportCase, case_id)
        if not case:
            raise HTTPException(404, "Case not found")
        assessment = session.scalar(select(Assessment).where(Assessment.case_id == case_id))
        messages = session.scalars(select(ChatMessage).where(ChatMessage.case_id == case_id).order_by(ChatMessage.sequence)).all()
        audio = session.scalar(select(AudioAsset).where(AudioAsset.case_id == case_id))
        return {
            "case": {"file": case.source_filename, "type": case.case_type, "agent": case.agent_id},
            "messages": [{"speaker": m.speaker, "text": m.text} for m in messages],
            "audio": {"transcription": audio.transcription, "metadata": audio.metadata_json} if audio else None,
            "assessment": {column.name: getattr(assessment, column.name) for column in Assessment.__table__.columns if column.name not in {"id", "case_id", "created_at"}} if assessment else None,
        }


@app.websocket("/ws/batches/{batch_id}")
async def batch_websocket(websocket: WebSocket, batch_id: str) -> None:
    await websocket.accept()
    client = aioredis.from_url(settings.redis_url, decode_responses=True)
    pubsub = client.pubsub()
    await pubsub.subscribe(f"support-lens:batch:{batch_id}")
    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message:
                await websocket.send_text(message["data"])
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(f"support-lens:batch:{batch_id}")
        await pubsub.close()
        await client.aclose()
