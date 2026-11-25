from dotenv import load_dotenv

load_dotenv()

from datetime import datetime

from sqlalchemy import select

from backend.celery_app import celery_app
from backend.database import SessionLocal
from backend.events import record_event
from backend.models import AnalysisJob, Assessment, AudioAsset, ChatMessage, SupportCase
from backend.services.rag_audit import assess_case, transcribe_audio


def save_assessment(session, case_id: str, result: dict) -> None:
    existing = session.scalar(select(Assessment).where(Assessment.case_id == case_id))
    if existing:
        session.delete(existing)
        session.flush()
    session.add(Assessment(case_id=case_id, **result))
    session.commit()


@celery_app.task(name="support_lens.process_case")
def process_case(job_id: str) -> None:
    with SessionLocal() as session:
        job = session.get(AnalysisJob, job_id)
        if not job:
            return
        case = session.get(SupportCase, job.case_id)
        job.status, case.status = "processing", "processing"
        session.commit()
        record_event(session, job.batch_id, job.id, "processing", f"Processing {case.case_type}: {case.source_filename}")
        try:
            if case.case_type == "audio":
                audio = session.scalar(select(AudioAsset).where(AudioAsset.case_id == case.id))
                record_event(session, job.batch_id, job.id, "processing", "Sending audio to OpenAI for transcription")
                transcript = transcribe_audio(audio.file_path)
                audio.transcription = transcript
                session.commit()
                payload = {
                    "case_type": "audio",
                    "audio_metadata": {"duration_seconds": audio.duration_seconds, "language": audio.language, **audio.metadata_json},
                    "agent_id": case.agent_id,
                    "transcript": transcript,
                }
            else:
                messages = session.scalars(select(ChatMessage).where(ChatMessage.case_id == case.id).order_by(ChatMessage.sequence)).all()
                payload = {
                    "case_type": "chat",
                    "agent_id": case.agent_id,
                    "messages": [{"speaker": message.speaker, "text": message.text} for message in messages],
                }
            record_event(session, job.batch_id, job.id, "retrieval", "Calling the required RAG policy-retrieval tool")
            assessment = assess_case(case.case_type, payload)
            save_assessment(session, case.id, assessment)
            job.status, case.status, job.completed_at = "completed", "completed", datetime.utcnow()
            session.commit()
            record_event(session, job.batch_id, job.id, "completed", f"Assessment complete: {case.source_filename}")
        except Exception as error:
            job.status, case.status, job.error_message = "failed", "failed", str(error)
            session.commit()
            record_event(session, job.batch_id, job.id, "failed", f"Failed {case.source_filename}: {error}")
