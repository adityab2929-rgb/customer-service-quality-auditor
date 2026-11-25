"""Seed existing chats and 18 spoken synthetic-audio cases into PostgreSQL."""

import json
import time
from pathlib import Path

from openai import OpenAI
from sqlalchemy import select

from backend.config import settings
from backend.database import SessionLocal, init_db
from backend.models import AudioAsset, ChatMessage, SupportCase


def transcript_from_messages(messages: list[dict]) -> str:
    return "\n".join(f"{item['speaker'].title()}: {item['text']}" for item in messages)


def create_spoken_audio(text: str, output_path: Path, voice: str) -> float:
    """Generate a real spoken MP3 locally using the OpenAI speech API."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI()
    speech = client.audio.speech.create(model=settings.openai_tts_model, voice=voice, input=text[:4000])
    speech.stream_to_file(output_path)
    # Useful estimated duration for metadata; the actual file is sent for transcription later.
    return round(max(4.0, len(text.split()) / 2.4), 1)


def seed_data() -> dict[str, int]:
    init_db()
    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    counts = {"chats": 0, "audio": 0}
    voices = ["alloy", "ash", "coral", "sage"]
    with SessionLocal() as session:
        for index, chat_path in enumerate(sorted(settings.chat_dir.glob("*.json"))):
            chat = json.loads(chat_path.read_text(encoding="utf-8"))
            messages = chat["messages"]
            if not session.scalar(select(SupportCase).where(SupportCase.source_filename == chat_path.name)):
                chat_case = SupportCase(
                    case_type="chat",
                    source_filename=chat_path.name,
                    agent_id=chat["agent_id"],
                    issue_hint=chat["issue"],
                    metadata_json={"chat_number": chat["chat_number"]},
                )
                session.add(chat_case)
                session.flush()
                for sequence, message in enumerate(messages):
                    session.add(ChatMessage(case_id=chat_case.id, sequence=sequence, speaker=message["speaker"], text=message["text"]))
                counts["chats"] += 1

            audio_name = f"{chat_path.stem}.mp3"
            if session.scalar(select(SupportCase).where(SupportCase.source_filename == audio_name)):
                continue
            transcript = transcript_from_messages(messages)
            audio_path = settings.audio_dir / audio_name
            duration = create_spoken_audio(transcript, audio_path, voices[index % len(voices)])
            audio_case = SupportCase(
                case_type="audio",
                source_filename=audio_name,
                agent_id=chat["agent_id"],
                issue_hint=chat["issue"],
                metadata_json={"derived_from_chat": chat_path.name, "recording_type": "synthetic support call"},
            )
            session.add(audio_case)
            session.flush()
            session.add(AudioAsset(
                case_id=audio_case.id,
                file_path=str(audio_path),
                duration_seconds=duration,
                language="en",
                metadata_json={"voice": voices[index % len(voices)], "generated_at": int(time.time())},
            ))
            counts["audio"] += 1
        session.commit()
    return counts


if __name__ == "__main__":
    print(seed_data())
