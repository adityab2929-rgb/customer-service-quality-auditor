from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://support_lens:support_lens@localhost:5432/support_lens"
    redis_url: str = "redis://localhost:6379/0"
    api_base_url: str = "http://localhost:8000"
    openai_audit_model: str = "gpt-4o-mini"
    openai_transcription_model: str = "gpt-transcribe"
    openai_tts_model: str = "gpt-4o-mini-tts"
    vector_db_dir: Path = PROJECT_DIR / "vector-db"
    chat_dir: Path = PROJECT_DIR / "customer-chats"
    audio_dir: Path = PROJECT_DIR / "synthetic-audio"

    model_config = SettingsConfigDict(env_file=PROJECT_DIR / ".env", extra="ignore")


settings = Settings()
