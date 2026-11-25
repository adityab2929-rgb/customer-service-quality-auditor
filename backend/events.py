import json

import redis

from backend.config import settings
from backend.models import AnalysisEvent


def record_event(session, batch_id: str, job_id: str, level: str, message: str) -> None:
    event = AnalysisEvent(batch_id=batch_id, job_id=job_id, level=level, message=message)
    session.add(event)
    session.commit()
    try:
        redis.Redis.from_url(settings.redis_url, decode_responses=True).publish(
            f"support-lens:batch:{batch_id}",
            json.dumps({"level": level, "message": message, "job_id": job_id}),
        )
    except redis.RedisError:
        # The database remains the durable source of logs if Redis is temporarily unavailable.
        pass
