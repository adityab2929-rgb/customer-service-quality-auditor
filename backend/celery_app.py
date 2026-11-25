from celery import Celery

from backend.config import settings


celery_app = Celery("support_lens", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(task_serializer="json", result_serializer="json", accept_content=["json"], task_track_started=True)

import backend.tasks