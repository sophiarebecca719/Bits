"""Celery application instance and task routing."""

from celery import Celery
from aiops import config

app = Celery(
    "aiops",
    broker=config.REDIS_URL,
    backend=config.REDIS_URL,
    include=[
        "aiops.tasks.classify",
        "aiops.tasks.log_fetch",
        "aiops.tasks.search",
        "aiops.tasks.rca",
        "aiops.tasks.notify",
        "aiops.tasks.confluence_kb",
    ],
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "aiops.tasks.classify.*": {"queue": "classify"},
        "aiops.tasks.log_fetch.*": {"queue": "logs"},
        "aiops.tasks.search.*": {"queue": "search"},
        "aiops.tasks.rca.*": {"queue": "rca"},
        "aiops.tasks.notify.*": {"queue": "notify"},
        "aiops.tasks.confluence_kb.*": {"queue": "confluence"},
    },
    task_default_retry_delay=30,   # seconds
    task_max_retries=3,
)
