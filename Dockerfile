# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY aiops/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY aiops/ /app/aiops/

# Model artifacts (bind-mount or copy from a model registry at deploy time)
# COPY artifacts/ /app/artifacts/

ENV PYTHONUNBUFFERED=1 \
    ARTIFACTS_DIR=/app/artifacts

# ── API server ────────────────────────────────────────────────────────────────
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "aiops.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Celery worker ─────────────────────────────────────────────────────────────
FROM base AS worker
CMD ["celery", "-A", "aiops.celery_app", "worker", "--loglevel=info", \
     "--queues=classify,logs,search,rca,notify,confluence,celery"]
