"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Required environment variable '{key}' is not set.")
    return val


# ---------------------------------------------------------------------------
# Jira
# ---------------------------------------------------------------------------
JIRA_BASE_URL: str = os.getenv("JIRA_BASE_URL", "")          # e.g. https://myorg.atlassian.net
JIRA_USER: str = os.getenv("JIRA_USER", "")                   # service-account email
JIRA_API_TOKEN: str = os.getenv("JIRA_API_TOKEN", "")
JIRA_PROJECT_KEY: str = os.getenv("JIRA_PROJECT_KEY", "OPS")

# ---------------------------------------------------------------------------
# Confluence
# ---------------------------------------------------------------------------
CONFLUENCE_BASE_URL: str = os.getenv("CONFLUENCE_BASE_URL", JIRA_BASE_URL)
CONFLUENCE_SPACE_KEY: str = os.getenv("CONFLUENCE_SPACE_KEY", "KB")
CONFLUENCE_PARENT_PAGE_ID: str = os.getenv("CONFLUENCE_PARENT_PAGE_ID", "")

# ---------------------------------------------------------------------------
# Azure Storage (mobile logs)
# ---------------------------------------------------------------------------
AZURE_STORAGE_CONNECTION_STRING: str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_FILE_SHARE_NAME: str = os.getenv("AZURE_FILE_SHARE_NAME", "mobile-logs")
AZURE_LOG_DIRECTORY: str = os.getenv("AZURE_LOG_DIRECTORY", "")

# ---------------------------------------------------------------------------
# Datadog (server logs + monitoring)
# ---------------------------------------------------------------------------
DATADOG_API_KEY: str = os.getenv("DATADOG_API_KEY", "")
DATADOG_APP_KEY: str = os.getenv("DATADOG_APP_KEY", "")
DATADOG_SITE: str = os.getenv("DATADOG_SITE", "datadoghq.com")
DATADOG_LOG_LOOKBACK_HOURS: int = int(os.getenv("DATADOG_LOG_LOOKBACK_HOURS", "6"))

# ---------------------------------------------------------------------------
# OpenAI / Azure OpenAI
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# ---------------------------------------------------------------------------
# Email / SMTP
# ---------------------------------------------------------------------------
SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER: str = os.getenv("SMTP_USER", "")
SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
ALERT_EMAIL_RECIPIENTS: list[str] = [
    e.strip()
    for e in os.getenv("ALERT_EMAIL_RECIPIENTS", "").split(",")
    if e.strip()
]

# ---------------------------------------------------------------------------
# Celery / Redis
# ---------------------------------------------------------------------------
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ---------------------------------------------------------------------------
# Model artifacts
# ---------------------------------------------------------------------------
ARTIFACTS_DIR: str = os.getenv("ARTIFACTS_DIR", "artifacts")

# ---------------------------------------------------------------------------
# Log alert thresholds
# ---------------------------------------------------------------------------
ERROR_COUNT_THRESHOLD: int = int(os.getenv("ERROR_COUNT_THRESHOLD", "50"))
WARNING_COUNT_THRESHOLD: int = int(os.getenv("WARNING_COUNT_THRESHOLD", "200"))
FATAL_COUNT_THRESHOLD: int = int(os.getenv("FATAL_COUNT_THRESHOLD", "5"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K_SIMILAR_TICKETS: int = int(os.getenv("TOP_K_SIMILAR_TICKETS", "5"))
TOP_K_LOG_LINES: int = int(os.getenv("TOP_K_LOG_LINES", "20"))
TOP_K_CONFLUENCE: int = int(os.getenv("TOP_K_CONFLUENCE", "3"))

# ChromaDB
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "/tmp/aiops_chroma")
