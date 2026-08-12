"""Celery tasks: fetch logs from Azure File Share (mobile) or Datadog (server)."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any

from aiops.celery_app import app
from aiops import config
from aiops.models import loader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Log re-ranking helper
# ---------------------------------------------------------------------------

def _rerank_logs(query: str, log_lines: list[str]) -> list[str]:
    """Score log lines against the ticket query and return top-k."""
    reranker = loader.get("log_reranker")
    if reranker is None or not log_lines:
        return log_lines[: config.TOP_K_LOG_LINES]

    try:
        pairs = [(query, line) for line in log_lines]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, log_lines), key=lambda x: x[0], reverse=True)
        return [line for _, line in ranked[: config.TOP_K_LOG_LINES]]
    except Exception as exc:
        logger.warning("Log reranking failed: %s", exc)
        return log_lines[: config.TOP_K_LOG_LINES]


# ---------------------------------------------------------------------------
# Azure File Share — mobile logs
# ---------------------------------------------------------------------------

def _fetch_azure_logs(service: str, env: str, lookback_hours: int = 6) -> list[str]:
    """Fetch log lines from Azure File Share for mobile issues."""
    try:
        from azure.storage.fileshare import ShareServiceClient

        conn_str = config.AZURE_STORAGE_CONNECTION_STRING
        if not conn_str:
            logger.warning("AZURE_STORAGE_CONNECTION_STRING not set — skipping mobile logs")
            return []

        svc = ShareServiceClient.from_connection_string(conn_str)
        share_client = svc.get_share_client(config.AZURE_FILE_SHARE_NAME)

        directory = config.AZURE_LOG_DIRECTORY or service
        dir_client = share_client.get_directory_client(directory)

        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        lines: list[str] = []

        for item in dir_client.list_directories_and_files():
            if item["is_directory"]:
                continue
            if item.get("last_modified") and item["last_modified"] < cutoff:
                continue

            file_client = dir_client.get_file_client(item["name"])
            content = file_client.download_file().readall().decode("utf-8", errors="replace")
            lines.extend(content.splitlines())

        logger.info("Fetched %d log lines from Azure File Share (service=%s)", len(lines), service)
        return lines
    except Exception as exc:
        logger.error("Azure log fetch failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Datadog — server logs
# ---------------------------------------------------------------------------

def _fetch_datadog_logs(service: str, env: str, lookback_hours: int | None = None) -> list[str]:
    """Fetch log lines from Datadog Logs API for server issues."""
    try:
        import requests

        if not config.DATADOG_API_KEY:
            logger.warning("DATADOG_API_KEY not set — skipping Datadog logs")
            return []

        hours = lookback_hours or config.DATADOG_LOG_LOOKBACK_HOURS
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=hours)

        url = f"https://api.{config.DATADOG_SITE}/api/v2/logs/events/search"
        headers = {
            "DD-API-KEY": config.DATADOG_API_KEY,
            "DD-APPLICATION-KEY": config.DATADOG_APP_KEY,
            "Content-Type": "application/json",
        }
        query_filter = f"service:{service}"
        if env:
            query_filter += f" env:{env}"
        query_filter += " status:(error OR warn OR critical)"

        payload = {
            "filter": {
                "query": query_filter,
                "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            "sort": "timestamp",
            "page": {"limit": 1000},
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=20)
        resp.raise_for_status()

        data = resp.json()
        events = data.get("data", [])
        lines: list[str] = []
        for evt in events:
            msg = evt.get("attributes", {}).get("message", "")
            if msg:
                lines.append(msg)

        logger.info("Fetched %d log lines from Datadog (service=%s)", len(lines), service)
        return lines
    except Exception as exc:
        logger.error("Datadog log fetch failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

@app.task(bind=True, max_retries=3, default_retry_delay=30, name="aiops.tasks.log_fetch.fetch_logs")
def fetch_logs(self, ticket: dict[str, Any]) -> dict[str, Any]:
    """Fetch and re-rank logs relevant to the ticket.

    Determines source (Azure = mobile, Datadog = server) from ticket category.

    Args:
        ticket: dict with ``key``, ``category``, ``service``, ``env``.

    Returns:
        dict with ``top_log_lines`` (list[str]) and ``source`` ("azure" | "datadog" | "none").
    """
    loader.load_all()

    category = (ticket.get("category") or ticket.get("final_category") or "").lower()
    service = ticket.get("service", "")
    env = ticket.get("env", "")
    query = f"{ticket.get('summary', '')} {ticket.get('description', '')}"

    is_mobile = any(kw in category for kw in ("mobile", "android", "ios", "app"))

    if is_mobile:
        raw_lines = _fetch_azure_logs(service, env)
        source = "azure"
    else:
        raw_lines = _fetch_datadog_logs(service, env)
        source = "datadog"

    if not raw_lines:
        return {"top_log_lines": [], "source": "none"}

    top_lines = _rerank_logs(query, raw_lines)

    # Threshold monitoring: count errors/warnings
    error_count = sum(1 for l in raw_lines if "error" in l.lower() or "exception" in l.lower())
    warn_count = sum(1 for l in raw_lines if "warn" in l.lower())
    fatal_count = sum(1 for l in raw_lines if "fatal" in l.lower() or "critical" in l.lower())

    thresholds_exceeded = (
        error_count >= config.ERROR_COUNT_THRESHOLD
        or warn_count >= config.WARNING_COUNT_THRESHOLD
        or fatal_count >= config.FATAL_COUNT_THRESHOLD
    )

    if thresholds_exceeded:
        logger.warning(
            "Log thresholds exceeded for %s — errors=%d, warnings=%d, fatals=%d",
            ticket.get("key"), error_count, warn_count, fatal_count,
        )
        from aiops.tasks.notify import send_threshold_alert
        send_threshold_alert.delay(
            ticket=ticket,
            error_count=error_count,
            warn_count=warn_count,
            fatal_count=fatal_count,
            top_lines=top_lines[:10],
        )

    return {
        "top_log_lines": top_lines,
        "source": source,
        "error_count": error_count,
        "warn_count": warn_count,
        "fatal_count": fatal_count,
    }
