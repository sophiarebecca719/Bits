"""FastAPI application: Jira & Datadog webhook endpoints + pipeline orchestration."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from aiops import config
from aiops.models import loader
from aiops.tasks.classify import classify_ticket
from aiops.tasks.log_fetch import fetch_logs
from aiops.tasks.search import search_context
from aiops.tasks.rca import generate_rca
from aiops.tasks.notify import send_rca_notification, send_datadog_alert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Ops Automation",
    description=(
        "End-to-end AI Ops pipeline: Jira ingestion → classification → "
        "log fetching → RCA generation → notifications."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    """Pre-load ML models so the first request is not slow."""
    loader.load_all()
    logger.info("AI Ops application started.")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Jira webhook
# ---------------------------------------------------------------------------

@app.post("/webhook/jira", tags=["webhooks"])
async def jira_webhook(request: Request) -> JSONResponse:
    """Receive Jira issue-created / issue-updated events.

    Jira should POST to this endpoint via a *webhook* configured in
    Project Settings → System → Webhooks.
    """
    try:
        payload: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    event = payload.get("webhookEvent", "")
    issue = payload.get("issue", {})
    fields = issue.get("fields", {})

    if not issue:
        return JSONResponse({"status": "ignored", "reason": "no issue in payload"})

    # Build normalised ticket dict
    ticket: dict[str, Any] = {
        "key": issue.get("key", ""),
        "summary": fields.get("summary", ""),
        "description": _extract_description(fields.get("description")),
        "comments_text": "",
        "top_error_lines": "",
        "env": _extract_custom_field(fields, "customfield_env", "unknown"),
        "service": _extract_custom_field(fields, "customfield_service", "unknown"),
        "affected_users": _extract_custom_field(fields, "customfield_affected_users", 0),
        "downtime_minutes": _extract_custom_field(fields, "customfield_downtime_minutes", 0),
        "error_count": 0,
        "fatal_count": 0,
        "timeout_count": 0,
        "auth_error_count": 0,
    }

    logger.info("Received Jira event '%s' for ticket %s", event, ticket["key"])

    # Fire-and-forget full pipeline via Celery chain
    _queue_pipeline(ticket)

    return JSONResponse({"status": "accepted", "ticket_key": ticket["key"]})


# ---------------------------------------------------------------------------
# Datadog webhook
# ---------------------------------------------------------------------------

@app.post("/webhook/datadog", tags=["webhooks"])
async def datadog_webhook(request: Request) -> JSONResponse:
    """Receive Datadog monitor alert events.

    Configure in Datadog: Integrations → Webhooks → add URL pointing here.
    """
    try:
        payload: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    logger.info("Received Datadog alert: %s", payload.get("monitor_name", "unknown"))

    # Send immediate email notification
    send_datadog_alert.delay(alert_payload=payload)

    # Create a Jira ticket for the alert if not already tracked
    jira_key = _create_jira_ticket_for_datadog(payload)

    # Build a synthetic ticket and queue the full RCA pipeline
    if jira_key:
        # Datadog sends tags as a list of "key:value" strings
        raw_tags = payload.get("tags", [])
        tag_dict: dict[str, str] = {}
        if isinstance(raw_tags, list):
            for tag in raw_tags:
                if ":" in tag:
                    k, v = tag.split(":", 1)
                    tag_dict[k] = v
        elif isinstance(raw_tags, dict):
            tag_dict = raw_tags

        ticket: dict[str, Any] = {
            "key": jira_key,
            "summary": f"[Datadog] {payload.get('monitor_name', 'Monitor Alert')}",
            "description": (
                f"Metric: {payload.get('metric', 'N/A')}\n"
                f"Value: {payload.get('value', 'N/A')} "
                f"(threshold: {payload.get('threshold', 'N/A')})\n"
                f"Snapshot: {payload.get('snapshot_url', '')}"
            ),
            "env": tag_dict.get("env", "unknown"),
            "service": tag_dict.get("service", "unknown"),
            "error_count": 0,
            "fatal_count": 0,
            "timeout_count": 0,
            "auth_error_count": 0,
            "affected_users": 0,
            "downtime_minutes": 0,
            "comments_text": "",
            "top_error_lines": "",
        }
        _queue_pipeline(ticket)

    return JSONResponse({"status": "accepted", "jira_key": jira_key})


# ---------------------------------------------------------------------------
# Manual trigger (for testing / backfill)
# ---------------------------------------------------------------------------

@app.post("/analyze", tags=["api"])
async def analyze_ticket(ticket: dict[str, Any]) -> JSONResponse:
    """Manually trigger the full analysis pipeline for a ticket dict.

    Useful for testing without a live Jira webhook.
    """
    if not ticket.get("key"):
        raise HTTPException(status_code=400, detail="'key' field is required")
    _queue_pipeline(ticket)
    return JSONResponse({"status": "queued", "ticket_key": ticket["key"]})


# ---------------------------------------------------------------------------
# Pipeline helper
# ---------------------------------------------------------------------------

def _queue_pipeline(ticket: dict[str, Any]) -> None:
    """Queue the full analysis pipeline as a Celery chord/chain."""
    from celery import chain

    pipeline = chain(
        classify_ticket.s(ticket),
        _merge_classify.s(ticket),
        _run_parallel_context.s(),
        _merge_context.s(ticket),
        generate_rca.s(ticket),                     # type: ignore[arg-type]
        _notify_after_rca.s(ticket),
    )
    pipeline.delay()


# ---------------------------------------------------------------------------
# Intermediate glue tasks (lightweight — run in the default queue)
# ---------------------------------------------------------------------------

from aiops.celery_app import app as celery_app  # noqa: E402


@celery_app.task(name="aiops.main._merge_classify")
def _merge_classify(classify_result: dict[str, Any], ticket: dict[str, Any]) -> dict[str, Any]:
    """Merge classification results back into the ticket dict."""
    merged = {**ticket, **classify_result}
    return merged


@celery_app.task(name="aiops.main._run_parallel_context")
def _run_parallel_context(ticket: dict[str, Any]) -> dict[str, Any]:
    """Run log fetch and search in parallel via Celery workers."""
    from celery import group

    job = group(
        fetch_logs.s(ticket),
        search_context.s(ticket),
    )
    result = job.apply_async()
    log_result, search_result = result.get(timeout=120)
    return {**log_result, **search_result}


@celery_app.task(name="aiops.main._merge_context")
def _merge_context(context: dict[str, Any], ticket: dict[str, Any]) -> dict[str, Any]:
    """Attach incident_risk from ticket into context before RCA."""
    context["incident_risk"] = ticket.get("incident_risk", 0.0)
    return context


@celery_app.task(name="aiops.main._notify_after_rca")
def _notify_after_rca(rca_result: dict[str, Any], ticket: dict[str, Any]) -> None:
    """Send RCA notification email after generation."""
    send_rca_notification.delay(ticket=ticket, rca_text=rca_result.get("rca_text", ""))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_description(desc: Any) -> str:
    """Extract plain text from Jira's Atlassian Document Format description."""
    if desc is None:
        return ""
    if isinstance(desc, str):
        return desc
    # ADF: traverse content nodes
    parts: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "text":
                parts.append(node.get("text", ""))
            for child in node.get("content", []):
                _walk(child)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(desc)
    return " ".join(parts).strip()


def _extract_custom_field(fields: dict[str, Any], field_id: str, default: Any) -> Any:
    val = fields.get(field_id)
    if val is None:
        return default
    if isinstance(val, dict):
        return val.get("value", default)
    return val


def _create_jira_ticket_for_datadog(payload: dict[str, Any]) -> str | None:
    """Create a Jira issue for a Datadog monitor alert; return the issue key."""
    try:
        import requests
        from requests.auth import HTTPBasicAuth

        if not config.JIRA_BASE_URL or not config.JIRA_API_TOKEN:
            return None

        monitor_name = payload.get("monitor_name", "Datadog Monitor Alert")
        url = f"{config.JIRA_BASE_URL}/rest/api/3/issue"
        auth = HTTPBasicAuth(config.JIRA_USER, config.JIRA_API_TOKEN)

        issue_payload = {
            "fields": {
                "project": {"key": config.JIRA_PROJECT_KEY},
                "summary": f"[Datadog Alert] {monitor_name}",
                "issuetype": {"name": "Incident"},
                "priority": {"name": "High"},
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Automated Jira ticket created by AI Ops.\n"
                                        f"Monitor: {monitor_name}\n"
                                        f"Metric: {payload.get('metric', 'N/A')}\n"
                                        f"Value: {payload.get('value', 'N/A')}\n"
                                        f"Threshold: {payload.get('threshold', 'N/A')}"
                                    ),
                                }
                            ],
                        }
                    ],
                },
            }
        }

        resp = requests.post(url, json=issue_payload, auth=auth, timeout=15)
        if resp.ok:
            key = resp.json().get("key")
            logger.info("Created Jira ticket %s for Datadog alert", key)
            return key
        else:
            logger.error("Failed to create Jira ticket: %s %s", resp.status_code, resp.text)
            return None
    except Exception as exc:
        logger.error("Exception creating Jira ticket: %s", exc)
        return None
