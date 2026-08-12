"""Celery tasks: classify a Jira ticket (priority + category) and post back."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from aiops.celery_app import app
from aiops import config
from aiops.models import loader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_text(ticket: dict[str, Any]) -> str:
    parts = [
        ticket.get("summary", ""),
        ticket.get("description", ""),
        ticket.get("comments_text", ""),
        ticket.get("top_error_lines", ""),
    ]
    return " [SEP] ".join(p for p in parts if p)


def _classify_category(text: str) -> str | None:
    """Run DeBERTa category model; returns label string or None."""
    import torch

    model = loader.get("cat_model")
    tok = loader.get("cat_tokenizer")
    le = loader.get("cat_le")
    if model is None or tok is None or le is None:
        return None

    try:
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=384)
        with torch.no_grad():
            logits = model(**inputs).logits
        idx = int(torch.argmax(logits, dim=-1).item())
        return str(le.inverse_transform([idx])[0])
    except Exception as exc:
        logger.warning("Category inference failed: %s", exc)
        return None


def _classify_priority(ticket: dict[str, Any]) -> str | None:
    """Run LightGBM priority model; returns P1–P5 string or None."""
    booster = loader.get("prio_booster")
    tfidf = loader.get("prio_tfidf")
    le = loader.get("prio_le")
    if booster is None or tfidf is None or le is None:
        return None

    try:
        text = _build_text(ticket)
        tfidf_feat = tfidf.transform([text]).toarray()

        numeric_cols = [
            "affected_users", "downtime_minutes", "error_count",
            "fatal_count", "timeout_count", "auth_error_count",
        ]
        num_feat = np.array([[float(ticket.get(c, 0) or 0) for c in numeric_cols]])

        ohe = loader.get("prio_ohe")
        cat_cols = ["env", "service"]
        cat_vals = [[str(ticket.get(c, "unknown")) for c in cat_cols]]
        if ohe is not None:
            cat_feat = ohe.transform(cat_vals)
        else:
            cat_feat = np.zeros((1, 1))

        X = np.hstack([tfidf_feat, num_feat, cat_feat])
        proba = booster.predict(X)
        idx = int(np.argmax(proba, axis=1)[0])
        return str(le.inverse_transform([idx])[0])
    except Exception as exc:
        logger.warning("Priority inference failed: %s", exc)
        return None


def _predict_incident_risk(ticket: dict[str, Any]) -> float:
    """Return probability [0,1] that this ticket precedes an incident."""
    booster = loader.get("incident_booster")
    if booster is None:
        return 0.0
    try:
        tfidf = loader.get("prio_tfidf")
        text = _build_text(ticket)
        tfidf_feat = tfidf.transform([text]).toarray() if tfidf else np.zeros((1, 100))
        numeric_cols = [
            "affected_users", "downtime_minutes", "error_count",
            "fatal_count", "timeout_count", "auth_error_count",
        ]
        num_feat = np.array([[float(ticket.get(c, 0) or 0) for c in numeric_cols]])
        X = np.hstack([tfidf_feat, num_feat])
        proba = booster.predict(X)
        return float(proba[0]) if proba.ndim == 1 else float(proba[0, 1])
    except Exception as exc:
        logger.warning("Incident risk inference failed: %s", exc)
        return 0.0


def _post_jira_comment(ticket_key: str, body: str) -> None:
    """Post a plain-text comment to a Jira issue."""
    import requests
    from requests.auth import HTTPBasicAuth

    url = f"{config.JIRA_BASE_URL}/rest/api/3/issue/{ticket_key}/comment"
    auth = HTTPBasicAuth(config.JIRA_USER, config.JIRA_API_TOKEN)
    payload = {
        "body": {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": body}],
                }
            ],
        }
    }
    resp = requests.post(url, json=payload, auth=auth, timeout=15)
    if not resp.ok:
        logger.error("Failed to post Jira comment: %s %s", resp.status_code, resp.text)


def _update_jira_fields(ticket_key: str, fields: dict[str, Any]) -> None:
    """Update arbitrary Jira issue fields."""
    import requests
    from requests.auth import HTTPBasicAuth

    url = f"{config.JIRA_BASE_URL}/rest/api/3/issue/{ticket_key}"
    auth = HTTPBasicAuth(config.JIRA_USER, config.JIRA_API_TOKEN)
    resp = requests.put(url, json={"fields": fields}, auth=auth, timeout=15)
    if not resp.ok:
        logger.error("Failed to update Jira fields: %s %s", resp.status_code, resp.text)


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

@app.task(bind=True, max_retries=3, default_retry_delay=30, name="aiops.tasks.classify.classify_ticket")
def classify_ticket(self, ticket: dict[str, Any]) -> dict[str, Any]:
    """Classify a ticket and post results back to Jira.

    Args:
        ticket: dict containing at minimum ``key``, ``summary``, ``description``.

    Returns:
        dict with ``category``, ``priority``, ``incident_risk``.
    """
    loader.load_all()

    ticket_key = ticket.get("key", "UNKNOWN")
    logger.info("Classifying ticket %s", ticket_key)

    # Acknowledge receipt
    _post_jira_comment(
        ticket_key,
        "🤖 AI Ops: Ticket received. Classification and RCA analysis in progress…",
    )

    text = _build_text(ticket)
    category = _classify_category(text) or ticket.get("final_category", "Unknown")
    priority = _classify_priority(ticket) or ticket.get("final_priority", "P3")
    incident_risk = _predict_incident_risk(ticket)

    # Post classification summary
    comment = (
        f"📊 *Classification Results*\n"
        f"• Category: {category}\n"
        f"• Priority: {priority}\n"
        f"• Incident Risk Score: {incident_risk:.0%}\n"
    )
    _post_jira_comment(ticket_key, comment)

    # Update Jira priority field if we have a valid mapping
    priority_map = {"P1": "Highest", "P2": "High", "P3": "Medium", "P4": "Low", "P5": "Lowest"}
    jira_priority = priority_map.get(priority, "Medium")
    _update_jira_fields(ticket_key, {"priority": {"name": jira_priority}})

    result = {"category": category, "priority": priority, "incident_risk": incident_risk}
    logger.info("Ticket %s classified: %s", ticket_key, result)
    return result
