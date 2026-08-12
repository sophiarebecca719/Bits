"""Celery tasks: generate Root Cause Analysis via LLM and post to Jira."""

from __future__ import annotations

import logging
from typing import Any

from aiops.celery_app import app
from aiops import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM client helper
# ---------------------------------------------------------------------------

def _call_llm(prompt: str) -> str:
    """Call OpenAI or Azure OpenAI and return the response text."""
    # Prefer Azure OpenAI if endpoint is configured, otherwise fall back to OpenAI
    if config.AZURE_OPENAI_ENDPOINT and config.AZURE_OPENAI_API_KEY:
        return _call_azure_openai(prompt)
    if config.OPENAI_API_KEY:
        return _call_openai(prompt)
    raise RuntimeError("No LLM credentials configured (OPENAI_API_KEY or AZURE_OPENAI_*).")


def _call_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2048,
    )
    return resp.choices[0].message.content.strip()


def _call_azure_openai(prompt: str) -> str:
    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
    )
    resp = client.chat.completions.create(
        model=config.AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2048,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# RCA prompt builder
# ---------------------------------------------------------------------------

def _build_rca_prompt(
    ticket: dict[str, Any],
    similar_tickets: list[dict[str, Any]],
    confluence_pages: list[dict[str, Any]],
    top_log_lines: list[str],
    incident_risk: float,
) -> str:
    similar_text = "\n".join(
        f"- [{t.get('metadata', {}).get('key', 'N/A')}] "
        f"(similarity {t.get('similarity', 0):.0%}): {t.get('text', '')[:300]}"
        for t in similar_tickets
    ) or "None found."

    confluence_text = "\n".join(
        f"- [{p.get('title', '')}]({p.get('url', '')}): {p.get('excerpt', '')[:300]}"
        for p in confluence_pages
    ) or "None found."

    logs_text = "\n".join(top_log_lines[:20]) or "No logs retrieved."

    return f"""You are an expert Site Reliability Engineer performing a Root Cause Analysis.

## Ticket Details
**Key**: {ticket.get('key', 'N/A')}
**Summary**: {ticket.get('summary', '')}
**Description**: {ticket.get('description', '')}
**Category**: {ticket.get('category', '')}
**Priority**: {ticket.get('priority', '')}
**Service**: {ticket.get('service', '')}
**Environment**: {ticket.get('env', '')}
**Incident Risk Score**: {incident_risk:.0%}

## Similar Past Tickets
{similar_text}

## Relevant Confluence Knowledge Base
{confluence_text}

## Top Relevant Log Lines
{logs_text}

## Instructions
Produce a structured RCA with the following sections:

1. **Root Cause** — What is the primary technical cause?
2. **Contributing Factors** — Secondary conditions that amplified the issue.
3. **Impact** — Services/users affected and severity.
4. **Timeline** — Inferred sequence of events leading to the issue.
5. **Resolution Steps** — Concrete actions to resolve the issue now.
6. **Prevention / Follow-up** — Long-term fixes, monitoring improvements, runbook updates.

Be concise, technical, and actionable. Base your analysis strictly on the evidence provided.
"""


# ---------------------------------------------------------------------------
# Jira comment helper
# ---------------------------------------------------------------------------

def _post_jira_comment(ticket_key: str, body: str) -> None:
    import requests
    from requests.auth import HTTPBasicAuth

    if not config.JIRA_BASE_URL or not config.JIRA_API_TOKEN:
        logger.warning("Jira credentials not configured — skipping comment post")
        return

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


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

@app.task(bind=True, max_retries=3, default_retry_delay=60, name="aiops.tasks.rca.generate_rca")
def generate_rca(self, ticket: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Generate an RCA and post it to Jira.

    Args:
        ticket: classified ticket dict (key, summary, description, category, priority, …).
        context: combined output of classify, log_fetch, and search tasks:
            {similar_tickets, confluence_pages, top_log_lines, incident_risk}.

    Returns:
        dict with ``rca_text`` and ``confluence_created`` flag.
    """
    ticket_key = ticket.get("key", "UNKNOWN")
    logger.info("Generating RCA for %s", ticket_key)

    similar = context.get("similar_tickets", [])
    confluence = context.get("confluence_pages", [])
    logs = context.get("top_log_lines", [])
    incident_risk = float(context.get("incident_risk", 0.0))

    try:
        prompt = _build_rca_prompt(ticket, similar, confluence, logs, incident_risk)
        rca_text = _call_llm(prompt)
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        raise self.retry(exc=exc)

    # Post RCA to Jira
    header = (
        f"🔍 *Root Cause Analysis — {ticket_key}*\n"
        f"────────────────────────────────────\n"
    )
    _post_jira_comment(ticket_key, header + rca_text)

    # If no Confluence pages found, create a knowledge article
    confluence_created = False
    if not confluence:
        logger.info("No Confluence pages found — creating knowledge article for %s", ticket_key)
        from aiops.tasks.confluence_kb import create_kb_article
        create_kb_article.delay(ticket=ticket, rca_text=rca_text)
        confluence_created = True

    return {"rca_text": rca_text, "confluence_created": confluence_created}
