"""Celery tasks: find similar historical tickets and relevant Confluence pages."""

from __future__ import annotations

import logging
from typing import Any

from aiops.celery_app import app
from aiops import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Similar ticket retrieval via ChromaDB
# ---------------------------------------------------------------------------

def _get_chroma_collection():
    """Return (or lazily create) a ChromaDB collection of historical tickets."""
    try:
        import chromadb

        client = chromadb.Client()
        return client.get_or_create_collection("tickets")
    except Exception as exc:
        logger.warning("ChromaDB unavailable: %s", exc)
        return None


def _search_similar_tickets(query: str, top_k: int) -> list[dict[str, Any]]:
    """Query ChromaDB for the top-k most similar historical tickets."""
    collection = _get_chroma_collection()
    if collection is None:
        return []

    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        tickets = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            tickets.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "similarity": round(1 - dist, 4),
                }
            )
        return tickets
    except Exception as exc:
        logger.error("Similar ticket search failed: %s", exc)
        return []


def index_ticket(ticket: dict[str, Any]) -> None:
    """Upsert a ticket into the ChromaDB collection (call after resolution)."""
    collection = _get_chroma_collection()
    if collection is None:
        return
    try:
        text = (
            f"{ticket.get('summary', '')} "
            f"{ticket.get('description', '')} "
            f"{ticket.get('comments_text', '')}"
        )
        collection.upsert(
            ids=[ticket.get("key", "")],
            documents=[text],
            metadatas=[
                {
                    "key": ticket.get("key", ""),
                    "category": ticket.get("category", ""),
                    "priority": ticket.get("priority", ""),
                    "service": ticket.get("service", ""),
                }
            ],
        )
    except Exception as exc:
        logger.error("Failed to index ticket: %s", exc)


# ---------------------------------------------------------------------------
# Confluence search
# ---------------------------------------------------------------------------

def _search_confluence(query: str, top_k: int) -> list[dict[str, Any]]:
    """Search Confluence using CQL and return top-k matching pages."""
    try:
        import requests
        from requests.auth import HTTPBasicAuth

        if not config.CONFLUENCE_BASE_URL or not config.JIRA_API_TOKEN:
            logger.warning("Confluence credentials not configured — skipping")
            return []

        url = f"{config.CONFLUENCE_BASE_URL}/wiki/rest/api/content/search"
        auth = HTTPBasicAuth(config.JIRA_USER, config.JIRA_API_TOKEN)
        cql = (
            f'space = "{config.CONFLUENCE_SPACE_KEY}" AND text ~ "{query}" '
            f'ORDER BY relevance DESC'
        )
        params = {
            "cql": cql,
            "limit": top_k,
            "expand": "body.storage,metadata.labels",
        }
        resp = requests.get(url, params=params, auth=auth, timeout=15)
        resp.raise_for_status()

        results = resp.json().get("results", [])
        pages = []
        for r in results:
            body_val = (
                r.get("body", {}).get("storage", {}).get("value", "")
            )
            # Strip HTML tags for plain-text summary
            import re
            plain = re.sub(r"<[^>]+>", " ", body_val)[:1000]
            pages.append(
                {
                    "title": r.get("title", ""),
                    "url": (
                        config.CONFLUENCE_BASE_URL
                        + r.get("_links", {}).get("webui", "")
                    ),
                    "excerpt": plain.strip(),
                }
            )
        logger.info("Confluence returned %d pages for query '%s'", len(pages), query[:60])
        return pages
    except Exception as exc:
        logger.error("Confluence search failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

@app.task(bind=True, max_retries=3, default_retry_delay=30, name="aiops.tasks.search.search_context")
def search_context(self, ticket: dict[str, Any]) -> dict[str, Any]:
    """Find similar tickets and Confluence articles for the given ticket.

    Args:
        ticket: dict with ``key``, ``summary``, ``description``.

    Returns:
        dict with ``similar_tickets`` (list) and ``confluence_pages`` (list).
    """
    query = f"{ticket.get('summary', '')} {ticket.get('description', '')}".strip()
    if not query:
        return {"similar_tickets": [], "confluence_pages": []}

    similar = _search_similar_tickets(query, config.TOP_K_SIMILAR_TICKETS)
    confluence = _search_confluence(query[:200], config.TOP_K_CONFLUENCE)

    logger.info(
        "Ticket %s: found %d similar tickets, %d Confluence pages",
        ticket.get("key"), len(similar), len(confluence),
    )
    return {"similar_tickets": similar, "confluence_pages": confluence}
