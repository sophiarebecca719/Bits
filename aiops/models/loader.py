"""Load trained model artifacts once at process startup.

Artifacts expected under ``config.ARTIFACTS_DIR``:
  classification_deberta/   — DeBERTa seq-classification model + tokenizer + label_encoder.joblib
  priority_lgbm/            — LightGBM booster (priority_model.txt) + tfidf_vectorizer.joblib
                              + priority_label_encoder.joblib + ohe_encoder.joblib
  log_reranker/             — cross-encoder checkpoint
  incident_risk/            — LightGBM booster (incident_risk_model.txt)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Any] = {}


def _art(name: str) -> Path:
    from aiops import config
    return Path(config.ARTIFACTS_DIR) / name


def _load_category_model() -> None:
    """Load DeBERTa category classifier (optional — falls back to None)."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        path = _art("classification_deberta")
        if not path.exists():
            logger.warning("Category model not found at %s — skipping", path)
            return

        tok = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSequenceClassification.from_pretrained(str(path))
        model.eval()
        le = joblib.load(path / "label_encoder.joblib")

        _REGISTRY["cat_tokenizer"] = tok
        _REGISTRY["cat_model"] = model
        _REGISTRY["cat_le"] = le
        logger.info("Category model loaded from %s", path)
    except Exception as exc:
        logger.warning("Could not load category model: %s", exc)


def _load_priority_model() -> None:
    """Load LightGBM priority classifier."""
    try:
        import lightgbm as lgb

        path = _art("priority_lgbm")
        if not path.exists():
            logger.warning("Priority model not found at %s — skipping", path)
            return

        booster = lgb.Booster(model_file=str(path / "priority_model.txt"))
        tfidf = joblib.load(path / "tfidf_vectorizer.joblib")
        le = joblib.load(path / "priority_label_encoder.joblib")
        ohe = joblib.load(path / "ohe_encoder.joblib") if (path / "ohe_encoder.joblib").exists() else None

        _REGISTRY["prio_booster"] = booster
        _REGISTRY["prio_tfidf"] = tfidf
        _REGISTRY["prio_le"] = le
        _REGISTRY["prio_ohe"] = ohe
        logger.info("Priority model loaded from %s", path)
    except Exception as exc:
        logger.warning("Could not load priority model: %s", exc)


def _load_log_reranker() -> None:
    """Load cross-encoder log re-ranker."""
    try:
        from sentence_transformers.cross_encoder import CrossEncoder

        path = _art("log_reranker")
        if not path.exists():
            logger.warning("Log reranker not found at %s — skipping", path)
            return

        _REGISTRY["log_reranker"] = CrossEncoder(str(path))
        logger.info("Log reranker loaded from %s", path)
    except Exception as exc:
        logger.warning("Could not load log reranker: %s", exc)


def _load_incident_risk() -> None:
    """Load LightGBM incident-risk predictor."""
    try:
        import lightgbm as lgb

        path = _art("incident_risk")
        if not path.exists():
            logger.warning("Incident risk model not found at %s — skipping", path)
            return

        _REGISTRY["incident_booster"] = lgb.Booster(
            model_file=str(path / "incident_risk_model.txt")
        )
        logger.info("Incident risk model loaded from %s", path)
    except Exception as exc:
        logger.warning("Could not load incident risk model: %s", exc)


def load_all() -> None:
    """Call once at application startup to populate the model registry."""
    if _REGISTRY:
        return  # already loaded
    _load_category_model()
    _load_priority_model()
    _load_log_reranker()
    _load_incident_risk()
    logger.info("Model registry ready: %s", list(_REGISTRY.keys()))


def get(name: str) -> Any:
    """Retrieve a loaded artefact by name; returns None if unavailable."""
    return _REGISTRY.get(name)
