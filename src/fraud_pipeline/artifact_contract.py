"""Training / serving artifact contract helpers."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fraud_pipeline.exceptions import PredictionError, TrainingError
from fraud_pipeline.schemas import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def assert_feature_columns_contract(
    columns: Sequence[str],
    *,
    expected: Sequence[str] = FEATURE_COLUMNS,
) -> tuple[str, ...]:
    """Ensure artifact feature_columns match the serving FEATURE_COLUMNS contract."""
    actual = tuple(columns)
    expected_t = tuple(expected)
    if actual != expected_t:
        raise TrainingError(
            "feature_columns.json contract mismatch: "
            f"got {len(actual)} columns {list(actual)!r}, "
            f"expected {len(expected_t)} columns {list(expected_t)!r}"
        )
    return actual


def parse_threshold_payload(payload: Any) -> float:
    """Extract a unit-interval threshold from threshold.json contents."""
    if isinstance(payload, (int, float)):
        value = float(payload)
    elif isinstance(payload, dict):
        if "threshold" not in payload:
            raise PredictionError("threshold.json missing 'threshold' key")
        value = float(payload["threshold"])
    else:
        raise PredictionError(f"Unsupported threshold.json payload: {type(payload)!r}")

    if not 0.0 <= value <= 1.0:
        raise PredictionError(f"threshold out of [0, 1]: {value}")
    return value


def load_threshold_json(path: Path | str) -> float:
    """Load decision threshold from a local threshold.json file."""
    text = Path(path).read_text(encoding="utf-8")
    return parse_threshold_payload(json.loads(text))


def load_threshold_gcs(uri: str) -> float:
    """Load decision threshold from a GCS object (file or directory)."""
    if not uri.startswith("gs://"):
        raise PredictionError(f"threshold GCS URI must start with gs://, got {uri!r}")

    from google.cloud import storage

    object_uri = uri.rstrip("/")
    if not object_uri.endswith("threshold.json"):
        object_uri = f"{object_uri}/threshold.json"

    _, _, rest = object_uri.partition("gs://")
    bucket_name, _, blob_name = rest.partition("/")
    if not bucket_name or not blob_name:
        raise PredictionError(f"Invalid threshold GCS URI: {uri!r}")

    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    if not blob.exists():
        raise PredictionError(f"threshold.json not found at {object_uri}")
    payload = json.loads(blob.download_as_text(encoding="utf-8"))
    threshold = parse_threshold_payload(payload)
    logger.info("Loaded prediction threshold %.6f from %s", threshold, object_uri)
    return threshold


def resolve_prediction_threshold(
    *,
    override: float | None,
    env_threshold: float | None,
    threshold_gcs_uri: str | None,
) -> float:
    """CLI override > PREDICTION_THRESHOLD env > threshold.json GCS > 0.5."""
    if override is not None:
        return float(override)
    if env_threshold is not None:
        return float(env_threshold)
    if threshold_gcs_uri:
        return load_threshold_gcs(threshold_gcs_uri)
    logger.warning(
        "No PREDICTION_THRESHOLD or THRESHOLD_GCS_URI set; defaulting to 0.5"
    )
    return 0.5
