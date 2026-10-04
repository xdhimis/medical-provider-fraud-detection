"""Step 2: Build provider-level features in BigQuery via SQL."""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path

from fraud_pipeline.clients import bigquery_client, ensure_dataset
from fraud_pipeline.config import Settings, get_settings
from fraud_pipeline.exceptions import DataLoadError

logger = logging.getLogger(__name__)


def _load_sql() -> str:
    # Prefer package resource; fall back to sibling file path for editable installs.
    try:
        return (
            resources.files("fraud_pipeline.sql")
            .joinpath("provider_features.sql")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError, TypeError):
        path = Path(__file__).resolve().parents[1] / "sql" / "provider_features.sql"
        return path.read_text(encoding="utf-8")


def build_features(settings: Settings | None = None) -> str:
    """Run provider feature SQL. Returns destination table id."""
    settings = settings or get_settings()
    bq = bigquery_client(settings)
    ensure_dataset(
        bq,
        project=settings.gcp_project,
        dataset_id=settings.bq_dataset_features,
        location=settings.bq_location,
    )

    sql = _load_sql().format(
        project=settings.gcp_project,
        raw_dataset=settings.bq_dataset_raw,
        features_dataset=settings.bq_dataset_features,
    )
    dest = settings.features_table_id
    logger.info("Building feature table %s", dest)
    try:
        job = bq.query(sql)
        job.result()
    except Exception as exc:
        raise DataLoadError(f"Feature SQL failed: {exc}") from exc

    table = bq.get_table(dest)
    logger.info("Feature table ready: %s (%s rows)", dest, table.num_rows)
    return dest
