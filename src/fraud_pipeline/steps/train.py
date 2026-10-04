"""Step 4: Train XGBoost from BigQuery features and upload artifacts to GCS."""

from __future__ import annotations

import logging
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import storage

from fraud_pipeline.clients import bigquery_client, storage_client
from fraud_pipeline.config import Settings, get_settings
from fraud_pipeline.exceptions import TrainingError
from fraud_pipeline.schemas import TrainingArtifacts
from fraud_pipeline.train_lib import train_xgboost, write_local_artifacts

logger = logging.getLogger(__name__)


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _upload_dir(client: storage.Client, local_dir: Path, gcs_uri: str) -> None:
    if not gcs_uri.startswith("gs://"):
        raise TrainingError(f"Invalid GCS URI: {gcs_uri}")
    _, _, rest = gcs_uri.partition("gs://")
    bucket_name, _, prefix = rest.partition("/")
    bucket = client.bucket(bucket_name)
    for path in local_dir.rglob("*"):
        if path.is_file():
            blob_name = f"{prefix.rstrip('/')}/{path.relative_to(local_dir).as_posix()}"
            bucket.blob(blob_name).upload_from_filename(str(path))
            logger.info("Uploaded gs://%s/%s", bucket_name, blob_name)


def train_local(settings: Settings | None = None) -> TrainingArtifacts:
    """Pull features from BigQuery, train locally, write artifacts to GCS."""
    settings = settings or get_settings()
    bq = bigquery_client(settings)
    query = f"""
        SELECT *
        FROM `{settings.features_table_id}`
        WHERE is_fraud IS NOT NULL
    """
    logger.info("Loading training frame from %s", settings.features_table_id)
    try:
        frame = bq.query(query).to_dataframe(create_bqstorage_client=False)
    except Exception as exc:
        raise TrainingError(f"Failed to load features from BigQuery: {exc}") from exc

    if frame.empty:
        raise TrainingError("Feature table returned zero labeled rows")

    result = train_xgboost(
        frame,
        test_size=settings.train_test_size,
        random_seed=settings.train_random_seed,
    )

    run_id = _new_run_id()
    gcs_uri = settings.artifact_uri(run_id)
    with tempfile.TemporaryDirectory(prefix="fraud-train-") as tmp:
        local_dir = Path(tmp)
        write_local_artifacts(result, local_dir)
        _upload_dir(storage_client(settings), local_dir, gcs_uri)

    artifacts = TrainingArtifacts(run_id=run_id, gcs_uri=gcs_uri)
    logger.info(
        "Training complete run_id=%s threshold=%.4f metrics=%s artifacts=%s",
        run_id,
        result.threshold,
        result.metrics,
        gcs_uri,
    )
    return artifacts
