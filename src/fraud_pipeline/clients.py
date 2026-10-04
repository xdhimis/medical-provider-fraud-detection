"""Shared GCP client helpers (ADC / WIF — no embedded credentials)."""

from __future__ import annotations

from google.cloud import bigquery, storage
from google.cloud import aiplatform

from fraud_pipeline.config import Settings


def init_vertex(settings: Settings) -> None:
    """Initialize the Vertex AI SDK for the configured project/region."""
    aiplatform.init(project=settings.gcp_project, location=settings.gcp_region)


def bigquery_client(settings: Settings) -> bigquery.Client:
    return bigquery.Client(project=settings.gcp_project, location=settings.bq_location)


def storage_client(settings: Settings) -> storage.Client:
    return storage.Client(project=settings.gcp_project)


def ensure_dataset(
    client: bigquery.Client,
    *,
    project: str,
    dataset_id: str,
    location: str,
) -> bigquery.Dataset:
    """Create a BigQuery dataset if it does not exist (idempotent)."""
    ref = bigquery.Dataset(f"{project}.{dataset_id}")
    ref.location = location
    return client.create_dataset(ref, exists_ok=True)
