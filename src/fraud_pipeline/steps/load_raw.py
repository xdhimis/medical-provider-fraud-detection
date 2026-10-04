"""Step 1: Load Kaggle CSVs from GCS into BigQuery raw tables."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from fraud_pipeline.clients import bigquery_client, ensure_dataset, storage_client
from fraud_pipeline.config import Settings, get_settings
from fraud_pipeline.exceptions import DataLoadError
from fraud_pipeline.schemas import GCS_FILE_TO_TABLE, TABLE_SCHEMAS

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LoadResult:
    table_id: str
    source_uri: str
    output_rows: int


def _list_csv_blobs(settings: Settings) -> list[str]:
    client = storage_client(settings)
    prefix = settings.gcs_raw_prefix.strip("/") + "/"
    blobs = client.list_blobs(settings.gcs_bucket, prefix=prefix)
    uris = [
        f"gs://{settings.gcs_bucket}/{b.name}"
        for b in blobs
        if b.name.lower().endswith(".csv")
    ]
    if not uris:
        raise DataLoadError(
            f"No CSV files found under gs://{settings.gcs_bucket}/{prefix}"
        )
    return sorted(uris)


def _map_uri_to_table(uri: str) -> str | None:
    name = uri.rsplit("/", 1)[-1]
    # Prefer the most specific substrings first.
    ordered = sorted(GCS_FILE_TO_TABLE.items(), key=lambda kv: len(kv[0]), reverse=True)
    for needle, table in ordered:
        if needle in name:
            # Avoid matching Train- inside Train_Beneficiarydata via Train- rule:
            if needle == "Train-" and any(
                other in name
                for other in (
                    "Beneficiarydata",
                    "Inpatientdata",
                    "Outpatientdata",
                )
            ):
                continue
            return table
    return None


def load_raw(settings: Settings | None = None) -> list[LoadResult]:
    """Load all mapped CSVs into BigQuery with WRITE_TRUNCATE (idempotent)."""
    settings = settings or get_settings()
    bq = bigquery_client(settings)
    ensure_dataset(
        bq,
        project=settings.gcp_project,
        dataset_id=settings.bq_dataset_raw,
        location=settings.bq_location,
    )

    uris = _list_csv_blobs(settings)
    grouped: dict[str, list[str]] = {}
    unmatched: list[str] = []
    for uri in uris:
        table = _map_uri_to_table(uri)
        if table is None:
            unmatched.append(uri)
            continue
        grouped.setdefault(table, []).append(uri)

    if unmatched:
        logger.warning("Skipping unmatched CSV objects: %s", unmatched)

    required = set(TABLE_SCHEMAS)
    missing = required - set(grouped)
    if missing:
        raise DataLoadError(
            f"Missing required raw files for tables {sorted(missing)}. "
            f"Found: {sorted(grouped)}"
        )

    results: list[LoadResult] = []
    for table_name, source_uris in sorted(grouped.items()):
        table_id = f"{settings.gcp_project}.{settings.bq_dataset_raw}.{table_name}"
        job_config = bigquery.LoadJobConfig(
            schema=TABLE_SCHEMAS[table_name],
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            allow_quoted_newlines=True,
            ignore_unknown_values=True,
            autodetect=False,
            # Kaggle CSVs use literal "NA" for missing dates/numbers (e.g. DOD).
            null_markers=["NA", ""],
        )
        logger.info("Loading %s -> %s", source_uris, table_id)
        try:
            job = bq.load_table_from_uri(source_uris, table_id, job_config=job_config)
            job.result()
            table = bq.get_table(table_id)
        except Exception as exc:
            raise DataLoadError(f"Failed loading {table_id}: {exc}") from exc

        results.append(
            LoadResult(
                table_id=table_id,
                source_uri=",".join(source_uris),
                output_rows=int(table.num_rows or 0),
            )
        )
        logger.info("Loaded %s rows into %s", table.num_rows, table_id)

    return results


def table_exists(settings: Settings, table_name: str) -> bool:
    bq = bigquery_client(settings)
    table_id = f"{settings.gcp_project}.{settings.bq_dataset_raw}.{table_name}"
    try:
        bq.get_table(table_id)
        return True
    except NotFound:
        return False
