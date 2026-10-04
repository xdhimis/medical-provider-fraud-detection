"""POC cleanup: stop spend by undeploying/deleting Vertex + optional data resources.

Order (highest burn first):
  1. Undeploy + delete Endpoints matching ENDPOINT_DISPLAY_NAME
  2. Delete Feature Online Store (force removes FeatureViews)
  3. Delete Model Registry entries matching MODEL_DISPLAY_NAME
  4. (--all) Delete BigQuery raw/features datasets
  5. (--all) Delete GCS objects under ARTIFACT_GCS_PREFIX (models/), not raw CSVs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from google.api_core import exceptions as gax_exceptions
from google.cloud import aiplatform, bigquery, storage

from fraud_pipeline.clients import init_vertex
from fraud_pipeline.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class CleanupReport:
    dry_run: bool
    actions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        logger.info(message)
        self.actions.append(message)

    def fail(self, message: str) -> None:
        logger.warning(message)
        self.errors.append(message)


def cleanup_poc(
    *,
    settings: Settings | None = None,
    delete_data: bool = False,
    dry_run: bool = False,
) -> CleanupReport:
    """Tear down POC resources. Set delete_data=True to also drop BQ + model artifacts."""
    settings = settings or get_settings()
    report = CleanupReport(dry_run=dry_run)

    _cleanup_endpoints(settings, report=report, dry_run=dry_run)
    _cleanup_feature_store(settings, report=report, dry_run=dry_run)
    _cleanup_models(settings, report=report, dry_run=dry_run)

    if delete_data:
        _cleanup_bq_datasets(settings, report=report, dry_run=dry_run)
        _cleanup_gcs_model_artifacts(settings, report=report, dry_run=dry_run)
    else:
        report.note(
            "Skipped BigQuery/GCS data cleanup (pass delete_data=True / --all to include)"
        )

    return report


def _cleanup_endpoints(settings: Settings, *, report: CleanupReport, dry_run: bool) -> None:
    init_vertex(settings)
    endpoints = list(
        aiplatform.Endpoint.list(
            filter=f'display_name="{settings.endpoint_display_name}"',
            order_by="create_time desc",
        )
    )
    if not endpoints:
        report.note(
            f"No endpoints found with display_name={settings.endpoint_display_name}"
        )
        return

    for endpoint in endpoints:
        report.note(f"Endpoint {endpoint.resource_name}")
        deployed = list(endpoint.gca_resource.deployed_models)
        if dry_run:
            report.note(
                f"[dry-run] would undeploy {len(deployed)} model(s) and delete endpoint"
            )
            continue
        for dm in deployed:
            report.note(f"Undeploying deployed_model_id={dm.id}")
            try:
                endpoint.undeploy(deployed_model_id=dm.id, sync=True)
            except Exception as exc:
                report.fail(f"Undeploy failed for {dm.id}: {exc}")
        try:
            endpoint.delete(force=True, sync=True)
            report.note(f"Deleted endpoint {endpoint.resource_name}")
        except Exception as exc:
            report.fail(f"Endpoint delete failed: {exc}")


def _cleanup_feature_store(
    settings: Settings, *, report: CleanupReport, dry_run: bool
) -> None:
    init_vertex(settings)
    store_id = settings.feature_online_store
    report.note(f"Feature Online Store id={store_id}")
    if dry_run:
        report.note("[dry-run] would delete Feature Online Store (force=True)")
        return

    try:
        from vertexai.resources.preview import feature_store

        store = feature_store.FeatureOnlineStore(store_id)
        # force=True removes nested FeatureViews.
        delete_fn = getattr(store, "delete", None)
        if delete_fn is None:
            report.fail("FeatureOnlineStore.delete not available in this SDK version")
            return
        try:
            delete_fn(force=True)
        except TypeError:
            delete_fn(force=True, sync=True)
        report.note(f"Deleted Feature Online Store {store_id}")
    except gax_exceptions.NotFound:
        report.note(f"Feature Online Store not found: {store_id}")
    except Exception as exc:
        if "not found" in str(exc).lower() or "404" in str(exc):
            report.note(f"Feature Online Store not found: {store_id}")
        else:
            report.fail(f"Feature Online Store delete failed: {exc}")


def _cleanup_models(settings: Settings, *, report: CleanupReport, dry_run: bool) -> None:
    init_vertex(settings)
    models = list(
        aiplatform.Model.list(
            filter=f'display_name="{settings.model_display_name}"',
            order_by="create_time desc",
        )
    )
    if not models:
        report.note(f"No models found with display_name={settings.model_display_name}")
        return

    for model in models:
        report.note(f"Model {model.resource_name}")
        if dry_run:
            report.note("[dry-run] would delete model")
            continue
        try:
            model.delete(sync=True)
            report.note(f"Deleted model {model.resource_name}")
        except Exception as exc:
            report.fail(f"Model delete failed: {exc}")


def _cleanup_bq_datasets(
    settings: Settings, *, report: CleanupReport, dry_run: bool
) -> None:
    client = bigquery.Client(project=settings.gcp_project)
    for ds in (settings.bq_dataset_raw, settings.bq_dataset_features):
        dataset_id = f"{settings.gcp_project}.{ds}"
        report.note(f"BigQuery dataset {dataset_id}")
        if dry_run:
            report.note("[dry-run] would delete dataset (contents=True)")
            continue
        try:
            client.delete_dataset(dataset_id, delete_contents=True, not_found_ok=True)
            report.note(f"Deleted dataset {dataset_id}")
        except Exception as exc:
            report.fail(f"Dataset delete failed for {dataset_id}: {exc}")


def _cleanup_gcs_model_artifacts(
    settings: Settings, *, report: CleanupReport, dry_run: bool
) -> None:
    """Delete only ARTIFACT_GCS_PREFIX (default models/). Leaves raw/train CSVs alone."""
    client = storage.Client(project=settings.gcp_project)
    prefix = settings.artifact_gcs_prefix.strip("/") + "/"
    report.note(f"GCS artifacts gs://{settings.gcs_bucket}/{prefix}")
    blobs = list(client.list_blobs(settings.gcs_bucket, prefix=prefix))
    report.note(f"Found {len(blobs)} object(s) under models prefix")
    if dry_run:
        report.note("[dry-run] would delete those GCS objects (raw CSVs untouched)")
        return
    deleted = 0
    for blob in blobs:
        try:
            blob.delete()
            deleted += 1
        except Exception as exc:
            report.fail(f"Failed deleting gs://{settings.gcs_bucket}/{blob.name}: {exc}")
    report.note(f"Deleted {deleted} GCS object(s) under {prefix}")
