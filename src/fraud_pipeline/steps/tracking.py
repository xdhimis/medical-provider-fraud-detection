"""Vertex Experiments + ML Metadata helpers (no endpoint required)."""

from __future__ import annotations

import logging
import re
from typing import Any

from google.cloud import aiplatform

from fraud_pipeline.clients import init_vertex
from fraud_pipeline.config import Settings, get_settings
from fraud_pipeline.schemas import TrainingArtifacts

logger = logging.getLogger(__name__)


def experiment_run_name(run_id: str) -> str:
    """Vertex experiment runs: lowercase letters, digits, hyphens; max 128 chars."""
    cleaned = re.sub(r"[^a-z0-9-]", "-", run_id.lower()).strip("-")
    return (cleaned or "run")[:127]


def scalar_metrics(metrics: dict[str, Any]) -> dict[str, float | int | str]:
    """Keep only values Vertex ``log_metrics`` / ``log_params`` accept."""
    out: dict[str, float | int | str] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[key] = value
        elif isinstance(value, str):
            out[key] = value
    return out


def run_id_from_artifact_uri(artifact_uri: str) -> str:
    return artifact_uri.rstrip("/").rsplit("/", 1)[-1]


def log_training_run(
    artifacts: TrainingArtifacts,
    metrics: dict[str, Any],
    *,
    settings: Settings | None = None,
) -> str | None:
    """Start an experiment run, log params/metrics, record ML Metadata lineage."""
    settings = settings or get_settings()
    if not settings.vertex_experiment:
        logger.info("VERTEX_EXPERIMENT unset; skipping experiment logging")
        return None

    try:
        init_vertex(settings)
    except ValueError as exc:
        raise ValueError(
            f"VERTEX_EXPERIMENT={settings.vertex_experiment!r} conflicts with an "
            "existing Metadata context (often the pipeline named fraud-detection). "
            "Set VERTEX_EXPERIMENT=fraud-xgb (or another unused name)."
        ) from exc
    run_name = experiment_run_name(artifacts.run_id)
    aiplatform.start_run(run_name)
    try:
        aiplatform.log_params(
            {
                "run_id": artifacts.run_id,
                "gcs_uri": artifacts.gcs_uri,
                "features_table": settings.features_table_id,
                "train_test_size": settings.train_test_size,
                "train_random_seed": settings.train_random_seed,
            }
        )
        aiplatform.log_metrics(scalar_metrics(metrics))
    finally:
        # Close the experiment run before extra Metadata RPCs. Nested
        # Execution.create inside start_run can 503 (gRPC auth plugin).
        aiplatform.end_run()
    try:
        _record_train_lineage(settings, artifacts, metrics)
    except Exception as exc:
        logger.warning("ML Metadata lineage skipped: %s", exc)
    logger.info(
        "Logged experiment=%s run=%s",
        settings.vertex_experiment,
        run_name,
    )
    return run_name


def log_registry_model(
    artifact_uri: str,
    model_resource_name: str,
    *,
    settings: Settings | None = None,
) -> None:
    """Resume the train experiment run and attach the Registry resource name."""
    settings = settings or get_settings()
    if not settings.vertex_experiment:
        return
    init_vertex(settings)
    run_name = experiment_run_name(run_id_from_artifact_uri(artifact_uri))
    try:
        aiplatform.start_run(run_name, resume=True)
        try:
            aiplatform.log_params({"model_resource_name": model_resource_name})
        finally:
            aiplatform.end_run()
    except Exception as exc:
        logger.warning("Experiment registry param skipped: %s", exc)


def list_experiment_rows(*, settings: Settings | None = None) -> list[dict[str, Any]]:
    """Return experiment runs as dicts for CLI / tests."""
    settings = settings or get_settings()
    if not settings.vertex_experiment:
        return []
    init_vertex(settings)
    frame = aiplatform.get_experiment_df(
        experiment=settings.vertex_experiment,
        include_time_series=False,
    )
    return frame.to_dict(orient="records") if not frame.empty else []


def _record_train_lineage(
    settings: Settings,
    artifacts: TrainingArtifacts,
    metrics: dict[str, Any],
) -> None:
    """Link BQ features → train execution → GCS model + metrics artifacts."""
    from google.cloud.aiplatform_v1.types.execution import Execution as GcaExecution

    execution = aiplatform.Execution.create(
        schema_title="system.ContainerExecution",
        display_name=f"train-{artifacts.run_id}",
        metadata={"run_id": artifacts.run_id, "gcs_uri": artifacts.gcs_uri},
    )
    dataset = aiplatform.Artifact.create(
        schema_title="system.Dataset",
        uri=settings.features_bq_uri,
        display_name="provider_features",
        metadata={"table": settings.features_table_id},
    )
    model_art = aiplatform.Artifact.create(
        schema_title="system.Model",
        uri=artifacts.gcs_uri,
        display_name="train-artifacts",
        metadata={"run_id": artifacts.run_id},
    )
    metrics_art = aiplatform.Artifact.create(
        schema_title="system.Artifact",
        uri=artifacts.metrics_uri,
        display_name="metrics.json",
        metadata=scalar_metrics(metrics),
    )
    execution.assign_input_artifacts([dataset])
    execution.assign_output_artifacts([model_art, metrics_art])
    execution.update(state=GcaExecution.State.COMPLETE)
