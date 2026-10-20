"""Submit training as a Vertex Custom Job (same ``train_local`` inside the image)."""

from __future__ import annotations

import logging

from google.cloud import aiplatform

from fraud_pipeline.clients import init_vertex
from fraud_pipeline.config import Settings, get_settings
from fraud_pipeline.exceptions import TrainingError
from fraud_pipeline.schemas import TrainingArtifacts
from fraud_pipeline.steps.train import _new_run_id

logger = logging.getLogger(__name__)

_TRAIN_COMMAND = ["fraud-pipeline", "train", "--local"]


def build_worker_pool_spec(
    settings: Settings,
    *,
    image_uri: str,
    run_id: str,
    machine_type: str,
) -> dict:
    """Worker spec that runs in-process training. Never calls ``--vertex``."""
    env = settings.as_env()
    env["FRAUD_RUN_ID"] = run_id
    return {
        "machine_spec": {"machine_type": machine_type},
        "replica_count": 1,
        "container_spec": {
            "image_uri": image_uri,
            "command": list(_TRAIN_COMMAND),
            "env": [{"name": key, "value": value} for key, value in sorted(env.items())],
        },
    }


def train_vertex(
    settings: Settings | None = None,
    *,
    image_uri: str | None = None,
    machine_type: str | None = None,
    service_account: str | None = None,
) -> TrainingArtifacts:
    """Submit a Custom Job and wait. Artifacts land at the pre-assigned ``run_id``."""
    settings = settings or get_settings()
    image = (image_uri or settings.pipeline_image or "").strip()
    if not image:
        raise TrainingError(
            "Set PIPELINE_IMAGE (or pass --image) to the Artifact Registry image "
            "before train --vertex"
        )
    machine = machine_type or settings.train_machine_type
    account = service_account if service_account is not None else settings.vertex_service_account
    run_id = _new_run_id()
    gcs_uri = settings.artifact_uri(run_id)

    init_vertex(settings)
    job = aiplatform.CustomJob(
        display_name=f"fraud-train-{run_id}",
        worker_pool_specs=[
            build_worker_pool_spec(
                settings,
                image_uri=image,
                run_id=run_id,
                machine_type=machine,
            )
        ],
        base_output_dir=gcs_uri,
        staging_bucket=f"gs://{settings.gcs_bucket}",
        project=settings.gcp_project,
        location=settings.gcp_region,
    )
    logger.info("Submitting Custom Job image=%s run_id=%s", image, run_id)
    try:
        job.run(service_account=account or None, sync=True)
    except Exception as exc:
        raise TrainingError(f"Vertex Custom Job failed for run_id={run_id}: {exc}") from exc

    logger.info("Custom Job finished artifacts=%s", gcs_uri)
    return TrainingArtifacts(run_id=run_id, gcs_uri=gcs_uri)
