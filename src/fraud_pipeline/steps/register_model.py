"""Step 5: Register trained artifact in Vertex AI Model Registry."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from google.cloud import aiplatform

from fraud_pipeline.clients import init_vertex
from fraud_pipeline.config import Settings, get_settings
from fraud_pipeline.exceptions import ModelRegistryError

logger = logging.getLogger(__name__)

# Pin a prebuilt Vertex XGBoost serving image. Update when bumping xgboost.
XGBOOST_SERVING_IMAGE = (
    "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest"
)


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    resource_name: str
    display_name: str
    artifact_uri: str


def register_model(
    artifact_uri: str,
    *,
    settings: Settings | None = None,
    serving_container_image: str = XGBOOST_SERVING_IMAGE,
) -> RegisteredModel:
    """Upload a model directory (containing model.bst) to Model Registry."""
    settings = settings or get_settings()
    if not artifact_uri.startswith("gs://"):
        raise ModelRegistryError(f"artifact_uri must be a GCS path, got {artifact_uri}")

    init_vertex(settings)
    try:
        model = aiplatform.Model.upload(
            display_name=settings.model_display_name,
            artifact_uri=artifact_uri,
            serving_container_image_uri=serving_container_image,
            description="Medicare provider fraud XGBoost classifier",
            labels={
                "pipeline": "fraud-detection",
                "entity": "provider",
            },
            sync=True,
        )
    except Exception as exc:
        raise ModelRegistryError(f"Model.upload failed: {exc}") from exc

    logger.info("Registered model %s from %s", model.resource_name, artifact_uri)
    return RegisteredModel(
        resource_name=model.resource_name,
        display_name=settings.model_display_name,
        artifact_uri=artifact_uri,
    )


def latest_model_resource(settings: Settings | None = None) -> str:
    """Resolve the most recently created model with the configured display name."""
    settings = settings or get_settings()
    if settings.model_resource_name:
        return settings.model_resource_name

    init_vertex(settings)
    models = aiplatform.Model.list(
        filter=f'display_name="{settings.model_display_name}"',
        order_by="create_time desc",
    )
    if not models:
        raise ModelRegistryError(
            f"No models found with display_name={settings.model_display_name}"
        )
    return models[0].resource_name
