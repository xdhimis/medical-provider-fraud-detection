"""Step 5: Register trained artifact in Vertex AI Model Registry."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from google.cloud import aiplatform

from fraud_pipeline.clients import init_vertex
from fraud_pipeline.config import DEFAULT_XGBOOST_SERVING_IMAGE, Settings, get_settings
from fraud_pipeline.exceptions import ModelRegistryError
from fraud_pipeline.resource_names import require_parent_model_id

logger = logging.getLogger(__name__)

# Re-export for callers / tests. Prefer settings.xgboost_serving_image at runtime.
XGBOOST_SERVING_IMAGE = DEFAULT_XGBOOST_SERVING_IMAGE


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    resource_name: str
    display_name: str
    artifact_uri: str


def register_model(
    artifact_uri: str,
    *,
    settings: Settings | None = None,
    serving_container_image: str | None = None,
) -> RegisteredModel:
    """Upload a model directory (containing model.bst) to Model Registry.

    If ``MODEL_RESOURCE_NAME`` is set to a full Vertex model resource name,
    uploads as a new version of that parent model. Otherwise creates a new model.
    """
    settings = settings or get_settings()
    if not artifact_uri.startswith("gs://"):
        raise ModelRegistryError(f"artifact_uri must be a GCS path, got {artifact_uri}")

    image = serving_container_image or settings.xgboost_serving_image

    parent_model = None
    if settings.model_resource_name:
        parent_model = require_parent_model_id(settings.model_resource_name)

    init_vertex(settings)
    upload_kwargs: dict = {
        "display_name": settings.model_display_name,
        "artifact_uri": artifact_uri,
        "serving_container_image_uri": image,
        "description": "Medicare provider fraud XGBoost classifier",
        "labels": {
            "pipeline": "fraud-detection",
            "entity": "provider",
        },
        "sync": True,
    }
    if parent_model:
        upload_kwargs["parent_model"] = parent_model
        logger.info("Registering new version under parent_model=%s", parent_model)

    try:
        model = aiplatform.Model.upload(**upload_kwargs)
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
