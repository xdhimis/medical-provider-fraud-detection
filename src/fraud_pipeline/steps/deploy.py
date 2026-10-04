"""Step 6: Create Endpoint and deploy the registered model."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from google.cloud import aiplatform

from fraud_pipeline.clients import init_vertex
from fraud_pipeline.config import Settings, get_settings
from fraud_pipeline.exceptions import DeploymentError
from fraud_pipeline.steps.register_model import latest_model_resource

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DeployResult:
    endpoint_resource_name: str
    model_resource_name: str
    display_name: str


def _get_or_create_endpoint(settings: Settings) -> aiplatform.Endpoint:
    if settings.endpoint_resource_name:
        return aiplatform.Endpoint(settings.endpoint_resource_name)

    existing = aiplatform.Endpoint.list(
        filter=f'display_name="{settings.endpoint_display_name}"',
        order_by="create_time desc",
    )
    if existing:
        logger.info("Reusing endpoint %s", existing[0].resource_name)
        return existing[0]

    logger.info("Creating endpoint %s", settings.endpoint_display_name)
    return aiplatform.Endpoint.create(display_name=settings.endpoint_display_name)


def deploy_model(
    model_resource_name: str | None = None,
    *,
    settings: Settings | None = None,
) -> DeployResult:
    """Deploy (or redeploy) the model onto a dedicated endpoint."""
    settings = settings or get_settings()
    init_vertex(settings)

    model_name = model_resource_name or latest_model_resource(settings)
    try:
        model = aiplatform.Model(model_name)
        endpoint = _get_or_create_endpoint(settings)
        endpoint.deploy(
            model=model,
            deployed_model_display_name=settings.model_display_name,
            machine_type=settings.endpoint_machine_type,
            min_replica_count=settings.endpoint_min_replica_count,
            max_replica_count=settings.endpoint_max_replica_count,
            traffic_percentage=100,
            sync=True,
        )
    except Exception as exc:
        raise DeploymentError(f"Endpoint deploy failed: {exc}") from exc

    logger.info(
        "Deployed model %s to endpoint %s",
        model_name,
        endpoint.resource_name,
    )
    return DeployResult(
        endpoint_resource_name=endpoint.resource_name,
        model_resource_name=model_name,
        display_name=settings.endpoint_display_name,
    )


def undeploy_all(*, settings: Settings | None = None) -> str:
    """Undeploy all models from the configured endpoint (stops serving cost)."""
    settings = settings or get_settings()
    init_vertex(settings)
    endpoint = _get_or_create_endpoint(settings)
    for deployed in list(endpoint.gca_resource.deployed_models):
        logger.info("Undeploying %s from %s", deployed.id, endpoint.resource_name)
        endpoint.undeploy(deployed_model_id=deployed.id, sync=True)
    return endpoint.resource_name
