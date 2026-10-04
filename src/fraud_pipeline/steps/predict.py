"""Step 7: Online prediction = Feature Store lookup -> Endpoint.predict.

This is the production serving path:

  provider_id
    --(FeatureOnlineClient.fetch)--> feature vector (FEATURE_COLUMNS order)
    --(Endpoint.predict)----------> fraud probability / label
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

from google.cloud import aiplatform

from fraud_pipeline.clients import init_vertex
from fraud_pipeline.config import Settings, get_settings
from fraud_pipeline.exceptions import PredictionError
from fraud_pipeline.schemas import FEATURE_COLUMNS
from fraud_pipeline.serving.feature_client import FeatureOnlineClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PredictionResult:
    provider_id: str
    probability: float
    is_fraud: bool
    threshold: float
    feature_vector: list[float]
    endpoint: str


def _resolve_endpoint(settings: Settings) -> aiplatform.Endpoint:
    init_vertex(settings)
    if settings.endpoint_resource_name:
        return aiplatform.Endpoint(settings.endpoint_resource_name)

    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{settings.endpoint_display_name}"',
        order_by="create_time desc",
    )
    if not endpoints:
        raise PredictionError(
            f"No endpoint found with display_name={settings.endpoint_display_name}"
        )
    return endpoints[0]


def _extract_probability(prediction: object) -> float:
    """Normalize Vertex predict() response shapes to a fraud probability."""
    if isinstance(prediction, (int, float)):
        return float(prediction)
    if isinstance(prediction, (list, tuple)):
        if not prediction:
            raise PredictionError("Empty prediction vector from endpoint")
        # Binary logistic: [prob_0, prob_1] or [prob_1]
        if len(prediction) == 1:
            return float(prediction[0])
        if len(prediction) == 2:
            return float(prediction[1])
        return float(prediction[0])
    raise PredictionError(f"Unsupported prediction type: {type(prediction)!r}")


def predict_provider(
    provider_id: str,
    *,
    settings: Settings | None = None,
    threshold: float | None = None,
) -> PredictionResult:
    """Fetch online features for a provider and score via the deployed endpoint."""
    settings = settings or get_settings()
    thr = settings.prediction_threshold if threshold is None else threshold

    features = FeatureOnlineClient(settings).fetch(provider_id)
    # Dense vector in training column order. Model.bst is saved without
    # feature names so the Vertex XGBoost container accepts a 2D matrix.
    instance = features.as_instance(FEATURE_COLUMNS)

    endpoint = _resolve_endpoint(settings)
    try:
        response = endpoint.predict(instances=[instance])
    except Exception as exc:
        raise PredictionError(f"endpoint.predict failed: {exc}") from exc

    if not response.predictions:
        raise PredictionError("Endpoint returned no predictions")

    probability = _extract_probability(response.predictions[0])
    result = PredictionResult(
        provider_id=provider_id,
        probability=probability,
        is_fraud=probability >= thr,
        threshold=thr,
        feature_vector=instance,
        endpoint=endpoint.resource_name,
    )
    logger.info("Prediction: %s", json.dumps(asdict(result), sort_keys=True))
    return result
