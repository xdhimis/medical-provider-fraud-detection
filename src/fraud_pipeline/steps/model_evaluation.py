"""Import training metrics as a Vertex Model Evaluation on a Registry model.

This is Agent Platform **Model Evaluation** for a custom tabular model: attach
the metrics you already computed at train time so they appear on the model
details page. It does not run the expensive batch-predict + Dataflow pipeline.

The local ``evaluate`` CLI step remains the cheap promotion gate (metric floors).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fraud_pipeline.clients import init_vertex
from fraud_pipeline.config import Settings, get_settings
from fraud_pipeline.exceptions import ModelEvaluationError
from fraud_pipeline.resource_names import validate_model_resource_name

logger = logging.getLogger(__name__)

CLASSIFICATION_METRICS_SCHEMA = (
    "gs://google-cloud-aiplatform/schema/modelevaluation/classification_metrics_1.0.0.yaml"
)


@dataclass(frozen=True, slots=True)
class ImportedEvaluation:
    resource_name: str
    display_name: str
    model_resource_name: str
    metrics: dict[str, Any]


def classification_metrics_from_train(metrics: dict[str, Any]) -> dict[str, Any]:
    """Map ``metrics.json`` keys onto Vertex classification schema fields."""
    try:
        au_prc = float(metrics["pr_auc"])
        au_roc = float(metrics["roc_auc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelEvaluationError(
            "metrics.json must include numeric pr_auc and roc_auc for Model Evaluation"
        ) from exc

    payload: dict[str, Any] = {"auPrc": au_prc, "auRoc": au_roc}
    threshold = metrics.get("threshold")
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    f1 = metrics.get("f1")
    if all(value is not None for value in (threshold, precision, recall, f1)):
        payload["confidenceMetrics"] = [
            {
                "confidenceThreshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1Score": float(f1),
            }
        ]
    return payload


def load_metrics_payload(metrics_uri: str) -> dict[str, Any]:
    """Load metrics.json from a local path or ``gs://`` URI."""
    if metrics_uri.startswith("gs://"):
        from google.cloud import storage

        _, _, rest = metrics_uri.partition("gs://")
        bucket_name, _, blob_name = rest.partition("/")
        if not bucket_name or not blob_name:
            raise ModelEvaluationError(f"Invalid GCS metrics URI: {metrics_uri}")
        text = storage.Client().bucket(bucket_name).blob(blob_name).download_as_text()
        payload = json.loads(text)
    else:
        payload = json.loads(Path(metrics_uri).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ModelEvaluationError(
            f"metrics.json must be an object, got {type(payload)!r}"
        )
    return payload


def import_model_evaluation(
    model_resource_name: str,
    metrics: dict[str, Any],
    *,
    display_name: str = "validation",
    settings: Settings | None = None,
    client: Any | None = None,
) -> ImportedEvaluation:
    """Upload classification metrics onto a Model Registry version."""
    settings = settings or get_settings()
    model_name = validate_model_resource_name(model_resource_name)
    vertex_metrics = classification_metrics_from_train(metrics)
    init_vertex(settings)

    from google.api_core.client_options import ClientOptions
    from google.cloud import aiplatform_v1

    gapic = client or aiplatform_v1.ModelServiceClient(
        client_options=ClientOptions(
            api_endpoint=f"{settings.gcp_region}-aiplatform.googleapis.com"
        )
    )
    evaluation = aiplatform_v1.ModelEvaluation(
        display_name=display_name,
        metrics_schema_uri=CLASSIFICATION_METRICS_SCHEMA,
        metrics=vertex_metrics,
    )
    try:
        imported = gapic.import_model_evaluation(
            parent=model_name,
            model_evaluation=evaluation,
        )
    except Exception as exc:
        raise ModelEvaluationError(
            f"import_model_evaluation failed for {model_name}: {exc}"
        ) from exc

    resource_name = getattr(imported, "name", "") or ""
    logger.info(
        "Imported Model Evaluation %s on %s (display_name=%s)",
        resource_name,
        model_name,
        display_name,
    )
    return ImportedEvaluation(
        resource_name=resource_name,
        display_name=display_name,
        model_resource_name=model_name,
        metrics=vertex_metrics,
    )


def import_evaluation_from_uri(
    model_resource_name: str,
    metrics_uri: str,
    *,
    display_name: str = "validation",
    settings: Settings | None = None,
    client: Any | None = None,
) -> ImportedEvaluation:
    metrics = load_metrics_payload(metrics_uri)
    return import_model_evaluation(
        model_resource_name,
        metrics,
        display_name=display_name,
        settings=settings,
        client=client,
    )
