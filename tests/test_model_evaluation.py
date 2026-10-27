"""Unit tests for Vertex Model Evaluation import (no live GCP)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fraud_pipeline.config import Settings
from fraud_pipeline.exceptions import ModelEvaluationError
from fraud_pipeline.steps.model_evaluation import (
    CLASSIFICATION_METRICS_SCHEMA,
    classification_metrics_from_train,
    import_evaluation_from_uri,
    import_model_evaluation,
)


def _settings() -> Settings:
    return Settings(gcp_project="demo-project", gcs_bucket="demo-bucket")  # type: ignore[call-arg]


def test_classification_metrics_mapping() -> None:
    mapped = classification_metrics_from_train(
        {
            "pr_auc": 0.81,
            "roc_auc": 0.9,
            "threshold": 0.74,
            "precision": 0.5,
            "recall": 0.4,
            "f1": 0.44,
        }
    )
    assert mapped["auPrc"] == pytest.approx(0.81)
    assert mapped["auRoc"] == pytest.approx(0.9)
    assert mapped["confidenceMetrics"][0]["f1Score"] == pytest.approx(0.44)
    assert mapped["confidenceMetrics"][0]["confidenceThreshold"] == pytest.approx(0.74)


def test_classification_metrics_require_auc() -> None:
    with pytest.raises(ModelEvaluationError, match="pr_auc"):
        classification_metrics_from_train({"f1": 0.4})


def test_import_model_evaluation_calls_gapic(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = MagicMock()
    fake_client.import_model_evaluation.return_value = MagicMock(
        name="projects/p/locations/us-central1/models/99/evaluations/e1"
    )
    # MagicMock.name is special; set via configure
    imported = MagicMock()
    imported.name = "projects/p/locations/us-central1/models/99/evaluations/e1"
    fake_client.import_model_evaluation.return_value = imported

    monkeypatch.setattr(
        "fraud_pipeline.steps.model_evaluation.init_vertex",
        lambda settings: None,
    )

    result = import_model_evaluation(
        "projects/p/locations/us-central1/models/99",
        {
            "pr_auc": 0.5,
            "roc_auc": 0.8,
            "threshold": 0.5,
            "precision": 0.4,
            "recall": 0.3,
            "f1": 0.34,
        },
        display_name="validation",
        settings=_settings(),
        client=fake_client,
    )
    assert result.resource_name.endswith("/evaluations/e1")
    kwargs = fake_client.import_model_evaluation.call_args.kwargs
    assert kwargs["parent"].endswith("/models/99")
    evaluation = kwargs["model_evaluation"]
    assert evaluation.display_name == "validation"
    assert evaluation.metrics_schema_uri == CLASSIFICATION_METRICS_SCHEMA
    assert evaluation.metrics["auPrc"] == pytest.approx(0.5)


def test_import_evaluation_from_local_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"pr_auc": 0.6, "roc_auc": 0.85}))
    fake_client = MagicMock()
    imported = MagicMock()
    imported.name = "projects/p/locations/us-central1/models/1/evaluations/e2"
    fake_client.import_model_evaluation.return_value = imported
    monkeypatch.setattr(
        "fraud_pipeline.steps.model_evaluation.init_vertex",
        lambda settings: None,
    )
    result = import_evaluation_from_uri(
        "projects/p/locations/us-central1/models/1",
        str(path),
        settings=_settings(),
        client=fake_client,
    )
    assert "evaluations/e2" in result.resource_name
    assert result.metrics["auRoc"] == pytest.approx(0.85)
