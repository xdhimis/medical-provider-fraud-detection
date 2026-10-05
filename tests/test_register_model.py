"""Unit tests for Model Registry helpers (no live Vertex calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fraud_pipeline.config import Settings
from fraud_pipeline.exceptions import ModelRegistryError
from fraud_pipeline.resource_names import parent_model_id, require_parent_model_id
from fraud_pipeline.steps.register_model import register_model


def test_parent_model_id_strips_version() -> None:
    name = "projects/p/locations/us-central1/models/123@2"
    assert parent_model_id(name) == "projects/p/locations/us-central1/models/123"


def test_require_parent_model_id_rejects_display_name() -> None:
    with pytest.raises(ModelRegistryError, match="full Vertex resource name"):
        require_parent_model_id("provider-fraud-xgb")


def test_require_parent_model_id_rejects_partial_path() -> None:
    with pytest.raises(ModelRegistryError, match="full Vertex resource name"):
        require_parent_model_id("projects/p/locations/us-central1/models/")


def test_register_model_passes_parent_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCP_PROJECT", "demo-project")
    monkeypatch.setenv("GCS_BUCKET", "demo-bucket")
    monkeypatch.setenv(
        "MODEL_RESOURCE_NAME",
        "projects/demo-project/locations/us-central1/models/99@1",
    )
    settings = Settings()  # type: ignore[call-arg]

    fake_model = MagicMock()
    fake_model.resource_name = "projects/demo-project/locations/us-central1/models/99@2"

    with (
        patch("fraud_pipeline.steps.register_model.init_vertex"),
        patch(
            "fraud_pipeline.steps.register_model.aiplatform.Model.upload",
            return_value=fake_model,
        ) as upload,
    ):
        result = register_model("gs://demo-bucket/models/run1", settings=settings)

    assert result.resource_name.endswith("@2")
    kwargs = upload.call_args.kwargs
    assert kwargs["parent_model"] == "projects/demo-project/locations/us-central1/models/99"
    assert "xgboost-cpu.2-1" in kwargs["serving_container_image_uri"]


def test_register_model_rejects_non_gcs_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCP_PROJECT", "demo-project")
    monkeypatch.setenv("GCS_BUCKET", "demo-bucket")
    settings = Settings()  # type: ignore[call-arg]
    with pytest.raises(ModelRegistryError, match="GCS path"):
        register_model("/tmp/model", settings=settings)
