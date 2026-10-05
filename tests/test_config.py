"""Unit tests for settings loading."""

from __future__ import annotations

import pytest

from fraud_pipeline.config import Settings, reset_settings_cache


@pytest.fixture(autouse=True)
def _clear_settings() -> None:
    reset_settings_cache()
    yield
    reset_settings_cache()


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCP_PROJECT", "demo-project")
    monkeypatch.setenv("GCS_BUCKET", "demo-bucket")
    monkeypatch.setenv("GCP_REGION", "us-central1")
    monkeypatch.setenv("ARTIFACT_GCS_PREFIX", "models")
    monkeypatch.setenv("PREDICTION_THRESHOLD", "")  # blank → None (load threshold.json)
    monkeypatch.delenv("MODEL_RESOURCE_NAME", raising=False)
    monkeypatch.delenv("ENDPOINT_RESOURCE_NAME", raising=False)
    settings = Settings()  # type: ignore[call-arg]
    assert settings.gcp_project == "demo-project"
    assert settings.features_table_id == "demo-project.fraud_features.provider_features"
    assert settings.features_bq_uri == "bq://demo-project.fraud_features.provider_features"
    assert settings.artifact_uri("run1") == "gs://demo-bucket/models/run1"
    assert settings.prediction_threshold is None
    assert "xgboost-cpu.2-1" in settings.xgboost_serving_image


def test_replica_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCP_PROJECT", "demo-project")
    monkeypatch.setenv("GCS_BUCKET", "demo-bucket")
    monkeypatch.setenv("ENDPOINT_MIN_REPLICA_COUNT", "2")
    monkeypatch.setenv("ENDPOINT_MAX_REPLICA_COUNT", "1")
    with pytest.raises(Exception):
        Settings()  # type: ignore[call-arg]


def test_model_resource_name_fail_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCP_PROJECT", "demo-project")
    monkeypatch.setenv("GCS_BUCKET", "demo-bucket")
    monkeypatch.setenv("MODEL_RESOURCE_NAME", "provider-fraud-xgb")
    with pytest.raises(Exception, match="full Vertex resource name"):
        Settings()  # type: ignore[call-arg]


def test_endpoint_resource_name_fail_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCP_PROJECT", "demo-project")
    monkeypatch.setenv("GCS_BUCKET", "demo-bucket")
    monkeypatch.setenv("ENDPOINT_RESOURCE_NAME", "projects/p/locations/us-central1/endpoints/")
    with pytest.raises(Exception, match="full Vertex resource name"):
        Settings()  # type: ignore[call-arg]


def test_valid_resource_names_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCP_PROJECT", "demo-project")
    monkeypatch.setenv("GCS_BUCKET", "demo-bucket")
    monkeypatch.setenv(
        "MODEL_RESOURCE_NAME",
        "projects/demo-project/locations/us-central1/models/123@2",
    )
    monkeypatch.setenv(
        "ENDPOINT_RESOURCE_NAME",
        "projects/demo-project/locations/us-central1/endpoints/456",
    )
    settings = Settings()  # type: ignore[call-arg]
    assert settings.model_resource_name.endswith("models/123@2")
    assert settings.endpoint_resource_name.endswith("endpoints/456")
