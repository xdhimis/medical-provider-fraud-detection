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
    settings = Settings()  # type: ignore[call-arg]
    assert settings.gcp_project == "demo-project"
    assert settings.features_table_id == "demo-project.fraud_features.provider_features"
    assert settings.features_bq_uri == "bq://demo-project.fraud_features.provider_features"
    assert settings.artifact_uri("run1") == "gs://demo-bucket/models/run1"


def test_replica_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCP_PROJECT", "demo-project")
    monkeypatch.setenv("GCS_BUCKET", "demo-bucket")
    monkeypatch.setenv("ENDPOINT_MIN_REPLICA_COUNT", "2")
    monkeypatch.setenv("ENDPOINT_MAX_REPLICA_COUNT", "1")
    with pytest.raises(Exception):
        Settings()  # type: ignore[call-arg]
