"""Unit tests for POC cleanup helpers (no live GCP calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fraud_pipeline.config import Settings, reset_settings_cache
from fraud_pipeline.steps.cleanup import CleanupReport, cleanup_poc


@pytest.fixture(autouse=True)
def _clear_settings() -> None:
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("GCP_PROJECT", "demo-project")
    monkeypatch.setenv("GCS_BUCKET", "demo-bucket")
    monkeypatch.setenv("ENDPOINT_DISPLAY_NAME", "provider-fraud-endpoint")
    monkeypatch.setenv("MODEL_DISPLAY_NAME", "provider-fraud-xgb")
    monkeypatch.setenv("FEATURE_ONLINE_STORE", "fraud_provider_store")
    return Settings()  # type: ignore[call-arg]


def test_cleanup_dry_run_does_not_call_delete(settings: Settings) -> None:
    fake_endpoint = MagicMock()
    fake_endpoint.resource_name = "projects/p/locations/l/endpoints/e"
    fake_endpoint.gca_resource.deployed_models = [MagicMock(id="dm1")]

    with (
        patch("fraud_pipeline.steps.cleanup.init_vertex"),
        patch(
            "fraud_pipeline.steps.cleanup.aiplatform.Endpoint.list",
            return_value=[fake_endpoint],
        ),
        patch(
            "fraud_pipeline.steps.cleanup.aiplatform.Model.list",
            return_value=[],
        ),
        patch(
            "fraud_pipeline.steps.cleanup.feature_store",
            create=True,
        ),
    ):
        # Patch FeatureOnlineStore import path used inside the function
        with patch.dict("sys.modules", {}):
            with patch(
                "vertexai.resources.preview.feature_store.FeatureOnlineStore",
                MagicMock(),
            ):
                report = cleanup_poc(settings=settings, delete_data=False, dry_run=True)

    assert isinstance(report, CleanupReport)
    assert report.dry_run is True
    fake_endpoint.undeploy.assert_not_called()
    fake_endpoint.delete.assert_not_called()
    assert any("dry-run" in a for a in report.actions)
    assert any("Skipped BigQuery/GCS" in a for a in report.actions)
