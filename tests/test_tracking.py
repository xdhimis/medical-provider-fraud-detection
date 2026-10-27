"""Unit tests for Experiments, lineage helpers, and Registry aliases."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fraud_pipeline.config import Settings
from fraud_pipeline.schemas import TrainingArtifacts
from fraud_pipeline.steps.register_model import assign_version_alias
from fraud_pipeline.steps.tracking import (
    experiment_run_name,
    list_experiment_rows,
    log_training_run,
    run_id_from_artifact_uri,
    scalar_metrics,
)


def _settings(**kwargs) -> Settings:
    data = {"gcp_project": "demo-project", "gcs_bucket": "demo-bucket"}
    data.update(kwargs)
    return Settings(**data)  # type: ignore[arg-type]


def test_experiment_run_name_sanitizes() -> None:
    assert experiment_run_name("20261008T000000Z-Abcd1234") == "20261008t000000z-abcd1234"


def test_scalar_metrics_drops_bools() -> None:
    assert scalar_metrics({"pr_auc": 0.5, "ok": True, "note": "x"}) == {
        "pr_auc": 0.5,
        "note": "x",
    }


def test_run_id_from_artifact_uri() -> None:
    assert run_id_from_artifact_uri("gs://b/models/run-1/") == "run-1"


def test_log_training_run_skips_when_experiment_unset() -> None:
    settings = _settings(vertex_experiment=None)
    artifacts = TrainingArtifacts(run_id="r1", gcs_uri="gs://b/models/r1")
    assert log_training_run(artifacts, {"pr_auc": 0.4}, settings=settings) is None


def test_log_training_run_logs_and_records_lineage() -> None:
    settings = _settings()
    artifacts = TrainingArtifacts(run_id="20261008T1Z-abcd", gcs_uri="gs://b/models/rid")
    execution = MagicMock()
    with (
        patch("fraud_pipeline.steps.tracking.init_vertex") as init,
        patch("fraud_pipeline.steps.tracking.aiplatform.start_run") as start,
        patch("fraud_pipeline.steps.tracking.aiplatform.log_params") as params,
        patch("fraud_pipeline.steps.tracking.aiplatform.log_metrics") as metrics,
        patch("fraud_pipeline.steps.tracking.aiplatform.end_run") as end,
        patch("fraud_pipeline.steps.tracking.aiplatform.Execution.create", return_value=execution),
        patch("fraud_pipeline.steps.tracking.aiplatform.Artifact.create", return_value=MagicMock()),
    ):
        name = log_training_run(artifacts, {"pr_auc": 0.7, "f1": 0.4}, settings=settings)

    init.assert_called_once()
    start.assert_called_once()
    params.assert_called_once()
    metrics.assert_called_once()
    end.assert_called_once()
    execution.assign_input_artifacts.assert_called_once()
    execution.assign_output_artifacts.assert_called_once()
    assert name == experiment_run_name(artifacts.run_id)
    assert end.call_count == 1


def test_log_training_run_survives_lineage_failure() -> None:
    settings = _settings()
    artifacts = TrainingArtifacts(run_id="r2", gcs_uri="gs://b/models/r2")
    with (
        patch("fraud_pipeline.steps.tracking.init_vertex"),
        patch("fraud_pipeline.steps.tracking.aiplatform.start_run"),
        patch("fraud_pipeline.steps.tracking.aiplatform.log_params"),
        patch("fraud_pipeline.steps.tracking.aiplatform.log_metrics"),
        patch("fraud_pipeline.steps.tracking.aiplatform.end_run"),
        patch(
            "fraud_pipeline.steps.tracking.aiplatform.Execution.create",
            side_effect=RuntimeError("503 before_request"),
        ),
    ):
        name = log_training_run(artifacts, {"pr_auc": 0.7}, settings=settings)
    assert name == experiment_run_name(artifacts.run_id)


def test_assign_version_alias() -> None:
    fake_model = MagicMock()
    fake_model.version_id = "2"
    settings = _settings()
    with (
        patch("fraud_pipeline.steps.register_model.init_vertex"),
        patch(
            "fraud_pipeline.steps.register_model.aiplatform.Model",
            return_value=fake_model,
        ),
    ):
        alias = assign_version_alias(
            "projects/p/locations/us-central1/models/9@2",
            "champion",
            settings=settings,
        )
    assert alias == "champion"
    fake_model.versioning_registry.add_version_aliases.assert_called_once_with(
        ["champion"], "2"
    )


def test_list_experiment_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame([{"run_name": "r1", "metric.pr_auc": 0.5}])
    monkeypatch.setattr("fraud_pipeline.steps.tracking.init_vertex", lambda settings: None)
    monkeypatch.setattr(
        "fraud_pipeline.steps.tracking.aiplatform.get_experiment_df",
        lambda experiment, include_time_series=False: frame,
    )
    rows = list_experiment_rows(settings=_settings())
    assert rows[0]["run_name"] == "r1"
