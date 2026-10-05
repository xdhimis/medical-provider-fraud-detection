"""Unit tests for artifact contract + predict threshold resolution."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fraud_pipeline.artifact_contract import (
    assert_feature_columns_contract,
    load_threshold_json,
    parse_threshold_payload,
    resolve_prediction_threshold,
)
from fraud_pipeline.exceptions import PredictionError, TrainingError
from fraud_pipeline.schemas import FEATURE_COLUMNS
from fraud_pipeline.steps.predict import PredictionResult, _extract_probability, predict_provider


def test_assert_feature_columns_contract_ok() -> None:
    assert assert_feature_columns_contract(FEATURE_COLUMNS) == FEATURE_COLUMNS


def test_assert_feature_columns_contract_mismatch() -> None:
    with pytest.raises(TrainingError, match="contract mismatch"):
        assert_feature_columns_contract(FEATURE_COLUMNS[:-1])


def test_parse_and_load_threshold_json(tmp_path: Path) -> None:
    path = tmp_path / "threshold.json"
    path.write_text(json.dumps({"threshold": 0.74}), encoding="utf-8")
    assert load_threshold_json(path) == 0.74
    assert parse_threshold_payload({"threshold": 0.5}) == 0.5


def test_resolve_threshold_precedence() -> None:
    assert resolve_prediction_threshold(
        override=0.9, env_threshold=0.7, threshold_gcs_uri=None
    ) == 0.9
    assert resolve_prediction_threshold(
        override=None, env_threshold=0.7, threshold_gcs_uri=None
    ) == 0.7
    assert resolve_prediction_threshold(
        override=None, env_threshold=None, threshold_gcs_uri=None
    ) == 0.5


def test_extract_probability_shapes() -> None:
    assert _extract_probability(0.8) == 0.8
    assert _extract_probability([0.8]) == 0.8
    assert _extract_probability([0.2, 0.8]) == 0.8


def test_predict_provider_instance_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCP_PROJECT", "demo-project")
    monkeypatch.setenv("GCS_BUCKET", "demo-bucket")
    monkeypatch.setenv("PREDICTION_THRESHOLD", "0.74")

    from fraud_pipeline.config import Settings, reset_settings_cache

    reset_settings_cache()
    settings = Settings()  # type: ignore[call-arg]

    fake_features = MagicMock()
    fake_features.as_instance.return_value = [float(i) for i in range(len(FEATURE_COLUMNS))]

    fake_endpoint = MagicMock()
    fake_endpoint.resource_name = (
        "projects/demo-project/locations/us-central1/endpoints/1"
    )
    fake_endpoint.predict.return_value = MagicMock(predictions=[[0.1, 0.9]])

    with (
        patch(
            "fraud_pipeline.steps.predict.FeatureOnlineClient"
        ) as client_cls,
        patch(
            "fraud_pipeline.steps.predict._resolve_endpoint",
            return_value=fake_endpoint,
        ),
    ):
        client_cls.return_value.fetch.return_value = fake_features
        result = predict_provider("PRV55912", settings=settings)

    assert isinstance(result, PredictionResult)
    assert result.threshold == 0.74
    assert result.is_fraud is True
    assert len(result.feature_vector) == len(FEATURE_COLUMNS)
    assert all(isinstance(x, float) for x in result.feature_vector)
    fake_endpoint.predict.assert_called_once()
    instances = fake_endpoint.predict.call_args.kwargs["instances"]
    assert len(instances) == 1
    assert len(instances[0]) == len(FEATURE_COLUMNS)


def test_parse_threshold_rejects_out_of_range() -> None:
    with pytest.raises(PredictionError, match="out of"):
        parse_threshold_payload({"threshold": 1.5})
