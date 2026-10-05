"""Unit tests for FeatureView payload flattening and instance shape."""

from __future__ import annotations

import pytest

from fraud_pipeline.exceptions import PredictionError
from fraud_pipeline.schemas import FEATURE_COLUMNS
from fraud_pipeline.serving.feature_client import (
    ProviderFeatures,
    _normalize_feature_response,
)


def test_normalize_key_values_features_list() -> None:
    response = {
        "features": [
            {"name": "claim_count", "value": {"double_value": 12.0}},
            {"name": "avg_reimbursed", "value": {"doubleValue": 100.5}},
            {"name": "provider_id", "value": {"string_value": "PRV1"}},
        ]
    }
    flat = _normalize_feature_response(response, provider_id="PRV1")
    assert flat["claim_count"] == 12.0
    assert flat["avg_reimbursed"] == 100.5
    assert flat["provider_id"] == "PRV1"


def test_normalize_already_flat_mapping() -> None:
    response = {"claim_count": 3, "total_reimbursed": 50.0}
    flat = _normalize_feature_response(response, provider_id="PRV1")
    assert flat["claim_count"] == 3
    assert flat["total_reimbursed"] == 50.0


def test_as_instance_is_dense_float_list_in_feature_order() -> None:
    values = {name: float(i) for i, name in enumerate(FEATURE_COLUMNS)}
    features = ProviderFeatures(provider_id="PRV55912", values=values)
    instance = features.as_instance(FEATURE_COLUMNS)
    assert isinstance(instance, list)
    assert len(instance) == len(FEATURE_COLUMNS)
    assert all(isinstance(x, float) for x in instance)
    assert instance[0] == 0.0
    assert instance[-1] == float(len(FEATURE_COLUMNS) - 1)


def test_as_instance_raises_on_missing_columns() -> None:
    features = ProviderFeatures(provider_id="PRV1", values={"claim_count": 1.0})
    with pytest.raises(PredictionError, match="missing columns"):
        features.as_instance(FEATURE_COLUMNS)
