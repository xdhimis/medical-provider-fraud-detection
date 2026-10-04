"""Guardrails for feature SQL packaging and load mapping."""

from __future__ import annotations

from importlib import resources

from fraud_pipeline.schemas import FEATURE_COLUMNS, GCS_FILE_TO_TABLE
from fraud_pipeline.steps.load_raw import _map_uri_to_table


def test_provider_features_sql_is_packaged() -> None:
    text = (
        resources.files("fraud_pipeline.sql")
        .joinpath("provider_features.sql")
        .read_text(encoding="utf-8")
    )
    assert "provider_features" in text
    assert "{project}" in text
    assert "{raw_dataset}" in text
    for col in FEATURE_COLUMNS:
        assert col in text, f"missing feature column in SQL: {col}"


def test_gcs_mapping_prefers_specific_files() -> None:
    assert (
        _map_uri_to_table("gs://b/raw/train/Train_Beneficiarydata-123.csv")
        == "beneficiary"
    )
    assert _map_uri_to_table("gs://b/raw/train/Train_Inpatientdata-123.csv") == "inpatient"
    assert (
        _map_uri_to_table("gs://b/raw/train/Train_Outpatientdata-123.csv") == "outpatient"
    )
    assert _map_uri_to_table("gs://b/raw/train/Train-1542865627584.csv") == "labels"
    assert _map_uri_to_table("gs://b/raw/train/readme.txt") is None
    assert set(GCS_FILE_TO_TABLE.values()) == {
        "labels",
        "beneficiary",
        "inpatient",
        "outpatient",
    }
