"""Typed settings loaded from environment / .env (secrets stay out of git)."""

from __future__ import annotations

from functools import lru_cache
from typing import Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from fraud_pipeline.resource_names import (
    validate_endpoint_resource_name,
    validate_model_resource_name,
)

# Version-locked Vertex prebuilt image (XGBoost 2.1 / Python 3.10).
# Google only publishes `:latest` per framework line — override with a digest pin:
#   gcloud container images describe \
#     us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest \
#     --format='value(image_summary.digest)'
# then set XGBOOST_SERVING_IMAGE=...@sha256:...
DEFAULT_XGBOOST_SERVING_IMAGE = (
    "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest"
)


class Settings(BaseSettings):
    """Runtime configuration for the fraud pipeline.

    Non-secret project settings come from env vars (or a local `.env`).
    Credentials are never loaded from this file — use ADC / WIF.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    gcp_project: str = Field(..., min_length=1, description="GCP project id")
    gcp_region: str = Field(default="us-central1", min_length=1)

    gcs_bucket: str = Field(..., min_length=1)
    gcs_raw_prefix: str = Field(default="raw/train")

    bq_dataset_raw: str = Field(default="fraud_raw")
    bq_dataset_features: str = Field(default="fraud_features")
    bq_location: str = Field(default="US", description="BigQuery dataset location")

    feature_online_store: str = Field(default="fraud_provider_store")
    feature_view: str = Field(default="provider_fraud_features")

    model_display_name: str = Field(default="provider-fraud-xgb")
    endpoint_display_name: str = Field(default="provider-fraud-endpoint")
    endpoint_machine_type: str = Field(default="n1-standard-4")
    endpoint_min_replica_count: int = Field(default=1, ge=1)
    endpoint_max_replica_count: int = Field(default=1, ge=1)

    model_resource_name: str | None = Field(
        default=None,
        description="Optional full Model resource name; auto-resolved if empty",
    )
    endpoint_resource_name: str | None = Field(
        default=None,
        description="Optional full Endpoint resource name; auto-resolved if empty",
    )

    artifact_gcs_prefix: str = Field(default="models")
    train_test_size: float = Field(default=0.2, gt=0.0, lt=0.5)
    train_random_seed: int = Field(default=42)
    # When unset, predict loads threshold.json via THRESHOLD_GCS_URI (or defaults to 0.5).
    prediction_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold_gcs_uri: str | None = Field(
        default=None,
        description="Optional gs://…/threshold.json (or artifact dir) for predict default",
    )
    xgboost_serving_image: str = Field(default=DEFAULT_XGBOOST_SERVING_IMAGE)

    log_level: str = Field(default="INFO")

    @field_validator(
        "gcp_project",
        "gcp_region",
        "gcs_bucket",
        "gcs_raw_prefix",
        "bq_dataset_raw",
        "bq_dataset_features",
        "feature_online_store",
        "feature_view",
        "model_display_name",
        "endpoint_display_name",
        "xgboost_serving_image",
        mode="before",
    )
    @classmethod
    def _strip_nonempty(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("must not be empty")
        return value

    @field_validator(
        "model_resource_name",
        "endpoint_resource_name",
        "threshold_gcs_uri",
        "prediction_threshold",
        mode="before",
    )
    @classmethod
    def _empty_str_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("model_resource_name")
    @classmethod
    def _validate_model_resource(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_model_resource_name(value)

    @field_validator("endpoint_resource_name")
    @classmethod
    def _validate_endpoint_resource(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_endpoint_resource_name(value)

    @field_validator("threshold_gcs_uri")
    @classmethod
    def _validate_threshold_uri(cls, value: str | None) -> str | None:
        if value is None:
            return None
        uri = value.strip()
        if not uri.startswith("gs://"):
            raise ValueError("THRESHOLD_GCS_URI must start with gs://")
        return uri

    @field_validator("xgboost_serving_image")
    @classmethod
    def _validate_serving_image(cls, value: str) -> str:
        # Prefer digest pins (…@sha256:…). Version-locked :latest on xgboost-cpu.2-1
        # is allowed because Google does not publish dated tags for this family.
        if "@sha256:" in value:
            return value
        if "xgboost-cpu.2-1" not in value:
            raise ValueError(
                "XGBOOST_SERVING_IMAGE should target xgboost-cpu.2-1 "
                f"(or a digest pin); got {value!r}"
            )
        return value

    @model_validator(mode="after")
    def _replicas_consistent(self) -> Self:
        if self.endpoint_max_replica_count < self.endpoint_min_replica_count:
            raise ValueError(
                "ENDPOINT_MAX_REPLICA_COUNT must be >= ENDPOINT_MIN_REPLICA_COUNT"
            )
        return self

    @property
    def raw_uri_prefix(self) -> str:
        prefix = self.gcs_raw_prefix.strip("/")
        return f"gs://{self.gcs_bucket}/{prefix}"

    @property
    def features_table_id(self) -> str:
        return f"{self.gcp_project}.{self.bq_dataset_features}.provider_features"

    @property
    def features_bq_uri(self) -> str:
        return f"bq://{self.features_table_id}"

    def artifact_uri(self, run_id: str) -> str:
        prefix = self.artifact_gcs_prefix.strip("/")
        return f"gs://{self.gcs_bucket}/{prefix}/{run_id}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton for a process."""
    return Settings()  # type: ignore[call-arg]


def reset_settings_cache() -> None:
    """Clear cached settings (tests / process workers)."""
    get_settings.cache_clear()
