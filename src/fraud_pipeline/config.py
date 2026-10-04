"""Typed settings loaded from environment / .env (secrets stay out of git)."""

from __future__ import annotations

from functools import lru_cache
from typing import Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    prediction_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

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
        mode="before",
    )
    @classmethod
    def _strip_nonempty(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("must not be empty")
        return value

    @field_validator("model_resource_name", "endpoint_resource_name", mode="before")
    @classmethod
    def _empty_str_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
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
