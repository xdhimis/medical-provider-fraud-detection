"""Domain exceptions for the pipeline."""

from __future__ import annotations


class FraudPipelineError(Exception):
    """Base error for this package."""


class ConfigError(FraudPipelineError):
    """Invalid or missing configuration."""


class DataLoadError(FraudPipelineError):
    """GCS / BigQuery load failures."""


class FeatureStoreError(FraudPipelineError):
    """Feature Online Store / FeatureView failures."""


class TrainingError(FraudPipelineError):
    """Model training failures."""


class ModelRegistryError(FraudPipelineError):
    """Model upload / registry failures."""


class DeploymentError(FraudPipelineError):
    """Endpoint create / deploy failures."""


class PredictionError(FraudPipelineError):
    """Online feature fetch or endpoint prediction failures."""
