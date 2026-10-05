"""Validators for Vertex AI resource name strings."""

from __future__ import annotations

import re

from fraud_pipeline.exceptions import ConfigError, ModelRegistryError

_MODEL_RESOURCE_RE = re.compile(
    r"^projects/[^/]+/locations/[^/]+/models/[^/@]+(?:@[^/]+)?$"
)
_ENDPOINT_RESOURCE_RE = re.compile(
    r"^projects/[^/]+/locations/[^/]+/endpoints/[^/]+$"
)


def validate_model_resource_name(resource_name: str) -> str:
    """Require a full Vertex Model resource name (optional @version)."""
    name = resource_name.strip()
    if not _MODEL_RESOURCE_RE.fullmatch(name):
        raise ConfigError(
            "MODEL_RESOURCE_NAME must be a full Vertex resource name like "
            "projects/PROJECT/locations/REGION/models/MODEL_ID "
            f"(got {resource_name!r}). Do not use a display name or partial path."
        )
    return name


def validate_endpoint_resource_name(resource_name: str) -> str:
    """Require a full Vertex Endpoint resource name."""
    name = resource_name.strip()
    if not _ENDPOINT_RESOURCE_RE.fullmatch(name):
        raise ConfigError(
            "ENDPOINT_RESOURCE_NAME must be a full Vertex resource name like "
            "projects/PROJECT/locations/REGION/endpoints/ENDPOINT_ID "
            f"(got {resource_name!r}). Do not use a display name or partial path."
        )
    return name


def parent_model_id(resource_name: str) -> str:
    """Strip @version so Model.upload creates the next version under the same model."""
    validated = validate_model_resource_name(resource_name)
    return validated.split("@", 1)[0]


def require_parent_model_id(resource_name: str) -> str:
    """Like parent_model_id but raises ModelRegistryError (register-model path)."""
    try:
        return parent_model_id(resource_name)
    except ConfigError as exc:
        raise ModelRegistryError(str(exc)) from exc
