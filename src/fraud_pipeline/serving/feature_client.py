"""Online Feature Store client — the serving-time feature path.

Training reads features from BigQuery. Online prediction reads the same
logical features from a Vertex AI FeatureView that syncs from that table.

Flow at inference
-----------------
  1. Caller provides entity id = provider_id
  2. FeatureView.read(provider_id) returns latest feature values
  3. Values are projected into FEATURE_COLUMNS order (training contract)
  4. Ordered vector is sent to the Vertex Endpoint
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from google.cloud import aiplatform
from google.api_core import exceptions as gax_exceptions

from fraud_pipeline.config import Settings
from fraud_pipeline.exceptions import FeatureStoreError, PredictionError
from fraud_pipeline.schemas import ENTITY_ID_COLUMN, FEATURE_COLUMNS, LABEL_COLUMN

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProviderFeatures:
    """Typed online feature payload for one provider."""

    provider_id: str
    values: dict[str, float]

    def as_instance(self, columns: Sequence[str] = FEATURE_COLUMNS) -> list[float]:
        """Return a dense vector in the exact column order used at training."""
        missing = [c for c in columns if c not in self.values]
        if missing:
            raise PredictionError(
                f"Online features for {self.provider_id} missing columns: {missing}"
            )
        return [float(self.values[c]) for c in columns]


class FeatureOnlineClient:
    """Thin wrapper around Vertex Feature Online Store reads."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._feature_view = None

    def _load_feature_view(self) -> Any:
        if self._feature_view is not None:
            return self._feature_view

        try:
            from vertexai.resources.preview.feature_store import FeatureView
        except ImportError as exc:  # pragma: no cover
            raise FeatureStoreError(
                "vertexai Feature Store SDK unavailable; upgrade google-cloud-aiplatform"
            ) from exc

        aiplatform.init(
            project=self._settings.gcp_project,
            location=self._settings.gcp_region,
        )
        self._feature_view = FeatureView(
            name=self._settings.feature_view,
            feature_online_store_id=self._settings.feature_online_store,
        )
        return self._feature_view

    def fetch(self, provider_id: str) -> ProviderFeatures:
        """Fetch latest online features for a provider entity."""
        if not provider_id or not provider_id.strip():
            raise PredictionError("provider_id is required")

        fv = self._load_feature_view()
        entity = provider_id.strip()
        try:
            # Current Vertex SDK: read(key=[entity_id, ...]) for composite keys.
            response = fv.read(key=[entity])
        except Exception as exc:
            raise PredictionError(
                f"FeatureView.read failed for {provider_id}: {exc}"
            ) from exc

        if hasattr(response, "to_dict"):
            response = response.to_dict()

        payload = _normalize_feature_response(response, provider_id=entity)
        values = {
            k: float(v)
            for k, v in payload.items()
            if k not in {ENTITY_ID_COLUMN, LABEL_COLUMN, "entity_id"}
            and v is not None
        }
        logger.info(
            "Fetched %d online features for provider_id=%s",
            len(values),
            provider_id,
        )
        return ProviderFeatures(provider_id=provider_id.strip(), values=values)


def _normalize_feature_response(response: Any, *, provider_id: str) -> Mapping[str, Any]:
    """Normalize SDK read() return shapes into a flat dict."""
    if response is None:
        raise PredictionError(f"No online features returned for {provider_id}")

    # Proto-style / dict responses (including FeatureViewReadResponse.to_dict())
    if isinstance(response, Mapping):
        flattened = _flatten_key_values_dict(response)
        if flattened:
            return flattened
        return dict(response)

    # List of keyspaces / entity responses
    if isinstance(response, (list, tuple)):
        if not response:
            raise PredictionError(f"Empty online feature response for {provider_id}")
        first = response[0]
        if isinstance(first, Mapping):
            flattened = _flatten_key_values_dict(first)
            return flattened or dict(first)
        if hasattr(first, "to_dict"):
            data = first.to_dict()
            if isinstance(data, Mapping):
                flattened = _flatten_key_values_dict(data)
                return flattened or dict(data)
        # Feature data often lives under .key_values / .features
        extracted = _extract_from_object(first)
        if extracted:
            return extracted

    if hasattr(response, "to_dict"):
        data = response.to_dict()
        if isinstance(data, Mapping):
            flattened = _flatten_key_values_dict(data)
            return flattened or dict(data)

    extracted = _extract_from_object(response)
    if extracted:
        return extracted

    raise PredictionError(
        f"Unrecognized FeatureView.read response type for {provider_id}: "
        f"{type(response)!r}"
    )


def _flatten_key_values_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten Vertex key_values.to_dict() → {feature_name: scalar}."""
    features = data.get("features")
    if not isinstance(features, (list, tuple)):
        # Already flat-ish feature map
        if any(k in data for k in FEATURE_COLUMNS):
            return dict(data)
        return {}

    out: dict[str, Any] = {}
    for item in features:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name") or item.get("feature_id")
        if name is None:
            continue
        raw = item.get("value")
        out[str(name)] = _scalar_from_feature_value(raw)
    return out


def _scalar_from_feature_value(raw: Any) -> Any:
    """Pick the first populated scalar from a FeatureValue dict/proto."""
    if raw is None or isinstance(raw, (int, float, str, bool)):
        return raw
    if isinstance(raw, Mapping):
        for key in (
            "double_value",
            "doubleValue",
            "int64_value",
            "int64Value",
            "float_value",
            "floatValue",
            "string_value",
            "stringValue",
            "bool_value",
            "boolValue",
        ):
            if key in raw and raw[key] is not None:
                return raw[key]
        # Sometimes nested as {"value": {...}}
        if "value" in raw:
            return _scalar_from_feature_value(raw["value"])
    for attr in (
        "double_value",
        "int64_value",
        "float_value",
        "string_value",
        "bool_value",
    ):
        if hasattr(raw, attr):
            val = getattr(raw, attr)
            if val is not None:
                return val
    return raw


def _extract_from_object(obj: Any) -> dict[str, Any] | None:
    """Best-effort extraction from SDK response objects."""
    for attr in ("key_values", "features", "data", "feature_values"):
        if not hasattr(obj, attr):
            continue
        value = getattr(obj, attr)
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, (list, tuple)):
            out: dict[str, Any] = {}
            for item in value:
                name = getattr(item, "name", None) or getattr(item, "feature_id", None)
                if name is None and isinstance(item, Mapping):
                    name = item.get("name") or item.get("feature_id")
                    val = item.get("value") or item.get("feature_value")
                else:
                    val = getattr(item, "value", None)
                    if val is not None and hasattr(val, "bool_value"):
                        # FeatureValue proto — pick first populated scalar
                        val = (
                            getattr(val, "double_value", None)
                            or getattr(val, "int64_value", None)
                            or getattr(val, "string_value", None)
                            or getattr(val, "bool_value", None)
                        )
                if name is not None:
                    out[str(name)] = val
            if out:
                return out
    return None


def ensure_online_store(settings: Settings) -> Any:
    """Create Bigtable Feature Online Store if missing (idempotent).

    Optimized online serving is deprecated / blocked for new stores.
    Use Bigtable with min=max=1 nodes for a cheap POC footprint.
    """
    from vertexai.resources.preview import feature_store

    aiplatform.init(project=settings.gcp_project, location=settings.gcp_region)
    store_id = settings.feature_online_store

    try:
        existing = feature_store.FeatureOnlineStore(store_id)
        # Touch a property / method that forces a get; fall through on 404.
        _ = existing.gca_resource  # type: ignore[attr-defined]
        logger.info("Feature Online Store already exists: %s", store_id)
        return existing
    except Exception as exc:
        if not _is_not_found(exc):
            # Resource may not support gca_resource — try create path carefully.
            logger.debug("FeatureOnlineStore get probe: %s", exc)

    try:
        logger.info(
            "Creating Bigtable Feature Online Store: %s (min=1, max=1 nodes)",
            store_id,
        )
        return feature_store.FeatureOnlineStore.create_bigtable_store(
            store_id,
            min_node_count=1,
            max_node_count=1,
            cpu_utilization_target=50,
        )
    except gax_exceptions.AlreadyExists:
        logger.info("Feature Online Store already exists (race): %s", store_id)
        return feature_store.FeatureOnlineStore(store_id)
    except Exception as exc:
        # Some SDK versions raise a generic error containing AlreadyExists.
        if "AlreadyExists" in type(exc).__name__ or "already exists" in str(exc).lower():
            logger.info("Feature Online Store already exists: %s", store_id)
            return feature_store.FeatureOnlineStore(store_id)
        raise FeatureStoreError(f"Failed to create Feature Online Store: {exc}") from exc


def ensure_feature_view(settings: Settings, online_store: Any | None = None) -> Any:
    """Create FeatureView from BigQuery provider_features if missing."""
    from vertexai.resources.preview import feature_store

    aiplatform.init(project=settings.gcp_project, location=settings.gcp_region)
    store = online_store or feature_store.FeatureOnlineStore(settings.feature_online_store)
    view_id = settings.feature_view
    source_uri = settings.features_bq_uri

    try:
        existing = feature_store.FeatureView(
            name=view_id,
            feature_online_store_id=settings.feature_online_store,
        )
        _ = getattr(existing, "gca_resource", existing)
        logger.info("FeatureView already exists: %s", view_id)
        return existing
    except Exception as exc:
        if not _is_not_found(exc):
            logger.debug("FeatureView get probe: %s", exc)

    try:
        logger.info(
            "Creating FeatureView %s from %s (entity=%s)",
            view_id,
            source_uri,
            ENTITY_ID_COLUMN,
        )
        return store.create_feature_view(
            name=view_id,
            source=feature_store.utils.FeatureViewBigQuerySource(
                uri=source_uri,
                entity_id_columns=[ENTITY_ID_COLUMN],
            ),
            # Plain 5-field cron (Vertex rejects some TZ=/step forms).
            # Manual sync() below still runs immediately for the POC.
            sync_config="0 0 * * *",
        )
    except Exception as exc:
        if "AlreadyExists" in type(exc).__name__ or "already exists" in str(exc).lower():
            logger.info("FeatureView already exists (race): %s", view_id)
            return feature_store.FeatureView(
                name=view_id,
                feature_online_store_id=settings.feature_online_store,
            )
        raise FeatureStoreError(f"Failed to create FeatureView: {exc}") from exc


def sync_feature_view(settings: Settings, feature_view: Any | None = None) -> str:
    """Trigger an on-demand sync from BigQuery -> online store. Returns operation name."""
    from vertexai.resources.preview import feature_store

    aiplatform.init(project=settings.gcp_project, location=settings.gcp_region)
    fv = feature_view or feature_store.FeatureView(
        name=settings.feature_view,
        feature_online_store_id=settings.feature_online_store,
    )
    try:
        sync_job = fv.sync()
    except Exception as exc:
        raise FeatureStoreError(f"FeatureView.sync failed: {exc}") from exc

    # Wait when the SDK returns a job/operation with result()/wait().
    op_name = getattr(sync_job, "resource_name", None) or getattr(sync_job, "name", None)
    wait = getattr(sync_job, "wait", None) or getattr(sync_job, "result", None)
    if callable(wait):
        logger.info("Waiting for FeatureView sync to complete...")
        wait()
    logger.info("FeatureView sync finished: %s", op_name or sync_job)
    return str(op_name or sync_job)


def _is_not_found(exc: BaseException) -> bool:
    if isinstance(exc, gax_exceptions.NotFound):
        return True
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return "notfound" in name or "not found" in text or "404" in text
