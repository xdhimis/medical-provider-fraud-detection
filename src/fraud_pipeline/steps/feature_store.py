"""Step 3: Provision Feature Online Store, FeatureView, and sync from BigQuery.

Why this step exists
--------------------
BigQuery holds the system-of-record feature table used for training.
The Feature Online Store holds a low-latency serving copy of the latest
row per provider_id. Online prediction (step predict) reads from the
FeatureView — never re-aggregates claims at request time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fraud_pipeline.config import Settings, get_settings
from fraud_pipeline.serving.feature_client import (
    ensure_feature_view,
    ensure_online_store,
    sync_feature_view,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FeatureStoreResult:
    online_store_id: str
    feature_view_id: str
    bigquery_source: str
    sync_operation: str


def provision_feature_store(settings: Settings | None = None) -> FeatureStoreResult:
    """Idempotently create store + view, then sync latest features online."""
    settings = settings or get_settings()
    store = ensure_online_store(settings)
    view = ensure_feature_view(settings, online_store=store)
    sync_op = sync_feature_view(settings, feature_view=view)
    result = FeatureStoreResult(
        online_store_id=settings.feature_online_store,
        feature_view_id=settings.feature_view,
        bigquery_source=settings.features_bq_uri,
        sync_operation=sync_op,
    )
    logger.info("Feature store ready: %s", result)
    return result
