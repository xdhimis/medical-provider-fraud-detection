"""Medicare provider fraud detection on Vertex AI.

Pipeline (run locally, later from GitHub Actions):

  1. load-raw         GCS CSVs -> BigQuery raw tables
  2. build-features   SQL provider aggregation -> feature table
  3. feature-store    Feature Online Store + FeatureView + sync
  4. train            XGBoost from BigQuery features (local or CustomJob later)
  5. register-model   Upload artifact to Vertex Model Registry
  6. deploy           Create/deploy Endpoint
  7. predict          Online Feature Store lookup -> Endpoint.predict

How the online Feature Store is used
-----------------------------------
Training never reads the online store. Training queries the BigQuery
feature table (`fraud_features.provider_features`) so you get reproducible
batch features and cheap historical joins.

At serving time the same BigQuery table is the source of a Vertex AI
FeatureView. After sync, `predict` does:

  provider_id
    -> FeatureView.read(provider_id)     # low-latency online lookup
    -> ordered feature vector            # same column order as training
    -> Endpoint.predict(instances=[...]) # registered XGBoost model

That split is the production pattern: BigQuery = system of record for
features; Feature Online Store = low-latency serving copy keyed by entity id.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
