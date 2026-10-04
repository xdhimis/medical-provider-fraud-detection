"""Unit tests for training helpers (no GCP calls)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_pipeline.schemas import FEATURE_COLUMNS, LABEL_COLUMN
from fraud_pipeline.train_lib import choose_threshold, prepare_xy, train_xgboost


def _synthetic_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    # Make a somewhat separable label from total_reimbursed-like signal.
    score = data["total_reimbursed"] + 0.5 * data["claim_count"]
    data[LABEL_COLUMN] = (score > np.median(score)).astype(int)
    data["provider_id"] = [f"PRV{i:05d}" for i in range(n)]
    return pd.DataFrame(data)


def test_prepare_xy_requires_label() -> None:
    frame = _synthetic_frame()
    x, y = prepare_xy(frame)
    assert list(x.columns) == list(FEATURE_COLUMNS)
    assert set(y.unique()) <= {0, 1}


def test_choose_threshold_returns_unit_interval() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.7])
    thr = choose_threshold(y_true, y_prob, min_precision=0.2)
    assert 0.0 <= thr <= 1.0


def test_train_xgboost_smoke() -> None:
    frame = _synthetic_frame()
    result = train_xgboost(frame, test_size=0.25, random_seed=7)
    assert result.feature_columns == FEATURE_COLUMNS
    assert "roc_auc" in result.metrics
    assert 0.0 <= result.threshold <= 1.0
    proba = result.model.predict_proba(frame[list(FEATURE_COLUMNS)])[:, 1]
    assert len(proba) == len(frame)
