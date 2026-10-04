"""Shared training logic used by the local train step (and later CustomJobs)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from fraud_pipeline.exceptions import TrainingError
from fraud_pipeline.schemas import FEATURE_COLUMNS, LABEL_COLUMN

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainResult:
    model: xgb.XGBClassifier
    feature_columns: tuple[str, ...]
    threshold: float
    metrics: dict[str, Any]


def prepare_xy(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    label_column: str = LABEL_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate and split a feature frame into X / y."""
    required = list(feature_columns) + [label_column]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise TrainingError(f"Feature table missing columns: {missing}")

    labeled = frame.dropna(subset=[label_column]).copy()
    if labeled.empty:
        raise TrainingError("No labeled providers available for training")

    y = labeled[label_column].astype(int)
    x = labeled.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return x, y


def choose_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    min_precision: float = 0.35,
) -> float:
    """Pick a PR-curve threshold that maximizes F1 while keeping precision usable."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5

    best_t = 0.5
    best_f1 = -1.0
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds, strict=False):
        if p < min_precision:
            continue
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def train_xgboost(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_seed: int = 42,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> TrainResult:
    """Train an XGBoost classifier with class imbalance handling + threshold tuning."""
    x, y = prepare_xy(frame, feature_columns=feature_columns)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        raise TrainingError(f"Need both classes; got pos={pos} neg={neg}")

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    scale_pos_weight = neg / pos
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=random_seed,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )

    val_prob = model.predict_proba(x_val)[:, 1]
    threshold = choose_threshold(y_val.to_numpy(), val_prob)
    val_pred = (val_prob >= threshold).astype(int)

    metrics = {
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "pos_rate": float(y.mean()),
        "scale_pos_weight": float(scale_pos_weight),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_val, val_prob)),
        "pr_auc": float(average_precision_score(y_val, val_prob)),
        "precision": float(precision_score(y_val, val_pred, zero_division=0)),
        "recall": float(recall_score(y_val, val_pred, zero_division=0)),
        "f1": float(f1_score(y_val, val_pred, zero_division=0)),
    }
    logger.info("Validation metrics: %s", json.dumps(metrics, sort_keys=True))
    return TrainResult(
        model=model,
        feature_columns=tuple(feature_columns),
        threshold=threshold,
        metrics=metrics,
    )


def write_local_artifacts(result: TrainResult, output_dir: Path) -> dict[str, Path]:
    """Persist model + metadata next to each other (Vertex upload expects a directory)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.bst"
    result.model.save_model(model_path)

    columns_path = output_dir / "feature_columns.json"
    columns_path.write_text(
        json.dumps(list(result.feature_columns), indent=2) + "\n",
        encoding="utf-8",
    )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result.metrics, indent=2) + "\n", encoding="utf-8")

    threshold_path = output_dir / "threshold.json"
    threshold_path.write_text(
        json.dumps({"threshold": result.threshold}, indent=2) + "\n",
        encoding="utf-8",
    )

    # Booster-only export for Vertex prebuilt XGBoost containers.
    booster_path = output_dir / "model.bst"
    assert booster_path.exists()

    return {
        "model": model_path,
        "columns": columns_path,
        "metrics": metrics_path,
        "threshold": threshold_path,
    }


def train_result_summary(result: TrainResult) -> dict[str, Any]:
    return {
        "threshold": result.threshold,
        "feature_columns": list(result.feature_columns),
        "metrics": result.metrics,
    }
