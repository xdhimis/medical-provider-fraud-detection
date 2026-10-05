"""Unit tests for the evaluate metric gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fraud_pipeline.exceptions import TrainingError
from fraud_pipeline.steps.evaluate import evaluate_metrics, evaluate_metrics_file


def test_evaluate_metrics_passes() -> None:
    result = evaluate_metrics({"pr_auc": 0.5, "f1": 0.4, "roc_auc": 0.8})
    assert result.passed
    assert result.failures == []


def test_evaluate_metrics_fails_below_floor() -> None:
    result = evaluate_metrics({"pr_auc": 0.1, "f1": 0.4, "roc_auc": 0.8})
    assert not result.passed
    assert any("pr_auc" in f for f in result.failures)


def test_evaluate_metrics_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"pr_auc": 0.01, "f1": 0.01, "roc_auc": 0.01}))
    with pytest.raises(TrainingError, match="Evaluate gate failed"):
        evaluate_metrics_file(path)
