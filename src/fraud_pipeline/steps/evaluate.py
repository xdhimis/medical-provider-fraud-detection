"""Step: evaluate gate — fail the run when metrics regress below a floor."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fraud_pipeline.exceptions import TrainingError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvaluateResult:
    passed: bool
    metrics: dict[str, Any]
    floors: dict[str, float]
    failures: list[str]


DEFAULT_FLOORS: dict[str, float] = {
    "pr_auc": 0.30,
    "f1": 0.25,
    "roc_auc": 0.70,
}


def evaluate_metrics(
    metrics: dict[str, Any],
    *,
    floors: dict[str, float] | None = None,
) -> EvaluateResult:
    """Compare validation metrics to fixed floors (prior-run compare comes later)."""
    floors = floors or DEFAULT_FLOORS
    failures: list[str] = []
    for key, minimum in floors.items():
        if key not in metrics:
            failures.append(f"missing metric {key!r}")
            continue
        value = float(metrics[key])
        if value < minimum:
            failures.append(f"{key}={value:.4f} < floor {minimum:.4f}")

    passed = not failures
    if passed:
        logger.info("Evaluate gate passed: %s", json.dumps(metrics, sort_keys=True))
    else:
        logger.error("Evaluate gate failed: %s", "; ".join(failures))
    return EvaluateResult(
        passed=passed,
        metrics=metrics,
        floors=floors,
        failures=failures,
    )


def evaluate_metrics_file(
    path: Path | str,
    *,
    floors: dict[str, float] | None = None,
) -> EvaluateResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TrainingError(f"metrics.json must be an object, got {type(payload)!r}")
    result = evaluate_metrics(payload, floors=floors)
    if not result.passed:
        raise TrainingError(
            "Evaluate gate failed: " + "; ".join(result.failures)
        )
    return result
