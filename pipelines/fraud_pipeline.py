"""Vertex AI Pipeline definition (Phase 2 scaffold).

Components call existing ``fraud_pipeline.steps.*`` entrypoints — they do not
reimplement SQL / train / predict. Wire a container image + submit helper next.

Usage (once ``[pipelines]`` extras + image are ready)::

    from pipelines.fraud_pipeline import fraud_detection_pipeline
    # compile / submit via google.cloud.aiplatform.PipelineJob
"""

from __future__ import annotations

from typing import NamedTuple


class PipelineStep(NamedTuple):
    name: str
    cli: str
    notes: str


# Target DAG from plan/20261004T003601-next-session.md Phase 2.
PIPELINE_STEPS: tuple[PipelineStep, ...] = (
    PipelineStep("load-raw", "fraud-pipeline load-raw", "GCS CSVs → BigQuery raw"),
    PipelineStep(
        "build-features",
        "fraud-pipeline build-features",
        "SQL → provider_features",
    ),
    PipelineStep(
        "feature-store",
        "fraud-pipeline feature-store",
        "Sync Feature Online Store",
    ),
    PipelineStep(
        "train",
        "fraud-pipeline train --local",
        "Swap to --vertex CustomJob when wired",
    ),
    PipelineStep(
        "evaluate",
        "python -c 'from fraud_pipeline.steps.evaluate import evaluate_metrics_file; ...'",
        "Fail run if PR-AUC / F1 below floors",
    ),
    PipelineStep(
        "register-model",
        "fraud-pipeline register-model --artifact-uri $ARTIFACT_URI",
        "Model Registry version",
    ),
    PipelineStep(
        "deploy",
        "fraud-pipeline deploy",
        "Optional / gated after evaluate",
    ),
)


def fraud_detection_pipeline() -> list[dict[str, str]]:
    """Return a serializable DAG description (KFP DSL lands in a follow-up)."""
    return [
        {"name": step.name, "cli": step.cli, "notes": step.notes}
        for step in PIPELINE_STEPS
    ]


def describe_pipeline() -> str:
    lines = ["Vertex fraud-detection pipeline (scaffold)", ""]
    for i, step in enumerate(PIPELINE_STEPS, start=1):
        lines.append(f"{i}. {step.name}: {step.cli}")
        lines.append(f"   {step.notes}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_pipeline())
