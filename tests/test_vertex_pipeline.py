"""Vertex Custom Job spec, launcher, and pipeline compile (no live GCP)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fraud_pipeline.config import Settings
from fraud_pipeline.exceptions import TrainingError
from fraud_pipeline.launch import apply_config_env
from fraud_pipeline.steps.vertex_train import build_worker_pool_spec, train_vertex


def _settings() -> Settings:
    return Settings(
        gcp_project="demo-project",
        gcs_bucket="demo-bucket",
        gcp_region="us-central1",
        pipeline_image="us-docker.pkg.dev/demo/fraud:dev",
        train_machine_type="n1-standard-4",
    )


def test_worker_spec_runs_local_train_with_shared_run_id() -> None:
    spec = build_worker_pool_spec(
        _settings(),
        image_uri="us-docker.pkg.dev/demo/fraud:dev",
        run_id="20261008T000000Z-abcd1234",
        machine_type="n1-standard-4",
    )
    container = spec["container_spec"]
    assert container["command"] == ["fraud-pipeline", "train", "--local"]
    assert "--vertex" not in container["command"]
    env = {item["name"]: item["value"] for item in container["env"]}
    assert env["FRAUD_RUN_ID"] == "20261008T000000Z-abcd1234"
    assert env["GCP_PROJECT"] == "demo-project"
    assert env["GCS_BUCKET"] == "demo-bucket"


def test_train_vertex_requires_image() -> None:
    settings = _settings()
    settings = settings.model_copy(update={"pipeline_image": None})
    with pytest.raises(TrainingError, match="PIPELINE_IMAGE"):
        train_vertex(settings)


def test_train_vertex_submits_custom_job(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class FakeJob:
        def run(self, service_account=None, sync=True) -> None:
            captured["service_account"] = service_account
            captured["sync"] = sync

    def fake_custom_job(**kwargs):
        captured["kwargs"] = kwargs
        return FakeJob()

    monkeypatch.setattr("fraud_pipeline.steps.vertex_train.init_vertex", lambda settings: None)
    monkeypatch.setattr(
        "fraud_pipeline.steps.vertex_train.aiplatform.CustomJob",
        fake_custom_job,
    )
    monkeypatch.setattr(
        "fraud_pipeline.steps.vertex_train._new_run_id",
        lambda: "20261008T000000Z-abcd1234",
    )

    artifacts = train_vertex(_settings(), service_account="sa@demo.iam.gserviceaccount.com")
    assert artifacts.run_id == "20261008T000000Z-abcd1234"
    assert artifacts.gcs_uri.endswith("/20261008T000000Z-abcd1234")
    assert captured["kwargs"]["worker_pool_specs"][0]["container_spec"]["image_uri"].endswith(
        "fraud:dev"
    )
    assert captured["service_account"] == "sa@demo.iam.gserviceaccount.com"
    assert captured["sync"] is True


def test_apply_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BQ_LOCATION", raising=False)
    monkeypatch.setenv(
        "FRAUD_CONFIG_JSON",
        json.dumps({"BQ_LOCATION": "us-central1", "EMPTY": "  "}),
    )
    apply_config_env()
    import os

    assert os.environ["BQ_LOCATION"] == "us-central1"
    assert "EMPTY" not in os.environ


def test_compile_pipeline(tmp_path: Path) -> None:
    pytest.importorskip("kfp")
    from pipelines.fraud_pipeline import compile_pipeline

    dest = tmp_path / "fraud_detection.yaml"
    compile_pipeline("us-docker.pkg.dev/demo/fraud:dev", str(dest))
    text = dest.read_text(encoding="utf-8")
    assert "fraud_pipeline.launch" in text
    assert "train-component" in text
    assert "evaluation-component" in text
    assert "set-alias" in text
    assert "us-docker.pkg.dev/demo/fraud:dev" in text
    assert "deploy_model" in text
