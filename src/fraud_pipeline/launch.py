"""Container entrypoint for Vertex Pipeline components.

Applies ``FRAUD_CONFIG_JSON`` (a JSON object of env vars) then runs a CLI
command or a step that writes KFP output parameters.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def apply_config_env() -> None:
    raw = os.environ.get("FRAUD_CONFIG_JSON", "").strip()
    if not raw:
        return
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SystemExit("FRAUD_CONFIG_JSON must be a JSON object")
    for key, value in payload.items():
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        os.environ[str(key)] = text


def _write_text(path: str, text: str) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")


def _train_component(gcs_uri_path: str, metrics_uri_path: str) -> None:
    from fraud_pipeline.config import reset_settings_cache

    reset_settings_cache()
    from fraud_pipeline.steps.train import train_local

    artifacts = train_local()
    _write_text(gcs_uri_path, artifacts.gcs_uri)
    _write_text(metrics_uri_path, artifacts.metrics_uri)


def _register_component(artifact_uri: str, resource_name_path: str) -> None:
    from fraud_pipeline.config import reset_settings_cache

    reset_settings_cache()
    from fraud_pipeline.steps.register_model import register_model

    model = register_model(artifact_uri)
    _write_text(resource_name_path, model.resource_name)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) >= 2 and args[0] == "--config-json":
        os.environ["FRAUD_CONFIG_JSON"] = args[1]
        args = args[2:]
    apply_config_env()
    if not args:
        raise SystemExit("launch expects a fraud-pipeline command")
    if args[0] == "train-component":
        if len(args) != 3:
            raise SystemExit("train-component expects <gcs_uri_path> <metrics_uri_path>")
        _train_component(args[1], args[2])
        return
    if args[0] == "register-component":
        if len(args) != 3:
            raise SystemExit("register-component expects <artifact_uri> <resource_name_path>")
        _register_component(args[1], args[2])
        return

    from fraud_pipeline.cli import app

    app(args=args)


if __name__ == "__main__":
    main()
