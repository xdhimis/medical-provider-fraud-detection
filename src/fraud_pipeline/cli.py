"""Typer CLI — one command per pipeline step (local now, GHA later)."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Optional

import typer
from rich import print as rprint

from fraud_pipeline.config import get_settings, reset_settings_cache
from fraud_pipeline.logging_setup import setup_logging

app = typer.Typer(
    name="fraud-pipeline",
    help="Medicare provider fraud detection on Vertex AI (production Python steps).",
    add_completion=False,
    no_args_is_help=True,
)


def _boot() -> None:
    reset_settings_cache()
    settings = get_settings()
    setup_logging(settings.log_level)


@app.command("load-raw")
def cmd_load_raw() -> None:
    """GCS CSVs -> BigQuery raw tables."""
    _boot()
    from fraud_pipeline.steps.load_raw import load_raw

    results = load_raw()
    for row in results:
        rprint(
            {
                "table_id": row.table_id,
                "source_uri": row.source_uri,
                "output_rows": row.output_rows,
            }
        )


@app.command("build-features")
def cmd_build_features() -> None:
    """Aggregate claim/beneficiary raw tables into provider feature table."""
    _boot()
    from fraud_pipeline.steps.build_features import build_features

    table_id = build_features()
    rprint({"features_table": table_id})


@app.command("feature-store")
def cmd_feature_store() -> None:
    """Create/sync Feature Online Store + FeatureView from BigQuery features.

    Online serving reads provider features by provider_id from this store.
    Training continues to read BigQuery directly.
    """
    _boot()
    from fraud_pipeline.steps.feature_store import provision_feature_store

    result = provision_feature_store()
    rprint(
        {
            "online_store_id": result.online_store_id,
            "feature_view_id": result.feature_view_id,
            "bigquery_source": result.bigquery_source,
            "sync_operation": result.sync_operation,
        }
    )


@app.command("train")
def cmd_train(
    local: bool = typer.Option(
        True,
        "--local/--vertex",
        help="Train in-process from BigQuery (default). --vertex reserved for CustomJob.",
    ),
) -> None:
    """Train XGBoost and upload artifacts to GCS."""
    _boot()
    if not local:
        raise typer.BadParameter(
            "Vertex CustomJob training is not wired yet; use --local for now."
        )
    from fraud_pipeline.steps.train import train_local

    artifacts = train_local()
    rprint(
        {
            "run_id": artifacts.run_id,
            "gcs_uri": artifacts.gcs_uri,
            "model_uri": artifacts.model_uri,
            "metrics_uri": artifacts.metrics_uri,
            "threshold_uri": artifacts.threshold_uri,
        }
    )


@app.command("register-model")
def cmd_register_model(
    artifact_uri: str = typer.Option(
        ...,
        "--artifact-uri",
        help="GCS directory from train step, e.g. gs://bucket/models/<run_id>",
    ),
) -> None:
    """Upload trained artifacts to Vertex Model Registry."""
    _boot()
    from fraud_pipeline.steps.register_model import register_model

    model = register_model(artifact_uri)
    rprint(
        {
            "resource_name": model.resource_name,
            "display_name": model.display_name,
            "artifact_uri": model.artifact_uri,
            "hint": "Save resource_name as MODEL_RESOURCE_NAME in .env for versioned re-registers",
        }
    )


@app.command("deploy")
def cmd_deploy(
    model_resource_name: Optional[str] = typer.Option(
        None,
        "--model-resource-name",
        help="Optional Model resource name; defaults to latest matching display name.",
    ),
) -> None:
    """Create endpoint (if needed) and deploy the registered model."""
    _boot()
    from fraud_pipeline.steps.deploy import deploy_model

    result = deploy_model(model_resource_name)
    rprint(
        {
            "endpoint_resource_name": result.endpoint_resource_name,
            "model_resource_name": result.model_resource_name,
            "display_name": result.display_name,
        }
    )


@app.command("undeploy")
def cmd_undeploy() -> None:
    """Undeploy all models from the endpoint (stop replica billing)."""
    _boot()
    from fraud_pipeline.steps.deploy import undeploy_all

    endpoint = undeploy_all()
    rprint({"undeployed_endpoint": endpoint})


@app.command("cleanup")
def cmd_cleanup(
    all_resources: bool = typer.Option(
        False,
        "--all",
        help="Also delete BigQuery datasets and GCS model artifacts (raw CSVs kept).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print planned actions without deleting anything.",
    ),
) -> None:
    """Tear down POC resources to stop spend (endpoint, feature store, models)."""
    _boot()
    from fraud_pipeline.steps.cleanup import cleanup_poc

    report = cleanup_poc(delete_data=all_resources, dry_run=dry_run)
    rprint(
        {
            "dry_run": report.dry_run,
            "actions": report.actions,
            "errors": report.errors,
        }
    )
    if report.errors:
        raise typer.Exit(code=1)


@app.command("predict")
def cmd_predict(
    provider_id: str = typer.Option(..., "--provider-id", help="Provider entity id"),
    threshold: Optional[float] = typer.Option(
        None,
        "--threshold",
        help="Override decision threshold (default: PREDICTION_THRESHOLD)",
    ),
) -> None:
    """Online Feature Store lookup + Endpoint.predict for one provider."""
    _boot()
    from fraud_pipeline.steps.predict import predict_provider

    result = predict_provider(provider_id, threshold=threshold)
    rprint(asdict(result))


@app.command("show-config")
def cmd_show_config() -> None:
    """Print resolved non-secret settings (for debugging local / CI env)."""
    _boot()
    settings = get_settings()
    payload = settings.model_dump()
    rprint(payload)


if __name__ == "__main__":
    app()
