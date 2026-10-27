"""Compile and submit the fraud-detection Vertex pipeline.

Examples::

    python -m pipelines.submit --compile pipelines/fraud_detection.yaml
    python -m pipelines.submit --submit
    python -m pipelines.submit --submit --deploy
    python -m pipelines.submit --schedule "0 6 * * 1"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraud_pipeline.config import get_settings
from pipelines.fraud_pipeline import compile_pipeline


def _parameter_values(*, deploy: bool) -> dict:
    settings = get_settings()
    return {
        "config_json": json.dumps(settings.as_env()),
        "deploy_model": deploy,
    }


def build_job(
    *,
    template_path: str,
    display_name: str,
    deploy: bool,
    enable_caching: bool,
):
    from google.cloud import aiplatform

    settings = get_settings()
    return aiplatform.PipelineJob(
        display_name=display_name,
        template_path=template_path,
        pipeline_root=f"gs://{settings.gcs_bucket}/pipeline-root",
        parameter_values=_parameter_values(deploy=deploy),
        enable_caching=enable_caching,
        project=settings.gcp_project,
        location=settings.gcp_region,
        labels={"pipeline": "fraud-detection"},
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compile or submit the Vertex fraud pipeline")
    parser.add_argument(
        "--compile",
        metavar="PATH",
        help="Write the compiled pipeline YAML and exit unless --submit/--schedule is set",
    )
    parser.add_argument("--submit", action="store_true", help="Submit one PipelineJob")
    parser.add_argument(
        "--schedule",
        metavar="CRON",
        help="Create a Vertex pipeline schedule (example: '0 6 * * 1')",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy when the evaluate gate passes",
    )
    parser.add_argument("--image", default="", help="Artifact Registry image (else PIPELINE_IMAGE)")
    parser.add_argument("--display-name", default="fraud-detection")
    parser.add_argument("--enable-caching", action="store_true")
    args = parser.parse_args(argv)

    if not args.compile and not args.submit and not args.schedule:
        parser.error("pass --compile PATH and/or --submit and/or --schedule CRON")

    settings = get_settings()
    image = (args.image or settings.pipeline_image or "").strip()
    if not image:
        parser.error("set PIPELINE_IMAGE or pass --image")

    template = args.compile or "pipelines/fraud_detection.yaml"
    Path(template).parent.mkdir(parents=True, exist_ok=True)
    compile_pipeline(image, template)
    print(f"compiled {template}")

    if not args.submit and not args.schedule:
        return

    job = build_job(
        template_path=template,
        display_name=args.display_name,
        deploy=args.deploy,
        enable_caching=args.enable_caching,
    )
    account = settings.vertex_service_account
    if args.schedule:
        schedule = job.create_schedule(
            cron=args.schedule,
            display_name=args.display_name,
            max_concurrent_run_count=1,
            service_account=account,
        )
        print(f"schedule {schedule.resource_name}")
        return

    submit_kwargs: dict = {"service_account": account}
    if settings.vertex_experiment:
        submit_kwargs["experiment"] = settings.vertex_experiment
    job.submit(**submit_kwargs)
    print(f"submitted {job.resource_name}")


if __name__ == "__main__":
    main()
