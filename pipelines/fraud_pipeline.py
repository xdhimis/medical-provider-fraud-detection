"""Vertex AI Pipeline: one container image, existing CLI / step modules.

Train runs in-process inside the pipeline worker (Vertex already schedules that
worker). ``fraud-pipeline train --vertex`` is the standalone Custom Job path.

Compile and submit::

    python -m pipelines.submit --compile pipelines/fraud_detection.yaml
    python -m pipelines.submit --submit --deploy
    python -m pipelines.submit --schedule "0 6 * * 1"
"""

from typing import NamedTuple

# ``--config-json`` is applied before the CLI command. KFP ContainerSpec has no env field.
_LAUNCH = ["python", "-m", "fraud_pipeline.launch", "--config-json"]


class PipelineStep(NamedTuple):
    name: str
    cli: str
    notes: str


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
        "In-worker training (the pipeline task is the Custom Job)",
    ),
    PipelineStep(
        "evaluate",
        "fraud-pipeline evaluate --metrics-uri gs://…/metrics.json",
        "Fail the run when metrics are below floors",
    ),
    PipelineStep(
        "register-model",
        "fraud-pipeline register-model --artifact-uri gs://…/<run_id>",
        "Model Registry version; runs only after evaluate passes",
    ),
    PipelineStep(
        "deploy",
        "fraud-pipeline deploy",
        "Optional; pipeline parameter deploy=true",
    ),
)


def describe_pipeline() -> str:
    lines = ["Vertex fraud-detection pipeline", ""]
    for i, step in enumerate(PIPELINE_STEPS, start=1):
        lines.append(f"{i}. {step.name}: {step.cli}")
        lines.append(f"   {step.notes}")
    return "\n".join(lines)


def make_pipeline(image: str):
    """Return a KFP pipeline function bound to ``image``."""
    from kfp import dsl

    if not image.strip():
        raise ValueError("pipeline image is required")

    @dsl.container_component
    def load_raw(config_json: str):
        return dsl.ContainerSpec(
            image=image,
            command=_LAUNCH,
            args=[config_json, "load-raw"],
        )

    @dsl.container_component
    def build_features(config_json: str):
        return dsl.ContainerSpec(
            image=image,
            command=_LAUNCH,
            args=[config_json, "build-features"],
        )

    @dsl.container_component
    def feature_store(config_json: str):
        return dsl.ContainerSpec(
            image=image,
            command=_LAUNCH,
            args=[config_json, "feature-store"],
        )

    @dsl.container_component
    def train(
        config_json: str,
        gcs_uri: dsl.OutputPath(str),
        metrics_uri: dsl.OutputPath(str),
    ):
        return dsl.ContainerSpec(
            image=image,
            command=_LAUNCH,
            args=[config_json, "train-component", gcs_uri, metrics_uri],
        )

    @dsl.container_component
    def evaluate(config_json: str, metrics_uri: str):
        return dsl.ContainerSpec(
            image=image,
            command=_LAUNCH,
            args=[config_json, "evaluate", "--metrics-uri", metrics_uri],
        )

    @dsl.container_component
    def register_model(
        config_json: str,
        artifact_uri: str,
        resource_name: dsl.OutputPath(str),
    ):
        return dsl.ContainerSpec(
            image=image,
            command=_LAUNCH,
            args=[config_json, "register-component", artifact_uri, resource_name],
        )

    @dsl.container_component
    def deploy(config_json: str, model_resource_name: str):
        return dsl.ContainerSpec(
            image=image,
            command=_LAUNCH,
            args=[config_json, "deploy", "--model-resource-name", model_resource_name],
        )

    @dsl.pipeline(
        name="fraud-detection",
        description=(
            "load-raw, features, feature store, train, evaluate, register, optional deploy"
        ),
    )
    def fraud_detection_pipeline(config_json: str, deploy_model: bool = False):
        raw = load_raw(config_json=config_json).set_caching_options(False)
        features = build_features(config_json=config_json).set_caching_options(False).after(raw)
        store = feature_store(config_json=config_json).set_caching_options(False).after(features)
        trained = (
            train(config_json=config_json)
            .set_caching_options(False)
            .set_cpu_limit("4")
            .set_memory_limit("15G")
            .after(store)
        )
        gated = evaluate(
            config_json=config_json,
            metrics_uri=trained.outputs["metrics_uri"],
        ).set_caching_options(False)
        registered = (
            register_model(
                config_json=config_json,
                artifact_uri=trained.outputs["gcs_uri"],
            )
            .set_caching_options(False)
            .after(gated)
        )
        with dsl.If(deploy_model == True, name="deploy"):  # noqa: E712
            deploy(
                config_json=config_json,
                model_resource_name=registered.outputs["resource_name"],
            ).set_caching_options(False)

    return fraud_detection_pipeline


def compile_pipeline(image: str, package_path: str) -> str:
    """Compile the pipeline to a YAML file Vertex PipelineJob can submit."""
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=make_pipeline(image),
        package_path=package_path,
    )
    return package_path


if __name__ == "__main__":
    print(describe_pipeline())
