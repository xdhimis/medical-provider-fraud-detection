# Vertex Pipeline code runbook

How the **Vertex AI Pipeline** code in this repo works — read this to understand the design, not just the commands. For step-by-step ops (build image, submit, cleanup), see the root [README.md](../README.md).

---

## 1. Big picture

You already have working **step modules** (`fraud_pipeline.steps.*`) and a **CLI** (`fraud-pipeline load-raw`, `train`, …).

The Vertex pipeline does **not** rewrite SQL, training, or register logic. It only:

1. Puts those commands in a **DAG** (ordered graph of tasks).
2. Runs each task in the **same Docker image**.
3. Passes your `.env` settings into each task as JSON.
4. Passes a few **outputs** between tasks (`gcs_uri`, `metrics_uri`, model resource name).

```
Laptop                         Google Cloud
──────                         ────────────
pipelines/submit.py  ──submit──►  PipelineJob (orchestrator)
                                       │
                    for each DAG node  │
                                       ▼
                              pull PIPELINE_IMAGE
                                       │
                                       ▼
                         python -m fraud_pipeline.launch …
                                       │
                                       ▼
                         same steps as local CLI
                         (BigQuery, GCS, Feature Store, …)
```

**DAG** = Directed Acyclic Graph: steps with arrows (“must finish before”), no loops.

```
load-raw → build-features → feature-store → train → evaluate → register-model
                                                              → import-evaluation
                                                              └→ deploy (only if flag)
```

Two different “evaluate” ideas:

| Step | What it does |
|------|----------------|
| `evaluate` | Local gate: fail the run if `metrics.json` is below floors |
| `import-evaluation` | Agent Platform Model Evaluation: attach those metrics to the Registry model so they show in the console |

`import-evaluation` does **not** start a batch-predict / Dataflow job. It imports the metrics you already computed at train time.

Pre-deploy MLOps (no endpoint):

| Piece | Code | Console |
|-------|------|---------|
| Experiments | `steps/tracking.py` from `train` | Vertex AI → Experiments |
| ML Metadata | same module (`Execution` + `Artifact`) | Vertex AI → Metadata |
| Model Evaluation | `import-evaluation` | Model Registry → Evaluations |
| Alias | `set-alias` (`champion`) | Model Registry version aliases |

---

## 2. Files to know (in reading order)

| File | Role |
|------|------|
| `Dockerfile` | Image that contains the package + CLI. Every pipeline task runs this image. |
| `pipelines/fraud_pipeline.py` | Defines the DAG in KFP (`@dsl.container_component` + `@dsl.pipeline`). |
| `pipelines/submit.py` | Compiles Python → YAML, then creates a Vertex `PipelineJob`. |
| `src/fraud_pipeline/launch.py` | What each container actually starts: apply config, then run a command. |
| `src/fraud_pipeline/config.py` | Loads settings; `as_env()` turns them into a JSON blob for workers. |
| `src/fraud_pipeline/steps/*.py` | Real work (unchanged idea from local/GitHub). |
| `src/fraud_pipeline/steps/vertex_train.py` | **Separate** path: `train --vertex` Custom Job (not the full DAG). |
| `pipelines/fraud_detection.yaml` | Compiled IR (gitignored). What Vertex executes. |

---

## 3. Compile vs submit

### Compile (`python -m pipelines.submit --compile …`)

`submit.py` calls `compile_pipeline(image, path)` in `fraud_pipeline.py`:

- Takes the Python DAG (`make_pipeline(image)`).
- Uses the **KFP compiler** to write `pipelines/fraud_detection.yaml`.
- Bakes in the **image URI** for every component.
- Does **not** talk to GCP yet (except you need `PIPELINE_IMAGE` / `--image` decided).

Open the YAML: you will see components like `comp-load-raw`, `comp-train`, container `command` / `args`, and pipeline parameters `config_json` and `deploy_model`.

### Submit (`--submit`)

`submit.py`:

1. Compiles again (so YAML matches current code + image).
2. Reads your local `.env` via `get_settings()`.
3. Builds parameters:

```python
{
  "config_json": json.dumps(settings.as_env()),  # GCP_PROJECT, GCS_BUCKET, …
  "deploy_model": True/False,                    # from --deploy flag
}
```

4. Creates `aiplatform.PipelineJob(template_path=…, parameter_values=…, pipeline_root=gs://…/pipeline-root)`.
5. Calls `job.submit(service_account=VERTEX_SERVICE_ACCOUNT)`.

Your laptop exits after submit. **Workers run in GCP.** Watch **Vertex AI → Pipelines**.

---

## 4. How config reaches a worker

Pipeline tasks have **no** local `.env` file.

Flow:

1. Submit serializes settings → one string `config_json`.
2. Each component receives `config_json` as a pipeline parameter.
3. Container args look like:

```text
python -m fraud_pipeline.launch --config-json '{…}' load-raw
```

4. `launch.py`:
   - Puts that JSON into env as `FRAUD_CONFIG_JSON`.
   - `apply_config_env()` copies each key into `os.environ` (`GCP_PROJECT`, …).
   - Then runs the CLI (`fraud-pipeline` via Typer) or a special component helper.

So: **same `Settings` / `get_settings()` code as local**, but env vars were injected by the launcher.

**Pitfall:** if `.env` has a stale `MODEL_RESOURCE_NAME` after cleanup, that value is inside `config_json` and `register-model` tries to version a deleted parent model → task fails. Leave it blank on a clean project.

---

## 5. Anatomy of one DAG task

### Simple tasks (load-raw, build-features, feature-store, evaluate, deploy)

Defined in `make_pipeline()` as `@dsl.container_component` functions that return a `ContainerSpec`:

```python
@dsl.container_component
def load_raw(config_json: str):
    return dsl.ContainerSpec(
        image=image,                    # your Artifact Registry image
        command=_LAUNCH,                # python -m fraud_pipeline.launch --config-json
        args=[config_json, "load-raw"], # then the CLI subcommand
    )
```

Inside the container that becomes roughly:

```bash
python -m fraud_pipeline.launch --config-json '…' load-raw
# → launch applies env → fraud_pipeline.cli → steps.load_raw.load_raw()
```

KFP’s `ContainerSpec` has **no** `env=` field in this SDK version, so config is passed on the **command line** as `--config-json`, not as Docker env vars.

### Wiring order (the DAG)

In `fraud_detection_pipeline`:

```python
raw = load_raw(...).set_caching_options(False)
features = build_features(...).after(raw)
store = feature_store(...).after(features)
trained = train(...).after(store)
gated = evaluate(metrics_uri=trained.outputs["metrics_uri"])
registered = register_model(artifact_uri=trained.outputs["gcs_uri"]).after(gated)
```

- `.after(x)` = wait for task `x`.
- `set_caching_options(False)` = always re-run (POC-friendly; avoids stale cache).
- Train gets more CPU/memory for XGBoost.

`evaluate` does **not** need `.after(trained)` explicitly when it consumes `trained.outputs[…]` — using an output already creates a dependency. `register_model` still uses `.after(gated)` so registration only runs if evaluate succeeded.

### Optional deploy

```python
with dsl.If(deploy_model == True, name="deploy"):
    deploy(..., model_resource_name=registered.outputs["resource_name"])
```

`deploy_model` comes from `--deploy` on submit. Default `False` → register only (cheaper).

---

## 6. Tasks that pass data: train and register

KFP parameters between steps are often written as **files** on a shared path (`OutputPath`).

### Train component

```python
def train(config_json, gcs_uri: OutputPath(str), metrics_uri: OutputPath(str)):
    args=[config_json, "train-component", gcs_uri, metrics_uri]
```

`launch.py` special-cases `train-component`:

1. Calls `train_local()` (same function as `fraud-pipeline train --local`).
2. Writes the GCS artifact directory URI into the `gcs_uri` output file.
3. Writes `…/metrics.json` into the `metrics_uri` output file.

Next steps read those strings:

- `evaluate --metrics-uri <that metrics uri>`
- `register-model` with `<that gcs uri>`

### Register component

`register-component` calls `register_model(artifact_uri)` and writes the Vertex **model resource name** to an output file so `deploy` can use `--model-resource-name`.

Why not only call the Typer CLI? The CLI prints JSON to stdout; KFP needs **typed outputs** in files. The small helpers in `launch.py` bridge that gap.

---

## 7. Two ways to train (do not confuse them)

| | Pipeline train task | `fraud-pipeline train --vertex` |
|--|---------------------|----------------------------------|
| Trigger | Part of the DAG | You run CLI on laptop / CI |
| Code | `launch` → `train_local()` | `vertex_train.train_vertex()` → Custom Job → container runs `train --local` |
| File | `launch.py` + `steps/train.py` | `steps/vertex_train.py` |
| Why both | Full automated chain | Train alone without load-raw / feature-store / register |

Important: the Custom Job container must run **`--local`**, never `--vertex` again (that would recurse). `build_worker_pool_spec` hard-codes:

```python
["fraud-pipeline", "train", "--local"]
```

and sets `FRAUD_RUN_ID` so the submitter and worker agree on the GCS path.

---

## 8. Image and entrypoint

`Dockerfile`:

- Installs the package.
- `ENTRYPOINT ["fraud-pipeline"]` for simple `docker run … show-config`.
- Pipeline tasks **override** the command to `python -m fraud_pipeline.launch …`, so ENTRYPOINT is ignored for DAG nodes.

Workers must be **linux/amd64** (use `docker build --platform linux/amd64` on Apple Silicon).

Identity inside GCP: `VERTEX_SERVICE_ACCOUNT` (e.g. deployer SA) needs rights for BigQuery, GCS, Vertex, Artifact Registry pull, etc. Same family of roles as WIF for GitHub.

---

## 9. Mental model: GitHub vs Vertex

| | GitHub Actions | Vertex Pipeline |
|--|----------------|-----------------|
| Orchestrator | GitHub runner | Vertex PipelineJob |
| How you start | Manual workflow_dispatch per step | One `submit` for the whole DAG |
| Auth | WIF → deployer SA | Pipeline runs as `VERTEX_SERVICE_ACCOUNT` |
| Code reused | `fraud-pipeline` CLI | Same CLI via `launch.py` |
| Config | Repo **Variables** / secrets | `config_json` from local `.env` at submit time |

GitHub = remote remote-control for **one** step.  
Vertex = cloud orchestration for **all** steps with dependencies and a metric gate.

---

## 10. Failure patterns (code-level)

| Symptom | Likely cause in this design |
|---------|-----------------------------|
| Fail at register-model, parent not found | `MODEL_RESOURCE_NAME` in `config_json` points at deleted model |
| Image pull / exec format error | Built for arm64; rebuild amd64 |
| Evaluate fails | Metrics below floors in `steps/evaluate.py` — register correctly skipped |
| Train OK, evaluate missing URI | Train component didn’t write output paths (check `train-component` logs) |
| Submit OK but nothing runs | Look at Pipeline run in console; submit only creates the job |

---

## 11. Suggested reading path in the repo

1. Skim `PIPELINE_STEPS` at the top of `pipelines/fraud_pipeline.py` (human list of the DAG).  
2. Read `fraud_detection_pipeline()` (wiring + `If` for deploy).  
3. Read one simple component (`load_raw`) and then `train` / `register_model`.  
4. Read `launch.py` top to bottom (~80 lines).  
5. Read `submit.py` `main()` + `_parameter_values()`.  
6. Optionally read `vertex_train.py` for the standalone Custom Job path.  
7. Open a compiled `pipelines/fraud_detection.yaml` and find `fraud_pipeline.launch` and `train-component` — that is what Vertex actually schedules.

When you change step logic, edit `src/fraud_pipeline/steps/…`.  
When you change **order / gates / outputs**, edit `pipelines/fraud_pipeline.py` and recompile.
