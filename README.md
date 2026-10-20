# Medical Provider Fraud Detection (Vertex AI)

Production-style Python pipeline for Medicare **provider-level** fraud detection on Google Cloud.
Inspired by the Kaggle [Medical Provider Fraud Detection System](https://www.kaggle.com/code/aishwaryasingh777/medical-provider-fraud-detection-system) notebook and the [Healthcare Provider Fraud Detection Analysis](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis) dataset — implemented as **CLI steps**, not a Workbench notebook.

## How the online Feature Store is used

```
                    ┌─────────────────────┐
   GCS CSVs ──────► │ BigQuery raw tables │
                    └─────────┬───────────┘
                              │ SQL aggregation
                              ▼
                    ┌──────────────────────────┐
                    │ BigQuery provider_features│  ◄── training reads HERE
                    │  (1 row / provider_id)    │
                    └─────────┬────────────────┘
                              │ FeatureView sync
                              ▼
                    ┌──────────────────────────┐
                    │ Feature Online Store      │  ◄── predict reads HERE
                    │  FeatureView(provider_id) │
                    └─────────┬────────────────┘
                              │ ordered feature vector
                              ▼
                    ┌──────────────────────────┐
                    │ Vertex Endpoint (XGBoost) │
                    └──────────────────────────┘
```

| Path | Source | Why |
|------|--------|-----|
| **Training** | BigQuery `provider_features` | Reproducible batch features, cheap joins, full labeled history |
| **Online predict** | Feature Online Store `FeatureView.read(provider_id)` | Low-latency latest features keyed by entity id, same column contract as training |

The online store is **not** used during training. It is a serving copy of the latest BigQuery feature row per `provider_id`. After each `build-features` (or on a schedule), run `feature-store` to sync. `predict` fetches features online, builds the vector in `FEATURE_COLUMNS` order, then calls the endpoint.

## Prerequisites

1. GCP project with billing
2. Enable APIs: `bigquery`, `storage`, `aiplatform`
3. Upload the four **Train** CSVs under `gs://$GCS_BUCKET/$GCS_RAW_PREFIX/`
4. Local auth: `gcloud auth application-default login` (no SA JSON in the repo)

## Setup

```bash
cd medical-provider-fraud-detection
# Prefer Python 3.11 (see .python-version). On macOS XGBoost needs OpenMP:
#   brew install libomp
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# edit .env with your GCP_PROJECT, GCS_BUCKET, etc.

# Local ADC auth (no service-account JSON in the repo):
gcloud auth application-default login
```

## Run locally (one step at a time)

```bash
fraud-pipeline show-config
fraud-pipeline load-raw
fraud-pipeline build-features
fraud-pipeline feature-store
fraud-pipeline train --local
# copy gcs_uri from train output:
fraud-pipeline register-model --artifact-uri gs://YOUR_BUCKET/models/RUN_ID
fraud-pipeline deploy
fraud-pipeline predict --provider-id PRV55912
# when done testing — stop spend:
fraud-pipeline undeploy
# or full POC teardown:
fraud-pipeline cleanup --dry-run
fraud-pipeline cleanup              # endpoint + feature store + models
fraud-pipeline cleanup --all        # also BQ datasets + gs://…/models/ (keeps raw CSVs)
```

## Cleanup (save money after the POC)

Highest cost is a **deployed endpoint**. Undeploy first; then delete the rest.

| Command | Effect |
|---------|--------|
| `fraud-pipeline undeploy` | Remove replicas only (endpoint shell may remain) |
| `fraud-pipeline cleanup` | Undeploy + delete endpoints, Feature Online Store, Model Registry entries |
| `fraud-pipeline cleanup --all` | Above + delete `fraud_raw` / `fraud_features` datasets and GCS `models/` prefix |

Raw training CSVs under `GCS_RAW_PREFIX` are **not** deleted. Equivalent script: `python scripts/cleanup_poc.py [--dry-run] [--all]`.

## Config

All settings come from environment variables (see `.env.example`). Credentials never live in config — use ADC locally and Workload Identity Federation in GitHub Actions.

## Layout

```
src/fraud_pipeline/
  cli.py                 # typer entrypoint
  config.py              # pydantic-settings
  schemas.py             # feature column contract + BQ schemas
  train_lib.py           # XGBoost training (no GCP I/O)
  sql/provider_features.sql
  serving/feature_client.py   # Feature Online Store read path
  steps/                 # one module per pipeline stage
```

## Tests

```bash
pytest -q
```

## GitHub Actions / WIF

`.github/workflows/pipeline.yml` mirrors the CLI. Auth uses Workload Identity Federation
(no SA JSON keys). Setup guide: [docs/wif-setup.md](docs/wif-setup.md).

The workflow auth step is enabled (`if: ${{ true }}`) once `GCP_WIF_PROVIDER` and `GCP_DEPLOYER_SA` are set.

GHA is a good first automation layer (manual `workflow_dispatch` per step). The longer-term path for scheduled retrain + deploy gates is **Vertex AI Pipelines** below.

## Phase 2: Vertex AI Pipelines

| Piece | How to run it |
|-------|----------------|
| Container | `Dockerfile` (`python -m fraud_pipeline.launch` inside pipeline tasks) |
| DAG | `pipelines/fraud_pipeline.py` |
| Submit / weekly schedule | `python -m pipelines.submit` |
| Metric gate | `evaluate` step; register runs only if it passes |
| Standalone Custom Job | `fraud-pipeline train --vertex` |

Install the compiler extra once: `pip install -e ".[pipelines]"`.

```bash
# Build and push (image must be in Artifact Registry)
docker build -t us-central1-docker.pkg.dev/$GCP_PROJECT/fraud/fraud-pipeline:dev .
docker push us-central1-docker.pkg.dev/$GCP_PROJECT/fraud/fraud-pipeline:dev

# .env: PIPELINE_IMAGE=… and optional VERTEX_SERVICE_ACCOUNT=…
python -m pipelines.submit --compile pipelines/fraud_detection.yaml
python -m pipelines.submit --submit                  # train + register, no deploy
python -m pipelines.submit --submit --deploy         # also deploy after the gate
python -m pipelines.submit --schedule "0 6 * * 1"    # Vertex schedule, Mondays 06:00 UTC
```

The pipeline train task runs `train_local` in the pipeline worker. `train --vertex` submits that same training container as its own Custom Job when you do not want the full DAG. Deploy stays off unless you pass `--deploy`. Canary traffic split is still later.

## Automated retrain

The GitHub workflow runs one CLI step per dispatch. The Vertex pipeline runs the whole chain without a laptop.

### Target flow

```
on demand / schedule / new data
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Vertex AI Pipeline                                       │
│  load-raw → build-features → feature-store sync           │
│       → train (Custom Job) → evaluate gate                │
│       → register-model → (optional) deploy / canary       │
└───────────────────────────────────────────────────────────┘
        │
        ├── artifacts → GCS + Model Registry
        └── serving   → Feature Online Store + Endpoint
```

### What changes vs local CLI

| Piece | CLI / GitHub | Vertex pipeline |
|-------|-------------|-----------------|
| Orchestration | One `fraud-pipeline` command per run | `python -m pipelines.submit` (PipelineJob) |
| Training compute | `--local` on the runner | Same `train_lib.py` inside the pipeline worker, or `train --vertex` |
| Triggers | Manual dispatch | Submit anytime, or `--schedule` (weekly cron) |
| Deploy | Separate `deploy` step | Only when you pass `--deploy`, and only after evaluate passes |
| Rollback | Redeploy previous model by hand | Keep prior Model Registry version; traffic split / undeploy |

### Still later

- Canary deploy (traffic split, then promote)
- Event trigger when new raw files land in GCS

### Retrain anytime

```bash
python -m pipelines.submit --submit --display-name fraud-retrain-$(date +%Y%m%d)
```

Or from the Vertex AI console: **Pipelines → Create run**. Each run gets a new `run_id`, new GCS artifacts, and a new Model Registry version. Feature Store sync stays in the DAG so online features match the batch table used for training.

### Why Vertex Pipelines (not GKE / Cloud Run for training)

This stack is already Vertex-native (BigQuery, Feature Store, Model Registry, Endpoint). Pipelines reuse that IAM and lineage. Cloud Run fits serving APIs, not long training jobs; GKE is only needed if you already run a custom K8s ML platform.

### Suggested rollout

1. Local and GitHub CLI path (`load-raw` → `predict`) — done
2. Push the pipeline image and run `python -m pipelines.submit --submit`
3. Turn on `--deploy` only after the evaluate gate looks right
4. Attach `--schedule` for a weekly retrain
5. Canary traffic split later

## Future enhancement: FeatureGroup (shared offline + online schema)

Google recommends defining features once in a **FeatureGroup** (Feature Registry) and binding both offline training and online serving to that group. That way adding a feature updates one registry definition instead of risking train/serve skew.

### What we do today (POC)

| Path | Source | Contract |
|------|--------|----------|
| Offline (train) | BigQuery `fraud_features.provider_features` | `FEATURE_COLUMNS` in `schemas.py` |
| Online (predict) | Feature Online Store FeatureView synced from the **same** BQ table | Same `FEATURE_COLUMNS` order |

BigQuery is the offline feature store. The online store is a low-latency copy. Schema consistency is **manual**: edit `sql/provider_features.sql` and `FEATURE_COLUMNS` together, then rebuild features, sync online, retrain, and redeploy.

### Gaps vs FeatureGroup

| Concern | Today (POC) | Future (FeatureGroup) |
|---------|-------------|------------------------|
| Single source of truth | BQ table + Python tuple (manual sync) | FeatureGroup registry |
| Add a feature | Edit SQL + `FEATURE_COLUMNS` + retrain + FeatureView sync + redeploy | Register feature once; offline materialization and online serving bind to the group |
| Train / serve skew risk | Possible if one of those steps is skipped | Lower — both paths resolve the same feature defs |
| Lineage / discovery | Implicit via repo + BQ | Feature Registry UI / APIs |
| Point-in-time / historical training sets | Full-table SQL rebuild | Optional FeatureGroup offline materialization with as-of semantics |

### Planned direction (not built yet)

- Create a FeatureGroup for provider entity features; register each column in `FEATURE_COLUMNS` (and any new ones) there
- Point FeatureView (online) and training dataset export (offline) at that FeatureGroup / shared BigQuery source
- Generate or validate `FEATURE_COLUMNS` from the FeatureGroup so the model contract cannot drift silently
- Keep the rule: **new features always require a retrain + model version** — FeatureGroup prevents schema mismatch; it does not skip redeploy
