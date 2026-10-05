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

## GitHub Actions

`.github/workflows/pipeline.yml` mirrors the CLI. Wire `google-github-actions/auth` + repo variables before enabling the auth step (`if: ${{ false }}` is a safety latch).

GHA is a good first automation layer (manual `workflow_dispatch` per step). The longer-term path for scheduled retrain + deploy gates is **Vertex AI Pipelines** below.

## Future development: Vertex AI Pipelines (automated retrain)

Today each stage is run by hand (or one GHA job at a time). Next step is to wrap the **same Python steps** in a Vertex Pipeline so training and promotion are automated — no laptop required.

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

| Piece | Now (POC) | Future (automated) |
|-------|-----------|--------------------|
| Orchestration | You run `fraud-pipeline …` | Vertex Pipeline run (KFP / PipelineJob) |
| Training compute | `--local` on your machine | Vertex **Custom Training Job** (same `train_lib.py`) |
| Triggers | Manual | Console/SDK anytime, Cloud Scheduler (e.g. weekly), or new data in GCS/BQ |
| Deploy | Always deploy after register | Deploy only if metrics beat current model (gate) |
| Rollback | Redeploy previous model by hand | Keep prior Model Registry version; traffic split / undeploy |

### Planned repo additions (not built yet)

- `pipelines/fraud_pipeline.py` — Kubeflow / Vertex pipeline definition calling existing step modules (or container entrypoints)
- `train --vertex` — submit Custom Job instead of in-process fit
- Evaluate component — compare `metrics.json` to production model; fail the run if below threshold
- Optional canary deploy — e.g. 10% traffic, then promote
- Scheduler — Cloud Scheduler → PipelineJob for periodic retrain

### Retrain anytime

Once the pipeline exists, retrain is just a **new pipeline run**:

```bash
# conceptual — after pipelines/ is implemented
# python -m pipelines.submit --display-name fraud-retrain-$(date +%Y%m%d)
```

Or from the Vertex AI console: **Pipelines → Create run**. Each run gets a new `run_id`, new GCS artifacts, and a new Model Registry version. Feature Store sync stays in the DAG so online features match the batch table used for training.

### Why Vertex Pipelines (not GKE / Cloud Run for training)

This stack is already Vertex-native (BigQuery, Feature Store, Model Registry, Endpoint). Pipelines reuse that IAM and lineage. Cloud Run fits serving APIs, not long training jobs; GKE is only needed if you already run a custom K8s ML platform.

### Suggested rollout

1. Finish local POC (`load-raw` → `predict` → `cleanup`)
2. Implement `train --vertex` (Custom Job)
3. Define Vertex Pipeline that chains the existing steps
4. Add metric gate + optional canary
5. Attach Cloud Scheduler / event trigger for continuous retrain

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
