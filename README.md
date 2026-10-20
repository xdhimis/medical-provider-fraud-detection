# Medical Provider Fraud Detection (Vertex AI)

Production-style Python pipeline for Medicare **provider-level** fraud detection on Google Cloud.
Inspired by the Kaggle [Medical Provider Fraud Detection System](https://www.kaggle.com/code/aishwaryasingh777/medical-provider-fraud-detection-system) notebook and the [Healthcare Provider Fraud Detection Analysis](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis) dataset — implemented as **CLI steps**, not a Workbench notebook.

## What is proven

| Path | Status |
|------|--------|
| Local CLI (`load-raw` → `predict`) | Done |
| GitHub Actions + WIF (same CLI, one step per dispatch) | Done |
| Vertex AI Pipeline DAG through Model Registry | Done (`--submit`, no deploy) |
| Weekly schedule / canary deploy | Not yet |

After a POC, run `fraud-pipeline cleanup --all` to stop spend (keeps raw CSVs in GCS). Artifact Registry images are cheap; optional to delete.

---

## Architecture

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
| **Training** | BigQuery `provider_features` | Batch features, labeled history |
| **Online predict** | Feature Online Store `FeatureView.read(provider_id)` | Low-latency features, same `FEATURE_COLUMNS` contract |

The online store is **not** used during training. Sync it with `feature-store` after `build-features`.

---

## From scratch (clean GCP)

Use this when the project is empty except a GCS bucket + Train CSVs (or after `cleanup --all`).

### 0. Prerequisites

1. GCP project with billing  
2. Enable APIs:

```bash
export GCP_PROJECT=your-gcp-project-id
export GCP_REGION=us-central1

gcloud config set project "$GCP_PROJECT"
gcloud services enable \
  bigquery.googleapis.com \
  storage.googleapis.com \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  iamcredentials.googleapis.com   # required for WIF / SA impersonation
```

3. Upload the four **Train** CSVs under `gs://$GCS_BUCKET/$GCS_RAW_PREFIX/`  
4. Local auth: `gcloud auth application-default login` (never put SA JSON in the repo)

### 1. Local Python setup

```bash
cd medical-provider-fraud-detection
# Prefer Python 3.11 (see .python-version). On macOS XGBoost needs OpenMP:
#   brew install libomp
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# edit .env — see Config notes below
gcloud auth application-default login
fraud-pipeline show-config
```

**Config notes (common pitfalls)**

| Variable | Guidance |
|----------|----------|
| `BQ_LOCATION` | Multi-region `US`/`EU`, or a region like `us-central1`. Datasets cannot move after create. |
| `MODEL_RESOURCE_NAME` / `ENDPOINT_RESOURCE_NAME` | Leave **blank** after cleanup. Stale IDs break `register-model` (parent model not found). |
| `PIPELINE_IMAGE` | Set after you push the Docker image (Vertex path). |
| `VERTEX_SERVICE_ACCOUNT` | Deployer SA used by PipelineJob / Custom Job (e.g. `fraud-pipeline-deployer@…`). |

### 2A. Local CLI path (full E2E)

```bash
fraud-pipeline load-raw
fraud-pipeline build-features
fraud-pipeline feature-store          # Bigtable — costs money
fraud-pipeline train --local
# copy metrics_uri / gcs_uri from train output:
fraud-pipeline evaluate --metrics-uri gs://BUCKET/…/metrics.json
fraud-pipeline register-model --artifact-uri gs://BUCKET/…/RUN_ID
fraud-pipeline deploy                 # endpoint — costs money
fraud-pipeline predict --provider-id PRV55912
```

### 2B. GitHub Actions path (one step at a time)

1. Complete WIF: [docs/wif-setup.md](docs/wif-setup.md)  
   - Repo secrets: `GCP_WIF_PROVIDER`, `GCP_DEPLOYER_SA`  
   - Repo variables: mirror `.env` (`GCP_PROJECT`, `GCS_BUCKET`, `BQ_LOCATION`, `ENDPOINT_MACHINE_TYPE`, …)  
2. Enable IAM Credentials API (`iamcredentials.googleapis.com`) if not already.  
3. Actions → **fraud-pipeline** → Run workflow → start with `show-config`, then:

`load-raw` → `build-features` → `feature-store` → `train` → `evaluate` (paste `metrics_uri`) → `register-model` (paste `artifact_uri`) → `deploy` → `predict` (paste `provider_id`)

Workflow file: `.github/workflows/pipeline.yml` (auth uses WIF; no SA JSON).

### 2C. Vertex AI Pipeline path (full DAG)

This runs the same CLI steps as a **DAG** in GCP (not GitHub step-by-step).

**DAG:** `load-raw` → `build-features` → `feature-store` → `train` → `evaluate` → `register-model` → optional `deploy`

#### A. Docker daemon

CLI alone is not enough — start Docker Desktop or Colima (`colima start`), then `docker ps` must work.

#### B. Artifact Registry + image (linux/amd64)

```bash
export GCP_PROJECT=your-gcp-project-id
export GCP_REGION=us-central1
export IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/fraud/fraud-pipeline:dev"

# once
gcloud artifacts repositories create fraud \
  --repository-format=docker \
  --location="$GCP_REGION" \
  --description="Fraud pipeline images" || true
gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev"

# Apple Silicon: force amd64 for Vertex workers
docker build --platform linux/amd64 -t "$IMAGE" .
docker push "$IMAGE"
```

Seeing two rows in Artifact Registry (tagged `dev` + untagged digest) after one push is normal.

#### C. `.env` for Vertex

```bash
PIPELINE_IMAGE=us-central1-docker.pkg.dev/PROJECT/fraud/fraud-pipeline:dev
VERTEX_SERVICE_ACCOUNT=fraud-pipeline-deployer@PROJECT.iam.gserviceaccount.com
MODEL_RESOURCE_NAME=          # blank on a clean project
ENDPOINT_RESOURCE_NAME=
```

#### D. Compile and submit

```bash
pip install -e ".[pipelines]"
python -m pipelines.submit --compile pipelines/fraud_detection.yaml
# → writes local pipelines/fraud_detection.yaml (gitignored)

python -m pipelines.submit --submit              # through register-model; no deploy
# python -m pipelines.submit --submit --deploy   # only when you want an endpoint
# python -m pipelines.submit --schedule "0 6 * * 1"
```

`--submit` returns after the job is **created**. Watch progress in  
**Vertex AI → Pipelines → Runs**. Nodes should go green in order; `deploy` stays skipped unless `--deploy`.

Standalone Custom Job (train only, no full DAG): `fraud-pipeline train --vertex` (same `PIPELINE_IMAGE`).

#### E. Verify side effects

| Step | Where |
|------|--------|
| load-raw / features | BigQuery `fraud_raw`, `fraud_features` |
| feature-store | Feature Online Store `fraud_provider_store` |
| train | `gs://$GCS_BUCKET/$ARTIFACT_GCS_PREFIX/<run_id>/` |
| register | Model Registry display name `provider-fraud-xgb` |

---

## Cleanup (save money)

Highest cost: **Feature Online Store (Bigtable)** and a **deployed endpoint**.

```bash
fraud-pipeline cleanup --dry-run
fraud-pipeline cleanup              # endpoint + feature store + Model Registry
fraud-pipeline cleanup --all        # also BQ datasets + GCS models/ (keeps raw CSVs)
```

After cleanup:

- Leave `MODEL_RESOURCE_NAME=` and `ENDPOINT_RESOURCE_NAME=` blank.  
- Artifact Registry `:dev` image — keep for the next run, or delete if you want zero images.  
- Raw CSVs under `GCS_RAW_PREFIX` are never deleted by cleanup.

---

## Config

All settings come from environment / `.env` (see `.env.example`). Credentials never live in config — ADC locally, WIF in GitHub Actions, deployer SA for Vertex jobs.

## Layout

```
src/fraud_pipeline/
  cli.py                 # typer entrypoint
  config.py              # pydantic-settings
  launch.py              # Vertex component entry (config JSON → CLI)
  schemas.py             # feature column contract + BQ schemas
  train_lib.py           # XGBoost training (no GCP I/O)
  sql/provider_features.sql
  serving/feature_client.py
  steps/                 # one module per stage (+ vertex_train.py)
pipelines/
  fraud_pipeline.py      # KFP DAG
  submit.py              # compile / submit / schedule
Dockerfile               # image for pipeline + Custom Job
.github/workflows/pipeline.yml
docs/wif-setup.md
```

## Tests

```bash
pip install -e ".[dev,pipelines]"
pytest -q
```

---

## Path comparison

| Piece | Local / GitHub | Vertex pipeline |
|-------|----------------|-----------------|
| Orchestration | One CLI command (or one GHA dispatch) | `python -m pipelines.submit` |
| Training | `train --local` on laptop/runner | Same `train_lib` inside the pipeline worker (or `train --vertex`) |
| Triggers | Manual | Submit anytime, or `--schedule` |
| Deploy | Explicit `deploy` step | Only with `--deploy`, after evaluate passes |

### Still later

- Canary deploy (traffic split)  
- Trigger on new GCS raw files  
- FeatureGroup as single schema source (offline + online)

### FeatureGroup (future)

Today schema is manual: keep `sql/provider_features.sql` and `FEATURE_COLUMNS` in sync. FeatureGroup would register features once so train/serve cannot drift; **new features still require retrain + new model version**.
