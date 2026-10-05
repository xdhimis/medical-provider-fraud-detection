# Workload Identity Federation for GitHub Actions

This repo’s workflow (`.github/workflows/pipeline.yml`) uses **OIDC + WIF**.
Do **not** put a service-account JSON key in GitHub secrets or in `.env`.

## 1. Create a deployer service account

```bash
PROJECT_ID=your-gcp-project-id
SA_NAME=fraud-pipeline-deployer
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create "$SA_NAME" \
  --project="$PROJECT_ID" \
  --display-name="Fraud pipeline GHA deployer"

# Minimum POC roles (tighten later):
for ROLE in \
  roles/bigquery.dataEditor \
  roles/bigquery.jobUser \
  roles/storage.objectAdmin \
  roles/aiplatform.user
do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="$ROLE"
done
```

## 2. Create a Workload Identity Pool + GitHub provider

```bash
POOL_ID=github-pool
PROVIDER_ID=github-provider
REPO=xdhimis/medical-provider-fraud-detection   # owner/name

gcloud iam workload-identity-pools create "$POOL_ID" \
  --project="$PROJECT_ID" \
  --location="global" \
  --display-name="GitHub Actions pool"

gcloud iam workload-identity-pools providers create-oidc "$PROVIDER_ID" \
  --project="$PROJECT_ID" \
  --location="global" \
  --workload-identity-pool="$POOL_ID" \
  --display-name="GitHub provider" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.actor=assertion.actor" \
  --attribute-condition="assertion.repository=='${REPO}'"
```

## 3. Allow the repo to impersonate the SA

```bash
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
PRINCIPAL="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/attribute.repository/${REPO}"

gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --project="$PROJECT_ID" \
  --role="roles/iam.workloadIdentityUser" \
  --member="$PRINCIPAL"

WIF_PROVIDER="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/providers/${PROVIDER_ID}"
echo "GCP_WIF_PROVIDER=${WIF_PROVIDER}"
echo "GCP_DEPLOYER_SA=${SA_EMAIL}"
```

## 4. GitHub configuration

**Secrets** (Settings → Secrets and variables → Actions):

| Secret | Value |
|--------|--------|
| `GCP_WIF_PROVIDER` | full provider resource name from step 3 |
| `GCP_DEPLOYER_SA` | `fraud-pipeline-deployer@PROJECT.iam.gserviceaccount.com` |

**Variables** (same page → Variables): set `GCP_PROJECT`, `GCP_REGION`, `GCS_BUCKET`, and the other `vars.*` referenced in `pipeline.yml`.

## 5. Enable the auth step

In `.github/workflows/pipeline.yml`, change:

```yaml
if: ${{ false }}  # enable after configuring WIF secrets/vars
```

to:

```yaml
if: ${{ true }}
```

(or remove the `if:` line).

## Local vs CI

| Environment | Auth |
|-------------|------|
| Laptop | `gcloud auth application-default login` |
| GitHub Actions | WIF via `google-github-actions/auth@v2` (this doc) |
| Vertex Pipeline workers | Runtime SA on the Custom Job / Pipeline (no user ADC) |
