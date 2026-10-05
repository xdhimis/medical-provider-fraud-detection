# Container image for CLI steps (local, GHA, and future Vertex Pipeline components).
# Build:
#   docker build -t REGION-docker.pkg.dev/PROJECT/fraud/fraud-pipeline:dev .
# Run:
#   docker run --rm -e GCP_PROJECT=... -e GCS_BUCKET=... fraud-pipeline:dev show-config

FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY pipelines ./pipelines

RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["fraud-pipeline"]
CMD ["show-config"]
