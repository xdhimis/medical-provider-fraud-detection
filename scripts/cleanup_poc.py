#!/usr/bin/env python3
"""CLI wrapper around fraud_pipeline.steps.cleanup (POC teardown).

Examples:
  python scripts/cleanup_poc.py --dry-run
  python scripts/cleanup_poc.py
  python scripts/cleanup_poc.py --all
"""

from __future__ import annotations

import argparse
import json
import sys

from fraud_pipeline.config import get_settings
from fraud_pipeline.logging_setup import setup_logging
from fraud_pipeline.steps.cleanup import cleanup_poc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tear down Vertex/BQ POC resources to stop spend."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also delete BigQuery datasets and GCS model artifacts (not raw CSVs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without deleting.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level)
    report = cleanup_poc(delete_data=args.all, dry_run=args.dry_run)
    print(
        json.dumps(
            {
                "dry_run": report.dry_run,
                "actions": report.actions,
                "errors": report.errors,
            },
            indent=2,
        )
    )
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
