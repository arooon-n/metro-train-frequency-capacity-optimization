#!/usr/bin/env python
"""Rebuild data/processed/cleaned_data.parquet from the raw workbook.

Equivalent to the cleaning stage of profile_dataset.py, kept separate so the
pipeline can be re-run after configuration/mapping changes without reprinting
the full sheet listing.

Usage:
    python scripts/preprocess.py [--data-file path.xlsx] [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.loader import SchemaDetectionError, detect_schema, profile_workbook
from src.preprocessing.pipeline import build_profile, preprocess
from src.utils import config as cfg
from src.utils.io import save_cleaned, save_profile


def main() -> int:
    ap = argparse.ArgumentParser(description="Preprocess NUMBAT workbook")
    ap.add_argument("--data-file", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg.ensure_dirs()
    try:
        path = cfg.resolve_data_file(args.data_file)
    except cfg.DataFileMissingError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    mapping = cfg.load_schema_mapping()
    try:
        schema, frames = detect_schema(path, mapping)
    except SchemaDetectionError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    cleaned, report = preprocess(frames, schema, mapping)
    info = profile_workbook(path, mapping)
    profile = build_profile(info, schema, report, cleaned)
    profile["quality"]["blocking_issues"] = bool(
        report.null_demand_rows or report.null_key_rows
        or report.negative_load_rows or report.negative_frequency_rows
    )
    save_profile(profile)

    if profile["quality"]["blocking_issues"] and not args.force:
        print(
            "Quality issues detected — inspect data/processed/dataset_profile.json. "
            "Re-run with --force to write the parquet anyway.",
            file=sys.stderr,
        )
        return 4

    out = save_cleaned(cleaned)
    print(f"Cleaned data written: {out} ({len(cleaned)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
