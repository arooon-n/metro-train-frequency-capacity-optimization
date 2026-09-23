#!/usr/bin/env python
"""Profile the NUMBAT workbook: sheets, columns, detection, quality checks.

Outputs:
    data/processed/dataset_profile.json          (always, if workbook found)
    data/processed/cleaned_data.parquet          (only if validation succeeds
                                                  or --force is given)

Usage:
    python scripts/profile_dataset.py
    python scripts/profile_dataset.py --data-file path/to/numbat.xlsx
    $env:DATA_FILE = "path\to\numbat.xlsx"; python scripts/profile_dataset.py
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
    ap = argparse.ArgumentParser(description="Profile a TfL NUMBAT workbook")
    ap.add_argument("--data-file", default=None, help="path to NUMBAT .xlsx (or set DATA_FILE)")
    ap.add_argument("--force", action="store_true",
                    help="write cleaned_data.parquet even if quality checks flag issues")
    args = ap.parse_args()

    cfg.ensure_dirs()
    try:
        path = cfg.resolve_data_file(args.data_file)
    except cfg.DataFileMissingError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Workbook: {path}")
    mapping = cfg.load_schema_mapping()

    try:
        info = profile_workbook(path, mapping)
        print(f"Sheets found ({info['sheet_count']}):")
        for s in info["sheets"]:
            print(f"  - {s['name']}: {s['rows']} rows x {s['columns_count']} cols")

        schema, frames = detect_schema(path, mapping)
    except SchemaDetectionError as exc:
        print(str(exc), file=sys.stderr)
        # still save a sheet-level profile so the user has something to inspect
        try:
            info = profile_workbook(path, mapping)
            info["detection_error"] = str(exc)
            save_profile(info)
            print(f"Partial profile saved to {cfg.PROFILE_PATH}", file=sys.stderr)
        except Exception:
            pass
        return 3

    print("\nDetected roles:")
    for role, r in schema.roles.items():
        print(f"  - {role}: sheet='{r.sheet}' cols={r.columns}")

    cleaned, report = preprocess(frames, schema, mapping)
    profile = build_profile(info, schema, report, cleaned)

    problems = (
        report.negative_load_rows
        + report.negative_frequency_rows
        + report.null_key_rows
        + report.null_demand_rows
        + report.duplicate_rows_removed
    )
    profile["quality"]["blocking_issues"] = bool(
        report.null_demand_rows or report.null_key_rows
        or report.negative_load_rows or report.negative_frequency_rows
    )

    p = save_profile(profile)
    print(f"\nProfile written: {p}")

    print("Quality summary:")
    for k, v in report.to_dict().items():
        if k == "missing_15min_periods" and v:
            print(f"  - {k}: {len(v)} series with gaps (see profile JSON)")
        else:
            print(f"  - {k}: {v}")

    if report.missing_15min_periods:
        print("\nWARNING: missing 15-minute periods detected for some series "
              "(see dataset_profile.json -> quality.missing_15min_periods).")

    if profile["quality"]["blocking_issues"] and not args.force:
        print(
            "\nVALIDATION FAILED: negative/null values were excluded during "
            "cleaning (details above). Review the profile, fix the source file "
            "or mapping, then rerun. Use --force to write cleaned_data.parquet "
            "anyway for exploratory work.",
            file=sys.stderr,
        )
        return 4

    out = save_cleaned(cleaned)
    print(f"Cleaned data written: {out} ({len(cleaned)} rows)")
    print("Next: configure config/capacity_by_line.csv, then python scripts/preprocess.py "
          "or run the models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
