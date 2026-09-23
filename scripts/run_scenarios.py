#!/usr/bin/env python
"""Run the configured sensitivity scenarios (baseline, +10%, +20%, reduced
fleet, target-utilisation sweeps).

Outputs:
    outputs/scenario_results.csv
    outputs/runs/<ts>_scenarios/ (full per-scenario result tables + metadata)

Usage:
    python scripts/run_scenarios.py --model model1
    python scripts/run_scenarios.py --model model3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import add_common_args, base_params, load_tables

from src.scenarios.runner import run_scenarios
from src.utils import config as cfg
from src.utils.manifest import new_run_dir, snapshot_config, write_json, write_run_metadata


def main() -> int:
    ap = argparse.ArgumentParser(description="Scenario analysis")
    add_common_args(ap)
    ap.add_argument("--model", default="model1", choices=["model1", "model2", "model3"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    links, line_df, _ = load_tables(args)
    params = base_params(args)
    params["scenario_model"] = args.model

    out_dir = Path(args.out_dir) if args.out_dir else cfg.OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running scenarios with base model: {args.model}")
    summary, runs = run_scenarios(
        links, line_df, model=args.model, params=None,
        solver_name=params.get("solver"),
        progress=lambda n: print(f"  - {n} ..."),
    )

    summary_path = out_dir / "scenario_results.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nScenario summary: {summary_path}")
    print(summary.to_string(index=False))

    run_dir = new_run_dir(prefix="scenarios")
    snapshot_config(run_dir)
    summary.to_csv(run_dir / "results.csv", index=False)
    for name, run in runs.items():
        if run.results is not None and not run.results.empty:
            run.results.to_csv(run_dir / f"{name}.csv", index=False)
        if run.error or run.infeasible_explanation:
            write_json(
                run_dir / f"{name}_notes.json",
                {"error": run.error, "infeasible_explanation": run.infeasible_explanation,
                 "status": run.solver_status},
            )
    write_run_metadata(
        run_dir,
        dataset_path=cfg.resolve_data_file(None) if _has_data() else None,
        model_name=f"scenarios({args.model})",
        scenario="sweep",
        parameters=params,
        solver_name=params.get("solver") or "appsi_highs",
        solver_status="; ".join(f"{k}:{v.solver_status}" for k, v in runs.items()),
        selected_mode=params.get("mode"),
        selected_lines=params.get("lines"),
    )
    print(f"Run dir: {run_dir}")
    return 0


def _has_data() -> bool:
    try:
        cfg.resolve_data_file(None)
        return True
    except cfg.DataFileMissingError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
