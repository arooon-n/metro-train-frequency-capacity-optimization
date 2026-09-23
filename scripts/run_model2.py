#!/usr/bin/env python
"""Run Model 2 — Capacity-Constrained Service Allocation.

Objective: maximise SERVED LINE-BOARDING demand (line-level coverage) under a
train-equivalent service budget.

Usage:
    python scripts/run_model2.py
    python scripts/run_model2.py --fleet-multiplier 0.85
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import add_common_args, finish, prepare


def main() -> int:
    ap = argparse.ArgumentParser(description="Model 2: capacity-constrained allocation")
    add_common_args(ap)
    ap.add_argument("--require-budget", action="store_true",
                    help="fail if fleet_budget.csv is empty")
    args = ap.parse_args()

    from src.models.common import ModelDataError
    from src.models.model2 import solve_model2
    from src.validation.validator import validate_model2

    data, params, links, line_df, station = prepare(args)
    params["model"] = "model2"
    try:
        result = solve_model2(
            data,
            require_budget=args.require_budget,
            solver_name=params.get("solver"),
            solver_options=params.get("solver_options"),
        )
    except ModelDataError as exc:
        raise SystemExit(f"CONFIGURATION ERROR:\n{exc}")
    report = validate_model2(result, data)


    from src.visualization import plots

    figs = {"baseline_vs_optimized": plots.baseline_vs_optimized(
        result.results, xcol="trains_allocated"
    )}
    figs = {k: v for k, v in figs.items() if v is not None}

    finish(
        args,
        model_name="model2",
        results=result.results,
        report=report,
        data=data,
        params={**params, **result.parameters, "warnings": result.warnings},
        solver=result.solver,
        figures=figs,
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
