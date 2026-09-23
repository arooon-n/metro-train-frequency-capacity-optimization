#!/usr/bin/env python
"""Run Model 3 — Weighted Goal Programming for multi-objective service planning.

Weights come from config/goal_weights.yaml (editable, never hardcoded).

Usage:
    python scripts/run_model3.py
    python scripts/run_model3.py --target-utilization 0.75 --demand-multiplier 1.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import add_common_args, finish, prepare


def main() -> int:
    ap = argparse.ArgumentParser(description="Model 3: weighted goal programming")
    add_common_args(ap)
    args = ap.parse_args()

    from src.models.common import ModelDataError
    from src.models.model3 import solve_model3
    from src.utils.config import load_goal_weights
    from src.validation.validator import validate_model3

    data, params, links, line_df, station = prepare(args)
    params["model"] = "model3"
    weights = load_goal_weights()
    try:
        result = solve_model3(
            data,
            weights_cfg=weights,
            target_utilization=params["target_utilization"],
            solver_name=params.get("solver"),
            solver_options=params.get("solver_options"),
        )
    except ModelDataError as exc:
        raise SystemExit(f"CONFIGURATION ERROR:\n{exc}")
    report = validate_model3(result, data)


    from src.visualization import plots

    figs = {"baseline_vs_optimized": plots.baseline_vs_optimized(
        result.results, xcol="trains_allocated"
    )}
    figs = {k: v for k, v in figs.items() if v is not None}

    finish(
        args,
        model_name="model3",
        results=result.results,
        report=report,
        data=data,
        params={
            **params, **result.parameters,
            "goal_weights": weights, "goal_totals": result.goal_totals,
            "warnings": result.warnings,
        },
        solver=result.solver,
        figures=figs,
    )
    if result.goal_totals:
        print("Goal totals:", result.goal_totals)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
