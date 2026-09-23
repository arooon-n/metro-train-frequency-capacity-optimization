#!/usr/bin/env python
"""Run Model 1 — Minimum Service Frequency Optimization.

Usage:
    python scripts/run_model1.py
    python scripts/run_model1.py --lines Victoria Central --demand-multiplier 1.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import add_common_args, finish, prepare


def main() -> int:
    ap = argparse.ArgumentParser(description="Model 1: minimum service frequency")
    add_common_args(ap)
    args = ap.parse_args()

    from src.analysis import eda
    from src.models.common import ModelDataError
    from src.models.model1 import solve_model1
    from src.utils import config as cfg
    from src.validation.validator import validate_model1
    from src.visualization import plots

    data, params, links, line_df, station = prepare(args)
    params["model"] = "model1"
    try:
        result = solve_model1(
            data, solver_name=params.get("solver"), solver_options=params.get("solver_options")
        )
    except ModelDataError as exc:
        raise SystemExit(f"CONFIGURATION ERROR:\n{exc}")

    report = validate_model1(result, data)

    util = eda.baseline_utilization(links, cfg.load_capacity_config())
    figs = {}
    heat = plots.utilization_heatmap(util)
    if heat is not None:
        figs["utilization_heatmap"] = heat
    figs["demand_heatmap"] = plots.demand_heatmap(links)
    figs["frequency_timeseries"] = plots.frequency_timeseries(links)

    finish(
        args,
        model_name="model1",
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
