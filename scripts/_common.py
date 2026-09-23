"""Shared CLI plumbing for the model/scenario scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import eda
from src.models.common import prepare_model_data
from src.utils import config as cfg
from src.utils.io import load_cleaned, load_profile, split_mode_lines
from src.utils.manifest import new_run_dir, snapshot_config, write_json, write_run_metadata


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--mode", default=None, help="filter to a single mode label")
    ap.add_argument("--lines", nargs="*", default=None, help="filter to specific lines")
    ap.add_argument("--demand-multiplier", type=float, default=1.0)
    ap.add_argument("--target-utilization", type=float, default=None,
                    help="override target utilisation (default: scenario_parameters.yaml)")
    ap.add_argument("--fleet-multiplier", type=float, default=1.0)
    ap.add_argument("--out-dir", default=None, help="output directory (default: outputs/)")


def base_params(args) -> dict:
    sp = cfg.load_scenario_parameters()
    target = args.target_utilization
    if target is None:
        target = float(sp.get("target_utilization", 0.8))
    return {
        "mode": args.mode,
        "lines": args.lines,
        "demand_multiplier": args.demand_multiplier,
        "fleet_multiplier": args.fleet_multiplier,
        "target_utilization": float(target),
        "solver": sp.get("solver"),
        "solver_options": sp.get("solver_options"),
    }


def load_tables(args):
    links, line_df, station = eda.split_tables(load_cleaned())
    links = split_mode_lines(links, args.mode, args.lines)
    line_df = split_mode_lines(line_df, args.mode, args.lines)
    if links.empty:
        raise SystemExit(
            "No rows after mode/line filter — check --mode/--lines against the profile."
        )
    return links, line_df, station


def prepare(args):
    from src.models.common import ModelDataError

    try:
        links, line_df, station = load_tables(args)
        p = base_params(args)
        data = prepare_model_data(
            links, line_df,
            demand_multiplier=p["demand_multiplier"],
            target_utilization=p["target_utilization"],
            fleet_multiplier=p["fleet_multiplier"],
            require_capacity=True,
            require_line_demand=False,
        )
    except ModelDataError as exc:
        raise SystemExit(f"CONFIGURATION ERROR:\n{exc}")
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    return data, p, links, line_df, station


def finish(
    args, *, model_name: str, results: pd.DataFrame, report, data, params,
    scenario: str = "baseline", solver=None, extra_figs: dict | None = None,
    figures: dict | None = None,
) -> Path:
    from src.visualization import plots
    from src.utils.manifest import write_run_metadata  # noqa: F401

    out_dir = Path(args.out_dir) if args.out_dir else cfg.OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = new_run_dir(prefix=model_name.replace(" ", "").lower())
    snapshot_config(run_dir)

    results.to_csv(run_dir / "results.csv", index=False)
    write_json(run_dir / "validation_report.json", report.to_dict())
    (run_dir / "validation_report.txt").write_text(report.to_text(), encoding="utf-8")

    # canonical copies in outputs/
    results.to_csv(out_dir / f"{model_name}_results.csv", index=False)
    import json as _json

    (out_dir / "validation_report.json").write_text(
        _json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "validation_report.txt").write_text(report.to_text(), encoding="utf-8")

    profile = load_profile()
    if profile:
        write_json(run_dir / "dataset_profile.json", profile)
    write_json(run_dir / "model_parameters.json", params)

    write_run_metadata(
        run_dir,
        dataset_path=cfg.resolve_data_file(None) if _has_data() else None,
        model_name=model_name,
        scenario=scenario,
        parameters=params,
        solver_name=getattr(solver, "solver_name", params.get("solver") or "appsi_highs"),
        solver_status=getattr(solver, "status", "unknown"),
        selected_mode=params.get("mode"),
        selected_lines=params.get("lines") or sorted(data.links["line"].unique().tolist()),
        dataset_year=_dataset_year(),
    )

    # figures
    figs: dict = dict(figures or {})
    if not results.empty:
        f = plots.baseline_vs_optimized(results)
        if f is not None:
            figs["baseline_vs_optimized"] = f
        f2 = plots.frequency_change_by_group(results)
        if f2 is not None:
            figs["frequency_change_by_group"] = f2
    for k, f in (extra_figs or {}).items():
        figs[k] = f

    fig_dir = cfg.FIGURES_DIR
    run_figs = run_dir / "figures"
    for name, f in figs.items():
        try:
            plots.save_figure(f, fig_dir / f"{name}.png")
        except Exception:
            plots.save_figure_html(f, fig_dir / f"{name}.html")
        plots.save_figure_html(f, run_figs / f"{name}.html")
        plots.save_figure_html(f, fig_dir / f"{name}.html")

    print(f"\nResults  : {out_dir / (model_name + '_results.csv')}")
    print(f"Validation: {out_dir / 'validation_report.txt'}")
    print(f"Run dir  : {run_dir}")
    print(f"Solver   : {getattr(solver, 'solver_name', '?')} -> {getattr(solver, 'status', '?')}")
    if getattr(report, "infeasible_explanation", None):
        print("\n" + report.infeasible_explanation)
    print(f"Validation overall: {'PASSED' if report.passed else 'FAILED'}")
    return run_dir


def _has_data() -> bool:
    try:
        cfg.resolve_data_file(None)
        return True
    except cfg.DataFileMissingError:
        return False


def _dataset_year():
    profile = load_profile()
    if not profile:
        return None
    name = str(profile.get("workbook", ""))
    for token in name.replace("_", " ").replace("-", " ").split():
        if token.isdigit() and len(token) == 4:
            return token
    return None
