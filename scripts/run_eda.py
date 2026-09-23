"""EDA / figure generation script.

Builds the standard exploratory figures into outputs/figures/ from the
cleaned dataset (requires data/processed/cleaned_data.parquet).

Usage:
    python scripts/run_eda.py [--mode MODE] [--lines L1 L2 ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.analysis import eda
from src.network.builder import network_statistics
from src.utils import config as cfg
from src.utils.io import load_cleaned, split_mode_lines
from src.visualization import plots


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate exploratory figures")
    ap.add_argument("--mode", default=None)
    ap.add_argument("--lines", nargs="*", default=None)
    args = ap.parse_args()

    cfg.ensure_dirs()
    try:
        cleaned = load_cleaned()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    links, line_df, station = eda.split_tables(cleaned)
    links = split_mode_lines(links, args.mode, args.lines)
    line_df = split_mode_lines(line_df, args.mode, args.lines)
    if links.empty:
        print("No rows after filter.", file=sys.stderr)
        return 3

    sp = cfg.load_scenario_parameters()
    cap = cfg.load_capacity_config()
    target_util = float(sp.get("target_utilization", 0.8))
    peak_windows = sp.get("peak_windows") or [["07:00", "10:00"], ["16:00", "19:00"]]

    # tables
    tables = {
        "demand_by_time": eda.demand_by_time(links, line_df),
        "demand_by_line": eda.demand_by_line(links, line_df),
        "demand_by_link": eda.demand_by_link(links),
        "peak_vs_offpeak": eda.peak_vs_offpeak(links, peak_windows),
        "frequency_by_time": eda.frequency_by_time(links),
        "high_load_links": eda.high_load_links(links),
        "underserved_periods": eda.underserved_periods(links, cap, target_util),
        "underutilized_periods": eda.underutilized_periods(links, cap, target_util),
    }
    util = eda.baseline_utilization(links, cap)
    tables["baseline_utilization"] = util[
        ["line", "time_period", "from_station", "to_station", "link_load",
         "link_frequency", "capacity_per_train", "baseline_utilization"]
    ] if "baseline_utilization" in util.columns else util

    for name, df in tables.items():
        p = cfg.OUTPUTS_DIR / f"eda_{name}.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        print(f"  table: {p}")

    # figures
    figs = {
        "demand_heatmap": plots.demand_heatmap(links),
        "demand_timeseries": plots.demand_timeseries(links),
        "line_boarders_timeseries": plots.line_boarders_timeseries(line_df),
        "frequency_timeseries": plots.frequency_timeseries(links),
        "demand_frequency_scatter": plots.demand_frequency_scatter(links),
        "line_comparison": plots.line_comparison(links),
        "top_demand_links": plots.top_demand_links(links),
        "utilization_heatmap": plots.utilization_heatmap(util),
    }
    stats = network_statistics(links)
    print("\nNetwork statistics:")
    for k in ("stations", "unique_links", "links_by_line", "is_fully_connected_by_line"):
        print(f"  {k}: {stats[k]}")
    branches = {k: v for k, v in stats["branch_analysis"].items() if v["junction_stations"]}
    if branches:
        print("  junctions detected (configure service_groups.csv if needed):")
        for k, v in branches.items():
            print(f"    {k}: {v['junction_stations']}")

    for name, fig in figs.items():
        if fig is None:
            continue
        html_path = plots.save_figure_html(fig, cfg.FIGURES_DIR / f"{name}.html")
        try:
            png = plots.save_figure(fig, cfg.FIGURES_DIR / f"{name}.png")
            print(f"  figure: {png}")
        except Exception as exc:
            print(f"  figure (html): {html_path}  [png unavailable: {exc}]")

    util_sum = eda.utilization_summary(util)
    print("\nBaseline utilisation summary:", util_sum)
    print("\nDone. Figures in outputs/figures/, tables in outputs/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
