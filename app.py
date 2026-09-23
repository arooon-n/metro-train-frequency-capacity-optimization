"""Streamlit dashboard — Data-Driven Urban Rail Train Frequency and Capacity
Optimization using TfL NUMBAT data.

Run:  streamlit run app.py

Every figure/table below is computed from data/processed/cleaned_data.parquet
plus config/*.csv|yaml — nothing is hardcoded, nothing is fabricated.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import eda
from src.models.common import ModelDataError, prepare_model_data
from src.models.model1 import solve_model1
from src.models.model2 import solve_model2
from src.models.model3 import solve_model3
from src.network.builder import network_statistics
from src.scenarios.runner import run_scenarios
from src.utils import config as cfg
from src.utils.io import load_cleaned, load_profile, split_mode_lines
from src.utils.manifest import new_run_dir, snapshot_config, write_json, write_run_metadata
from src.validation.validator import (
    validate_model1, validate_model2, validate_model3, write_report,
)
from src.visualization import plots

st.set_page_config(
    page_title="NUMBAT Rail Optimisation",
    page_icon=":train:",
    layout="wide",
)

DISCLAIMER = (
    "**Academic decision-support model — not an TfL operational timetable.** "
    "Optimised values are model recommendations under the configured assumptions. "
    "Link Frequency in NUMBAT is *scheduled/planned* supply, not observed operation."
)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading cleaned NUMBAT dataset…")
def get_data() -> tuple[pd.DataFrame, dict | None]:
    try:
        cleaned = load_cleaned()
    except FileNotFoundError as exc:
        return pd.DataFrame(), {"error": str(exc)}
    return cleaned, load_profile()


def capacity_df() -> pd.DataFrame:
    return cfg.load_capacity_config()


def constraints_df() -> pd.DataFrame:
    return cfg.load_service_constraints()


def fleet_df() -> pd.DataFrame:
    return cfg.load_fleet_budget()


def weights_cfg() -> dict:
    try:
        return cfg.load_goal_weights()
    except cfg.ConfigError as exc:
        st.error(str(exc))
        return {"weights": {"w1_unmet_demand": 1, "w2_frequency_target": 1,
                            "w3_crowding": 1, "w4_service_usage": 0.1},
                "normalize_terms": True}


def missing_data_banner() -> None:
    st.error(
        "**NUMBAT workbook / cleaned dataset not found.**\n\n"
        f"1. Download the official release from <{cfg.NUMBAT_SOURCE_URL}> "
        "(prefer NUMBAT 2025, Tuesday–Wednesday–Thursday 'TWT' weekday profile).\n"
        f"2. Place the `.xlsx` in `{cfg.RAW_DIR}`.\n"
        "3. Run `python scripts/profile_dataset.py` then `python scripts/preprocess.py`.\n"
        "4. Reload this page."
    )


def check_capacity(lines: list[str]) -> tuple[bool, list[str]]:
    cap = capacity_df()
    if cap.empty:
        return False, ["config/capacity_by_line.csv is EMPTY — capacity_per_train is a "
                       "mandatory manual input (never fabricated)."]
    have = set(cap["line"].dropna().astype(str))
    missing = [l for l in lines if l not in have]
    if missing:
        return False, [f"No capacity_per_train configured for lines: {missing}. "
                       "Add sourced values in config/capacity_by_line.csv."]
    return True, []


def run_manifest(model_name: str, results: pd.DataFrame, params: dict,
                 solver_status: str, scenario: str = "baseline") -> Path:
    run_dir = new_run_dir(prefix=f"ui_{model_name}")
    snapshot_config(run_dir)
    results.to_csv(run_dir / "results.csv", index=False)
    write_json(run_dir / "model_parameters.json", params)
    profile = load_profile()
    if profile:
        write_json(run_dir / "dataset_profile.json", profile)
    data_file = None
    try:
        data_file = cfg.resolve_data_file(None)
    except cfg.DataFileMissingError:
        pass
    write_run_metadata(
        run_dir,
        dataset_path=data_file,
        model_name=model_name,
        scenario=scenario,
        parameters=params,
        solver_name=str(params.get("solver", "appsi_highs")),
        solver_status=solver_status,
        selected_mode=params.get("mode"),
        selected_lines=params.get("lines"),
        dataset_year=_year_from_profile(profile),
    )
    return run_dir


def _year_from_profile(profile: dict | None) -> str | None:
    if not profile:
        return None
    for token in str(profile.get("workbook", "")).replace("_", " ").replace("-", " ").split():
        if token.isdigit() and len(token) == 4:
            return token
    return None


def results_frame(res) -> pd.DataFrame:
    return getattr(res, "results", pd.DataFrame())


def show_validation(report) -> None:
    if report.passed:
        st.success(f"Validation **PASSED** — solver status `{report.solver_status}`")
    else:
        st.error(f"Validation **FAILED** — solver status `{report.solver_status}`")
    if report.infeasible_explanation:
        st.warning(report.infeasible_explanation)
    rows = [{"check": c.name, "passed": c.passed, "severity": c.severity, "detail": c.detail}
            for c in report.checks]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if report.warnings:
        st.caption("Warnings: " + " | ".join(report.warnings))


# ----------------------------------------------------------------------------
# sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("NUMBAT Rail Optimisation")
st.sidebar.caption(DISCLAIMER)

cleaned, profile = get_data()
has_data = not cleaned.empty and (profile is None or "error" not in (profile or {}))

mode = st.sidebar.selectbox(
    "Mode filter",
    ["(all)"] + sorted(cleaned["mode"].dropna().astype(str).unique().tolist())
    if has_data else ["(all)"],
)
line_options = (
    sorted(cleaned.loc[
        (mode == "(all)") | (cleaned["mode"].astype(str) == mode), "line"
    ].dropna().astype(str).unique().tolist())
    if has_data else []
)
lines = st.sidebar.multiselect("Lines", line_options, default=line_options[:])

section = st.sidebar.radio(
    "Section",
    ["Data Overview", "Data Exploration", "Optimization", "Parameters",
     "Results", "Scenarios", "Validation", "Export"],
)

model_choice = st.sidebar.radio(
    "Optimization model",
    ["Model 1 — Min service frequency",
     "Model 2 — Capacity-constrained allocation",
     "Model 3 — Weighted goal programming"],
)
MODEL_KEY = {"Model 1 — Min service frequency": "model1",
             "Model 2 — Capacity-constrained allocation": "model2",
             "Model 3 — Weighted goal programming": "model3"}[model_choice]

if st.sidebar.button("Run optimisation", type="primary"):
    st.session_state["run_requested"] = True

if st.sidebar.button("Clear results"):
    for k in ("last_result", "last_report", "last_params", "scenario_output"):
        st.session_state.pop(k, None)
    st.session_state["run_requested"] = False
    st.rerun()


# ============================================================================
# sections
# ============================================================================
if not has_data:
    missing_data_banner()
    if profile and "error" in profile:
        st.code(profile["error"])
    st.stop()

links_all, line_all, station_all = eda.split_tables(cleaned)
links = split_mode_lines(links_all, None if mode == "(all)" else mode, lines)
line_df = split_mode_lines(line_all, None if mode == "(all)" else mode, lines)

# apply the time-range chosen in the Parameters section (persists in session state)
_tsel = st.session_state.get("time_periods") or []
if _tsel:
    links = links[links["time_period"].isin(_tsel)]
    line_df = line_df[line_df["time_period"].isin(_tsel)]

if links.empty:
    st.warning("No rows for the current mode/line filter.")
    st.stop()

sp = cfg.load_scenario_parameters()
cap = capacity_df()
target_util_default = float(sp.get("target_utilization", 0.8))

# ---------------------------------------------------------------- Data overview
if section == "Data Overview":
    st.title("1 · Data overview")
    cs = (profile or {}).get("cleaned_summary", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", (profile or {}).get("workbook", "n/a"))
    c2.metric("Year", _year_from_profile(profile) or "n/a")
    c3.metric("Day types", ", ".join(cs.get("day_types", [])) or "n/a")
    c4.metric("Source", "TfL NUMBAT")
    st.caption(cfg.NUMBAT_SOURCE_URL)

    n_links = links[["line", "from_station", "to_station", "direction"]].drop_duplicates()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lines", links["line"].nunique())
    c2.metric("Stations", int(pd.unique(pd.concat(
        [links["from_station"], links["to_station"]], ignore_index=True)).size))
    c3.metric("Links", len(n_links))
    c4.metric("Time periods", links["time_period"].nunique())

    st.subheader("Detected sheets / columns")
    st.json((profile or {}).get("detected_roles", {}))

    st.subheader("Quality report")
    st.json((profile or {}).get("quality", {}))

    st.subheader("Network statistics")
    stats = network_statistics(links)
    st.write({
        "stations": stats["stations"],
        "unique_links": stats["unique_links"],
        "links_by_line": stats["links_by_line"],
        "fully_connected_by_line": stats["is_fully_connected_by_line"],
    })
    branched = {k: v["junction_stations"] for k, v in stats["branch_analysis"].items()
                if v["junction_stations"]}
    if branched:
        st.warning(
            f"Junction/branch structures detected: {branched}. If a line must be split "
            "into separate decision units, configure config/service_groups.csv."
        )
    else:
        st.info("No junctions detected — service_group defaults to line.")

    st.subheader("Operational configuration status")
    cap_status = "EMPTY — fill config/capacity_by_line.csv" if cap.empty else \
        f"{len(cap)} rows configured"
    st.write({
        "capacity_by_line.csv": cap_status,
        "service_constraints.csv": "configured" if not constraints_df().empty else "empty (no bounds)",
        "fleet_budget.csv": "configured" if not fleet_df().empty else "empty (budget constraint disabled)",
        "goal_weights.yaml": weights_cfg()["weights"],
    })
    st.caption(cs.get("link_load_note", ""))

# ---------------------------------------------------------------- Exploration
elif section == "Data Exploration":
    st.title("2 · Data exploration")
    tab_a, tab_b, tab_c, tab_d = st.tabs(
        ["Demand", "Frequency", "Utilisation", "Tables"]
    )
    with tab_a:
        st.plotly_chart(plots.demand_heatmap(links), use_container_width=True)
        st.plotly_chart(plots.demand_timeseries(links), use_container_width=True)
        f = plots.line_boarders_timeseries(line_df)
        if f:
            st.plotly_chart(f, use_container_width=True)
        st.plotly_chart(plots.line_comparison(links), use_container_width=True)
        st.plotly_chart(plots.top_demand_links(links), use_container_width=True)
        pv = eda.peak_vs_offpeak(links, sp.get("peak_windows") or [["07:00", "10:00"], ["16:00", "19:00"]])
        st.subheader("Peak vs off-peak")
        st.dataframe(pv, use_container_width=True, hide_index=True)
    with tab_b:
        st.plotly_chart(plots.frequency_timeseries(links), use_container_width=True)
        st.plotly_chart(plots.demand_frequency_scatter(links), use_container_width=True)
        st.caption("Link Frequency = scheduled/planned trains per 15-min period.")
    with tab_c:
        util = eda.baseline_utilization(links, cap)
        fig = plots.utilization_heatmap(util)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.json(eda.utilization_summary(util))
        else:
            st.warning(
                "Utilisation requires capacity_per_train — config/capacity_by_line.csv "
                "is empty. Capacities are never invented."
            )
        under = eda.underserved_periods(links, cap, target_util_default)
        st.subheader(f"Potentially underserved periods (util > {target_util_default})")
        st.dataframe(under, use_container_width=True, hide_index=True)
    with tab_d:
        st.subheader("Demand by link (top 100)")
        st.dataframe(eda.demand_by_link(links).head(100), use_container_width=True, hide_index=True)
        st.subheader("Demand by time")
        st.dataframe(eda.demand_by_time(links, line_df), use_container_width=True, hide_index=True)
        st.subheader("High-load links")
        st.dataframe(eda.high_load_links(links), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- Optimization
elif section == "Optimization":
    st.title("3 · Optimization")
    st.info(f"Selected: **{model_choice}** — set parameters in the *Parameters* "
            "section (sidebar), then press **Run optimisation**.")
    st.markdown(
        """
**Model 1 — Minimum Service Frequency Optimization**

$$\\min \\sum_{r,t} x_{r,t} \\quad
\\text{s.t.}\\quad cap_r \\, x_{r,t} \\ge D_{l,t}\\cdot m \\;\\;\\forall l\\in r,
\\;\\; \\underline{x}_{r,t}\\le x_{r,t}\\le \\overline{x}_{r,t},
\\;\\; \\sum_r x_{r,t}\\le B_t$$

with $x\\in\\mathbb{Z}_+$. Closed-form check: $x^*=\\lceil \\max_l D_{l,t}\\,m / cap_r\\rceil$.

**Model 2 — Capacity-Constrained Service Allocation**
$$\\max \\sum_{r,t} y_{r,t}\\;\\;
\\text{s.t.}\\; y\\le D^{\\text{line boarders}},\\; y\\le cap\\cdot x,\\; \\sum_r x\\le B_t$$

**Model 3 — Weighted Goal Programming**
$$\\min\\; w_1\\Sigma d^{unmet}+w_2\\Sigma d^{freq}+w_3\\Sigma d^{crowd}+w_4\\Sigma x$$
"""
    )

    # ---------------- run request ----------------
    if st.session_state.get("run_requested"):
        demand_mult = float(st.session_state.get("demand_multiplier", 1.0))
        fleet_mult = float(st.session_state.get("fleet_multiplier", 1.0))
        target_u = float(st.session_state.get("target_utilization", target_util_default))
        wts = st.session_state.get("goal_weights", {})

        ok, missing = check_capacity(links["line"].dropna().unique().tolist())
        if not ok:
            for m in missing:
                st.error(m)
            st.stop()

        try:
            data = prepare_model_data(
                links, line_df,
                demand_multiplier=demand_mult,
                target_utilization=target_u,
                fleet_multiplier=fleet_mult,
                require_capacity=True,
                require_line_demand=MODEL_KEY in {"model2", "model3"},
            )
        except ModelDataError as exc:
            st.error(str(exc))
            st.session_state["run_requested"] = False
            st.stop()

        with st.spinner(f"Solving {MODEL_KEY} with HiGHS…"):
            if MODEL_KEY == "model1":
                res = solve_model1(data, solver_name=sp.get("solver"),
                                   solver_options=sp.get("solver_options"))
                report = validate_model1(res, data)
            elif MODEL_KEY == "model2":
                res = solve_model2(data, solver_name=sp.get("solver"),
                                   solver_options=sp.get("solver_options"))
                report = validate_model2(res, data)
            else:
                res = solve_model3(data, weights_cfg=weights_cfg() if not wts else
                                   {"weights": wts,
                                    "normalize_terms": weights_cfg().get("normalize_terms", True)},
                                   target_utilization=target_u,
                                   solver_name=sp.get("solver"),
                                   solver_options=sp.get("solver_options"))
                report = validate_model3(res, data)

        params = {
            "model": MODEL_KEY,
            "mode": mode,
            "lines": lines,
            "demand_multiplier": demand_mult,
            "fleet_multiplier": fleet_mult,
            "target_utilization": target_u,
            "goal_weights": wts or weights_cfg()["weights"],
            "solver": res.solver.solver_name,
            **{k: v for k, v in getattr(res, "parameters", {}).items() if k != "model"},
        }
        run_dir = run_manifest(MODEL_KEY, res.results, params, res.solver.status)
        vpath, tpath = write_report(report, run_dir)

        st.session_state["last_result"] = res
        st.session_state["last_report"] = report
        st.session_state["last_params"] = params
        st.session_state["last_model"] = MODEL_KEY
        st.session_state["run_requested"] = False

        if not res.solver.ok:
            st.error(f"Solver status: **{res.solver.status}**")
            if res.infeasible_explanation:
                st.warning(res.infeasible_explanation)
        else:
            st.success(
                f"Solved with `{res.solver.solver_name}` → **{res.solver.status}**. "
                f"Run saved to `{run_dir}`."
            )

# ---------------------------------------------------------------- Parameters
elif section == "Parameters":
    st.title("4 · Parameters")
    st.caption("All values below are configuration inputs — not hidden assumptions. "
               "They are snapshot into outputs/runs/<timestamp>/config_snapshot/.")

    p1, p2 = st.columns(2)
    with p1:
        st.subheader("Scenario scalars")
        st.slider(
            "Demand multiplier (× NUMBAT link load / line boarders)",
            0.5, 2.0, float(st.session_state.get("demand_multiplier", 1.0)), 0.05,
            key="demand_multiplier",
        )
        st.slider(
            "Fleet / service-budget multiplier",
            0.5, 1.5, float(st.session_state.get("fleet_multiplier", 1.0)), 0.05,
            key="fleet_multiplier",
        )
        st.slider(
            "Target utilisation (occupancy)", 0.3, 1.2,
            float(st.session_state.get("target_utilization", target_util_default)),
            0.05, key="target_utilization",
        )
        st.caption(f"Defaults come from config/scenario_parameters.yaml "
                   f"(target_utilization={target_util_default}).")
    with p2:
        st.subheader("Time / line scope")

        def _tp_key(p: str):
            try:
                h, m = str(p).split("-")[0].split(":")
                v = int(h) * 60 + int(m)
                return v + 24 * 60 if int(h) < 4 else v
            except Exception:
                return 10**6

        periods_sorted = sorted(links["time_period"].dropna().unique().tolist(), key=_tp_key)
        existing = [p for p in (st.session_state.get("time_periods") or [])
                    if p in periods_sorted]
        st.multiselect("Time periods (empty = all)", periods_sorted,
                       default=existing, key="time_periods")
        st.write(f"Rows in scope: **{len(links)}** link-periods, "
                 f"**{links['line'].nunique()}** lines "
                 f"(scope applies on the next interaction)")

    st.subheader("Frequency bounds (service_constraints)")
    sdf = constraints_df()
    if sdf.empty:
        st.info("config/service_constraints.csv is empty — models run without "
                "explicit min/max/target frequency bounds (target defaults to the "
                "observed NUMBAT frequency).")
    else:
        st.dataframe(sdf, use_container_width=True, hide_index=True)

    st.subheader("Train capacity (capacity_by_line)")
    if cap.empty:
        st.error("config/capacity_by_line.csv is EMPTY. Capacity per train is a "
                 "**mandatory manual input** — download source: TfL rolling stock / "
                 "vehicle specifications, record `source` and `source_url`.")
    else:
        st.dataframe(cap, use_container_width=True, hide_index=True)

    st.subheader("Fleet / train-equivalent service budget (fleet_budget)")
    fdf = fleet_df()
    if fdf.empty:
        st.info("No budget configured → budget constraint disabled (documented). "
                "This is a simplified train-equivalent service budget, NOT the "
                "TfL physical fleet.")
    else:
        st.dataframe(fdf, use_container_width=True, hide_index=True)

    st.subheader("Goal-programming weights (goal_weights.yaml)")
    wcfg = weights_cfg()
    wcols = st.columns(4)
    gw = st.session_state.setdefault("goal_weights", dict(wcfg["weights"]))
    gw["w1_unmet_demand"] = wcols[0].number_input(
        "w1 unmet demand", 0.0, 100.0, float(gw.get("w1_unmet_demand", 1.0)), 0.1, key="w1")
    gw["w2_frequency_target"] = wcols[1].number_input(
        "w2 frequency target", 0.0, 100.0, float(gw.get("w2_frequency_target", 1.0)), 0.1, key="w2")
    gw["w3_crowding"] = wcols[2].number_input(
        "w3 crowding", 0.0, 100.0, float(gw.get("w3_crowding", 1.0)), 0.1, key="w3")
    gw["w4_service_usage"] = wcols[3].number_input(
        "w4 service usage", 0.0, 100.0, float(gw.get("w4_service_usage", 0.1)), 0.05, key="w4")
    st.session_state["goal_weights"] = gw
    st.caption(f"normalize_terms = {wcfg.get('normalize_terms')} "
               "(edit config/goal_weights.yaml to change the flag).")

# ---------------------------------------------------------------- Results
elif section == "Results":
    st.title("5 · Results")
    res = st.session_state.get("last_result")
    report = st.session_state.get("last_report")
    if res is None:
        st.info("No optimisation run yet — go to **Optimization** and press "
                "*Run optimisation*.")
    else:
        if report is not None:
            show_validation(report)
        df = results_frame(res)
        if df.empty:
            st.error("Empty result table.")
            if getattr(res, "infeasible_explanation", None):
                st.warning(res.infeasible_explanation)
        else:
            xcol = ("optimized_frequency" if "optimized_frequency" in df.columns
                    else "trains_allocated")
            st.subheader("Before vs after (baseline = NUMBAT scheduled frequency)")
            f = plots.baseline_vs_optimized(df, xcol=xcol)
            if f:
                st.plotly_chart(f, use_container_width=True)
            f2 = plots.frequency_change_by_group(df, xcol=xcol)
            if f2:
                st.plotly_chart(f2, use_container_width=True)

            m = {}
            x = pd.to_numeric(df[xcol], errors="coerce")
            m["total train-equivalents (baseline)"] = float(
                pd.to_numeric(df["observed_frequency"], errors="coerce").sum())
            m["total train-equivalents (optimised)"] = float(x.sum())
            m["average frequency"] = float(x.mean())
            m["peak frequency"] = float(x.max())
            if "demand_satisfaction" in df.columns:
                m["mean demand satisfaction"] = float(
                    pd.to_numeric(df["demand_satisfaction"], errors="coerce").mean())
            if "unmet_demand" in df.columns:
                m["unmet demand"] = float(
                    pd.to_numeric(df["unmet_demand"], errors="coerce").sum())
            if "optimized_utilization" in df.columns:
                m["max utilisation (optimised)"] = float(
                    pd.to_numeric(df["optimized_utilization"], errors="coerce").max())
                m["mean utilisation (optimised)"] = float(
                    pd.to_numeric(df["optimized_utilization"], errors="coerce").mean())
            if "observed_frequency" in df.columns:
                delta = x - pd.to_numeric(df["observed_frequency"], errors="coerce")
                m["groups needing increased service"] = int((delta > 1e-9).sum())
                m["groups where service could be reduced"] = int((delta < -1e-9).sum())
            st.subheader("Summary metrics")
            cols = st.columns(4)
            for i, (k, v) in enumerate(m.items()):
                cols[i % 4].metric(k, f"{v:,.2f}" if isinstance(v, float) else v)
            st.caption("Reductions are **model recommendations under the configured "
                       "assumptions**, not operationally deployable cuts.")
            st.subheader("Full result table")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download results CSV",
                               df.to_csv(index=False).encode("utf-8"),
                               file_name="results.csv", mime="text/csv")

# ---------------------------------------------------------------- Scenarios
elif section == "Scenarios":
    st.title("6 · Scenario analysis")
    st.caption("Scenarios rerun the SAME model code with parameter overrides "
               "(config/scenario_parameters.yaml).")
    base_model = st.selectbox("Base model for demand/fleet scenarios",
                              ["model1", "model2", "model3"])
    if st.button("Run scenario sweep", type="primary"):
        with st.spinner("Running scenarios…"):
            summary, runs = run_scenarios(
                links, line_df, model=base_model, progress=lambda n: st.write(f"– {n}")
            )
        st.session_state["scenario_output"] = summary
        out = cfg.OUTPUTS_DIR / "scenario_results.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)
        st.success(f"Saved {out}")
    so = st.session_state.get("scenario_output")
    if so is not None:
        st.dataframe(so, use_container_width=True, hide_index=True)
        st.download_button("Download scenario CSV", so.to_csv(index=False).encode("utf-8"),
                           file_name="scenario_results.csv", mime="text/csv")
        numeric_cols = [c for c in so.columns if so[c].dtype.kind in "fi"
                        and c in {"total_train_equivalents", "demand_satisfaction",
                                  "unmet_demand", "mean_utilization", "max_utilization",
                                  "crowded_links"}]
        if numeric_cols and "scenario" in so.columns:
            st.line_chart(so.set_index("scenario")[numeric_cols])
        bad = so[so["status"].astype(str).isin(["infeasible", "error", "skipped"])] \
            if "status" in so.columns else pd.DataFrame()
        if not bad.empty:
            st.warning("Some scenarios did not solve — see skip reasons:")
            st.dataframe(bad, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- Validation
elif section == "Validation":
    st.title("7 · Validation")
    report = st.session_state.get("last_report")
    if report is None:
        st.info("Run an optimisation first.")
        # fall back to saved report
        saved = cfg.OUTPUTS_DIR / "validation_report.json"
        if saved.exists():
            st.subheader("Last saved validation report")
            st.json(json.loads(saved.read_text(encoding="utf-8")))
    else:
        show_validation(report)
        st.subheader("Report files")
        st.code(report.to_text())

# ---------------------------------------------------------------- Export
elif section == "Export":
    st.title("8 · Export")
    res = st.session_state.get("last_result")
    report = st.session_state.get("last_report")
    params = st.session_state.get("last_params", {})

    if res is None:
        st.info("Run an optimisation first.")
    else:
        df = results_frame(res)
        c1, c2, c3 = st.columns(3)
        c1.download_button("Results CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="results.csv", mime="text/csv")
        c2.download_button("Model run JSON",
                           json.dumps({"parameters": params,
                                       "solver": {"name": res.solver.solver_name,
                                                  "status": res.solver.status,
                                                  "objective": res.solver.objective},
                                       "results": df.to_dict(orient="records")},
                                      indent=2, default=str).encode("utf-8"),
                           file_name="model_run.json", mime="application/json")
        if report is not None:
            c3.download_button("Validation JSON",
                               json.dumps(report.to_dict(), indent=2, default=str).encode("utf-8"),
                               file_name="validation_report.json",
                               mime="application/json")
            st.download_button("Validation TXT",
                               report.to_text().encode("utf-8"),
                               file_name="validation_report.txt", mime="text/plain")

        st.subheader("Figures")
        fig_files = sorted(cfg.FIGURES_DIR.glob("*.html")) + sorted(cfg.FIGURES_DIR.glob("*.png"))
        if fig_files:
            for f in fig_files:
                st.download_button(f"Download {f.name}", f.read_bytes(),
                                   file_name=f.name, key=f"fig_{f.name}")
        else:
            st.info("No figures yet — run scripts/run_eda.py or a model script.")

        st.subheader("Latest runs")
        runs = sorted(cfg.RUNS_DIR.glob("*"), reverse=True)[:10]
        if runs:
            st.write("\n".join(str(r) for r in runs))
        st.caption("Each run directory contains config_snapshot/, results.csv, "
                   "validation_report.json, dataset_profile.json and run_metadata.json.")

st.sidebar.caption(
    f"Dataset: {(profile or {}).get('workbook', 'n/a')} · "
    f"day types: {', '.join((profile or {}).get('cleaned_summary', {}).get('day_types', []))}"
)
