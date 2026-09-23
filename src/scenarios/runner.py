"""Reusable scenario runner.

Scenarios are pure parameter overrides on top of the SAME model code — no
optimisation logic is duplicated. Defined by config/scenario_parameters.yaml:

  1. baseline                 demand x1.00, fleet x1.00, target util from config
  2. demand_+10%              demand x increased_demand_multiplier
  3. demand_+20%              demand x high_demand_multiplier
  4. reduced_fleet            fleet x reduced_fleet_multiplier (needs budget file)
  5. target_utilization sweeps (Model 3 only)

Each scenario records a comparable set of metrics + change vs baseline and the
full results table is saved to outputs/scenario_results.csv.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.models.common import ModelData, prepare_model_data
from src.models.model1 import solve_model1
from src.models.model2 import solve_model2
from src.models.model3 import solve_model3
from src.utils import config as cfg


@dataclass
class Scenario:
    name: str
    description: str
    demand_multiplier: float = 1.0
    fleet_multiplier: float = 1.0
    target_utilization: float | None = None
    model: str = "model1"          # model1 | model2 | model3
    active: bool = True
    skip_reason: str | None = None


@dataclass
class ScenarioRun:
    scenario: Scenario
    solver_status: str
    metrics: dict[str, Any]
    results: pd.DataFrame
    infeasible_explanation: str | None = None
    error: str | None = None


def build_scenarios(
    params: dict[str, Any] | None = None,
    model: str = "model1",
) -> list[Scenario]:
    p = params or cfg.load_scenario_parameters()
    target_util = float(p.get("target_utilization", 0.8))
    fleet_base = float(p.get("fleet_multiplier", 1.0))
    scen = [
        Scenario("baseline", "Observed NUMBAT demand, configured budget",
                 float(p.get("baseline_demand_multiplier", 1.0)), fleet_base,
                 target_util, model),
        Scenario("demand_+10%", f"Demand x {p.get('increased_demand_multiplier', 1.10)}",
                 float(p.get("increased_demand_multiplier", 1.10)), fleet_base,
                 target_util, model),
        Scenario("demand_+20%", f"Demand x {p.get('high_demand_multiplier', 1.20)}",
                 float(p.get("high_demand_multiplier", 1.20)), fleet_base,
                 target_util, model),
        Scenario("reduced_fleet",
                 f"Fleet/service budget x {p.get('reduced_fleet_multiplier', 0.85)}",
                 float(p.get("baseline_demand_multiplier", 1.0)),
                 fleet_base * float(p.get("reduced_fleet_multiplier", 0.85)),
                 target_util, model),
    ]
    for i, u in enumerate(p.get("target_utilization_scenarios") or []):
        scen.append(
            Scenario(
                name=f"target_util_{float(u):.2f}",
                description=f"Goal programming with target utilisation = {u}",
                demand_multiplier=float(p.get("baseline_demand_multiplier", 1.0)),
                fleet_multiplier=fleet_base,
                target_utilization=float(u),
                model="model3",
            )
        )
    return scen


def _solve(data: ModelData, model: str, scenario: Scenario) -> Any:
    if model == "model1":
        return solve_model1(data)
    if model == "model2":
        return solve_model2(data, require_budget=False)
    if model == "model3":
        return solve_model3(data, target_utilization=scenario.target_utilization)
    raise ValueError(f"Unknown model '{model}' (expected model1|model2|model3)")


def scenario_metrics(
    result: Any, data: ModelData, scenario: Scenario
) -> dict[str, Any]:
    df = getattr(result, "results", pd.DataFrame())
    if df is None or df.empty or not result.solver.ok:
        return {
            "solver_status": result.solver.status,
            "total_train_equivalents": np.nan,
            "mean_frequency": np.nan,
            "peak_frequency": np.nan,
            "demand_satisfaction": np.nan,
            "unmet_demand": np.nan,
            "mean_utilization": np.nan,
            "max_utilization": np.nan,
            "crowded_links": np.nan,
            "groups_increased": np.nan,
            "groups_reduced": np.nan,
        }

    xcol = "optimized_frequency" if "optimized_frequency" in df.columns else "trains_allocated"
    x = pd.to_numeric(df[xcol], errors="coerce")

    # crowding at link level under this scenario's allocation
    xmap = {}
    for sg, tp, v in zip(df["service_group"], df["time_period"], x):
        if pd.notna(v):
            xmap[(sg, tp)] = float(v)
    crowded = 0
    util_vals = []
    links = data.link_rows_for_group_time()
    for row in links.itertuples(index=False):
        cap = data.capacity.get(row.service_group)
        xv = xmap.get((row.service_group, row.time_period))
        if cap is None or xv is None or cap * xv <= 0:
            continue
        u = float(row.link_load) / (cap * xv)
        util_vals.append(u)
        if u > data.target_utilization + 1e-9:
            crowded += 1

    if "demand_served" in df.columns and df["demand_served"].notna().any():
        served = float(pd.to_numeric(df["demand_served"], errors="coerce").sum())
        dsc_ser = df["demand_scaled"] if "demand_scaled" in df.columns else df.get("demand")
        dsc = float(pd.to_numeric(dsc_ser, errors="coerce").fillna(0).sum())
        unmet = float(pd.to_numeric(df["unmet_demand"], errors="coerce").fillna(0).sum())
        satisfaction = served / dsc if dsc else 1.0
    else:
        # Model 1: satisfaction from link capacity coverage
        sat = []
        for row in links.itertuples(index=False):
            cap = data.capacity.get(row.service_group)
            xv = xmap.get((row.service_group, row.time_period))
            if cap and xv is not None and row.link_load > 0:
                sat.append(min(1.0, (cap * xv) / float(row.link_load)))
            satisfaction = float(np.mean(sat)) if sat else np.nan
            dsc = float(links["link_load"].sum())

            def _unmet() -> float:
                total = 0.0
                for row in links.itertuples(index=False):
                    cap = data.capacity.get(row.service_group, 0) or 0
                    xv = xmap.get((row.service_group, row.time_period), 0) or 0
                    total += max(0.0, float(row.link_load) - cap * xv)
                return total

            unmet = _unmet()


    return {
        "solver_status": result.solver.status,
        "total_train_equivalents": float(x.sum()),
        "mean_frequency": float(x.mean()),
        "peak_frequency": float(x.max()),
        "demand_satisfaction": satisfaction,
        "unmet_demand": unmet,
        "mean_utilization": float(np.mean(util_vals)) if util_vals else np.nan,
        "max_utilization": float(np.max(util_vals)) if util_vals else np.nan,
        "crowded_links": int(crowded),
        "groups_increased": None,
        "groups_reduced": None,
    }


def run_scenarios(
    links: pd.DataFrame,
    line_demand: pd.DataFrame,
    *,
    model: str = "model1",
    params: dict[str, Any] | None = None,
    solver_name: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, ScenarioRun]]:
    p = params or cfg.load_scenario_parameters()
    scen_list = build_scenarios(p, model=model)
    base_cap = cfg.load_capacity_config()
    fleet_df = cfg.load_fleet_budget()
    has_budget = not fleet_df.empty

    runs: dict[str, ScenarioRun] = {}
    rows: list[dict[str, Any]] = []
    baseline_x: pd.DataFrame | None = None

    for sc in scen_list:
        if progress:
            progress(sc.name)
        if sc.model == "model2" and not has_budget and sc.name == "reduced_fleet":
            sc.active = False
            sc.skip_reason = "fleet_budget.csv is empty — reduced-fleet scenario disabled"
        if sc.name == "reduced_fleet" and not has_budget:
            runs[sc.name] = ScenarioRun(
                scenario=sc, solver_status="skipped", metrics={}, results=pd.DataFrame(),
                error=sc.skip_reason or "no fleet budget configured",
            )
            rows.append({"scenario": sc.name, "description": sc.description,
                         "status": "skipped", "skip_reason": sc.skip_reason})
            continue

        try:
            data = prepare_model_data(
                links, line_demand,
                demand_multiplier=sc.demand_multiplier,
                target_utilization=sc.target_utilization or float(p.get("target_utilization", 0.8)),
                fleet_multiplier=sc.fleet_multiplier,
                require_capacity=True,
                require_line_demand=(sc.model in {"model2", "model3"}),
            )
            result = _solve(data, sc.model, sc)
        except Exception as exc:  # config/data problems must not kill the sweep
            runs[sc.name] = ScenarioRun(
                scenario=sc, solver_status="error", metrics={}, results=pd.DataFrame(),
                error=str(exc),
            )
            rows.append({"scenario": sc.name, "description": sc.description,
                         "status": "error", "skip_reason": str(exc)})
            continue

        metrics = scenario_metrics(result, data, sc)
        if sc.name == "baseline" and getattr(result, "results", None) is not None \
                and not result.results.empty:
            xcol = "optimized_frequency" if "optimized_frequency" in result.results.columns \
                else "trains_allocated"
            baseline_x = result.results[["service_group", "time_period", xcol]].rename(
                columns={xcol: "baseline_frequency"}
            )

        if baseline_x is not None and getattr(result, "results", None) is not None \
                and not result.results.empty:
            xcol = "optimized_frequency" if "optimized_frequency" in result.results.columns \
                else "trains_allocated"
            merged = result.results.merge(
                baseline_x, on=["service_group", "time_period"], how="left"
            )
            delta = pd.to_numeric(merged[xcol], errors="coerce") - pd.to_numeric(
                merged["baseline_frequency"], errors="coerce"
            )
            metrics["groups_increased"] = int((delta > 1e-9).sum())
            metrics["groups_reduced"] = int((delta < -1e-9).sum())
            metrics["delta_train_equivalents_vs_baseline"] = float(delta.sum())

        runs[sc.name] = ScenarioRun(
            scenario=sc,
            solver_status=result.solver.status,
            metrics=metrics,
            results=result.results,
            infeasible_explanation=getattr(result, "infeasible_explanation", None),
        )
        row = {
            "scenario": sc.name,
            "description": sc.description,
            "model": sc.model,
            "demand_multiplier": sc.demand_multiplier,
            "fleet_multiplier": sc.fleet_multiplier,
            "target_utilization": sc.target_utilization,
            "status": result.solver.status,
            **metrics,
        }
        if sc.name != "baseline":
            base = runs.get("baseline")
            if base and base.metrics:
                for k in (
                    "total_train_equivalents", "demand_satisfaction", "unmet_demand",
                    "mean_utilization", "max_utilization", "crowded_links",
                ):
                    if k in metrics and k in base.metrics and base.metrics[k] is not None:
                        bv, mv = base.metrics[k], metrics[k]
                        if bv is not None and mv is not None and not (
                            isinstance(bv, float) and np.isnan(bv)
                        ):
                            row[f"delta_{k}_vs_baseline"] = (
                                mv - bv if not isinstance(mv, str) else None
                            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary, runs
