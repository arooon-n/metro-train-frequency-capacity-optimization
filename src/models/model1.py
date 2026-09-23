"""MODEL 1 — Minimum Service Frequency Optimization (MILP).

Decision : x[r,t] in Z+  = trains scheduled per 15-minute period for
           service group r (line by default).
Objective: minimise total train-equivalent service   sum_{r,t} x[r,t]
Subject  : cap[r] * x[r,t] >= D[l,t] * demand_multiplier   for every link l in r
           min_freq[r,t] <= x[r,t] <= max_freq[r,t]        (when configured)
           sum_r x[r,t] <= fleet_budget[t]                 (only if configured)

Because cap[r]*x[r,t] is identical for every link of the same group, the family
of per-link capacity constraints is mathematically equivalent to a single
constraint driven by the MAXIMUM link load of that (group, period). We build
the max-link form (identical feasible set, far fewer constraints) and the
validator re-checks EVERY link afterwards.

Closed-form sanity check (no budget, no binding upper bound):

    required_frequency[r,t] = ceil( max_link_load[r,t] * demand_multiplier
                                    / capacity[r] )

Solver: Pyomo + HiGHS (highspy) by default — no paid solver required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from src.models.common import ModelData, ModelDataError
from src.models.solver_utils import SolverResult, solve_model
from src.utils import config as cfg


@dataclass
class Model1Result:
    results: pd.DataFrame
    solver: SolverResult
    analytical: pd.DataFrame
    parameters: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    infeasible_explanation: str | None = None


def analytical_required_frequency(data: ModelData) -> pd.DataFrame:
    """Closed-form solution: ceil(max scaled link load / capacity), clipped to bounds."""
    agg = data.scaled_max_load()
    rows = []
    for row in agg.itertuples(index=False):
        cap = data.capacity.get(row.service_group)
        if cap is None or cap <= 0:
            continue
        x = float(np.ceil(row.max_link_load / cap - 1e-9))
        lo = data.min_freq.get((row.service_group, row.time_period))
        hi = data.max_freq.get((row.service_group, row.time_period))
        if lo is not None:
            x = max(x, float(lo))
        if hi is not None:
            x = min(x, float(hi))
        rows.append(
            {
                "service_group": row.service_group,
                "time_period": row.time_period,
                "max_link_load": row.max_link_load,
                "capacity_per_train": cap,
                "analytical_frequency": max(x, 0.0),
            }
        )
    return pd.DataFrame(rows)


def explain_infeasibility(data: ModelData, agg: pd.DataFrame) -> str | None:
    """Human-readable reason when the model cannot be feasible."""
    problems = []
    for row in agg.itertuples(index=False):
        cap = data.capacity.get(row.service_group)
        if cap is None:
            continue
        need = float(np.ceil(row.max_link_load / cap - 1e-9)) if cap > 0 else np.inf
        hi = data.max_freq.get((row.service_group, row.time_period))
        lo = data.min_freq.get((row.service_group, row.time_period))
        if hi is not None and need > hi:
            problems.append(
                f"[{row.service_group} @ {row.time_period}] needs >= {need:.0f} trains "
                f"to cover max link load {row.max_link_load:.1f} with capacity {cap:.0f}, "
                f"but max_frequency_per_15min = {hi:.0f} (capacity < demand at the "
                f"frequency ceiling)."
            )
        if lo is not None and data.fleet_budget:
            pass
    if data.fleet_budget:
        budget_problems = _budget_conflicts(data, agg)
        problems.extend(budget_problems)
    if not problems:
        return None
    head = (
        "Model 1 is INFEASIBLE under the current configuration. Likely causes:\n"
    )
    body = "\n".join(f"  - {p}" for p in problems[:10])
    more = f"\n  ... and {len(problems) - 10} more." if len(problems) > 10 else ""
    tail = (
        "\nHow to resolve (choose one, do NOT edit solver output):\n"
        "  - raise max_frequency_per_15min in config/service_constraints.csv\n"
        "  - review capacity_per_train in config/capacity_by_line.csv (must be sourced)\n"
        "  - raise / remove fleet_budget in config/fleet_budget.csv\n"
        "  - lower demand_multiplier in your scenario configuration"
    )
    return head + body + more + tail


def _budget_conflicts(data: ModelData, agg: pd.DataFrame) -> list[str]:
    out = []
    need_sum = agg.groupby("time_period")["max_link_load"].sum()  # rough lower bound
    for period, _ in need_sum.items():
        budget = data.fleet_budget.get(str(period))
        if budget is None:
            continue
        rows = agg[agg["time_period"] == period]
        min_need = 0.0
        for row in rows.itertuples(index=False):
            cap = data.capacity.get(row.service_group)
            if cap and cap > 0:
                need = np.ceil(row.max_link_load / cap - 1e-9)
                lo = data.min_freq.get((row.service_group, period))
                min_need += max(need, lo if lo is not None else 0)
        if min_need > budget:
            out.append(
                f"[{period}] minimum capacity-covering service needs ~{min_need:.0f} "
                f"train-equivalents but fleet_budget = {budget:.0f}."
            )
    return out


def solve_model1(
    data: ModelData,
    *,
    solver_name: str | None = None,
    solver_options: dict | None = None,
) -> Model1Result:
    if not data.capacity:
        raise ModelDataError(
            "Model 1 requires capacity_per_train — fill config/capacity_by_line.csv."
        )

    agg = data.scaled_max_load()
    periods = sorted(set(agg["time_period"]) | set(data.periods))
    groups = sorted(set(agg["service_group"]) | set(data.service_groups))
    keys = [(r, t) for r in groups for t in periods]
    load = {
        (r, t): float(v)
        for (r, t), v in agg.set_index(["service_group", "time_period"])["max_link_load"].items()
    }

    m = pyo.ConcreteModel("Model1_MinimumServiceFrequency")
    m.R = pyo.Set(initialize=groups)
    m.T = pyo.Set(initialize=periods)

    def bounds(model, r, t):
        lo = data.min_freq.get((r, t))
        hi = data.max_freq.get((r, t))
        lb = 0 if lo is None else lo
        ub = hi
        return (lb, ub)

    m.x = pyo.Var(keys, domain=pyo.NonNegativeIntegers, bounds=bounds)

    # capacity coverage (max-link form == per-link family)
    def cap_rule(model, r, t):
        cap = data.capacity.get(r)
        if cap is None:
            return pyo.Constraint.Skip
        d = load.get((r, t), 0.0)
        return cap * m.x[r, t] >= d

    m.cap_con = pyo.Constraint([(r, t) for (r, t) in keys], rule=cap_rule)

    # optional train-equivalent service budget
    budget_active = bool(data.fleet_budget)
    if budget_active:
        def budget_rule(model, t):
            if t not in data.fleet_budget:
                return pyo.Constraint.Skip
            return pyo.quicksum(m.x[r, t] for r in groups) <= data.fleet_budget[t]

        m.budget = pyo.Constraint(periods, rule=budget_rule)

    m.objective = pyo.Objective(
        expr=pyo.quicksum(m.x[r, t] for (r, t) in keys), sense=pyo.minimize
    )

    if solver_options is None:
        sp = cfg.load_scenario_parameters()
        solver_name = solver_name or sp.get("solver")
        solver_options = sp.get("solver_options") or None

    sres = solve_model(m, solver_name, solver_options)

    warnings = list(data.warnings)
    if not sres.ok:
        expl = explain_infeasibility(data, agg) if sres.status == "infeasible" else (
            f"Solver returned status '{sres.status}': {sres.message}"
        )
        empty = _empty_results(agg, data, sres.status)
        return Model1Result(
            results=empty,
            solver=sres,
            analytical=analytical_required_frequency(data),
            parameters=_params(data, budget_active),
            warnings=warnings,
            infeasible_explanation=expl,
        )

    # ---- extract ------------------------------------------------------
    x_vals = {(r, t): int(round(pyo.value(m.x[r, t]))) for (r, t) in keys}
    analytical = analytical_required_frequency(data)
    ana = analytical.set_index(["service_group", "time_period"])["analytical_frequency"].to_dict()

    rows = []
    for (r, t) in keys:
        cap = data.capacity.get(r)
        obs = data.observed_freq.get((r, t))
        opt = x_vals[(r, t)]
        d = load.get((r, t), 0.0)
        base_u = (d / (cap * obs)) if cap and obs and obs > 0 else np.nan
        # baseline utilisation should use UNSCALED load for comparison
        d_raw = d / data.demand_multiplier if data.demand_multiplier else d
        if cap and obs and obs > 0:
            base_u = d_raw / (cap * obs)
        opt_u = (d / (cap * opt)) if cap and opt > 0 else (np.inf if cap and d > 0 else np.nan)
        sat = float(min(1.0, (cap * opt) / d)) if cap and d > 0 else 1.0
        a = ana.get((r, t))
        rows.append(
            {
                "time_period": t,
                "line": ", ".join(data.group_to_lines.get(r, [r])),
                "service_group": r,
                "observed_frequency": obs,
                "optimized_frequency": opt,
                "frequency_change": (opt - obs) if obs is not None else np.nan,
                "max_link_load": d_raw,
                "max_link_load_scaled": d,
                "capacity_per_train": cap,
                "baseline_utilization": base_u,
                "optimized_utilization": opt_u if np.isfinite(opt_u) else np.nan,
                "demand_satisfaction": sat,
                "analytical_frequency": a,
                "matches_analytical": (a is None or abs(a - opt) < 1e-6),
                "solver_status": sres.status,
            }
        )

    results = pd.DataFrame(rows)
    n_mismatch = int((~results["matches_analytical"].fillna(True)).sum())
    if n_mismatch:
        warnings.append(
            f"{n_mismatch} group-periods differ from the closed-form solution — "
            "check min_frequency bounds (min bounds can raise the optimum above "
            "ceil(max_link_load/capacity))."
        )
    return Model1Result(
        results=results,
        solver=sres,
        analytical=analytical,
        parameters=_params(data, budget_active),
        warnings=warnings,
        infeasible_explanation=None,
    )


def _params(data: ModelData, budget_active: bool) -> dict[str, Any]:
    return {
        "model": "Model 1 — Minimum Service Frequency Optimization",
        "demand_multiplier": data.demand_multiplier,
        "capacity_by_service_group": {k: data.capacity[k] for k in sorted(data.capacity)},
        "service_budget_active": budget_active,
        "fleet_budget": data.fleet_budget,
        "min_frequency_configured": bool(data.min_freq),
        "max_frequency_configured": bool(data.max_freq),
        "n_service_groups": len(data.service_groups),
        "n_time_periods": len(data.periods),
    }


def _empty_results(agg: pd.DataFrame, data: ModelData, status: str) -> pd.DataFrame:
    rows = []
    for row in agg.itertuples(index=False):
        rows.append(
            {
                "time_period": row.time_period,
                "line": ", ".join(data.group_to_lines.get(row.service_group, [row.service_group])),
                "service_group": row.service_group,
                "observed_frequency": data.observed_freq.get(
                    (row.service_group, row.time_period)
                ),
                "optimized_frequency": np.nan,
                "frequency_change": np.nan,
                "max_link_load": row.max_link_load,
                "capacity_per_train": data.capacity.get(row.service_group),
                "baseline_utilization": np.nan,
                "optimized_utilization": np.nan,
                "demand_satisfaction": np.nan,
                "solver_status": status,
            }
        )
    return pd.DataFrame(rows)
