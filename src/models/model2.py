"""MODEL 2 — Capacity-Constrained Service Allocation (MILP).

Given a limited train-equivalent service budget per period, allocate integer
train frequency x[r,t] to maximise SERVED LINE-BOARDING demand y[r,t].

    max  sum_{r,t} y[r,t]
    s.t. y[r,t] <= D[r,t]                 (cannot serve more than demanded)
         y[r,t] <= cap[r] * x[r,t]        (cannot serve more than capacity)
         sum_r x[r,t] <= fleet_budget[t]  (budget REQUIRED for this model's
                                           purpose; auto-scaled by fleet_multiplier)
         min_freq <= x <= max_freq,  x integer, y >= 0

D[r,t] is the NUMBAT LINE BOARDERS aggregated to the service group — i.e. the
objective measures "served line-boarding demand" / line-level demand coverage.
It is NOT a count of unique passenger journeys, and link loads are never summed
into journey counts.
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
class Model2Result:
    results: pd.DataFrame
    solver: SolverResult
    parameters: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    infeasible_explanation: str | None = None


def solve_model2(
    data: ModelData,
    *,
    require_budget: bool = False,
    solver_name: str | None = None,
    solver_options: dict | None = None,
) -> Model2Result:
    if not data.capacity:
        raise ModelDataError(
            "Model 2 requires capacity_per_train — fill config/capacity_by_line.csv."
        )

    demand_df = data.scaled_group_demand()
    periods = sorted(set(demand_df["time_period"]) | set(data.periods))
    groups = sorted(set(demand_df["service_group"]) | set(data.service_groups))
    keys = [(r, t) for r in groups for t in periods]
    demand = {
        (r, t): float(v)
        for (r, t), v in demand_df.set_index(["service_group", "time_period"])["line_boarders"].items()
    }

    if require_budget and not data.fleet_budget:
        raise ModelDataError(
            "Model 2 needs a train-equivalent service budget to be meaningful. "
            "Add rows to config/fleet_budget.csv (or use scenario fleet_multiplier "
            "with a configured budget). The budget is a simplified train-equivalent "
            "service budget, not the TfL physical fleet."
        )

    warnings = list(data.warnings)
    if not data.fleet_budget:
        warnings.append(
            "No fleet/service budget configured — Model 2 allocates up to the "
            "maximum frequency bounds only (budget constraint skipped)."
        )

    m = pyo.ConcreteModel("Model2_CapacityConstrainedAllocation")
    m.R = pyo.Set(initialize=groups)
    m.T = pyo.Set(initialize=periods)

    def x_bounds(model, r, t):
        lo = data.min_freq.get((r, t))
        hi = data.max_freq.get((r, t))
        return (0 if lo is None else lo, hi)

    m.x = pyo.Var(keys, domain=pyo.NonNegativeIntegers, bounds=x_bounds)
    m.y = pyo.Var(keys, domain=pyo.NonNegativeReals)

    def y_demand_rule(model, r, t):
        return m.y[r, t] <= demand.get((r, t), 0.0)

    def y_cap_rule(model, r, t):
        cap = data.capacity.get(r)
        if cap is None:
            return pyo.Constraint.Skip
        return m.y[r, t] <= cap * m.x[r, t]

    m.y_demand = pyo.Constraint(keys, rule=y_demand_rule)
    m.y_cap = pyo.Constraint(keys, rule=y_cap_rule)

    budget_active = bool(data.fleet_budget)
    if budget_active:
        def budget_rule(model, t):
            if t not in data.fleet_budget:
                return pyo.Constraint.Skip
            return pyo.quicksum(m.x[r, t] for r in groups) <= data.fleet_budget[t]

        m.budget = pyo.Constraint(periods, rule=budget_rule)

    m.objective = pyo.Objective(
        expr=pyo.quicksum(m.y[r, t] for (r, t) in keys), sense=pyo.maximize
    )

    if solver_options is None:
        sp = cfg.load_scenario_parameters()
        solver_name = solver_name or sp.get("solver")
        solver_options = sp.get("solver_options") or None

    sres = solve_model(m, solver_name, solver_options)

    parameters = {
        "model": "Model 2 — Capacity-Constrained Service Allocation",
        "objective": "maximise served line-boarding demand (line-level coverage)",
        "demand_multiplier": data.demand_multiplier,
        "service_budget_active": budget_active,
        "fleet_budget": data.fleet_budget,
        "capacity_by_service_group": {k: data.capacity[k] for k in sorted(data.capacity)},
        "n_service_groups": len(groups),
        "n_time_periods": len(periods),
    }

    if not sres.ok:
        expl = None
        if sres.status == "infeasible":
            expl = (
                "Model 2 is INFEASIBLE. Typical causes:\n"
                "  - min_frequency exceeds max_frequency for some (group, period)\n"
                "  - fleet_budget[t] is smaller than sum of minimum frequencies\n"
                "Check config/service_constraints.csv and config/fleet_budget.csv."
            )
        else:
            expl = f"Solver returned status '{sres.status}': {sres.message}"
        return Model2Result(
            results=_empty(demand_df, data, sres.status),
            solver=sres,
            parameters=parameters,
            warnings=warnings,
            infeasible_explanation=expl,
        )

    rows = []
    for (r, t) in keys:
        cap = data.capacity.get(r)
        d_raw = demand.get((r, t), 0.0)
        d_base = d_raw / data.demand_multiplier if data.demand_multiplier else d_raw
        x = int(round(pyo.value(m.x[r, t])))
        y = float(pyo.value(m.y[r, t]))
        supplied = (cap * x) if cap else np.nan
        rows.append(
            {
                "time_period": t,
                "line": ", ".join(data.group_to_lines.get(r, [r])),
                "service_group": r,
                "demand": d_base,                       # unscaled line boarders
                "demand_scaled": d_raw,
                "trains_allocated": x,
                "observed_frequency": data.observed_freq.get((r, t)),
                "capacity": cap,
                "capacity_supplied": supplied,
                "demand_served": y,
                "unmet_demand": d_raw - y,
                "coverage_ratio": (y / d_raw) if d_raw > 0 else 1.0,
                "utilization": (y / supplied) if supplied and supplied > 0 else np.nan,
                "solver_status": sres.status,
            }
        )

    results = pd.DataFrame(rows)
    return Model2Result(
        results=results, solver=sres, parameters=parameters, warnings=warnings
    )


def _empty(demand_df: pd.DataFrame, data: ModelData, status: str) -> pd.DataFrame:
    rows = []
    for row in demand_df.itertuples(index=False):
        d = float(row.line_boarders)
        rows.append(
            {
                "time_period": row.time_period,
                "line": ", ".join(data.group_to_lines.get(row.service_group, [row.service_group])),
                "service_group": row.service_group,
                "demand": d / data.demand_multiplier if data.demand_multiplier else d,
                "demand_scaled": d,
                "trains_allocated": np.nan,
                "capacity": data.capacity.get(row.service_group),
                "demand_served": np.nan,
                "unmet_demand": np.nan,
                "coverage_ratio": np.nan,
                "utilization": np.nan,
                "solver_status": status,
            }
        )
    return pd.DataFrame(rows)
