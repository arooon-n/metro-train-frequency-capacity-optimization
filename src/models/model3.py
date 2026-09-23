"""MODEL 3 — Weighted Goal Programming for multi-objective service planning.

Four explicit linear goals with deviation variables (weights come from
config/goal_weights.yaml — never hardcoded):

  G1  unmet line demand      y[r,t] + d_unmet_minus[r,t] = D[r,t]
                              y[r,t] <= cap[r] * x[r,t]
  G2  target frequency       x[r,t] + d_freq_minus[r,t] - d_freq_plus[r,t]
                              = target[r,t]
  G3  crowding per link l    D[l,t] - u*cap[r]*x[r,t] <= d_crowd_plus[l,t]
                              (u = target utilisation)
  G4  service usage          x[r,t] itself (train-equivalents)

  min  w1*SUM(d_unmet_minus)
     + w2*SUM(d_freq_minus + d_freq_plus)
     + w3*SUM(d_crowd_plus)
     + w4*SUM(x)

With normalize_terms: true each sum is divided by a documented reference scale
(total demand / number of entities) so the weights are comparable across units.
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
class Model3Result:
    results: pd.DataFrame
    solver: SolverResult
    parameters: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    infeasible_explanation: str | None = None
    goal_totals: dict[str, float] = field(default_factory=dict)


def solve_model3(
    data: ModelData,
    *,
    weights_cfg: dict[str, Any] | None = None,
    target_utilization: float | None = None,
    solver_name: str | None = None,
    solver_options: dict | None = None,
) -> Model3Result:
    if not data.capacity:
        raise ModelDataError(
            "Model 3 requires capacity_per_train — fill config/capacity_by_line.csv."
        )
    if weights_cfg is None:
        weights_cfg = cfg.load_goal_weights()
    w = weights_cfg["weights"]
    normalize = bool(weights_cfg.get("normalize_terms", True))

    u = target_utilization
    if u is None:
        u = weights_cfg.get("target_utilization_override")
    if u is None:
        u = data.target_utilization
    u = float(u)
    if not (0 < u <= 2):
        raise ModelDataError(f"target_utilization must be in (0, 2]; got {u}")

    demand_df = data.scaled_group_demand()
    link_df = data.link_rows_for_group_time()
    periods = sorted(set(demand_df["time_period"]) | set(data.periods))
    groups = sorted(set(demand_df["service_group"]) | set(data.service_groups))
    keys = [(r, t) for r in groups for t in periods]
    demand = {
        (r, t): float(v)
        for (r, t), v in demand_df.set_index(["service_group", "time_period"])["line_boarders"].items()
    }

    # one crowding goal per link observation (scaled demand)
    link_keys = list(range(len(link_df)))
    link_sg = dict(zip(link_keys, link_df["service_group"]))
    link_tp = dict(zip(link_keys, link_df["time_period"]))
    link_load = dict(zip(link_keys, link_df["link_load"].astype(float)))

    target = {}
    for k in keys:
        target[k] = data.target_freq.get(k, data.observed_freq.get(k, 0.0))

    warnings = list(data.warnings)

    m = pyo.ConcreteModel("Model3_WeightedGoalProgramming")
    m.R = pyo.Set(initialize=groups)
    m.T = pyo.Set(initialize=periods)
    m.K = pyo.Set(initialize=link_keys)

    def x_bounds(model, r, t):
        lo = data.min_freq.get((r, t))
        hi = data.max_freq.get((r, t))
        return (0 if lo is None else lo, hi)

    m.x = pyo.Var(keys, domain=pyo.NonNegativeIntegers, bounds=x_bounds)
    m.y = pyo.Var(keys, domain=pyo.NonNegativeReals)

    # deviation variables (all >= 0)
    m.d_unmet_minus = pyo.Var(keys, domain=pyo.NonNegativeReals)
    m.d_freq_minus = pyo.Var(keys, domain=pyo.NonNegativeReals)
    m.d_freq_plus = pyo.Var(keys, domain=pyo.NonNegativeReals)
    m.d_crowd_plus = pyo.Var(m.K, domain=pyo.NonNegativeReals)

    # G1
    def g1_rule(model, r, t):
        return m.y[r, t] + m.d_unmet_minus[r, t] == demand.get((r, t), 0.0)

    def y_cap_rule(model, r, t):
        cap = data.capacity.get(r)
        if cap is None:
            return pyo.Constraint.Skip
        return m.y[r, t] <= cap * m.x[r, t]

    # G2
    def g2_rule(model, r, t):
        return m.x[r, t] + m.d_freq_minus[r, t] - m.d_freq_plus[r, t] == target[k_target(r, t)]

    # G3
    def g3_rule(model, k):
        r, t = link_sg[k], link_tp[k]
        cap = data.capacity.get(r)
        if cap is None:
            return pyo.Constraint.Skip
        return link_load[k] - u * cap * m.x[r, t] <= m.d_crowd_plus[k]

    m.g1 = pyo.Constraint(keys, rule=g1_rule)
    m.y_cap = pyo.Constraint(keys, rule=y_cap_rule)
    m.g2 = pyo.Constraint(keys, rule=g2_rule)
    m.g3 = pyo.Constraint(m.K, rule=g3_rule)

    budget_active = bool(data.fleet_budget)
    if budget_active:
        def budget_rule(model, t):
            if t not in data.fleet_budget:
                return pyo.Constraint.Skip
            return pyo.quicksum(m.x[r, t] for r in groups) <= data.fleet_budget[t]

        m.budget = pyo.Constraint(periods, rule=budget_rule)

    # ---- scales for optional normalisation -----------------------------
    total_demand = float(sum(demand.values())) or 1.0
    n_rt = max(len(keys), 1)
    total_link_load = float(sum(link_load.values())) or 1.0
    s1 = total_demand if normalize else 1.0
    s2 = float(n_rt) if normalize else 1.0
    s3 = total_link_load if normalize else 1.0
    s4 = float(n_rt) if normalize else 1.0

    w1 = float(w["w1_unmet_demand"]) / s1
    w2 = float(w["w2_frequency_target"]) / s2
    w3 = float(w["w3_crowding"]) / s3
    w4 = float(w["w4_service_usage"]) / s4

    m.objective = pyo.Objective(
        expr=(
            w1 * pyo.quicksum(m.d_unmet_minus[r, t] for (r, t) in keys)
            + w2 * pyo.quicksum(
                m.d_freq_minus[r, t] + m.d_freq_plus[r, t] for (r, t) in keys
            )
            + w3 * pyo.quicksum(m.d_crowd_plus[k] for k in link_keys)
            + w4 * pyo.quicksum(m.x[r, t] for (r, t) in keys)
        ),
        sense=pyo.minimize,
    )

    if solver_options is None:
        sp = cfg.load_scenario_parameters()
        solver_name = solver_name or sp.get("solver")
        solver_options = sp.get("solver_options") or None

    sres = solve_model(m, solver_name, solver_options)

    parameters = {
        "model": "Model 3 — Weighted Goal Programming",
        "weights": w,
        "normalize_terms": normalize,
        "effective_weights": {"w1": w1, "w2": w2, "w3": w3, "w4": w4},
        "target_utilization": u,
        "demand_multiplier": data.demand_multiplier,
        "service_budget_active": budget_active,
        "fleet_budget": data.fleet_budget,
        "capacity_by_service_group": {k: data.capacity[k] for k in sorted(data.capacity)},
        "n_service_groups": len(groups),
        "n_time_periods": len(periods),
        "n_crowding_goals": len(link_keys),
    }

    if not sres.ok:
        expl = (
            "Model 3 is INFEASIBLE. Goal deviation variables are non-negative, so "
            "infeasibility here almost always comes from hard constraints: "
            "min/max frequency bounds conflicting, or fleet_budget below the sum of "
            "minimum frequencies. Inspect config/service_constraints.csv and "
            f"config/fleet_budget.csv. Solver said: {sres.status} {sres.message}"
            if sres.status == "infeasible"
            else f"Solver returned status '{sres.status}': {sres.message}"
        )
        return Model3Result(
            results=_empty(demand_df, data, sres.status),
            solver=sres,
            parameters=parameters,
            warnings=warnings,
            infeasible_explanation=expl,
        )

    # ---- extract --------------------------------------------------------
    goal_totals = {
        "unmet_line_demand": float(
            sum(pyo.value(m.d_unmet_minus[r, t]) for (r, t) in keys)
        ),
        "frequency_deviation": float(
            sum(
                pyo.value(m.d_freq_minus[r, t]) + pyo.value(m.d_freq_plus[r, t])
                for (r, t) in keys
            )
        ),
        "crowding_deviation": float(
            sum(pyo.value(m.d_crowd_plus[k]) for k in link_keys)
        ),
        "train_equivalent_usage": float(
            sum(pyo.value(m.x[r, t]) for (r, t) in keys)
        ),
        "objective_value": float(sres.objective) if sres.objective is not None else None,
    }

    crowd_by_group = {}
    for k in link_keys:
        r = link_sg[k]
        v = float(pyo.value(m.d_crowd_plus[k]))
        if v > 1e-6:
            crowd_by_group[r] = crowd_by_group.get(r, 0) + 1

    rows = []
    for (r, t) in keys:
        cap = data.capacity.get(r)
        d_raw = demand.get((r, t), 0.0)
        d_base = d_raw / data.demand_multiplier if data.demand_multiplier else d_raw
        x = int(round(pyo.value(m.x[r, t])))
        y = float(pyo.value(m.y[r, t]))
        rows.append(
            {
                "time_period": t,
                "line": ", ".join(data.group_to_lines.get(r, [r])),
                "service_group": r,
                "demand": d_base,
                "demand_scaled": d_raw,
                "trains_allocated": x,
                "observed_frequency": data.observed_freq.get((r, t)),
                "target_frequency": target[(r, t)],
                "capacity": cap,
                "capacity_supplied": (cap * x) if cap else np.nan,
                "demand_served": y,
                "unmet_demand": d_raw - y,
                "coverage_ratio": (y / d_raw) if d_raw > 0 else 1.0,
                "d_unmet_minus": float(pyo.value(m.d_unmet_minus[r, t])),
                "d_freq_minus": float(pyo.value(m.d_freq_minus[r, t])),
                "d_freq_plus": float(pyo.value(m.d_freq_plus[r, t])),
                "d_crowd_plus_group": crowd_by_group.get(r, 0),
                "utilization_of_demand": (
                    (cap * x) / d_raw if cap and d_raw > 0 else np.nan
                ),
                "solver_status": sres.status,
            }
        )

    return Model3Result(
        results=pd.DataFrame(rows),
        solver=sres,
        parameters=parameters,
        warnings=warnings,
        goal_totals=goal_totals,
    )


def k_target(r: str, t: str) -> tuple[str, str]:
    return (r, t)


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
                "trains_allocated": np.nan,
                "target_frequency": data.target_freq.get(
                    (row.service_group, row.time_period)
                ),
                "capacity": data.capacity.get(row.service_group),
                "demand_served": np.nan,
                "unmet_demand": np.nan,
                "solver_status": status,
            }
        )
    return pd.DataFrame(rows)
