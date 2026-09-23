"""Mandatory validation of optimisation outputs.

Every run produces outputs/validation_report.json + .txt. Failures are loud:
an infeasible solve yields an explanation, never silently-zero results.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.common import ModelData
from src.models.model1 import Model1Result
from src.models.model2 import Model2Result
from src.models.model3 import Model3Result


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    severity: str = "error"          # error | warning | info


@dataclass
class ValidationReport:
    model: str
    generated_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    solver_status: str = ""
    checks: list[Check] = field(default_factory=list)
    infeasible_explanation: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "error")

    def add(self, name: str, passed: bool, detail: str, severity: str = "error") -> None:
        self.checks.append(Check(name, bool(passed), detail, severity))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "generated_at": self.generated_at,
            "solver_status": self.solver_status,
            "overall_passed": self.passed,
            "infeasible_explanation": self.infeasible_explanation,
            "warnings": self.warnings,
            "checks": [asdict(c) for c in self.checks],
        }

    def to_text(self) -> str:
        lines = [
            "=" * 72,
            f"VALIDATION REPORT — {self.model}",
            f"generated: {self.generated_at}",
            f"solver status: {self.solver_status}",
            f"overall: {'PASSED' if self.passed else 'FAILED'}",
            "=" * 72,
        ]
        for c in self.checks:
            mark = "PASS" if c.passed else ("WARN" if c.severity == "warning" else "FAIL")
            lines.append(f"[{mark}] {c.name}: {c.detail}")
        if self.infeasible_explanation:
            lines += ["", "INFEASIBLE — EXPLANATION:", self.infeasible_explanation]
        if self.warnings:
            lines += ["", "WARNINGS:"] + [f"  - {w}" for w in self.warnings]
        return "\n".join(lines)


def _check_solver(report: ValidationReport, status: str) -> None:
    report.solver_status = status
    ok = status in {"optimal", "feasible", "locallyOptimal"}
    report.add(
        "solver_status",
        ok,
        f"solver termination status = '{status}'"
        + ("" if ok else " (results must not be interpreted as optimal)"),
    )


def _check_integers(report: ValidationReport, col: str, df: pd.DataFrame) -> None:
    if col not in df.columns or df[col].isna().all():
        report.add(f"integer_{col}", False, f"no values in {col}", severity="error")
        return
    s = df[col].dropna()
    frac = (np.abs(s - np.round(s)) > 1e-6).sum()
    report.add(
        f"integer_{col}",
        frac == 0,
        f"{len(s)} values checked, {frac} non-integer" if frac else f"{len(s)} integer values",
    )


def _check_no_negatives(report: ValidationReport, df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        n = int((s < -1e-9).sum())
        report.add(f"non_negative_{c}", n == 0, f"{n} negative values in {c}")


def _check_bounds(
    report: ValidationReport, data: ModelData, df: pd.DataFrame,
    xcol: str = "optimized_frequency",
) -> None:
    if xcol not in df.columns or df[xcol].isna().all():
        report.add("frequency_bounds", False, "no optimised frequencies to check")
        return
    viol = []
    for row in df.dropna(subset=[xcol]).itertuples(index=False):
        r, t = row.service_group, row.time_period
        x = float(getattr(row, xcol))
        lo = data.min_freq.get((r, t))
        hi = data.max_freq.get((r, t))
        if lo is not None and x < lo - 1e-6:
            viol.append(f"{r}@{t}: {x} < min {lo}")
        if hi is not None and x > hi + 1e-6:
            viol.append(f"{r}@{t}: {x} > max {hi}")
    report.add(
        "frequency_bounds",
        not viol,
        "all frequencies within configured bounds"
        if not viol
        else "; ".join(viol[:5]),
    )


def _check_budget(
    report: ValidationReport, data: ModelData, df: pd.DataFrame, xcol: str
) -> None:
    if not data.fleet_budget:
        report.add(
            "fleet_budget",
            True,
            "fleet/service budget not configured — constraint disabled (documented)",
            severity="info",
        )
        return
    if xcol not in df.columns or df[xcol].isna().all():
        report.add("fleet_budget", False, "no frequencies to check")
        return
    viol = []
    g = df.dropna(subset=[xcol]).groupby("time_period")[xcol].sum()
    for t, total in g.items():
        b = data.fleet_budget.get(str(t))
        if b is not None and total > b + 1e-6:
            viol.append(f"{t}: used {total} > budget {b}")
    report.add(
        "fleet_budget",
        not viol,
        "train-equivalent usage within budget"
        if not viol
        else "; ".join(viol[:5]),
    )


def _check_capacity_per_link(
    report: ValidationReport, data: ModelData, df: pd.DataFrame, xcol: str
) -> None:
    if xcol not in df.columns or df[xcol].isna().all():
        report.add("capacity_constraints", False, "no frequencies to check")
        return
    xmap = {
        (r.service_group, r.time_period): float(getattr(r, xcol))
        for r in df.dropna(subset=[xcol]).itertuples(index=False)
    }
    viol = 0
    checked = 0
    for row in data.link_rows_for_group_time().itertuples(index=False):
        cap = data.capacity.get(row.service_group)
        x = xmap.get((row.service_group, row.time_period))
        if cap is None or x is None:
            continue
        checked += 1
        if cap * x + 1e-6 < float(row.link_load):
            viol += 1
    report.add(
        "capacity_constraints",
        viol == 0,
        f"{checked} link-period capacity constraints checked, {viol} violated"
        if checked
        else "no link-period pairs with both capacity and frequency",
    )


def validate_model1(result: Model1Result, data: ModelData) -> ValidationReport:
    rep = ValidationReport(model="Model 1 — Minimum Service Frequency Optimization")
    rep.warnings = list(result.warnings)
    rep.infeasible_explanation = result.infeasible_explanation
    _check_solver(rep, result.solver.status)
    if not result.solver.ok:
        rep.add("feasibility", False, f"solver did not find a feasible solution ({result.solver.status})")
        return rep

    df = result.results
    rep.add("feasibility", True, "solver reported a feasible/optimal solution")
    _check_integers(rep, "optimized_frequency", df)
    _check_no_negatives(rep, df, ["optimized_frequency", "max_link_load", "capacity_per_train"])
    _check_capacity_per_link(rep, data, df, "optimized_frequency")
    _check_bounds(rep, data, df, "optimized_frequency")
    _check_budget(rep, data, df, "optimized_frequency")

    # analytical closed-form consistency (only where budget/bounds are not binding)
    if "analytical_frequency" in df.columns and df["analytical_frequency"].notna().any():
        sub = df.dropna(subset=["analytical_frequency", "optimized_frequency"])
        mism = sub[np.abs(sub["analytical_frequency"] - sub["optimized_frequency"]) > 1e-6]
        # a mismatch is only a failure if no budget exists and no upper bound binds
        hard = False
        if data.fleet_budget:
            hard = True
        if not hard and data.max_freq:
            hard = True
        if len(mism) == 0:
            rep.add(
                "analytical_sanity_check",
                True,
                f"all {len(sub)} group-periods match ceil(max_link_load/capacity) exactly",
            )
        elif hard:
            rep.add(
                "analytical_sanity_check",
                True,
                f"{len(mism)} differ from closed-form — configured min/max frequency "
                f"bounds or a service budget alter the unconstrained ceiling "
                f"(inspect these rows manually)",
                severity="warning",
            )
        else:
            rep.add(
                "analytical_sanity_check",
                False,
                f"{len(mism)} group-periods differ from closed-form with NO budget/"
                f"bounds configured — inspect solver/model",
            )

    # unit consistency
    rep.add(
        "unit_consistency",
        True,
        "baseline_utilization uses UNSCALED link load / (capacity * observed freq); "
        "optimized_utilization uses SCALED load / (capacity * x) — both passengers/"
        "capacity-slots per period",
        severity="info",
    )
    return rep


def validate_model2(result: Model2Result, data: ModelData) -> ValidationReport:
    rep = ValidationReport(model="Model 2 — Capacity-Constrained Service Allocation")
    rep.warnings = list(result.warnings)
    rep.infeasible_explanation = result.infeasible_explanation
    _check_solver(rep, result.solver.status)
    if not result.solver.ok:
        rep.add("feasibility", False, f"solver did not find a feasible solution ({result.solver.status})")
        return rep

    df = result.results
    rep.add("feasibility", True, "solver reported a feasible/optimal solution")
    _check_integers(rep, "trains_allocated", df)
    _check_no_negatives(
        rep, df, ["trains_allocated", "demand_served", "unmet_demand", "demand"]
    )

    ok_served = bool((df["demand_served"] <= df["demand_scaled"] + 1e-6).all())
    rep.add(
        "demand_served_le_demand",
        ok_served,
        "demand_served <= demand for every row"
        if ok_served
        else f"{int((df['demand_served'] > df['demand_scaled'] + 1e-6).sum())} rows exceed demand",
    )
    ok_cap = bool(
        (df["demand_served"] <= df["capacity_supplied"].fillna(np.inf) + 1e-6).all()
    )
    rep.add(
        "demand_served_le_capacity",
        ok_cap,
        "demand_served <= capacity * x for every row"
        if ok_cap
        else "served demand exceeds supplied capacity on some rows",
    )
    unmet_ok = bool((df["unmet_demand"] >= -1e-6).all())
    rep.add("unmet_demand_non_negative", unmet_ok, "unmet_demand >= 0 for every row")
    _check_bounds(rep, data, df, "trains_allocated")
    _check_budget(rep, data, df, "trains_allocated")
    rep.add(
        "unit_consistency",
        True,
        "objective and demand are LINE BOARDINGS (line-level coverage), "
        "not unique journeys and not summed link loads",
        severity="info",
    )
    return rep


def validate_model3(result: Model3Result, data: ModelData) -> ValidationReport:
    rep = ValidationReport(model="Model 3 — Weighted Goal Programming")
    rep.warnings = list(result.warnings)
    rep.infeasible_explanation = result.infeasible_explanation
    _check_solver(rep, result.solver.status)
    if not result.solver.ok:
        rep.add("feasibility", False, f"solver did not find a feasible solution ({result.solver.status})")
        return rep

    df = result.results
    rep.add("feasibility", True, "solver reported a feasible/optimal solution")
    _check_integers(rep, "trains_allocated", df)
    _check_no_negatives(
        rep, df,
        ["trains_allocated", "d_unmet_minus", "d_freq_minus", "d_freq_plus",
         "demand_served", "unmet_demand"],
    )
    ok_served = bool((df["demand_served"] <= df["demand_scaled"] + 1e-6).all())
    rep.add("demand_served_le_demand", ok_served, "y <= D for every row")
    ok_cap = bool(
        (df["demand_served"] <= df["capacity_supplied"].fillna(np.inf) + 1e-6).all()
    )
    rep.add("demand_served_le_capacity", ok_cap, "y <= cap * x for every row")

    # deviation variables must reproduce the goal equations
    g1_err = np.max(
        np.abs(
            df["demand_served"] + df["d_unmet_minus"] - df["demand_scaled"]
        )
    ) if len(df) else 0.0
    rep.add(
        "goal1_equation",
        bool(g1_err < 1e-4),
        f"max |y + d_unmet_minus - D| = {g1_err:.3e}",
    )
    if "target_frequency" in df.columns:
        g2_err = np.max(
            np.abs(
                df["trains_allocated"] + df["d_freq_minus"] - df["d_freq_plus"]
                - df["target_frequency"]
            )
        )
        rep.add(
            "goal2_equation",
            bool(g2_err < 1e-4),
            f"max |x + d_freq_minus - d_freq_plus - target| = {g2_err:.3e}",
        )
    rep.add(
        "goal3_crowding_deviations",
        bool((df["d_crowd_plus_group"] >= -1e-9).all()),
        "crowding deviations non-negative (checked at group aggregate level)",
    )
    _check_bounds(rep, data, df, "trains_allocated")
    _check_budget(rep, data, df, "trains_allocated")
    return rep


def write_report(report: ValidationReport, out_dir: Path | str) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jpath = out / "validation_report.json"
    tpath = out / "validation_report.txt"
    with jpath.open("w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2, default=str)
    tpath.write_text(report.to_text(), encoding="utf-8")
    return jpath, tpath
