"""Pyomo solver utilities.

Default solver: HiGHS through Pyomo (highspy) — free, no licence.

Pyomo 6.9+ wraps ``SolverFactory('appsi_highs')`` in a legacy adapter that may
return either an APPSI ``Results`` object or a classic ``SolverResults``
object, so this module normalises both shapes into one status vocabulary:

    optimal | feasible | infeasible | unbounded | maxTimeLimit | error | <other>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pyomo.environ as pyo

DEFAULT_SOLVER = "appsi_highs"


class SolverUnavailableError(RuntimeError):
    pass


@dataclass
class SolverResult:
    status: str                       # e.g. "optimal", "infeasible", "error"
    solver_name: str
    termination: str
    objective: float | None = None
    message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in {"optimal", "feasible", "locallyOptimal"}


def _normalise_termination(term: str) -> str:
    t = str(term).strip()
    low = t.lower()
    if "optimal" in low and "infeasible" not in low:
        return "optimal"
    if "feasible" in low and "infeasible" not in low:
        return "feasible"
    if "infeasible" in low and "unbounded" not in low:
        return "infeasible"
    if "unbounded" in low and "infeasible" not in low:
        return "unbounded"
    if "infeasible" in low and "unbounded" in low:
        return "infeasibleOrUnbounded"
    if "maxtime" in low or "max time" in low or "timelimit" in low:
        return "maxTimeLimit"
    if "error" in low:
        return "error"
    if "ok" == low:
        return "optimal"
    return t


def get_solver(name: str | None = None, options: dict | None = None):
    name = name or DEFAULT_SOLVER
    candidates = [name]
    for alt in ("appsi_highs", "cbc", "glpk"):
        if alt not in candidates:
            candidates.append(alt)
    last_err = ""
    for cand in candidates:
        try:
            opt = pyo.SolverFactory(cand)
            if opt is None:
                last_err = f"{cand}: Factory returned None"
                continue
            if hasattr(opt, "available"):
                try:
                    if not opt.available(exception_flag=False):
                        last_err = f"{cand} reported not available"
                        continue
                except TypeError:
                    pass
            if options:
                try:
                    opt.options.update(options)
                except Exception as exc:
                    last_err = f"{cand}: could not apply options ({exc})"
                    # keep going — options are best-effort
            return opt, cand
        except Exception as exc:
            last_err = f"{cand}: {exc}"
            continue
    raise SolverUnavailableError(
        "No supported MILP solver found (tried: "
        f"{candidates}). Last error: {last_err}. "
        "Install with: pip install highspy   (recommended, free HiGHS)"
    )


def solve_model(
    model: pyo.ConcreteModel,
    name: str | None = None,
    options: dict | None = None,
) -> SolverResult:
    opt, used = get_solver(name, options)
    res = None
    load_issue: Exception | None = None
    try:
        res = opt.solve(model)
    except Exception as exc:
        load_issue = exc
        msg = str(exc).lower()
        if "feasible solution was not found" in msg or "solution can be loaded" in msg:
            # Solver finished WITHOUT a feasible solution (typically infeasible).
            # Re-solve without loading so we can read the true termination status.
            try:
                res = opt.solve(model, load_solutions=False)
            except Exception:
                res = None
            if res is None:
                return SolverResult(
                    status="infeasible",
                    solver_name=used,
                    termination="no feasible solution",
                    message=str(load_issue),
                )
        else:
            return SolverResult(
                status="error", solver_name=used, termination="error", message=str(exc)
            )

    # --- normalise the result object (APPSI Results vs legacy SolverResults) ---
    term = None
    if hasattr(res, "termination_condition") and not hasattr(res, "solver"):
        term = str(res.termination_condition)                 # APPSI Results
    elif hasattr(res, "solver") and hasattr(res.solver, "termination_condition"):
        term = str(res.solver.termination_condition)          # legacy SolverResults
    elif hasattr(res, "termination_condition"):
        term = str(res.termination_condition)
    else:
        term = str(res)

    status = _normalise_termination(term)
    if load_issue is not None and status in {"optimal", "feasible"}:
        # solution could not be loaded despite a nominal optimal status
        status = "error"

    obj = None
    try:
        obj = float(pyo.value(model.objective))
    except Exception:
        obj = None
    if status in {"infeasible", "unbounded", "error", "infeasibleOrUnbounded"}:
        obj = None

    return SolverResult(
        status=status, solver_name=used, termination=term, objective=obj,
        message=str(load_issue) if load_issue is not None else "",
    )
