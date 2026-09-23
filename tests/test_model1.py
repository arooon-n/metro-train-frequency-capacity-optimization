"""Model 1 tests: feasibility, analytical sanity check, infeasibility handling."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.models.common import ModelDataError, prepare_model_data
from src.models.model1 import (
    analytical_required_frequency,
    explain_infeasibility,
    solve_model1,
)
from src.validation.validator import validate_model1
from tests.conftest import TEST_PERIODS


def test_model1_feasible_and_valid(model_data):
    res = solve_model1(model_data)
    assert res.solver.ok, f"solver failed: {res.solver.status} {res.solver.message}"
    df = res.results
    assert not df.empty
    assert (df["optimized_frequency"] >= 0).all()
    assert np.allclose(df["optimized_frequency"],
                       np.round(df["optimized_frequency"]))
    report = validate_model1(res, model_data)
    assert report.passed, report.to_text()


def test_model1_analytical_sanity_check(model_data):
    """MILP must equal ceil(max_link_load / capacity) when unconstrained."""
    res = solve_model1(model_data)
    ana = analytical_required_frequency(model_data)
    merged = res.results.merge(
        ana, on=["service_group", "time_period"], how="left", suffixes=("", "_a")
    )
    assert merged["analytical_frequency"].notna().all()
    # budget in the test config is 100/period — ample; upper bound 20 also ample
    mismatch = merged[
        np.abs(merged["optimized_frequency"] - merged["analytical_frequency"]) > 1e-6
    ]
    assert mismatch.empty, mismatch.to_string()

    # spot-check one cell by hand
    cap = model_data.capacity["TestLineA"]
    peak_load = 300  # TEST DATA peak link load on TestLineA
    expected = math.ceil(peak_load / cap)
    row = merged[(merged["service_group"] == "TestLineA")
                 & (merged["time_period"] == "07:15-07:30")].iloc[0]
    assert row["optimized_frequency"] == expected == 3


def test_model1_capacity_constraint_respected(model_data):
    res = solve_model1(model_data)
    xmap = {(r.service_group, r.time_period): r.optimized_frequency
            for r in res.results.itertuples(index=False)}
    for row in model_data.link_rows_for_group_time().itertuples(index=False):
        cap = model_data.capacity[row.service_group]
        x = xmap[(row.service_group, row.time_period)]
        assert cap * x >= row.link_load - 1e-6


def test_model1_budget_binding_changes_result(model_data):
    """Budget is respected when feasible; impossible budgets are explained.

    Note (OR insight): in Model 1 the budget is an UPPER bound on a
    minimise-sum objective whose lower bounds are the capacity-covering
    ceilings — so a budget below the sum of ceilings makes the model
    INFEASIBLE rather than 'less than closed-form'. min_frequency, by
    contrast, can push the optimum ABOVE the closed-form ceiling.
    """
    # feasible budget (unconstrained need per period = 3, 5, 3)
    model_data.fleet_budget = {p: 6 for p in TEST_PERIODS}  # TEST budget
    res = solve_model1(model_data)
    assert res.solver.ok
    usage = res.results.groupby("time_period")["optimized_frequency"].sum()
    for period, used in usage.items():
        assert used <= 6 + 1e-6
    report = validate_model1(res, model_data)
    assert report.passed

    # impossible budget => infeasible + explanation, never silent zeros
    model_data.fleet_budget = {p: 4 for p in TEST_PERIODS}
    res2 = solve_model1(model_data)
    assert not res2.solver.ok
    assert res2.infeasible_explanation
    assert "fleet_budget" in res2.infeasible_explanation


def test_model1_infeasible_explains(model_data):
    """max_frequency below requirement => infeasible with an explanation, not zeros."""
    model_data.max_freq = {
        (g, p): 1 for g in model_data.service_groups for p in model_data.periods
    }
    res = solve_model1(model_data)
    assert not res.solver.ok
    assert res.infeasible_explanation
    assert "max_frequency_per_15min" in res.infeasible_explanation
    report = validate_model1(res, model_data)
    assert not report.passed
    assert report.infeasible_explanation


def test_model1_missing_capacity_raises(model_inputs):
    """Capacities must never be guessed — missing config stops the model."""
    links, line_df = model_inputs
    from src.utils import config as cfg

    # simulate an empty capacity file
    import pathlib
    import tempfile

    empty = pathlib.Path(tempfile.mkdtemp()) / "capacity_by_line.csv"
    empty.write_text(
        "mode,line,service_group,capacity_per_train,capacity_definition,source,"
        "source_url,notes\n",
        encoding="utf-8",
    )
    original = cfg.CAPACITY_PATH
    cfg.CAPACITY_PATH = empty
    try:
        with pytest.raises(ModelDataError) as exc:
            prepare_model_data(links, line_df, require_capacity=True)
        assert "capacity" in str(exc.value).lower()
    finally:
        cfg.CAPACITY_PATH = original


def test_analytical_frequency_clips_to_bounds(model_data):
    model_data.min_freq = {("TestLineA", p): 9 for p in model_data.periods}
    ana = analytical_required_frequency(model_data)
    sub = ana[ana["service_group"] == "TestLineA"]
    assert (sub["analytical_frequency"] >= 9).all()
