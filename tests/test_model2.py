"""Model 2 tests: served-demand constraints and budget compliance."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.common import ModelDataError
from src.models.model2 import solve_model2
from src.validation.validator import validate_model2
from tests.conftest import TEST_PERIODS


def test_model2_demand_served_constraints(model_data):
    res = solve_model2(model_data)
    assert res.solver.ok, res.solver.status
    df = res.results
    # y <= D
    assert (df["demand_served"] <= df["demand_scaled"] + 1e-6).all()
    # y <= capacity * x
    assert (df["demand_served"] <= df["capacity_supplied"] + 1e-6).all()
    # no negatives, x integer
    assert (df["trains_allocated"] >= 0).all()
    assert np.allclose(df["trains_allocated"], np.round(df["trains_allocated"]))
    assert (df["unmet_demand"] >= -1e-6).all()
    report = validate_model2(res, model_data)
    assert report.passed, report.to_text()


def test_model2_respects_budget(model_data):
    res = solve_model2(model_data)
    assert res.solver.ok
    usage = res.results.groupby("time_period")["trains_allocated"].sum()
    for period, used in usage.items():
        assert used <= model_data.fleet_budget[period] + 1e-6


def test_model2_maximises_coverage_under_tight_budget(model_data):
    """Halving the budget cannot increase served demand."""
    free = solve_model2(model_data)
    model_data.fleet_budget = {p: 3 for p in TEST_PERIODS}  # TEST DATA tight budget
    tight = solve_model2(model_data)
    assert free.solver.ok and tight.solver.ok
    assert tight.results["demand_served"].sum() <= free.results["demand_served"].sum() + 1e-4
    usage = tight.results.groupby("time_period")["trains_allocated"].sum()
    assert (usage <= 3 + 1e-6).all()


def test_model2_requires_capacity(model_inputs):
    from src.utils import config as cfg
    import pathlib, tempfile

    empty = pathlib.Path(tempfile.mkdtemp()) / "capacity_by_line.csv"
    empty.write_text(
        "mode,line,service_group,capacity_per_train,capacity_definition,source,"
        "source_url,notes\n",
        encoding="utf-8",
    )
    original = cfg.CAPACITY_PATH
    cfg.CAPACITY_PATH = empty
    links, line_df = model_inputs
    try:
        with pytest.raises(ModelDataError):
            prepare = __import__("src.models.common", fromlist=["prepare_model_data"])
            prepare.prepare_model_data(links, line_df, require_capacity=True)
    finally:
        cfg.CAPACITY_PATH = original


def test_model2_require_budget_flag(model_data):
    """require_budget=True must fail loudly when no budget file exists."""
    from src.utils import config as cfg
    import pathlib, tempfile

    empty = pathlib.Path(tempfile.mkdtemp()) / "fleet_budget.csv"
    empty.write_text("time_period,fleet_budget,source,notes\n", encoding="utf-8")
    original = cfg.FLEET_BUDGET_PATH
    cfg.FLEET_BUDGET_PATH = empty
    try:
        model_data.fleet_budget = {}
        with pytest.raises(ModelDataError) as exc:
            solve_model2(model_data, require_budget=True)
        assert "budget" in str(exc.value).lower()
    finally:
        cfg.FLEET_BUDGET_PATH = original
