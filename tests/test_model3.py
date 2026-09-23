"""Model 3 tests: deviation variables satisfy the goal equations; weights are
loaded from configuration (never hardcoded)."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.model3 import solve_model3
from src.utils import config as cfg
from src.validation.validator import validate_model3


def test_model3_solves_and_validates(model_data):
    res = solve_model3(model_data)
    assert res.solver.ok, res.solver.status
    report = validate_model3(res, model_data)
    assert report.passed, report.to_text()


def test_model3_goal1_equation(model_data):
    """y + d_unmet_minus == D  (scaled demand) for every group-period."""
    res = solve_model3(model_data)
    df = res.results
    err = np.max(np.abs(
        df["demand_served"] + df["d_unmet_minus"] - df["demand_scaled"]
    ))
    assert err < 1e-4


def test_model3_goal2_equation(model_data):
    """x + d_freq_minus - d_freq_plus == target_frequency."""
    res = solve_model3(model_data)
    df = res.results
    err = np.max(np.abs(
        df["trains_allocated"] + df["d_freq_minus"] - df["d_freq_plus"]
        - df["target_frequency"]
    ))
    assert err < 1e-4


def test_model3_deviation_nonnegativity(model_data):
    res = solve_model3(model_data)
    df = res.results
    assert (df["d_unmet_minus"] >= -1e-9).all()
    assert (df["d_freq_minus"] >= -1e-9).all()
    assert (df["d_freq_plus"] >= -1e-9).all()
    assert (df["d_crowd_plus_group"] >= -1e-9).all()


def test_model3_weights_come_from_config(model_data):
    """Changing the yaml weights changes the recorded effective weights."""
    base_cfg = cfg.load_goal_weights()
    res_default = solve_model3(model_data, weights_cfg=base_cfg)
    assert res_default.parameters["weights"] == base_cfg["weights"]

    custom = {
        "weights": {
            "w1_unmet_demand": 50.0,
            "w2_frequency_target": 0.0,
            "w3_crowding": 0.0,
            "w4_service_usage": 0.0,
        },
        "normalize_terms": True,
    }
    res_custom = solve_model3(model_data, weights_cfg=custom)
    assert res_custom.solver.ok
    # heavy w1 weight => unmet demand must not be worse than the default run
    assert (res_custom.results["d_unmet_minus"].sum()
            <= res_default.results["d_unmet_minus"].sum() + 1e-4)


def test_model3_capacity_feeds_crowding_goal(model_data):
    """Goal 3 must be driven by capacity*x — crowding drops as x rises."""
    res = solve_model3(model_data)
    df = res.results
    # any group-period with positive crowding deviation must have had
    # demand above target_utilisation * capacity * x
    for row in df.itertuples(index=False):
        cap = model_data.capacity[row.service_group]
        u = model_data.target_utilization
        if row.d_crowd_plus_group > 1e-6:
            assert row.demand_scaled > u * cap * row.trains_allocated - 1e-4
