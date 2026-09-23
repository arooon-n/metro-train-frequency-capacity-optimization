"""Scenario multiplier logic + config loading + export tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.scenarios.runner import build_scenarios, run_scenarios
from src.utils import config as cfg
from src.utils.io import export_csv, save_cleaned
from src.utils.manifest import new_run_dir, snapshot_config, write_json, write_run_metadata


# ---------------------------------------------------------------- scenarios
def test_build_scenarios_multipliers_from_config():
    params = {
        "baseline_demand_multiplier": 1.0,
        "increased_demand_multiplier": 1.10,
        "high_demand_multiplier": 1.20,
        "fleet_multiplier": 1.0,
        "reduced_fleet_multiplier": 0.85,
        "target_utilization": 0.80,
        "target_utilization_scenarios": [0.7, 0.9],
    }
    scen = {s.name: s for s in build_scenarios(params, model="model1")}
    assert scen["baseline"].demand_multiplier == 1.0
    assert scen["demand_+10%"].demand_multiplier == pytest.approx(1.10)
    assert scen["demand_+20%"].demand_multiplier == pytest.approx(1.20)
    assert scen["reduced_fleet"].fleet_multiplier == pytest.approx(0.85)
    # utilisation sweeps always run the goal-programming model
    assert scen["target_util_0.70"].model == "model3"
    assert scen["target_util_0.90"].target_utilization == pytest.approx(0.9)


def test_run_scenarios_smoke(model_inputs):
    links, line_df = model_inputs
    summary, runs = run_scenarios(links, line_df, model="model1")
    assert not summary.empty
    assert "baseline" in runs
    assert runs["baseline"].solver_status in {"optimal", "feasible"}
    row = summary[summary["scenario"] == "baseline"].iloc[0]
    assert row["total_train_equivalents"] > 0
    # +10% demand cannot need fewer trains than baseline (capacity fixed)
    b = summary[summary["scenario"] == "baseline"].iloc[0]["total_train_equivalents"]
    d10 = summary[summary["scenario"] == "demand_+10%"].iloc[0]["total_train_equivalents"]
    d20 = summary[summary["scenario"] == "demand_+20%"].iloc[0]["total_train_equivalents"]
    assert d10 >= b - 1e-6
    assert d20 >= d10 - 1e-6


# ---------------------------------------------------------------- config
def test_load_scenario_parameters_defaults():
    p = cfg.load_scenario_parameters()
    assert p["baseline_demand_multiplier"] == 1.0
    assert p["increased_demand_multiplier"] == pytest.approx(1.10)
    assert p["high_demand_multiplier"] == pytest.approx(1.20)


def test_goal_weights_require_all_keys(monkeypatch, tmp_path):
    bad = tmp_path / "goal_weights.yaml"
    bad.write_text("weights:\n  w1_unmet_demand: 1.0\n", encoding="utf-8")
    monkeypatch.setattr(cfg, "GOAL_WEIGHTS_PATH", bad)
    with pytest.raises(cfg.ConfigError):
        cfg.load_goal_weights()


def test_capacity_rejects_nonpositive(monkeypatch, tmp_path):
    bad = tmp_path / "capacity_by_line.csv"
    bad.write_text(
        "mode,line,service_group,capacity_per_train,capacity_definition,source,"
        "source_url,notes\nUnderground,X,X,-5,cap,s,u,n\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "CAPACITY_PATH", bad)
    with pytest.raises(cfg.ConfigError):
        cfg.load_capacity_config()


def test_capacity_ignores_blank_rows(monkeypatch, tmp_path):
    f = tmp_path / "capacity_by_line.csv"
    f.write_text(
        "mode,line,service_group,capacity_per_train,capacity_definition,source,"
        "source_url,notes\nUnderground,X,X,,cap,s,u,missing value\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "CAPACITY_PATH", f)
    df = cfg.load_capacity_config()
    assert df.empty          # blank capacity rows dropped, never guessed


def test_fleet_budget_empty_is_ok(monkeypatch, tmp_path):
    f = tmp_path / "fleet_budget.csv"
    f.write_text("time_period,fleet_budget,source,notes\n", encoding="utf-8")
    monkeypatch.setattr(cfg, "FLEET_BUDGET_PATH", f)
    assert cfg.load_fleet_budget().empty


def test_fleet_budget_rejects_nonnumeric(monkeypatch, tmp_path):
    f = tmp_path / "fleet_budget.csv"
    f.write_text("time_period,fleet_budget,source,notes\n07:00-07:15,abc,s,n\n",
                 encoding="utf-8")
    monkeypatch.setattr(cfg, "FLEET_BUDGET_PATH", f)
    with pytest.raises(cfg.ConfigError):
        cfg.load_fleet_budget()


def test_service_constraints_min_gt_max(monkeypatch, tmp_path):
    f = tmp_path / "service_constraints.csv"
    f.write_text(
        "mode,line,service_group,min_frequency_per_15min,max_frequency_per_15min,"
        "target_frequency_per_15min,notes\nU,L,L,10,2,5,n\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "CONSTRAINTS_PATH", f)
    with pytest.raises(cfg.ConfigError):
        cfg.load_service_constraints()


# ---------------------------------------------------------------- export
def test_export_csv(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    p = export_csv(df, tmp_path / "out.csv")
    assert p.exists()
    back = pd.read_csv(p)
    assert list(back.columns) == ["a", "b"]


def test_save_and_load_cleaned_roundtrip(tmp_path, cleaned_df):
    p = save_cleaned(cleaned_df, tmp_path / "cleaned.parquet")
    assert p.exists()
    back = pd.read_parquet(p)
    assert len(back) == len(cleaned_df)
    assert set(back["record_type"]) >= {"link", "line_boarders"}


def test_run_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(cfg, "CONFIG_DIR", cfg.CONFIG_DIR)  # real configs snapshotted
    run_dir = new_run_dir(prefix="test")
    snapshot_config(run_dir)
    assert (run_dir / "config_snapshot" / "scenario_parameters.yaml").exists()
    write_json(run_dir / "model_parameters.json", {"x": 1})
    meta = write_run_metadata(
        run_dir,
        dataset_path=None,
        model_name="model1",
        scenario="baseline",
        parameters={"demand_multiplier": 1.0},
        solver_name="appsi_highs",
        solver_status="optimal",
        selected_lines=["TestLineA"],
    )
    data = json.loads(Path(meta).read_text(encoding="utf-8"))
    assert data["model_name"] == "model1"
    assert data["solver_status"] == "optimal"
    assert data["selected_lines"] == ["TestLineA"]
