"""End-to-end pipeline test on the TEST DATA workbook.

Covers: profile -> preprocess -> 3 models -> validation -> scenario sweep ->
result export, with all outputs redirected to a temporary directory so that
TEST DATA can never be mistaken for project results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from src.ingestion.loader import detect_schema, profile_workbook
from src.preprocessing.pipeline import build_profile, preprocess
from src.utils import config as cfg
from src.validation.validator import validate_model1, validate_model2, validate_model3
from src.utils.io import save_cleaned, save_profile


@pytest.fixture()
def outputs(tmp_path, monkeypatch):
    """Redirect every output location into tmp."""
    out = tmp_path / "outputs"
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", out)
    monkeypatch.setattr(cfg, "FIGURES_DIR", out / "figures")
    monkeypatch.setattr(cfg, "RUNS_DIR", out / "runs")
    monkeypatch.setattr(cfg, "REPORTS_DIR", out / "reports")
    for d in (out, out / "figures", out / "runs", out / "reports"):
        d.mkdir(parents=True, exist_ok=True)
    return out


def test_full_pipeline(test_workbook, config_override, outputs, monkeypatch, tmp_path):
    mapping = cfg.load_schema_mapping()

    # --- profile ------------------------------------------------------
    info = profile_workbook(test_workbook, mapping)
    assert info["sheet_count"] == 3
    schema, frames = detect_schema(test_workbook, mapping)

    # --- preprocess ---------------------------------------------------
    cleaned, report = preprocess(frames, schema, mapping)
    profile = build_profile(info, schema, report, cleaned)
    save_profile(profile, tmp_path / "dataset_profile.json")
    cleaned_path = save_cleaned(cleaned, tmp_path / "cleaned_data.parquet")
    assert cleaned_path.exists()
    assert not profile["quality"]["blocking_issues"] or report.null_demand_rows == 0

    # --- load back like the scripts do --------------------------------
    links, line_df, station = __import__("src.analysis.eda", fromlist=["x"]).split_tables(
        pd.read_parquet(cleaned_path)
    )
    assert not links.empty and not line_df.empty

    # --- Model 1 ------------------------------------------------------
    from _common import finish
    from src.models.common import prepare_model_data
    from src.models.model1 import solve_model1
    from src.models.model2 import solve_model2
    from src.models.model3 import solve_model3

    class Args:
        mode = None
        lines = None
        out_dir = str(outputs)

    data = prepare_model_data(links, line_df, demand_multiplier=1.0,
                              target_utilization=0.8, fleet_multiplier=1.0,
                              require_capacity=True)
    r1 = solve_model1(data)
    v1 = validate_model1(r1, data)
    assert v1.passed, v1.to_text()
    run1 = finish(Args(), model_name="model1", results=r1.results, report=v1,
                  data=data,
                  params={"demand_multiplier": 1.0, "target_utilization": 0.8},
                  solver=r1.solver)
    assert (outputs / "model1_results.csv").exists()
    assert (outputs / "validation_report.txt").exists()
    assert (run1 / "results.csv").exists()
    assert (run1 / "validation_report.json").exists()
    assert (run1 / "run_metadata.json").exists()
    assert (run1 / "config_snapshot" / "capacity_by_line.csv").exists()
    meta = json.loads((run1 / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["model_name"] == "model1"
    assert meta["solver_status"] == "optimal"

    # --- Model 2 ------------------------------------------------------
    r2 = solve_model2(data)
    v2 = validate_model2(r2, data)
    assert v2.passed, v2.to_text()
    finish(Args(), model_name="model2", results=r2.results, report=v2, data=data,
           params={}, solver=r2.solver)
    assert (outputs / "model2_results.csv").exists()

    # --- Model 3 ------------------------------------------------------
    r3 = solve_model3(data)
    v3 = validate_model3(r3, data)
    assert v3.passed, v3.to_text()
    finish(Args(), model_name="model3", results=r3.results, report=v3, data=data,
           params={"goal_totals": r3.goal_totals}, solver=r3.solver)
    assert (outputs / "model3_results.csv").exists()

    # --- scenarios ----------------------------------------------------
    from src.scenarios.runner import run_scenarios

    summary, runs = run_scenarios(links, line_df, model="model1")
    assert not summary.empty
    summary.to_csv(outputs / "scenario_results.csv", index=False)
    assert set(["baseline", "demand_+10%", "demand_+20%"]) <= set(runs)

    # every saved result must carry a solver status column
    for name in ("model1_results.csv", "model2_results.csv", "model3_results.csv"):
        df = pd.read_csv(outputs / name)
        assert "solver_status" in df.columns, name
        assert (df["solver_status"] == "optimal").all(), name
