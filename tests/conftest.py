"""Pytest fixtures.

*** ALL DATA BUILT HERE IS TEST DATA — NEVER USED FOR PROJECT RESULTS. ***

The fixtures create a tiny artificial NUMBAT-like workbook (2 lines, 3 stations,
3 time periods) plus temporary configuration files. The real project always runs
on the official NUMBAT workbook in data/raw/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import config as cfg  # noqa: E402

TEST_DATA_NOTICE = "TEST DATA — artificial workbook for unit tests only"

# canonical TEST DATA tables -------------------------------------------------
TEST_PERIODS = ["07:00-07:15", "07:15-07:30", "07:30-07:45"]
TEST_LINES = {
    "TestLineA": ["Alpha", "Beta", "Gamma"],     # simple corridor
    "TestLineB": ["Delta", "Epsilon"],           # short corridor
}


def link_rows(day_type: str = "TWT") -> pd.DataFrame:
    rows = []
    loads = {("TestLineA", 0): [120, 80], ("TestLineA", 1): [300, 260],
             ("TestLineA", 2): [140, 90],
             ("TestLineB", 0): [60], ("TestLineB", 1): [150], ("TestLineB", 2): [70]}
    for (line, pi), vals in loads.items():
        stations = TEST_LINES[line]
        for seg, val in enumerate(vals):
            rows.append({
                "Mode": "Underground",
                "Line": line,
                "From": stations[seg],
                "To": stations[seg + 1],
                "Direction": "Northbound",
                "Time": TEST_PERIODS[pi],
                "Day Type": day_type,
                "Link Load": val,
            })
    return pd.DataFrame(rows)


def frequency_rows(day_type: str = "TWT") -> pd.DataFrame:
    df = link_rows(day_type)
    # scheduled frequency: 6 trains / 15 min early, 12 at peak-ish period
    df["Link Frequency"] = df["Time"].map(
        {"07:00-07:15": 6, "07:15-07:30": 12, "07:30-07:45": 8}
    )
    return df.drop(columns=["Link Load"])


def line_boarders_rows(day_type: str = "TWT") -> pd.DataFrame:
    rows = []
    demands = {
        "TestLineA": [400, 1500, 600],
        "TestLineB": [200, 700, 300],
    }
    for line, vals in demands.items():
        for t, v in zip(TEST_PERIODS, vals):
            rows.append({
                "Mode": "Underground", "Line": line, "Time": t,
                "Day Type": day_type, "Line Boarders": v,
            })
    return pd.DataFrame(rows)


def write_workbook(path: Path, *, drop_sheets: tuple[str, ...] = (),
                   links: pd.DataFrame | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        if "link_load" not in drop_sheets:
            (links if links is not None else link_rows()).to_excel(
                xl, sheet_name="Link Load", index=False
            )
        if "link_frequency" not in drop_sheets:
            frequency_rows().to_excel(xl, sheet_name="Link Frequency", index=False)
        if "line_boarders" not in drop_sheets:
            line_boarders_rows().to_excel(xl, sheet_name="Line Boarders", index=False)
    return path


@pytest.fixture(scope="session")
def test_workbook(tmp_path_factory) -> Path:
    """TEST DATA workbook (clean)."""
    p = tmp_path_factory.mktemp("test_data") / "TEST_DATA_numbat_synthetic.xlsx"
    return write_workbook(p)


@pytest.fixture()
def config_override(tmp_path, monkeypatch):
    """Point every operational config file at temporary TEST CONFIG files."""
    cap = tmp_path / "capacity_by_line.csv"
    cap.write_text(
        "mode,line,service_group,capacity_per_train,capacity_definition,source,source_url,notes\n"
        "Underground,TestLineA,TestLineA,100,TEST DATA capacity,test,test,TEST DATA\n"
        "Underground,TestLineB,TestLineB,80,TEST DATA capacity,test,test,TEST DATA\n",
        encoding="utf-8",
    )
    con = tmp_path / "service_constraints.csv"
    con.write_text(
        "mode,line,service_group,min_frequency_per_15min,"
        "max_frequency_per_15min,target_frequency_per_15min,notes\n"
        "Underground,TestLineA,TestLineA,1,20,10,TEST DATA\n"
        "Underground,TestLineB,TestLineB,1,20,8,TEST DATA\n",
        encoding="utf-8",
    )
    fb = tmp_path / "fleet_budget.csv"
    fb.write_text(
        "time_period,fleet_budget,source,notes\n"
        + "".join(f"{p},100,test,TEST DATA\n" for p in TEST_PERIODS),
        encoding="utf-8",
    )
    sg = tmp_path / "service_groups.csv"
    sg.write_text("mode,line,service_group,from_station,to_station,notes\n",
                  encoding="utf-8")

    monkeypatch.setattr(cfg, "CAPACITY_PATH", cap)
    monkeypatch.setattr(cfg, "CONSTRAINTS_PATH", con)
    monkeypatch.setattr(cfg, "FLEET_BUDGET_PATH", fb)
    monkeypatch.setattr(cfg, "SERVICE_GROUPS_PATH", sg)
    return tmp_path


@pytest.fixture()
def cleaned_df(test_workbook) -> pd.DataFrame:
    """Cleaned canonical dataset built from the TEST DATA workbook."""
    from src.ingestion.loader import detect_schema
    from src.preprocessing.pipeline import preprocess

    mapping = cfg.load_schema_mapping()
    schema, frames = detect_schema(test_workbook, mapping)
    cleaned, _report = preprocess(frames, schema, mapping)
    return cleaned


@pytest.fixture()
def model_inputs(cleaned_df, config_override):
    """(links, line_demand) tables ready for prepare_model_data."""
    from src.analysis import eda

    links, line_df, _ = eda.split_tables(cleaned_df)
    return links, line_df


@pytest.fixture()
def model_data(model_inputs):
    from src.models.common import prepare_model_data

    links, line_df = model_inputs
    return prepare_model_data(
        links, line_df,
        demand_multiplier=1.0, target_utilization=0.8,
        fleet_multiplier=1.0, require_capacity=True,
    )
