"""Workbook loading + schema detection tests (TEST DATA workbook)."""

from __future__ import annotations

import pytest

from src.ingestion.loader import (
    SchemaDetectionError,
    detect_schema,
    list_sheets,
    profile_workbook,
    read_sheet,
)
from src.utils import config as cfg
from tests.conftest import write_workbook


def test_list_sheets(test_workbook):
    sheets = list_sheets(test_workbook)
    assert "Link Load" in sheets
    assert "Link Frequency" in sheets
    assert "Line Boarders" in sheets


def test_profile_workbook_reports_shapes(test_workbook):
    info = profile_workbook(test_workbook, cfg.load_schema_mapping())
    assert info["sheet_count"] == 3
    assert all(s["rows"] > 0 for s in info["sheets"])


def test_schema_detection_finds_roles(test_workbook):
    schema, frames = detect_schema(test_workbook, cfg.load_schema_mapping())
    assert schema.roles["link_load"].sheet == "Link Load"
    assert schema.roles["link_frequency"].sheet == "Link Frequency"
    assert schema.roles["line_boarders"].sheet == "Line Boarders"
    # logical -> actual column mapping
    assert schema.roles["link_load"].columns["link_load"] == "Link Load"
    assert schema.roles["link_load"].columns["from_station"] == "From"
    assert schema.roles["line_boarders"].columns["line_boarders"] == "Line Boarders"
    assert set(frames) == {"Link Load", "Link Frequency", "Line Boarders"}


def test_missing_required_sheet_raises(tmp_path):
    """Required sheet absent => explicit SchemaDetectionError, never a guess."""
    bad = write_workbook(tmp_path / "TEST_DATA_missing.xlsx",
                         drop_sheets=("line_boarders",))
    with pytest.raises(SchemaDetectionError) as exc:
        detect_schema(bad, cfg.load_schema_mapping())
    assert "line_boarders" in str(exc.value) or "Line Boarders" in str(exc.value)


def test_read_sheet_header_detection(tmp_path):
    """Sheet with a title row above the header is still readable."""
    import pandas as pd

    from tests.conftest import link_rows

    raw = link_rows()
    path = tmp_path / "TEST_DATA_titlerow.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        pd.DataFrame([["TEST DATA — title", None, None]]).to_excel(
            xl, sheet_name="Link Load", index=False, header=False
        )
        raw.to_excel(xl, sheet_name="Link Load", startrow=1, index=False)
        from tests.conftest import frequency_rows, line_boarders_rows

        frequency_rows().to_excel(xl, sheet_name="Link Frequency", index=False)
        line_boarders_rows().to_excel(xl, sheet_name="Line Boarders", index=False)
    df = read_sheet(path, "Link Load")
    assert "Link Load" in [str(c) for c in df.columns]
