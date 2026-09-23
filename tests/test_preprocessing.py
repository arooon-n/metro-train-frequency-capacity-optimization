"""Preprocessing tests: time normalisation, duplicates, negatives, day types."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.loader import detect_schema
from src.preprocessing.pipeline import preprocess
from src.utils import config as cfg
from src.utils.time_utils import (
    expected_time_periods,
    missing_periods,
    normalise_time_column,
    parse_timeband,
)
from tests.conftest import link_rows, write_workbook


@pytest.mark.parametrize(
    "raw,label",
    [
        ("04:00-04:15", "04:00-04:15"),
        ("0400-0415", "04:00-04:15"),
        ("07:45-08:00", "07:45-08:00"),
        ("7:00-7:15", "07:00-07:15"),
    ],
)
def test_parse_timeband(raw, label):
    got, start, end, ok = parse_timeband(raw)
    assert ok
    assert got == label
    assert end - start == 15


def test_parse_timeband_free_text_preserved():
    got, start, end, ok = parse_timeband("AM Peak")
    assert not ok
    assert got == "AM Peak"      # preserved, never rewritten
    assert start is None


def test_expected_periods_count():
    assert len(expected_time_periods()) == 84
    assert expected_time_periods()[0] == "04:00-04:15"


def test_missing_periods():
    obs = expected_time_periods()[1:3]
    miss = missing_periods(obs)
    assert expected_time_periods()[0] in miss
    assert len(miss) == 82


def test_normalise_time_column():
    s = pd.Series(["07:00-07:15", "nonsense"])
    df = normalise_time_column(s)
    assert bool(df.loc[0, "time_parsed"]) is True
    assert bool(df.loc[1, "time_parsed"]) is False
    assert df.loc[1, "time_period"] == "nonsense"


def test_preprocess_clean_workbook(test_workbook):
    mapping = cfg.load_schema_mapping()
    schema, frames = detect_schema(test_workbook, mapping)
    cleaned, report = preprocess(frames, schema, mapping)
    links = cleaned[cleaned["record_type"] == "link"]
    assert not links.empty
    assert report.duplicate_rows_removed == 0
    assert report.negative_load_rows == 0
    assert set(links["day_type"]) == {"TWT"}
    assert "line_raw" in links.columns          # originals preserved
    assert {"link_load", "link_frequency"} <= set(links.columns)


def test_duplicate_detection(tmp_path):
    base = link_rows()
    dup = pd.concat([base, base.head(2)], ignore_index=True)
    p = write_workbook(tmp_path / "TEST_DATA_dupes.xlsx", links=dup)
    mapping = cfg.load_schema_mapping()
    schema, frames = detect_schema(p, mapping)
    _cleaned, report = preprocess(frames, schema, mapping)
    assert report.duplicate_rows_removed == 2


def test_negative_value_detection(tmp_path):
    base = link_rows()
    base.loc[0, "Link Load"] = -50
    p = write_workbook(tmp_path / "TEST_DATA_neg.xlsx", links=base)
    mapping = cfg.load_schema_mapping()
    schema, frames = detect_schema(p, mapping)
    cleaned, report = preprocess(frames, schema, mapping)
    assert report.negative_load_rows == 1
    links = cleaned[cleaned["record_type"] == "link"]
    assert (links["link_load"] >= 0).all()


def test_null_demand_not_silently_filled(tmp_path):
    base = link_rows()
    base.loc[3, "Link Load"] = None
    p = write_workbook(tmp_path / "TEST_DATA_nulls.xlsx", links=base)
    mapping = cfg.load_schema_mapping()
    schema, frames = detect_schema(p, mapping)
    cleaned, report = preprocess(frames, schema, mapping)
    assert report.null_demand_rows == 1
    links = cleaned[cleaned["record_type"] == "link"]
    assert links["link_load"].notna().all()      # dropped, not filled


def test_day_type_filter_keeps_only_twt(tmp_path):
    multi = pd.concat(
        [link_rows("TWT"), link_rows("Saturday")], ignore_index=True
    )
    p = write_workbook(tmp_path / "TEST_DATA_days.xlsx", links=multi)
    mapping = cfg.load_schema_mapping()
    schema, frames = detect_schema(p, mapping)
    cleaned, report = preprocess(frames, schema, mapping)
    links = cleaned[cleaned["record_type"] == "link"]
    assert set(links["day_type"]) == {"TWT"}
    assert "Saturday" in report.day_types_present
