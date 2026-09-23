"""Data cleaning pipeline: raw NUMBAT sheets -> canonical long-format tables.

Guarantees:
  * raw files in data/raw/ are never modified;
  * original labels are preserved in ``*_raw`` columns;
  * duplicates and impossible (negative) values are detected and EXCLUDED with
    an audit trail — never silently repaired or filled;
  * 15-minute resolution and day type / mode / direction are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.ingestion.loader import DetectedSchema, SchemaDetectionError
from src.utils.naming import display_label, norm_key
from src.utils.time_utils import missing_periods, normalise_time_column

LINK_COLUMNS = [
    "record_type", "mode", "line", "service_group", "from_station", "to_station",
    "direction", "day_type", "time_period", "time_start", "time_end", "time_parsed",
    "link_load", "link_frequency",
]
LINE_COLUMNS = [
    "record_type", "mode", "line", "service_group", "direction", "day_type",
    "time_period", "time_start", "time_end", "time_parsed", "line_boarders",
]
STATION_COLUMNS = [
    "record_type", "mode", "line", "station", "day_type",
    "time_period", "time_start", "time_end", "time_parsed", "metric", "value",
]


@dataclass
class QualityReport:
    duplicate_rows_removed: int = 0
    negative_load_rows: int = 0
    negative_frequency_rows: int = 0
    null_key_rows: int = 0
    null_demand_rows: int = 0
    inconsistent_link_relationships: int = 0
    missing_15min_periods: dict[str, list[str]] = field(default_factory=dict)
    unmatched_frequency_links: int = 0
    day_types_present: list[str] = field(default_factory=list)
    day_type_filter_applied: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "negative_load_rows": self.negative_load_rows,
            "negative_frequency_rows": self.negative_frequency_rows,
            "null_key_rows": self.null_key_rows,
            "null_demand_rows": self.null_demand_rows,
            "inconsistent_link_relationships": self.inconsistent_link_relationships,
            "missing_15min_periods": self.missing_15min_periods,
            "unmatched_frequency_links": self.unmatched_frequency_links,
            "day_types_present": self.day_types_present,
            "day_type_filter_applied": self.day_type_filter_applied,
            "notes": self.notes,
        }


def _standardise_names(series: pd.Series, aliases: dict[str, str]) -> pd.Series:
    """Map messy original labels to canonical display labels via norm_key."""
    if not aliases:
        return series
    lookup = {norm_key(k): v for k, v in aliases.items()}
    return series.map(lambda v: lookup.get(norm_key(v), v) if pd.notna(v) else v)


def _extract(df: pd.DataFrame, columns: dict[str, str], role: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for logical, actual in columns.items():
        out[logical] = df[actual]
    return out


def _apply_day_type(
    df: pd.DataFrame,
    schema: DetectedSchema,
    day_filter: list[str],
    report: QualityReport,
    role_name: str = "line_boarders",
) -> pd.DataFrame:
    role = schema.roles.get(role_name)
    if "day_type" not in df.columns or df["day_type"].isna().all():
        hint = role.day_type_hint if role else None
        df["day_type"] = hint or "unspecified"
        if not hint:
            report.notes.append("No day_type column found; day_type set to 'unspecified'.")
        return df

    present = sorted({str(v) for v in df["day_type"].dropna().unique()})
    # accumulate across tables (links, line boarders, stations each call this)
    report.day_types_present = sorted(set(report.day_types_present) | set(present))
    if day_filter:
        keys = {norm_key(v) for v in day_filter}
        mask = df["day_type"].map(lambda v: norm_key(v) in keys or any(
            k in norm_key(v) or norm_key(v) in k for k in keys if k
        ))
        if mask.any():
            matched = sorted({str(v) for v in df.loc[mask, "day_type"].dropna().unique()})
            report.day_type_filter_applied = ", ".join(matched)
            return df.loc[mask].copy()
        report.notes.append(
            f"Configured day-type filter {day_filter} matched nothing; "
            f"day types present: {present}. Keeping ALL day types."
        )
    return df


def _clean_links(
    load_df: pd.DataFrame,
    freq_df: pd.DataFrame,
    schema: DetectedSchema,
    mapping: dict[str, Any],
    report: QualityReport,
) -> pd.DataFrame:
    loads = _extract(load_df, schema.roles["link_load"].columns, "link_load")
    freqs = _extract(freq_df, schema.roles["link_frequency"].columns, "link_frequency")

    for frame in (loads, freqs):
        if "mode" not in frame.columns:
            frame["mode"] = mapping.get("default_mode", "rail")
        elif frame["mode"].isna().all():
            frame["mode"] = mapping.get("default_mode", "rail")

    line_aliases = mapping.get("line_aliases") or {}
    station_aliases = mapping.get("station_aliases") or {}

    links = _merge_load_frequency(loads, freqs, report)

    # preserve originals, then normalise
    links["line_raw"] = links["line"]
    links["from_station_raw"] = links["from_station"]
    links["to_station_raw"] = links["to_station"]
    links["line"] = _standardise_names(links["line"], line_aliases)
    links["from_station"] = _standardise_names(links["from_station"], station_aliases)
    links["to_station"] = _standardise_names(links["to_station"], station_aliases)

    links = _apply_day_type(links, schema, mapping.get("day_type_filter") or [], report, "link_load")

    # recompute frequency-match coverage AFTER the day-type filter so that
    # rows for other day types (with no frequency sheet) are not reported
    report.notes = [n for n in report.notes if "no matching Link Frequency" not in n]
    if "link_frequency" in links.columns:
        unmatched = int(links["link_frequency"].isna().sum())
        report.unmatched_frequency_links = unmatched
        if unmatched:
            report.notes.append(
                f"{unmatched} link-load rows have no matching Link Frequency row "
                "(join on mode/line/from/to/day_type/time)."
            )

    links = links.rename(columns={"time_period": "time_raw"})
    t = normalise_time_column(links["time_raw"])
    links = pd.concat([links, t], axis=1)

    # key nulls
    key_null = links[["line", "from_station", "to_station", "time_period"]].isna().any(axis=1)
    report.null_key_rows = int(key_null.sum())
    links = links.loc[~key_null]

    # duplicates on natural key
    dup_keys = ["mode", "line", "from_station", "to_station", "day_type", "time_period"]
    dup_mask = links.duplicated(subset=dup_keys, keep="first")
    report.duplicate_rows_removed += int(dup_mask.sum())
    links = links.loc[~dup_mask]

    # negative / null numeric demand
    links["link_load"] = pd.to_numeric(links["link_load"], errors="coerce")
    links["link_frequency"] = pd.to_numeric(links["link_frequency"], errors="coerce")
    neg_load = links["link_load"] < 0
    report.negative_load_rows = int(neg_load.sum())
    links = links.loc[~neg_load.fillna(False)]
    neg_freq = links["link_frequency"] < 0
    report.negative_frequency_rows = int(neg_freq.sum())
    links = links.loc[~neg_freq.fillna(False)]

    null_demand = links["link_load"].isna()
    report.null_demand_rows += int(null_demand.sum())
    links = links.loc[~null_demand]

    # inconsistent station/link relationships: same pair, same period, conflicting direction
    pair_dir = links.groupby(
        ["line", "from_station", "to_station", "day_type", "time_period"], dropna=False
    )["direction"].nunique(dropna=True)
    report.inconsistent_link_relationships = int((pair_dir > 1).sum())

    # direction: derive simple forward/backward if absent (based on first-seen order per line)
    if "direction" not in links.columns or links["direction"].isna().all():
        links["direction"] = "unspecified"
        report.notes.append("No direction column found; direction set to 'unspecified'.")
    links["direction"] = links["direction"].astype("string").fillna("unspecified")

    # missing 15-minute periods (per line & day type, parsed times only)
    parsed = links.loc[links["time_parsed"]]
    for (line, day), grp in parsed.groupby(["line", "day_type"], dropna=False):
        miss = missing_periods(grp["time_period"].unique())
        if miss:
            report.missing_15min_periods[f"{line} | {day}"] = miss
    if (~links["time_parsed"]).any():
        report.notes.append(
            f"{int((~links['time_parsed']).sum())} link rows have non-ISO time labels; "
            "they are kept verbatim in time_period and excluded from 15-min gap checks."
        )

    links["record_type"] = "link"
    links["service_group"] = links["line"]  # placeholder; real mapping applied later
    return links.reset_index(drop=True)


def _merge_load_frequency(
    loads: pd.DataFrame, freqs: pd.DataFrame, report: QualityReport
) -> pd.DataFrame:
    keys = [c for c in ("mode", "line", "from_station", "to_station", "day_type", "time_period")
            if c in loads.columns and c in freqs.columns]
    loads = loads.copy()
    freqs = freqs.copy()

    # join on normalised keys but keep original label columns untouched
    tmp_keys: list[str] = []
    for k in keys:
        ck = f"__k_{k}"
        tmp_keys.append(ck)
        loads[ck] = loads[k].map(lambda v: norm_key(v) if pd.notna(v) else v)
        freqs[ck] = freqs[k].map(lambda v: norm_key(v) if pd.notna(v) else v)

    if "link_frequency" in loads.columns and "link_load" not in loads.columns:
        loads, freqs = freqs, loads
        tmp_keys = [f"__k_{k}" for k in keys]

    keep = tmp_keys + [c for c in ("direction", "link_frequency") if c in freqs.columns]
    freqs = freqs[keep].drop_duplicates(subset=tmp_keys, keep="first")
    if "direction" in freqs.columns and "direction" in loads.columns:
        freqs = freqs.rename(columns={"direction": "direction_freq"})
    merged = loads.merge(freqs, on=tmp_keys, how="left")
    merged = merged.drop(columns=tmp_keys, errors="ignore")

    if "direction" not in merged.columns and "direction_freq" in merged.columns:
        merged["direction"] = merged["direction_freq"]
    if "direction_freq" in merged.columns:
        merged = merged.drop(columns=["direction_freq"])

    if "link_frequency" in merged.columns:
        unmatched = int(merged["link_frequency"].isna().sum())
        report.unmatched_frequency_links = unmatched
        if unmatched:
            report.notes.append(
                f"{unmatched} link-load rows have no matching Link Frequency row "
                "(join on mode/line/from/to/day_type/time)."
            )
    return merged


def _clean_line_boarders(
    df: pd.DataFrame, schema: DetectedSchema, mapping: dict[str, Any], report: QualityReport
) -> pd.DataFrame:
    lb = _extract(df, schema.roles["line_boarders"].columns, "line_boarders")
    if "mode" not in lb.columns or lb["mode"].isna().all():
        lb["mode"] = mapping.get("default_mode", "rail")
    line_aliases = mapping.get("line_aliases") or {}
    lb["line_raw"] = lb["line"]
    lb["line"] = _standardise_names(lb["line"], line_aliases)
    lb = _apply_day_type(lb, schema, mapping.get("day_type_filter") or [], report, "line_boarders")
    lb = lb.rename(columns={"time_period": "time_raw"})
    t = normalise_time_column(lb["time_raw"])
    lb = pd.concat([lb, t], axis=1)

    key_null = lb[["line", "time_period"]].isna().any(axis=1)
    report.null_key_rows += int(key_null.sum())
    lb = lb.loc[~key_null]

    dup_keys = ["mode", "line", "day_type", "time_period"]
    dup = lb.duplicated(subset=dup_keys, keep="first")
    report.duplicate_rows_removed += int(dup.sum())
    lb = lb.loc[~dup]

    lb["line_boarders"] = pd.to_numeric(lb["line_boarders"], errors="coerce")
    neg = lb["line_boarders"] < 0
    report.negative_load_rows += int(neg.sum())
    lb = lb.loc[~neg.fillna(False)]
    nulls = lb["line_boarders"].isna()
    report.null_demand_rows += int(nulls.sum())
    lb = lb.loc[~nulls]

    if "direction" not in lb.columns:
        lb["direction"] = "unspecified"
    lb["record_type"] = "line_boarders"
    lb["service_group"] = lb["line"]
    return lb.reset_index(drop=True)


def _clean_station(
    df: pd.DataFrame, schema: DetectedSchema, mapping: dict[str, Any], report: QualityReport
) -> pd.DataFrame | None:
    role = schema.roles.get("station")
    if role is None:
        return None
    metric_cols = {
        "station_boarders": "boarders",
        "station_alighters": "alighters",
        "station_entry": "entry",
        "station_exit": "exit",
    }
    available = {k: v for k, v in metric_cols.items() if k in role.columns}
    if not available or "station" not in role.columns:
        return None

    base = _extract(df, role.columns, "station")
    if "mode" not in base.columns or base["mode"].isna().all():
        base["mode"] = mapping.get("default_mode", "rail")
    base = _apply_day_type(base, schema, mapping.get("day_type_filter") or [], report, "station")
    base = base.rename(columns={"time_period": "time_raw"})
    t = normalise_time_column(base["time_raw"])
    base = pd.concat([base, t], axis=1)

    frames = []
    for field, metric in available.items():
        tmp = base.copy()
        tmp["metric"] = metric
        tmp["value"] = pd.to_numeric(tmp[field], errors="coerce")
        frames.append(tmp)
    st = pd.concat(frames, ignore_index=True)
    st = st.dropna(subset=["station", "value"])
    neg = st["value"] < 0
    report.negative_load_rows += int(neg.sum())
    st = st.loc[~neg]
    st["record_type"] = "station"
    cols = ["record_type", "mode", "line" if "line" in st.columns else "mode", "station",
            "day_type", "time_period", "time_start", "time_end", "time_parsed", "metric", "value"]
    cols = list(dict.fromkeys(cols))
    for c in cols:
        if c not in st.columns:
            st[c] = pd.NA
    if "line" not in st.columns:
        st["line"] = pd.NA
    return st[STATION_COLUMNS].reset_index(drop=True)


def preprocess(
    frames: dict[str, pd.DataFrame],
    schema: DetectedSchema,
    mapping: dict[str, Any],
) -> tuple[pd.DataFrame, QualityReport]:
    """Build the canonical cleaned dataset (record_type: link/line_boarders/station)."""
    report = QualityReport()
    if "link_load" not in schema.roles:
        raise SchemaDetectionError("link_load sheet was not detected — cannot preprocess.")

    load_frame = frames[schema.roles["link_load"].sheet]
    freq_frame = frames[schema.roles["link_frequency"].sheet]
    links = _clean_links(load_frame, freq_frame, schema, mapping, report)

    frames_out = [links]

    if "line_boarders" in schema.roles:
        lb = _clean_line_boarders(
            frames[schema.roles["line_boarders"].sheet], schema, mapping, report
        )
        frames_out.append(lb)
    else:
        raise SchemaDetectionError("line_boarders sheet was not detected — cannot preprocess.")

    st = None
    if "station" in schema.roles:
        st = _clean_station(frames[schema.roles["station"].sheet], schema, mapping, report)
    if st is not None and not st.empty:
        frames_out.append(st)

    cleaned = pd.concat(frames_out, ignore_index=True, sort=False)

    if cleaned.empty:
        raise SchemaDetectionError(
            "Preprocessing produced an empty dataset — check day-type filter and columns."
        )
    return cleaned, report


def build_profile(
    workbook_info: dict[str, Any],
    schema: DetectedSchema,
    report: QualityReport,
    cleaned: pd.DataFrame | None,
) -> dict[str, Any]:
    links = cleaned[cleaned["record_type"] == "link"] if cleaned is not None else pd.DataFrame()
    lb = cleaned[cleaned["record_type"] == "line_boarders"] if cleaned is not None else pd.DataFrame()
    profile: dict[str, Any] = {
        **workbook_info,
        "detected_roles": {
            role: {"sheet": r.sheet, "columns": r.columns, "day_type_hint": r.day_type_hint}
            for role, r in schema.roles.items()
        },
        "quality": report.to_dict(),
        "cleaned_summary": {},
    }
    profile["quality"]["blocking_issues"] = bool(
        report.null_demand_rows
        or report.null_key_rows
        or report.negative_load_rows
        or report.negative_frequency_rows
    )
    if not links.empty:
        profile["cleaned_summary"] = {
            "link_rows": int(len(links)),
            "line_boarders_rows": int(len(lb)),
            "modes": sorted(map(str, links["mode"].dropna().unique())),
            "lines": sorted(map(str, links["line"].dropna().unique())),
            "stations": int(
                pd.unique(
                    pd.concat([links["from_station"], links["to_station"]], ignore_index=True)
                ).size
            ),
            "unique_links": int(
                links[["line", "from_station", "to_station", "direction"]]
                .drop_duplicates().shape[0]
            ),
            "day_types": sorted(map(str, links["day_type"].dropna().unique())),
            "time_periods": int(links["time_period"].nunique()),
            "link_load_total_sum": float(links["link_load"].sum()),
            "link_load_note": (
                "Sum of link loads is a link-passenger-km-like exposure measure, "
                "NOT unique passenger journeys."
            ),
            "line_boarders_total_sum": float(lb["line_boarders"].sum()) if not lb.empty else None,
            "frequency_range": [
                float(links["link_frequency"].min()) if links["link_frequency"].notna().any() else None,
                float(links["link_frequency"].max()) if links["link_frequency"].notna().any() else None,
            ],
        }
    return profile
