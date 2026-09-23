"""NUMBAT workbook ingestion: sheet listing, flexible sheet/column detection.

Design rules:
  * never guess: a required role that cannot be detected raises SchemaDetectionError
    listing every sheet/column actually present, so the user can edit
    config/schema_mapping.yaml instead of us inventing a mapping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.naming import norm_key
from src.utils.config import NUMBAT_SOURCE_URL


class SchemaDetectionError(RuntimeError):
    """Raised when required sheets/columns cannot be identified automatically."""


def list_sheets(path: Path | str) -> list[str]:
    xl = pd.ExcelFile(path, engine="openpyxl")
    try:
        return list(xl.sheet_names)
    finally:
        xl.close()


def _norm_col(name: object) -> str:
    return norm_key(name)


def _find_column(columns: list[str], aliases: list[str]) -> str | None:
    normed = {_norm_col(c): c for c in columns}
    for alias in aliases:
        a = norm_key(alias)
        if not a:
            continue
        if a in normed:
            return normed[a]
    for alias in aliases:
        a = norm_key(alias)
        if not a:
            continue
        for nk, original in normed.items():
            if a in nk:
                return original
    return None


def _score_sheet(sheet_name: str, aliases: list[str]) -> int:
    sn = norm_key(sheet_name)
    best = 0
    for alias in aliases:
        a = norm_key(alias)
        if not a:
            continue
        if sn == a:
            best = max(best, 100)
        elif a in sn:
            best = max(best, 60)
    return best


def _day_type_from_sheet_name(sheet_name: str, day_aliases: list[str]) -> str | None:
    sn = norm_key(sheet_name)
    for a in day_aliases:
        ak = norm_key(a)
        if ak and ak in sn:
            return str(a)
    return None


def read_sheet(path: Path | str, sheet: str, header_search_rows: int = 5) -> pd.DataFrame:
    """Read a sheet, searching the first rows for a plausible header row."""
    raw = pd.read_excel(path, sheet_name=sheet, header=0, engine="openpyxl", dtype=object)
    if raw.shape[1] == 0:
        return raw

    def quality(frame: pd.DataFrame) -> int:
        cols = [c for c in frame.columns if not str(c).startswith("Unnamed")]
        if not cols:
            return -1
        return sum(1 for c in cols if norm_key(c)) - 3 * sum(
            1 for c in cols if str(c).startswith("Unnamed")
        )

    best, best_q = raw, quality(raw)
    for hdr in range(1, max(1, header_search_rows)):
        try:
            cand = pd.read_excel(
                path, sheet_name=sheet, header=hdr, engine="openpyxl", dtype=object
            )
        except Exception:
            continue
        q = quality(cand)
        if q > best_q:
            best, best_q = cand, q
    best.columns = [str(c).strip() for c in best.columns]
    return best


@dataclass
class SheetRole:
    role: str
    sheet: str
    columns: dict[str, str]          # logical field -> actual column name
    day_type_hint: str | None = None


@dataclass
class DetectedSchema:
    workbook: str
    sheets: list[str]
    sheet_shapes: dict[str, list[int]]
    roles: dict[str, SheetRole] = field(default_factory=dict)

    def column_listing(self) -> dict[str, list[str]]:
        return {s: list(cols) for s, cols in self.sheet_shapes.items()}


def detect_schema(
    path: Path | str,
    mapping: dict[str, Any],
) -> tuple[DetectedSchema, dict[str, pd.DataFrame]]:
    """Detect sheets/columns for every logical role; read each sheet once."""
    path = Path(path)
    sheets = list_sheets(path)
    header_rows = int(mapping.get("header_search_rows", 5))
    sheet_cfg: dict[str, Any] = mapping.get("sheets", {})
    col_cfg: dict[str, Any] = mapping.get("columns", {})

    frames: dict[str, pd.DataFrame] = {}
    shapes: dict[str, list[int]] = {}
    col_index: dict[str, list[str]] = {}
    for s in sheets:
        df = read_sheet(path, s, header_search_rows=header_rows)
        frames[s] = df
        shapes[s] = [int(df.shape[0]), int(df.shape[1])]
        col_index[s] = [str(c) for c in df.columns]

    detected = DetectedSchema(workbook=path.name, sheets=sheets, sheet_shapes=shapes)

    # ---- pick a sheet for each role -------------------------------------
    role_scores: dict[str, list[tuple[int, str]]] = {}
    for role, cfg in sheet_cfg.items():
        aliases = list(cfg.get("aliases", [role]))
        scored = [(_score_sheet(s, aliases), s) for s in sheets]
        scored = [x for x in scored if x[0] > 0]
        scored.sort(key=lambda x: -x[0])
        role_scores[role] = scored

    def required_columns(role: str) -> dict[str, Any]:
        out = {}
        for field_name, fcfg in col_cfg.items():
            applies = fcfg.get("applies_to")
            if applies and role not in applies:
                continue
            out[field_name] = fcfg
        return out

    def match_columns(role: str, sheet: str, need: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
        cols = col_index[sheet]
        found: dict[str, str] = {}
        missing: list[str] = []
        for field_name, fcfg in need.items():
            if role == "line_boarders" and field_name in ("from_station", "to_station", "link_load", "link_frequency"):
                continue
            if role != "link_load" and field_name in ("link_load",):
                continue
            if role != "link_frequency" and field_name in ("link_frequency",):
                continue
            if role != "line_boarders" and field_name == "line_boarders":
                continue
            if role == "line_boarders" and field_name == "line_boarders":
                pass
            col = _find_column(cols, list(fcfg.get("aliases", [field_name])))
            if col is not None:
                found[field_name] = col
            elif fcfg.get("required", False):
                missing.append(field_name)
        return found, missing

    # sheets that carry required columns, in score order, then by column fit
    role_need = required_columns_roles(col_cfg)

    # optional roles must ALSO be justified by a sheet-name match (and any
    # defining column) so they never silently fall back onto the wrong sheet
    defining_cols = {"station": ["station"], "od_flow": [], "journey_time": []}

    for role, cfg in sheet_cfg.items():
        need = role_need[role]
        is_required = bool(cfg.get("required", False))
        aliases = list(cfg.get("aliases", [role]))
        # field-level applies_to filtering is inside match_columns via col_cfg
        candidates = role_scores.get(role, [])
        # add all sheets as fallback candidates (a sheet may fit without name match)
        ordered = [s for _, s in candidates] + [s for s in sheets if s not in [x[1] for x in candidates]]
        best: tuple[str, dict[str, str]] | None = None
        last_missing: list[str] = []
        for sheet in ordered:
            if not is_required:
                if _score_sheet(sheet, aliases) <= 0:
                    continue
                if any(c not in col_index[sheet] or
                       _find_column(col_index[sheet], [c]) is None
                       for c in defining_cols.get(role, [])):
                    # defining columns absent (checked via found dict below)
                    pass
            found, missing = match_columns(role, sheet, need)
            req_missing = list(missing)
            if not req_missing:
                if not is_required:
                    defining = defining_cols.get(role, [])
                    if any(d not in found for d in defining):
                        last_missing = defining
                        continue
                if best is None or len(found) > len(best[1]):
                    best = (sheet, found)
                    if len(found) == len(need):
                        break
            else:
                last_missing = req_missing
        if best is None:
            if cfg.get("required", False):
                raise SchemaDetectionError(
                    _schema_error_message(
                        role=role,
                        path=path,
                        sheets=sheets,
                        col_index=col_index,
                        missing_fields=last_missing or list(need),
                        mapping_hint="config/schema_mapping.yaml",
                    )
                )
            continue
        sheet, found = best
        hint = _day_type_from_sheet_name(
            sheet, ["TWT", "Tue/Wed/Thu", "Tuesday/Wednesday/Thursday", "weekday", "Saturday", "Sunday", "Friday", "Monday"]
        )
        detected.roles[role] = SheetRole(role=role, sheet=sheet, columns=found, day_type_hint=hint)

    for required_role in ("link_load", "link_frequency", "line_boarders"):
        if required_role not in detected.roles:
            raise SchemaDetectionError(
                f"Required sheet/role '{required_role}' could not be identified in "
                f"{path.name}. Sheets found: {sheets}. "
                f"Edit {mapping.get('_hint', 'config/schema_mapping.yaml')} aliases and rerun."
            )

    return detected, frames


def required_columns_roles(col_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build the per-role required/optional column checklist."""
    roles = {
        "link_load": ["mode", "line", "from_station", "to_station", "direction",
                      "day_type", "time_period", "link_load"],
        "link_frequency": ["mode", "line", "from_station", "to_station", "direction",
                           "day_type", "time_period", "link_frequency"],
        "line_boarders": ["mode", "line", "day_type", "time_period", "line_boarders"],
        "station": ["mode", "line", "station", "day_type", "time_period"],
        "od_flow": ["line", "day_type", "time_period"],
        "journey_time": ["line", "day_type", "time_period"],
    }
    out: dict[str, dict[str, Any]] = {}
    for role, fields in roles.items():
        need = {}
        for f in fields:
            if f in col_cfg:
                need[f] = col_cfg[f]
        out[role] = need
    return out


def _schema_error_message(
    role: str,
    path: Path,
    sheets: list[str],
    col_index: dict[str, list[str]],
    missing_fields: list[str],
    mapping_hint: str,
) -> str:
    lines = [
        "==================== SCHEMA DETECTITON FAILED ====================",  # fixed below
    ]
    lines[0] = "==================== SCHEMA DETECTION FAILED ===================="
    lines += [
        f"Workbook : {path}",
        f"Role     : {role}",
        f"Missing  : {missing_fields}",
        "",
        "Sheets and their columns:",
    ]
    for s in sheets:
        lines.append(f"  - {s}: {col_index[s]}")
    lines += [
        "",
        "How to fix (do NOT let the tool guess):",
        f"  1. Open {mapping_hint}",
        "  2. Add the real sheet/column names to the relevant aliases list",
        "  3. Rerun: python scripts/profile_dataset.py",
        "================================================================",
    ]
    return "\n".join(lines)


def profile_workbook(path: Path | str, mapping: dict[str, Any]) -> dict[str, Any]:
    """Cheap profile: sheet names, shapes, columns — used before full detection."""
    path = Path(path)
    sheets = list_sheets(path)
    info: dict[str, Any] = {
        "workbook": path.name,
        "source_url": NUMBAT_SOURCE_URL,
        "sheet_count": len(sheets),
        "sheets": [],
    }
    header_rows = int(mapping.get("header_search_rows", 5))
    for s in sheets:
        df = read_sheet(path, s, header_search_rows=header_rows)
        info["sheets"].append(
            {
                "name": s,
                "rows": int(df.shape[0]),
                "columns_count": int(df.shape[1]),
                "columns": [str(c) for c in df.columns],
            }
        )
    return info
