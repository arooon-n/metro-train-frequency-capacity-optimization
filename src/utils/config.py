"""Project-wide paths and configuration loading.

All paths are resolved relative to the project root so the package works from
any working directory (scripts, tests, Streamlit).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RUNS_DIR = OUTPUTS_DIR / "runs"
REPORTS_DIR = OUTPUTS_DIR / "reports"

NUMBAT_SOURCE_URL = "https://crowding.data.tfl.gov.uk/"

SCHEMA_MAPPING_PATH = CONFIG_DIR / "schema_mapping.yaml"
CAPACITY_PATH = CONFIG_DIR / "capacity_by_line.csv"
CONSTRAINTS_PATH = CONFIG_DIR / "service_constraints.csv"
FLEET_BUDGET_PATH = CONFIG_DIR / "fleet_budget.csv"
SCENARIO_PARAMS_PATH = CONFIG_DIR / "scenario_parameters.yaml"
GOAL_WEIGHTS_PATH = CONFIG_DIR / "goal_weights.yaml"
SERVICE_GROUPS_PATH = CONFIG_DIR / "service_groups.csv"


class ConfigError(RuntimeError):
    """Raised when a required configuration is missing or unusable."""


class DataFileMissingError(FileNotFoundError):
    """Raised when no NUMBAT workbook is available in data/raw."""


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, FIGURES_DIR, RUNS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file must contain a mapping: {path}")
    return data


def load_schema_mapping() -> dict[str, Any]:
    return load_yaml(SCHEMA_MAPPING_PATH)


def load_scenario_parameters() -> dict[str, Any]:
    return load_yaml(SCENARIO_PARAMS_PATH)


def load_goal_weights() -> dict[str, Any]:
    cfg = load_yaml(GOAL_WEIGHTS_PATH)
    weights = cfg.get("weights") or {}
    required = {"w1_unmet_demand", "w2_frequency_target", "w3_crowding", "w4_service_usage"}
    missing = required - set(weights)
    if missing:
        raise ConfigError(f"goal_weights.yaml is missing keys: {sorted(missing)}")
    for k, v in weights.items():
        if not isinstance(v, (int, float)) or v < 0:
            raise ConfigError(f"goal weight '{k}' must be a non-negative number (got {v!r})")
    return cfg


def _read_csv_config(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    df = pd.read_csv(path, comment="#", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(how="all")
    # drop rows that are entirely empty strings
    for c in df.columns:
        df[c] = df[c].astype("string").str.strip()
    df = df.replace({"": pd.NA, "nan": pd.NA})
    df = df.dropna(how="all")
    return df.reset_index(drop=True)


def load_capacity_config() -> pd.DataFrame:
    """Train capacity per line/service group. Empty capacity rows are dropped
    (never guessed); callers must check coverage of the lines they model."""
    df = _read_csv_config(CAPACITY_PATH)
    if df.empty:
        return df
    if "capacity_per_train" not in df.columns:
        raise ConfigError("capacity_by_line.csv must contain a capacity_per_train column")
    num = pd.to_numeric(df["capacity_per_train"], errors="coerce")
    bad = df[num.notna() & (num <= 0)]
    if len(bad):
        raise ConfigError(
            "capacity_by_line.csv contains non-positive capacities: "
            + bad.to_dict(orient="records").__repr__()
        )
    df = df[num.notna()].copy()
    df["capacity_per_train"] = num[num.notna()].astype(float)
    return df.reset_index(drop=True)


def load_service_constraints() -> pd.DataFrame:
    df = _read_csv_config(CONSTRAINTS_PATH)
    if df.empty:
        return df
    for col in ("min_frequency_per_15min", "max_frequency_per_15min", "target_frequency_per_15min"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "min_frequency_per_15min" in df and "max_frequency_per_15min" in df:
        both = df["min_frequency_per_15min"].notna() & df["max_frequency_per_15min"].notna()
        viol = df[both & (df["min_frequency_per_15min"] > df["max_frequency_per_15min"])]
        if len(viol):
            raise ConfigError(
                "service_constraints.csv has min > max frequency for rows: "
                + viol.to_dict(orient="records").__repr__()
            )
    return df.reset_index(drop=True)


def load_fleet_budget() -> pd.DataFrame:
    """Train-equivalent service budget per time period. Empty file => disabled."""
    df = _read_csv_config(FLEET_BUDGET_PATH)
    if df.empty:
        return df
    if "fleet_budget" not in df.columns or "time_period" not in df.columns:
        raise ConfigError("fleet_budget.csv must contain time_period and fleet_budget columns")
    df["fleet_budget"] = pd.to_numeric(df["fleet_budget"], errors="coerce")
    bad = df["fleet_budget"].isna()
    if bad.any():
        raise ConfigError(
            f"fleet_budget.csv has non-numeric fleet_budget values on rows "
            f"{list(df.index[bad])}"
        )
    if (df["fleet_budget"] < 0).any():
        raise ConfigError("fleet_budget.csv contains negative budgets")
    return df.reset_index(drop=True)


def load_service_groups() -> pd.DataFrame:
    df = _read_csv_config(SERVICE_GROUPS_PATH)
    if df.empty:
        return df
    needed = {"line", "service_group"}
    missing = needed - set(df.columns)
    if missing:
        raise ConfigError(f"service_groups.csv missing columns: {sorted(missing)}")
    return df.reset_index(drop=True)


def resolve_data_file(explicit: str | Path | None = None) -> Path:
    """Locate the NUMBAT workbook.

    Priority: explicit argument -> DATA_FILE env var -> first .xlsx in data/raw.
    Never invents a filename; raises a fully-worded error if nothing is found.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("DATA_FILE")
    if env:
        candidates.append(Path(env))
    if RAW_DIR.exists():
        candidates.extend(sorted(RAW_DIR.glob("*.xlsx")))
        candidates.extend(sorted(RAW_DIR.glob("*.xlsm")))

    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()

    raise DataFileMissingError(
        "\n"
        "==================== NUMBAT WORKBOOK NOT FOUND ====================\n"
        f"Looked for (in order): {[str(c) for c in candidates] or ['data/raw/*.xlsx']}\n\n"
        "How to fix:\n"
        f"  1. Download the official NUMBAT release from: {NUMBAT_SOURCE_URL}\n"
        "     (prefer NUMBAT 2025, Tuesday-Wednesday-Thursday 'TWT' weekday profile)\n"
        "  2. Place the downloaded .xlsx workbook inside: "
        f"{RAW_DIR}\n"
        "  3. Optionally point at it directly, e.g.\n"
        "       set DATA_FILE=path\\to\\numbat.xlsx      (PowerShell: $env:DATA_FILE=...)\n"
        "       python scripts/profile_dataset.py --data-file path\\to\\numbat.xlsx\n"
        "  4. Rerun the profiling script:\n"
        "       python scripts/profile_dataset.py\n"
        "=================================================================="
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
