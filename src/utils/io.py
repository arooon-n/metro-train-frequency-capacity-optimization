"""I/O helpers for processed data and result exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import config as cfg

CLEANED_PATH = cfg.PROCESSED_DIR / "cleaned_data.parquet"
PROFILE_PATH = cfg.PROCESSED_DIR / "dataset_profile.json"


class CleanedDataMissingError(FileNotFoundError):
    pass


def save_cleaned(df: pd.DataFrame, path: Path | str = CLEANED_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_cleaned(path: Path | str = CLEANED_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise CleanedDataMissingError(
            f"Cleaned dataset not found: {path}\n"
            "Run first:\n"
            "  1. place the official NUMBAT workbook in data/raw/\n"
            "  2. python scripts/profile_dataset.py\n"
            "  3. python scripts/preprocess.py"
        )
    return pd.read_parquet(path)


def load_profile(path: Path | str = PROFILE_PATH) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_profile(profile: dict[str, Any], path: Path | str = PROFILE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2, default=str)
    return path


def split_mode_lines(
    df: pd.DataFrame,
    mode: str | None = None,
    lines: list[str] | None = None,
) -> pd.DataFrame:
    out = df
    if mode and "mode" in out.columns:
        out = out[out["mode"].astype(str).str.casefold() == str(mode).casefold()]
    if lines:
        out = out[out["line"].isin(lines)]
    return out.copy()


def export_csv(df: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
