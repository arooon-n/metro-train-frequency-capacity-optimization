"""Reproducible run manifests: every optimisation run writes outputs/runs/<ts>/."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.utils import config as cfg


def new_run_dir(prefix: str = "run") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = cfg.RUNS_DIR / f"{ts}_{prefix}"
    n = 1
    while path.exists():
        path = cfg.RUNS_DIR / f"{ts}_{prefix}_{n}"
        n += 1
    path.mkdir(parents=True, exist_ok=False)
    (path / "figures").mkdir(exist_ok=True)
    return path


def snapshot_config(run_dir: Path) -> Path:
    """Copy the exact configuration files used for this run."""
    snap = run_dir / "config_snapshot"
    snap.mkdir(exist_ok=True)
    for p in sorted(cfg.CONFIG_DIR.iterdir()):
        if p.is_file():
            shutil.copy2(p, snap / p.name)
    return snap


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def write_run_metadata(
    run_dir: Path,
    *,
    dataset_path: Path | None,
    model_name: str,
    scenario: str,
    parameters: dict[str, Any],
    solver_name: str,
    solver_status: str,
    selected_mode: str | None = None,
    selected_lines: list[str] | None = None,
    selected_time_periods: list[str] | None = None,
    dataset_year: str | int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    meta: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "scenario": scenario,
        "solver_name": solver_name,
        "solver_status": solver_status,
        "parameter_values": parameters,
        "selected_mode": selected_mode,
        "selected_lines": selected_lines,
        "selected_time_periods": selected_time_periods,
    }
    if dataset_path is not None and Path(dataset_path).exists():
        meta["dataset_filename"] = Path(dataset_path).name
        meta["dataset_hash"] = cfg.file_sha256(Path(dataset_path))
        meta["dataset_path"] = str(dataset_path)
    else:
        meta["dataset_filename"] = None
        meta["dataset_hash"] = None
    if dataset_year is not None:
        meta["dataset_year"] = str(dataset_year)
    if extra:
        meta.update(extra)
    path = run_dir / "run_metadata.json"
    write_json(path, meta)
    with (run_dir / "config_snapshot" / "run_parameters.yaml").open(
        "w", encoding="utf-8"
    ) as fh:
        yaml.safe_dump(parameters, fh, sort_keys=False)
    return path
