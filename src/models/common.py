"""Shared preparation of model inputs: service groups, capacity, bounds, budgets.

Nothing here invents numbers. Missing capacity/bounds/budgets are represented
as explicit "not configured" states that the models must handle honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils import config as cfg
from src.utils.naming import norm_key


class ModelDataError(ValueError):
    """Raised when model inputs are incomplete or inconsistent."""


@dataclass
class ModelData:
    """Everything the three models need, at service-group x time-period level."""

    links: pd.DataFrame                     # link-level table with service_group
    line_demand: pd.DataFrame               # line_boarders per (line, time)
    periods: list[str]
    service_groups: list[str]
    capacity: dict[str, float]              # service_group -> capacity_per_train
    capacity_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    min_freq: dict[tuple[str, str], float] = field(default_factory=dict)
    max_freq: dict[tuple[str, str], float] = field(default_factory=dict)
    target_freq: dict[tuple[str, str], float] = field(default_factory=dict)
    observed_freq: dict[tuple[str, str], float] = field(default_factory=dict)
    fleet_budget: dict[str, float] = field(default_factory=dict)   # empty => disabled
    demand_multiplier: float = 1.0
    target_utilization: float = 0.80
    group_to_lines: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # --- convenience aggregations -------------------------------------
    def max_link_load(self) -> pd.DataFrame:
        """max link load per (service_group, time_period) — drives Model 1."""
        return (
            self.links.groupby(["service_group", "time_period"], as_index=False)
            .agg(
                max_link_load=("link_load", "max"),
                link_load_sum=("link_load", "sum"),
                n_links=("link_load", "size"),
            )
        )

    def group_demand(self) -> pd.DataFrame:
        """Line boarders aggregated to service groups (Model 2/3 Goal 1)."""
        return (
            self.line_demand.groupby(["service_group", "time_period"], as_index=False)
            .agg(line_boarders=("line_boarders", "sum"))
        )

    def scaled_max_load(self) -> pd.DataFrame:
        df = self.max_link_load()
        df["max_link_load"] = df["max_link_load"] * self.demand_multiplier
        return df

    def scaled_group_demand(self) -> pd.DataFrame:
        df = self.group_demand()
        df["line_boarders"] = df["line_boarders"] * self.demand_multiplier
        return df

    def link_rows_for_group_time(self) -> pd.DataFrame:
        out = self.links[["service_group", "time_period", "line",
                          "from_station", "to_station", "link_load"]].copy()
        out["link_load"] = out["link_load"] * self.demand_multiplier
        return out


def assign_service_groups(links: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Attach a service_group to every link.

    Default: service_group = line. If config/service_groups.csv has rows, a
    link belongs to the mapping row for its line whose [from_station,
    to_station] range contains the link's from_station; whole-line rows
    (empty range) match any link of that line. Unmapped lines fall back to
    the line name with no fabricated grouping.
    """
    out = links.copy()
    if mapping is None or mapping.empty:
        out["service_group"] = out["line"]
        return out

    mapping = mapping.copy()
    for c in ("from_station", "to_station"):
        if c not in mapping.columns:
            mapping[c] = pd.NA
    mapping["_key"] = mapping["line"].map(norm_key)

    sg = []
    for row in out.itertuples(index=False):
        key = norm_key(row.line)
        cand = mapping[mapping["_key"] == key]
        chosen = None
        if not cand.empty:
            for m in cand.itertuples(index=False):
                fs = getattr(m, "from_station", pd.NA)
                ts = getattr(m, "to_station", pd.NA)
                if pd.isna(fs) and pd.isna(ts):
                    chosen = m.service_group
                    break
                if pd.isna(fs) or pd.isna(ts):
                    chosen = m.service_group
                    break
                # directional range check using normalised labels
                if norm_key(fs) == norm_key(row.from_station) or norm_key(ts) == norm_key(row.to_station):
                    chosen = m.service_group
                    break
        sg.append(chosen if chosen is not None else row.line)
    out["service_group"] = sg
    return out


def prepare_model_data(
    links: pd.DataFrame,
    line_demand: pd.DataFrame,
    *,
    demand_multiplier: float = 1.0,
    target_utilization: float = 0.80,
    fleet_multiplier: float = 1.0,
    require_capacity: bool = True,
    require_line_demand: bool = False,
) -> ModelData:
    links = links.copy()
    line_demand = line_demand.copy()
    if links.empty:
        raise ModelDataError("No link rows available for the optimisation models.")

    sg_map = cfg.load_service_groups()
    links = assign_service_groups(links, sg_map)

    if line_demand.empty and require_line_demand:
        raise ModelDataError(
            "Line Boarders data is required for this model but is missing from the "
            "cleaned dataset. Check the line_boarders sheet detection in the profile."
        )
    if not line_demand.empty:
        if sg_map is not None and not sg_map.empty:
            line_demand = line_demand.merge(
                sg_map[["line", "service_group"]].drop_duplicates("line"),
                on="line", how="left",
            )
            line_demand["service_group"] = line_demand["service_group"].fillna(
                line_demand["line"]
            )
        else:
            line_demand["service_group"] = line_demand["line"]

    periods = sorted(links["time_period"].dropna().unique().tolist())
    service_groups = sorted(links["service_group"].dropna().unique().tolist())

    # ---- capacity (mandatory for capacity-constrained models) ----------
    cap_df = cfg.load_capacity_config()
    capacity: dict[str, float] = {}
    capacity_meta: dict[str, dict[str, Any]] = {}
    if not cap_df.empty:
        for row in cap_df.to_dict(orient="records"):
            sg = row.get("service_group")
            line = row.get("line")
            value = row.get("capacity_per_train")
            if pd.isna(value):
                continue
            key = sg if isinstance(sg, str) and sg.strip() else line
            if not isinstance(key, str) or not key.strip():
                continue
            capacity[key] = float(value)
            capacity_meta[key] = {
                "capacity_definition": row.get("capacity_definition"),
                "source": row.get("source"),
                "source_url": row.get("source_url"),
            }
            # also index by line for fallback lookup
            if isinstance(line, str) and line.strip() and line not in capacity:
                capacity[line] = float(value)
                capacity_meta[line] = capacity_meta[key]

    missing_cap = [g for g in service_groups if g not in capacity]
    if missing_cap and require_capacity:
        raise ModelDataError(
            "MISSING TRAIN CAPACITY for service groups: "
            f"{missing_cap}. NUMBAT does not provide capacity_per_train — fill "
            "config/capacity_by_line.csv from a citable source (column "
            "capacity_per_train) and rerun. Capacities are never guessed."
        )
    if missing_cap and not require_capacity:
        pass  # utilisation simply stays NaN for those groups

    # ---- frequency bounds / targets ------------------------------------
    con = cfg.load_service_constraints()
    min_freq: dict[tuple[str, str], float] = {}
    max_freq: dict[tuple[str, str], float] = {}
    target_freq: dict[tuple[str, str], float] = {}
    if not con.empty:
        for row in con.to_dict(orient="records"):
            sg = row.get("service_group")
            line = row.get("line")
            base = sg if isinstance(sg, str) and sg.strip() else line
            if not isinstance(base, str) or not base.strip():
                continue
            targets = [base]
            if isinstance(line, str) and line.strip() and line != base:
                targets.append(line)
            for t in targets:
                if pd.notna(row.get("min_frequency_per_15min")):
                    v = float(row["min_frequency_per_15min"])
                    for p in periods:
                        min_freq[(t, p)] = v
                if pd.notna(row.get("max_frequency_per_15min")):
                    v = float(row["max_frequency_per_15min"])
                    for p in periods:
                        max_freq[(t, p)] = v
                if pd.notna(row.get("target_frequency_per_15min")):
                    v = float(row["target_frequency_per_15min"])
                    for p in periods:
                        target_freq[(t, p)] = v

    # ---- observed frequency (median across links of the group/period) ---
    obs = (
        links.dropna(subset=["link_frequency"])
        .groupby(["service_group", "time_period"])["link_frequency"]
        .median()
    )
    observed_freq = {(g, p): float(v) for (g, p), v in obs.items()}

    # fill missing targets from observed frequency (documented behaviour)
    for key, v in observed_freq.items():
        target_freq.setdefault(key, v)

    # ---- fleet / train-equivalent service budget -----------------------
    fb = cfg.load_fleet_budget()
    fleet_budget: dict[str, float] = {}
    if not fb.empty:
        for row in fb.to_dict(orient="records"):
            fleet_budget[str(row["time_period"])] = float(row["fleet_budget"]) * fleet_multiplier

    group_to_lines = (
        links.groupby("service_group")["line"].apply(lambda s: sorted(set(map(str, s)))).to_dict()
    )

    warnings: list[str] = []
    if not fleet_budget:
        warnings.append(
            "fleet_budget.csv is empty — the service-budget constraint is DISABLED."
        )
    if missing_cap and not require_capacity:
        warnings.append(f"No capacity configured for: {missing_cap}; utilisation = NaN.")
    inconsistent = [
        g for g in service_groups
        if len({round(v, 6) for (gg, _), v in observed_freq.items() if gg == g}) > 1
    ]
    if inconsistent:
        warnings.append(
            f"Observed scheduled frequency varies across periods for: {inconsistent[:5]}"
            + ("..." if len(inconsistent) > 5 else "")
        )

    return ModelData(
        links=links,
        line_demand=line_demand,
        periods=periods,
        service_groups=service_groups,
        capacity=capacity,
        capacity_meta=capacity_meta,
        min_freq=min_freq,
        max_freq=max_freq,
        target_freq=target_freq,
        observed_freq=observed_freq,
        fleet_budget=fleet_budget,
        demand_multiplier=float(demand_multiplier),
        target_utilization=float(target_utilization),
        group_to_lines=group_to_lines,
        warnings=warnings,
    )
