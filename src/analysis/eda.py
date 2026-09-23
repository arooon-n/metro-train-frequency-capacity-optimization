"""Exploratory analyses (pure functions returning tidy DataFrames).

Utilisation is only computed when an explicit capacity_per_train exists —
capacity is NEVER invented here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.time_utils import label_from_minutes


def split_tables(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    links = cleaned[cleaned["record_type"] == "link"].copy()
    line_df = cleaned[cleaned["record_type"] == "line_boarders"].copy()
    station = cleaned[cleaned["record_type"] == "station"].copy()
    return links, line_df, station


def _is_peak(time_start, peak_windows: list[list[str]]) -> bool:
    if pd.isna(time_start):
        return False
    t = int(time_start)
    for start, end in peak_windows:
        sh, sm = (int(x) for x in str(start).split(":"))
        eh, em = (int(x) for x in str(end).split(":"))
        s, e = sh * 60 + sm, eh * 60 + em
        if s <= t < e:
            return True
    return False


def demand_by_time(links: pd.DataFrame, line_df: pd.DataFrame) -> pd.DataFrame:
    a = links.groupby("time_period", as_index=False).agg(
        link_load_sum=("link_load", "sum"),
        link_load_max=("link_load", "max"),
        mean_frequency=("link_frequency", "mean"),
    )
    a["source"] = "link_load"
    if not line_df.empty:
        b = line_df.groupby("time_period", as_index=False)["line_boarders"].sum()
        b["source"] = "line_boarders"
        b = b.rename(columns={"line_boarders": "link_load_sum"})
        b["link_load_max"] = np.nan
        b["mean_frequency"] = np.nan
        return pd.concat([a, b[b.columns]], ignore_index=True)
    return a


def demand_by_line(links: pd.DataFrame, line_df: pd.DataFrame) -> pd.DataFrame:
    a = links.groupby("line", as_index=False).agg(
        link_load_sum=("link_load", "sum"),
        peak_link_load=("link_load", "max"),
        mean_frequency=("link_frequency", "mean"),
    )
    if not line_df.empty:
        b = line_df.groupby("line", as_index=False)["line_boarders"].sum()
        a = a.merge(b, on="line", how="outer")
    return a.fillna(0)


def demand_by_link(links: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    g = links.groupby(
        ["line", "direction", "from_station", "to_station"], as_index=False
    ).agg(
        link_load_sum=("link_load", "sum"),
        peak_link_load=("link_load", "max"),
        mean_frequency=("link_frequency", "mean"),
    ).sort_values("peak_link_load", ascending=False)
    return g.head(top_n) if top_n else g


def peak_vs_offpeak(
    links: pd.DataFrame, peak_windows: list[list[str]]
) -> pd.DataFrame:
    df = links.copy()
    df["is_peak"] = df["time_start"].map(lambda t: _is_peak(t, peak_windows))
    out = df.groupby("is_peak", as_index=False).agg(
        link_load_sum=("link_load", "sum"),
        rows=("link_load", "size"),
        mean_frequency=("link_frequency", "mean"),
    )
    out["period_type"] = out["is_peak"].map({True: "peak", False: "off-peak"})
    total = out["link_load_sum"].sum()
    out["share_of_demand"] = out["link_load_sum"] / total if total else np.nan
    return out


def frequency_by_time(links: pd.DataFrame) -> pd.DataFrame:
    return links.groupby("time_period", as_index=False).agg(
        mean_frequency=("link_frequency", "mean"),
        min_frequency=("link_frequency", "min"),
        max_frequency=("link_frequency", "max"),
        trains_per_period_sum=("link_frequency", "sum"),
    )


def demand_frequency_relationship(links: pd.DataFrame) -> pd.DataFrame:
    """Link-period observations for scatter analysis."""
    return links[["line", "time_period", "from_station", "to_station",
                  "link_load", "link_frequency"]].dropna(subset=["link_load"])


def high_load_links(links: pd.DataFrame, quantile: float = 0.95, top_n: int = 50) -> pd.DataFrame:
    thresh = links["link_load"].quantile(quantile)
    out = links[links["link_load"] >= thresh].sort_values("link_load", ascending=False)
    return out.head(top_n)


def baseline_utilization(
    links: pd.DataFrame, capacity: pd.DataFrame
) -> pd.DataFrame:
    """utilisation = link_load / (capacity_per_train * scheduled_frequency).

    Requires an explicit capacity table; rows without capacity stay NaN
    (never filled with a guess).
    """
    if capacity is None or capacity.empty:
        out = links.copy()
        out["baseline_utilization"] = np.nan
        out["capacity_per_train"] = np.nan
        out["capacity_available"] = False
        return out

    cap = capacity.copy()
    key = "service_group" if "service_group" in cap.columns and cap["service_group"].notna().any() else "line"
    cap = cap[["line", key, "capacity_per_train"]].drop_duplicates(subset=["line"])
    df = links.merge(cap[["line", "capacity_per_train"]], on="line", how="left")
    denom = df["capacity_per_train"] * df["link_frequency"]
    df["baseline_utilization"] = np.where(
        denom > 0, df["link_load"] / denom, np.nan
    )
    df["capacity_available"] = df["capacity_per_train"].notna()
    return df


def underserved_periods(
    links: pd.DataFrame, capacity: pd.DataFrame, target_util: float
) -> pd.DataFrame:
    util = baseline_utilization(links, capacity)
    if util["capacity_available"].sum() == 0:
        return pd.DataFrame(
            columns=["line", "time_period", "max_utilization", "observed_frequency",
                     "required_frequency"]
        )
    g = util.dropna(subset=["baseline_utilization"]).groupby(
        ["line", "time_period"], as_index=False
    ).agg(
        max_utilization=("baseline_utilization", "max"),
        max_link_load=("link_load", "max"),
        observed_frequency=("link_frequency", "median"),
        capacity_per_train=("capacity_per_train", "first"),
    )
    out = g[g["max_utilization"] > target_util].copy()
    out["required_frequency"] = np.ceil(
        out["max_link_load"] / out["capacity_per_train"] / target_util
    )
    return out.sort_values("max_utilization", ascending=False)


def underutilized_periods(
    links: pd.DataFrame, capacity: pd.DataFrame, target_util: float, floor: float = 0.25
) -> pd.DataFrame:
    util = baseline_utilization(links, capacity)
    if util["capacity_available"].sum() == 0:
        return pd.DataFrame(columns=["line", "time_period", "max_utilization"])
    g = util.dropna(subset=["baseline_utilization"]).groupby(
        ["line", "time_period"], as_index=False
    ).agg(
        max_utilization=("baseline_utilization", "max"),
        observed_frequency=("link_frequency", "median"),
        max_link_load=("link_load", "max"),
    )
    return g[g["max_utilization"] < floor].sort_values("max_utilization")


def utilization_summary(util_df: pd.DataFrame) -> dict:
    s = util_df["baseline_utilization"].dropna()
    if s.empty:
        return {"available": False,
                "note": "No capacity configured — utilisation not computed."}
    return {
        "available": True,
        "mean": float(s.mean()),
        "max": float(s.max()),
        "p95": float(s.quantile(0.95)),
        "count_above_1": int((s > 1.0).sum()),
        "count_obs": int(s.size),
    }


def time_order(periods) -> list[str]:
    """Sort canonical labels chronologically (handles 24:00+ wraps)."""
    def key(p: str):
        try:
            head = str(p).split("-")[0]
            h, m = head.split(":")
            v = int(h) * 60 + int(m)
            if int(h) < 4:
                v += 24 * 60
            return v
        except Exception:
            return 10**6
    return sorted(periods, key=key)


__all__ = [
    "split_tables", "demand_by_time", "demand_by_line", "demand_by_link",
    "peak_vs_offpeak", "frequency_by_time", "demand_frequency_relationship",
    "high_load_links", "baseline_utilization", "underserved_periods",
    "underutilized_periods", "utilization_summary", "time_order",
]
