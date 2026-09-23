"""Plotly figure builders. All figures are pure functions of the input tables,
so every chart on the dashboard can be reproduced from saved CSV results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.analysis.eda import time_order

PALETTE = px.colors.qualitative.Set2


def _order_time(df: pd.DataFrame, col: str = "time_period") -> pd.DataFrame:
    order = time_order(df[col].dropna().unique())
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df.sort_values(col)


def demand_heatmap(links: pd.DataFrame) -> go.Figure:
    piv = links.pivot_table(
        index="line", columns="time_period", values="link_load",
        aggfunc="sum", observed=False,
    )
    piv = piv.reindex(columns=time_order(piv.columns))
    fig = go.Figure(
        data=go.Heatmap(
            z=piv.values,
            x=[str(c) for c in piv.columns],
            y=[str(i) for i in piv.index],
            colorscale="YlOrRd",
            colorbar=dict(title="Link load sum"),
            hovertemplate="line=%{y}<br>period=%{x}<br>load=%{z:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Demand heatmap (summed link load by line and 15-min period)",
        xaxis_title="15-minute period", yaxis_title="Line",
        height=420, margin=dict(l=60, r=20, t=60, b=80),
    )
    return fig


def demand_timeseries(links: pd.DataFrame) -> go.Figure:
    df = links.groupby("time_period", as_index=False)["link_load"].sum()
    df = _order_time(df)
    fig = go.Figure(go.Scatter(
        x=df["time_period"].astype(str), y=df["link_load"], mode="lines+markers",
        name="Link load sum", line=dict(color=PALETTE[0]),
    ))
    fig.update_layout(
        title="Demand by 15-minute period (summed link loads — not unique journeys)",
        xaxis_title="15-minute period", yaxis_title="Link load",
        height=400, template="plotly_white",
    )
    return fig


def line_boarders_timeseries(line_df: pd.DataFrame) -> go.Figure | None:
    if line_df.empty:
        return None
    df = line_df.groupby("time_period", as_index=False)["line_boarders"].sum()
    df = _order_time(df)
    fig = go.Figure(go.Scatter(
        x=df["time_period"].astype(str), y=df["line_boarders"], mode="lines+markers",
        name="Line boarders", line=dict(color=PALETTE[1]),
    ))
    fig.update_layout(
        title="Line boardings by 15-minute period",
        xaxis_title="15-minute period", yaxis_title="Line boarders",
        height=400, template="plotly_white",
    )
    return fig


def frequency_timeseries(links: pd.DataFrame) -> go.Figure:
    df = links.groupby("time_period", as_index=False).agg(
        mean_frequency=("link_frequency", "mean"),
        min_frequency=("link_frequency", "min"),
        max_frequency=("link_frequency", "max"),
    )
    df = _order_time(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time_period"].astype(str), y=df["max_frequency"],
        mode="lines", name="max", line=dict(dash="dot", color=PALETTE[2]),
    ))
    fig.add_trace(go.Scatter(
        x=df["time_period"].astype(str), y=df["mean_frequency"],
        mode="lines+markers", name="mean", line=dict(color=PALETTE[2]),
    ))
    fig.add_trace(go.Scatter(
        x=df["time_period"].astype(str), y=df["min_frequency"],
        mode="lines", name="min", line=dict(dash="dot", color=PALETTE[2]),
        fill="tonexty", fillcolor="rgba(102,194,165,0.15)",
    ))
    fig.update_layout(
        title="Scheduled (planned) link frequency by period",
        xaxis_title="15-minute period", yaxis_title="Trains / 15 min",
        height=400, template="plotly_white",
    )
    return fig


def demand_frequency_scatter(links: pd.DataFrame) -> go.Figure:
    df = links.dropna(subset=["link_frequency"]).copy()
    fig = px.scatter(
        df, x="link_frequency", y="link_load", color="line",
        hover_data=["time_period", "from_station", "to_station"],
        title="Demand vs scheduled frequency (one point = one link-period)",
        labels={
            "link_frequency": "Scheduled frequency (trains / 15 min)",
            "link_load": "Link load (passengers / period)",
        },
    )
    fig.update_layout(height=450, template="plotly_white")
    return fig


def line_comparison(links: pd.DataFrame) -> go.Figure:
    df = links.groupby("line", as_index=False).agg(
        link_load_sum=("link_load", "sum"),
        peak_link_load=("link_load", "max"),
        mean_frequency=("link_frequency", "mean"),
    ).sort_values("link_load_sum", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["line"], x=df["link_load_sum"], name="Summed link load",
        orientation="h", marker_color=PALETTE[3],
    ))
    fig.update_layout(
        title="Line comparison — demand (summed link load)",
        xaxis_title="Link load sum", yaxis_title="Line",
        height=max(350, 28 * max(len(df), 1)), template="plotly_white",
    )
    return fig


def top_demand_links(links: pd.DataFrame, top_n: int = 20) -> go.Figure:
    df = links.groupby(["line", "from_station", "to_station"], as_index=False).agg(
        peak_link_load=("link_load", "max")
    ).sort_values("peak_link_load", ascending=False).head(top_n)
    df["link_label"] = df["from_station"] + " → " + df["to_station"] + " (" + df["line"] + ")"
    df = df.sort_values("peak_link_load")
    fig = go.Figure(go.Bar(
        y=df["link_label"], x=df["peak_link_load"], orientation="h",
        marker_color=PALETTE[4],
    ))
    fig.update_layout(
        title=f"Top {len(df)} links by peak load",
        xaxis_title="Peak link load", height=max(380, 26 * len(df)),
        template="plotly_white",
    )
    return fig


def utilization_heatmap(util_df: pd.DataFrame, value_col: str = "baseline_utilization") -> go.Figure | None:
    if value_col not in util_df.columns or util_df[value_col].dropna().empty:
        return None
    piv = util_df.pivot_table(
        index="line", columns="time_period", values=value_col, aggfunc="max",
        observed=False,
    )
    piv = piv.reindex(columns=time_order(piv.columns))
    fig = go.Figure(data=go.Heatmap(
        z=piv.values,
        x=[str(c) for c in piv.columns],
        y=[str(i) for i in piv.index],
        colorscale="RdYlGn_r", zmid=1.0,
        colorbar=dict(title="utilisation"),
        hovertemplate="line=%{y}<br>period=%{x}<br>u=%{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Capacity utilisation heatmap (max link utilisation per line-period)",
        xaxis_title="15-minute period", yaxis_title="Line",
        height=420, margin=dict(l=60, r=20, t=60, b=80),
    )
    return fig


def baseline_vs_optimized(results: pd.DataFrame, xcol: str = "optimized_frequency") -> go.Figure | None:
    if results.empty or xcol not in results.columns:
        return None
    df = results.copy()
    obs_col = "observed_frequency" if "observed_frequency" in df.columns else None
    if obs_col is None:
        return None
    agg = df.groupby("time_period", as_index=False).agg(
        baseline=(obs_col, "sum"), optimized=(xcol, "sum")
    )
    agg = _order_time(agg)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg["time_period"].astype(str), y=agg["baseline"],
        name="Baseline (NUMBAT scheduled)", marker_color="#9ecae1",
    ))
    fig.add_trace(go.Bar(
        x=agg["time_period"].astype(str), y=agg["optimized"],
        name="Optimized (model recommendation)", marker_color="#3182bd",
    ))
    fig.update_layout(
        barmode="group",
        title="Baseline vs optimised train-equivalent service by period",
        xaxis_title="15-minute period", yaxis_title="Trains / 15 min (sum over groups)",
        height=430, template="plotly_white",
    )
    return fig


def frequency_change_by_group(results: pd.DataFrame, xcol: str = "optimized_frequency") -> go.Figure | None:
    if results.empty or "observed_frequency" not in results.columns or xcol not in results.columns:
        return None
    df = results.copy()
    df["delta"] = pd.to_numeric(df[xcol], errors="coerce") - pd.to_numeric(
        df["observed_frequency"], errors="coerce"
    )
    agg = df.groupby("service_group", as_index=False)["delta"].sum().sort_values("delta")
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in agg["delta"]]
    fig = go.Figure(go.Bar(x=agg["delta"], y=agg["service_group"], orientation="h",
                           marker_color=colors))
    fig.update_layout(
        title="Change in train-equivalents vs baseline (by service group, all periods)",
        xaxis_title="Δ trains (model recommendation − baseline)",
        yaxis_title="Service group", height=max(350, 30 * len(agg)),
        template="plotly_white",
    )
    return fig


def network_graph(G, line: str | None = None) -> go.Figure | None:
    """Optional static network view (spring layout, first time period only)."""
    try:
        import networkx as nx
    except Exception:
        return None
    if G.number_of_nodes() == 0:
        return None
    pos = nx.spring_layout(G, seed=42, k=1.2 / max(G.number_of_nodes(), 1) ** 0.5)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(color="#888", width=1), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                             text=[str(n) for n in G.nodes()],
                             textposition="top center", marker=dict(size=10, color="#1f77b4"),
                             hovertext=[str(n) for n in G.nodes()]))
    title = "Reconstructed network (topology)"
    if line:
        title += f" — filter: {line}"
    fig.update_layout(title=title, showlegend=False, height=520,
                      template="plotly_white",
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    return fig


def save_figure(fig: go.Figure | None, path: Path | str, width: int = 1200, height: int = 450) -> Path | None:
    if fig is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path), width=width, height=height, scale=1)
    return path


def save_figure_html(fig: go.Figure | None, path: Path | str) -> Path | None:
    """HTML export works without the kaleido dependency."""
    if fig is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path
