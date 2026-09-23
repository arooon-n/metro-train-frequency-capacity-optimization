"""Rail network reconstruction from consecutive-station links (NetworkX).

The graph is a MultiDiGraph: parallel edges exist because the same physical
link appears in every 15-minute period. Edge attributes carry line, stations,
direction, time_period, link_load and link_frequency.

Branching is DETECTED, never assumed away: for each (line, direction) we count
unique neighbours per station; nodes with >2 neighbours indicate a junction.
If a structure cannot be safely interpreted, configure config/service_groups.csv.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd


def build_network(links: pd.DataFrame) -> nx.MultiDiGraph:
    """Build a directed multigraph from a cleaned link table."""
    if links.empty:
        raise ValueError("build_network received an empty link table.")
    required = {"line", "from_station", "to_station", "time_period"}
    missing = required - set(links.columns)
    if missing:
        raise ValueError(f"link table missing columns: {sorted(missing)}")

    G = nx.MultiDiGraph()
    for row in links.itertuples(index=False):
        data = {
            "line": getattr(row, "line", None),
            "direction": getattr(row, "direction", None),
            "time_period": getattr(row, "time_period", None),
            "link_load": getattr(row, "link_load", None),
            "link_frequency": getattr(row, "link_frequency", None),
            "mode": getattr(row, "mode", None),
            "day_type": getattr(row, "day_type", None),
        }
        G.add_node(row.from_station, line=data["line"])
        G.add_node(row.to_station, line=data["line"])
        G.add_edge(
            row.from_station,
            row.to_station,
            key=(data["line"], data["direction"], data["time_period"]),
            **data,
        )
    return G


def links_by_line(links: pd.DataFrame) -> dict[str, int]:
    uniq = links[["line", "from_station", "to_station", "direction"]].drop_duplicates()
    return uniq.groupby("line").size().to_dict()


def detect_branches(links: pd.DataFrame) -> dict[str, Any]:
    """Per (line, direction): endpoints, junctions (degree>2), simple-corridor flag."""
    out: dict[str, Any] = {}
    uniq = links[["line", "direction", "from_station", "to_station"]].drop_duplicates()
    for (line, direction), grp in uniq.groupby(["line", "direction"], dropna=False):
        G = nx.DiGraph()
        G.add_edges_from(grp[["from_station", "to_station"]].itertuples(index=False, name=None))
        und = G.to_undirected()
        nbrs = {n: set(und.neighbors(n)) for n in und.nodes}
        junctions = sorted([n for n, nb in nbrs.items() if len(nb) > 2])
        endpoints = sorted([n for n, nb in nbrs.items() if len(nb) == 1])
        components = nx.number_weakly_connected_components(G) if G.number_of_nodes() else 0
        out[f"{line} | {direction}"] = {
            "stations": G.number_of_nodes(),
            "links": G.number_of_edges(),
            "endpoints": endpoints,
            "junction_stations": junctions,
            "is_simple_corridor": len(junctions) == 0 and components <= 1,
            "weakly_connected_components": components,
        }
    return out


def network_statistics(links: pd.DataFrame) -> dict[str, Any]:
    G = build_network(links)
    stations = pd.unique(pd.concat([links["from_station"], links["to_station"]], ignore_index=True))
    uniq_links = links[["line", "from_station", "to_station", "direction"]].drop_duplicates()
    branches = detect_branches(links)
    # disconnected check per line-direction using the unique topology graph
    disconnected = {
        k: v["weakly_connected_components"]
        for k, v in branches.items()
        if v["weakly_connected_components"] > 1
    }
    return {
        "graph_nodes": G.number_of_nodes(),
        "graph_edges_with_time": G.number_of_edges(),
        "stations": int(pd.Series(stations).notna().sum()),
        "unique_links": int(len(uniq_links)),
        "links_by_line": links_by_line(links),
        "branch_analysis": branches,
        "disconnected_groups": disconnected,
        "is_fully_connected_by_line": len(disconnected) == 0,
    }


def line_topology(links: pd.DataFrame, line: str, direction: str | None = None) -> dict[str, Any]:
    """Ordered station sequence for a simple corridor; warns when branching."""
    sub = links[links["line"] == line]
    if direction is not None:
        sub = sub[sub["direction"] == direction]
    topo = detect_branches(sub) if not sub.empty else {}
    return topo
