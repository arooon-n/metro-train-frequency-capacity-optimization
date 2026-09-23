"""Network construction tests (TEST DATA topology)."""

from __future__ import annotations

import pandas as pd

from src.analysis import eda
from src.network.builder import (
    build_network,
    detect_branches,
    links_by_line,
    network_statistics,
)


def test_build_network_counts(model_inputs):
    links, _ = model_inputs
    G = build_network(links)
    # TestLineA: Alpha, Beta, Gamma ; TestLineB: Delta, Epsilon
    assert set(G.nodes) == {"Alpha", "Beta", "Gamma", "Delta", "Epsilon"}
    # edges are per time period (MultiDiGraph): A has 2 segments x 3 periods = 6
    # plus B 1 segment x 3 periods = 3
    assert G.number_of_edges() == 9


def test_links_by_line(model_inputs):
    links, _ = model_inputs
    counts = links_by_line(links)
    uniq = links[["line", "from_station", "to_station", "direction"]].drop_duplicates()
    assert counts["TestLineA"] == 2
    assert counts["TestLineB"] == 1
    assert sum(counts.values()) == len(uniq)


def test_simple_corridor_no_branches(model_inputs):
    links, _ = model_inputs
    br = detect_branches(links)
    a = br["TestLineA | Northbound"]
    assert a["is_simple_corridor"]
    assert a["junction_stations"] == []
    assert a["endpoints"] == ["Alpha", "Gamma"]


def test_branch_detection_on_y_topology():
    rows = []
    # trunk M1->M2, branches M2->B1 and M2->B2
    edges = [("M1", "M2"), ("M2", "B1"), ("M2", "B2")]
    for i, (f, t) in enumerate(edges):
        rows.append({
            "line": "Branchy", "direction": "N", "from_station": f, "to_station": t,
            "time_period": "07:00-07:15", "link_load": 10, "link_frequency": 4,
            "mode": "rail", "day_type": "TWT",
        })
    df = pd.DataFrame(rows)
    br = detect_branches(df)
    key = "Branchy | N"
    assert br[key]["junction_stations"] == ["M2"]
    assert not br[key]["is_simple_corridor"]


def test_disconnected_network_flag():
    rows = []
    for f, t in [("A", "B"), ("C", "D")]:
        rows.append({
            "line": "Broken", "direction": "N", "from_station": f, "to_station": t,
            "time_period": "07:00-07:15", "link_load": 5, "link_frequency": 2,
            "mode": "rail", "day_type": "TWT",
        })
    stats = network_statistics(pd.DataFrame(rows))
    assert not stats["is_fully_connected_by_line"]
    assert stats["disconnected_groups"]


def test_network_statistics_from_model_inputs(model_inputs):
    links, _ = model_inputs
    stats = network_statistics(links)
    assert stats["stations"] == 5
    assert stats["unique_links"] == 3
    assert stats["is_fully_connected_by_line"]
    assert set(stats["links_by_line"]) == {"TestLineA", "TestLineB"}
