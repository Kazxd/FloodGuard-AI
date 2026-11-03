"""E6: Dijkstra vs A* on the real Velachery graph (re-run of manuscript Section V-D).

Design (the original pair-selection rule is not recorded, so this one is declared here):
heavy scenario (200 mm, 4 m), proportional cost, w = 5; 100 pairs = 100 random
non-shelter starts (seed 2026), each paired with one shelter drawn uniformly (same RNG).
Runtime = median of 10 repeats per algorithm (repo's compare_algorithms).
Also: wall time of one full best-shelter query (A*, 5 shelters) for each w.
"""
from __future__ import annotations

import json
import platform
import statistics
import time

import numpy as np

from common import OUT, REPO, SCENARIOS, build_graph, edge_table, score_rf
from routing.algorithms import compare_algorithms
from routing.cost import apply_costs
from routing.risk import load_model_bundle
from routing.shelters import find_shelters, route_to_best_shelter


def main():
    multi, g = build_graph()
    E = edge_table(multi, g)
    score_rf(g, E, *SCENARIOS["heavy"], load_model_bundle(REPO / "models/flood_model.joblib"))
    shelters = find_shelters(g)
    rng = np.random.default_rng(2026)
    nodes = [n for n in g.nodes if n not in shelters]
    starts = rng.choice(nodes, 100, replace=False)
    targets = rng.choice(shelters, 100, replace=True)
    apply_costs(g, 5.0)
    rows = []
    for s, t in zip(starts, targets):
        d, a = compare_algorithms(g, int(s), int(t), repeats=10)
        rows.append({"cost_equal": abs(d.cost - a.cost) <= 1e-6 * max(1.0, d.cost),
                     "nodes_d": d.nodes_visited, "nodes_a": a.nodes_visited,
                     "ms_d": 1000 * d.runtime_s, "ms_a": 1000 * a.runtime_s})
    nd = np.array([r["nodes_d"] for r in rows]); na = np.array([r["nodes_a"] for r in rows])
    res = {"pairs": len(rows), "all_costs_equal": all(r["cost_equal"] for r in rows),
           "mean_nodes_dijkstra": float(nd.mean()), "mean_nodes_astar": float(na.mean()),
           "node_reduction_pct": float(100 * (1 - na.sum() / nd.sum())),
           "median_ms_dijkstra": float(statistics.median(r["ms_d"] for r in rows)),
           "median_ms_astar": float(statistics.median(r["ms_a"] for r in rows)),
           "platform": f"{platform.python_version()} / {platform.processor() or platform.machine()}"}
    q = {}
    for w in (0, 1, 2, 5, 10, 20):
        apply_costs(g, float(w))
        times = []
        for s in starts[:50]:
            t0 = time.perf_counter()
            route_to_best_shelter(g, int(s), shelters)
            times.append(1000 * (time.perf_counter() - t0))
        q[w] = float(statistics.median(times))
    res["median_full_query_ms_by_w"] = q
    json.dump(res, open(OUT / "astar_bench.json", "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
