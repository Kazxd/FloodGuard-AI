"""Property-based tests for Dijkstra and A* (Phase 6).

These tests build their own small graphs, so they do not depend on the OSM
download or on the trained model. They check *properties* that must hold for
any correct search implementation, not just one hand-picked example.

Run from the project root:
    python -m unittest tests.test_search_properties -v
"""
from __future__ import annotations

import math
import random
import unittest
from typing import Callable, Tuple

import networkx as nx

from routing.algorithms import (NoRouteError, SearchResult, astar,
                                compare_algorithms, dijkstra)
from routing.graph_builder import haversine_m

# Spacing between neighbouring grid nodes in degrees (about 111 m at the equator).
GRID_STEP_DEG = 0.001
BASE_LAT, BASE_LON = 13.0, 80.0

RiskFn = Callable[[int, int], float]


def add_road(graph: nx.DiGraph, a: int, b: int, risk: float, weight: float) -> None:
    """Add a two-way road whose cost is length + weight * risk.

    The length is the great-circle distance between the end nodes, so the
    A* heuristic is exactly admissible and consistent on these graphs.
    """
    length = haversine_m(graph.nodes[a]["y"], graph.nodes[a]["x"],
                         graph.nodes[b]["y"], graph.nodes[b]["x"])
    for u, v in ((a, b), (b, a)):
        graph.add_edge(u, v, length=length, flood_prob=risk,
                       cost=length + weight * risk)


def build_grid(size: int, risk_fn: RiskFn, weight: float = 500.0) -> nx.DiGraph:
    """Build a size x size grid; node id = row * size + col."""
    graph = nx.DiGraph()
    for row in range(size):
        for col in range(size):
            graph.add_node(row * size + col,
                           y=BASE_LAT + row * GRID_STEP_DEG,
                           x=BASE_LON + col * GRID_STEP_DEG)
    for row in range(size):
        for col in range(size):
            node = row * size + col
            if col + 1 < size:
                add_road(graph, node, node + 1, risk_fn(node, node + 1), weight)
            if row + 1 < size:
                add_road(graph, node, node + size, risk_fn(node, node + size), weight)
    return graph


def build_two_route_graph(weight: float) -> Tuple[nx.DiGraph, int, int]:
    """Short risky route S-A-T versus a longer, completely safe route S-B-T."""
    graph = nx.DiGraph()
    coords = {0: (13.0000, 80.0000),   # S
              1: (13.0000, 80.0050),   # A (on the straight line: short, risky)
              2: (13.0050, 80.0050),   # B (detour: long, safe)
              3: (13.0000, 80.0100)}   # T
    for node, (lat, lon) in coords.items():
        graph.add_node(node, y=lat, x=lon)
    add_road(graph, 0, 1, 0.9, weight)
    add_road(graph, 1, 3, 0.9, weight)
    add_road(graph, 0, 2, 0.0, weight)
    add_road(graph, 2, 3, 0.0, weight)
    return graph, 0, 3


class TestSearchCorrectness(unittest.TestCase):
    """Dijkstra and A* must agree on the optimal cost."""

    def test_both_algorithms_find_same_optimal_cost(self) -> None:
        """A* is optimal only if its heuristic is admissible: costs must match."""
        rng = random.Random(7)
        risks = {}

        def risk_fn(a: int, b: int) -> float:
            return risks.setdefault((min(a, b), max(a, b)), rng.random())

        graph = build_grid(8, risk_fn)
        nodes = list(graph.nodes)
        for _ in range(40):
            src, dst = rng.sample(nodes, 2)
            d, a = dijkstra(graph, src, dst), astar(graph, src, dst)
            self.assertAlmostEqual(d.cost, a.cost, places=6,
                                   msg=f"cost mismatch for {src}->{dst}")

    def test_astar_visits_fewer_nodes_in_open_grid(self) -> None:
        """Goal-directed search should expand a narrow corridor, not a diamond."""
        graph = build_grid(9, lambda a, b: 0.0)
        src, dst = 4 * 9 + 0, 4 * 9 + 8   # middle-left to middle-right
        d, a = dijkstra(graph, src, dst), astar(graph, src, dst)
        self.assertLess(a.nodes_visited, d.nodes_visited)

    def test_path_is_connected_and_metrics_consistent(self) -> None:
        """Every hop must be a real edge; reported distance = sum of lengths."""
        graph = build_grid(6, lambda a, b: 0.3)
        result = astar(graph, 0, 35)
        self.assertEqual(result.path[0], 0)
        self.assertEqual(result.path[-1], 35)
        for u, v in zip(result.path, result.path[1:]):
            self.assertTrue(graph.has_edge(u, v))
        expected = sum(graph[u][v]["length"] for u, v in zip(result.path, result.path[1:]))
        self.assertAlmostEqual(result.distance_m, expected, places=6)
        self.assertAlmostEqual(result.mean_risk, 0.3, places=6)
        self.assertGreaterEqual(result.runtime_s, 0.0)


class TestEdgeCases(unittest.TestCase):
    """Unusual inputs must produce clear, predictable behaviour."""

    def test_source_equals_target(self) -> None:
        graph = build_grid(4, lambda a, b: 0.5)
        for fn in (dijkstra, astar):
            result = fn(graph, 5, 5)
            self.assertEqual(result.path, [5])
            self.assertEqual(result.cost, 0.0)
            self.assertEqual(result.distance_m, 0.0)
            self.assertEqual(result.nodes_visited, 1)

    def test_unknown_node_raises_key_error(self) -> None:
        graph = build_grid(3, lambda a, b: 0.0)
        with self.assertRaises(KeyError):
            dijkstra(graph, 0, 999)
        with self.assertRaises(KeyError):
            astar(graph, 999, 0)

    def test_disconnected_target_raises_no_route(self) -> None:
        graph = build_grid(3, lambda a, b: 0.0)
        graph.add_node(100, y=BASE_LAT + 1.0, x=BASE_LON + 1.0)  # island
        for fn in (dijkstra, astar):
            with self.assertRaises(NoRouteError):
                fn(graph, 0, 100)

    def test_blocked_roads_force_no_route(self) -> None:
        """Infinite cost = impassable. Blocking a whole column cuts the grid."""
        graph = build_grid(3, lambda a, b: 0.0)
        for u, v in list(graph.edges):
            if {u % 3, v % 3} == {0, 1}:      # every edge between column 0 and 1
                graph[u][v]["cost"] = math.inf
        for fn in (dijkstra, astar):
            with self.assertRaises(NoRouteError):
                fn(graph, 0, 2)

    def test_search_avoids_single_blocked_road(self) -> None:
        graph, src, dst = build_two_route_graph(weight=0.0)
        graph[0][1]["cost"] = math.inf   # block the short route
        result = astar(graph, src, dst)
        self.assertEqual(result.path, [0, 2, 3])


class TestFloodWeightBehaviour(unittest.TestCase):
    """The flood weight must trade distance against risk as designed."""

    def test_zero_weight_prefers_shortest_route(self) -> None:
        graph, src, dst = build_two_route_graph(weight=0.0)
        self.assertEqual(astar(graph, src, dst).path, [0, 1, 3])

    def test_large_weight_prefers_safest_route(self) -> None:
        graph, src, dst = build_two_route_graph(weight=1000.0)
        self.assertEqual(astar(graph, src, dst).path, [0, 2, 3])

    def test_risk_never_increases_as_weight_grows(self) -> None:
        """Mean risk of the chosen route is non-increasing in the weight."""
        previous = math.inf
        for weight in (0.0, 10.0, 100.0, 1000.0, 10000.0):
            graph, src, dst = build_two_route_graph(weight)
            risk = astar(graph, src, dst).mean_risk
            self.assertLessEqual(risk, previous + 1e-12)
            previous = risk


class TestCompareAlgorithms(unittest.TestCase):
    """The benchmark helper used by the Analytics page."""

    def test_returns_both_results_in_order(self) -> None:
        graph = build_grid(5, lambda a, b: 0.1)
        results = compare_algorithms(graph, 0, 24, repeats=3)
        self.assertEqual([r.algorithm for r in results], ["Dijkstra", "A*"])
        self.assertTrue(all(isinstance(r, SearchResult) for r in results))
        self.assertAlmostEqual(results[0].cost, results[1].cost, places=6)

    def test_invalid_repeats_rejected(self) -> None:
        graph = build_grid(3, lambda a, b: 0.0)
        with self.assertRaises(ValueError):
            compare_algorithms(graph, 0, 8, repeats=0)


if __name__ == "__main__":
    unittest.main()