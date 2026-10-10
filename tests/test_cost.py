"""Tests for routing/cost.py: the flood-aware edge cost (Phase 6).

The cost function connects the ML output (flood probability) to the search
algorithms, and the A* correctness proof depends on it, so it is tested both
with exact examples and with mathematical properties.

Run from the project root:
    python -m unittest tests.test_cost -v
"""
from __future__ import annotations

import math
import random
import unittest

from routing.algorithms import NoRouteError, astar, dijkstra
from routing.cost import COST_MODES, apply_costs, edge_cost
from routing.graph_builder import build_grid_graph


def risky_grid(seed: int = 3, size: int = 8):
    """Grid whose two-way roads carry random (but symmetric) flood risk."""
    rng = random.Random(seed)
    graph = build_grid_graph(size, size)
    risk = {}
    for u, v, data in graph.edges(data=True):
        key = (min(u, v), max(u, v))
        data["flood_prob"] = risk.setdefault(key, rng.random())
    return graph


def exposure(graph, path) -> float:
    """Length-weighted total flood risk along a path (sum of length * risk)."""
    return sum(graph[a][b]["length"] * graph[a][b]["flood_prob"]
               for a, b in zip(path, path[1:]))


class TestEdgeCostFormulas(unittest.TestCase):
    """Exact values for both cost formulations."""

    def test_additive_formula_matches_project_spec(self) -> None:
        # Cost = Distance + Flood Risk x Weight
        self.assertAlmostEqual(edge_cost(100.0, 0.5, 5.0, mode="additive"), 102.5)

    def test_proportional_formula(self) -> None:
        # Cost = Distance x (1 + Weight x Risk)
        self.assertAlmostEqual(edge_cost(100.0, 0.5, 5.0, mode="proportional"), 350.0)

    def test_default_mode_is_proportional(self) -> None:
        self.assertEqual(edge_cost(100.0, 0.5, 5.0),
                         edge_cost(100.0, 0.5, 5.0, mode="proportional"))

    def test_zero_weight_gives_plain_distance_in_both_modes(self) -> None:
        for mode in COST_MODES:
            self.assertEqual(edge_cost(137.0, 0.9, 0.0, mode=mode), 137.0)

    def test_zero_risk_gives_plain_distance_in_both_modes(self) -> None:
        for mode in COST_MODES:
            self.assertEqual(edge_cost(137.0, 0.0, 10.0, mode=mode), 137.0)


class TestEdgeCostProperties(unittest.TestCase):
    """Properties the search algorithms rely on."""

    def test_cost_is_never_below_length(self) -> None:
        """Needed so the straight-line heuristic never overestimates."""
        rng = random.Random(1)
        for _ in range(500):
            length, risk, weight = rng.uniform(0, 500), rng.random(), rng.uniform(0, 25)
            for mode in COST_MODES:
                self.assertGreaterEqual(edge_cost(length, risk, weight, mode), length)

    def test_cost_does_not_decrease_with_risk_or_weight(self) -> None:
        for mode in COST_MODES:
            risks = [edge_cost(100.0, r / 10, 5.0, mode) for r in range(11)]
            weights = [edge_cost(100.0, 0.5, w, mode) for w in range(0, 26, 5)]
            self.assertEqual(risks, sorted(risks))
            self.assertEqual(weights, sorted(weights))

    def test_blocking_threshold(self) -> None:
        self.assertTrue(math.isinf(edge_cost(100.0, 0.8, 5.0, block_threshold=0.7)))
        self.assertTrue(math.isinf(edge_cost(100.0, 0.7, 5.0, block_threshold=0.7)))
        self.assertTrue(math.isfinite(edge_cost(100.0, 0.69, 5.0, block_threshold=0.7)))
        self.assertTrue(math.isfinite(edge_cost(100.0, 1.0, 5.0, block_threshold=None)))


class TestEdgeCostValidation(unittest.TestCase):
    """Bad inputs must fail loudly, not silently produce odd routes."""

    def test_invalid_arguments_raise(self) -> None:
        bad_calls = [
            dict(length_m=100.0, risk=0.5, weight=1.0, mode="quadratic"),
            dict(length_m=100.0, risk=0.5, weight=-1.0),
            dict(length_m=-5.0, risk=0.5, weight=1.0),
            dict(length_m=100.0, risk=1.5, weight=1.0),
            dict(length_m=100.0, risk=-0.1, weight=1.0),
        ]
        for kwargs in bad_calls:
            with self.assertRaises(ValueError, msg=str(kwargs)):
                edge_cost(**kwargs)


class TestApplyCosts(unittest.TestCase):
    """Writing costs onto a whole graph."""

    def test_every_edge_receives_a_cost(self) -> None:
        graph = risky_grid()
        apply_costs(graph, 5.0)
        for _, _, data in graph.edges(data=True):
            self.assertAlmostEqual(
                data["cost"], edge_cost(data["length"], data["flood_prob"], 5.0))

    def test_zero_weight_cost_equals_length(self) -> None:
        graph = risky_grid()
        apply_costs(graph, 0.0)
        for _, _, data in graph.edges(data=True):
            self.assertAlmostEqual(data["cost"], data["length"])

    def test_missing_flood_probability_raises(self) -> None:
        graph = build_grid_graph(3, 3)
        with self.assertRaises(ValueError):
            apply_costs(graph, 5.0)

    def test_mode_and_blocking_are_passed_through(self) -> None:
        graph = risky_grid()
        apply_costs(graph, 5.0, mode="additive", block_threshold=0.9)
        for _, _, data in graph.edges(data=True):
            if data["flood_prob"] >= 0.9:
                self.assertTrue(math.isinf(data["cost"]))
            else:
                self.assertAlmostEqual(data["cost"], data["length"] + 5.0 * data["flood_prob"])


class TestCostWithSearch(unittest.TestCase):
    """The cost function combined with Dijkstra and A*."""

    def test_astar_matches_dijkstra_for_every_mode_and_weight(self) -> None:
        graph = risky_grid(seed=11, size=9)
        nodes = list(graph.nodes)
        rng = random.Random(5)
        for mode in COST_MODES:
            for weight in (0.0, 1.0, 5.0, 25.0):
                apply_costs(graph, weight, mode=mode)
                for _ in range(10):
                    src, dst = rng.sample(nodes, 2)
                    self.assertAlmostEqual(dijkstra(graph, src, dst).cost,
                                           astar(graph, src, dst).cost, places=6)

    def test_zero_weight_route_is_the_shortest_distance_route(self) -> None:
        graph = risky_grid()
        apply_costs(graph, 0.0)
        result = astar(graph, 0, 63)
        self.assertAlmostEqual(result.cost, result.distance_m, places=6)

    def test_route_exposure_never_increases_as_weight_grows(self) -> None:
        """Parametric shortest paths: more weight on risk => no more exposure."""
        graph = risky_grid(seed=21, size=9)
        previous = math.inf
        for weight in (0.0, 1.0, 5.0, 25.0, 100.0, 1000.0):
            apply_costs(graph, weight)           # proportional: cost = L + w * exposure
            path = astar(graph, 0, 80).path
            current = exposure(graph, path)
            self.assertLessEqual(current, previous + 1e-9)
            previous = current

    def test_blocked_roads_are_never_used(self) -> None:
        """Roads at or above the threshold are impassable (a pair may be cut off)."""
        graph = risky_grid(seed=2, size=9)
        apply_costs(graph, 5.0, block_threshold=0.9)
        nodes = list(graph.nodes)
        rng = random.Random(8)
        routes_found = 0
        for _ in range(15):
            src, dst = rng.sample(nodes, 2)
            try:
                result = astar(graph, src, dst)
            except NoRouteError:
                continue                      # blocking can disconnect a pair
            routes_found += 1
            for a, b in zip(result.path, result.path[1:]):
                self.assertLess(graph[a][b]["flood_prob"], 0.9)
        self.assertGreaterEqual(routes_found, 5)


if __name__ == "__main__":
    unittest.main()