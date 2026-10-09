"""Unit tests for the routing engine (run offline on a synthetic grid).

Run from the project root:  python -m unittest tests.test_routing -v
"""
from __future__ import annotations

import unittest

import networkx as nx

from routing.algorithms import NoRouteError, astar, compare_algorithms, dijkstra
from routing.cost import apply_costs, edge_cost
from routing.graph_builder import attach_static_features, build_grid_graph
from routing.risk import assign_flood_risk, load_model_bundle

ROWS, COLS = 15, 15
SOURCE, TARGET = 0, ROWS * COLS - 1


def make_graph(rainfall: float = 200.0, river: float = 4.0) -> nx.DiGraph:
    """Grid graph with static features and ML flood risk attached."""
    graph = build_grid_graph(ROWS, COLS)
    attach_static_features(graph)
    assign_flood_risk(graph, rainfall, river, load_model_bundle())
    return graph


class RoutingTests(unittest.TestCase):
    """Correctness tests for costs, search algorithms and risk behaviour."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = make_graph()

    def test_cost_formulas(self) -> None:
        self.assertEqual(edge_cost(100, 0.5, 10, "additive"), 105.0)
        self.assertEqual(edge_cost(100, 0.5, 10, "proportional"), 600.0)
        self.assertEqual(edge_cost(100, 0.9, 10, block_threshold=0.8), float("inf"))
        with self.assertRaises(ValueError):
            edge_cost(100, 1.5, 1)

    def test_risk_in_unit_interval(self) -> None:
        probs = [d["flood_prob"] for _, _, d in self.graph.edges(data=True)]
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_dijkstra_and_astar_agree(self) -> None:
        for weight in (0.0, 5.0, 20.0):
            apply_costs(self.graph, weight)
            d = dijkstra(self.graph, SOURCE, TARGET)
            a = astar(self.graph, SOURCE, TARGET)
            self.assertAlmostEqual(d.cost, a.cost, places=6)

    def test_astar_expands_no_more_nodes(self) -> None:
        apply_costs(self.graph, 5.0)
        d, a = compare_algorithms(self.graph, SOURCE, TARGET, repeats=3)
        self.assertLessEqual(a.nodes_visited, d.nodes_visited)

    def test_higher_weight_reduces_risk_exposure(self) -> None:
        apply_costs(self.graph, 0.0)
        base = dijkstra(self.graph, SOURCE, TARGET)
        apply_costs(self.graph, 50.0)
        safe = dijkstra(self.graph, SOURCE, TARGET)
        self.assertLessEqual(safe.mean_risk * safe.distance_m,
                             base.mean_risk * base.distance_m + 1e-6)
        self.assertGreaterEqual(safe.distance_m, base.distance_m - 1e-6)

    def test_blocking_everything_raises(self) -> None:
        apply_costs(self.graph, 5.0, block_threshold=0.0)
        with self.assertRaises(NoRouteError):
            astar(self.graph, SOURCE, TARGET)

    def test_unknown_node(self) -> None:
        apply_costs(self.graph, 5.0)
        with self.assertRaises(KeyError):
            dijkstra(self.graph, SOURCE, 10_000)

    def test_same_source_and_target(self) -> None:
        apply_costs(self.graph, 5.0)
        result = astar(self.graph, 7, 7)
        self.assertEqual(result.path, [7])
        self.assertEqual(result.distance_m, 0.0)


if __name__ == "__main__":
    unittest.main()
