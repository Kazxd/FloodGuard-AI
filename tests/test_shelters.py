"""Tests for routing/shelters.py (Phase 6).

Run from the project root:
    python -m unittest tests.test_shelters -v
"""
from __future__ import annotations

import unittest

import networkx as nx

from routing.algorithms import NoRouteError, dijkstra
from routing.graph_builder import (attach_static_features, build_grid_graph,
                                   haversine_m)
from routing.shelters import find_shelters, node_elevation, route_to_best_shelter


def make_costed_grid(rows: int = 8, cols: int = 8) -> nx.DiGraph:
    """Grid with static features and cost = length (no flood penalty)."""
    graph = build_grid_graph(rows, cols)
    attach_static_features(graph)
    for _, _, data in graph.edges(data=True):
        data["flood_prob"] = 0.0
        data["cost"] = data["length"]
    return graph


class TestNodeElevation(unittest.TestCase):
    """Node elevation is the mean of its incident edge elevations."""

    def test_mean_of_incident_edges(self) -> None:
        graph = nx.DiGraph()
        for a, b, elev in ((0, 1, 2.0), (1, 2, 4.0)):
            graph.add_edge(a, b, elevation_m=elev)
            graph.add_edge(b, a, elevation_m=elev)
        result = node_elevation(graph)
        self.assertEqual(result, {0: 2.0, 1: 3.0, 2: 4.0})

    def test_missing_elevation_raises(self) -> None:
        graph = nx.DiGraph()
        graph.add_edge(0, 1, length=10.0)
        with self.assertRaises(ValueError):
            node_elevation(graph)


class TestFindShelters(unittest.TestCase):
    """Greedy selection of high, well-separated shelters."""

    def setUp(self) -> None:
        self.graph = make_costed_grid()
        self.elevation = node_elevation(self.graph)

    def test_returns_requested_number_of_distinct_nodes(self) -> None:
        shelters = find_shelters(self.graph, count=3, min_separation_m=200.0)
        self.assertEqual(len(shelters), 3)
        self.assertEqual(len(set(shelters)), 3)

    def test_first_shelter_is_the_highest_node(self) -> None:
        shelters = find_shelters(self.graph, count=3, min_separation_m=200.0)
        self.assertEqual(self.elevation[shelters[0]], max(self.elevation.values()))

    def test_shelters_are_chosen_in_non_increasing_elevation(self) -> None:
        shelters = find_shelters(self.graph, count=4, min_separation_m=200.0)
        heights = [self.elevation[s] for s in shelters]
        self.assertEqual(heights, sorted(heights, reverse=True))

    def test_minimum_separation_is_respected(self) -> None:
        separation = 300.0
        shelters = find_shelters(self.graph, count=5, min_separation_m=separation)
        for i, a in enumerate(shelters):
            for b in shelters[i + 1:]:
                gap = haversine_m(self.graph.nodes[a]["y"], self.graph.nodes[a]["x"],
                                  self.graph.nodes[b]["y"], self.graph.nodes[b]["x"])
                self.assertGreaterEqual(gap, separation)

    def test_returns_fewer_when_area_is_too_small(self) -> None:
        shelters = find_shelters(self.graph, count=5, min_separation_m=1_000_000.0)
        self.assertEqual(len(shelters), 1)

    def test_invalid_count_raises(self) -> None:
        with self.assertRaises(ValueError):
            find_shelters(self.graph, count=0)


class TestRouteToBestShelter(unittest.TestCase):
    """Choosing the cheapest reachable shelter."""

    def test_picks_the_cheapest_shelter(self) -> None:
        graph = make_costed_grid(6, 6)
        result = route_to_best_shelter(graph, 0, [35, 1])   # far corner vs neighbour
        self.assertEqual(result.path[-1], 1)

    def test_default_and_dijkstra_agree_on_cost(self) -> None:
        graph = make_costed_grid(6, 6)
        default = route_to_best_shelter(graph, 0, [35, 20, 5])
        explicit = route_to_best_shelter(graph, 0, [35, 20, 5], search=dijkstra)
        self.assertAlmostEqual(default.cost, explicit.cost, places=6)

    def test_unreachable_shelters_are_skipped(self) -> None:
        graph = make_costed_grid(4, 4)
        graph.add_node(999, y=0.0, x=0.0)   # island
        result = route_to_best_shelter(graph, 0, [999, 15])
        self.assertEqual(result.path[-1], 15)

    def test_raises_when_no_shelter_is_reachable(self) -> None:
        graph = make_costed_grid(4, 4)
        graph.add_node(999, y=0.0, x=0.0)
        with self.assertRaises(NoRouteError):
            route_to_best_shelter(graph, 0, [999])
        with self.assertRaises(NoRouteError):
            route_to_best_shelter(graph, 0, [])

    def test_equal_cost_is_broken_by_shorter_distance(self) -> None:
        """With equal cost, the route that is shorter in metres wins."""
        graph = nx.DiGraph()
        graph.add_edge("S", "A", length=100.0, flood_prob=0.0, cost=100.0)
        graph.add_edge("S", "B", length=50.0, flood_prob=0.5, cost=100.0)
        result = route_to_best_shelter(graph, "S", ["A", "B"], search=dijkstra)
        self.assertEqual(result.path[-1], "B")


if __name__ == "__main__":
    unittest.main()