"""Tests for routing/graph_builder.py (Phase 6).

All tests run offline: they use the synthetic grid and hand-made OSM-style
graphs, so OSMnx and an internet connection are not needed.

Run from the project root:
    python -m unittest tests.test_graph_builder -v
"""
from __future__ import annotations

import unittest

import networkx as nx

from routing.graph_builder import (_normalise_highway, attach_static_features,
                                   build_grid_graph, haversine_m, nearest_node,
                                   simplify_to_digraph)
from utils.config import ROAD_TYPES, VALID_RANGES

STATIC_FIELDS = ("elevation_m", "distance_from_river_m", "historical_flood_freq")


class TestHaversine(unittest.TestCase):
    """Great-circle distance is the base of every length and heuristic."""

    def test_same_point_is_zero(self) -> None:
        self.assertEqual(haversine_m(13.0, 80.0, 13.0, 80.0), 0.0)

    def test_one_degree_of_latitude_is_about_111_km(self) -> None:
        self.assertAlmostEqual(haversine_m(0.0, 80.0, 1.0, 80.0), 111_195.0, delta=100.0)

    def test_distance_is_symmetric(self) -> None:
        forward = haversine_m(12.97, 80.21, 13.05, 80.27)
        backward = haversine_m(13.05, 80.27, 12.97, 80.21)
        self.assertAlmostEqual(forward, backward, places=6)


class TestGridGraph(unittest.TestCase):
    """The synthetic grid drives the unit tests and the offline demo."""

    def test_node_and_edge_counts(self) -> None:
        rows, cols = 4, 6
        graph = build_grid_graph(rows, cols)
        self.assertEqual(graph.number_of_nodes(), rows * cols)
        undirected_roads = rows * (cols - 1) + cols * (rows - 1)
        self.assertEqual(graph.number_of_edges(), 2 * undirected_roads)

    def test_every_road_is_two_way_with_equal_length(self) -> None:
        graph = build_grid_graph(5, 5)
        for u, v, data in graph.edges(data=True):
            self.assertTrue(graph.has_edge(v, u))
            self.assertAlmostEqual(graph[v][u]["length"], data["length"], places=6)

    def test_edge_lengths_are_close_to_the_requested_spacing(self) -> None:
        graph = build_grid_graph(5, 5, spacing_m=120.0)
        for _, _, data in graph.edges(data=True):
            self.assertAlmostEqual(data["length"], 120.0, delta=2.4)  # within 2 %

    def test_road_types_are_valid(self) -> None:
        graph = build_grid_graph(10, 10)
        kinds = {d["road_type"] for _, _, d in graph.edges(data=True)}
        self.assertTrue(kinds <= set(ROAD_TYPES))

    def test_grid_is_strongly_connected(self) -> None:
        self.assertTrue(nx.is_strongly_connected(build_grid_graph(6, 6)))

    def test_too_small_grid_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_grid_graph(1, 5)
        with self.assertRaises(ValueError):
            build_grid_graph(5, 1)


class TestHighwayNormalisation(unittest.TestCase):
    """OSM 'highway' tags arrive as strings, lists or stringified lists."""

    def test_plain_tags(self) -> None:
        self.assertEqual(_normalise_highway("primary"), "primary")
        self.assertEqual(_normalise_highway("motorway"), "trunk")
        self.assertEqual(_normalise_highway("living_street"), "residential")

    def test_link_suffix_is_removed(self) -> None:
        self.assertEqual(_normalise_highway("trunk_link"), "trunk")

    def test_list_and_stringified_list_use_first_entry(self) -> None:
        self.assertEqual(_normalise_highway(["secondary", "tertiary"]), "secondary")
        self.assertEqual(_normalise_highway("['primary', 'secondary']"), "primary")

    def test_unknown_tag_falls_back_to_residential(self) -> None:
        self.assertEqual(_normalise_highway("footway"), "residential")
        self.assertEqual(_normalise_highway(None), "residential")


class TestSimplifyToDigraph(unittest.TestCase):
    """Conversion from an OSMnx MultiDiGraph to the simple routing graph."""

    @staticmethod
    def _multigraph() -> nx.MultiDiGraph:
        multi = nx.MultiDiGraph()
        for node in range(1, 6):
            multi.add_node(node, x=80.0 + node * 0.001, y=13.0)
        # 1, 2, 3 form a strongly connected loop.
        for a, b in ((1, 2), (2, 3), (3, 1), (2, 1), (3, 2), (1, 3)):
            multi.add_edge(a, b, length=100.0, highway="residential")
        multi.add_edge(1, 2, length=50.0, highway="primary")   # parallel, shorter
        multi.add_edge(2, 2, length=30.0, highway="residential")  # self-loop
        multi.add_edge(3, 4, length=0.0, highway="residential")   # zero length
        multi.add_edge(4, 5, length=80.0, highway="residential")  # dead-end branch
        return multi

    def test_keeps_shortest_parallel_edge(self) -> None:
        graph = simplify_to_digraph(self._multigraph())
        self.assertEqual(graph[1][2]["length"], 50.0)
        self.assertEqual(graph[1][2]["road_type"], "primary")

    def test_drops_self_loops_zero_length_and_dead_ends(self) -> None:
        graph = simplify_to_digraph(self._multigraph())
        self.assertEqual(set(graph.nodes), {1, 2, 3})
        self.assertFalse(graph.has_edge(2, 2))
        self.assertTrue(nx.is_strongly_connected(graph))

    def test_empty_graph_raises(self) -> None:
        with self.assertRaises(ValueError):
            simplify_to_digraph(nx.MultiDiGraph())


class TestStaticFeatures(unittest.TestCase):
    """Elevation, river distance and flood history attached to each road."""

    def setUp(self) -> None:
        self.graph = build_grid_graph(10, 10)
        attach_static_features(self.graph)

    def test_every_edge_has_all_features_inside_valid_ranges(self) -> None:
        for _, _, data in self.graph.edges(data=True):
            for field in STATIC_FIELDS:
                self.assertIn(field, data)
                low, high = VALID_RANGES[field]
                self.assertGreaterEqual(data[field], low)
                self.assertLessEqual(data[field], high)

    def test_same_seed_gives_identical_features(self) -> None:
        other = build_grid_graph(10, 10)
        attach_static_features(other)
        for u, v, data in self.graph.edges(data=True):
            for field in STATIC_FIELDS:
                self.assertEqual(data[field], other[u][v][field])

    def test_different_seed_changes_features(self) -> None:
        other = build_grid_graph(10, 10)
        attach_static_features(other, seed=999)
        changed = any(data["elevation_m"] != other[u][v]["elevation_m"]
                      for u, v, data in self.graph.edges(data=True))
        self.assertTrue(changed)

    def test_both_directions_of_a_road_share_the_same_features(self) -> None:
        """One physical road must have one elevation and one flood history."""
        for u, v, data in self.graph.edges(data=True):
            for field in STATIC_FIELDS:
                self.assertEqual(data[field], self.graph[v][u][field],
                                 f"{field} differs between {u}->{v} and {v}->{u}")

    def test_roads_near_the_river_are_lower_than_roads_far_away(self) -> None:
        """The synthetic river runs along the south edge of the grid."""
        graph = self.graph
        min_y = min(d["y"] for _, d in graph.nodes(data=True))
        max_y = max(d["y"] for _, d in graph.nodes(data=True))
        south, north = [], []
        for u, v, data in graph.edges(data=True):
            if graph.nodes[u]["y"] == min_y and graph.nodes[v]["y"] == min_y:
                south.append(data)
            if graph.nodes[u]["y"] == max_y and graph.nodes[v]["y"] == max_y:
                north.append(data)
        mean = lambda rows, key: sum(r[key] for r in rows) / len(rows)  # noqa: E731
        self.assertLess(mean(south, "distance_from_river_m"),
                        mean(north, "distance_from_river_m"))
        self.assertLess(mean(south, "elevation_m"), mean(north, "elevation_m"))

    def test_real_river_geometry_is_used_when_given(self) -> None:
        graph = build_grid_graph(8, 8)
        row_lat = graph.nodes[3 * 8]["y"]                 # latitude of grid row 3
        lon_min = min(d["x"] for _, d in graph.nodes(data=True))
        lon_max = max(d["x"] for _, d in graph.nodes(data=True))
        attach_static_features(graph, river_lines=[[(lon_min, row_lat), (lon_max, row_lat)]])
        on_river = [d["distance_from_river_m"] for u, v, d in graph.edges(data=True)
                    if graph.nodes[u]["y"] == row_lat and graph.nodes[v]["y"] == row_lat]
        self.assertTrue(on_river)
        self.assertLess(max(on_river), 1.0)               # roads lying on the river


class TestNearestNode(unittest.TestCase):
    """Snapping clicked map coordinates to intersections."""

    def test_exact_coordinate_returns_that_node(self) -> None:
        graph = build_grid_graph(5, 5)
        for node in (0, 7, 24):
            data = graph.nodes[node]
            self.assertEqual(nearest_node(graph, data["y"], data["x"]), node)

    def test_nearby_point_snaps_to_closest_node(self) -> None:
        graph = build_grid_graph(5, 5)
        data = graph.nodes[12]
        self.assertEqual(nearest_node(graph, data["y"] + 0.00001, data["x"] - 0.00001), 12)


if __name__ == "__main__":
    unittest.main()