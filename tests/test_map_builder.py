"""Tests for dashboard/map_builder.py (Phase 6).

The map is rendered to HTML and inspected as text, so no browser is needed.

Run from the project root:
    python -m unittest tests.test_map_builder -v
"""
from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any

import folium
import networkx as nx

from dashboard.map_builder import (COLOR_BASELINE, build_map, risk_class,
                                   save_map)
from routing.algorithms import astar
from routing.cost import apply_costs
from routing.graph_builder import build_grid_graph
from utils.config import (COLOR_FLOODED, COLOR_MODERATE, COLOR_ROUTE,
                          COLOR_SAFE, RISK_FLOODED_MIN, RISK_SAFE_MAX)

LEVELS = (0.1, 0.5, 0.9)   # one safe, one moderate, one flooded value


def banded_grid(size: int = 5) -> nx.DiGraph:
    """Grid whose roads cycle through safe / moderate / flooded risk."""
    graph = build_grid_graph(size, size)
    assigned = {}
    for u, v, data in graph.edges(data=True):
        key = (min(u, v), max(u, v))
        if key not in assigned:
            assigned[key] = LEVELS[len(assigned) % len(LEVELS)]
        data["flood_prob"] = assigned[key]
    apply_costs(graph, 5.0)
    return graph


def render(fmap: folium.Map) -> str:
    """Return the full HTML of a Folium map."""
    return fmap.get_root().render()


class TestRiskClass(unittest.TestCase):
    """Colour bands for flood probability."""

    def test_bands(self) -> None:
        self.assertEqual(risk_class(0.0), "safe")
        self.assertEqual(risk_class(RISK_SAFE_MAX - 0.001), "safe")
        self.assertEqual(risk_class(RISK_SAFE_MAX), "moderate")
        self.assertEqual(risk_class(RISK_FLOODED_MIN - 0.001), "moderate")
        self.assertEqual(risk_class(RISK_FLOODED_MIN), "flooded")
        self.assertEqual(risk_class(1.0), "flooded")

    def test_out_of_range_probability_raises(self) -> None:
        for bad in (-0.01, 1.01):
            with self.assertRaises(ValueError):
                risk_class(bad)


class TestBuildMap(unittest.TestCase):
    """Content of the generated map."""

    def setUp(self) -> None:
        self.graph = banded_grid()
        self.roads = self.graph.number_of_edges() // 2
        self.route = astar(self.graph, 0, 24)
        self.baseline = astar(self.graph, 0, 24)

    def test_returns_a_folium_map(self) -> None:
        self.assertIsInstance(build_map(self.graph), folium.Map)

    def test_each_physical_road_is_drawn_exactly_once(self) -> None:
        html = render(build_map(self.graph))
        self.assertEqual(html.count("L.polyline("), self.roads)

    def test_routes_add_one_line_each(self) -> None:
        html = render(build_map(self.graph, self.route, self.baseline, 0, 24))
        self.assertEqual(html.count("L.polyline("), self.roads + 2)

    def test_all_risk_colours_and_route_colours_appear(self) -> None:
        html = render(build_map(self.graph, self.route, self.baseline, 0, 24))
        for colour in (COLOR_SAFE, COLOR_MODERATE, COLOR_FLOODED, COLOR_ROUTE, COLOR_BASELINE):
            self.assertIn(colour, html)

    def test_start_destination_and_shelter_markers(self) -> None:
        # shelter 24 is also the destination, so only two shelter markers remain
        html = render(build_map(self.graph, start=0, destination=24, shelters=[24, 12, 20]))
        self.assertEqual(html.count("L.marker("), 1 + 1 + 2)
        for label in ("Start", "Destination", "Shelter 2", "Shelter 3"):
            self.assertIn(label, html)
        self.assertNotIn("Shelter 1", html)

    def test_no_markers_when_none_requested(self) -> None:
        self.assertEqual(render(build_map(self.graph)).count("L.marker("), 0)

    def test_legend_and_layer_names_are_present(self) -> None:
        html = render(build_map(self.graph))
        self.assertIn("FloodGuard Lite", html)
        for name in ("Safe roads", "Moderate-risk roads", "Flooded roads"):
            self.assertIn(name, html)

    def test_tooltip_shows_road_risk(self) -> None:
        html = render(build_map(self.graph))
        self.assertIn("flood risk 90%", html)

    def test_building_a_map_raises_no_warnings(self) -> None:
        """Warnings (for example about map tiles) can mean a blank background."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_map(self.graph)
        self.assertEqual([str(w.message) for w in caught], [])


class TestBuildMapErrors(unittest.TestCase):
    """Invalid graphs must be rejected with clear errors."""

    def test_empty_graph_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_map(nx.DiGraph())

    def test_graph_without_flood_probability_raises(self) -> None:
        graph = build_grid_graph(3, 3)   # no assign_flood_risk call
        with self.assertRaises(ValueError):
            build_map(graph)


class TestSaveMap(unittest.TestCase):
    """Writing the map to disk."""

    def test_creates_folders_and_writes_html(self) -> None:
        fmap = build_map(banded_grid(3))
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "folder" / "map.html"
            returned: Any = save_map(fmap, target)
            self.assertEqual(returned, target)
            self.assertTrue(target.exists())
            self.assertIn("<html", target.read_text(encoding="utf-8").lower())


if __name__ == "__main__":
    unittest.main()