"""Tests for shelters and the Folium map (map tests skip if folium is absent).

Run from the project root:  python -m unittest tests.test_phase4 -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from routing.algorithms import NoRouteError
from routing.cost import apply_costs
from routing.graph_builder import haversine_m
from routing.shelters import find_shelters, route_to_best_shelter
from tests.test_routing import SOURCE, make_graph

try:
    import folium  # noqa: F401
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False


class ShelterTests(unittest.TestCase):
    """Shelter selection and best-shelter routing."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = make_graph()

    def test_shelters_are_separated(self) -> None:
        shelters = find_shelters(self.graph, count=3, min_separation_m=500)
        self.assertGreaterEqual(len(shelters), 2)
        for i, a in enumerate(shelters):
            for b in shelters[i + 1:]:
                gap = haversine_m(self.graph.nodes[a]["y"], self.graph.nodes[a]["x"],
                                  self.graph.nodes[b]["y"], self.graph.nodes[b]["x"])
                self.assertGreaterEqual(gap, 500)

    def test_best_shelter_is_cheapest(self) -> None:
        apply_costs(self.graph, 5.0)
        shelters = find_shelters(self.graph, count=3, min_separation_m=500)
        best = route_to_best_shelter(self.graph, SOURCE, shelters)
        self.assertIn(best.path[-1], shelters)
        for s in shelters:
            single = route_to_best_shelter(self.graph, SOURCE, [s])
            self.assertLessEqual(best.cost, single.cost + 1e-9)

    def test_no_shelter_reachable(self) -> None:
        apply_costs(self.graph, 5.0, block_threshold=0.0)
        with self.assertRaises(NoRouteError):
            route_to_best_shelter(self.graph, SOURCE, [10, 20])


@unittest.skipUnless(HAS_FOLIUM, "folium not installed")
class MapTests(unittest.TestCase):
    """Map rendering smoke tests."""

    def test_map_saved_with_legend(self) -> None:
        from dashboard.map_builder import build_map, risk_class, save_map
        graph = make_graph()
        shelters = find_shelters(graph, count=3, min_separation_m=500)
        apply_costs(graph, 5.0)
        route = route_to_best_shelter(graph, SOURCE, shelters)
        fmap = build_map(graph, route, None, SOURCE, route.path[-1], shelters)
        with tempfile.TemporaryDirectory() as tmp:
            out = save_map(fmap, Path(tmp) / "m.html")
            self.assertTrue(out.exists())
            self.assertIn("FloodGuard Lite", out.read_text(encoding="utf-8"))
        self.assertEqual(risk_class(0.1), "safe")
        self.assertEqual(risk_class(0.45), "moderate")
        self.assertEqual(risk_class(0.9), "flooded")
        with self.assertRaises(ValueError):
            risk_class(1.2)


if __name__ == "__main__":
    unittest.main()
