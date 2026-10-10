"""Tests for the Streamlit dashboard (Phase 6).

* ``TestDashboardHelpers`` checks the functions behind the pages.
* ``TestDashboardPages`` uses Streamlit's built-in ``AppTest`` to run the app
  headlessly, open every page, press the buttons, and confirm nothing crashes.

The OSM download is replaced by an offline stub, so the tests are fast and
need no internet connection. They are skipped automatically until the model
and metrics exist (``python -m models.train_model``).

Run from the project root:
    python -m unittest tests.test_dashboard_smoke -v
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import streamlit as st
from streamlit.testing.v1 import AppTest

from dashboard import streamlit_app
from routing.algorithms import NoRouteError
from utils.config import METRICS_PATH, MODEL_PATH

MAIN_FILE = str(Path(__file__).resolve().parent.parent / "main.py")
TIMEOUT_S = 120
PAGES = ["Home", "Flood Prediction", "Route Planner", "Analytics"]
ARTIFACTS_READY = MODEL_PATH.exists() and METRICS_PATH.exists()


class OfflineDashboardCase(unittest.TestCase):
    """Base class: stub the OSM calls and start every test with empty caches."""

    def setUp(self) -> None:
        st.cache_resource.clear()
        st.cache_data.clear()
        for target, kwargs in (
                ("dashboard.streamlit_app.load_osm_graph",
                 {"side_effect": RuntimeError("offline test")}),
                ("dashboard.streamlit_app.fetch_river_lines", {"return_value": None})):
            patcher = mock.patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)


def open_page(page: str) -> AppTest:
    """Start the app and switch to the named page."""
    app = AppTest.from_file(MAIN_FILE, default_timeout=TIMEOUT_S).run()
    if page != "Home":
        app.sidebar.radio[0].set_value(page).run()
    return app


@unittest.skipUnless(ARTIFACTS_READY, "Run `python -m models.train_model` first")
class TestDashboardHelpers(OfflineDashboardCase):
    """Functions used by the pages."""

    def test_scenario_graph_has_risk_and_cost_on_every_edge(self) -> None:
        graph = streamlit_app.build_scenario_graph(200.0, 4.0, 5.0)
        for _, _, data in graph.edges(data=True):
            self.assertTrue(0.0 <= data["flood_prob"] <= 1.0)
            self.assertGreaterEqual(data["cost"], data["length"])

    def test_scenarios_do_not_modify_the_cached_base_graph(self) -> None:
        streamlit_app.build_scenario_graph(400.0, 8.0, 25.0)
        base = streamlit_app.get_base_graph()
        for _, _, data in base.edges(data=True):
            self.assertNotIn("flood_prob", data)
            self.assertNotIn("cost", data)

    def test_base_graph_is_loaded_only_once(self) -> None:
        self.assertIs(streamlit_app.get_base_graph(), streamlit_app.get_base_graph())

    def test_two_scenarios_are_independent(self) -> None:
        dry = streamlit_app.build_scenario_graph(0.0, 0.0, 5.0)
        wet = streamlit_app.build_scenario_graph(400.0, 8.0, 5.0)
        mean = lambda g: sum(d["flood_prob"] for _, _, d in g.edges(data=True)) / g.number_of_edges()  # noqa: E731, E501
        self.assertLess(mean(dry), mean(wet))

    def test_route_result_to_nearest_shelter(self) -> None:
        graph = streamlit_app.get_base_graph()
        start = min(graph.nodes)
        result = streamlit_app.compute_route_result(200.0, 4.0, 5.0, start, None)
        self.assertEqual(result["route"].path[0], start)
        self.assertIn(result["destination"], streamlit_app.get_shelters())
        self.assertEqual(result["baseline"].path[-1], result["destination"])
        # plain distance can never be longer than the flood-aware route
        self.assertLessEqual(result["baseline"].distance_m, result["route"].distance_m + 1e-6)

    def test_route_result_to_chosen_destination(self) -> None:
        nodes = sorted(streamlit_app.get_base_graph().nodes)
        result = streamlit_app.compute_route_result(200.0, 4.0, 5.0, nodes[0], nodes[-1])
        self.assertEqual(result["route"].path[-1], nodes[-1])

    def test_zero_weight_route_equals_shortest_route(self) -> None:
        nodes = sorted(streamlit_app.get_base_graph().nodes)
        result = streamlit_app.compute_route_result(200.0, 4.0, 0.0, nodes[0], nodes[-1])
        self.assertAlmostEqual(result["route"].distance_m, result["baseline"].distance_m, places=6)

    def test_unknown_node_is_reported(self) -> None:
        with self.assertRaises((KeyError, NoRouteError)):
            streamlit_app.compute_route_result(200.0, 4.0, 5.0, -999, None)

    def test_benchmark_compares_both_algorithms(self) -> None:
        table = streamlit_app.algorithm_benchmark(200.0, 4.0, 5.0, pairs=6)
        self.assertEqual(list(table["Algorithm"]), ["Dijkstra", "A*"])
        by_name = table.set_index("Algorithm")
        self.assertAlmostEqual(by_name.loc["Dijkstra", "Cost"], by_name.loc["A*", "Cost"], places=2)
        self.assertLessEqual(by_name.loc["A*", "Nodes visited"],
                             by_name.loc["Dijkstra", "Nodes visited"])

    def test_missing_metrics_file_gives_empty_structure(self) -> None:
        metrics = streamlit_app.load_metrics(Path("no_such_metrics.json"))
        self.assertEqual(metrics["results"], {})


@unittest.skipUnless(ARTIFACTS_READY, "Run `python -m models.train_model` first")
class TestDashboardPages(OfflineDashboardCase):
    """Every page must render, and every button must work, without exceptions."""

    def test_navigation_lists_the_four_required_pages(self) -> None:
        self.assertEqual(list(open_page("Home").sidebar.radio[0].options), PAGES)

    def test_home_page(self) -> None:
        app = open_page("Home")
        self.assertEqual(len(app.exception), 0)
        self.assertIn("FloodGuard Lite", [t.value for t in app.title])

    def test_prediction_page_before_and_after_clicking_predict(self) -> None:
        app = open_page("Flood Prediction")
        self.assertEqual(len(app.exception), 0)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertTrue(any("Flood probability" in m.value for m in app.markdown))

    def test_prediction_sliders_use_the_configured_ranges(self) -> None:
        from utils.config import VALID_RANGES
        app = open_page("Flood Prediction")
        self.assertEqual((app.slider[0].min, app.slider[0].max), VALID_RANGES["rainfall_mm"])
        self.assertEqual((app.slider[1].min, app.slider[1].max), VALID_RANGES["river_level_m"])

    def test_route_planner_waits_for_the_compute_button(self) -> None:
        app = open_page("Route Planner")
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 0)          # nothing computed yet
        self.assertEqual(len(app.info), 1)

    def test_compute_route_to_nearest_shelter(self) -> None:
        app = open_page("Route Planner")
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(app.metric), 4)
        self.assertGreaterEqual(len(app.dataframe), 1)

    def test_result_stays_after_another_rerun(self) -> None:
        app = open_page("Route Planner")
        app.button[0].click().run()
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 4)

    def test_compute_route_survives_extreme_scenarios(self) -> None:
        app = open_page("Route Planner")
        app.slider[0].set_value(400.0)   # rainfall
        app.slider[1].set_value(8.0)     # river level
        app.slider[2].set_value(25.0)    # flood weight
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 4)

    def test_same_start_and_destination_gives_a_warning(self) -> None:
        app = open_page("Route Planner")
        app.selectbox[1].set_value(app.selectbox[0].value)   # destination := start
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.warning), 1)
        self.assertEqual(len(app.metric), 0)

    def test_analytics_page(self) -> None:
        app = open_page("Analytics")
        self.assertEqual(len(app.exception), 0)
        self.assertGreaterEqual(len(app.dataframe), 2)   # model table + algorithm table


if __name__ == "__main__":
    unittest.main()