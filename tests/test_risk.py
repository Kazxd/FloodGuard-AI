"""Tests for routing/risk.py: assigning ML flood probabilities to roads (Phase 6).

Most tests use a tiny fake model so they are fast and fully predictable; one
integration test trains a real Random Forest to check the whole chain
(static features -> feature engineering -> model -> edge risk).

NOTE: if your module is not ``routing/risk.py``, change the import below.

Run from the project root:
    python -m unittest tests.test_risk -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from models.train_model import clean_data
from routing.graph_builder import attach_static_features, build_grid_graph
from routing.risk import assign_flood_risk, load_model_bundle
from tests.test_ml_pipeline import make_dataset
from utils.config import (FEATURE_COLUMNS, RANDOM_SEED, TARGET_COLUMN,
                          VALID_RANGES)
from utils.features import build_features


class FakeModel:
    """Predictable stand-in: risk grows with rainfall and river level."""

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        rain = frame["rainfall_mm"] / VALID_RANGES["rainfall_mm"][1]
        river = frame["river_level_m"] / VALID_RANGES["river_level_m"][1]
        prob = (0.5 * rain + 0.5 * river).clip(0.0, 1.0).to_numpy()
        return np.column_stack([1.0 - prob, prob])


def fake_bundle() -> Dict[str, Any]:
    return {"model": FakeModel(), "feature_columns": FEATURE_COLUMNS,
            "model_name": "Fake", "threshold": 0.5}


def featured_grid() -> Any:
    graph = build_grid_graph(6, 6)
    attach_static_features(graph)
    return graph


def mean_risk(graph: Any) -> float:
    values = [d["flood_prob"] for _, _, d in graph.edges(data=True)]
    return sum(values) / len(values)


class TestAssignFloodRisk(unittest.TestCase):
    """Behaviour of assign_flood_risk with a predictable model."""

    def test_every_edge_gets_a_probability_between_zero_and_one(self) -> None:
        graph = featured_grid()
        assign_flood_risk(graph, 200.0, 4.0, fake_bundle())
        for _, _, data in graph.edges(data=True):
            self.assertIn("flood_prob", data)
            self.assertGreaterEqual(data["flood_prob"], 0.0)
            self.assertLessEqual(data["flood_prob"], 1.0)

    def test_heavier_scenario_gives_higher_risk(self) -> None:
        dry, wet = featured_grid(), featured_grid()
        assign_flood_risk(dry, 0.0, 0.0, fake_bundle())
        assign_flood_risk(wet, 400.0, 8.0, fake_bundle())
        self.assertLess(mean_risk(dry), mean_risk(wet))

    def test_scenario_boundaries_are_accepted(self) -> None:
        graph = featured_grid()
        for rain, river in ((0.0, 0.0), (400.0, 8.0)):
            assign_flood_risk(graph, rain, river, fake_bundle())

    def test_out_of_range_inputs_raise_and_leave_graph_unchanged(self) -> None:
        for rain, river in ((-1.0, 2.0), (401.0, 2.0), (100.0, -0.1), (100.0, 8.5)):
            graph = featured_grid()
            with self.assertRaises(ValueError):
                assign_flood_risk(graph, rain, river, fake_bundle())
            self.assertTrue(all("flood_prob" not in d for _, _, d in graph.edges(data=True)))

    def test_graph_without_static_features_is_rejected(self) -> None:
        graph = build_grid_graph(4, 4)   # no attach_static_features call
        with self.assertRaises(ValueError) as ctx:
            assign_flood_risk(graph, 100.0, 2.0, fake_bundle())
        self.assertIn("attach_static_features", str(ctx.exception))

    def test_graph_without_edges_is_rejected(self) -> None:
        import networkx as nx
        graph = nx.DiGraph()
        graph.add_node(0, x=80.0, y=13.0)
        with self.assertRaises(ValueError):
            assign_flood_risk(graph, 100.0, 2.0, fake_bundle())

    def test_rerunning_with_a_new_scenario_overwrites_old_values(self) -> None:
        graph = featured_grid()
        assign_flood_risk(graph, 400.0, 8.0, fake_bundle())
        high = mean_risk(graph)
        assign_flood_risk(graph, 0.0, 0.0, fake_bundle())
        self.assertLess(mean_risk(graph), high)


class TestLoadModelBundle(unittest.TestCase):
    """Loading the artifact written by models.train_model."""

    def test_missing_file_gives_helpful_error(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            load_model_bundle(Path("no_such_model.joblib"))
        self.assertIn("train_model", str(ctx.exception))

    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bundle.joblib"
            joblib.dump(fake_bundle(), path)
            loaded = load_model_bundle(path)
        self.assertEqual(loaded["model_name"], "Fake")
        self.assertEqual(list(loaded["feature_columns"]), list(FEATURE_COLUMNS))


class TestRiskWithRealModel(unittest.TestCase):
    """Whole chain with a genuinely trained Random Forest."""

    @classmethod
    def setUpClass(cls) -> None:
        df = clean_data(make_dataset(800))
        model = RandomForestClassifier(n_estimators=40, random_state=RANDOM_SEED)
        model.fit(build_features(df), df[TARGET_COLUMN].astype(int))
        cls.bundle = {"model": model, "feature_columns": FEATURE_COLUMNS,
                      "model_name": "Random Forest", "threshold": 0.5}

    def test_probabilities_are_valid_and_respond_to_the_scenario(self) -> None:
        dry, wet = featured_grid(), featured_grid()
        assign_flood_risk(dry, 0.0, 0.0, self.bundle)
        assign_flood_risk(wet, 400.0, 8.0, self.bundle)
        for graph in (dry, wet):
            for _, _, data in graph.edges(data=True):
                self.assertTrue(0.0 <= data["flood_prob"] <= 1.0)
        self.assertLess(mean_risk(dry), mean_risk(wet))


if __name__ == "__main__":
    unittest.main()