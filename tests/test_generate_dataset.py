"""Tests for data/generate_dataset.py: the synthetic flood dataset (Phase 6).

Because the labels come from a known equation, we can test the generator
against its own documentation: valid ranges, label probabilities that match
the stored ``true_probability``, and the physical trends it claims to model
(more rain, lower ground, nearer the river and smaller roads flood more).

Run from the project root:
    python -m unittest tests.test_generate_dataset -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from data import generate_dataset
from data.generate_dataset import (ROAD_TYPE_EFFECT, ROAD_TYPE_PROBS, generate,
                                   sigmoid)
from models.train_model import clean_data
from utils.config import (FEATURE_COLUMNS, RANDOM_SEED, ROAD_TYPES,
                          TARGET_COLUMN, VALID_RANGES)
from utils.features import build_features

EXPECTED_COLUMNS = {*VALID_RANGES, "road_type", "true_probability", TARGET_COLUMN}


class TestSigmoid(unittest.TestCase):
    """Log-odds to probability."""

    def test_known_values_and_symmetry(self) -> None:
        self.assertAlmostEqual(float(sigmoid(np.array(0.0))), 0.5)
        z = np.array([-3.0, -1.0, 2.0])
        np.testing.assert_allclose(sigmoid(z) + sigmoid(-z), 1.0)

    def test_output_stays_between_zero_and_one(self) -> None:
        out = sigmoid(np.array([-40.0, 0.0, 40.0]))
        self.assertTrue(np.all((out >= 0.0) & (out <= 1.0)))


class TestGeneratorSettings(unittest.TestCase):
    """The lookup tables must agree with the project configuration."""

    def test_road_type_tables_match_config(self) -> None:
        self.assertEqual(set(ROAD_TYPE_EFFECT), set(ROAD_TYPES))
        self.assertEqual(len(ROAD_TYPE_PROBS), len(ROAD_TYPES))
        self.assertAlmostEqual(sum(ROAD_TYPE_PROBS), 1.0)

    def test_smaller_roads_have_larger_flood_effect(self) -> None:
        """ROAD_TYPES is ordered from small local roads to trunk roads."""
        effects = [ROAD_TYPE_EFFECT[r] for r in ROAD_TYPES]
        self.assertEqual(effects, sorted(effects, reverse=True))


class TestGeneratedData(unittest.TestCase):
    """Structure and validity of the generated table."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = generate(20_000)

    def test_shape_and_columns(self) -> None:
        self.assertEqual(len(self.df), 20_000)
        self.assertEqual(set(self.df.columns), EXPECTED_COLUMNS)

    def test_no_missing_values(self) -> None:
        self.assertFalse(self.df.isna().any().any())

    def test_features_are_inside_the_valid_ranges(self) -> None:
        for column, (low, high) in VALID_RANGES.items():
            self.assertGreaterEqual(self.df[column].min(), low, column)
            self.assertLessEqual(self.df[column].max(), high, column)

    def test_labels_and_probabilities_are_valid(self) -> None:
        self.assertTrue(set(self.df[TARGET_COLUMN].unique()) <= {0, 1})
        self.assertTrue(self.df["true_probability"].between(0.0, 1.0).all())

    def test_road_types_are_known_and_follow_the_requested_mix(self) -> None:
        self.assertTrue(set(self.df["road_type"]) <= set(ROAD_TYPES))
        shares = self.df["road_type"].value_counts(normalize=True)
        for road, expected in zip(ROAD_TYPES, ROAD_TYPE_PROBS):
            self.assertAlmostEqual(shares[road], expected, delta=0.02)

    def test_both_classes_are_present_with_a_realistic_flood_rate(self) -> None:
        self.assertTrue(0.15 < self.df[TARGET_COLUMN].mean() < 0.55)

    def test_labels_are_drawn_from_the_stored_probabilities(self) -> None:
        """Calibration: observed flood rate ~ mean of true_probability."""
        self.assertAlmostEqual(self.df[TARGET_COLUMN].mean(),
                               self.df["true_probability"].mean(), delta=0.01)
        for low, high in ((0.0, 0.2), (0.4, 0.6), (0.8, 1.0)):
            band = self.df[self.df["true_probability"].between(low, high)]
            if len(band) > 200:
                self.assertAlmostEqual(band[TARGET_COLUMN].mean(),
                                       band["true_probability"].mean(), delta=0.05)

    def test_cleaning_keeps_every_generated_row(self) -> None:
        self.assertEqual(len(clean_data(self.df)), len(self.df))


class TestReproducibility(unittest.TestCase):
    """Same seed, same data; different seed, different data."""

    def test_same_seed_gives_identical_tables(self) -> None:
        pd.testing.assert_frame_equal(generate(500, seed=5), generate(500, seed=5))

    def test_different_seed_changes_the_table(self) -> None:
        self.assertFalse(generate(500, seed=5).equals(generate(500, seed=6)))

    def test_non_positive_size_is_rejected(self) -> None:
        for bad in (0, -10):
            with self.assertRaises(ValueError):
                generate(bad)


class TestPhysicalTrends(unittest.TestCase):
    """The generator should model the flood physics it claims to."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = generate(20_000)

    def _mean_risk(self, mask: pd.Series) -> float:
        return float(self.df.loc[mask, "true_probability"].mean())

    def test_heavy_rain_floods_more_than_light_rain(self) -> None:
        low, high = self.df["rainfall_mm"].quantile([0.25, 0.75])
        self.assertGreater(self._mean_risk(self.df["rainfall_mm"] >= high),
                           self._mean_risk(self.df["rainfall_mm"] <= low))

    def test_low_ground_floods_more_than_high_ground(self) -> None:
        low, high = self.df["elevation_m"].quantile([0.25, 0.75])
        self.assertGreater(self._mean_risk(self.df["elevation_m"] <= low),
                           self._mean_risk(self.df["elevation_m"] >= high))

    def test_roads_near_the_river_flood_more_when_the_river_is_high(self) -> None:
        wet = self.df["river_level_m"] >= self.df["river_level_m"].quantile(0.75)
        near = self.df["distance_from_river_m"] <= 300
        far = self.df["distance_from_river_m"] >= 2000
        self.assertGreater(self._mean_risk(wet & near), self._mean_risk(wet & far))

    def test_local_roads_flood_more_than_trunk_roads(self) -> None:
        self.assertGreater(self._mean_risk(self.df["road_type"] == "residential"),
                           self._mean_risk(self.df["road_type"] == "trunk"))

    def test_flood_history_is_higher_for_low_ground(self) -> None:
        low, high = self.df["elevation_m"].quantile([0.25, 0.75])
        self.assertGreater(
            self.df.loc[self.df["elevation_m"] <= low, "historical_flood_freq"].mean(),
            self.df.loc[self.df["elevation_m"] >= high, "historical_flood_freq"].mean())


class TestNoLeakage(unittest.TestCase):
    """true_probability is the answer key and must never reach the model."""

    def test_true_probability_is_not_a_feature(self) -> None:
        self.assertNotIn("true_probability", FEATURE_COLUMNS)

    def test_feature_builder_ignores_the_answer_columns(self) -> None:
        features = build_features(generate(100))
        self.assertEqual(list(features.columns), list(FEATURE_COLUMNS))


class TestLearnability(unittest.TestCase):
    """A model should learn the pattern, but cannot beat the label noise."""

    def test_model_is_good_but_below_the_theoretical_ceiling(self) -> None:
        df = generate(4000)
        x, y = build_features(df), df[TARGET_COLUMN]
        x_tr, x_te, y_tr, y_te, _, p_te = train_test_split(
            x, y, df["true_probability"], test_size=0.25, stratify=y,
            random_state=RANDOM_SEED)
        model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                       random_state=RANDOM_SEED).fit(x_tr, y_tr)
        model_auc = roc_auc_score(y_te, model.predict_proba(x_te)[:, 1])
        ceiling = roc_auc_score(y_te, p_te)       # best any model could do
        self.assertGreater(model_auc, 0.7)
        self.assertLess(model_auc, ceiling + 0.01)
        self.assertLess(ceiling, 0.99)            # labels are genuinely noisy


class TestMain(unittest.TestCase):
    """The command-line entry point writes the CSV."""

    def test_main_writes_the_dataset_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "road_segments.csv"
            with mock.patch.object(generate_dataset, "RAW_DATA_PATH", target):
                generate_dataset.main()
            self.assertTrue(target.exists())
            written = pd.read_csv(target)
        self.assertEqual(len(written), 6000)
        self.assertEqual(set(written.columns), EXPECTED_COLUMNS)


if __name__ == "__main__":
    unittest.main()