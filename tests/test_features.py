"""Tests for utils/features.py: feature engineering shared by training and
prediction (Phase 6).

A mistake here causes "train/serve skew": the model is trained on one feature
layout and asked to predict on another, which silently ruins predictions.

Run from the project root:
    python -m unittest tests.test_features -v
"""
from __future__ import annotations

import math
import unittest

import pandas as pd

from utils.config import FEATURE_COLUMNS, NUMERIC_FEATURES, ROAD_TYPES
from utils.features import PROXIMITY_SCALE_M, build_features


def sample_frame() -> pd.DataFrame:
    """Two hand-made road segments with easy-to-check values."""
    return pd.DataFrame({
        "rainfall_mm": [100.0, 300.0],
        "river_level_m": [2.0, 6.0],
        "elevation_m": [4.0, 9.0],
        "distance_from_river_m": [0.0, 500.0],
        "historical_flood_freq": [1.0, 7.0],
        "road_type": ["residential", "primary"],
    }, index=[10, 11])


class TestFeatureLayout(unittest.TestCase):
    """The output must always have exactly the layout the model expects."""

    def test_columns_match_config_exactly_and_in_order(self) -> None:
        self.assertEqual(list(build_features(sample_frame()).columns), list(FEATURE_COLUMNS))

    def test_row_count_and_index_are_preserved(self) -> None:
        frame = sample_frame()
        features = build_features(frame)
        self.assertEqual(list(features.index), list(frame.index))

    def test_no_missing_values_for_clean_input(self) -> None:
        self.assertFalse(build_features(sample_frame()).isna().any().any())

    def test_single_row_input_works(self) -> None:
        """The Flood Prediction page sends exactly one row."""
        features = build_features(sample_frame().iloc[[0]])
        self.assertEqual(features.shape, (1, len(FEATURE_COLUMNS)))

    def test_extra_columns_are_ignored(self) -> None:
        frame = sample_frame().assign(flooded=[0, 1], note="x")
        self.assertEqual(list(build_features(frame).columns), list(FEATURE_COLUMNS))

    def test_input_is_not_modified(self) -> None:
        frame = sample_frame()
        before = frame.copy(deep=True)
        build_features(frame)
        pd.testing.assert_frame_equal(frame, before)

    def test_output_is_deterministic(self) -> None:
        pd.testing.assert_frame_equal(build_features(sample_frame()),
                                      build_features(sample_frame()))


class TestEngineeredValues(unittest.TestCase):
    """Check the domain formulas with numbers worked out by hand."""

    def setUp(self) -> None:
        self.features = build_features(sample_frame())

    def test_numeric_features_are_copied_unchanged(self) -> None:
        frame = sample_frame()
        for column in NUMERIC_FEATURES:
            self.assertEqual(list(self.features[column]), list(frame[column]))

    def test_river_proximity_decays_exponentially(self) -> None:
        # distance 0 -> 1.0 ; distance == scale -> 1/e
        self.assertAlmostEqual(self.features.loc[10, "river_proximity"], 1.0)
        self.assertAlmostEqual(self.features.loc[11, "river_proximity"],
                               math.exp(-500.0 / PROXIMITY_SCALE_M))

    def test_river_exposure_is_level_times_proximity(self) -> None:
        for row in (10, 11):
            self.assertAlmostEqual(
                self.features.loc[row, "river_exposure"],
                self.features.loc[row, "river_level_m"] * self.features.loc[row, "river_proximity"])

    def test_rain_per_elevation(self) -> None:
        self.assertAlmostEqual(self.features.loc[10, "rain_per_elevation"], 100.0 / (1.0 + 4.0))
        self.assertAlmostEqual(self.features.loc[11, "rain_per_elevation"], 300.0 / (1.0 + 9.0))

    def test_zero_elevation_does_not_divide_by_zero(self) -> None:
        frame = sample_frame()
        frame["elevation_m"] = 0.0
        self.assertTrue(math.isfinite(build_features(frame)["rain_per_elevation"].max()))


class TestRoadTypeEncoding(unittest.TestCase):
    """One-hot encoding with a fixed category order."""

    def test_exactly_one_road_column_is_set_per_row(self) -> None:
        features = build_features(sample_frame())
        road_columns = [f"road_{r}" for r in ROAD_TYPES]
        self.assertTrue((features[road_columns].sum(axis=1) == 1).all())

    def test_correct_column_is_set(self) -> None:
        features = build_features(sample_frame())
        self.assertEqual(features.loc[10, "road_residential"], 1)
        self.assertEqual(features.loc[11, "road_primary"], 1)
        self.assertEqual(features.loc[10, "road_primary"], 0)

    def test_every_road_type_in_config_is_supported(self) -> None:
        frame = pd.concat([sample_frame().iloc[[0]]] * len(ROAD_TYPES), ignore_index=True)
        frame["road_type"] = ROAD_TYPES
        features = build_features(frame)
        for i, road in enumerate(ROAD_TYPES):
            self.assertEqual(features.loc[i, f"road_{road}"], 1)


class TestValidation(unittest.TestCase):
    """Bad data must raise a clear error instead of producing wrong features."""

    def test_missing_column_raises_and_names_it(self) -> None:
        frame = sample_frame().drop(columns=["river_level_m"])
        with self.assertRaises(ValueError) as ctx:
            build_features(frame)
        self.assertIn("river_level_m", str(ctx.exception))

    def test_missing_road_type_column_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_features(sample_frame().drop(columns=["road_type"]))

    def test_unknown_road_type_raises_and_names_it(self) -> None:
        frame = sample_frame()
        frame.loc[10, "road_type"] = "footpath"
        with self.assertRaises(ValueError) as ctx:
            build_features(frame)
        self.assertIn("footpath", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()