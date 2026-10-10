"""Consistency checks for utils/config.py (Phase 6).

A wrong constant here breaks the model, the routing cost or the map without
raising an error, so these tests guard the invariants other modules rely on.

Run from the project root:
    python -m unittest tests.test_config -v
"""
from __future__ import annotations

import unittest

from routing.graph_builder import _HIGHWAY_MAP
from utils import config


class TestConfigConsistency(unittest.TestCase):
    """Constants must agree with each other."""

    def test_valid_ranges_cover_exactly_the_numeric_features(self) -> None:
        self.assertEqual(set(config.VALID_RANGES), set(config.NUMERIC_FEATURES))

    def test_every_range_is_ordered(self) -> None:
        for name, (low, high) in config.VALID_RANGES.items():
            self.assertLess(low, high, f"{name} has min >= max")

    def test_feature_columns_are_built_from_the_parts(self) -> None:
        expected = (config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES
                    + [f"road_{r}" for r in config.ROAD_TYPES])
        self.assertEqual(config.FEATURE_COLUMNS, expected)

    def test_feature_columns_are_unique_and_exclude_target(self) -> None:
        columns = config.FEATURE_COLUMNS
        self.assertEqual(len(columns), len(set(columns)))
        self.assertNotIn(config.TARGET_COLUMN, columns)

    def test_osm_highway_map_only_produces_known_road_types(self) -> None:
        """A road type missing from ROAD_TYPES would have no one-hot column."""
        self.assertTrue(set(_HIGHWAY_MAP.values()) <= set(config.ROAD_TYPES))


class TestConfigThresholds(unittest.TestCase):
    """Numeric settings must be in sensible ranges."""

    def test_flood_weight_is_non_negative(self) -> None:
        """A* stays optimal only if cost >= length, which needs weight >= 0."""
        self.assertGreaterEqual(config.DEFAULT_FLOOD_WEIGHT, 0.0)

    def test_risk_bands_are_ordered_and_within_unit_interval(self) -> None:
        self.assertLess(config.RISK_SAFE_MAX, config.RISK_FLOODED_MIN)
        self.assertGreaterEqual(config.RISK_SAFE_MAX, 0.0)
        self.assertLessEqual(config.RISK_FLOODED_MIN, 1.0)

    def test_training_settings_are_valid(self) -> None:
        self.assertTrue(0.0 < config.TEST_SIZE < 1.0)
        self.assertGreaterEqual(config.CV_FOLDS, 2)
        self.assertTrue(0.0 < config.DECISION_THRESHOLD < 1.0)

    def test_map_colours_are_hex_codes(self) -> None:
        for colour in (config.COLOR_SAFE, config.COLOR_MODERATE,
                       config.COLOR_FLOODED, config.COLOR_ROUTE):
            self.assertRegex(colour, r"^#[0-9a-fA-F]{6}$")


if __name__ == "__main__":
    unittest.main()