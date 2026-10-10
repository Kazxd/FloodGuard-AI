"""Tests for the machine-learning pipeline (Phase 6).

Covers data cleaning, feature construction, model candidates, evaluation
metrics, figure generation, and (when it exists) the saved model artifact.
The tests build their own small dataset from ``VALID_RANGES`` in
``utils.config``, so they stay correct if you change the ranges later.

Run from the project root:
    python -m unittest tests.test_ml_pipeline -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from models.train_model import (clean_data, evaluate, get_candidates, load_data,
                                plot_confusion, plot_importance)
from utils.config import (FEATURE_COLUMNS, MODEL_PATH, RANDOM_SEED, ROAD_TYPES,
                          TARGET_COLUMN, VALID_RANGES)
from utils.features import build_features

METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc", "brier")


def make_dataset(rows: int = 400, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create a small, valid, learnable dataset using the project's own schema.

    The label depends on the (normalised) inputs plus noise, so a tree model
    can learn it, and both classes are guaranteed to be present.
    """
    rng = np.random.default_rng(seed)
    data = {col: rng.uniform(lo, hi, rows) for col, (lo, hi) in VALID_RANGES.items()}
    data["road_type"] = rng.choice(ROAD_TYPES, rows)
    df = pd.DataFrame(data)
    signal = sum((df[c] - lo) / (hi - lo) for c, (lo, hi) in VALID_RANGES.items())
    signal = signal + rng.normal(0, 0.3, rows)
    df[TARGET_COLUMN] = (signal > signal.median()).astype(int)
    return df


class TestCleanData(unittest.TestCase):
    """clean_data must keep good rows and drop each kind of bad row."""

    def setUp(self) -> None:
        self.df = make_dataset(100)
        self.first_col = next(iter(VALID_RANGES))
        self.first_hi = VALID_RANGES[self.first_col][1]

    def test_valid_rows_are_kept(self) -> None:
        self.assertEqual(len(clean_data(self.df)), len(self.df))

    def test_duplicates_are_dropped(self) -> None:
        dup = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        self.assertEqual(len(clean_data(dup)), len(self.df))

    def test_missing_values_are_dropped(self) -> None:
        bad = self.df.copy()
        bad.loc[0, self.first_col] = np.nan
        self.assertEqual(len(clean_data(bad)), len(self.df) - 1)

    def test_unknown_road_type_is_dropped(self) -> None:
        bad = self.df.copy()
        bad.loc[0, "road_type"] = "not_a_road"
        self.assertEqual(len(clean_data(bad)), len(self.df) - 1)

    def test_out_of_range_value_is_dropped(self) -> None:
        bad = self.df.copy()
        bad.loc[0, self.first_col] = self.first_hi + 1
        self.assertEqual(len(clean_data(bad)), len(self.df) - 1)

    def test_index_is_reset(self) -> None:
        bad = self.df.copy()
        bad.loc[0, "road_type"] = "not_a_road"
        cleaned = clean_data(bad)
        self.assertEqual(list(cleaned.index), list(range(len(cleaned))))


class TestLoadData(unittest.TestCase):
    """load_data must fail clearly when the dataset is missing."""

    def test_missing_file_raises_helpful_error(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            load_data(Path("definitely_missing_dataset.csv"))
        self.assertIn("generate_dataset", str(ctx.exception))

    def test_csv_round_trip(self) -> None:
        df = make_dataset(20)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            df.to_csv(path, index=False)
            loaded = load_data(path)
        self.assertEqual(len(loaded), len(df))
        self.assertEqual(set(loaded.columns), set(df.columns))


class TestFeatures(unittest.TestCase):
    """Training and prediction must see exactly the same feature layout."""

    def test_columns_match_config_in_order(self) -> None:
        x = build_features(make_dataset(50))
        self.assertEqual(list(x.columns), list(FEATURE_COLUMNS))

    def test_no_missing_values_and_row_count_preserved(self) -> None:
        df = make_dataset(50)
        x = build_features(df)
        self.assertEqual(len(x), len(df))
        self.assertFalse(x.isna().any().any())

    def test_target_is_not_a_feature(self) -> None:
        """Guard against data leakage: the label must never be an input."""
        self.assertNotIn(TARGET_COLUMN, FEATURE_COLUMNS)


class TestCandidates(unittest.TestCase):
    """The project compares exactly Decision Tree and Random Forest."""

    def test_candidate_names(self) -> None:
        self.assertEqual(set(get_candidates()), {"Decision Tree", "Random Forest"})

    def test_each_candidate_has_grid_and_probabilities(self) -> None:
        for name, (estimator, grid) in get_candidates().items():
            self.assertTrue(grid, f"{name} has an empty tuning grid")
            self.assertTrue(hasattr(estimator, "predict_proba"))


class TestEvaluationAndProbabilities(unittest.TestCase):
    """Train a small model and check metrics and probability range."""

    @classmethod
    def setUpClass(cls) -> None:
        df = clean_data(make_dataset(600))
        x, y = build_features(df), df[TARGET_COLUMN].astype(int)
        cls.x_train, cls.x_test, cls.y_train, cls.y_test = train_test_split(
            x, y, test_size=0.25, stratify=y, random_state=RANDOM_SEED)
        cls.model = RandomForestClassifier(
            n_estimators=30, random_state=RANDOM_SEED).fit(cls.x_train, cls.y_train)

    def test_metrics_are_present_and_between_zero_and_one(self) -> None:
        metrics = evaluate(self.model, self.x_test, self.y_test)
        for key in METRIC_KEYS:
            self.assertIn(key, metrics)
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)

    def test_confusion_matrix_shape_and_total(self) -> None:
        cm = evaluate(self.model, self.x_test, self.y_test)["confusion_matrix"]
        self.assertEqual(np.array(cm).shape, (2, 2))
        self.assertEqual(int(np.sum(cm)), len(self.y_test))

    def test_probabilities_are_valid(self) -> None:
        proba = self.model.predict_proba(self.x_test)[:, 1]
        self.assertTrue(np.all(proba >= 0.0) and np.all(proba <= 1.0))

    def test_model_beats_random_guessing(self) -> None:
        """A learnable problem must give ROC-AUC clearly above 0.5."""
        self.assertGreater(evaluate(self.model, self.x_test, self.y_test)["roc_auc"], 0.7)


class TestFigures(unittest.TestCase):
    """Plot helpers must write PNG files into the figures folder."""

    def test_confusion_and_importance_figures_are_saved(self) -> None:
        df = clean_data(make_dataset(200))
        model = RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED).fit(
            build_features(df), df[TARGET_COLUMN].astype(int))
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("models.train_model.FIGURES_DIR", Path(tmp)):
                plot_confusion("Random Forest", [[10, 2], [3, 15]])
                plot_importance(model)
            names = {p.name for p in Path(tmp).iterdir()}
        self.assertIn("confusion_random_forest.png", names)
        self.assertIn("feature_importance.png", names)


@unittest.skipUnless(MODEL_PATH.exists(), "Run `python -m models.train_model` first")
class TestSavedModelArtifact(unittest.TestCase):
    """Checks on the real trained model that the dashboard will load."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = joblib.load(MODEL_PATH)

    def test_artifact_contains_expected_keys(self) -> None:
        for key in ("model", "model_name", "feature_columns", "threshold"):
            self.assertIn(key, self.artifact)

    def test_saved_feature_columns_match_config(self) -> None:
        self.assertEqual(list(self.artifact["feature_columns"]), list(FEATURE_COLUMNS))

    def test_predictions_are_probabilities(self) -> None:
        x = build_features(make_dataset(100))
        proba = self.artifact["model"].predict_proba(x)[:, 1]
        self.assertTrue(np.all(proba >= 0.0) and np.all(proba <= 1.0))

    @unittest.skipUnless({"rainfall_mm", "river_level_m"} <= set(VALID_RANGES),
                         "needs rainfall_mm and river_level_m in VALID_RANGES")
    def test_wetter_conditions_do_not_lower_flood_risk(self) -> None:
        """Domain sanity check: extreme rain and river level should raise risk."""
        base = make_dataset(300)
        dry, wet = base.copy(), base.copy()
        dry["rainfall_mm"], dry["river_level_m"] = (VALID_RANGES["rainfall_mm"][0],
                                                    VALID_RANGES["river_level_m"][0])
        wet["rainfall_mm"], wet["river_level_m"] = (VALID_RANGES["rainfall_mm"][1],
                                                    VALID_RANGES["river_level_m"][1])
        model = self.artifact["model"]
        p_dry = model.predict_proba(build_features(dry))[:, 1].mean()
        p_wet = model.predict_proba(build_features(wet))[:, 1].mean()
        self.assertGreater(p_wet, p_dry)


if __name__ == "__main__":
    unittest.main()