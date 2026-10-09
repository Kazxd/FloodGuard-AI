"""Train and compare Decision Tree vs Random Forest flood classifiers.

Pipeline: load -> clean -> feature engineering -> stratified split ->
cross-validated tuning -> evaluation -> save best model and reports.

Run from the project root:  python -m models.train_model
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend: no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from utils.config import (CV_FOLDS, DECISION_THRESHOLD, FEATURE_COLUMNS, FIGURES_DIR,
                          METRICS_PATH, MODEL_PATH, RANDOM_SEED, RAW_DATA_PATH,
                          ROAD_TYPES, TARGET_COLUMN, TEST_SIZE, VALID_RANGES)
from utils.features import build_features

logger = logging.getLogger(__name__)


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset CSV.

    Raises:
        FileNotFoundError: if the dataset has not been generated yet.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m data.generate_dataset")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, nulls, unknown road types and out-of-range rows."""
    before = len(df)
    df = df.drop_duplicates().dropna(subset=list(VALID_RANGES) + ["road_type", TARGET_COLUMN])
    df = df[df["road_type"].isin(ROAD_TYPES)]
    for col, (lo, hi) in VALID_RANGES.items():
        df = df[df[col].between(lo, hi)]
    logger.info("Cleaning kept %d of %d rows", len(df), before)
    return df.reset_index(drop=True)


def explore(df: pd.DataFrame) -> None:
    """Log basic exploratory statistics and save a class-balance figure."""
    logger.info("Class balance:\n%s", df[TARGET_COLUMN].value_counts(normalize=True).round(3))
    logger.info("Summary:\n%s", df.describe().round(2).T)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    df[TARGET_COLUMN].value_counts().sort_index().plot.bar(ax=ax, color=["#2e7d32", "#c62828"])
    ax.set_xticklabels(["Not flooded", "Flooded"], rotation=0)
    ax.set_title("Class balance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=150)
    plt.close(fig)


def get_candidates() -> Dict[str, Tuple[Any, Dict[str, list]]]:
    """Return the two candidate models with their tuning grids."""
    return {
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_SEED),
            {"max_depth": [3, 5, 8, 12], "min_samples_leaf": [1, 10, 30]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
            {"n_estimators": [100, 300], "max_depth": [8, 12, None],
             "min_samples_leaf": [1, 5]},
        ),
    }


def evaluate(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Compute classification metrics on the held-out test set."""
    proba = model.predict_proba(x_test)[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "brier": brier_score_loss(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def plot_confusion(name: str, cm: list) -> None:
    """Save a confusion-matrix figure for one model."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(np.array(cm), display_labels=["Safe", "Flooded"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"confusion_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)


def plot_importance(model: Any) -> None:
    """Save a feature-importance bar chart for a fitted tree-based model."""
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    imp.plot.barh(ax=ax, color="#1565c0")
    ax.set_title("Feature importance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the full training pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        df = clean_data(load_data())
    except (FileNotFoundError, pd.errors.ParserError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1)

    explore(df)
    x = build_features(df)
    y = df[TARGET_COLUMN].astype(int)

    # Stratified split keeps the flood/safe ratio equal in train and test.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)

    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results: Dict[str, Dict[str, Any]] = {}
    fitted: Dict[str, Any] = {}
    for name, (estimator, grid) in get_candidates().items():
        logger.info("Tuning %s ...", name)
        search = GridSearchCV(estimator, grid, scoring="f1", cv=cv, n_jobs=-1)
        search.fit(x_train, y_train)
        fitted[name] = search.best_estimator_
        results[name] = evaluate(search.best_estimator_, x_test, y_test)
        results[name]["best_params"] = search.best_params_
        results[name]["cv_f1"] = float(search.best_score_)
        plot_confusion(name, results[name]["confusion_matrix"])
        logger.info("%s: F1=%.3f AUC=%.3f", name, results[name]["f1"], results[name]["roc_auc"])

    # Select by ROC-AUC: routing consumes probabilities, not hard labels.
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    if hasattr(fitted[best_name], "feature_importances_"):
        plot_importance(fitted[best_name])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump({"model": fitted[best_name], "model_name": best_name,
                     "feature_columns": FEATURE_COLUMNS,
                     "threshold": DECISION_THRESHOLD}, MODEL_PATH)
        with open(METRICS_PATH, "w", encoding="utf-8") as fh:
            json.dump({"best_model": best_name, "results": results}, fh, indent=2, default=str)
    except OSError as exc:
        logger.error("Could not save artifacts: %s", exc)
        raise SystemExit(1)
    logger.info("Best model: %s (saved to %s)", best_name, MODEL_PATH)


if __name__ == "__main__":
    main()
