"""Assign ML-predicted flood probabilities to every road edge."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import networkx as nx
import pandas as pd

from utils.config import MODEL_PATH, NUMERIC_FEATURES, VALID_RANGES
from utils.features import build_features

logger = logging.getLogger(__name__)

_STATIC_EDGE_FIELDS = ["elevation_m", "distance_from_river_m",
                       "historical_flood_freq", "road_type"]


def load_model_bundle(path: Path = MODEL_PATH) -> Dict[str, Any]:
    """Load the saved model bundle produced by ``models.train_model``.

    Raises:
        FileNotFoundError: if the model has not been trained yet.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m models.train_model")
    return joblib.load(path)


def _validate_scenario(rainfall_mm: float, river_level_m: float) -> None:
    """Ensure scenario inputs lie within the trained feature ranges."""
    for name, value in (("rainfall_mm", rainfall_mm), ("river_level_m", river_level_m)):
        lo, hi = VALID_RANGES[name]
        if not lo <= value <= hi:
            raise ValueError(f"{name}={value} outside valid range [{lo}, {hi}]")


def assign_flood_risk(graph: nx.DiGraph, rainfall_mm: float, river_level_m: float,
                      bundle: Dict[str, Any]) -> None:
    """Predict flood probability for all edges and store it as ``flood_prob``.

    All edges are scored in one vectorised call (fast even for thousands of
    edges). The graph must already have static features attached.
    """
    _validate_scenario(rainfall_mm, river_level_m)
    edges = list(graph.edges(data=True))
    if not edges:
        raise ValueError("Graph has no edges")
    rows = []
    for _, _, data in edges:
        missing = [f for f in _STATIC_EDGE_FIELDS if f not in data]
        if missing:
            raise ValueError(f"Edge missing static features {missing}; "
                             "call attach_static_features first")
        rows.append({f: data[f] for f in _STATIC_EDGE_FIELDS})
    frame = pd.DataFrame(rows)
    frame["rainfall_mm"] = rainfall_mm
    frame["river_level_m"] = river_level_m
    frame = frame[NUMERIC_FEATURES + ["road_type"]]

    proba = bundle["model"].predict_proba(build_features(frame)[bundle["feature_columns"]])[:, 1]
    for (_, _, data), p in zip(edges, proba):
        data["flood_prob"] = float(p)
    logger.info("Assigned flood risk to %d edges (mean %.3f)", len(edges), float(proba.mean()))
