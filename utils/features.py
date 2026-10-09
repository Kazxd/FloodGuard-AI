"""Feature engineering shared by training and inference.

Using one function in both places prevents train/serve skew: the model
sees exactly the same columns, in the same order, at prediction time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.config import FEATURE_COLUMNS, NUMERIC_FEATURES, ROAD_TYPES

PROXIMITY_SCALE_M: float = 500.0  # distance at which river influence decays by 1/e


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw road-segment records into the model feature matrix.

    Args:
        df: DataFrame with the numeric columns in ``NUMERIC_FEATURES``
            plus a categorical ``road_type`` column.

    Returns:
        DataFrame with columns exactly ``FEATURE_COLUMNS``.

    Raises:
        ValueError: if required columns are missing or a road type is unknown.
    """
    required = NUMERIC_FEATURES + ["road_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    unknown = set(df["road_type"].unique()) - set(ROAD_TYPES)
    if unknown:
        raise ValueError(f"Unknown road types: {sorted(unknown)}")

    out = df[NUMERIC_FEATURES].astype(float).copy()

    # Engineered features encode domain knowledge the trees would otherwise
    # have to discover through many splits.
    out["river_proximity"] = np.exp(-df["distance_from_river_m"] / PROXIMITY_SCALE_M)
    out["river_exposure"] = df["river_level_m"] * out["river_proximity"]
    out["rain_per_elevation"] = df["rainfall_mm"] / (1.0 + df["elevation_m"])

    # One-hot encode road type with a fixed category order.
    for road_type in ROAD_TYPES:
        out[f"road_{road_type}"] = (df["road_type"] == road_type).astype(int)

    return out[FEATURE_COLUMNS]
