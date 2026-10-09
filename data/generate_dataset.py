"""Generate a synthetic road-segment flood dataset for Velachery-like terrain.

No public dataset provides per-road flood labels with these features, so
labels are drawn from a documented hydrological rule (logistic risk model
plus noise). The generating equation is reported in the paper; the ML task
therefore measures *learnability* of the risk pattern, not real-world
accuracy.

Run from the project root:  python -m data.generate_dataset
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from utils.config import RANDOM_SEED, RAW_DATA_PATH, ROAD_TYPES, VALID_RANGES

logger = logging.getLogger(__name__)

# Road-type adjustment to flood log-odds: small local roads drain worst.
ROAD_TYPE_EFFECT: Dict[str, float] = {
    "residential": 0.4,
    "tertiary": 0.2,
    "secondary": 0.0,
    "primary": -0.3,
    "trunk": -0.5,
}
ROAD_TYPE_PROBS = [0.45, 0.25, 0.15, 0.10, 0.05]


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function mapping log-odds to probability."""
    return 1.0 / (1.0 + np.exp(-z))


def generate(n_samples: int = 6000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create synthetic (road segment, weather scenario) samples.

    Args:
        n_samples: number of rows to generate.
        seed: random seed for reproducibility.

    Returns:
        DataFrame with features, ``true_probability`` and binary ``flooded``.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)

    road_type = rng.choice(ROAD_TYPES, size=n_samples, p=ROAD_TYPE_PROBS)
    elevation = np.clip(rng.normal(8.0, 3.0, n_samples), *VALID_RANGES["elevation_m"])
    distance = np.clip(rng.exponential(800.0, n_samples), *VALID_RANGES["distance_from_river_m"])
    rainfall = np.clip(rng.gamma(2.0, 50.0, n_samples), *VALID_RANGES["rainfall_mm"])
    river = np.clip(0.012 * rainfall + rng.normal(0.8, 0.8, n_samples),
                    *VALID_RANGES["river_level_m"])

    # Historical flood frequency: higher for low, near-river roads.
    hist_rate = np.clip(3.0 + 0.25 * (8.0 - elevation) + 1.5 * np.exp(-distance / 500.0), 0.1, 9.0)
    hist = np.clip(rng.poisson(hist_rate), *VALID_RANGES["historical_flood_freq"])

    proximity = np.exp(-distance / 500.0)
    effect = np.array([ROAD_TYPE_EFFECT[r] for r in road_type])
    log_odds = (
        -4.0
        + 0.012 * rainfall
        + 0.55 * river * proximity
        + 0.35 * hist
        - 0.25 * (elevation - 8.0)
        + effect
        + rng.normal(0.0, 0.5, n_samples)  # unobserved factors (drainage, blockage)
    )
    prob = sigmoid(log_odds)
    flooded = rng.binomial(1, prob)

    return pd.DataFrame({
        "rainfall_mm": rainfall.round(1),
        "river_level_m": river.round(2),
        "elevation_m": elevation.round(2),
        "distance_from_river_m": distance.round(0),
        "historical_flood_freq": hist,
        "road_type": road_type,
        "true_probability": prob.round(4),
        "flooded": flooded,
    })


def main() -> None:
    """Generate the dataset and write it to ``RAW_DATA_PATH``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        df = generate()
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)
    except OSError as exc:
        logger.error("Could not write dataset: %s", exc)
        raise
    logger.info("Saved %d rows to %s (flood rate %.1f%%)",
                len(df), RAW_DATA_PATH, 100 * df["flooded"].mean())


if __name__ == "__main__":
    main()
