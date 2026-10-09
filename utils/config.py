"""Central configuration for FloodGuard Lite.

Every path, constant and valid range lives here so that all modules
(data generation, training, routing, dashboard) stay consistent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_PATH: Path = DATA_DIR / "raw" / "road_segments.csv"
MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_PATH: Path = MODELS_DIR / "flood_model.joblib"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
METRICS_PATH: Path = REPORTS_DIR / "metrics.json"

RANDOM_SEED: int = 42
PLACE_NAME: str = "Velachery, Chennai, Tamil Nadu, India"

ROAD_TYPES: List[str] = ["residential", "tertiary", "secondary", "primary", "trunk"]

NUMERIC_FEATURES: List[str] = [
    "rainfall_mm",              # 24h rainfall (mm)
    "river_level_m",            # river level above normal (m)
    "elevation_m",              # road elevation above sea level (m)
    "distance_from_river_m",    # distance to nearest river/water body (m)
    "historical_flood_freq",    # flood events per 10 years
]

# Valid (min, max) per numeric feature; used for cleaning and UI sliders.
VALID_RANGES: Dict[str, Tuple[float, float]] = {
    "rainfall_mm": (0.0, 400.0),
    "river_level_m": (0.0, 8.0),
    "elevation_m": (0.0, 25.0),
    "distance_from_river_m": (0.0, 5000.0),
    "historical_flood_freq": (0.0, 10.0),
}

ENGINEERED_FEATURES: List[str] = [
    "river_proximity",
    "river_exposure",
    "rain_per_elevation",
]

FEATURE_COLUMNS: List[str] = (
    NUMERIC_FEATURES
    + ENGINEERED_FEATURES
    + [f"road_{rt}" for rt in ROAD_TYPES]
)

TARGET_COLUMN: str = "flooded"
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5
DECISION_THRESHOLD: float = 0.5

# ---- Routing (Phase 3) ----
GRAPH_PATH: Path = DATA_DIR / "processed" / "velachery.graphml"
DEFAULT_FLOOD_WEIGHT: float = 5.0   # penalty strength for flood risk

# ---- Map display (Phase 4) ----
RISK_SAFE_MAX: float = 0.3
RISK_FLOODED_MIN: float = 0.7
COLOR_SAFE: str = "#2e7d32"
COLOR_MODERATE: str = "#f9a825"
COLOR_FLOODED: str = "#c62828"
COLOR_ROUTE: str = "#1565c0"
