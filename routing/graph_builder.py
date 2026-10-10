"""Road-network graph construction for FloodGuard Lite.

Nodes are intersections (``x`` = longitude, ``y`` = latitude). Edges are
road segments carrying ``length`` (m), ``road_type`` and the static flood
features (elevation, distance from river, historical flood frequency).

OSMnx is imported lazily so the rest of the package (and the unit tests)
work offline using :func:`build_grid_graph`.
"""
from __future__ import annotations

import ast
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from utils.config import GRAPH_PATH, PLACE_NAME, RANDOM_SEED

logger = logging.getLogger(__name__)

LonLatLine = Sequence[Tuple[float, float]]
EARTH_RADIUS_M: float = 6_371_000.0

_HIGHWAY_MAP = {
    "residential": "residential", "living_street": "residential",
    "unclassified": "residential", "service": "residential",
    "tertiary": "tertiary", "secondary": "secondary", "primary": "primary",
    "trunk": "trunk", "motorway": "trunk",
}


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two lat/lon points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _normalise_highway(value: Any) -> str:
    """Map an OSM ``highway`` tag (str, list or stringified list) to ROAD_TYPES."""
    if isinstance(value, str) and value.startswith("["):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    tag = str(value).replace("_link", "")
    return _HIGHWAY_MAP.get(tag, "residential")


def simplify_to_digraph(multi: nx.MultiDiGraph) -> nx.DiGraph:
    """Collapse parallel edges (keep the shortest) and keep the largest
    strongly connected component so every node pair is routable."""
    graph = nx.DiGraph()
    for node, data in multi.nodes(data=True):
        graph.add_node(node, x=float(data["x"]), y=float(data["y"]))
    for u, v, data in multi.edges(data=True):
        length = float(data.get("length", 0.0))
        if u == v or length <= 0:
            continue
        if graph.has_edge(u, v) and graph[u][v]["length"] <= length:
            continue
        graph.add_edge(u, v, length=length, road_type=_normalise_highway(data.get("highway")))
    if graph.number_of_nodes() == 0:
        raise ValueError("Graph is empty after simplification")
    core = max(nx.strongly_connected_components(graph), key=len)
    return graph.subgraph(core).copy()


def load_osm_graph(place: str = PLACE_NAME, cache_path: Path = GRAPH_PATH) -> nx.DiGraph:
    """Download (first run) or load the cached drivable road graph.

    Raises:
        RuntimeError: if OSMnx is missing or the download fails.
    """
    try:
        import osmnx as ox
    except ImportError as exc:
        raise RuntimeError("osmnx is not installed: pip install osmnx") from exc
    try:
        if cache_path.exists():
            multi = ox.load_graphml(cache_path)
        else:
            logger.info("Downloading road network for %s ...", place)
            multi = ox.graph_from_place(place, network_type="drive")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            ox.save_graphml(multi, cache_path)
    except Exception as exc:  # network / geocoding failures vary widely
        raise RuntimeError(f"Could not obtain OSM graph for '{place}': {exc}") from exc
    return simplify_to_digraph(multi)


def fetch_river_lines(place: str = PLACE_NAME) -> Optional[List[List[Tuple[float, float]]]]:
    """Fetch waterway geometries from OSM; return None if unavailable."""
    try:
        import osmnx as ox
        gdf = ox.features_from_place(
            place, tags={"waterway": ["river", "canal", "stream", "drain"]})
    except Exception as exc:
        logger.warning("Could not fetch waterways (%s); using synthetic river.", exc)
        return None
    lines: List[List[Tuple[float, float]]] = []
    for geom in gdf.geometry:
        parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        lines.extend([list(p.coords) for p in parts if p.geom_type == "LineString"])
    return lines or None


def build_grid_graph(rows: int = 15, cols: int = 15, spacing_m: float = 120.0,
                     origin: Tuple[float, float] = (12.975, 80.215)) -> nx.DiGraph:
    """Build a bidirectional synthetic grid near Velachery for offline use/tests.

    Every 5th row/column is a ``primary`` road, every 3rd a ``tertiary``;
    the rest are ``residential``.
    """
    if rows < 2 or cols < 2:
        raise ValueError("rows and cols must be >= 2")
    lat0, lon0 = origin
    dlat = spacing_m / 110_540.0
    dlon = spacing_m / (111_320.0 * math.cos(math.radians(lat0)))
    graph = nx.DiGraph()
    for r in range(rows):
        for c in range(cols):
            graph.add_node(r * cols + c, x=lon0 + c * dlon, y=lat0 + r * dlat)

    def road_type(index: int) -> str:
        return "primary" if index % 5 == 0 else "tertiary" if index % 3 == 0 else "residential"

    def link(a: int, b: int, kind: str) -> None:
        length = haversine_m(graph.nodes[a]["y"], graph.nodes[a]["x"],
                             graph.nodes[b]["y"], graph.nodes[b]["x"])
        graph.add_edge(a, b, length=length, road_type=kind)
        graph.add_edge(b, a, length=length, road_type=kind)

    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            if c + 1 < cols:
                link(node, node + 1, road_type(r))
            if r + 1 < rows:
                link(node, node + cols, road_type(c))
    return graph


def _to_local_m(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Equirectangular projection to metres around (lon0, lat0)."""
    x = (np.asarray(lon) - lon0) * 111_320.0 * math.cos(math.radians(lat0))
    y = (np.asarray(lat) - lat0) * 110_540.0
    return x, y


def _distance_to_polyline(px: float, py: float, line: np.ndarray) -> float:
    """Minimum distance from a point to a polyline (array of shape (n, 2))."""
    if len(line) == 1:
        return float(np.hypot(line[0, 0] - px, line[0, 1] - py))
    a, b = line[:-1], line[1:]
    ab = b - a
    denom = np.maximum((ab ** 2).sum(axis=1), 1e-12)
    t = np.clip(((px - a[:, 0]) * ab[:, 0] + (py - a[:, 1]) * ab[:, 1]) / denom, 0.0, 1.0)
    cx, cy = a[:, 0] + t * ab[:, 0], a[:, 1] + t * ab[:, 1]
    return float(np.hypot(cx - px, cy - py).min())


def attach_static_features(graph: nx.DiGraph,
                           river_lines: Optional[Sequence[LonLatLine]] = None,
                           seed: int = RANDOM_SEED) -> None:
    """Add elevation, river distance and historical flood frequency to edges."""
    xs = np.array([d["x"] for _, d in graph.nodes(data=True)])
    ys = np.array([d["y"] for _, d in graph.nodes(data=True)])
    lon0, lat0 = float(xs.mean()), float(ys.mean())
 
    if river_lines:
        lines = [np.column_stack(_to_local_m(*zip(*ln), lon0, lat0)) for ln in river_lines]
    else:
        south = (ys.min() - lat0) * 110_540.0
        west, east = _to_local_m(np.array([xs.min(), xs.max()]), np.array([lat0, lat0]), lon0, lat0)[0]
        lines = [np.array([[west, south], [east, south]])]
 
    rng = np.random.default_rng(seed)
    # A road drawn in both directions is ONE physical road, so its features are
    # computed once per undirected pair and shared by both directed edges.
    computed: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
    for u, v, data in graph.edges(data=True):
        key = (min(u, v), max(u, v))
        if key not in computed:
            mx, my = _to_local_m((graph.nodes[u]["x"] + graph.nodes[v]["x"]) / 2,
                                 (graph.nodes[u]["y"] + graph.nodes[v]["y"]) / 2, lon0, lat0)
            dist = min(_distance_to_polyline(float(mx), float(my), ln) for ln in lines)
            elev = 5.0 + 0.004 * dist + 1.2 * math.sin(mx / 600) * math.cos(my / 700) \
                + rng.normal(0, 0.3)
            elev = float(np.clip(elev, 0.5, 25.0))
            rate = float(np.clip(3.0 + 0.25 * (8.0 - elev)
                                 + 1.5 * math.exp(-dist / 500.0), 0.1, 9.0))
            computed[key] = (float(np.clip(dist, 0.0, 5000.0)), elev,
                             float(min(rng.poisson(rate), 10)))
        (data["distance_from_river_m"], data["elevation_m"],
         data["historical_flood_freq"]) = computed[key]


def nearest_node(graph: nx.DiGraph, lat: float, lon: float) -> int:
    """Return the graph node closest to a lat/lon coordinate."""
    return min(graph.nodes, key=lambda n: haversine_m(
        lat, lon, graph.nodes[n]["y"], graph.nodes[n]["x"]))
