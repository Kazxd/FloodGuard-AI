"""Interactive Folium map for FloodGuard Lite.

Colours: green = safe, yellow = moderate risk, red = flooded,
blue = recommended evacuation route (dashed grey = plain shortest-distance
route, drawn for comparison). Start, destination and shelters get markers.

Run a self-contained demo from the project root:
    python -m dashboard.map_builder      ->  reports/demo_map.html
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import folium
import networkx as nx

from routing.algorithms import SearchResult, astar
from routing.cost import apply_costs
from routing.graph_builder import (attach_static_features, build_grid_graph,
                                   fetch_river_lines, load_osm_graph)
from routing.risk import assign_flood_risk, load_model_bundle
from routing.shelters import find_shelters, node_elevation, route_to_best_shelter
from utils.config import (COLOR_FLOODED, COLOR_MODERATE, COLOR_ROUTE, COLOR_SAFE,
                          DEFAULT_FLOOD_WEIGHT, REPORTS_DIR, RISK_FLOODED_MIN,
                          RISK_SAFE_MAX)

logger = logging.getLogger(__name__)

COLOR_BASELINE = "#424242"
Coord = Tuple[float, float]


def risk_class(probability: float) -> str:
    """Classify a flood probability as ``safe``, ``moderate`` or ``flooded``."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability {probability} outside [0, 1]")
    if probability < RISK_SAFE_MAX:
        return "safe"
    return "moderate" if probability < RISK_FLOODED_MIN else "flooded"


RISK_COLORS: Dict[str, str] = {
    "safe": COLOR_SAFE, "moderate": COLOR_MODERATE, "flooded": COLOR_FLOODED}


def _node_coord(graph: nx.DiGraph, node: int) -> Coord:
    """Return (lat, lon) of a node."""
    return graph.nodes[node]["y"], graph.nodes[node]["x"]


def _legend_html() -> str:
    """Small fixed HTML legend explaining the colour scheme."""
    rows = [(COLOR_SAFE, f"Safe (&lt; {RISK_SAFE_MAX:.0%})"),
            (COLOR_MODERATE, f"Moderate ({RISK_SAFE_MAX:.0%}&ndash;{RISK_FLOODED_MIN:.0%})"),
            (COLOR_FLOODED, f"Flooded (&ge; {RISK_FLOODED_MIN:.0%})"),
            (COLOR_ROUTE, "Recommended route"),
            (COLOR_BASELINE, "Shortest-distance route")]
    items = "".join(
        f'<div><span style="display:inline-block;width:18px;height:4px;'
        f'background:{c};margin-right:6px;vertical-align:middle"></span>{t}</div>'
        for c, t in rows)
    return ('<div style="position:fixed;bottom:24px;left:24px;z-index:9999;'
            'background:white;padding:8px 12px;border:1px solid #999;'
            'border-radius:6px;font-size:12px;line-height:1.6">'
            f'<b>FloodGuard Lite</b>{items}</div>')


def _add_roads(graph: nx.DiGraph, fmap: folium.Map) -> None:
    """Draw each physical road once, coloured by flood risk, in toggleable layers."""
    layers = {
        "safe": folium.FeatureGroup(name="Safe roads"),
        "moderate": folium.FeatureGroup(name="Moderate-risk roads"),
        "flooded": folium.FeatureGroup(name="Flooded roads"),
    }
    seen: set[FrozenSet[int]] = set()
    for u, v, data in graph.edges(data=True):
        key = frozenset((u, v))
        if key in seen:
            continue
        seen.add(key)
        if "flood_prob" not in data:
            raise ValueError("Edges lack flood_prob; call assign_flood_risk first")
        prob = data["flood_prob"]
        if graph.has_edge(v, u):  # same road, other direction: show the worse value
            prob = max(prob, graph[v][u]["flood_prob"])
        kind = risk_class(prob)
        folium.PolyLine(
            [_node_coord(graph, u), _node_coord(graph, v)],
            color=RISK_COLORS[kind], weight=4 if kind == "flooded" else 3, opacity=0.8,
            tooltip=f"{data.get('road_type', 'road').title()} road | "
                    f"flood risk {prob:.0%} | {data['length']:.0f} m",
        ).add_to(layers[kind])
    for layer in layers.values():
        layer.add_to(fmap)


def _add_route(graph: nx.DiGraph, fmap: folium.Map, result: SearchResult,
               label: str, color: str, weight: int, dashed: bool = False) -> None:
    """Draw a route polyline with a metrics tooltip."""
    coords: List[Coord] = [_node_coord(graph, n) for n in result.path]
    folium.PolyLine(
        coords, color=color, weight=weight, opacity=0.9,
        dash_array="10 8" if dashed else None,
        tooltip=f"{label}: {result.distance_m / 1000:.2f} km, "
                f"mean risk {result.mean_risk:.0%}, max risk {result.max_risk:.0%}",
    ).add_to(fmap)


def build_map(graph: nx.DiGraph, route: Optional[SearchResult] = None,
              baseline: Optional[SearchResult] = None, start: Optional[int] = None,
              destination: Optional[int] = None, shelters: Sequence[int] = (),
              zoom: int = 14) -> folium.Map:
    """Assemble the full interactive map.

    Args:
        graph: graph with ``flood_prob`` on every edge.
        route: recommended (flood-aware) route, drawn in blue.
        baseline: plain shortest-distance route, drawn dashed grey.
        start / destination: node ids for the endpoint markers.
        shelters: node ids of shelters.
        zoom: initial zoom (the map is then fitted to the graph bounds).
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot draw an empty graph")
    lats = [d["y"] for _, d in graph.nodes(data=True)]
    lons = [d["x"] for _, d in graph.nodes(data=True)]
    fmap = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)],
                      zoom_start=zoom, tiles="cartodbpositron")

    _add_roads(graph, fmap)
    if baseline is not None:
        _add_route(graph, fmap, baseline, "Shortest-distance route", COLOR_BASELINE, 5, True)
    if route is not None:
        _add_route(graph, fmap, route, "Recommended route", COLOR_ROUTE, 7)

    for index, node in enumerate(shelters, start=1):
        if node == destination:
            continue  # destination marker takes priority
        folium.Marker(_node_coord(graph, node), tooltip=f"Shelter {index}",
                      icon=folium.Icon(color="purple", icon="home", prefix="fa")).add_to(fmap)
    if start is not None:
        folium.Marker(_node_coord(graph, start), tooltip="Start",
                      icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(fmap)
    if destination is not None:
        folium.Marker(_node_coord(graph, destination), tooltip="Destination",
                      icon=folium.Icon(color="red", icon="flag", prefix="fa")).add_to(fmap)

    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    folium.LayerControl(collapsed=True).add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(_legend_html()))
    return fmap


def save_map(fmap: folium.Map, path: Path) -> Path:
    """Write the map to an HTML file and return its path."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fmap.save(str(path))
    except OSError as exc:
        logger.error("Could not save map to %s: %s", path, exc)
        raise
    return path


def main() -> None:
    """Build a demo evacuation scenario and save it as an HTML map."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        graph = load_osm_graph()
        rivers = fetch_river_lines()
    except RuntimeError as exc:
        logger.warning("%s -- falling back to the synthetic grid graph.", exc)
        graph, rivers = build_grid_graph(), None

    attach_static_features(graph, rivers)
    assign_flood_risk(graph, rainfall_mm=200.0, river_level_m=4.0,
                      bundle=load_model_bundle())

    shelters = find_shelters(graph)
    elevation = node_elevation(graph)
    start = min(elevation, key=elevation.get)  # lowest-lying intersection

    apply_costs(graph, DEFAULT_FLOOD_WEIGHT)
    safe = route_to_best_shelter(graph, start, shelters)
    destination = safe.path[-1]
    apply_costs(graph, 0.0)  # plain distance for the comparison route
    baseline = astar(graph, start, destination)

    fmap = build_map(graph, safe, baseline, start, destination, shelters)
    out = save_map(fmap, REPORTS_DIR / "demo_map.html")
    logger.info("Safe route: %.2f km, mean risk %.1f%% | Shortest: %.2f km, mean risk %.1f%%",
                safe.distance_m / 1000, 100 * safe.mean_risk,
                baseline.distance_m / 1000, 100 * baseline.mean_risk)
    logger.info("Map saved to %s", out)


if __name__ == "__main__":
    main()
