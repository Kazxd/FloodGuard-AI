"""Shared setup for the FloodGuard Lite audit experiments.

Imports the UNMODIFIED repository at ~/FloodGuard-AI (read-only) and builds:
  * the Velachery DiGraph with static features (real DEM, OSM waterways from the
    repository's committed Overpass cache),
  * an undirected edge table with geometry,
  * observed 2015 / 2017 crowdsourced flood labels (osm-in/flood-map).
Nothing in the repository is written to.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString

import os

REPO = Path(os.environ.get("FLOODGUARD_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))

from routing.graph_builder import attach_static_features, simplify_to_digraph  # noqa: E402
from routing.risk import assign_flood_risk, load_model_bundle  # noqa: E402

SCRATCH = Path("/tmp/claude-0/-home-claude/28361c35-759c-54fc-9377-e0929e3128d3/scratchpad")
FLOODMAP = Path(os.environ.get("FLOODMAP_DATA", SCRATCH / "fm" / "flood-map" / "data"))
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)
UTM = "EPSG:32644"
SCENARIOS = {"moderate": (100.0, 2.0), "heavy": (200.0, 4.0), "extreme": (300.0, 6.0)}


def cached_waterways():
    """Waterway polylines from the repository's committed Overpass cache."""
    lines = []
    for f in sorted((REPO / "cache").glob("*.json")):
        d = json.load(open(f))
        if not isinstance(d, dict):
            continue
        nodes = {e["id"]: (e["lon"], e["lat"]) for e in d["elements"] if e["type"] == "node"}
        for e in d["elements"]:
            t = e.get("tags", {})
            if e["type"] == "way" and (t.get("waterway") or t.get("natural") in ("water", "wetland")):
                pts = [nodes[n] for n in e["nodes"] if n in nodes]
                if len(pts) >= 2:
                    lines.append(pts)
    # the two cache files with waterways are identical queries; dedupe
    uniq = {tuple(map(tuple, ln)) for ln in lines}
    return [list(u) for u in uniq]


def geojson_waterways(path):
    """Waterway polylines from a saved GeoJSON; polygons become outlines (as in the repo)."""
    lines = []
    for geom in gpd.read_file(path).to_crs("EPSG:4326").geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type in ("Polygon", "MultiPolygon"):
            geom = geom.boundary
        parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        lines.extend([list(p.coords) for p in parts if p.geom_type == "LineString"])
    return lines


def waterways():
    """FLOODGUARD_WATERWAYS=<file.geojson> selects a saved layer; default is the repo cache."""
    path = os.environ.get("FLOODGUARD_WATERWAYS")
    return geojson_waterways(path) if path else cached_waterways()


def build_graph(hist_seed: int = 42):
    multi = ox.load_graphml(REPO / "data/processed/velachery.graphml")
    g = simplify_to_digraph(multi)
    attach_static_features(g, river_lines=waterways(), seed=hist_seed,
                           dem_path=REPO / "data/external/velachery_dem.tif")
    return multi, g


def edge_table(multi, g) -> gpd.GeoDataFrame:
    """One row per undirected physical road in g, with geometry (UTM)."""
    _, e = ox.graph_to_gdfs(multi)
    e = e.reset_index()
    geom = {}
    for r in e.itertuples():
        k = (min(r.u, r.v), max(r.u, r.v))
        if k not in geom or r.length < geom[k][0]:
            geom[k] = (r.length, r.geometry)
    rows = []
    for u, v, d in g.edges(data=True):
        k = (min(u, v), max(u, v))
        if any(r["key"] == k for r in rows[-1:]):
            pass
        rows.append({"key": k, **{f: d[f] for f in ("length", "road_type", "elevation_m",
                                                     "distance_from_river_m",
                                                     "historical_flood_freq")},
                     "geometry": geom.get(k, (0, LineString([(g.nodes[u]["x"], g.nodes[u]["y"]),
                                                             (g.nodes[v]["x"], g.nodes[v]["y"])])))[1]})
    df = pd.DataFrame(rows).drop_duplicates("key").reset_index(drop=True)
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").to_crs(UTM)


def flood_lines(fname: str) -> gpd.GeoDataFrame:
    df = gpd.read_file(FLOODMAP / fname)
    df["is_flooded"] = df["is_flooded"].astype(str).str.lower().isin(["true", "1"])
    return df.to_crs(UTM)


def label_edges(edges: gpd.GeoDataFrame, lines: gpd.GeoDataFrame, buf_m: float = 15.0,
                min_frac: float = 0.5) -> np.ndarray:
    """1 if >= min_frac of the edge's length lies within buf_m of a flooded street."""
    fl = lines[lines["is_flooded"]]
    if fl.empty:
        return np.zeros(len(edges), int)
    zone = fl.buffer(buf_m).union_all()
    inter = edges.geometry.intersection(zone).length
    return (inter / edges.geometry.length.clip(lower=1e-6) >= min_frac).astype(int).values


def score_rf(g: nx.DiGraph, edges: gpd.GeoDataFrame, rain: float, river: float, bundle=None):
    bundle = bundle or load_model_bundle(REPO / "models/flood_model.joblib")
    assign_flood_risk(g, rain, river, bundle)
    p = {(min(u, v), max(u, v)): d["flood_prob"] for u, v, d in g.edges(data=True)}
    return np.array([p[k] for k in edges["key"]])
