"""Report exactly which DEM and which waterway layer the experiment used.

    python -m experiments.audit_inputs

DEM part: prints file date, size, resolution, CRS, bounds, nodata, embedded metadata tags and the
elevation range, so the product (and its resolution) can be identified and written in the paper.
Waterway part: repeats the OpenStreetMap query used by ``fetch_river_lines`` and counts what it
returns, split into river/canal/stream/drain lines and lake/wetland polygons, so the paper's
description of the waterway layer can be checked against the data. It also lists any cached
waterway files under ``data/``.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import Counter
from pathlib import Path

from routing.graph_builder import fetch_river_lines
from utils.config import DATA_DIR, DEM_PATH, PLACE_NAME

# Keep identical to routing/graph_builder.py::fetch_river_lines
WATER_TAGS = {"waterway": ["river", "canal", "stream", "drain"], "natural": ["water", "wetland"]}
BUFFER_M = 2000.0


def audit_dem(path: Path = DEM_PATH) -> None:
    print("=== DEM ===")
    if not path.exists():
        print(f"No DEM file at {path}")
        return
    import numpy as np
    import rasterio
    stat = path.stat()
    print(f"file: {path}\nsize: {stat.st_size / 1024:.0f} KB   modified: {dt.datetime.fromtimestamp(stat.st_mtime):%Y-%m-%d %H:%M}")
    with rasterio.open(path) as src:
        res_x, res_y = src.res
        mid_lat = (src.bounds.top + src.bounds.bottom) / 2
        metres = res_y * 111_320
        print(f"CRS: {src.crs}   size: {src.width} x {src.height} px   dtype: {src.dtypes[0]}   nodata: {src.nodata}")
        print(f"bounds (W,S,E,N): {src.bounds.left:.4f}, {src.bounds.bottom:.4f}, {src.bounds.right:.4f}, {src.bounds.top:.4f}")
        print(f"pixel size: {res_x:.6f} x {res_y:.6f} deg  (about {metres:.0f} m north-south, "
              f"{res_x * 111_320 * math.cos(math.radians(mid_lat)):.0f} m east-west at latitude {mid_lat:.2f})")
        print(f"dataset tags: {src.tags()}\nband tags: {src.tags(1)}")
        data = src.read(1).astype("float64")
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
    print(f"elevation: min {np.nanmin(data):.1f} m, median {np.nanmedian(data):.2f} m, max {np.nanmax(data):.1f} m")
    print("Pixel size ~0.000278 deg = 1 arc-second (30 m class: Copernicus GLO-30, SRTM, NASADEM, ALOS); "
          "~0.000833 deg = 3 arc-seconds (90 m class). The file alone may not name the product: check the "
          "OpenTopography download page or confirmation e-mail, and record dataset name, resolution and date.")


def audit_waterways(place: str = PLACE_NAME) -> None:
    print("\n=== Waterway layer ===")
    cached = [p for p in DATA_DIR.rglob("*") if p.is_file() and any(k in p.name.lower() for k in ("river", "water", "wetland", "lake"))]
    print("cached water-related files under data/:", [str(p.relative_to(DATA_DIR)) for p in cached] or "none")
    try:
        import osmnx as ox
        west, south, east, north = [float(v) for v in ox.geocode_to_gdf(place).total_bounds]
        margin = BUFFER_M / 111_000.0
        gdf = ox.features_from_bbox(bbox=(west - margin, south - margin, east + margin, north + margin), tags=WATER_TAGS)
    except Exception as exc:
        print(f"OpenStreetMap query failed ({exc}). The layer could not be audited.")
        return
    kinds = Counter()
    names = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type in ("Polygon", "MultiPolygon"):
            kinds[f"polygon natural={row.get('natural')}"] += 1
        else:
            kinds[f"line waterway={row.get('waterway')}"] += 1
        name = row.get("name")
        if isinstance(name, str):
            names.append(name)
    print(f"features returned by the query: {len(gdf)}")
    for kind, n in sorted(kinds.items()):
        print(f"  {kind}: {n}")
    marsh = sorted({n for n in names if "pallikaranai" in n.lower()})
    print("named Pallikaranai features:", marsh or "none")
    lines = fetch_river_lines(place)
    print(f"fetch_river_lines() returned: {'None (synthetic river would be used)' if lines is None else str(len(lines)) + ' polylines'}")
    has_poly = any(k.startswith("polygon") for k in kinds)
    print("lakes/wetlands present in the layer:", "yes" if has_poly else "NO - Section IV-B must not claim them")


if __name__ == "__main__":
    audit_dem()
    audit_waterways()
