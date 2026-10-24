"""Real terrain elevation for FloodGuard Lite from a Digital Elevation Model.

A DEM is a raster (GeoTIFF) where every pixel stores a height above sea
level. This module can

* download the small window covering the study area from the free
  Copernicus GLO-30 DEM (30 m pixels, hosted on AWS without a login) or crop
  a DEM file you downloaded yourself;
* sample the height at any longitude/latitude with bilinear interpolation.

Run from the project root to create the local DEM file once:

    python -m routing.elevation                 # download Copernicus tile
    python -m routing.elevation --from-file my_dem.tif   # crop your own DEM

``rasterio`` is imported lazily so the rest of the package still works
without it.
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from utils.config import DEM_PATH

logger = logging.getLogger(__name__)

Bounds = Tuple[float, float, float, float]  # (west, south, east, north) in degrees
COPERNICUS_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def tile_name(lat: float, lon: float) -> str:
    """Name of the 1x1 degree Copernicus GLO-30 tile containing a point."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    lat_i, lon_i = abs(math.floor(lat)), abs(math.floor(lon))
    return f"Copernicus_DSM_COG_10_{ns}{lat_i:02d}_00_{ew}{lon_i:03d}_00_DEM"


def tile_url(lat: float, lon: float) -> str:
    """HTTPS address of the Copernicus GLO-30 tile containing a point."""
    name = tile_name(lat, lon)
    return f"{COPERNICUS_BASE}/{name}/{name}.tif"


def download_dem(bounds: Bounds, out_path: Path = DEM_PATH,
                 source: Optional[Union[str, Path]] = None) -> Path:
    """Save the part of a DEM that covers ``bounds`` to ``out_path``.

    Args:
        bounds: (west, south, east, north) in degrees.
        out_path: where to write the cropped GeoTIFF.
        source: URL or file path of the DEM. Defaults to the Copernicus
            GLO-30 tile that contains ``bounds`` (must be a single tile).

    Raises:
        RuntimeError: if rasterio is missing or the DEM cannot be read.
        ValueError: if ``bounds`` are invalid or span more than one tile.
    """
    west, south, east, north = bounds
    if not (west < east and south < north):
        raise ValueError(f"Invalid bounds {bounds}: need west<east and south<north")
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError as exc:
        raise RuntimeError("rasterio is not installed: pip install rasterio") from exc

    if source is None:
        eps = 1e-9
        corners = {tile_name(lat, lon) for lat in (south, north - eps)
                   for lon in (west, east - eps)}
        if len(corners) != 1:
            raise ValueError("Study area spans several Copernicus tiles; download "
                             "a DEM yourself and use --from-file.")
        source = tile_url(south, west)

    try:
        with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            with rasterio.open(str(source)) as src:
                window = from_bounds(west, south, east, north, transform=src.transform)
                window = window.round_offsets().round_lengths()
                data = src.read(1, window=window)
                profile = {
                    "driver": "GTiff", "height": data.shape[0], "width": data.shape[1],
                    "count": 1, "dtype": "float32", "crs": src.crs,
                    "transform": src.window_transform(window), "nodata": src.nodata,
                    "compress": "deflate",
                }
    except Exception as exc:  # network / format errors vary widely
        raise RuntimeError(f"Could not read DEM from {source}: {exc}") from exc

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data.astype("float32"), 1)
    logger.info("Saved %dx%d DEM window to %s", data.shape[1], data.shape[0], out_path)
    return out_path


class DemSampler:
    """Look up terrain height (metres) at longitude/latitude points.

    Uses bilinear interpolation between the four nearest pixel centres.
    Pixels flagged as no-data are ignored (the remaining neighbours are
    re-weighted). Points outside the raster return NaN.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        try:
            import rasterio
        except ImportError as exc:
            raise RuntimeError("rasterio is not installed: pip install rasterio") from exc
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DEM file not found: {path}")
        with rasterio.open(path) as src:
            if src.crs is None or src.crs.to_epsg() != 4326:
                raise ValueError("DEM must be in geographic coordinates (EPSG:4326)")
            array = src.read(1).astype("float64")
            transform, nodata = src.transform, src.nodata
        if nodata is not None:
            array[array == nodata] = np.nan
        array[~np.isfinite(array)] = np.nan
        self._array = array
        self._inverse = ~transform

    @property
    def median(self) -> float:
        """Median height of all valid pixels (used to fill gaps)."""
        return float(np.nanmedian(self._array))

    def sample(self, lons: Sequence[float], lats: Sequence[float]) -> np.ndarray:
        """Heights at the given points (NaN where no data is available)."""
        lons = np.atleast_1d(np.asarray(lons, dtype=float))
        lats = np.atleast_1d(np.asarray(lats, dtype=float))
        inv = self._inverse
        cols = inv.a * lons + inv.b * lats + inv.c
        rows = inv.d * lons + inv.e * lats + inv.f
        cols, rows = cols - 0.5, rows - 0.5          # pixel centres sit at +0.5
        c0, r0 = np.floor(cols).astype(int), np.floor(rows).astype(int)
        fc, fr = cols - c0, rows - r0
        height, width = self._array.shape
        total, weight_sum = np.zeros(lons.shape), np.zeros(lons.shape)
        for dr, dc, w in ((0, 0, (1 - fr) * (1 - fc)), (0, 1, (1 - fr) * fc),
                          (1, 0, fr * (1 - fc)), (1, 1, fr * fc)):
            r, c = r0 + dr, c0 + dc
            inside = (r >= 0) & (r < height) & (c >= 0) & (c < width)
            values = np.full(lons.shape, np.nan)
            values[inside] = self._array[r[inside], c[inside]]
            ok = np.isfinite(values) & (w > 0)
            total[ok] += w[ok] * values[ok]
            weight_sum[ok] += w[ok]
        safe = np.where(weight_sum > 0, weight_sum, 1.0)
        return np.where(weight_sum > 0, total / safe, np.nan)


def main() -> None:
    """Create ``data/external/velachery_dem.tif`` for the cached road graph."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Prepare the Velachery DEM.")
    parser.add_argument("--from-file", help="crop this local DEM instead of downloading")
    parser.add_argument("--buffer", type=float, default=0.01,
                        help="degrees of margin around the road network (default 0.01)")
    args = parser.parse_args()

    from routing.graph_builder import load_osm_graph  # lazy: needs osmnx
    graph = load_osm_graph()
    xs = [d["x"] for _, d in graph.nodes(data=True)]
    ys = [d["y"] for _, d in graph.nodes(data=True)]
    b = args.buffer
    bounds = (min(xs) - b, min(ys) - b, max(xs) + b, max(ys) + b)
    path = download_dem(bounds, DEM_PATH, source=args.from_file)

    values = DemSampler(path).sample(xs, ys)
    logger.info("Junction elevations: min %.1f m, mean %.1f m, max %.1f m (%d without data)",
                np.nanmin(values), np.nanmean(values), np.nanmax(values),
                int(np.isnan(values).sum()))


if __name__ == "__main__":
    main()
