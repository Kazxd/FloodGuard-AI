"""Tests for DEM sampling and DEM-based elevation (routing/elevation.py)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from routing.elevation import DemSampler, download_dem, tile_name, tile_url
from routing.graph_builder import attach_static_features, build_grid_graph
from utils.config import VALID_RANGES

try:
    import rasterio
    from rasterio.transform import from_origin
    HAVE_RASTERIO = True
except ImportError:  # pragma: no cover
    HAVE_RASTERIO = False

WEST, NORTH, STEP = 80.10, 13.10, 0.001   # raster covers lon 80.10-80.30, lat 12.90-13.10


def write_plane_dem(path: Path, nodata: float = -9999.0, hole: bool = False) -> None:
    """Write a DEM whose height rises 0.5 m per 0.01 degree eastwards (10 + ...)."""
    cols, rows = 200, 200
    lon_centres = WEST + (np.arange(cols) + 0.5) * STEP
    plane = 10.0 + 50.0 * (lon_centres - WEST)            # linear in longitude
    data = np.tile(plane, (rows, 1)).astype("float32")
    if hole:
        data[100:105, 100:105] = nodata
    with rasterio.open(path, "w", driver="GTiff", height=rows, width=cols, count=1,
                       dtype="float32", crs="EPSG:4326",
                       transform=from_origin(WEST, NORTH, STEP, STEP), nodata=nodata) as dst:
        dst.write(data, 1)


class TestTileNames(unittest.TestCase):
    def test_velachery_tile(self) -> None:
        self.assertEqual(tile_name(12.98, 80.22), "Copernicus_DSM_COG_10_N12_00_E080_00_DEM")
        self.assertTrue(tile_url(12.98, 80.22).endswith("N12_00_E080_00_DEM.tif"))

    def test_southern_and_western_hemispheres(self) -> None:
        self.assertIn("S34_00_E018_00", tile_name(-33.5, 18.4))
        self.assertIn("N51_00_W001_00", tile_name(51.5, -0.5))

    def test_bad_bounds_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            download_dem((80.3, 12.9, 80.1, 13.0))

    def test_area_spanning_two_tiles_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            download_dem((80.5, 12.5, 81.5, 12.9))


@unittest.skipUnless(HAVE_RASTERIO, "rasterio not installed")
class TestDemSampler(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "dem.tif"
        write_plane_dem(self.path)
        self.sampler = DemSampler(self.path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_bilinear_reproduces_a_linear_surface(self) -> None:
        lons = np.array([80.15, 80.2005, 80.25])
        values = self.sampler.sample(lons, np.full(3, 13.0))
        np.testing.assert_allclose(values, 10.0 + 50.0 * (lons - WEST), atol=1e-3)

    def test_points_outside_the_raster_are_nan(self) -> None:
        self.assertTrue(np.isnan(self.sampler.sample([79.0], [13.0])[0]))
        self.assertTrue(np.isnan(self.sampler.sample([80.2], [14.0])[0]))

    def test_nodata_pixels_are_ignored(self) -> None:
        write_plane_dem(self.path, hole=True)
        sampler = DemSampler(self.path)
        lon = WEST + 102.5 * STEP        # centre of the hole
        self.assertTrue(np.isnan(sampler.sample([lon], [NORTH - 102.5 * STEP])[0]))
        edge = sampler.sample([WEST + 105.5 * STEP], [NORTH - 102.5 * STEP])[0]
        self.assertTrue(np.isfinite(edge))

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            DemSampler(Path(self.tmp.name) / "nope.tif")

    def test_crop_from_local_file(self) -> None:
        out = Path(self.tmp.name) / "cropped.tif"
        download_dem((80.15, 12.95, 80.20, 13.00), out, source=self.path)
        with rasterio.open(out) as src:
            self.assertEqual((src.width, src.height), (50, 50))
        np.testing.assert_allclose(DemSampler(out).sample([80.175], [12.975]),
                                   10.0 + 50.0 * (80.175 - WEST), atol=1e-3)


@unittest.skipUnless(HAVE_RASTERIO, "rasterio not installed")
class TestRealElevationOnGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "dem.tif"
        write_plane_dem(self.path)
        self.graph = build_grid_graph(10, 10)       # near (12.975, 80.215)
        attach_static_features(self.graph, dem_path=self.path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_elevation_comes_from_the_dem(self) -> None:
        for u, v, data in self.graph.edges(data=True):
            mid_lon = (self.graph.nodes[u]["x"] + self.graph.nodes[v]["x"]) / 2
            expected = 10.0 + 50.0 * (mid_lon - WEST)
            self.assertAlmostEqual(data["elevation_m"], expected, places=2)

    def test_features_stay_inside_valid_ranges(self) -> None:
        for _, _, data in self.graph.edges(data=True):
            low, high = VALID_RANGES["elevation_m"]
            self.assertTrue(low <= data["elevation_m"] <= high)

    def test_both_directions_share_elevation(self) -> None:
        for u, v, data in self.graph.edges(data=True):
            self.assertEqual(data["elevation_m"], self.graph[v][u]["elevation_m"])

    def test_without_dem_elevation_is_still_synthetic(self) -> None:
        other = build_grid_graph(10, 10)
        attach_static_features(other)
        differs = any(abs(d["elevation_m"] - other[u][v]["elevation_m"]) > 1e-6
                      for u, v, d in self.graph.edges(data=True))
        self.assertTrue(differs)


if __name__ == "__main__":
    unittest.main()
