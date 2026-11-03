# FloodGuard Lite: evaluation (audit) experiments

These scripts produce every number in Sections V-A and V-C to V-F of the paper. They
import the FloodGuard Lite code in this repository read-only and write only to `out/`.

## Commits

| Role | Commit |
| --- | --- |
| System implementation used for all reported results (model, graph, DEM, waterway cache) | `4d194de` (24 Oct 2026) |
| Evaluation scripts and their outputs first added | `d1e1a76` (3 Nov 2026) |
| Reproducibility package (import fix, pinned waterway layer, clustered bootstrap) | the commit containing this README |

The implementation files that the experiments read (`routing/`, `utils/`, `models/flood_model.joblib`,
`data/processed/velachery.graphml`, `data/external/velachery_dem.tif`, `cache/6b73b1e7….json`,
`cache/c5fb3153….json`) are unchanged since `4d194de`.

## Inputs

* Flood reports: `git clone https://github.com/osm-in/flood-map && git -C flood-map checkout 5790681`
* Waterways: the paper uses the 9 Oct 2026 Overpass responses `cache/6b73b1e7….json` and
  `cache/c5fb3153….json` (identical waterway-only queries: 14 drains, 2 canals). These are
  pinned in `common.py`. `cache/2ec40626….json` (27 Oct 2026 lakes/water query) and
  `data/raw/lakes_wetlands.geojson` were added later and are **not** used for any reported result.
  Including them changes every risk-based number (e.g. Random Forest 2015 AUC 0.692 → 0.696).

## Run

    export FLOODGUARD_REPO=$PWD/..                # this repository
    export FLOODMAP_DATA=/path/to/flood-map/data  # required
    pip install -r ../requirements-lock.txt rasterio==1.5.2 geopandas==1.2.0 scipy==1.18.1
    python exp_validation.py         # Table V + sensitivity  -> out/validation.json, out/edges_labelled.gpkg
    python exp_routing.py            # Table IV, Table VI     -> out/routing.json, out/routing_od_{2015,2017}.csv
    python exp_routing_clustered.py  # Table VI intervals by unique start / spatial block -> out/routing_clustered.json
    python exp_astar.py              # Section V-D            -> out/astar_bench.json (timings are machine-specific)
    python exp_synthetic_stats.py    # Table III, nested CV (5 x 5, outer seeds 42-46) -> out/synthetic_stats.json
    python make_figures.py           # Figs. 1-3              -> out/figures/
    python show_val.py               # print Table V

Approximate run time on 2 vCPUs: validation 3 min, routing 4.5 min, synthetic stats 6 min.

Options:

    FLOODGUARD_WATERWAY_CACHE=a.json,b.json   use other cache files (sensitivity only)
    FLOODGUARD_WATERWAYS=<file.geojson>       use a saved GeoJSON layer instead of the cache
    NESTED_REPEATS=1                          quicker nested CV (one outer repeat)

## Verified

Re-run on Python 3.13.18 (x86_64) with the pinned versions: `validation.json`, `routing.json`,
the per-route CSVs, `astar_bench.json` node counts and `synthetic_stats.json` reproduce the
values printed in the paper. `pytest` at the repository root: 209 passed.

## Changes relative to `d1e1a76`

* `exp_routing.py`: `from alidation import geo_X` → `from exp_validation import geo_X` (the
  committed file did not run).
* `common.py`: waterway cache pinned to the paper's two files instead of `cache/*.json`;
  `FLOODMAP_DATA` default no longer points to a temporary scratch directory.
* `exp_synthetic_stats.py`: feature-level Bayes reference row added; outer folds use seeds 42–46 as stated in the paper (previously
  `RepeatedStratifiedKFold`); logistic regression standardised as stated in Table III; full
  threshold metrics and calibration intercepts for every row.
* `exp_routing_clustered.py`: new; cluster bootstrap over unique starts, de-duplicated
  Wilcoxon tests with Holm correction, and spatial block bootstrap of start locations.
* `out/validation.json` regenerated: the copy committed in `d1e1a76` had been produced with
  the later lakes/water cache file and did not match the paper.
