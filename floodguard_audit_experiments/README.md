# FloodGuard Lite audit experiments

Runs against an UNMODIFIED checkout of Kazxd/FloodGuard-AI (commit 4d194de) and
osm-in/flood-map (commit 5790681). Nothing in either repo is written.

    export FLOODGUARD_REPO=~/FloodGuard-AI
    export FLOODMAP_DATA=~/flood-map/data
    pip install -r $FLOODGUARD_REPO/requirements-lock.txt rasterio geopandas scipy
    python exp_validation.py        # E1+E2 -> out/validation.json, out/edges_labelled.gpkg
    python exp_routing.py           # E3+E3b -> out/routing.json, out/routing_od_{2015,2017}.csv
    python exp_synthetic_stats.py   # E4 -> out/synthetic_stats.json
    python show_val.py              # print the E1/E2 table

Waterways come from the repo's committed Overpass cache (drains + canals only).

Options:
    FLOODGUARD_WATERWAYS=<file.geojson>  use a saved waterway layer instead of the repo cache
    NESTED_REPEATS=5                     5x5 outer folds in exp_synthetic_stats.py (slow)
    python exp_astar.py                  Dijkstra vs A* benchmark -> out/astar_bench.json
    python make_figures.py               Figs. 1-3 -> out/figures/
