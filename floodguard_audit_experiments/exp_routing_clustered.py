"""E3c: dependence-aware uncertainty for the observed-label routing evaluation (Table VI).

Reads out/routing_od_{2015,2017}.csv written by exp_routing.py; no routing is re-run.

Why: the 900 routes per event come from 3 seeds x 300 starts but only 699 starts are
unique. Because the destination is fixed by the start (nearest shelter by road distance)
and routing is deterministic, a repeated start reproduces an identical route, so the 201
repeats are exact duplicates. A bootstrap over the 900 rows treats them as independent
and understates uncertainty. Nearby starts also share road segments.

Three analyses, all paired against the shortest route (w = 0):
  A. cluster bootstrap over the 699 unique starts, each start keeping all its rows
     (the estimand stays the 900-route pooled mean used in the paper);
  B. de-duplicated analysis: one row per unique start (n = 699), percentile bootstrap
     and Wilcoxon signed-rank on the 699 independent starts;
  C. spatial block bootstrap: starts grouped by the 500 m x 500 m UTM cell of the start
     node (same grid as exp_validation.py), resampling cells.
Holm correction is applied across the Wilcoxon tests of the policies reported in Table VI.
Output: out/routing_clustered.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
REPO = Path(os.environ.get("FLOODGUARD_REPO", HERE.parent))
B = 2000
BLOCK_M = 500.0
UTM = "EPSG:32644"
REPORTED = [("geo", 5), ("geo_lr", 5), ("rf", 5), ("geo_lr", 1), ("rf", 2)]   # Table VI rows
POLICIES = ("geo", "geo_lr", "rf")
WEIGHTS = (1, 2, 5, 10, 20)


def start_cells(starts):
    """500 m UTM cell id of each start node (needs osmnx; skipped if unavailable)."""
    try:
        import geopandas as gpd
        import osmnx as ox
        from shapely.geometry import Point
    except ImportError:
        return None
    g = ox.load_graphml(REPO / "data/processed/velachery.graphml")
    pts = gpd.GeoSeries([Point(g.nodes[s]["x"], g.nodes[s]["y"]) for s in starts],
                        crs="EPSG:4326").to_crs(UTM)
    return np.array([f"{int(np.floor(p.x / BLOCK_M))}_{int(np.floor(p.y / BLOCK_M))}" for p in pts])


def group_bootstrap(groups, cols, rng):
    """Resample groups with replacement; returns B x len(cols) matrix of column sums."""
    ug, inv = np.unique(groups, return_inverse=True)
    sums = np.zeros((len(ug), len(cols)))
    np.add.at(sums, inv, np.column_stack(cols))
    idx = rng.integers(0, len(ug), (B, len(ug)))
    return sums[idx].sum(axis=1), np.bincount(inv)[idx].sum(axis=1)


def pct(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def analyse(df, cells):
    rng = np.random.default_rng(20261103)
    starts = df["start"].to_numpy()
    first = ~df["start"].duplicated()
    dd = df[first].reset_index(drop=True)
    base_obs, base_len = df["shortest_w0_obs"].to_numpy(), df["shortest_w0_len"].to_numpy()
    res = {"n_routes": int(len(df)), "n_unique_starts": int(dd.shape[0]),
           "n_duplicate_rows": int(len(df) - dd.shape[0]),
           "start_multiplicity": {int(k): int(v) for k, v in
                                  df["start"].value_counts().value_counts().sort_index().items()},
           "duplicates_identical": bool(df.drop(columns="seed").groupby("start").nunique().max().max() == 1),
           "policies": {}}
    if cells is not None:
        cell_of = dict(zip(dd["start"], cells))
        res["n_start_cells"] = int(len(set(cells)))
    for pol in POLICIES:
        for w in WEIGHTS:
            tag = f"{pol}_w{w}"
            obs, ln = df[f"{tag}_obs"].to_numpy(), df[f"{tag}_len"].to_numpy()
            d_obs, d_len = obs - base_obs, ln - base_len
            r = {"mean_d_obs_m": float(d_obs.mean()),
                 "obs_avoided_pct": float(100 * (1 - obs.sum() / base_obs.sum())),
                 "extra_len_pct": float(100 * (ln.sum() / base_len.sum() - 1))}
            # naive (as in the submitted paper): rows treated as independent
            i = rng.integers(0, len(df), (B, len(df)))
            r["naive_ci_d_obs"] = pct(d_obs[i].mean(axis=1))
            # A. cluster bootstrap over unique starts
            s, n = group_bootstrap(starts, [d_obs, obs, base_obs], rng)
            r["cluster_ci_d_obs"] = pct(s[:, 0] / n)
            r["cluster_ci_avoided_pct"] = pct(100 * (1 - s[:, 1] / s[:, 2]))
            # B. de-duplicated (one row per unique start)
            dd_obs = (dd[f"{tag}_obs"] - dd["shortest_w0_obs"]).to_numpy()
            i = rng.integers(0, len(dd_obs), (B, len(dd_obs)))
            r["dedup_mean_d_obs_m"] = float(dd_obs.mean())
            r["dedup_obs_avoided_pct"] = float(100 * (1 - dd[f"{tag}_obs"].sum() / dd["shortest_w0_obs"].sum()))
            r["dedup_ci_d_obs"] = pct(dd_obs[i].mean(axis=1))
            nz = dd_obs[dd_obs != 0]
            r["dedup_n_changed"] = int(len(nz))
            r["dedup_wilcoxon_p"] = float(wilcoxon(nz).pvalue) if len(nz) > 10 else None
            # C. spatial blocks of start locations
            if cells is not None:
                blocks = np.array([cell_of[x] for x in starts])
                s, n = group_bootstrap(blocks, [d_obs, obs, base_obs], rng)
                r["spatial_ci_d_obs"] = pct(s[:, 0] / n)
                r["spatial_ci_avoided_pct"] = pct(100 * (1 - s[:, 1] / s[:, 2]))
            res["policies"][tag] = r
    # head-to-head at w = 5 (Section V-F text): RF minus comparator, observed flooded metres
    res["head_to_head_w5"] = {}
    for other in ("geo", "geo_lr"):
        d = (df["rf_w5_obs"] - df[f"{other}_w5_obs"]).to_numpy()
        s, n = group_bootstrap(starts, [d], rng)
        h = {"mean_m": float(d.mean()), "cluster_ci": pct(s[:, 0] / n)}
        if cells is not None:
            s, n = group_bootstrap(np.array([cell_of[x] for x in starts]), [d], rng)
            h["spatial_ci"] = pct(s[:, 0] / n)
        res["head_to_head_w5"][f"rf_minus_{other}"] = h
    tags = [f"{p}_w{w}" for p, w in REPORTED]
    ps = [res["policies"][t]["dedup_wilcoxon_p"] for t in tags]
    for t, a in zip(tags, holm(ps)):
        res["policies"][t]["dedup_wilcoxon_p_holm"] = a
    res["holm_family"] = tags
    return res


def main():
    out = {"B": B, "note": __doc__.strip().splitlines()[0], "events": {}}
    cells = None
    for ev in ("2015", "2017"):
        df = pd.read_csv(OUT / f"routing_od_{ev}.csv")
        if cells is None:
            uniq = df.loc[~df["start"].duplicated(), "start"].tolist()
            cells = start_cells(uniq)
        out["events"][ev] = analyse(df, cells)
    json.dump(out, open(OUT / "routing_clustered.json", "w"), indent=2)
    for ev, r in out["events"].items():
        print(f"\n{ev}: {r['n_routes']} routes, {r['n_unique_starts']} unique starts, "
              f"{r['n_duplicate_rows']} duplicate rows, {r.get('n_start_cells')} start cells")
        for p, w in REPORTED:
            x = r["policies"][f"{p}_w{w}"]
            f = lambda c: "[{:.1f}, {:.1f}]".format(*c)
            print(f"  {p:>6} w={w:<2} avoided {x['obs_avoided_pct']:+5.1f}%  mean d {x['mean_d_obs_m']:+6.1f} m"
                  f"  naive {f(x['naive_ci_d_obs'])} cluster {f(x['cluster_ci_d_obs'])}"
                  f" spatial {f(x.get('spatial_ci_d_obs', [np.nan, np.nan]))}"
                  f" | dedup {x['dedup_mean_d_obs_m']:+6.1f} {f(x['dedup_ci_d_obs'])}"
                  f" p={x['dedup_wilcoxon_p']:.2g} holm={x['dedup_wilcoxon_p_holm']:.2g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
