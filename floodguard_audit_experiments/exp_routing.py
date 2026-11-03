"""E3 (+E5 partial): routing evaluation against observed flood extents, and an
independent re-implementation of the paper's Table II experiment (IV-H).

Policies (identical origin-destination pairs):
  shortest   : w = 0
  geo        : unsupervised geography risk p = (zmax - z) / (zmax - zmin) from DEM elevation
  geo_lr     : logistic(elev, log dist, road class) trained on the OTHER event
  rf         : committed Random Forest, scenario per event
OD design: 3 seeds (26, 123, 2026) x 300 random non-shelter starts; destination fixed to
the shelter nearest by shortest-path distance (so every policy uses the same O and D).
Outcome: observed flooded metres on the route (crowdsourced labels), any-contact rate,
route length. Paired bootstrap over starts (B=2000) + Wilcoxon signed-rank.
"""
from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression

from common import (OUT, REPO, SCENARIOS, build_graph, edge_table, flood_lines,
                    label_edges, score_rf)
from exp_validation import geo_X
from routing.cost import apply_costs
from routing.risk import load_model_bundle
from routing.shelters import find_shelters, route_to_best_shelter

SEEDS = (26, 123, 2026)
N_START = 300
WEIGHTS = (0, 1, 2, 5, 10, 20)
rng_boot = np.random.default_rng(7)


def set_risk(g, key2p):
    for u, v, d in g.edges(data=True):
        d["flood_prob"] = float(key2p[(min(u, v), max(u, v))])


def path_edges(path):
    return [(min(a, b), max(a, b)) for a, b in zip(path[:-1], path[1:])]


def route(g, s, t, w):
    apply_costs(g, w)
    return nx.dijkstra_path(g, s, t, weight="cost")


def boot_ci(x):
    x = np.asarray(x)
    m = [x[rng_boot.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    multi, g = build_graph()
    E = edge_table(multi, g)
    keys = list(E["key"])
    length = dict(zip(keys, E["length"]))
    labels = {"2015": label_edges(E, flood_lines("chennai-flooded-streets-Dec2.geojson")),
              "2017": label_edges(E, flood_lines("chennai-flooded-streets-23-Nov-2017.geojson"))}
    lab = {ev: dict(zip(keys, y)) for ev, y in labels.items()}
    bundle = load_model_bundle(REPO / "models/flood_model.joblib")
    shelters = find_shelters(g)
    z = E["elevation_m"].values
    geo = dict(zip(keys, (z.max() - z) / (z.max() - z.min())))
    Xg = geo_X(E)

    # ---------- Part A: re-implementation of Table II (model-internal exposure) ----------
    nodes = [n for n in g.nodes if n not in shelters]
    table2 = {}
    for scen, (rain, river) in SCENARIOS.items():
        rf = dict(zip(keys, score_rf(g, E, rain, river, bundle)))
        set_risk(g, rf)
        res = {w: {"len": 0.0, "exp": 0.0} for w in WEIGHTS}
        for sd in SEEDS:
            starts = np.random.default_rng(sd).choice(nodes, N_START, replace=False)
            for w in WEIGHTS:
                apply_costs(g, w)
                for s in starts:
                    r = route_to_best_shelter(g, int(s), shelters)
                    pe = path_edges(r.path)
                    res[w]["len"] += sum(length[k] for k in pe)
                    res[w]["exp"] += sum(length[k] * rf[k] for k in pe)
        b = res[0]
        table2[scen] = {w: [100 * (res[w]["len"] / b["len"] - 1),
                            100 * (1 - res[w]["exp"] / b["exp"])] for w in WEIGHTS if w}
    # ---------- Part B: observed-label routing evaluation, identical OD pairs ----------
    out = {"shelters": [int(s) for s in shelters], "table2_reimpl_pooled": table2, "events": {}}
    for ev, scen in (("2015", "extreme"), ("2017", "moderate")):
        other = "2017" if ev == "2015" else "2015"
        risks = {
            "geo": geo,
            "geo_lr": dict(zip(keys, LogisticRegression(max_iter=1000)
                               .fit(Xg, labels[other]).predict_proba(Xg)[:, 1])),
            "rf": dict(zip(keys, score_rf(g, E, *SCENARIOS[scen], bundle))),
        }
        rows = []
        set_risk(g, risks["rf"])
        for sd in SEEDS:
            starts = np.random.default_rng(sd).choice(nodes, N_START, replace=False)
            for s in starts:
                s = int(s)
                dist = {t: nx.dijkstra_path_length(g, s, t, weight="length") for t in shelters}
                t = min(dist, key=dist.get)
                rec = {"seed": sd, "start": s, "dest": t}
                for pol in ("shortest", "geo", "geo_lr", "rf"):
                    for w in (WEIGHTS if pol != "shortest" else (0,)):
                        set_risk(g, risks.get(pol, geo))
                        pe = path_edges(route(g, s, t, w))
                        tag = f"{pol}_w{w}"
                        rec[f"{tag}_len"] = sum(length[k] for k in pe)
                        rec[f"{tag}_obs"] = sum(length[k] * lab[ev][k] for k in pe)
                        rec[f"{tag}_touch"] = int(any(lab[ev][k] for k in pe))
                rows.append(rec)
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"routing_od_{ev}.csv", index=False)
        summ = {"n_routes": len(df), "scenario": scen, "unique_starts": int(df["start"].nunique())}
        base_len, base_obs = df["shortest_w0_len"], df["shortest_w0_obs"]
        summ["shortest"] = {"mean_len": float(base_len.mean()), "mean_obs_m": float(base_obs.mean()),
                            "touch_rate": float(df["shortest_w0_touch"].mean())}
        for pol in ("geo", "geo_lr", "rf"):
            for w in WEIGHTS[1:]:
                tag = f"{pol}_w{w}"
                d_obs = df[f"{tag}_obs"] - base_obs
                d_len = df[f"{tag}_len"] - base_len
                nz = d_obs[d_obs != 0]
                summ[tag] = {
                    "extra_len_pct": float(100 * (df[f"{tag}_len"].sum() / base_len.sum() - 1)),
                    "obs_avoided_pct": float(100 * (1 - df[f"{tag}_obs"].sum() / base_obs.sum())),
                    "mean_d_obs_m": float(d_obs.mean()), "d_obs_ci": boot_ci(d_obs),
                    "mean_d_len_m": float(d_len.mean()), "d_len_ci": boot_ci(d_len),
                    "touch_rate": float(df[f"{tag}_touch"].mean()),
                    "wilcoxon_p": float(wilcoxon(nz).pvalue) if len(nz) > 10 else None,
                    "routes_changed": int((d_len != 0).sum())}
        # head-to-head at w=5: RF vs geo, paired
        for a, b in (("rf", "geo"), ("rf", "geo_lr")):
            d = df[f"{a}_w5_obs"] - df[f"{b}_w5_obs"]
            nz = d[d != 0]
            summ[f"{a}_vs_{b}_w5"] = {"mean_d_obs_m": float(d.mean()), "ci": boot_ci(d),
                                      "wilcoxon_p": float(wilcoxon(nz).pvalue) if len(nz) > 10 else None}
        out["events"][ev] = summ
    json.dump(out, open(OUT / "routing.json", "w"), indent=2)


if __name__ == "__main__":
    main()
