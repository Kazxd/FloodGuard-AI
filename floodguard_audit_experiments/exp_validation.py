"""E1+E2: validate frozen synthetic-trained models against observed flood labels
and compare with prevalence / geography baselines.

Labels: osm-in/flood-map crowdsourced flooded streets
  2015: 'Dec2' snapshot (11 AM, 2 Dec 2015)   -> primary event
  2017: '23-Nov-2017'                         -> second, held-out event
Edge positive if >=50% of its length lies within 15 m of a reported flooded street.
Negatives = not reported (noisy; see limitations).

Uncertainty: spatial block bootstrap (500 m UTM cells), B=2000, paired across models.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from common import (OUT, REPO, SCENARIOS, build_graph, edge_table, flood_lines,
                    label_edges, score_rf)
from routing.risk import load_model_bundle
from utils.config import FEATURE_COLUMNS, NUMERIC_FEATURES, RANDOM_SEED, TEST_SIZE
from utils.features import build_features

B = 2000
BLOCK_M = 500.0
rng = np.random.default_rng(2026)
ROAD_RANK = {"residential": 4, "tertiary": 3, "secondary": 2, "primary": 1, "trunk": 0}


def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def calib_slope_intercept(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m = LogisticRegression(C=1e6).fit(lp.reshape(-1, 1), y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def metrics(y, s, is_prob):
    out = {"roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s)}
    if is_prob:
        out["brier"] = brier_score_loss(y, s)
        out["ece"] = ece(y, s)
        out["calib_slope"], out["calib_intercept"] = calib_slope_intercept(y, s)
    return out


def block_bootstrap(blocks, y, scores, ref="RF (committed)"):
    ub = np.unique(blocks)
    members = {b: np.where(blocks == b)[0] for b in ub}
    aucs = {k: [] for k in scores}
    for _ in range(B):
        idx = np.concatenate([members[b] for b in rng.choice(ub, len(ub), replace=True)])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        for k, s in scores.items():
            aucs[k].append(roc_auc_score(yy, s[idx]))
    res = {}
    for k in scores:
        a = np.array(aucs[k])
        d = a - np.array(aucs[ref])
        res[k] = {"auc_ci": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
                  "diff_vs_ref": float(np.mean(d)),
                  "diff_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                  "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
    return res


def static_frame(E):
    return E[["elevation_m", "distance_from_river_m", "historical_flood_freq", "road_type"]].copy()


def geo_X(E):
    X = pd.DataFrame({"elev": E["elevation_m"], "logdist": np.log1p(E["distance_from_river_m"])})
    for rt in ROAD_RANK:
        X[rt] = (E["road_type"] == rt).astype(int)
    return X


def main():
    multi, g = build_graph()
    E = edge_table(multi, g)
    c = E.geometry.centroid
    blocks = (np.floor(c.x / BLOCK_M).astype(int).astype(str) + "_"
              + np.floor(c.y / BLOCK_M).astype(int).astype(str)).values

    labels = {"2015": label_edges(E, flood_lines("chennai-flooded-streets-Dec2.geojson")),
              "2017": label_edges(E, flood_lines("chennai-flooded-streets-23-Nov-2017.geojson"))}
    # explicit negatives available in the 2015 snapshot
    L15 = flood_lines("chennai-flooded-streets-Dec2.geojson")
    explicit_neg = label_edges(E, L15.assign(is_flooded=~L15["is_flooded"]))

    bundle = load_model_bundle(REPO / "models/flood_model.joblib")
    # Decision Tree with the repo's tuned params, fit on the repo's own train split
    raw = pd.read_csv(REPO / "data/raw/road_segments.csv")
    Xs = build_features(raw)
    xtr, _, ytr, _ = train_test_split(Xs, raw["flooded"], test_size=TEST_SIZE,
                                      stratify=raw["flooded"], random_state=RANDOM_SEED)
    dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=30,
                                random_state=RANDOM_SEED).fit(xtr, ytr)

    event_scen = {"2015": "extreme", "2017": "moderate"}
    report = {"n_edges": int(len(E)), "km": float(E.length.sum() / 1000),
              "n_blocks": int(len(set(blocks))),
              "explicit_negative_edges_2015": int(explicit_neg.sum()), "events": {}}

    # supervised geography baseline: trained on one event, tested on the other
    Xg = geo_X(E)
    geo_fit = {ev: LogisticRegression(max_iter=1000).fit(Xg, labels[ev]) for ev in labels}

    for ev, y in labels.items():
        rain, river = SCENARIOS[event_scen[ev]]
        other = "2017" if ev == "2015" else "2015"
        sf = static_frame(E)
        sf["rainfall_mm"], sf["river_level_m"] = rain, river
        scores = {
            "RF (committed)": score_rf(g, E, rain, river, bundle),
            "Decision Tree (repo params)": dt.predict_proba(
                build_features(sf[NUMERIC_FEATURES + ["road_type"]])[FEATURE_COLUMNS])[:, 1],
            "Prevalence (other event)": np.full(len(E), labels[other].mean()),
            "Low elevation (-elev)": -E["elevation_m"].values,
            "Near water (-dist)": -E["distance_from_river_m"].values,
            "Road class (minor=risky)": E["road_type"].map(ROAD_RANK).values.astype(float),
            "Synthetic flood history": E["historical_flood_freq"].values.astype(float),
            f"Geo logistic (trained {other})": geo_fit[other].predict_proba(Xg)[:, 1],
        }
        # constant scores: AUC=0.5 by definition; add tiny jitter-free handling
        res = {}
        for k, s in scores.items():
            is_prob = k.startswith(("RF", "Decision", "Prevalence", "Geo logistic"))
            if np.ptp(s) == 0:
                res[k] = {"roc_auc": 0.5, "pr_auc": float(y.mean()),
                          "brier": brier_score_loss(y, s), "ece": float(abs(y.mean() - s[0]))}
            else:
                res[k] = metrics(y, s, is_prob)
        boot = block_bootstrap(blocks, y, {k: v for k, v in scores.items() if np.ptp(v) > 0})
        for k in boot:
            res[k].update(boot[k])
        # RF sensitivity to the random synthetic flood-history draw
        seeds = []
        for sd in range(20):
            _, gs = build_graph(hist_seed=sd)
            seeds.append(roc_auc_score(y, score_rf(gs, E, rain, river, bundle)))
        # RF sensitivity to scenario choice
        scen = {s: roc_auc_score(y, score_rf(g, E, *SCENARIOS[s], bundle)) for s in SCENARIOS}
        report["events"][ev] = {
            "positives": int(y.sum()), "prevalence": float(y.mean()),
            "scenario_used": event_scen[ev], "models": res,
            "rf_auc_over_20_history_seeds": [float(np.min(seeds)), float(np.median(seeds)),
                                             float(np.max(seeds))],
            "rf_auc_by_scenario": scen}

    # label-construction sensitivity for the key comparison (RF vs -elev, 2015)
    sens = []
    for buf in (10, 15, 25):
        for frac in (0.3, 0.5, 0.7):
            y = label_edges(E, flood_lines("chennai-flooded-streets-Dec2.geojson"), buf, frac)
            sens.append({"buffer_m": buf, "min_frac": frac, "prev": float(y.mean()),
                         "rf_auc": roc_auc_score(y, score_rf(g, E, *SCENARIOS["extreme"], bundle)),
                         "elev_auc": roc_auc_score(y, -E["elevation_m"].values)})
    report["label_sensitivity_2015"] = sens
    json.dump(report, open(OUT / "validation.json", "w"), indent=2)
    E.assign(y2015=labels["2015"], y2017=labels["2017"]).drop(columns="key").to_file(
        OUT / "edges_labelled.gpkg")
    print(json.dumps(report, indent=1)[:6000])


if __name__ == "__main__":
    main()
