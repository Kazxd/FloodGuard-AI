"""E4: statistical robustness of the synthetic-benchmark results (Table I).

1. Recompute Table I + oracle ceiling; bootstrap 95% CIs; paired DT-vs-RF AUC difference.
2. Calibration against the stored true probability (ECE, slope, RMSE to truth).
3. Nested CV (5-fold outer; inner 3-fold GridSearchCV with the repo's grids; 2-CPU budget)
   where DT-vs-RF selection happens INSIDE the outer training fold -> unbiased estimate.
4. Variation over 10 independently generated synthetic datasets (generator seeds).
5. Scenario extrapolation: training support near each routing scenario.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     train_test_split)
from sklearn.tree import DecisionTreeClassifier

from common import OUT, REPO, SCENARIOS
from data.generate_dataset import generate
from models.train_model import clean_data, get_candidates
from utils.config import RANDOM_SEED, TEST_SIZE
from utils.features import build_features

rng = np.random.default_rng(11)


def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def slope(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return float(LogisticRegression(C=1e6).fit(lp.reshape(-1, 1), y).coef_[0, 0])


def boot(y, preds, B=2000):
    out = {k: [] for k in preds}
    diffs = []
    n = len(y)
    for _ in range(B):
        i = rng.integers(0, n, n)
        a = {k: roc_auc_score(y[i], p[i]) for k, p in preds.items()}
        for k in a:
            out[k].append(a[k])
        diffs.append(a["Random Forest"] - a["Decision Tree"])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}
    d = np.array(diffs)
    return ci, {"mean": float(d.mean()), "ci": [float(np.percentile(d, 2.5)),
                                                 float(np.percentile(d, 97.5))],
                "p_two_sided": float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def main():
    df = clean_data(pd.read_csv(REPO / "data/raw/road_segments.csv"))
    X, y = build_features(df), df["flooded"].astype(int)
    xtr, xte, ytr, yte, _, pte = train_test_split(X, y, df["true_probability"], test_size=TEST_SIZE,
                                                  stratify=y, random_state=RANDOM_SEED)
    yte_a, pte_a = yte.values, pte.values
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    preds, rep = {}, {}
    for name, (est, grid) in get_candidates().items():
        s = GridSearchCV(est, grid, scoring="f1", cv=cv, n_jobs=-1).fit(xtr, ytr)
        p = s.best_estimator_.predict_proba(xte)[:, 1]
        preds[name] = p
        rep[name] = {"auc": roc_auc_score(yte_a, p), "acc": accuracy_score(yte_a, p >= .5),
                     "f1": f1_score(yte_a, p >= .5), "brier": brier_score_loss(yte_a, p),
                     "ece": ece(yte_a, p), "calib_slope": slope(yte_a, p),
                     "rmse_to_true_prob": float(np.sqrt(np.mean((p - pte_a) ** 2)))}
    preds["Oracle"] = pte_a
    rep["Oracle"] = {"auc": roc_auc_score(yte_a, pte_a), "acc": accuracy_score(yte_a, pte_a >= .5),
                     "f1": f1_score(yte_a, pte_a >= .5), "brier": brier_score_loss(yte_a, pte_a),
                     "ece": ece(yte_a, pte_a)}
    lr = LogisticRegression(max_iter=2000).fit(xtr, ytr).predict_proba(xte)[:, 1]
    preds["Logistic regression"] = lr
    rep["Logistic regression"] = {"auc": roc_auc_score(yte_a, lr), "brier": brier_score_loss(yte_a, lr),
                                  "ece": ece(yte_a, lr),
                                  "rmse_to_true_prob": float(np.sqrt(np.mean((lr - pte_a) ** 2)))}
    rep["Prevalence"] = {"auc": 0.5, "brier": brier_score_loss(yte_a, np.full(len(yte_a), ytr.mean()))}
    ci, diff = boot(yte_a, preds)
    for k in ci:
        rep[k]["auc_ci"] = ci[k]
    out = {"table1_recomputed": rep, "rf_minus_dt_auc": diff}

    # dataset-seed variation (fixed repo-tuned hyperparameters)
    seeds = []
    for sd in range(10):
        d2 = clean_data(generate(seed=1000 + sd))
        X2, y2 = build_features(d2), d2["flooded"]
        a, b, c, e = train_test_split(X2, y2, test_size=TEST_SIZE, stratify=y2, random_state=sd)
        rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=sd,
                                    n_jobs=-1).fit(a, c)
        dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=30, random_state=sd).fit(a, c)
        seeds.append({"rf": roc_auc_score(e, rf.predict_proba(b)[:, 1]),
                      "dt": roc_auc_score(e, dt.predict_proba(b)[:, 1])})
    s = pd.DataFrame(seeds)
    out["dataset_seed_variation"] = {"rf_mean_sd": [s.rf.mean(), s.rf.std()],
                                     "dt_mean_sd": [s.dt.mean(), s.dt.std()],
                                     "rf_better_in": int((s.rf > s.dt).sum())}

    # scenario support in training data
    sup = {}
    for k, (rain, river) in SCENARIOS.items():
        near = (df.rainfall_mm.sub(rain).abs() <= 25) & (df.river_level_m.sub(river).abs() <= 0.5)
        sup[k] = {"rows_within_25mm_0.5m": int(near.sum()),
                  "river_level_percentile_given_rain_pm25": float(
                      (df.loc[df.rainfall_mm.sub(rain).abs() <= 25, "river_level_m"] < river).mean())}
    out["scenario_support"] = sup
    json.dump(out, open(OUT / "synthetic_stats.json", "w"), indent=2, default=float)
    # nested CV: selection inside the outer fold
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=int(os.environ.get("NESTED_REPEATS", "1")), random_state=RANDOM_SEED)
    nested, chosen = [], []
    for tr, te in outer.split(X, y):
        best, best_s, best_m = None, -1, None
        for name, (est, grid) in get_candidates().items():
            s = GridSearchCV(clone(est), grid, scoring="roc_auc",
                             cv=StratifiedKFold(3, shuffle=True, random_state=0), n_jobs=-1
                             ).fit(X.iloc[tr], y.iloc[tr])
            if s.best_score_ > best_s:
                best, best_s, best_m = name, s.best_score_, s.best_estimator_
        p = best_m.predict_proba(X.iloc[te])[:, 1]
        nested.append(roc_auc_score(y.iloc[te], p))
        chosen.append(best)
    out["nested_cv"] = {"auc_mean": float(np.mean(nested)), "auc_sd": float(np.std(nested, ddof=1)),
                        "auc_min_max": [float(min(nested)), float(max(nested))],
                        "selected_counts": pd.Series(chosen).value_counts().to_dict()}

    json.dump(out, open(OUT / "synthetic_stats.json", "w"), indent=2, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
