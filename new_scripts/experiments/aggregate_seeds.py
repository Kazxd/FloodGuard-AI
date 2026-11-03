"""Aggregate several routing-experiment runs (different seeds) into the paper numbers.

Reads the ``summary.csv`` file written by ``experiments.routing_experiment`` for each seed
and reproduces Table II, the headline w = 5 numbers, the at-risk counts and Fig. 2.
Nothing is typed in by hand: every value comes from the CSV files.

    python -m experiments.aggregate_seeds reports/routing_experiment/seed_26/summary.csv ^
        reports/routing_experiment/seed_123/summary.csv reports/routing_experiment/seed_2026/summary.csv

Outputs (in ``reports/routing_experiment/aggregate/``): ``table_ii.csv``, ``headline_w5.csv``,
``at_risk_w5.csv`` and ``tradeoff_mean.png``.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from utils.config import REPORTS_DIR  # noqa: E402

logger = logging.getLogger(__name__)
HEADLINE_W = 5.0
SCENARIO_ORDER = ["moderate", "heavy", "extreme"]
REQUIRED = {"scenario", "subset", "weight", "routes", "extra_distance_pct", "exposure_reduction_pct",
            "mean_risk", "mean_max_risk"}


def load_runs(paths: List[Path]) -> pd.DataFrame:
    """Stack the summary files, tagging each row with the file it came from."""
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = REQUIRED - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame["run"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def seed_mean(runs: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Mean over runs for each scenario and weight (simple mean of the per-run values)."""
    part = runs[runs["subset"] == subset]
    grouped = part.groupby(["scenario", "weight"], sort=False)
    out = grouped.agg(extra_distance_pct=("extra_distance_pct", "mean"),
                      exposure_reduction_pct=("exposure_reduction_pct", "mean"),
                      exposure_min=("exposure_reduction_pct", "min"),
                      exposure_max=("exposure_reduction_pct", "max"),
                      routes_total=("routes", "sum"), runs=("run", "nunique")).reset_index()
    out["scenario"] = pd.Categorical(out["scenario"], SCENARIO_ORDER, ordered=True)
    return out.sort_values(["scenario", "weight"]).reset_index(drop=True)


def headline(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-scenario headline numbers at w = 5 (all starts), with ranges across runs."""
    part = runs[(runs["subset"] == "all") & (runs["weight"] == HEADLINE_W)]
    base = runs[(runs["subset"] == "all") & (runs["weight"] == 0.0)]
    rows = []
    for scenario in SCENARIO_ORDER:
        p, b = part[part["scenario"] == scenario], base[base["scenario"] == scenario]
        if p.empty:
            continue
        rows.append({"scenario": scenario, "runs": len(p), "routes_total": int(p["routes"].sum()),
                     "extra_distance_pct": p["extra_distance_pct"].mean(),
                     "extra_min": p["extra_distance_pct"].min(), "extra_max": p["extra_distance_pct"].max(),
                     "exposure_avoided_pct": p["exposure_reduction_pct"].mean(),
                     "avoided_min": p["exposure_reduction_pct"].min(),
                     "avoided_max": p["exposure_reduction_pct"].max(),
                     "risk_shortest": b["mean_risk"].mean(), "risk_w5": p["mean_risk"].mean(),
                     "max_risk_shortest": b["mean_max_risk"].mean(), "max_risk_w5": p["mean_max_risk"].mean()})
    return pd.DataFrame(rows)


def at_risk(runs: pd.DataFrame) -> pd.DataFrame:
    """At-risk subset at w = 5: route counts and both simple and route-weighted means."""
    part = runs[(runs["subset"] == "at_risk") & (runs["weight"] == HEADLINE_W)]
    rows = []
    for scenario in SCENARIO_ORDER:
        p = part[part["scenario"] == scenario]
        if p.empty:
            continue
        n = p["routes"]
        rows.append({"scenario": scenario, "routes_total": int(n.sum()),
                     "routes_per_run": ", ".join(str(int(v)) for v in n),
                     "extra_simple": p["extra_distance_pct"].mean(),
                     "extra_weighted": (p["extra_distance_pct"] * n).sum() / n.sum(),
                     "avoided_simple": p["exposure_reduction_pct"].mean(),
                     "avoided_weighted": (p["exposure_reduction_pct"] * n).sum() / n.sum()})
    return pd.DataFrame(rows)


def plot(mean_all: pd.DataFrame, path: Path) -> None:
    """Mean trade-off curves (extra distance vs exposure avoided), one per scenario."""
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for scenario in SCENARIO_ORDER:
        g = mean_all[mean_all["scenario"] == scenario]
        if not g.empty:
            ax.plot(g["extra_distance_pct"], g["exposure_reduction_pct"], marker="o", ms=3.5, lw=1.2, label=scenario)
    ax.set_xlabel("Extra distance vs shortest route (%)", fontsize=7.5)
    ax.set_ylabel("Flood exposure avoided (%)", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Aggregate routing-experiment summaries over seeds")
    parser.add_argument("summaries", nargs="+", type=Path, help="summary.csv files, one per seed")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR / "routing_experiment" / "aggregate")
    args = parser.parse_args()

    runs = load_runs(args.summaries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mean_all = seed_mean(runs, "all")
    table = mean_all[mean_all["weight"] > 0].copy()
    table["cell"] = table["extra_distance_pct"].map("{:.1f}".format) + "/" + table["exposure_reduction_pct"].map("{:.1f}".format)
    table_ii = table.pivot(index="scenario", columns="weight", values="cell")
    table_ii.to_csv(args.output_dir / "table_ii.csv")
    head, risk = headline(runs), at_risk(runs)
    head.to_csv(args.output_dir / "headline_w5.csv", index=False)
    risk.to_csv(args.output_dir / "at_risk_w5.csv", index=False)
    plot(mean_all, args.output_dir / "tradeoff_mean.png")

    pd.set_option("display.width", 200)
    print(f"\nRuns: {len(args.summaries)}\n\nTable II (extra distance % / exposure avoided %):\n{table_ii.to_string()}")
    print(f"\nHeadline w = {HEADLINE_W:g} (all starts):\n{head.round(3).to_string(index=False)}")
    print(f"\nAt-risk starts, w = {HEADLINE_W:g}:\n{risk.round(3).to_string(index=False)}")
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
