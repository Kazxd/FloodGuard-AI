"""Figures for the revised manuscript, drawn only from recorded outputs in out/."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from common import OUT

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False})
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
W = ["1", "2", "5", "10", "20"]


def fig1():
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 48); ax.axis("off")
    kinds = {"measured": "#dbe9f6", "derived": "#e4f2e1", "simulated": "#fdf0d5",
             "synthetic": "#f6dede", "process": "#ffffff"}

    def box(x, y, w, h, text, kind):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=kinds[kind], ec="#444", lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.0)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333"))

    box(1, 38, 18, 7, "Synthetic samples\n(6,000; logistic rule)", "synthetic")
    box(24, 38, 17, 7, "Features, Eq. (3)\n+ one-hot road class", "process")
    box(46, 38, 20, 7, "DT / RF grid search\n(5-fold CV, F1)", "process")
    box(71, 38, 27, 7, "Model chosen by test ROC AUC\n(nested CV reported, Table III)", "process")
    arrow(19.5, 41.5, 23.5, 41.5); arrow(41.5, 41.5, 45.5, 41.5); arrow(66.5, 41.5, 70.5, 41.5)

    box(1, 24, 18, 8, "OSM roads (cached\nGraphML), DEM,\nOSM waterways", "measured")
    box(24, 24, 17, 8, "Edge features: DEM\nelevation, distance to\nwater; history (synth.)", "derived")
    box(46, 24, 20, 8, "Scenario: rainfall,\nriver level\n(3 fixed levels)", "simulated")
    box(71, 24, 27, 8, "Edge risk p_e = model output\nCost, Eq. (4)", "process")
    arrow(19.5, 28, 23.5, 28); arrow(66.5, 28, 70.5, 28)
    ax.plot([32.5, 32.5, 76], [23.5, 21.5, 21.5], color="#333", lw=0.7)
    arrow(76, 21.5, 76, 23.5)
    arrow(84.5, 37.5, 84.5, 32.5)

    box(46, 9, 20, 8, "Dijkstra / A*\nto best of 5 high-\nground shelters", "process")
    box(71, 9, 27, 8, "Evaluation: predicted exposure\n(Table IV) and observed\nflooded length (Table VI)", "process")
    box(1, 9, 40, 8, "Crowdsourced flooded streets (osm-in/flood-map),\n2 Dec 2015 and 23 Nov 2017;\nsegment label = within 15 m (derived)", "measured")
    arrow(84.5, 23.5, 66.5, 15); arrow(66.5, 13, 70.5, 13)
    ax.plot([21, 21, 92], [8.5, 5, 5], color="#333", lw=0.7); arrow(92, 5, 92, 8.5)
    for i, (k, lab) in enumerate([("measured", "measured"), ("derived", "derived"),
                                  ("simulated", "simulated"), ("synthetic", "synthetic")]):
        ax.add_patch(plt.Rectangle((2 + 13 * i, 0.5), 2.5, 2.2, fc=kinds[k], ec="#444", lw=0.5))
        ax.text(5.2 + 13 * i, 1.6, lab, va="center", fontsize=6.4)
    fig.savefig(FIG / "fig1_pipeline.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig2():
    t = json.load(open(OUT / "routing.json"))["table2_reimpl_pooled"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for scen, mk in (("moderate", "o"), ("heavy", "s"), ("extreme", "^")):
        xs = [0] + [t[scen][w][0] for w in W]
        ys = [0] + [t[scen][w][1] for w in W]
        ax.plot(xs, ys, marker=mk, ms=3.5, lw=1, color="#333", label=scen)
        ax.annotate("w=5", (t[scen]["5"][0], t[scen]["5"][1]), xytext=(3, -8),
                    textcoords="offset points", fontsize=6)
    ax.set_xlabel("Extra route length vs w = 0 (%)")
    ax.set_ylabel("Predicted exposure avoided (%)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig2_tradeoff.png", dpi=300); plt.close(fig)


def fig3():
    r = json.load(open(OUT / "routing.json"))["events"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=False)
    sty = {"geo": ("Elevation risk", "o", "#777"), "geo_lr": ("Geography logistic", "s", "#2b6cb0"),
           "rf": ("Random Forest", "^", "#c0392b")}
    for ax, ev in zip(axs, ("2015", "2017")):
        for pol, (lab, mk, col) in sty.items():
            xs = [0] + [r[ev][f"{pol}_w{w}"]["extra_len_pct"] for w in W]
            ys = [0] + [r[ev][f"{pol}_w{w}"]["obs_avoided_pct"] for w in W]
            ax.plot(xs, ys, marker=mk, ms=3.5, lw=1, color=col, label=lab)
        ax.axhline(0, color="#999", lw=0.6, ls="--")
        ax.set_title(f"{ev} labels", fontsize=8)
        ax.set_xlabel("Extra route length vs shortest (%)")
    axs[0].set_ylabel("Observed flooded length avoided (%)")
    axs[1].legend(frameon=False, fontsize=6.5)
    fig.tight_layout(); fig.savefig(FIG / "fig3_observed_routing.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3()
