"""Figure: time spent aggregating vs population size and body size (manuscript Fig. 3).

Panel A: hours at the FSA per fish-season across population-size groups (violins and
points colored by projected total length), with the reporting-model fitted tenure at
mean body length and its 95% CI connecting the group means. Panel B: hours adjusted to
mean population size against projected total length, with the reporting-model fit and
95% CI; stars mark the fitted tenure at the smallest and largest projected lengths
observed among aggregating fish.

Run AFTER tenure_dataset.py, from this directory:  python3 tenure_figure.py

Output (written to ../../output/):  Fig3_aggregation_tenure.png
"""
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from matplotlib.colors import Normalize

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
p = pd.read_csv(os.path.join(OUT, "tenure_fish_seasons.csv")).dropna(subset=["length_proj"])
p = p.copy()
p["ln_hours"] = np.log(p.hours)
p["ln_pop"] = np.log(p.popsize)
p["year_f"] = p.year.astype(str)

# the reporting model (same specification as tenure_model.py)
m = smf.mixedlm("ln_hours ~ length_proj + ln_pop", p, groups=p.year_f).fit(reml=False)
b = m.params
V = m.cov_params()
names = ["Intercept", "length_proj", "ln_pop"]


def size_group(row):
    """Population-size display groups: Cayman Brac, then Little Cayman recovery stages."""
    if row.island == "CB":
        return 520
    if row.popsize <= 3000:
        return 2000
    if row.popsize <= 4500:
        return 4100
    if row.popsize <= 6000:
        return 5200
    if row.popsize <= 7000:
        return 6700
    return 8300


p["grp"] = p.apply(size_group, axis=1)
groups = sorted(p.grp.unique())
labels = {g: f"{p[p.grp == g].popsize.mean():,.0f}" for g in groups}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.5, 6.1), width_ratios=[1.15, 1])

norm = Normalize(vmin=p.length_proj.min(), vmax=p.length_proj.max())
rng = np.random.default_rng(7)
Lbar = p.length_proj.mean()
xs, fit, lo, hi = [], [], [], []
for k, g in enumerate(groups):
    sub = p[p.grp == g]
    vp = axA.violinplot([sub.hours.values], positions=[k], widths=0.82, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor("none")
        body.set_edgecolor("#555555")
        body.set_alpha(1.0)
        body.set_linewidth(0.9)
    x = k + rng.uniform(-0.16, 0.16, len(sub))
    axA.scatter(x, sub.hours, c=viridis(norm(sub.length_proj)), s=36, alpha=0.85,
                edgecolors="white", linewidths=0.4, zorder=3)
    X = np.array([1.0, Lbar, np.log(sub.popsize.mean())])
    eta = float(X @ b[names].values)
    se = float(np.sqrt(X @ V.loc[names, names].values @ X))
    xs.append(k)
    fit.append(np.exp(eta))
    lo.append(np.exp(eta - 1.96 * se))
    hi.append(np.exp(eta + 1.96 * se))
axA.fill_between(xs, lo, hi, color="#2b6cb0", alpha=0.15, zorder=4)
axA.plot(xs, fit, "-o", color="#2b6cb0", lw=2.2, ms=6.5, mec="white", mew=0.8, zorder=5)
axA.set_xticks(range(len(groups)))
axA.set_xticklabels([labels[g] for g in groups])
axA.set_ylim(0, None)
axA.set_xlabel("Spawning Population Size")
axA.set_ylabel("Hours Spent Aggregating")
axA.spines[["top", "right"]].set_visible(False)
axA.annotate(r"$-23\%$ per doubling of population size ($P = 3\times10^{-5}$)",
             xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
             fontsize=11.5, color="#333333")
sm = plt.cm.ScalarMappable(cmap=viridis, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, ax=axA, pad=0.015)
cb.set_label("Projected total length (cm)")
axA.text(-0.10, 1.02, "A", transform=axA.transAxes, fontsize=17, fontweight="bold")

# ---- panel B: the body-size effect at mean population size ----
p["adj_hours"] = np.exp(p.ln_hours - b["ln_pop"] * (p.ln_pop - p.ln_pop.mean()))
xg = np.linspace(p.length_proj.min(), p.length_proj.max(), 120)
X = np.column_stack([np.ones_like(xg), xg, np.full_like(xg, p.ln_pop.mean())])
eta = X @ b[names].values
se = np.sqrt(np.einsum("ij,jk,ik->i", X, V.loc[names, names].values, X))
mk = {"LC": ("o", "#2b6cb0"), "CB": ("^", "#c05621")}
for island, (marker, color) in mk.items():
    g = p[p.island == island]
    axB.scatter(g.length_proj, g.adj_hours, marker=marker, s=44, color=color, alpha=0.75,
                edgecolors="white", linewidths=0.4, zorder=3,
                label={"LC": "Little Cayman", "CB": "Cayman Brac"}[island])
axB.plot(xg, np.exp(eta), "--", color="#222222", lw=2, zorder=4)
axB.fill_between(xg, np.exp(eta - 1.96 * se), np.exp(eta + 1.96 * se),
                 color="#222222", alpha=0.14, zorder=2)
Lmin, Lmax = float(p.length_proj.min()), float(p.length_proj.max())
fitted = lambda L: float(np.exp(b["Intercept"] + b["length_proj"] * L
                                + b["ln_pop"] * p.ln_pop.mean()))
for L, h, dx, dy, ha in [(Lmin, fitted(Lmin), 8, 8, "left"),
                         (Lmax, fitted(Lmax), -8, 10, "right")]:
    axB.plot([L], [h], marker="*", ms=18, color="#b7791f", mec="#7b5310", zorder=5)
    axB.annotate(f"{L:.0f} cm: {h:.0f} h", (L, h), xytext=(dx, dy),
                 textcoords="offset points", ha=ha, fontsize=11.5,
                 color="#7b5310", fontweight="bold")
perc = 100 * (np.exp(10 * b["length_proj"]) - 1)
axB.annotate(rf"$+{perc:.0f}\%$ per 10 cm ($P = 0.004$)",
             xy=(0.03, 0.97), xycoords="axes fraction", va="top",
             fontsize=11.5, color="#333333")
axB.set_ylim(0, None)
axB.set_xlabel("Projected total length (cm)")
axB.set_ylabel("Hours, adjusted to mean population size")
axB.spines[["top", "right"]].set_visible(False)
axB.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.80), fontsize=10.5)
axB.text(-0.10, 1.02, "B", transform=axB.transAxes, fontsize=17, fontweight="bold")

fig.tight_layout()
path = os.path.join(OUT, "Fig3_aggregation_tenure.png")
fig.savefig(path, dpi=200, facecolor="white")
print("saved", path)
