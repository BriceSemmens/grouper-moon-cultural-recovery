"""Figure: spawning-season travel distances across the recovery (manuscript Fig. 2).

Main panel: cumulative travel distance for every migratory fish-season, ranked and
colored by the spawning population size of the fish's island-season, with silhouettes of
Little Cayman and Cayman Brac drawn at the same scale as the x axis. Inset: distance
against population size on logarithmic axes, with points and fitted line rendered as
partial residuals at mean detection effort from the selected model (the same model
reported by travel_model.py).

Run AFTER travel_distance.py, from this directory:  python3 travel_figure.py

Output (written to ../../output/):  Fig2_travel_distance.png
"""
import os
import pickle

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")

SURF, INK, INK2, GRID = "#ffffff", "#111111", "#4a4a4a", "#e6e6e6"
LAND, LAND_EDGE = "#9aa38f", "#6f7766"
COL = {"LC": "#2b6cb0", "CB": "#c05621"}
MK = {"LC": "o", "CB": "^"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "axes.edgecolor": "#444444", "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK})

a = (pd.read_csv(os.path.join(OUT, "travel_fish_seasons.csv"))
     .query("distance_km > 0")                    # residents excluded (see travel_model.py)
     .sort_values("distance_km").reset_index(drop=True))
n = len(a)

# island coastlines for the scale silhouettes
cl = pd.read_csv(os.path.join(DATA, "coastlines_LC_CB.csv"))
polys = [(g.lon.to_numpy(), g.lat.to_numpy()) for _, g in cl.groupby("polygon_id")]

# ---- the selected model, refit here so the inset is self-contained ----
d = a.copy()
d["y"] = np.log(d.distance_km)
mu_p, sd_p = d.popsize.mean(), d.popsize.std()
d["P"] = (d.popsize - mu_p) / sd_p
d["N"] = (np.log(d.n_det) - np.log(d.n_det).mean()) / np.log(d.n_det).std()
best = smf.mixedlm("y ~ P + N", d, groups=d.animal_id).fit(reml=False)
V = best.cov_params().loc[["Intercept", "P"], ["Intercept", "P"]].to_numpy()
b0, bP = best.params["Intercept"], best.params["P"]
d["y_adj"] = d.y - best.params["N"] * d.N        # partial residuals at mean effort

# ---- layout ----
LAT0 = np.mean([p[1].mean() for p in polys])
LON0 = min(p[0].min() for p in polys)
KMX = 111.320 * np.cos(np.radians(LAT0))
KMY = 110.574
isl = [((lon - LON0) * KMX, (lat - LAT0) * KMY) for lon, lat in polys]

XMAX = float(a.distance_km.max()) * 1.06
X0 = 3.0
FIGW, FIGH = 7.6, 9.4
fig, ax = plt.subplots(figsize=(FIGW, FIGH), dpi=200, facecolor=SURF)
ax.set_facecolor(SURF)
norm = matplotlib.colors.Normalize(a.popsize.min(), a.popsize.max())
cmap = matplotlib.cm.viridis
ax.barh(np.arange(n), a.distance_km, height=0.82, color=cmap(norm(a.popsize)), linewidth=0)
Y_BOT = -0.115 * n
ax.set_xlim(0, XMAX)
ax.set_ylim(Y_BOT, n)
ax.set_yticks([0, 40, 80, 120, 153])
ax.set_xlabel("Distance traveled during the spawning season (km)", fontsize=13)
ax.set_ylabel("Tagged fish-season, ranked", fontsize=13)
ax.grid(True, axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                  fraction=0.045, pad=0.02, shrink=0.42, anchor=(0.0, 0.985))
cb.set_label("Aggregation size", fontsize=11)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=9)
fig.tight_layout()
fig.canvas.draw()
bb = ax.get_position()
axw_in, axh_in = bb.width * FIGW, bb.height * FIGH
y_per_km = (axw_in * (n - Y_BOT)) / (XMAX * axh_in)   # isotropic km on both axes

isl_h = max(y.max() - y.min() for _, y in isl) * y_per_km
Y_ISL = -0.050 * n - isl_h / 2
for x, y in isl:
    ax.add_patch(Polygon(np.column_stack([X0 + x, Y_ISL + y * y_per_km]), closed=True,
                         facecolor=LAND, edgecolor=LAND_EDGE, linewidth=0.7, zorder=4))
ax.axhline(0, color="#d8d8d8", linewidth=0.8, zorder=2)
ax.set_ylim(Y_BOT, n)

# ---- inset: distance vs population size, partial residuals ----
axi = ax.inset_axes([0.40, 0.26, 0.58, 0.40])
axi.set_facecolor("#fcfcfc")
rng = np.random.default_rng(7)                       # jitter for overplotted x values
for island in ["LC", "CB"]:
    g = d[d.island == island]
    x = g.popsize * np.exp(rng.normal(0, 0.035, len(g)))
    axi.scatter(x, np.exp(g.y_adj), marker=MK[island], s=17, alpha=0.6,
                facecolor=COL[island], edgecolor="white", linewidth=0.3,
                label="Little Cayman" if island == "LC" else "Cayman Brac")
px = np.linspace(np.log(a.popsize.min()), np.log(a.popsize.max()), 80)
Pz = (np.exp(px) - mu_p) / sd_p
eta = b0 + bP * Pz
se = np.sqrt(V[0, 0] + Pz ** 2 * V[1, 1] + 2 * Pz * V[0, 1])
axi.fill_between(np.exp(px), np.exp(eta - 1.96 * se), np.exp(eta + 1.96 * se),
                 color="#4a5568", alpha=0.18, linewidth=0, zorder=3)
axi.plot(np.exp(px), np.exp(eta), color="#4a5568", linewidth=1.8, linestyle="--", zorder=4)
axi.set_xscale("log")
axi.set_yscale("log")
axi.set_xticks([520, 2000, 4000, 8000], ["520", "2,000", "4,000", "8,000"], fontsize=8)
axi.set_yticks([2, 10, 50, 235], ["2", "10", "50", "235"], fontsize=8)
axi.minorticks_off()
axi.tick_params(length=2.5)
axi.set_xlabel("Aggregation size (fish)", fontsize=9, labelpad=1.5)
axi.set_ylabel("Distance (km, log scale)", fontsize=9, labelpad=1.5)
axi.grid(True, color="#ececec", linewidth=0.6)
axi.set_axisbelow(True)
for s_ in ("top", "right"):
    axi.spines[s_].set_visible(False)
axi.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.14, 0.0),
           handletextpad=0.1, borderaxespad=0.2, labelspacing=0.25)
axi.annotate(f"population size β = {bP:+.2f}, P = {best.pvalues['P']:.3f}",
             xy=(0.97, 0.965), xycoords="axes fraction", ha="right", va="top",
             fontsize=8.2, color=INK2)
path = os.path.join(OUT, "Fig2_travel_distance.png")
fig.savefig(path, facecolor=SURF)
print("saved", path)
