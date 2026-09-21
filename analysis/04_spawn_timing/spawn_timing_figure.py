"""Figure: spawning has compressed toward the full moon (manuscript Fig. 4).

Panel A: observed first (onset) and peak spawning at Little Cayman, 2002-2023, in days
after the full moon, with quasi-Poisson fits and 95% CIs. Panel B: acoustic peak arrival
day of tagged fish with OLS fit and 95% CI; diver-observed peaks from panel A shown in
gray for comparison.

Run from this directory:  python3 spawn_timing_figure.py

Output (written to ../../output/):  Fig4_spawn_timing.png
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output")
os.makedirs(OUT, exist_ok=True)

C_PEAK, C_ONSET, INK, MUT = "#2b6cb0", "#c05621", "#1a202c", "#718096"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

obs = pd.read_csv(os.path.join(DATA, "spawn_timing_observed.csv"))
ac = (pd.read_csv(os.path.join(DATA, "spawn_timing_acoustic.csv"))
      .query("island == 'LC'").dropna(subset=["peak_mid"]))

xs = np.linspace(obs.season.min() - 0.5, obs.season.max() + 0.5, 120)
m_pk = smf.glm("obs_peak ~ season", obs, family=sm.families.Poisson()).fit(scale="X2")
m_on = smf.glm("obs_onset ~ season", obs, family=sm.families.Poisson()).fit(scale="X2")
pr_pk = m_pk.get_prediction(pd.DataFrame({"season": xs})).summary_frame()
pr_on = m_on.get_prediction(pd.DataFrame({"season": xs})).summary_frame()

fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=200, sharey=True,
                           gridspec_kw={"width_ratios": [1.45, 1]})
for ax in (a, b):
    for sp in ax.spines.values():
        sp.set_color("#333333")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUT, linewidth=1, linestyle=":")

a.fill_between(xs, pr_pk.mean_ci_lower, pr_pk.mean_ci_upper, color=C_PEAK, alpha=0.15, lw=0)
a.fill_between(xs, pr_on.mean_ci_lower, pr_on.mean_ci_upper, color=C_ONSET, alpha=0.15, lw=0)
a.plot(xs, pr_pk["mean"], color=C_PEAK, linewidth=2)
a.plot(xs, pr_on["mean"], color=C_ONSET, linewidth=2, linestyle="--")
a.scatter(obs.season, obs.obs_peak, s=42, color=C_PEAK, zorder=3,
          edgecolor="white", linewidth=0.8, label="peak spawning")
a.scatter(obs.season, obs.obs_onset, s=46, color=C_ONSET, marker="^", zorder=3,
          edgecolor="white", linewidth=0.8, label="first spawning")
pk_dec = 100 * (np.exp(10 * m_pk.params.season) - 1)
on_dec = 100 * (np.exp(10 * m_on.params.season) - 1)
a.annotate(f"peak  {pk_dec:+.0f} %/decade  (P = {m_pk.pvalues.season:.3f})",
           (0.03, 0.95), xycoords="axes fraction", color=C_PEAK, fontsize=10, va="top")
a.annotate(f"first  {on_dec:+.0f} %/decade  (P = {m_on.pvalues.season:.3f})",
           (0.03, 0.88), xycoords="axes fraction", color=C_ONSET, fontsize=10, va="top")
a.set_xlabel("Year")
a.set_ylabel("Days after full moon")
a.text(-0.075, 1.02, "A", transform=a.transAxes, fontsize=15, fontweight="bold")
a.legend(loc="lower left", frameon=False, fontsize=9)
a.annotate("full moon", (obs.season.max() + 0.4, 0.08), fontsize=8.5, color=MUT,
           va="bottom", ha="right")

m_ac = smf.ols("peak_mid ~ season", ac).fit()
xs2 = np.linspace(ac.season.min() - 0.5, ac.season.max() + 0.5, 120)
pr_ac = m_ac.get_prediction(pd.DataFrame({"season": xs2})).summary_frame()
b.fill_between(xs2, pr_ac.mean_ci_lower, pr_ac.mean_ci_upper, color=C_PEAK, alpha=0.15, lw=0)
b.plot(xs2, pr_ac["mean"], color=C_PEAK, linewidth=2)
b.scatter(ac.season, ac.peak_mid, s=42, color=C_PEAK, zorder=3,
          edgecolor="white", linewidth=0.8, label="acoustic peak day (tagged fish)")
b.scatter(obs.season, obs.obs_peak, s=26, color=MUT, alpha=0.45, zorder=2,
          label="diver-observed peak (panel A)")
b.annotate(f"{m_ac.params.season * 10:+.1f} d/decade (P = {m_ac.pvalues.season:.2f})",
           (0.03, 0.95), xycoords="axes fraction", color=INK, fontsize=10, va="top")
b.set_xlabel("Year")
b.text(-0.06, 1.02, "B", transform=b.transAxes, fontsize=15, fontweight="bold")
b.legend(loc="lower left", frameon=False, fontsize=9)

fig.tight_layout()
path = os.path.join(OUT, "Fig4_spawn_timing.png")
fig.savefig(path, facecolor="white")
print("saved", path)
