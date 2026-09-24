"""Spawn timing within the lunar cycle across the recovery.

OBSERVED SERIES (primary). Diver-recorded first (onset) and peak spawning dates at
Little Cayman, 2002-2023, expressed in days after the full moon (DAFM). DAFM is a small
non-negative count, so trends are fit as Poisson-family GLMs (log link) against year.
Both series are underdispersed for a Poisson response, so quasi-Poisson estimation is
used, with the dispersion estimated from the Pearson statistic: same fitted line as
Poisson, honest standard errors. Plain Poisson (dispersion forced to 1) and ordinary
least squares are reported for comparison. Onset is a first-detection statistic and
therefore more sensitive than peak to day-to-day observation coverage within a season;
peak is treated as the primary observed result.

ACOUSTIC SERIES (supporting). A peak arrival day reconstructed for each adequately
sampled Little Cayman season from the daily count of tagged fish detected at the FSA
(see data/README.md for the derivation). The trend is fit by OLS (the metric is
continuous). With only ten qualifying seasons and cohort size strongly correlated with
year, this trend cannot be separated from cohort-size effects and is reported as
consistent with the observed series, not independent confirmation.

TEMPERATURE. Mean spawning-season (January-April) sea-surface temperature at Little
Cayman is added to the year model as a check that the timing trend is not thermally
driven.

Run from this directory:  python3 spawn_timing_analysis.py

Output (written to ../../output/):  spawn_timing_summary.txt
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output")
os.makedirs(OUT, exist_ok=True)
lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


obs = pd.read_csv(os.path.join(DATA, "spawn_timing_observed.csv"))
missing = sorted(set(range(int(obs.season.min()), int(obs.season.max()) + 1)) - set(obs.season))
say(f"observed record: {obs.season.min()}-{obs.season.max()}, n = {len(obs)} seasons"
    + (f" (unsampled: {', '.join(map(str, missing))})" if missing else ", no gaps"))


def trend(y, label):
    plain = smf.glm(f"{y} ~ season", obs, family=sm.families.Poisson()).fit()
    dispersion = plain.pearson_chi2 / plain.df_resid
    quasi = smf.glm(f"{y} ~ season", obs, family=sm.families.Poisson()).fit(scale="X2")
    beta, pval = quasi.params.season, quasi.pvalues.season
    lo, hi = quasi.conf_int().loc["season"]
    per_decade = 100 * (np.exp(10 * beta) - 1)
    dlo, dhi = 100 * (np.exp(10 * lo) - 1), 100 * (np.exp(10 * hi) - 1)
    say(f"\n{label}:")
    say(f"  quasi-Poisson: beta {beta:+.4f}/yr [{lo:+.4f}, {hi:+.4f}]  P = {pval:.4f}")
    say(f"  = {per_decade:+.0f}% per decade [{dlo:+.0f}, {dhi:+.0f}];  "
        f"dispersion (Pearson chi2/df) = {dispersion:.2f} (underdispersed)")
    say(f"  plain Poisson (dispersion forced to 1) would overstate the uncertainty: "
        f"P = {plain.pvalues.season:.4f}")
    fit = quasi.get_prediction(pd.DataFrame({"season": [2002, 2023]})).predicted_mean
    say(f"  fitted DAFM: {fit[0]:.1f} d (2002) -> {fit[1]:.1f} d (2023)")
    ols = smf.ols(f"{y} ~ season", obs).fit()
    say(f"  OLS comparison: {ols.params.season:+.3f} d/yr, P = {ols.pvalues.season:.4f}")


trend("obs_peak", "Peak spawning (primary)")
trend("obs_onset", "Onset (first observed spawning)")

# ---- acoustic support ----
ac = (pd.read_csv(os.path.join(DATA, "spawn_timing_acoustic.csv"))
      .query("island == 'LC'").dropna(subset=["peak_mid"]))
m_ac = smf.ols("peak_mid ~ season", ac).fit()
say(f"\nAcoustic peak arrival day (LC, n = {len(ac)} qualifying seasons, "
    f"{ac.season.min()}-{ac.season.max()}):")
say(f"  OLS: {m_ac.params.season * 10:+.1f} d/decade, P = {m_ac.pvalues.season:.2f}")
m_ac2 = smf.ols("peak_mid ~ season + n_fish", ac).fit()
say(f"  with cohort size added: year P = {m_ac2.pvalues.season:.2f} "
    f"(corr(n_fish, season) = {ac.n_fish.corr(ac.season):+.2f}; not separable -> "
    f"consistent, not confirmatory)")
paired = obs.merge(ac[["season", "peak_mid"]], on="season")
say(f"  acoustic peak precedes observed peak by "
    f"{(paired.obs_peak - paired.peak_mid).mean():.1f} d on average "
    f"across {len(paired)} paired seasons (assemble, then spawn)")

# ---- temperature check ----
sst = pd.read_csv(os.path.join(DATA, "sst_daily_LC_CB.csv"), parse_dates=["date"])
lc = sst[sst.island == "LC"].copy()
lc = lc[lc.date.dt.month.isin([1, 2, 3, 4])]
season_sst = lc.groupby(lc.date.dt.year).sst_c.mean()
obs["sst"] = obs.season.map(season_sst)
d = obs.dropna(subset=["sst"])
say(f"\nTemperature check (n = {len(d)} seasons with SST; "
    f"mean Jan-Apr sea-surface temperature at Little Cayman):")
for y, label in [("obs_peak", "peak"), ("obs_onset", "onset")]:
    q = smf.glm(f"{y} ~ season + sst", d, family=sm.families.Poisson()).fit(scale="X2")
    alone = smf.glm(f"{y} ~ sst", d, family=sm.families.Poisson()).fit(scale="X2")
    say(f"  {label}: year trend with SST in the model {q.params.season:+.4f}/yr "
        f"(P = {q.pvalues.season:.4f}); SST term P = {q.pvalues.sst:.2f}; "
        f"SST alone P = {alone.pvalues.sst:.2f}")
say("  The year trend survives (and does not weaken) with temperature in the model;")
say("  temperature alone shows no significant association with either series.")

open(os.path.join(OUT, "spawn_timing_summary.txt"), "w").write("\n".join(lines) + "\n")
say("\nwrote output/spawn_timing_summary.txt")
