"""Aggregation-tenure models: hours at the FSA vs population size and body size.

Response: ln(hours at the FSA per fish-season). Model selection is run on the
static-length model set; the reporting model substitutes growth-projected length into
the AIC-best structure (length + ln population size + random year intercept). The
random-year structure absorbs shared season-level conditions and, importantly, prevents
the partial collinearity between projected length and calendar time from loading onto
the population-size estimate.

Run AFTER tenure_dataset.py, from this directory:  python3 tenure_model.py

Outputs (written to ../../output/):
    tenure_model_selection.csv   the AIC table (static-length set)
    tenure_model_summary.txt     reporting-model coefficients and derived effect sizes

Expected headline results (n = 156 fish-seasons): ln(population size) -0.370
(95% CI -0.545 to -0.195; P = 3.4e-05), i.e. -23% hours per doubling of population size;
projected length +0.0230 per cm (95% CI +0.0075 to +0.0386; P = 0.0037), i.e. +26% per
10 cm. No sex effect. Fitted tenure at the observed size extremes: 116 h (48 cm) vs
274 h (85 cm), a 2.4-fold contrast.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output")
lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


d = pd.read_csv(os.path.join(OUT, "tenure_fish_seasons.csv"))
d["ln_hours"] = np.log(d.hours)
d["ln_pop"] = np.log(d.popsize)
d["year_f"] = d.year.astype(str)
say(f"fish-seasons: {len(d)}   fish: {d.animal_id.nunique()}   years: {d.year_f.nunique()}")

# ---- model selection on the static-length set ----
models = [
    ("length + log(popsize) + (1|year)", smf.mixedlm("ln_hours ~ length + ln_pop", d, groups=d.year_f).fit(reml=False)),
    ("length + log(popsize) + (1|fish)", smf.mixedlm("ln_hours ~ length + ln_pop", d, groups=d.animal_id).fit(reml=False)),
    ("length + log(popsize)",            smf.ols("ln_hours ~ length + ln_pop", d).fit()),
    ("log(popsize)",                     smf.ols("ln_hours ~ ln_pop", d).fit()),
    ("island",                           smf.ols("ln_hours ~ island", d).fit()),
    ("length",                           smf.ols("ln_hours ~ length", d).fit()),
]
tab = pd.DataFrame([(nm, m.aic) for nm, m in models], columns=["model", "AIC"])
tab["dAIC"] = (tab.AIC - tab.AIC.min()).round(1)
tab["AIC"] = tab.AIC.round(1)
tab = tab.sort_values("AIC").reset_index(drop=True)
say("\n-- model selection (response = ln hours; static length) --")
say(tab.to_string(index=False))
tab.to_csv(os.path.join(OUT, "tenure_model_selection.csv"), index=False)

# ---- reporting model: projected length in the AIC-best structure ----
m = smf.mixedlm("ln_hours ~ length_proj + ln_pop", d, groups=d.year_f).fit(reml=False)
ci = m.conf_int()
say("\nReporting model: projected length + ln(popsize) + (1|year)")
say(f"  projected length  {m.params.length_proj:+.4f} "
    f"[{ci.loc['length_proj', 0]:+.4f}, {ci.loc['length_proj', 1]:+.4f}]  "
    f"P = {m.pvalues.length_proj:.4g}")
say(f"  ln(popsize)       {m.params.ln_pop:+.4f} "
    f"[{ci.loc['ln_pop', 0]:+.4f}, {ci.loc['ln_pop', 1]:+.4f}]  P = {m.pvalues.ln_pop:.4g}")
say(f"  random year SD {np.sqrt(m.cov_re.iloc[0, 0]):.3f}, residual SD {np.sqrt(m.scale):.3f}")

pd_lo, pd_hi = np.exp(np.log(2) * ci.loc["ln_pop"]) - 1
say(f"  per doubling of population size: {100 * (np.exp(np.log(2) * m.params.ln_pop) - 1):+.0f}% "
    f"[{100 * pd_lo:+.0f}, {100 * pd_hi:+.0f}]")
cm_lo, cm_hi = np.exp(10 * ci.loc["length_proj"]) - 1
say(f"  per 10 cm projected length: {100 * (np.exp(10 * m.params.length_proj) - 1):+.0f}% "
    f"[{100 * cm_lo:+.0f}, {100 * cm_hi:+.0f}]")

Lmin, Lmax = d.length_proj.min(), d.length_proj.max()
fitted = lambda L: np.exp(m.params.Intercept + m.params.length_proj * L
                          + m.params.ln_pop * d.ln_pop.mean())
say(f"  fitted tenure at size extremes (mean popsize): "
    f"{Lmin:.0f} cm -> {fitted(Lmin):.0f} h, {Lmax:.0f} cm -> {fitted(Lmax):.0f} h "
    f"(ratio {fitted(Lmax) / fitted(Lmin):.2f})")

# pooled OLS comparison (static length)
o = smf.ols("ln_hours ~ length + ln_pop", d).fit()
say(f"\nPooled OLS, static length: length {o.params.length:+.4f} P = {o.pvalues.length:.4g}; "
    f"ln(popsize) {o.params.ln_pop:+.4f} P = {o.pvalues.ln_pop:.3g}")

# Little Cayman only (drops the cross-island contrast; the longitudinal signal remains)
lc = d[d.island == "LC"]
mlc = smf.mixedlm("ln_hours ~ length_proj + ln_pop", lc, groups=lc.year_f).fit(reml=False)
cilc = mlc.conf_int().loc["ln_pop"]
say(f"\nLittle Cayman only (n = {len(lc)}): ln(popsize) {mlc.params.ln_pop:+.4f} "
    f"[{cilc[0]:+.4f}, {cilc[1]:+.4f}]  P = {mlc.pvalues.ln_pop:.4g}")

# sex: likelihood-ratio test on the subset with sex recorded
ds = d[d.sex.astype(str).str.lower().isin(["m", "f"])].copy()
ds["Sex"] = ds.sex.str.lower()
m0 = smf.mixedlm("ln_hours ~ length_proj + ln_pop", ds, groups=ds.year_f).fit(reml=False)
m1 = smf.mixedlm("ln_hours ~ length_proj + ln_pop + Sex", ds, groups=ds.year_f).fit(reml=False)
lrt = 2 * (m1.llf - m0.llf)
say(f"\nSex (n = {len(ds)}): LRT chi2 = {lrt:.2f}, P = {stats.chi2.sf(lrt, 1):.3f}; "
    f"median hours {ds.groupby('Sex').hours.median().to_dict()}")

open(os.path.join(OUT, "tenure_model_summary.txt"), "w").write("\n".join(lines) + "\n")
say("\nwrote output/tenure_model_selection.csv and tenure_model_summary.txt")
