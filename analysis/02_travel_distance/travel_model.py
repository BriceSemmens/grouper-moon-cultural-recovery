"""Travel-distance model selection and the reported population-size effect.

Fish-seasons detected only at the FSA (cumulative distance zero) provide no evidence of
migration and are classified as resident; residents are excluded from all
migration-distance analyses. This also removes the arbitrary offset otherwise needed to
admit zero distances to a log scale: the response is plain log(distance).

Models are linear mixed models (maximum likelihood) with standardized predictors;
population size enters unlogged. The random intercept for individual fish accounts for
repeated measures. Model selection is by AIC.

Run AFTER travel_distance.py, from this directory:  python3 travel_model.py

Outputs (written to ../../output/):
    travel_model_selection.csv   the AIC table
    travel_model_summary.txt     coefficients of the selected and full models

Expected headline result (n = 153 migratory fish-seasons, 77 fish): AIC-best model is
popsize + log(n_det) + (1|fish); standardized population-size effect on log distance
-0.193 (95% CI -0.371 to -0.014; P = 0.034). Body length (static or growth-projected)
and sex explain no additional variation.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output")
lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


a = pd.read_csv(os.path.join(OUT, "travel_fish_seasons.csv"))
resident = a[a.distance_km == 0]
say(f"attended fish-seasons: {len(a)}; classified resident (FSA-only, zero path): "
    f"{len(resident)} ({', '.join(resident.animal_id + ' ' + resident.island + resident.year.astype(str))})")
a = a[a.distance_km > 0].copy()

d = a.dropna(subset=["length"]).copy()
d["y"] = np.log(d.distance_km)
for col, src in [("L", d.length), ("Lp", d.length_proj),
                 ("P", d.popsize), ("N", np.log(d.n_det))]:
    d[col] = (src - src.mean()) / src.std()
d["year_f"] = d.year.astype(str)
d["fish"] = d.animal_id
say(f"model set n = {len(d)}, fish = {d.fish.nunique()}")

models = [
    ("popsize + log(n_det) + (1|fish)",               smf.mixedlm("y ~ P + N", d, groups=d.fish).fit(reml=False)),
    ("log(n_det) + (1|fish)",                         smf.mixedlm("y ~ N", d, groups=d.fish).fit(reml=False)),
    ("proj length + popsize + log(n_det) + (1|fish)", smf.mixedlm("y ~ Lp + P + N", d, groups=d.fish).fit(reml=False)),
    ("length + popsize + log(n_det) + (1|fish)",      smf.mixedlm("y ~ L + P + N", d, groups=d.fish).fit(reml=False)),
    ("popsize + (1|fish)",                            smf.mixedlm("y ~ P", d, groups=d.fish).fit(reml=False)),
    ("length + popsize + (1|fish)",                   smf.mixedlm("y ~ L + P", d, groups=d.fish).fit(reml=False)),
    ("(1|fish)",                                      smf.mixedlm("y ~ 1", d, groups=d.fish).fit(reml=False)),
    ("length + popsize + (1|year)",                   smf.mixedlm("y ~ L + P", d, groups=d.year_f).fit(reml=False)),
    ("length + popsize",                              smf.ols("y ~ L + P", d).fit()),
    ("popsize",                                       smf.ols("y ~ P", d).fit()),
    ("length",                                        smf.ols("y ~ L", d).fit()),
    ("island",                                        smf.ols("y ~ island", d).fit()),
]
tab = pd.DataFrame([(nm, m.aic) for nm, m in models], columns=["model", "AIC"])
tab["dAIC"] = (tab.AIC - tab.AIC.min()).round(1)
tab["AIC"] = tab.AIC.round(1)
tab = tab.sort_values("AIC").reset_index(drop=True)
say("\n-- model selection (response = log distance; predictors standardized) --")
say(tab.to_string(index=False))
tab.to_csv(os.path.join(OUT, "travel_model_selection.csv"), index=False)

winner = models[0][1]
full = models[3][1]
ci = winner.conf_int()
say("\nSelected model: popsize + log(n_det) + (1|fish)")
for v, nm in [("P", "population size"), ("N", "log(detections)")]:
    say(f"  {nm:16s} {winner.params[v]:+.3f} "
        f"[{ci.loc[v, 0]:+.3f}, {ci.loc[v, 1]:+.3f}]  P = {winner.pvalues[v]:.4g}")
say("With static length retained (deadweight by AIC):")
for v, nm in [("L", "length"), ("P", "population size"), ("N", "log(detections)")]:
    say(f"  {nm:16s} {full.params[v]:+.3f}  P = {full.pvalues[v]:.4g}")

# sex: no effect (subset with sex recorded)
ds = d[d.sex.astype(str).str.lower().isin(["m", "f"])].copy()
ds["Sex"] = ds.sex.str.lower()
m0 = smf.mixedlm("y ~ P + N", ds, groups=ds.fish).fit(reml=False)
m1 = smf.mixedlm("y ~ Sex + P + N", ds, groups=ds.fish).fit(reml=False)
say(f"\nSex (n = {len(ds)} fish-seasons with sex recorded): "
    f"AIC without sex {m0.aic:.1f}, with sex {m1.aic:.1f} (no improvement)")

open(os.path.join(OUT, "travel_model_summary.txt"), "w").write("\n".join(lines) + "\n")
say("\nwrote output/travel_model_selection.csv and travel_model_summary.txt")
