# R_check: cross-language verification

The analyses of record for this paper are the Python scripts under
`analysis/` — those are what produced every number in the manuscript. The
scripts in this folder refit the key statistical models in R (`lme4` mixed
models, base-`glm` quasi-Poisson) as an independent cross-language check.
They are verification, not analysis: they read the same screened fish-season
tables the Python pipeline writes and confirm that the reported effects do
not depend on the software used to fit them.

## Contents

| script | verifies | reads |
| --- | --- | --- |
| `01_travel_model_check.R` | travel-distance mixed model (population-size effect on migration distance) | `output/travel_fish_seasons.csv` |
| `02_tenure_model_check.R` | aggregation-tenure mixed model (population-size and body-size effects on FSA hours), plus the Little Cayman-only refit | `output/tenure_fish_seasons.csv` |
| `03_spawn_timing_check.R` | quasi-Poisson trends in days-after-full-moon for the observed peak and onset series | `data/spawn_timing_observed.csv` |

## Running

Scripts 01 and 02 read fish-season tables produced by the Python pipeline,
so run that first (see the top-level README); script 03 reads packaged data
directly. Then, from this directory:

```
Rscript 01_travel_model_check.R
Rscript 02_tenure_model_check.R
Rscript 03_spawn_timing_check.R
```

Requires R (tested on 4.3) with the `lme4` package installed
(`install.packages("lme4")`); script 03 needs base R only.

## What agreement looks like

Each script's header lists the Python values it should reproduce, and each
prints a pointer to the corresponding `output/*_summary.txt` from the Python
run. Coefficients, dispersions, and AIC differences agree to the printed
precision. Two convention differences are expected and are noted in the
script headers: the mixed-model Wald standard errors are computed slightly
differently by `lme4` and `statsmodels`, so confidence-interval endpoints
and P values can shift in the third decimal; and R's `summary()` quotes
t-based P values for quasi-Poisson fits where `statsmodels` quotes normal
(z) ones, so script 03 prints both. Neither changes any conclusion.
