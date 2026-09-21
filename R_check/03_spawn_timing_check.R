# Cross-language verification of the spawn-timing trends (see
# analysis/04_spawn_timing/spawn_timing_analysis.py, which is the analysis of
# record).
#
# Refits the quasi-Poisson trends in days-after-full-moon (DAFM) for the
# observed peak and onset series, directly from data/spawn_timing_observed.csv
# (no Python outputs needed). R's quasipoisson family estimates the dispersion
# from the Pearson statistic, the same estimator statsmodels uses with
# scale="X2", so coefficients, standard errors, and dispersions should agree
# essentially exactly. One convention difference: R's summary() quotes t-based
# P values for quasi families while statsmodels quotes normal (z) ones, so
# both are printed; with n = 22 seasons they differ slightly, and the z line
# is the one to compare against the Python output.
#
# Python (statsmodels) values to reproduce (1x per-year log-scale slopes):
#   peak   -0.0275/yr [-0.0469, -0.0081]  P(z) = 0.0054   dispersion 0.40
#   onset  -0.0415/yr [-0.0652, -0.0178]  P(z) = 0.0006   dispersion 0.46
#   OLS comparison: peak -0.128 d/yr (P = 0.010); onset -0.152 d/yr (P = 0.003)
#
# Usage:  Rscript 03_spawn_timing_check.R     (from this directory)

obs <- read.csv(file.path("..", "data", "spawn_timing_observed.csv"))
cat(sprintf("observed record: %d-%d, n = %d seasons\n\n",
            min(obs$season), max(obs$season), nrow(obs)))

trend <- function(y, label) {
  q <- glm(reformulate("season", y), data = obs, family = quasipoisson())
  co <- summary(q)$coefficients
  est <- co["season", "Estimate"]; se <- co["season", "Std. Error"]
  pz <- 2 * pnorm(-abs(est / se))                    # normal convention (statsmodels)
  pt <- co["season", "Pr(>|t|)"]                     # R's native t convention
  cat(sprintf("%s:\n", label))
  cat(sprintf("  quasi-Poisson: beta %+.4f/yr [%+.4f, %+.4f]  P(z) = %.4f  (R t-based P = %.4f)\n",
              est, est - qnorm(0.975) * se, est + qnorm(0.975) * se, pz, pt))
  cat(sprintf("  = %+.0f%% per decade;  dispersion (Pearson chi2/df) = %.2f\n",
              100 * (exp(10 * est) - 1), summary(q)$dispersion))
  o <- lm(reformulate("season", y), data = obs)
  cat(sprintf("  OLS comparison: %+.3f d/yr, P = %.4f\n\n",
              coef(o)["season"], summary(o)$coefficients["season", "Pr(>|t|)"]))
}

trend("obs_peak", "Peak spawning (primary)")
trend("obs_onset", "Onset (first observed spawning)")

cat("Compare with output/spawn_timing_summary.txt from the Python run.\n")
