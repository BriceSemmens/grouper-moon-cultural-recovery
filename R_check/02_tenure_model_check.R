# Cross-language verification of the aggregation-tenure model (see
# analysis/03_aggregation_tenure/tenure_model.py, which is the analysis of
# record).
#
# Refits the reporting model -- ln(hours at the FSA) ~ growth-projected length
# + ln(population size) + random year intercept, maximum likelihood -- in
# lme4, from the fish-season table the Python pipeline writes
# (output/tenure_fish_seasons.csv). Run the Python pipeline first
# (tenure_dataset.py) so that file exists. Also refits the Little
# Cayman-only model reported in the supplement.
#
# Confidence intervals and P values are Wald/normal, matching the statsmodels
# convention. Coefficients agree to the printed precision; Wald standard
# errors are computed slightly differently by the two implementations, so CI
# endpoints and P values can move in the third decimal.
#
# Python (statsmodels) values to reproduce, n = 156 fish-seasons:
#   projected length  +0.0230  [+0.0075, +0.0386]   P = 0.0037
#   ln(popsize)       -0.3697  [-0.5445, -0.1950]   P = 3.4e-05
#   per doubling of population size: -23% ; per 10 cm projected length: +26%
#   Little Cayman only (n = 134): ln(popsize) -0.4909 [-0.9175, -0.0642], P = 0.024
#
# Usage:  Rscript 02_tenure_model_check.R     (from this directory)

suppressPackageStartupMessages(library(lme4))

d <- read.csv(file.path("..", "output", "tenure_fish_seasons.csv"))
d$ln_hours <- log(d$hours)
d$ln_pop <- log(d$popsize)
d$year_f <- factor(d$year)
cat(sprintf("fish-seasons: %d   fish: %d   years: %d\n\n",
            nrow(d), length(unique(d$animal_id)), nlevels(d$year_f)))

wald_line <- function(m, v, label) {
  co <- summary(m)$coefficients
  est <- co[v, "Estimate"]; se <- co[v, "Std. Error"]
  lo <- est - qnorm(0.975) * se; hi <- est + qnorm(0.975) * se
  p <- 2 * pnorm(-abs(est / se))
  cat(sprintf("  %-17s %+.4f [%+.4f, %+.4f]  P = %.4g\n", label, est, lo, hi, p))
  invisible(c(est = est, lo = lo, hi = hi))
}

m <- lmer(ln_hours ~ length_proj + ln_pop + (1 | year_f), d, REML = FALSE)
cat("Reporting model: projected length + ln(popsize) + (1|year)\n")
wald_line(m, "length_proj", "projected length")
pp <- wald_line(m, "ln_pop", "ln(popsize)")
vc <- as.data.frame(VarCorr(m))
cat(sprintf("  random year SD %.3f, residual SD %.3f\n",
            vc$sdcor[vc$grp == "year_f"], vc$sdcor[vc$grp == "Residual"]))
cat(sprintf("  per doubling of population size: %+.0f%% [%+.0f, %+.0f]\n",
            100 * (exp(log(2) * pp["est"]) - 1),
            100 * (exp(log(2) * pp["lo"]) - 1),
            100 * (exp(log(2) * pp["hi"]) - 1)))
lp <- summary(m)$coefficients["length_proj", "Estimate"]
cat(sprintf("  per 10 cm projected length: %+.0f%%\n", 100 * (exp(10 * lp) - 1)))

lc <- d[d$island == "LC", ]
mlc <- lmer(ln_hours ~ length_proj + ln_pop + (1 | year_f), lc, REML = FALSE)
cat(sprintf("\nLittle Cayman only (n = %d):\n", nrow(lc)))
wald_line(mlc, "ln_pop", "ln(popsize)")

cat("\nCompare with output/tenure_model_summary.txt from the Python run.\n")
