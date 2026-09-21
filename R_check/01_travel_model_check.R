# Cross-language verification of the travel-distance model (see
# analysis/02_travel_distance/travel_model.py, which is the analysis of record).
#
# Refits the AIC-selected linear mixed model -- log(distance) ~ standardized
# population size + standardized log(detections) + random intercept per fish,
# maximum likelihood -- in lme4, from the same fish-season table the Python
# pipeline writes (output/travel_fish_seasons.csv). Run the Python pipeline
# first (travel_distance.py) so that file exists.
#
# Conventions matched to the Python analysis: predictors standardized with the
# n-1 sample SD; fish-seasons with zero cumulative distance (FSA-only
# residents) excluded; missing lengths dropped so every model sees the same
# rows; confidence intervals and P values are Wald/normal, as reported by
# statsmodels. Coefficients and AIC agree to the printed precision; the two
# implementations compute Wald standard errors slightly differently, so CI
# endpoints and P values can move in the third decimal (e.g. CI upper bound
# -0.016 here vs -0.014 in Python). That is a convention difference, not a
# result difference.
#
# Python (statsmodels) values to reproduce, n = 153 fish-seasons, 77 fish:
#   population size   -0.193  [-0.371, -0.014]   P = 0.034
#   log(detections)   +0.197  [+0.051, +0.343]   P = 0.0083
#   dAIC of the same model without population size: +2.3
#
# Usage:  Rscript 01_travel_model_check.R     (from this directory)

suppressPackageStartupMessages(library(lme4))

d <- read.csv(file.path("..", "output", "travel_fish_seasons.csv"))
d <- d[d$distance_km > 0 & !is.na(d$length), ]
d$y <- log(d$distance_km)
d$P <- as.numeric(scale(d$popsize))
d$N <- as.numeric(scale(log(d$n_det)))
d$Lp <- as.numeric(scale(d$length_proj))
d$L <- as.numeric(scale(d$length))
cat(sprintf("model set n = %d, fish = %d\n\n", nrow(d), length(unique(d$animal_id))))

wald <- function(m) {
  co <- summary(m)$coefficients
  data.frame(estimate = co[, "Estimate"], se = co[, "Std. Error"],
             lo = co[, "Estimate"] - qnorm(0.975) * co[, "Std. Error"],
             hi = co[, "Estimate"] + qnorm(0.975) * co[, "Std. Error"],
             P = 2 * pnorm(-abs(co[, "Estimate"] / co[, "Std. Error"])))
}

models <- list(
  "popsize + log(n_det) + (1|fish)"               = lmer(y ~ P + N + (1 | animal_id), d, REML = FALSE),
  "proj length + popsize + log(n_det) + (1|fish)" = lmer(y ~ Lp + P + N + (1 | animal_id), d, REML = FALSE),
  "length + popsize + log(n_det) + (1|fish)"      = lmer(y ~ L + P + N + (1 | animal_id), d, REML = FALSE),
  "log(n_det) + (1|fish)"                         = lmer(y ~ N + (1 | animal_id), d, REML = FALSE),
  "popsize + (1|fish)"                            = lmer(y ~ P + (1 | animal_id), d, REML = FALSE),
  "(1|fish)"                                      = lmer(y ~ 1 + (1 | animal_id), d, REML = FALSE)
)
aics <- sapply(models, AIC)
tab <- data.frame(model = names(aics), AIC = round(aics, 1),
                  dAIC = round(aics - min(aics), 1), row.names = NULL)
cat("-- model selection (mixed models refit in lme4; ML) --\n")
print(tab[order(tab$AIC), ], row.names = FALSE)

cat("\nSelected model: popsize + log(n_det) + (1|fish)\n")
w <- wald(models[[1]])
for (v in c("P", "N")) {
  nm <- c(P = "population size", N = "log(detections)")[v]
  cat(sprintf("  %-16s %+.3f [%+.3f, %+.3f]  P = %.4g\n",
              nm, w[v, "estimate"], w[v, "lo"], w[v, "hi"], w[v, "P"]))
}
cat("\nCompare with output/travel_model_summary.txt from the Python run.\n")
