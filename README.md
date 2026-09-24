# Grouper Moon: behavioral reorganization across the recovery of a Nassau Grouper metapopulation

Data and analysis code for:

> Semmens BX, McCoy CM, Pattengill-Semmens CV, Heppell SA, Johnson BC, Candelmo A,
> Waterhouse L, Bush PG. *Cultural reorganization across two decades of recovery in a
> critically endangered reef fish.* (in review)

Two decades (2005-2025) of whole-island passive acoustic telemetry of Nassau Grouper
(*Epinephelus striatus*) at Little Cayman and Cayman Brac, Cayman Islands, collected by
the Grouper Moon Project (Reef Environmental Education Foundation and the Cayman Islands
Department of Environment).

## Repository layout

```
data/                          packaged data (see data/README.md for the dictionary)
analysis/
  00_shared/gmp_shared.py      data loading + the fish-season inclusion rule
                               (imported by every analysis; the rule lives here once)
  01_attendance/               attendance analysis + manuscript Fig. 1 / fig. S2
  02_travel_distance/          travel-distance metric, models + Fig. 2
  03_aggregation_tenure/       hours-at-FSA dataset, models + Fig. 3
  04_spawn_timing/             spawn-timing trends, acoustic support, SST check + Fig. 4
  05_map_and_movie/            study-site map (fig. S1) + movement animation (Movie S1)
output/                        everything the scripts write (tables, figures, summaries)
R_check/                       independent R refits of the key models (verification)
```

## Running the analyses

Python >= 3.10 with the packages in `requirements.txt`
(`pip install -r requirements.txt`). Each script is run from its own directory and
writes to `output/`. Within a folder, run scripts in the order listed:

```
cd analysis/01_attendance        && python3 attendance_analysis.py && python3 attendance_figure.py
cd ../02_travel_distance         && python3 travel_distance.py && python3 travel_model.py && python3 travel_figure.py
cd ../03_aggregation_tenure      && python3 tenure_dataset.py && python3 tenure_model.py && python3 tenure_figure.py
cd ../04_spawn_timing            && python3 spawn_timing_analysis.py && python3 spawn_timing_figure.py
cd ../05_map_and_movie           && python3 site_map_figure.py && python3 movement_animation.py
```

Notes: the site map's Caribbean locator inset requires the `basemap` toolkit and is
skipped with a notice if it is not installed; the movement animation requires `ffmpeg`
on the PATH (`--still` renders two PNG frames instead of the full movie).

## Headline results the code reproduces

| Analysis | Result |
|---|---|
| Attendance | 156/158 fish-seasons attended = 98.7% (95% CI 95.5-99.7) |
| Travel distance | standardized population-size effect on log distance -0.193 (95% CI -0.371 to -0.014), P = 0.034; n = 153 migratory fish-seasons |
| Aggregation tenure | -23% hours per doubling of population size (95% CI -31 to -13), P = 3.4e-05; +26% per 10 cm projected length (95% CI +8 to +47), P = 0.004 |
| Spawn timing | peak spawning -23% DAFM per decade (P = 0.002); onset -34% per decade (P = 0.0001); acoustic arrival -1.3 d/decade (P = 0.06, consistent, not confirmatory) |

Each model script's docstring states its expected output, and the written summaries in
`output/` carry the full numbers.

The scripts in `R_check/` refit the key models in R (`lme4`, base `glm`) as an
independent cross-language check; they reproduce the values above from the same
screened fish-season tables. See `R_check/README.md`.

## Provenance

Analyses were originally developed in R by B. X. Semmens; this repository
implements them in Python (ported and verified with AI assistance), with R
cross-checks of the key models under `R_check/`.

## Data use

The underlying data were collected under research permits from the Cayman Islands
Department of Environment. Please contact the corresponding author
(bsemmens@ucsd.edu) before reusing the data in other work, and cite the paper above.
