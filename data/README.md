# Data dictionary

All files are plain CSV (one gzip-compressed), readable from any environment. Times are
UTC unless stated; the Cayman Islands sit at UTC-5 year-round.

## detections_nassau_LC_CB.csv.gz
Acoustic detections of tagged Nassau Grouper on the Little Cayman (LC) and Cayman Brac
(CB) receiver arrays, 2005-2025 (2,673,316 rows, 125 fish). Assembled from the raw
receiver downloads; transmitter codes are resolved to animals through their deployment
windows (tag numbers were re-used across animals), and detections outside a
transmitter's window on an animal are removed.

| column | meaning |
|---|---|
| animal_id | fish identifier (GMP-nnnn) |
| datetime_utc | detection timestamp, UTC |
| station | receiver station name (canonical) |
| receiver | receiver serial number |
| transmitter | full transmitter code |

## tagging_metadata.csv
One row per acoustically tagged Nassau Grouper at LC/CB (plus any fish appearing in the
detection file). Capture and surgical-implantation methods are described in the paper's
supplement. Columns: animal_id, capture_date, island, location_name, transmitter,
tag_valid_from/until (deployment window), length_cm (total length at tagging),
weight_kg, sex_best (best available sex assignment), tag_family, tag_delay_min_s /
tag_delay_max_s (nominal random transmission delay), tag_est_life_days.
One fish (GMP-0052, tagged at Little Cayman in 2005) was never detected on any
receiver and therefore appears in this table but not in the detection file (126
rows here; 125 fish with detections).

## station_locations.csv
One position per canonical receiver station (deployment-mean position: receivers were
re-moored at slightly different spots over the years). `island` is LC, CB, or LR (a
14-receiver line array from an unrelated 2017-18 study, shown on the site map for
context; no analysis uses it).

## receiver_station_month_status.csv
Station-by-month occupancy record for the LC and CB arrays, 2005-2025. `status` is
`active` (deployed, detections that month), `silent` (deployed, no detections),
`absent` (no receiver), or `unknown` (no source covers the month); `status_basis` says
how each status was established. Silence is common and does not indicate absence.

## population_size.csv
Island spawning-population estimates used as model covariates, from the mark-resight /
video census program of Waterhouse et al. 2020 (PNAS 117:1587-1595) continued through
2024. `basis` gives the provenance of every value, including the two interpolated
Little Cayman seasons and the constant Cayman Brac value.

## spawn_timing_observed.csv
Diver-observed spawning dates at Little Cayman, 2002-2023 (no gaps): season, first
observed spawning (obs_onset), and peak observed spawning (obs_peak), both in days
after the full moon.

## spawn_timing_acoustic.csv
Acoustic peak arrival day per island-season, derived from the detection file: for each
season, the daily count of tagged fish detected at the FSA is formed from fish tagged
before 1 January of that season (mid-season-tagged fish generate a detection ramp that
reflects tagging, not arrival); `peak_mid` is the day with the most fish detected,
taking the middle day of any tied run (taking the first tied day would pin small late
cohorts to their arrival plateau and manufacture exactly the trend under test).
`n_all`/`n_fish` are cohort sizes before/after the tagging filter; `centroid`,
`arr_dafm`, `dep_dafm` are alternative summaries (detection-weighted centroid, arrival,
departure). Days are relative to the season's spawning full moon.

## sst_daily_LC_CB.csv
Daily satellite sea-surface temperature (deg C) at Little Cayman and Cayman Brac,
2004-2025, used in the spawn-timing temperature check.

## coastlines_LC_CB.csv
Island coastline polygons (GSHHG full resolution; Wessel & Smith 1996), used for the
map figures and the movement animation. Columns: polygon_id, lon, lat.
