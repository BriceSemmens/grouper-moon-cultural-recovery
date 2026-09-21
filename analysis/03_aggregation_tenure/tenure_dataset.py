"""Aggregation tenure dataset: hours at the FSA per attended fish-season.

For every fish-season that attended under the shared inclusion rule, the response is the
number of distinct local clock-hours with at least one detection at the island's FSA
stations, 1 January - 30 April. Covariates: length at tagging, length projected forward
to the season on the von Bertalanffy curve (see gmp_shared.project_length), sex, and the
island-season spawning population size.

Run from this directory:  python3 tenure_dataset.py

Output (written to ../../output/):
    tenure_fish_seasons.csv   one row per attended fish-season
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_shared"))
import gmp_shared as G

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output")
os.makedirs(OUT, exist_ok=True)

det = G.load_detections()
meta = G.load_metadata()
pop = G.load_population()
recruit = G.recruited(det, meta)

rows = []
for island, year in G.SEASONS:
    _, att, per_fish = G.in_scope(det, meta, island, year, recruit=recruit)
    for a in sorted(att):
        rows.append(dict(animal_id=a, island=island, year=year,
                         hours=int(per_fish.hours[a]), n_det=int(per_fish.n_det[a])))

d = pd.DataFrame(rows)
d["length"] = d.animal_id.map(meta.length_cm)
d["sex"] = d.animal_id.map(meta.sex_best)
d["tag_date"] = pd.to_datetime(d.animal_id.map(meta.capture_date), errors="coerce")
dt_years = (pd.to_datetime(d.year.astype(str) + "-02-01") - d.tag_date).dt.days / 365.25
d["dt_yr"] = dt_years
d["length_proj"] = [G.project_length(l, i, t) for l, i, t in zip(d.length, d.island, dt_years)]
d["popsize"] = [pop[(i, y)] for i, y in zip(d.island, d.year)]
d = d.sort_values(["island", "year", "animal_id"]).reset_index(drop=True)
d.to_csv(os.path.join(OUT, "tenure_fish_seasons.csv"), index=False)
print(f"fish-seasons: {len(d)}   fish: {d.animal_id.nunique()}")
print(d.groupby(d.island + d.year.astype(str))
       .agg(n=("hours", "size"), median_h=("hours", "median")).to_string())
