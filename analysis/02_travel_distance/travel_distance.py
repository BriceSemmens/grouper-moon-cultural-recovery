"""Spawning-season travel distance per fish-season.

For every fish-season that ATTENDED the aggregation under the shared inclusion rule,
computes the cumulative distance traveled among receiver stations during the season:
detections are sorted in time and the great-circle distance between consecutive
detections is summed whenever the station changes. This is a minimum estimate of the
distance actually traveled. The two adjacent Little Cayman FSA receivers (LC REC 1 and
LC REC 16) are treated as a single location throughout, so movement between them never
contributes distance.

Because cumulative path length accrues with every station-to-station transition, it
scales with detection count; the model script controls for this with log(detections),
and density-invariant summaries (hourly- and daily-binned path length, maximum
displacement between any two visited stations) are carried on every row as checks.

Run from this directory:  python3 travel_distance.py

Output (written to ../../output/):
    travel_fish_seasons.csv   one row per attended fish-season with the distance metric,
                              diagnostics, covariates, and population size
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_shared"))
import gmp_shared as G

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output")
os.makedirs(OUT, exist_ok=True)

EARTH_R = 6378.145                     # km; matches the metric's original implementation
FSA_MERGE = {"LC REC 16": "LC REC 1"}  # one FSA location
NO_COORD = {"LC Bld Bay MPA"}          # no surveyed position; detections cannot enter a path


def great_circle_km(lon1, lat1, lon2, lat2):
    rad = np.pi / 180
    a1, b1, a2, b2 = lat1 * rad, lon1 * rad, lat2 * rad, lon2 * rad
    h = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((b2 - b1) / 2) ** 2
    return EARTH_R * 2 * np.arctan2(np.sqrt(h), np.sqrt(1 - h))


def path_length(lon, lat, key):
    """Sum of step distances where the station key changes between consecutive rows."""
    if len(key) < 2:
        return 0.0
    step = great_circle_km(lon[:-1], lat[:-1], lon[1:], lat[1:])
    return float(np.sum(np.where(key[:-1] != key[1:], step, 0.0)))


st = pd.read_csv(os.path.join(G.DATA_DIR, "station_locations.csv")).set_index("station")
LAT, LON = st.lat.to_dict(), st.lon.to_dict()
LAT["LC REC 16"], LON["LC REC 16"] = LAT["LC REC 1"], LON["LC REC 1"]

det = G.load_detections()
meta = G.load_metadata()
pop = G.load_population()
recruit = G.recruited(det, meta)

# attended fish-seasons under the shared rule, for the analysis seasons
attended = {}
for island, year in G.SEASONS:
    _, att, _ = G.in_scope(det, meta, island, year, recruit=recruit)
    attended[(island, year)] = att
print(f"attended fish-seasons across analysis seasons: "
      f"{sum(len(v) for v in attended.values())}")

season_det = det[det.local.dt.month.isin([1, 2, 3, 4])].copy()
season_det["year"] = season_det.local.dt.year
season_det["island"] = season_det.station.str[:2]
season_det = season_det[~season_det.station.isin(NO_COORD)]
season_det["loc"] = season_det.station.replace(FSA_MERGE)
season_det["lat"] = season_det["loc"].map(LAT)
season_det["lon"] = season_det["loc"].map(LON)

rows = []
for (island, year) in G.SEASONS:
    g = season_det[(season_det.island == island) & (season_det.year == year)
                   & season_det.animal_id.isin(attended[(island, year)])]
    for animal in sorted(attended[(island, year)]):
        d = g[g.animal_id == animal].sort_values("ts", kind="stable")
        key = d["loc"].to_numpy()
        raw = path_length(d.lon.to_numpy(), d.lat.to_numpy(), key)
        # density-invariant diagnostics: path over hourly/daily modal stations
        binned = {}
        for label, freq in [("hourly", "h"), ("daily", "D")]:
            mode = (d.assign(b=d.local.dt.floor(freq)).groupby("b")["loc"]
                    .agg(lambda s: s.mode().iloc[0]).sort_index().to_numpy())
            binned[label] = path_length(np.array([LON[x] for x in mode]),
                                        np.array([LAT[x] for x in mode]), mode)
        locs = list(dict.fromkeys(key))
        max_disp = max((great_circle_km(LON[p], LAT[p], LON[q], LAT[q])
                        for p, q in itertools.combinations(locs, 2)), default=0.0)
        rows.append(dict(animal_id=animal, island=island, year=year, distance_km=raw,
                         d_hourly_km=binned["hourly"], d_daily_km=binned["daily"],
                         max_displacement_km=max_disp, n_det=len(d),
                         n_locations=len(locs)))

a = pd.DataFrame(rows)
a["length"] = a.animal_id.map(meta.length_cm)
a["sex"] = a.animal_id.map(meta.sex_best)
a["tag_date"] = pd.to_datetime(a.animal_id.map(meta.capture_date), errors="coerce")
dt_years = (pd.to_datetime(a.year.astype(str) + "-02-01") - a.tag_date).dt.days / 365.25
a["length_proj"] = [G.project_length(l, i, t) for l, i, t in zip(a.length, a.island, dt_years)]
a["popsize"] = [pop[(i, y)] for i, y in zip(a.island, a.year)]
a = a.sort_values(["island", "year", "animal_id"]).reset_index(drop=True)
a.to_csv(os.path.join(OUT, "travel_fish_seasons.csv"), index=False)
print(f"fish-seasons: {len(a)}   fish: {a.animal_id.nunique()}")
print(a.groupby(a.island + a.year.astype(str))
       .agg(n=("distance_km", "size"), median_km=("distance_km", "median"),
            max_km=("distance_km", "max")).round(1).to_string())
