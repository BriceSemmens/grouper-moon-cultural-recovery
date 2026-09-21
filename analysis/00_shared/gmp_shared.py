"""Shared data loading and the fish-season inclusion rule.

Every analysis in this repository imports this module rather than re-implementing
data handling or eligibility criteria. That guarantees the attendance, travel-distance,
and aggregation-tenure analyses operate on identically screened data.

Data files (see ../../data/README.md):
    detections_nassau_LC_CB.csv.gz   acoustic detections of tagged Nassau Grouper on the
                                     Little Cayman (LC) and Cayman Brac (CB) arrays
    tagging_metadata.csv             one row per tagged fish
    population_size.csv              island spawning-population estimates by season
    station_locations.csv            receiver station positions

THE INCLUSION RULE
------------------
A fish-season (one fish in one island's spawning season) is IN SCOPE if all six hold:

  1. TAGGED BEFORE THE SEASON   capture date precedes 1 January of the season.
  2. ALIVE ON 1 JANUARY         the tag's last day with >= 2 detections anywhere falls on
                                or after 1 January of the season.
  3. REPRODUCTIVELY MATURE      fish tagged at total length > 55 cm -- the threshold for
                                reliable maturity in Nassau Grouper (NMFS 2013, Status
                                Review Report: Nassau Grouper (Epinephelus striatus),
                                NOAA National Marine Fisheries Service) -- are in
                                scope from tagging. Fish tagged at <= 55 cm enter scope in
                                the first season they demonstrably attend the aggregation
                                (same >= 3-detection test as the attendance numerator) and
                                remain in scope thereafter, including seasons they skip.
                                Rationale: a fixed size cut discards observed spawning
                                migrations by small-tagged fish on the authority of a
                                static length threshold, and ignores growth over a
                                two-decade record; recruiting on first demonstrated
                                attendance uses the animal's own record instead.
  4. NOT A YEAR-ROUND RESIDENT  fewer than 20 distinct off-season (June-November) days
                                with a detection at an FSA station. Year-round residents
                                would otherwise trivially register as attending.
  5. NO MID-SEASON TAG DEATH    excluded if the tag's final-ever detection precedes
                                1 March AND the fish logged < 24 distinct FSA
                                detection-hours that season.
  6. TAG VALIDITY               detections outside a transmitter's deployment window on
                                this animal are removed upstream (tag numbers were
                                re-used across animals); enforced in the packaged
                                detection file.

Shed / failed transmitters are truncated at the onset of failure (see SHED below): a
transmitter that loses all between-station movement and settles into a fixed, metronomic
single-station pattern is classified as shed or failed from that onset, and detections
from then on are removed before any screening.

ATTENDANCE: a fish-season counts as attended if the tag logged >= 3 detections at its
island's FSA station(s) between 1 January and 30 April (local time). Single detections
can arise from code collisions, hence the 3-detection floor.

FSA stations: LC = LC REC 1 + LC REC 16 (two adjacent receivers at one site);
              CB = CB EE REC.
"""
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- constants
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")

FSA = {"LC": ["LC REC 1", "LC REC 16"], "CB": ["CB EE REC"]}
ALL_FSA = [s for v in FSA.values() for s in v]

# Shed / failed transmitters, truncated at the diagnosed onset of failure.
# GMP-0066: shed 2008-02-05 (corroborated by collapse of daily depth variation).
# GMP-0118: failed 2018-05-26 (detections settle into a fixed single-station pattern).
SHED = {"GMP-0066": pd.Timestamp("2008-02-05"),
        "GMP-0118": pd.Timestamp("2018-05-26")}

MATURE_CM = 55.0            # length-at-maturity threshold (criterion 3; NMFS 2013 Status Review)
RESIDENT_MIN_DAYS = 20      # off-season FSA days that mark a year-round resident
OFFSEASON_MONTHS = [6, 7, 8, 9, 10, 11]
MIN_FSA_DET = 3             # attendance floor (code collisions produce singletons)
DEATH_CUTOFF_MD = "03-01"   # criterion 5
DEATH_MAX_HOURS = 24
SEASON_END_MD = "04-30"

# Analysis seasons with defensible array coverage (see materials and methods).
SEASONS = ([("LC", y) for y in [2006, 2007, 2008] + list(range(2016, 2025))] +
           [("CB", y) for y in range(2007, 2012)])

# von Bertalanffy growth (Stock et al. 2021, ICES JMS 78:277-292, Table 2, model m2):
# shared L-infinity, island-specific K. Used to project length-at-tagging forward.
VBGF_LINF = 80.2
VBGF_K = {"LC": 0.140, "CB": 0.160}


# ---------------------------------------------------------------- loading
def load_detections(data_dir=DATA_DIR):
    """Read the packaged detection file and attach derived time columns.

    Detections are logged in UTC; the Cayman Islands sit at UTC-5 year-round, and the
    season window (1 Jan - 30 Apr) is defined on the local clock, so a local timestamp
    is attached here and used for all windowing.
    """
    det = pd.read_csv(os.path.join(data_dir, "detections_nassau_LC_CB.csv.gz"),
                      parse_dates=["datetime_utc"])
    det = det.rename(columns={"datetime_utc": "ts"})
    for animal, onset in SHED.items():
        det = det[~((det.animal_id == animal) & (det.ts >= onset))]
    det["local"] = det.ts - pd.Timedelta(hours=5)
    det["day"] = det.local.dt.normalize()
    det["hour"] = det.local.dt.floor("h")
    det["station"] = det.station.fillna("").astype(str)
    return det


def load_metadata(data_dir=DATA_DIR):
    """One row per tagged fish, indexed by animal_id."""
    return pd.read_csv(os.path.join(data_dir, "tagging_metadata.csv")).set_index("animal_id")


def load_population(data_dir=DATA_DIR):
    """{(island_code, season): spawning population estimate}.

    Little Cayman: yearly series (mark-resight, with two interpolated seasons).
    Cayman Brac: held at the 2008 mark-resight estimate across its monitored seasons.
    See population_size.csv for provenance of every value.
    """
    pop = pd.read_csv(os.path.join(data_dir, "population_size.csv"))
    code = {"Little Cayman": "LC", "Cayman Brac": "CB"}
    return {(code[r.site], int(r.season)): float(r.estimate) for r in pop.itertuples()}


def project_length(length_cm, island, dt_years):
    """Project length-at-tagging forward dt_years on the von Bertalanffy curve.

    Fish tagged at or above L-infinity are held at their tagged length (the mean curve
    would shrink them, an artifact of L0 > Linf rather than biology). Missing lengths
    and negative dt pass through unchanged.
    """
    if pd.isna(length_cm) or pd.isna(dt_years) or dt_years < 0 or length_cm >= VBGF_LINF:
        return length_cm
    return VBGF_LINF - (VBGF_LINF - length_cm) * np.exp(-VBGF_K[island] * dt_years)


def wilson_interval(k, n, z=1.959963984540054):
    """Wilson score 95% interval for a binomial proportion."""
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - half) / denom, (center + half) / denom


# ---------------------------------------------------------------- the rule
def residents(det):
    """Criterion 4: animals with >= RESIDENT_MIN_DAYS off-season FSA detection days."""
    off = det[det.station.isin(ALL_FSA) & det.local.dt.month.isin(OFFSEASON_MONTHS)]
    n = off.groupby("animal_id").day.nunique()
    return set(n[n >= RESIDENT_MIN_DAYS].index)


def recruited(det, meta):
    """Criterion 3, second half: the first season each sub-threshold fish attends.

    For every fish tagged at <= MATURE_CM, find the first season in which it logged
    >= MIN_FSA_DET detections at its island's FSA stations inside the season window.
    Returns {animal_id: first attended season}; a fish absent from the mapping never
    demonstrated attendance and is never in scope. The attendance test used here is
    identical to the numerator's, so a fish is admitted by evidence it attends, not by
    an assumption about its size.
    """
    out = {}
    for a in meta.index:
        length = meta.length_cm.get(a)
        if pd.isna(length) or length > MATURE_CM:
            continue
        island = meta.island.get(a)
        if island not in FSA:
            continue
        d = det[(det.animal_id == a) & det.station.isin(FSA[island])
                & det.local.dt.month.isin([1, 2, 3, 4])]
        if d.empty:
            continue
        n = d.groupby(d.local.dt.year).size()
        ok = n[n >= MIN_FSA_DET]
        if len(ok):
            out[a] = int(ok.index.min())
    return out


def in_scope(det, meta, island, season, recruit=None):
    """Apply the rule to one island-season.

    Returns (denominator, attended, per_fish) where:
        denominator  list of in-scope animal_ids (criteria 1-6),
        attended     subset with >= MIN_FSA_DET detections at the island's FSA,
        per_fish     DataFrame indexed by animal_id with distinct FSA detection-hours
                     ('hours') and FSA detection count ('n_det') inside the window.

    Pass `recruit` (the map from recruited()) when looping over many seasons so it is
    computed once.
    """
    start = pd.Timestamp(f"{season}-01-01")
    end = pd.Timestamp(f"{season}-{SEASON_END_MD} 23:59:59")
    cap = pd.to_datetime(meta.capture_date, errors="coerce")
    if recruit is None:
        recruit = recruited(det, meta)

    # last day with >= 2 detections anywhere (criterion 2), last detection ever (crit. 5)
    per_day = det.groupby(["animal_id", "day"]).size().reset_index(name="n")
    alive_until = per_day[per_day.n >= 2].groupby("animal_id").day.max()
    last_ever = det.groupby("animal_id").local.max()
    res = residents(det)

    pool = []
    for a in meta.index:
        if meta.island.get(a) != island:
            continue
        if pd.isna(cap.get(a)) or cap[a] >= start:                       # 1
            continue
        if a not in alive_until.index or alive_until[a] < start:         # 2
            continue
        if (pd.notna(meta.length_cm.get(a))                              # 3
                and meta.length_cm[a] <= MATURE_CM
                and season < recruit.get(a, np.inf)):
            continue
        if a in res:                                                     # 4
            continue
        pool.append(a)

    # The window is filtered on LOCAL time, matching the season definition; filtering on
    # raw UTC would shift the window by five hours at each end.
    d = det[det.animal_id.isin(pool) & det.station.isin(FSA[island])
            & (det.local >= start) & (det.local <= end)]
    per_fish = d.groupby("animal_id").agg(hours=("hour", "nunique"), n_det=("ts", "size"))
    attended = set(per_fish[per_fish.n_det >= MIN_FSA_DET].index)
    cutoff = pd.Timestamp(f"{season}-{DEATH_CUTOFF_MD}")
    denominator = [a for a in pool                                       # 5
                   if not (last_ever.get(a, pd.NaT) < cutoff
                           and per_fish.hours.get(a, 0) < DEATH_MAX_HOURS)]
    return denominator, attended & set(denominator), per_fish
