"""Aggregation attendance of tagged Nassau Grouper.

For every monitored island-season, applies the shared inclusion rule and scores each
in-scope fish-season as attended (>= 3 detections at the island's FSA stations,
1 January - 30 April local) or not. Reports per-season and pooled attendance with
Wilson score 95% confidence intervals.

Run from this directory:  python3 attendance_analysis.py

Outputs (written to ../../output/):
    attendance_by_season.csv   per island-season counts and Wilson CIs
    attendance_summary.txt     the pooled numbers quoted in the manuscript

Expected headline result: 156/158 fish-seasons attended = 98.7% (95% CI 95.5-99.7);
Little Cayman 134/135 = 99.3%, Cayman Brac 22/23 = 95.7%. The two non-attending
fish-seasons are the only defensible instances in the record of a live, tagged,
mature fish skipping a spawning season.
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
recruit = G.recruited(det, meta)

rows, fish_rows = [], []
for island in G.FSA:
    for season in range(2005, 2026):
        den, att, _ = G.in_scope(det, meta, island, season, recruit=recruit)
        if not den:
            continue
        lo, hi = G.wilson_interval(len(att), len(den))
        rows.append(dict(island=island, season=season, n=len(den), attended=len(att),
                         pct=round(100 * len(att) / len(den), 1),
                         wilson_lo=round(100 * lo, 1), wilson_hi=round(100 * hi, 1)))
        fish_rows += [dict(island=island, season=season, animal_id=a, attended=a in att)
                      for a in den]

per_season = pd.DataFrame(rows)
per_season.to_csv(os.path.join(OUT, "attendance_by_season.csv"), index=False)
fish = pd.DataFrame(fish_rows)

lines = []


def head(label, sub):
    k, n = int(sub.attended.sum()), len(sub)
    lo, hi = G.wilson_interval(k, n)
    lines.append(f"{label:<40s} {k}/{n} = {100 * k / n:.1f}%  "
                 f"(95% CI {100 * lo:.1f}-{100 * hi:.1f}%)")


lines.append("Attendance at the spawning aggregation, tagged Nassau Grouper")
lines.append("(shed/failed transmitters truncated at failure onset; see gmp_shared.py)\n")
head("Both islands, all monitored seasons", fish)
head("  Little Cayman only", fish[fish.island == "LC"])
head("  Cayman Brac only", fish[fish.island == "CB"])
lines.append("")
lines.append("Non-attending fish-seasons: " + ", ".join(
    f"{r.animal_id} {r.island}{r.season}" for r in fish[~fish.attended].itertuples()))

report = "\n".join(lines)
open(os.path.join(OUT, "attendance_summary.txt"), "w").write(report + "\n")
print(report)
print("\nPer-season table written to output/attendance_by_season.csv")
