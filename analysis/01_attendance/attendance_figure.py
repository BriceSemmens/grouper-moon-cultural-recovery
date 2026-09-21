"""Figure: daily attendance at the FSA through a spawning season (manuscript Fig. 1
and fig. S2).

For each requested island-season, plots the daily proportion of in-scope tagged fish
detected at the island's FSA stations (bars, shaded by mean distinct detection-hours per
fish that day), the cumulative proportion detected at least once (black line), and the
winter full moons (open circles).

Run from this directory:  python3 attendance_figure.py

Outputs (written to ../../output/):
    Fig1_attendance_LC_2006.png       single-panel, Little Cayman 2006
    FigS2_attendance_LC_CB_2007.png   two-panel, both islands in 2007
"""
import os
import sys

import ephem
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm
import matplotlib.colors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.cm import viridis

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_shared"))
import gmp_shared as G

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output")
os.makedirs(OUT, exist_ok=True)

NAMES = {"LC": "Little Cayman", "CB": "Cayman Brac"}
LABEL_COLOR = {"LC": "#2b6cb0", "CB": "#c05621"}   # island label color only

det = G.load_detections()
meta = G.load_metadata()
recruit = G.recruited(det, meta)


def full_moons(year):
    """Full-moon timestamps (local) falling in Jan-Apr of the given year."""
    out, d = [], ephem.Date(f"{year - 1}/12/15")
    for _ in range(8):
        d = ephem.next_full_moon(d)
        t = pd.Timestamp(d.datetime()) - pd.Timedelta(hours=5)
        if t.year == year and t.month <= 4:
            out.append(t)
    return out


def panel_data(island, year):
    """Daily detected proportion, cumulative first-detection curve, and moons."""
    den, att, _ = G.in_scope(det, meta, island, year, recruit=recruit)
    n = len(den)
    if n == 0:
        return None
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-04-30 23:59:59")
    d = det[det.animal_id.isin(den) & det.station.isin(G.FSA[island])
            & (det.local >= start) & (det.local <= end)]
    per_fish_day = d.groupby(["day", "animal_id"]).hour.nunique().reset_index(name="n_hours")
    daily = (per_fish_day.groupby("day")
             .agg(n_fish=("animal_id", "size"), avg_hrs=("n_hours", "mean"))
             .reset_index())
    daily["prop"] = daily.n_fish / n
    days = pd.date_range(start, end.floor("D"), freq="D")
    daily = (daily.set_index("day").reindex(days).fillna({"n_fish": 0, "prop": 0})
             .rename_axis("day").reset_index())
    cum = (per_fish_day.groupby("animal_id").day.min().value_counts().sort_index().cumsum()
           .reindex(days, method="ffill").fillna(0) / n)
    return dict(island=island, year=year, n=n, daily=daily, days=days, cum=cum,
                moons=full_moons(year))


def draw(panels, filename):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    k = len(panels)
    fig, axes = plt.subplots(k, 1, figsize=(9.6, 4.3 if k == 1 else 3.6 * k),
                             dpi=200, sharex=True)
    if k == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.09, right=0.86, top=0.955,
                        bottom=0.185 if k == 1 else 0.10, hspace=0.14)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=24)
    for ax, p in zip(axes, panels):
        yr = p["year"]
        start = pd.Timestamp(f"{yr}-01-01")
        xend = pd.Timestamp(f"{yr}-04-01")
        ax.set_facecolor("#EBEBEB")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(True, color="white", linewidth=1.1)
        ax.set_axisbelow(True)
        m = p["daily"].n_fish > 0
        ax.bar(p["daily"].day[m], p["daily"].prop[m], width=0.92,
               color=[viridis(norm(v)) for v in p["daily"].avg_hrs[m]], zorder=2)
        ax.plot(p["days"], p["cum"].values, color="black", linewidth=1.8, zorder=3)
        ax.set_xlim(start - pd.Timedelta(days=2), xend + pd.Timedelta(days=2))
        ax.set_ylim(0, 1.06)
        ax.set_yticks([0, .25, .5, .75, 1.0])
        ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
        ax.tick_params(colors="#4d4d4d", length=3)
        ax.text(0.985, 0.905, f"{NAMES[p['island']]}, {yr}   (n = {p['n']})",
                transform=ax.transAxes, fontsize=13.5, va="top", ha="right",
                color=LABEL_COLOR[p["island"]])
        for t in p["moons"]:
            if t <= xend + pd.Timedelta(days=2):
                ax.plot([t], [1.0], marker="o", markersize=12, clip_on=False, zorder=5,
                        markerfacecolor="white", markeredgecolor="black",
                        markeredgewidth=1.4)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[-1].set_xlabel("Date", fontsize=15)
    fig.supylabel("Proportion of Tagged Fish", fontsize=15, x=0.015)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=viridis), ax=axes,
                      fraction=0.05, pad=0.03, shrink=0.5)
    cb.set_label("Avg. # hrs", fontsize=14, labelpad=8)
    cb.outline.set_visible(False)
    path = os.path.join(OUT, filename)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("saved", path)


draw([panel_data("LC", 2006)], "Fig1_attendance_LC_2006.png")
draw([p for p in (panel_data(i, 2007) for i in ("LC", "CB")) if p],
     "FigS2_attendance_LC_CB_2007.png")
