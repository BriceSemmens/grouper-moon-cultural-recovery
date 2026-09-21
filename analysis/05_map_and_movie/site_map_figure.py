#!/usr/bin/env python3
"""Study-site map: Little Cayman and Cayman Brac with the receiver arrays
(manuscript fig. S1).

Layout
  main    both islands with all Grouper Moon receiver stations, FSA receivers highlighted
  inset A Caribbean locator (requires the basemap toolkit; skipped with a notice if absent)
  inset B the Little Cayman FSA, where REC 1 and REC 16 sit ~430 m apart and would
          otherwise overplot at the main panel's scale

Notes
  - The 2017-18 line array from an unrelated study is drawn in recessive grey with a
    distinct marker: context, not a third category.
  - Isotropic scale: aspect = km_per_deg_lat / km_per_deg_lon at the study latitude, so
    the scale bar is exact; the printed check at the end must read equal.
  - Coastlines are GSHHG full resolution (packaged in data/coastlines_LC_CB.csv).

Run from this directory:  python3 site_map_figure.py
Output (written to ../../output/):  FigS1_site_map.png
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output")
os.makedirs(OUT, exist_ok=True)


RED, BLUE, GREY, INK, MUTED = "#D1562F", "#3286C4", "#8a8a8a", "#2b2b2b", "#6b6b6b"
LAND, SEA, COASTLINE = "#DEDAD2", "#EAF0F5", "#9d968a"
FSA = {"LC REC 1", "LC REC 16", "CB EE REC"}
LAT0 = 19.71
KM_LAT = 110.574
KM_LON = 111.320 * np.cos(np.radians(LAT0))
ASPECT = KM_LAT / KM_LON

XMIN, XMAX, YMIN, YMAX = -80.170, -79.680, 19.628, 19.796
FX0, FX1, FY0, FY1 = -80.1320, -80.1035, 19.6440, 19.6690

st = pd.read_csv(os.path.join(DATA, "station_locations.csv"))
assert "LC Bld Bay MPA" not in set(st.station), "Bld Bay should be merged into LC REC 5"
st["grp"] = st.island
gm = st[st.grp.isin(["LC", "CB"])].copy()
lr = st[st.grp == "LR"].copy()
print(f"{len(gm)} Grouper Moon receivers (LC {sum(gm.grp=='LC')}, CB {sum(gm.grp=='CB')}), "
      f"{len(lr)} lionfish line-array receivers")

def short(n):
    return (n.replace("LC REC ", "").replace("CB ", "").replace("LC ", "")
             .replace(" REC", "").replace("S Hole S", "S Hole").strip())

cl = pd.read_csv(os.path.join(DATA, "coastlines_LC_CB.csv"))
coast = [(g.lon.to_numpy(), g.lat.to_numpy()) for _, g in cl.groupby("polygon_id")]

# ---- figure geometry, derived from the data aspect -------------------------
FIG_W = 13.6
L, R = 0.050, 0.988
ax_w_in = (R - L) * FIG_W
ax_h_in = ax_w_in * ((YMAX - YMIN) * KM_LAT) / ((XMAX - XMIN) * KM_LON)
FIG_H = ax_h_in / 0.545 + 0.05
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=200)
ax = fig.add_axes([L, 1 - (ax_h_in / FIG_H) - 0.035, R - L, ax_h_in / FIG_H])

def draw_coast(a):
    for xs, ys in coast:
        a.fill(xs, ys, facecolor=LAND, edgecolor=COASTLINE, linewidth=0.9, zorder=2)

ax.set_facecolor(SEA)
draw_coast(ax)
core = gm[gm.station.isin(FSA)]
rest = gm[~gm.station.isin(FSA)]
ax.scatter(lr.lon, lr.lat, s=13, marker="s", c=GREY, edgecolors="none", zorder=4)
ax.scatter(rest.lon, rest.lat, s=52, marker="o", c=BLUE, edgecolors="white",
           linewidths=1.3, zorder=5)
ax.scatter(core.lon, core.lat, s=150, marker="*", c=RED, edgecolors="white",
           linewidths=1.1, zorder=6)

# labels: skip anything resolved in the FSA inset; alternate up/down along the
# crowded south shore so the 3 km-spaced stations do not collide
in_inset = gm.lon.between(FX0, FX1) & gm.lat.between(FY0, FY1)
south = gm[(~in_inset) & (gm.lat < 19.675) & (gm.grp == "LC")].sort_values("lon")
flip = {s: (14 if i % 2 else -17) for i, s in enumerate(south.station)}
for _, r in gm.iterrows():
    if in_inset.loc[r.name] or r.station == "CB EE REC":
        continue
    dy = flip.get(r.station, 11 if r.lat >= 19.695 else -16)
    ax.annotate(short(r.station), (r.lon, r.lat), ha="center", xytext=(0, dy),
                textcoords="offset points", fontsize=7.8, color=INK, zorder=7)

cb = gm[gm.station == "CB EE REC"].iloc[0]
ax.annotate("CB EE REC (FSA)", (cb.lon, cb.lat), textcoords="offset points",
            xytext=(0, -20), fontsize=8.5, color=RED, ha="center", zorder=8)
ax.annotate("14-receiver line array\n(lionfish study, 2017–18)",
            (lr.lon.mean(), lr.lat.mean()), textcoords="offset points",
            xytext=(0, 26), fontsize=7.8, color=MUTED, ha="center", zorder=8,
            linespacing=1.4,
            arrowprops=dict(arrowstyle="-", color=GREY, linewidth=0.8, shrinkB=4))

ax.text(-80.037, 19.6395, "LITTLE CAYMAN", fontsize=11.5, color="#7a7266",
        ha="center", style="italic", zorder=8)
ax.text(-79.800, 19.7770, "CAYMAN BRAC", fontsize=11.5, color="#7a7266",
        ha="center", style="italic", zorder=8)

# scale bar — in the open water between the islands, clear of every label
bar = 10.0 / KM_LON
x0, y0 = -79.995, 19.6440
ax.plot([x0, x0 + bar], [y0, y0], color=INK, linewidth=2.8, zorder=9,
        solid_capstyle="butt")
for xt in (x0, x0 + bar):
    ax.plot([xt, xt], [y0 - 0.0016, y0 + 0.0016], color=INK, linewidth=1.6, zorder=9)
ax.text(x0 + bar / 2, y0 + 0.0034, "10 km", ha="center", fontsize=8.8, color=INK, zorder=9)

ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX)
ax.set_aspect(ASPECT, adjustable="box")
xt = np.arange(-80.15, -79.68, 0.10)
ax.set_xticks(xt); ax.set_xticklabels([f"{abs(v):.2f}°W" for v in xt])
ax.set_yticks([19.65, 19.70, 19.75])
ax.set_yticklabels(["19.65°N", "19.70°N", "19.75°N"])
ax.tick_params(colors=MUTED, length=3, labelsize=8.5)
for s in ax.spines.values():
    s.set_color("#c8c8c8")

h = [plt.Line2D([], [], marker="*", linestyle="none", markersize=14, markerfacecolor=RED,
                markeredgecolor="white", label="FSA core receiver"),
     plt.Line2D([], [], marker="o", linestyle="none", markersize=8, markerfacecolor=BLUE,
                markeredgecolor="white", label="Array receiver"),
     plt.Line2D([], [], marker="s", linestyle="none", markersize=5, markerfacecolor=GREY,
                markeredgecolor="none", label="Line array (lionfish study)")]
ax.legend(handles=h, loc="lower right", frameon=True, facecolor="white",
          edgecolor="#d5d5d5", fontsize=8.8, handletextpad=0.5, borderpad=0.6)

ax.add_patch(Rectangle((FX0, FY0), FX1 - FX0, FY1 - FY0, fill=False,
                       edgecolor=RED, linewidth=1.4, zorder=9))

# ---- inset A: Caribbean locator -------------------------------------------
bot = 0.055
axl = fig.add_axes([L + 0.06, bot, 0.32, 0.30])
axl.set_facecolor(SEA)
try:
    from mpl_toolkits.basemap import Basemap
    bm = Basemap(projection="cyl", llcrnrlon=-85.0, urcrnrlon=-75.5,
                 llcrnrlat=16.8, urcrnrlat=23.6, resolution="i", ax=axl)
    bm.fillcontinents(color=LAND, lake_color=SEA)
    bm.drawcoastlines(linewidth=0.45, color=COASTLINE)
except Exception as e:
    print("locator: basemap unavailable —", e)
axl.plot([-80.05], [19.70], marker="o", markersize=9, markerfacecolor="none",
         markeredgecolor=RED, markeredgewidth=1.8, zorder=6)
axl.text(-80.05, 19.05, "Little Cayman\n& Cayman Brac", fontsize=7.4, color=RED,
         ha="center", va="top", linespacing=1.4, zorder=6)
axl.text(-81.6, 22.2, "CUBA", fontsize=7.6, color="#7a7266", ha="center")
axl.text(-77.3, 17.6, "JAMAICA", fontsize=7.6, color="#7a7266", ha="center")
axl.set_xlim(-85.0, -75.5); axl.set_ylim(16.8, 23.6)
axl.set_aspect(110.574 / (111.320 * np.cos(np.radians(20.2))))
axl.set_xticks([]); axl.set_yticks([])
for s in axl.spines.values():
    s.set_color("#c8c8c8")
axl.set_title("Caribbean context", fontsize=8.8, color=MUTED, pad=5)

# ---- inset B: the Little Cayman FSA ---------------------------------------
axf = fig.add_axes([0.56, bot, 0.34, 0.30])
axf.set_facecolor(SEA)
draw_coast(axf)
sub = gm[gm.lon.between(FX0, FX1) & gm.lat.between(FY0, FY1)]
for _, r in sub.iterrows():
    isf = r.station in FSA
    axf.scatter([r.lon], [r.lat], s=210 if isf else 62,
                marker="*" if isf else "o", c=RED if isf else BLUE,
                edgecolors="white", linewidths=1.2, zorder=6)
    axf.annotate(short(r.station), (r.lon, r.lat), textcoords="offset points",
                 xytext=(10, -3), fontsize=8.4, color=RED if isf else INK, zorder=7)
d_km = np.hypot((-80.12083 + 80.12328) * KM_LON, (19.64998 - 19.65315) * KM_LAT)
axf.annotate(f"{d_km*1000:.0f} m", (-80.1247, 19.6516), fontsize=8.0,
             color=MUTED, ha="right", zorder=7)
axf.plot([-80.12328, -80.12083], [19.65315, 19.64998], color=MUTED,
         linewidth=0.9, linestyle=(0, (3, 2)), zorder=5)
bar2 = 0.5 / KM_LON
axf.plot([FX0 + 0.0020, FX0 + 0.0020 + bar2], [FY0 + 0.0014] * 2, color=INK,
         linewidth=2.4, zorder=8, solid_capstyle="butt")
axf.text(FX0 + 0.0020 + bar2 / 2, FY0 + 0.0026, "500 m", ha="center",
         fontsize=8.0, color=INK, zorder=8)
axf.set_xlim(FX0, FX1); axf.set_ylim(FY0, FY1)
axf.set_aspect(ASPECT, adjustable="box")
axf.set_xticks([]); axf.set_yticks([])
for s in axf.spines.values():
    s.set_color(RED); s.set_linewidth(1.5)
axf.set_title("Little Cayman FSA (west end)", fontsize=8.8, color=MUTED, pad=5)

fig.savefig(os.path.join(OUT, "FigS1_site_map.png"), facecolor="white")
print("saved", os.path.join(OUT, "FigS1_site_map.png"))

bb = ax.get_window_extent()
xk = bb.width / ((XMAX - XMIN) * KM_LON)
yk = bb.height / ((YMAX - YMIN) * KM_LAT)
print(f"scale check: 10 km across = {10*xk:.2f} px, 10 km up = {10*yk:.2f} px")
