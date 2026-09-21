#!/usr/bin/env python3
"""Animated spawning-season movements of tagged Nassau Grouper, Little Cayman 2006
(manuscript Movie S1).

Top panel: each fish drawn just seaward of the receiver that most recently detected it
(receiver-to-receiver transitions are routed through water, never across land, using a
shortest-path grid restricted to sea cells). A solid box marks the active west-end
spawning aggregation site, a dashed box the historic east-end site. Middle panel: the
cohort collapsed onto the island's long axis through time (brightness = number of fish
per 0.6-km stretch of shoreline). Bottom panel: lunar phase and the number of tagged
fish detected on the array.

Fish are offset 0.2-0.6 km seaward of their receiver so overlapping animals stay
separable; a VR2W-class receiver hears out to roughly 0.5 km, so a fish drawn a few
hundred meters from its receiver remains inside the volume the detection constrains it
to. Offsets are confined to seaward directions using a signed distance-to-coast field,
and a final pass walks any position within the minimum clearance outward until it
clears; the script prints the minimum clearance actually achieved.

Requires ffmpeg (for the mp4 writer) and the ephem package (full moons).

Run from this directory:  python3 movement_animation.py
Output (written to ../../output/):  MovieS1_spawning_movement_LC_2006.mp4  (~2-3 min render)
"""
import os, hashlib
import numpy as np, pandas as pd, ephem
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import Normalize, LinearSegmentedColormap, PowerNorm
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import distance_transform_edt

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output")
COORDS = os.path.join(DATA, "station_locations.csv")

META = os.path.join(DATA, "tagging_metadata.csv")
OUT = os.path.join(OUTDIR, "MovieS1_spawning_movement_LC_2006.mp4")

START, END = pd.Timestamp("2006-01-06"), pd.Timestamp("2006-02-26")
STEP_MIN, FPS = 30, 30
TRAIL_H, MAX_TRANSIT_H, MAX_GAP_H = 8.0, 6.0, 6.0
NX, NY, DILATE = 700, 380, 2

# seaward scatter
ARC_DEG = 78.0          # half-width of the arc, centred on the offshore normal
R0_KM, R1_KM = 0.18, 0.58
MIN_CLEAR_KM = 0.24     # every drawn fish must be at least this far off the coast

SEA, DEEP, SHELF = "#0E2836", "#0A1E29", "#164055"
LAND, EDGE = "#E4DAC4", "#B9A987"
FSA_C, INK, MUTE, ACC = "#FF7A4D", "#F0F6F9", "#7E9AAA", "#4FC3D9"
RAMP = LinearSegmentedColormap.from_list("l", ["#B7F0F7", "#5FCBDD", "#2E9BC4", "#FFC24D"])

KM_LAT, KM_LON = 110.574, 111.320 * np.cos(np.radians(19.69))

# ------------------------------------------------------------------ data
co = pd.read_csv(COORDS).set_index("station")
meta = pd.read_csv(META).set_index("animal_id")
det = pd.read_csv(os.path.join(DATA, "detections_nassau_LC_CB.csv.gz"),
                  parse_dates=["datetime_utc"]).rename(columns={"datetime_utc": "ts"})
det = det[det.station.astype(str).str.startswith("LC") & det.station.isin(co.index)].copy()
det["local"] = det.ts - pd.Timedelta(hours=5)
det = det[(det.local >= START) & (det.local <= END)]
stations = sorted(det.station.unique())
fish = sorted(det.animal_id.unique())
print(f"{len(det):,} detections · {len(fish)} fish · {len(stations)} stations")

slon = np.array([co.lon[s] for s in stations]); slat = np.array([co.lat[s] for s in stations])
mx, my = 0.10 * (slon.max() - slon.min()), 0.26 * (slat.max() - slat.min())
X0, X1 = slon.min() - mx, slon.max() + mx
Y0, Y1 = slat.min() - my, slat.max() + my
print(f"extent {(X1-X0)*KM_LON:.1f} x {(Y1-Y0)*KM_LAT:.1f} km "
      f"(aspect {(X1-X0)*KM_LON/((Y1-Y0)*KM_LAT):.2f})")

_cl = pd.read_csv(os.path.join(DATA, "coastlines_LC_CB.csv"))
coast = [(g.lon.to_numpy(), g.lat.to_numpy()) for _, g in _cl.groupby("polygon_id")]
lcx, lcy = min(coast, key=lambda p: np.mean(p[0]))
lcx, lcy = np.asarray(lcx), np.asarray(lcy)
poly = MplPath(np.column_stack([lcx, lcy]))

# ------------------------------------------------- grid, routing mask, distance field
gx, gy = np.linspace(X0, X1, NX), np.linspace(Y0, Y1, NY)
GX, GY = np.meshgrid(gx, gy)
land0 = poly.contains_points(np.column_stack([GX.ravel(), GY.ravel()])).reshape(NY, NX)
dx_km, dy_km = (gx[1] - gx[0]) * KM_LON, (gy[1] - gy[0]) * KM_LAT

# SIGNED distance to the TRUE coastline: positive offshore, negative inland.
# Signed rather than one-sided so the gradient is defined everywhere, including
# on land — a point that somehow ends up inland can still be pushed out.
SD = (distance_transform_edt(~land0, sampling=(dy_km, dx_km))
      - distance_transform_edt(land0, sampling=(dy_km, dx_km)))
GYD, GXD = np.gradient(SD, dy_km, dx_km)          # km per km, points seaward
_g = np.hypot(GXD, GYD); _g[_g < 1e-9] = 1e-9
GXD, GYD = GXD / _g, GYD / _g

def samp(F, lon, lat):
    j = np.clip(np.round((np.asarray(lon) - X0) / (X1 - X0) * (NX - 1)).astype(int), 0, NX - 1)
    i = np.clip(np.round((np.asarray(lat) - Y0) / (Y1 - Y0) * (NY - 1)).astype(int), 0, NY - 1)
    return F[i, j]

off_m = samp(SD, slon, slat) * 1000
print("receiver distance offshore (m): " +
      ", ".join(f"{s.replace('LC REC ','R')} {d:.0f}"
                for s, d in sorted(zip(stations, off_m), key=lambda p: p[1])[:6]))

# the routing mask is dilated so Dijkstra paths stand off the shore; it is
# NEVER drawn — the true coastline above is what goes on the map
land = land0.copy()
for _ in range(DILATE):
    land = (land | np.roll(land, 1, 0) | np.roll(land, -1, 0)
                 | np.roll(land, 1, 1) | np.roll(land, -1, 1))
water = ~land
idx = -np.ones((NY, NX), int); idx[water] = np.arange(water.sum())
rows, cols, vals = [], [], []
for di, dj in [(0, 1), (1, 0), (1, 1), (1, -1)]:
    a = idx[max(0, -di):NY - max(0, di), max(0, -dj):NX - max(0, dj)]
    b = idx[max(0, di):NY - max(0, -di), max(0, dj):NX - max(0, -dj)]
    m = (a >= 0) & (b >= 0); w = np.hypot(di * dy_km, dj * dx_km)
    rows += [a[m], b[m]]; cols += [b[m], a[m]]; vals += [np.full(m.sum(), w)] * 2
G = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
               shape=(water.sum(),) * 2).tocsr()
wy, wx = np.where(water)

def cell(lon, lat):
    j = int(np.clip(np.searchsorted(gx, lon), 0, NX - 1))
    i = int(np.clip(np.searchsorted(gy, lat), 0, NY - 1))
    if idx[i, j] >= 0: return i, j
    d = (wy - i) ** 2 + ((wx - j) * dx_km / dy_km) ** 2
    k = d.argmin(); return int(wy[k]), int(wx[k])

cells = {s: cell(co.lon[s], co.lat[s]) for s in stations}

def chaikin(p, n=3):
    for _ in range(n):
        q = [p[0]]
        for a, b in zip(p[:-1], p[1:]):
            q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        q.append(p[-1]); p = np.array(q)
    return p

routes = {}
for s in stations:
    i, j = cells[s]
    _, pred = dijkstra(G, indices=idx[i, j], return_predecessors=True)
    for t in stations:
        if t == s or (s, t) in routes: continue
        n = idx[cells[t][0], cells[t][1]]
        seq = []
        while n >= 0: seq.append(n); n = pred[n]
        if len(seq) < 2: continue
        pts = np.column_stack([gx[wx[seq[::-1]]], gy[wy[seq[::-1]]]])
        pts = pts[:: max(1, len(pts) // 70)]
        pts = chaikin(np.vstack([[co.lon[s], co.lat[s]], pts, [co.lon[t], co.lat[t]]]))
        d = np.r_[0, np.cumsum(np.hypot(np.diff(pts[:, 0]) * KM_LON, np.diff(pts[:, 1]) * KM_LAT))]
        if d[-1] <= 0: continue
        u = np.linspace(0, d[-1], 100)
        routes[(s, t)] = np.column_stack([np.interp(u, d, pts[:, 0]), np.interp(u, d, pts[:, 1])])
        routes[(t, s)] = routes[(s, t)][::-1]
print(f"{len(routes)//2} routes · route points on land: "
      f"{sum(poly.contains_points(r).sum() for r in routes.values())}")

# ------------------------------------------------- along-island axis
P = np.column_stack([(slon - slon.min()) * KM_LON, (slat - slat.min()) * KM_LAT])
axis = np.linalg.svd(P - P.mean(0), full_matrices=False)[2][0]
if axis[0] < 0: axis = -axis
origin = P[np.argmin(slon)]
along = {s: float(np.dot(P[k] - origin, axis)) for k, s in enumerate(stations)}
ALONG_MAX = max(along.values())
FSA_ST = "LC REC 1" if "LC REC 1" in along else stations[int(np.argmin(slon))]
print(f"along-island axis: west end 0.0 km -> east end {ALONG_MAX:.1f} km; "
      f"{FSA_ST} at {along[FSA_ST]:.2f} km")
for s in sorted(stations, key=lambda z: along[z]):
    print(f"   {s:<12s} {along[s]:6.2f} km")

# ------------------------------------------------- SEAWARD per-fish offsets
# Offshore normal at each station, from the gradient of the signed distance
# field. Then each fish gets a deterministic angle inside +/- ARC_DEG of that
# normal and a deterministic radius — stable across frames (so a stationary
# fish does not twitch) and always with a seaward component.
noff = {s: np.array([float(samp(GXD, co.lon[s], co.lat[s])),
                     float(samp(GYD, co.lon[s], co.lat[s]))]) for s in stations}
for s in noff:
    noff[s] /= max(np.hypot(*noff[s]), 1e-9)

ARC = np.radians(ARC_DEG)
foff = {}
for a in fish:
    h = int(hashlib.md5(a.encode()).hexdigest()[:8], 16)
    th = ((h % 2000) / 2000 * 2 - 1) * ARC
    r = R0_KM + ((h >> 11) % 1000) / 1000 * (R1_KM - R0_KM)
    c, sn = np.cos(th), np.sin(th)
    for s in stations:
        nx_, ny_ = noff[s]
        v = np.array([c * nx_ - sn * ny_, sn * nx_ + c * ny_]) * r
        foff[(a, s)] = np.array([v[0] / KM_LON, v[1] / KM_LAT])

# ------------------------------------------------------------------ tracks
frames = pd.date_range(START, END, freq=f"{STEP_MIN}min")
NF = len(frames)
pos = np.full((len(fish), NF, 2), np.nan)
alo = np.full((len(fish), NF), np.nan)
here = np.zeros((len(fish), NF), bool)
fnum = lambda t: int(np.clip((t - START) / pd.Timedelta(minutes=STEP_MIN), 0, NF - 1))

det = det.sort_values("local")
for fi, a in enumerate(fish):
    g = det[det.animal_id == a]
    t, st = g.local.values, g.station.values
    gaps = np.r_[0.0, np.diff(t).astype("timedelta64[s]").astype(float) / 3600.0]
    for _, blk in pd.DataFrame({"t": t, "s": st, "b": np.cumsum(gaps > MAX_GAP_H)}).groupby("b"):
        tb, sb = blk.t.values, blk.s.values
        chg = np.r_[True, sb[1:] != sb[:-1]]
        starts = np.where(chg)[0]; ends = np.r_[starts[1:] - 1, len(sb) - 1]
        for r in range(len(starts)):
            s0 = sb[starts[r]]
            f0, f1 = fnum(pd.Timestamp(tb[starts[r]])), fnum(pd.Timestamp(tb[ends[r]]))
            pos[fi, f0:f1 + 1] = np.array([co.lon[s0], co.lat[s0]]) + foff[(a, s0)]
            alo[fi, f0:f1 + 1] = along[s0]
            here[fi, f0:f1 + 1] = True
            if r + 1 < len(starts):
                s1 = sb[starts[r + 1]]
                f2 = fnum(pd.Timestamp(tb[starts[r + 1]]))
                rt = routes.get((s0, s1))
                if rt is None or f2 <= f1: continue
                span = min(f2 - f1, int(MAX_TRANSIT_H * 60 / STEP_MIN))
                u = np.linspace(0, 1, max(span, 2))
                o0, o1 = foff[(a, s0)], foff[(a, s1)]
                seg = rt[(u * (len(rt) - 1)).astype(int)] + (o0 + (o1 - o0) * u[:, None])
                n = min(len(seg), f2 - f1)
                pos[fi, f1:f1 + n] = seg[:n]
                alo[fi, f1:f1 + n] = np.linspace(along[s0], along[s1], n)
                here[fi, f1:f1 + n] = True
                if f1 + n < f2:
                    pos[fi, f1 + n:f2] = rt[-1] + o1
                    alo[fi, f1 + n:f2] = along[s1]
                    here[fi, f1 + n:f2] = True

# --- verified clearance: push anything still too close to shore straight out
flat = pos.reshape(-1, 2)
ok = np.isfinite(flat[:, 0])
lon_, lat_ = flat[ok, 0].copy(), flat[ok, 1].copy()
d0 = samp(SD, lon_, lat_)
print(f"before clamp: {int((d0 < 0).sum())} positions inland, "
      f"{int((d0 < MIN_CLEAR_KM).sum())} within {MIN_CLEAR_KM*1000:.0f} m of shore "
      f"(min {d0.min()*1000:.0f} m)")
for _ in range(40):
    d = samp(SD, lon_, lat_)
    bad = d < MIN_CLEAR_KM
    if not bad.any(): break
    lon_[bad] += samp(GXD, lon_[bad], lat_[bad]) / KM_LON * 0.04
    lat_[bad] += samp(GYD, lon_[bad], lat_[bad]) / KM_LAT * 0.04
flat[ok, 0], flat[ok, 1] = lon_, lat_
pos = flat.reshape(pos.shape)
dfin = samp(SD, lon_, lat_)
print(f"after clamp:  min clearance {dfin.min()*1000:.0f} m · "
      f"positions inside the coastline: {int(poly.contains_points(np.column_stack([lon_, lat_])).sum())}")

present = here.sum(0)
lens = np.array([meta.length_cm.get(a, np.nan) for a in fish], float)
lens = np.where(np.isfinite(lens), lens, 65.0)
print(f"{NF} frames · {NF/FPS:.0f} s · peak {present.max()} fish")

md = pd.date_range(START, END, freq="6h")
mp = [ephem.Moon(d.to_pydatetime()).moon_phase for d in md]
fulls, dt = [], ephem.Date((START - pd.Timedelta(days=25)).to_pydatetime())
for _ in range(6):
    dt = ephem.next_full_moon(dt)
    tt = pd.Timestamp(dt.datetime()) - pd.Timedelta(hours=5)
    if tt > END: break
    if tt >= START: fulls.append(tt)

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.family": "DejaVu Sans"})
FIG_W, FIG_H = 12.4, 9.4
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=105)
fig.patch.set_facecolor(DEEP)
_ar = ((X1 - X0) * KM_LON) / ((Y1 - Y0) * KM_LAT)
_h = 0.545
_w = _h * FIG_H / FIG_W * _ar
axm = fig.add_axes([(1 - _w) / 2 - 0.02, 0.410, _w, _h])
axk = fig.add_axes([0.075, 0.150, 0.887, 0.235])
axb = fig.add_axes([0.075, 0.048, 0.887, 0.082])

# --- map
axm.set_xlim(X0, X1); axm.set_ylim(Y0, Y1)
axm.set_aspect(KM_LAT / KM_LON)
axm.set_xticks([]); axm.set_yticks([])
for sp in axm.spines.values(): sp.set_visible(False)
axm.set_facecolor(SEA)
for k in range(3):
    axm.fill(lcx, lcy, facecolor="none", edgecolor=SHELF,
             linewidth=(3 - k) * 6, alpha=0.16, zorder=1)
axm.fill(lcx, lcy, facecolor=LAND, edgecolor=EDGE, linewidth=1.2, zorder=3)   # TRUE coast
axm.scatter(slon, slat, s=120, facecolor="none", edgecolor="#3B6F86", lw=1.1, zorder=4)
axm.scatter(slon, slat, s=9, c="#3B6F86", zorder=4)
for s in stations:
    axm.annotate(s.replace("LC REC ", ""), (co.lon[s], co.lat[s]), textcoords="offset points",
                 xytext=(0, 11), ha="center", fontsize=7.5, color="#527F94", zorder=4)

# Boxes marking the two spawning sites: solid = active west-end FSA (REC 1 / REC 16),
# dashed = historic east-end site (REC 10). Both are nudged seaward like the fish-point
# offsets, labeled at their bottom-right corners, and drawn behind the fish markers.
from matplotlib.patches import Rectangle
_cx, _cy = float(np.mean(lcx)), float(np.mean(lcy))
def _site_box(sts, ls, label, pad_km=0.7, shift_km=0.45):
    xs=[co.lon[t] for t in sts if t in stations]; ys=[co.lat[t] for t in sts if t in stations]
    mx, my = np.mean(xs), np.mean(ys)
    v = np.array([(mx - _cx) * KM_LON, (my - _cy) * KM_LAT]); v /= np.linalg.norm(v)
    dx, dy = shift_km * v[0] / KM_LON, shift_km * v[1] / KM_LAT
    px, py = pad_km / KM_LON, pad_km / KM_LAT
    x0, y0 = min(xs) - px + dx, min(ys) - py + dy
    x1_, y1_ = max(xs) + px + dx, max(ys) + py + dy
    axm.add_patch(Rectangle((x0, y0), x1_ - x0, y1_ - y0, fill=False, edgecolor="#BBD7E4",
                            linestyle=ls, linewidth=1.3, alpha=0.85, zorder=5))
    axm.text(x1_, y0 - 0.12 * py, label, ha="right", va="top",
             fontsize=9, color="#BBD7E4", linespacing=1.15, zorder=5)
_site_box(["LC REC 1", "LC REC 16"], "-", "spawning\naggregation")
_site_box(["LC REC 10"], (0, (5, 4)), "historic\nspawning site")
bar = 2.0 / KM_LON
_bx = X1 - 0.010 - bar
axm.plot([_bx, _bx + bar], [Y0 + 0.0055] * 2, color=INK, lw=2.6, zorder=8)
axm.text(_bx + bar / 2, Y0 + 0.0080, "2 km", ha="center", fontsize=8.5, color=INK, zorder=8)
axm.text(np.mean(lcx), np.mean(lcy) - 0.0035, "LITTLE CAYMAN", fontsize=11.5,
         color="#9C8E70", ha="center", style="italic", zorder=4)
axm.text(0.988, 0.105,
         "each fish is drawn 0.2–0.6 km SEAWARD of the receiver that heard it, so\n"
         "overlapping animals stay separable — a VR2W hears out to roughly 0.5 km",
         transform=axm.transAxes, fontsize=7.8, color="#5C7C8C",
         ha="right", va="bottom", zorder=8)

trails = LineCollection([], linewidths=1.5, zorder=6); axm.add_collection(trails)
glow = axm.scatter([], [], s=260, c=[], cmap=RAMP, norm=Normalize(50, 85),
                   alpha=0.20, edgecolor="none", zorder=6)
dots = axm.scatter([], [], s=68, c=[], cmap=RAMP, norm=Normalize(50, 85),
                   edgecolor=DEEP, linewidths=0.9, zorder=7)
stamp = axm.text(0.010, 0.955, "", transform=axm.transAxes, fontsize=18, color=INK,
                 va="top", family="DejaVu Sans Mono")
sub = axm.text(0.010, 0.890, "", transform=axm.transAxes, fontsize=12, color=MUTE, va="top")

# --- density panel
KB, TB = 0.6, 6
ybins = np.arange(-0.3, ALONG_MAX + KB, KB)
H = np.zeros((len(ybins) - 1, NF // TB + 1))
for fi in range(len(fish)):
    y = alo[fi]; okm = ~np.isnan(y)
    if not okm.any(): continue
    np.add.at(H, (np.clip(np.digitize(y[okm], ybins) - 1, 0, len(ybins) - 2),
                  np.where(okm)[0] // TB), 1.0)
H /= TB
axk.set_facecolor("#08202D")
for sp in axk.spines.values(): sp.set_color("#1E4356")
DENS = LinearSegmentedColormap.from_list("d", ["#08202D", "#12455C", "#2E9BC4", "#8FE3F0", "#FFE9A8"])
axk.imshow(H, aspect="auto", origin="lower", cmap=DENS, interpolation="nearest",
           extent=[0, NF - 1, ybins[0], ybins[-1]],
           norm=PowerNorm(gamma=0.35, vmin=0,
                          vmax=np.percentile(H[H > 0], 98) if (H > 0).any() else 1))
# one tick per receiver, so the map -> axis mapping is visible
for s in stations:
    axk.plot([0, NF * 0.008], [along[s]] * 2, color="#9DC0D0", lw=1.0, zorder=4)
    axk.plot([NF * 0.992, NF - 1], [along[s]] * 2, color="#9DC0D0", lw=1.0, zorder=4)
axk.axhline(along[FSA_ST], color=FSA_C, lw=1.1, alpha=0.7)
axk.text(NF * 0.010, along[FSA_ST] + 0.35, "spawning aggregation (west end)",
         color=FSA_C, fontsize=9, va="bottom")
axk.text(NF * 0.985, ALONG_MAX - 0.4, "east end of the island",
         color="#8FB3C4", fontsize=9, va="top", ha="right")
for t in fulls:
    axk.axvline(fnum(t), color="#FFC24D", lw=1.0, ls=(0, (4, 3)))
axk.set_xlim(0, NF - 1); axk.set_ylim(ybins[0], ybins[-1])
axk.set_yticks(np.arange(0, ALONG_MAX + 1, 5))
axk.set_ylabel("distance east along the island\nfrom the aggregation (km)",
               fontsize=9.5, color=MUTE, labelpad=6)
axk.set_xticks([]); axk.tick_params(colors=MUTE, labelsize=8)
axk.text(0.998, 1.035, "where the fish are: brightness = number of fish in each 0.6 km "
         f"slice of shoreline   ·   ticks mark the {len(stations)} receivers",
         transform=axk.transAxes, ha="right", va="bottom", fontsize=8.5, color=MUTE)
khead = axk.axvline(0, color=INK, lw=1.6, zorder=5)

# --- moon + attendance
axb.set_facecolor("#0B2331")
for sp in axb.spines.values(): sp.set_color("#1E4356")
axb.plot(np.arange(NF), np.interp(np.arange(NF), np.linspace(0, NF - 1, len(mp)), mp),
         color="#6FA8BD", lw=1.3)
axb.fill_between(np.arange(NF), present / max(present.max(), 1), color=ACC, alpha=0.30, lw=0)
axb.plot(np.arange(NF), present / max(present.max(), 1), color=ACC, lw=1.2)
for t in fulls:
    axb.axvline(fnum(t), color="#FFC24D", lw=1.0, ls=(0, (4, 3)))
    axb.annotate(f"full moon {t:%d %b}", (fnum(t), 1.06), ha="center", va="bottom",
                 fontsize=8.5, color="#FFC24D", annotation_clip=False)
tk = pd.date_range(START, END, freq="7D")
axb.set_xticks([fnum(t) for t in tk]); axb.set_xticklabels([f"{t:%d %b}" for t in tk])
axb.set_xlim(0, NF - 1); axb.set_ylim(0, 1.05); axb.set_yticks([])
axb.tick_params(colors=MUTE, labelsize=8.5)
axb.text(0.006, 0.83, "moon phase", transform=axb.transAxes, fontsize=8.5, color="#6FA8BD", va="top")
axb.text(0.115, 0.83, "fish present", transform=axb.transAxes, fontsize=8.5, color=ACC, va="top")
bhead = axb.axvline(0, color=INK, lw=1.6)

cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(50, 85), cmap=RAMP),
                  ax=[axm], fraction=0.016, pad=0.006)
cb.set_label("total length (cm)", fontsize=9, color=MUTE)
cb.outline.set_visible(False); cb.ax.tick_params(colors=MUTE, labelsize=8)

TR = int(TRAIL_H * 60 / STEP_MIN)
def tint(t):
    h = t.hour + t.minute / 60
    return "#081C27" if (h < 5.5 or h > 21.5) else ("#0B2431" if (h < 7 or h > 20) else SEA)

def update(k):
    t = frames[k]
    axm.set_facecolor(tint(t))
    m = here[:, k]
    p = pos[m, k]
    dots.set_offsets(p); dots.set_array(lens[m])
    glow.set_offsets(p); glow.set_array(lens[m])
    sg, cl = [], []
    lo = max(0, k - TR)
    for fi in np.where(m)[0]:
        tr = pos[fi, lo:k + 1]; tr = tr[~np.isnan(tr[:, 0])]
        if len(tr) > 2:
            s_ = np.stack([tr[:-1], tr[1:]], 1); sg.extend(s_)
            cl.extend([(0.31, 0.76, 0.85, a) for a in np.linspace(0.05, 0.60, len(s_))])
    trails.set_segments(sg); trails.set_color(cl)
    khead.set_xdata([k, k]); bhead.set_xdata([k, k])
    dafm = min(((t - f).total_seconds() / 86400 for f in fulls), key=abs)
    stamp.set_text(f"{t:%a %d %b %Y   %H:%M}")
    sub.set_text(f"{int(m.sum())} fish present     {dafm:+.1f} d from full moon")
    return [trails, dots, glow, khead, bhead, stamp, sub]

if __name__ == "__main__":
    import sys
    if "--still" in sys.argv:
        # render two representative frames as PNGs instead of the full movie
        for k in (int(NF * 0.28), int(NF * 0.42)):
            update(k)
            fig.savefig(os.path.join(OUTDIR, f"MovieS1_still_{k}.png"),
                        facecolor=fig.get_facecolor(), dpi=105)
        print("stills written to output/")
    else:
        print("rendering ...")
        FuncAnimation(fig, update, frames=NF, blit=False).save(
            OUT, writer=FFMpegWriter(fps=FPS, bitrate=4600),
            savefig_kwargs=dict(facecolor=fig.get_facecolor()))
        print("saved", OUT)
