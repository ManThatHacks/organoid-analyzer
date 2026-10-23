#!/usr/bin/env python3
"""Organoid segmentation + morphometry, scikit-image / SciPy implementation.

An independent second method for cross-checking organoid_quant.py (OpenCV).
Same command line, same output files, different algorithm at every stage:

                      organoid_quant.py (OpenCV)        this script (scikit-image / SciPy)
  detection           HOUGH_GRADIENT_ALT on CLAHE image  classic Hough transform on a skeletonised
                                                         dark-ridge (Sato) map, 3 pyramid levels
  verification        intensity depth of the dark rim    ridge strength + angular coverage along
                                                         the circle
  outline             polar rim-minimum tracing          marker-controlled watershed on the ridge map
  touching objects    nearest normalised centre          watershed line along the shared rim

Usage:  python organoid_quant_skimage.py <image> <output_dir> [um_per_px] [--roi auto|field|well]
                                         [--well-diameter-mm D]
--roi and --well-diameter-mm work as in organoid_quant.py (shared well detection: well_roi.py).
Needs numpy, scipy, scikit-image (Pillow comes with scikit-image).
"""
import argparse, csv, json, os, sys
import numpy as np
from scipy import ndimage as ndi
from skimage import color, draw, exposure, filters, io, measure, morphology, segmentation, transform
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from well_roi import find_well, well_mask

ap = argparse.ArgumentParser(description="Organoid segmentation, scikit-image / SciPy implementation")
ap.add_argument("image"); ap.add_argument("output_dir")
ap.add_argument("um_per_px", nargs="?", type=float)
ap.add_argument("--roi", choices=["auto", "field", "well"], default="auto")
ap.add_argument("--well-diameter-mm", type=float)
args = ap.parse_args()
IMG, OUT, UM_PER_PX = args.image, args.output_dir, args.um_per_px
if UM_PER_PX and args.well_diameter_mm:
    sys.exit("give either um_per_px or --well-diameter-mm, not both")
WELL_VS_FIELD = 0.85       # auto ROI: use the well when its area is below this fraction of the lit field
R_MAX_FRAC = 0.14          # largest organoid radius searched in well mode, as a fraction of the well radius

# ---------------------------------------------------------------- parameters
FIELD_THRESH = 30          # grey level separating the illuminated disc from the black surround
FIELD_SHRINK = 15          # px pulled in from the field edge (drops the bright rim artefact)
FLAT_SIGMA = 40            # px, background flattening scale
RIDGE_SIGMAS = (1, 2, 3)   # Sato filter scales for the dark rim
# Hough levels: (downsample factor, radii at that level, ridge sigmas at that level).
# Together they cover radii 5-100 px in full-resolution pixels.
HOUGH_LEVELS = [(1, np.arange(5, 16), (1, 2)),
                (2, np.arange(7, 21), (1, 2)),
                (4, np.arange(9, 26), (1, 1.5))]
EDGE_PCTL = 90             # ridge percentile that becomes a Hough edge point
HOUGH_MIN = 0.35           # minimum fraction of the circle supported by edge points
RIM_PCTL = 90              # ridge percentile that counts as "rim present" at an angle
RIM_COV_MIN = 0.80         # rim present on >= 80 % of angles      } organoid-vs-debris
RIM_STRENGTH_MIN = 5.0     # median ridge response along the circle } boundary
AREA_MIN, CIRC_MIN, SOLIDITY_MIN = 80, 0.55, 0.85
SIZE_RATIO = (0.45, 1.8)   # final area / Hough circle area; rejects leaks and collapses


def load_gray(path):
    if not os.path.isfile(path):
        sys.exit("image not found: %s" % path)
    g = io.imread(path)                       # handles accented / non-ASCII paths
    if g.ndim == 3:
        g = color.rgb2gray(g[..., :3]) * 255 if g.shape[-1] in (3, 4) else g[..., 0]
    if g.dtype == np.uint16:
        g = g / 257.0
    elif g.dtype.kind == "f" and g.max() <= 1.0:
        g = g * 255
    return g.astype(np.float64)


g = load_gray(IMG)
H, W = g.shape
os.makedirs(OUT, exist_ok=True)
disk = morphology.disk

# ---------- 1. field of view ----------
field = ndi.binary_opening(g > FIELD_THRESH, disk(12))
lab_f = measure.label(field)
if lab_f.max() == 0:
    sys.exit("no illuminated field found (nothing brighter than %d)" % FIELD_THRESH)
field = ndi.binary_fill_holes(lab_f == 1 + np.argmax(np.bincount(lab_f.ravel())[1:]))
roi = ndi.binary_erosion(field, disk(FIELD_SHRINK))
roi_mode, well = "field", None
if args.roi != "field":
    well = find_well(g)
    if well is None and args.roi == "well":
        sys.exit("no well found in %s; run with --roi field" % IMG)
    if well is not None:
        wm = well_mask(g.shape, well)
        if args.roi == "well" or wm.sum() < WELL_VS_FIELD * field.sum():
            roi_mode = "well"
            field = wm & field              # flatten and analyse inside the well only, so the plate
            roi = field                     # plastic around it influences nothing
if args.well_diameter_mm:
    if roi_mode != "well":
        sys.exit('--well-diameter-mm needs a detected well (ROI mode is "%s")' % roi_mode)
    UM_PER_PX = args.well_diameter_mm * 1000 / (2 * well["r"])
if roi_mode == "well":                      # whole-well photos: organoids can be larger than 100 px
    r_max = R_MAX_FRAC * well["r"]
    if r_max > 100:
        HOUGH_LEVELS = HOUGH_LEVELS + [(8, np.arange(12, int(r_max / 8) + 2), (1, 1.5))]

# ---------- 2. flatten illumination, enhance dark rims ----------
num = filters.gaussian(np.where(field, g, 0), FLAT_SIGMA)
den = filters.gaussian(field.astype(float), FLAT_SIGMA)
flat = np.where(field, g - num / np.maximum(den, 1e-6), 0)
ridge = filters.sato(flat, sigmas=RIDGE_SIGMAS, black_ridges=True)
ridge[~roi] = 0

# ---------- 3. multi-scale Hough candidates ----------
cands = []
for f, radii, sig in HOUGH_LEVELS:
    small = transform.downscale_local_mean(flat, (f, f)) if f > 1 else flat
    rmask = transform.resize(roi, small.shape, order=0) > 0.5
    rr = filters.sato(small, sigmas=sig, black_ridges=True)
    rr[~rmask] = 0
    edges = morphology.skeletonize(rr > np.percentile(rr[rmask], EDGE_PCTL))
    hs = transform.hough_circle(edges, radii, normalize=True)
    acc, cx, cy, rad = transform.hough_circle_peaks(hs, radii, min_xdistance=3, min_ydistance=3,
                                                     threshold=HOUGH_MIN, total_num_peaks=5000)
    off = (f - 1) / 2
    cands += [(x * f + off, y * f + off, r * f) for x, y, r in zip(cx, cy, rad)]
C = np.array(cands, float).reshape(-1, 3)
if len(C):
    C = C[roi[np.clip(C[:, 1].astype(int), 0, H - 1), np.clip(C[:, 0].astype(int), 0, W - 1)]]
n_candidates = len(C)

# ---------- 4. rim verification along each circle ----------
NA = 72
ang = np.linspace(0, 2 * np.pi, NA, endpoint=False)
rim_thr = np.percentile(ridge[roi], RIM_PCTL)


def rim_stats(x, y, r):
    rs = np.linspace(0.85 * r, 1.15 * r, 5)
    X, Y = x + np.outer(np.cos(ang), rs), y + np.outer(np.sin(ang), rs)
    v = ndi.map_coordinates(ridge, [Y.ravel(), X.ravel()], order=1, mode="nearest")
    v = v.reshape(NA, -1).max(1)
    return float((v > rim_thr).mean()), float(np.median(v))


S = np.array([(*c, *rim_stats(*c)) for c in C]).reshape(-1, 5)     # x y r cov strength
S = S[(S[:, 3] >= RIM_COV_MIN) & (S[:, 4] >= RIM_STRENGTH_MIN)]

# ---------- 5. NMS: strongest rim first; no near-concentric duplicates, no centre inside a kept circle
kept = []
for i in np.argsort(-(S[:, 3] * S[:, 4])):
    x, y, r = S[i, :3]
    if all(np.hypot(x - k[0], y - k[1]) >= max(0.6 * max(r, k[2]), k[2] - 0.3 * r) for k in kept):
        kept.append(S[i])
K = np.array(kept).reshape(-1, 5)
n_after_nms = len(K)

# ---------- 6. marker-controlled watershed on the ridge map ----------
markers = np.zeros((H, W), np.int32)
reach = np.zeros((H, W), bool)
for j, (x, y, r, _, _) in enumerate(K, 1):
    markers[draw.disk((y, x), max(2, 0.35 * r), shape=(H, W))] = j
    reach[draw.disk((y, x), 1.35 * r + 3, shape=(H, W))] = True
bg = len(K) + 1
markers[~reach | ~roi] = bg
lab = segmentation.watershed(filters.gaussian(ridge, 1.0), markers) if len(K) else np.zeros((H, W), np.int32)
lab[lab == bg] = 0

keep = np.zeros(len(K) + 2, bool)
for p in measure.regionprops(lab):
    r = K[p.label - 1, 2]
    circ = 4 * np.pi * p.area / max(p.perimeter, 1) ** 2
    ratio = p.area / (np.pi * r * r)
    keep[p.label] = (p.area >= AREA_MIN and circ >= CIRC_MIN and p.solidity >= SOLIDITY_MIN
                     and SIZE_RATIO[0] <= ratio <= SIZE_RATIO[1])
src = np.where(keep[lab], lab, 0)
lab, fwd, inv = segmentation.relabel_sequential(src)

# ---------- 7. morphometry from the final masks ----------
recs = []
for p in measure.regionprops(lab, intensity_image=flat):
    k = K[inv[p.label] - 1]
    sl = tuple(slice(max(s.start - 20, 0), s.stop + 20) for s in p.slice)
    mm = lab[sl] == p.label
    band = mm & ~ndi.binary_erosion(mm, iterations=4)
    outer = ndi.binary_dilation(mm, iterations=10) & ~ndi.binary_dilation(mm, iterations=4)
    depth = float(np.median(flat[sl][outer]) - np.percentile(flat[sl][band], 25))
    per = float(p.perimeter)
    cy, cx = p.centroid
    MA, ma = p.axis_major_length, p.axis_minor_length
    r = dict(id=0, centroid_x=round(cx, 1), centroid_y=round(cy, 1),
             area_px2=int(p.area), eq_diameter_px=round(2 * np.sqrt(p.area / np.pi), 1),
             perimeter_px=round(per, 1),
             circularity=round(min(1.0, 4 * np.pi * p.area / max(per, 1) ** 2), 3),
             solidity=round(float(p.solidity), 3),
             minor_axis_px=round(ma, 1), major_axis_px=round(MA, 1),
             aspect_ratio=round(MA / max(ma, 1e-6), 2),
             rim_depth_gray=round(depth, 1), rim_coverage=round(float(k[3]), 2),
             _label=p.label)
    if UM_PER_PX:
        r["area_um2"] = round(p.area * UM_PER_PX ** 2, 1)
        r["eq_diameter_um"] = round(r["eq_diameter_px"] * UM_PER_PX, 1)
    recs.append(r)

# ids ordered by area, largest first. (organoid_quant.py numbers objects in detection
# order instead and only sorts the CSV rows by area, so the two never share IDs.)
recs.sort(key=lambda r: -r["area_px2"])
remap = np.zeros(lab.max() + 1, np.int32)
for i, r in enumerate(recs, 1):
    remap[r.pop("_label")] = i
    r["id"] = i
lab = remap[lab]

A = np.array([r["area_px2"] for r in recs], float)
D = np.array([r["eq_diameter_px"] for r in recs], float)


def stats(v, quart=False):
    if len(v) == 0:
        return None
    s = dict(min=float(v.min()), median=float(np.median(v)), mean=round(float(v.mean()), 1),
             max=float(v.max()), sd=round(float(v.std(ddof=1)), 1) if len(v) > 1 else 0.0)
    if quart:
        s.update(q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)))
    return s


summary = dict(image=IMG, method="scikit-image/scipy (Hough on ridge map + watershed)",
               image_size=[W, H], roi_area_px2=int(roi.sum()),
               n_candidates=int(n_candidates), n_after_nms=int(n_after_nms), n_organoids=len(recs),
               total_area_px2=float(A.sum()),
               pct_field_covered=round(100 * A.sum() / max(roi.sum(), 1), 2),
               area_px2=stats(A, quart=True), eq_diameter_px=stats(D), um_per_px=UM_PER_PX)
if UM_PER_PX:
    summary["total_area_um2"] = round(float(A.sum()) * UM_PER_PX ** 2, 1)
summary["roi_mode"] = roi_mode
if well is not None:
    summary["well"] = dict(cx=round(well["cx"], 1), cy=round(well["cy"], 1), r_floor_px=round(well["r"], 1),
                           r_wall_px=round(well["r_wall"], 1), used_as_roi=roi_mode == "well")
if args.well_diameter_mm:
    summary["scale_source"] = "well floor diameter %.3f mm = %.1f px" % (args.well_diameter_mm, 2 * well["r"])

cols = ["id", "centroid_x", "centroid_y", "area_px2", "eq_diameter_px", "perimeter_px",
        "circularity", "solidity", "minor_axis_px", "major_axis_px", "aspect_ratio",
        "rim_depth_gray", "rim_coverage"] + (["area_um2", "eq_diameter_um"] if UM_PER_PX else [])
with open(os.path.join(OUT, "organoids.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(recs)
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
io.imsave(os.path.join(OUT, "mask_labels.png"), lab.astype(np.uint16), check_contrast=False)
io.imsave(os.path.join(OUT, "mask_binary.png"), ((lab > 0) * 255).astype(np.uint8), check_contrast=False)

# ---------- 8. overlays ----------
base = exposure.equalize_adapthist(np.clip(g / 255, 0, 1), clip_limit=0.02)
base = np.dstack([base] * 3)
edge_px = morphology.dilation(segmentation.find_boundaries(lab, mode="inner"), disk(1))
rng = np.random.default_rng(0)
colmap = np.zeros((lab.max() + 1, 3))
colmap[1:] = rng.integers(70, 255, (lab.max(), 3)) / 255
ov = np.where((lab > 0)[..., None], 0.6 * base + 0.4 * colmap[lab], base)
ov[edge_px] = (1, 0, 0)
if roi_mode == "well":                       # show which region was analysed
    for t in range(3):
        ov[draw.circle_perimeter(int(round(well["cy"])), int(round(well["cx"])), int(round(well["r"])) + t, shape=(H, W))] = (1, 0.86, 0)
io.imsave(os.path.join(OUT, "overlay_mask.png"), (ov * 255).astype(np.uint8), check_contrast=False)

ann = base.copy()
ann[edge_px] = (1, 0, 0)
if roi_mode == "well":
    for t in range(3):
        ann[draw.circle_perimeter(int(round(well["cy"])), int(round(well["cx"])), int(round(well["r"])) + t, shape=(H, W))] = (1, 0.86, 0)
im = Image.fromarray((ann * 255).astype(np.uint8))
dr = ImageDraw.Draw(im)
for r in recs:
    dr.text((r["centroid_x"] - 6, r["centroid_y"] - 6), str(r["id"]), fill=(0, 30, 255))
im.save(os.path.join(OUT, "overlay_numbered.png"))

print(json.dumps(summary, indent=2))
