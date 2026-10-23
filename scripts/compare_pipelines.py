#!/usr/bin/env python3
"""Compare two organoid segmentations of the same image, object by object.

Usage:  python compare_pipelines.py <results_A> <results_B> [image] [out_dir]

Each results folder must contain mask_labels.png (16-bit label image, as
written by organoid_quant.py and organoid_quant_skimage.py). Objects are
paired one-to-one by maximum overlap (Hungarian assignment on IoU); a pair
counts as a match when IoU >= MATCH_IOU.

Writes to out_dir (default: results_B):
  comparison.json         counts, matching and area agreement
  comparison_pairs.csv    one row per object: matched pair, A-only or B-only
  comparison_overlay.png  yellow = matched, red = A only, cyan = B only
                          (only if the image is given)
Needs numpy, scipy, scikit-image.
"""
import csv, json, os, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage import io, color, exposure, segmentation, morphology, measure

MATCH_IOU = 0.30

if len(sys.argv) < 3:
    sys.exit(__doc__)
dir_a, dir_b = sys.argv[1], sys.argv[2]
image = sys.argv[3] if len(sys.argv) > 3 else None
out = sys.argv[4] if len(sys.argv) > 4 else dir_b
os.makedirs(out, exist_ok=True)
name_a = os.path.basename(os.path.normpath(dir_a))
name_b = os.path.basename(os.path.normpath(dir_b))

A = io.imread(os.path.join(dir_a, "mask_labels.png")).astype(np.int64)
B = io.imread(os.path.join(dir_b, "mask_labels.png")).astype(np.int64)
if A.shape != B.shape:
    sys.exit("label images differ in size: %s vs %s" % (A.shape, B.shape))
na, nb = int(A.max()), int(B.max())

# ---- pairwise overlaps via a joint histogram (fast, no per-object loops)
area_a = np.bincount(A.ravel(), minlength=na + 1)
area_b = np.bincount(B.ravel(), minlength=nb + 1)
both = (A > 0) & (B > 0)
inter = np.zeros((na + 1, nb + 1), np.int64)
np.add.at(inter, (A[both], B[both]), 1)
inter = inter[1:, 1:]
union = area_a[1:, None] + area_b[None, 1:] - inter
iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)

pairs = []
if na and nb:
    ra, rb = linear_sum_assignment(-iou)
    pairs = [(i + 1, j + 1, iou[i, j]) for i, j in zip(ra, rb) if iou[i, j] >= MATCH_IOU]
ma = {a for a, _, _ in pairs}
mb = {b for _, b, _ in pairs}
only_a = [a for a in range(1, na + 1) if a not in ma and area_a[a] > 0]
only_b = [b for b in range(1, nb + 1) if b not in mb and area_b[b] > 0]

def centroids(L, n):
    c = np.full((n + 1, 2), np.nan)
    for p in measure.regionprops(L):
        c[p.label] = p.centroid[::-1]          # (x, y)
    return c
ca, cb = centroids(A, na), centroids(B, nb)

pa = np.array([area_a[a] for a, _, _ in pairs], float)
pb = np.array([area_b[b] for _, b, _ in pairs], float)
rel = 100 * (pb - pa) / np.maximum(pa, 1)
summary = {
    "A": dir_a, "B": dir_b, "match_iou": MATCH_IOU,
    "n_A": na, "n_B": nb, "n_matched": len(pairs),
    "n_A_only": len(only_a), "n_B_only": len(only_b),
    "pct_of_A_found_by_B": round(100 * len(pairs) / max(na, 1), 1),
    "pct_of_B_found_by_A": round(100 * len(pairs) / max(nb, 1), 1),
    "total_area_A_px2": int(area_a[1:].sum()), "total_area_B_px2": int(area_b[1:].sum()),
    "total_area_diff_pct": round(100 * (area_b[1:].sum() - area_a[1:].sum()) / max(area_a[1:].sum(), 1), 2),
    "pixel_iou_all_objects": round(float(((A > 0) & (B > 0)).sum() / max(((A > 0) | (B > 0)).sum(), 1)), 3),
}
if pairs:
    summary["matched"] = {
        "median_iou": round(float(np.median([p[2] for p in pairs])), 3),
        "median_area_diff_pct": round(float(np.median(rel)), 2),
        "median_abs_area_diff_pct": round(float(np.median(np.abs(rel))), 2),
        "area_correlation_r": round(float(np.corrcoef(pa, pb)[0, 1]), 4) if len(pairs) > 2 else None,
        "matched_area_A_px2": int(pa.sum()), "matched_area_B_px2": int(pb.sum()),
    }
    summary["A_only_area_px2"] = int(sum(area_a[a] for a in only_a))
    summary["B_only_area_px2"] = int(sum(area_b[b] for b in only_b))

with open(os.path.join(out, "comparison.json"), "w") as f:
    json.dump(summary, f, indent=2)

with open(os.path.join(out, "comparison_pairs.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["status", "id_A", "id_B", "iou", "area_A_px2", "area_B_px2",
                "area_diff_pct", "x", "y"])
    for (a, b, v), r in zip(pairs, rel):
        w.writerow(["matched", a, b, round(v, 3), area_a[a], area_b[b], round(r, 1),
                    round(ca[a, 0], 1), round(ca[a, 1], 1)])
    for a in only_a:
        w.writerow(["A_only", a, "", "", area_a[a], "", "", round(ca[a, 0], 1), round(ca[a, 1], 1)])
    for b in only_b:
        w.writerow(["B_only", "", b, "", "", area_b[b], "", round(cb[b, 0], 1), round(cb[b, 1], 1)])

if image:
    g = io.imread(image)
    if g.ndim == 3:
        g = color.rgb2gray(g[..., :3])
    rgb = np.dstack([exposure.equalize_adapthist(g / max(g.max(), 1))] * 3)
    def outline(L, ids):
        keep = np.zeros(L.max() + 1, bool); keep[list(ids)] = True
        return morphology.dilation(segmentation.find_boundaries(np.where(keep[L], L, 0), mode="inner"),
                                   morphology.disk(1))
    rgb[outline(A, only_a)] = (1, 0.1, 0.1)
    rgb[outline(B, only_b)] = (0, 0.9, 1)
    rgb[outline(A, ma)] = (1, 0.9, 0)
    io.imsave(os.path.join(out, "comparison_overlay.png"), (rgb * 255).astype(np.uint8),
              check_contrast=False)

print(json.dumps(summary, indent=2))
print("\nA = %s, B = %s" % (name_a, name_b))
