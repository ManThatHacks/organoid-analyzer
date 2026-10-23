#!/usr/bin/env python3
"""Size-distribution figure from organoids.csv.

Usage:  python plot_distribution.py <results_dir> [um_per_px]
Needs matplotlib (not required by the pipeline itself).
"""
import csv, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1].rstrip("/\\") + os.sep
um = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
rows = list(csv.DictReader(open(d + "organoids.csv")))
if not rows:
    sys.exit("organoids.csv is empty")

A = np.array([float(r["area_px2"]) for r in rows])
D = np.array([float(r["eq_diameter_px"]) for r in rows])
C = np.array([float(r["circularity"]) for r in rows])
if um > 0:
    A, D = A * um * um, D * um
    ua, ud = "µm²", "µm"
else:
    ua, ud = "px²", "px"

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].hist(D, bins=25, color="#4C78A8", edgecolor="w")
ax[0].set_xlabel("equivalent diameter (%s)" % ud); ax[0].set_ylabel("count")
ax[0].set_title("Diameter — n=%d\nmedian %.1f %s" % (len(rows), np.median(D), ud))
ax[1].hist(A, bins=25, color="#72B7B2", edgecolor="w")
ax[1].set_xlabel("area (%s)" % ua)
ax[1].set_title("Area\nmedian %.0f %s   total %.0f %s" % (np.median(A), ua, A.sum(), ua))
ax[2].scatter(D, C, s=14, c="#E45756", alpha=.75)
ax[2].set_xlabel("equivalent diameter (%s)" % ud); ax[2].set_ylabel("circularity")
ax[2].set_ylim(0.5, 1.02); ax[2].set_title("Circularity vs size")
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(d + "size_distribution.png", dpi=130)
print("wrote " + d + "size_distribution.png")
