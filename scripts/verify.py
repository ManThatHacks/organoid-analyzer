#!/usr/bin/env python3
"""Check that your environment reproduces the published control-well result.

Usage:  python verify.py <results_dir>

Pass = your run agrees with the reference within tolerance. Small area
differences across OpenCV versions are expected and harmless; a different
object COUNT means something is actually wrong.
"""
import json, sys, os

REF = {"n_organoids": 125, "total_area_px2": 157708.0,
       "median_eq_diameter_px": 30.3, "n_candidates": 414}
COUNT_TOL, AREA_TOL_PCT = 0, 1.0

d = sys.argv[1].rstrip("/\\") + os.sep
s = json.load(open(d + "summary.json"))

print("                      yours      reference")
print("organoids         %9d %14d" % (s["n_organoids"], REF["n_organoids"]))
print("total area px2    %9.0f %14.0f" % (s["total_area_px2"], REF["total_area_px2"]))
print("median diam px    %9.1f %14.1f" % (s["eq_diameter_px"]["median"], REF["median_eq_diameter_px"]))
print("hough candidates  %9d %14d" % (s["n_candidates"], REF["n_candidates"]))

dn = abs(s["n_organoids"] - REF["n_organoids"])
da = 100 * abs(s["total_area_px2"] - REF["total_area_px2"]) / REF["total_area_px2"]
print("\ncount difference : %d (allowed %d)" % (dn, COUNT_TOL))
print("area  difference : %.3f%% (allowed %.1f%%)" % (da, AREA_TOL_PCT))

ok = dn <= COUNT_TOL and da <= AREA_TOL_PCT
print("\n" + ("PASS - your environment reproduces the reference result." if ok else
      "FAIL - see the README's reproducibility section."))
sys.exit(0 if ok else 1)
