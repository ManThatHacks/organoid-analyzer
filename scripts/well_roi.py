"""Find the well floor in a photo that shows the whole well (and the plate around it).

The well floor is a bright disc bounded by a dark wall. The search scores every candidate
circle by how much darker the wall is than the floor just inside it, *all the way round*
(a low percentile over angles), so partial arcs from plate rings and lens vignetting don't
win. The best circle is then refined at full resolution by locating the wall's inner edge
along 360 rays and fitting a circle to those points.

NumPy only, so both pipelines can use it without adding a dependency.
"""
import warnings
import numpy as np

COARSE = 8                      # search at 1/8 scale
R_FRAC = (0.25, 0.55)           # well radius, as a fraction of the shorter image side
CENTRE_FRAC = 0.22              # centre within this fraction of the shorter side from the image centre
MIN_SCORE = 8.0                 # grey levels: wall must be this much darker than the floor all round


def _blur(a, sigma):
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    a = np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="edge"), k, "valid"), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="edge"), k, "valid"), 1, a)


def _sample(img, x, y):
    """Bilinear sampling; NaN outside the image."""
    h, w = img.shape
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    ok = (x0 >= 0) & (y0 >= 0) & (x0 < w - 1) & (y0 < h - 1)
    x0c, y0c = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
    fx, fy = x - x0c, y - y0c
    v = (img[y0c, x0c] * (1 - fx) * (1 - fy) + img[y0c, x0c + 1] * fx * (1 - fy)
         + img[y0c + 1, x0c] * (1 - fx) * fy + img[y0c + 1, x0c + 1] * fx * fy)
    return np.where(ok, v, np.nan)


def _score_centre(s, cx, cy, radii, ang):
    """Score each radius for one centre: low percentile (over angles) of floor minus wall."""
    rr = np.arange(int(radii[0] * 0.8), int(radii[-1] * 1.08) + 2)
    P = _sample(s, cx + np.outer(np.cos(ang), rr), cy + np.outer(np.sin(ang), rr))   # angles x radii
    out = np.full(len(radii), -np.inf)
    for k, r in enumerate(radii):
        i0, i1 = int(0.84 * r) - rr[0], int(0.93 * r) - rr[0] + 1      # floor just inside
        w0, w1 = int(0.97 * r) - rr[0], int(1.05 * r) - rr[0] + 1      # the wall
        floor = np.nanmean(P[:, i0:i1], axis=1)
        wall = np.nanmin(P[:, w0:w1], axis=1)
        d = floor - wall
        valid = ~np.isnan(d)
        if valid.mean() < 0.6:
            continue
        out[k] = np.percentile(d[valid], 20)
    return out


def find_well(gray):
    """Return dict(cx, cy, r, r_wall, score) in full-resolution pixels, or None.

    r is the inner edge of the wall (use it as the region of interest); r_wall is the
    darkest point of the wall (use 2 * r_wall for calibration against the well diameter).
    """
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN rays outside the image are expected
        return _find_well(gray)


def _find_well(gray):
    g = np.asarray(gray, dtype=np.float64)
    H, W = g.shape
    F = COARSE
    s = g[: H // F * F, : W // F * F].reshape(H // F, F, W // F, F).mean(axis=(1, 3))
    s = _blur(s, 1.5)
    h, w = s.shape
    side = min(h, w)
    radii = np.arange(int(R_FRAC[0] * side), int(R_FRAC[1] * side) + 1)
    ang = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    span, step = int(CENTRE_FRAC * side), 4
    # coarse grid: keep every (centre, radius) that is a local maximum along the radius
    cands = []
    for cy in np.arange(h / 2 - span, h / 2 + span + 1, step):
        for cx in np.arange(w / 2 - span, w / 2 + span + 1, step):
            sc = _score_centre(s, cx, cy, radii, ang)
            for k in range(len(radii)):
                if np.isfinite(sc[k]) and sc[k] >= sc[max(k - 1, 0)] and sc[k] >= sc[min(k + 1, len(radii) - 1)]:
                    cands.append((sc[k], cx, cy, radii[k]))
    if not cands:
        return None
    top = max(c[0] for c in cands)
    if top < MIN_SCORE:
        return None
    # concentric plate rings sit OUTSIDE the well: take the smallest circle that scores nearly as well as the best
    good = [c for c in cands if c[0] >= 0.75 * top]
    rmin = min(c[3] for c in good)
    score, cx, cy, r = max((c for c in good if c[3] <= rmin * 1.04), key=lambda c: c[0])
    # fine centre search around it, radius kept near the chosen ring
    near = radii[(radii >= r * 0.96) & (radii <= r * 1.04)]
    for cy2 in np.arange(cy - 4, cy + 5):
        for cx2 in np.arange(cx - 4, cx + 5):
            sc = _score_centre(s, cx2, cy2, near, ang)
            k = int(np.argmax(sc))
            if sc[k] > score:
                score, cx, cy, r = sc[k], cx2, cy2, near[k]
    # ---- refine at full resolution: inner edge of the wall along 360 rays, robust circle fit
    cx, cy, r = (cx + 0.5) * F - 0.5, (cy + 0.5) * F - 0.5, r * F
    gs = _blur(g, 2.0) if max(H, W) <= 2500 else g
    a = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    rr = np.arange(int(0.85 * r), int(1.08 * r))
    P = _sample(gs, cx + np.outer(np.cos(a), rr), cy + np.outer(np.sin(a), rr))
    lo, hi = np.searchsorted(rr, 0.96 * r), np.searchsorted(rr, 1.06 * r)   # the wall is near the coarse radius;
    edge, wallr = [], []                                                    # further out are other plate rings
    for k in range(len(a)):
        p = P[k]
        if np.isnan(p).mean() > 0.2:
            continue
        p = np.convolve(np.nan_to_num(p, nan=np.nanmedian(p)), np.ones(5) / 5, "same")
        jw = int(np.argmin(p[lo:hi])) + lo                           # wall minimum
        floor = np.median(p[: max(3, len(p) // 6)])
        half = (floor + p[jw]) / 2
        j = jw
        while j > 0 and p[j] < half:                                 # walk inwards to the inner edge
            j -= 1
        edge.append((a[k], rr[j])); wallr.append((a[k], rr[jw]))
    if len(edge) < 120:
        return dict(cx=cx, cy=cy, r=0.97 * r, r_wall=r, score=float(score), refined=False)

    def fit(points, x0, y0):
        A = np.array([[x0 + rad * np.cos(t), y0 + rad * np.sin(t)] for t, rad in points])
        keep = np.ones(len(A), bool)
        for _ in range(3):                                           # algebraic fit, drop outliers
            X = A[keep]
            M = np.column_stack([X[:, 0], X[:, 1], np.ones(len(X))])
            b = -(X ** 2).sum(1)
            D, E, Fc = np.linalg.lstsq(M, b, rcond=None)[0]
            xc, yc = -D / 2, -E / 2
            rc = np.sqrt(xc ** 2 + yc ** 2 - Fc)
            res = np.abs(np.hypot(A[:, 0] - xc, A[:, 1] - yc) - rc)
            keep = res < max(3.0, 2.5 * np.median(res[keep]))
        return xc, yc, rc, keep.mean()

    xc, yc, rin, frac = fit(edge, cx, cy)
    _, _, rwall, _ = fit(wallr, cx, cy)
    return dict(cx=float(xc), cy=float(yc), r=float(rin), r_wall=float(rwall), score=float(score),
                refined=True, inlier_fraction=round(float(frac), 2))


def well_mask(shape, well, margin_frac=0.01):
    H, W = shape
    yy, xx = np.ogrid[:H, :W]
    return (xx - well["cx"]) ** 2 + (yy - well["cy"]) ** 2 <= (well["r"] * (1 - margin_frac)) ** 2
