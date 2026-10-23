#!/usr/bin/env python3
"""Organoid segmentation + morphometry for low-contrast brightfield well images.

Pipeline: ROI -> multi-scale Hough candidates -> NMS -> dark-rim verification
-> per-object radial rim refinement -> overlap resolution -> morphometry.

Usage:  python organoid_quant.py <image> <output_dir> [um_per_px] [--roi auto|field|well]
                                 [--well-diameter-mm D]

ROI: "field" analyses the illuminated field of view (images like the control well, where
everything outside the well is black). "well" finds the well floor and analyses only inside
it (photos that show the plate around the well). "auto" (default) uses the well when it is
clearly smaller than the illuminated field, otherwise the field.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, cv2, csv, sys, json, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from well_roi import find_well, well_mask

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('image'); ap.add_argument('output_dir')
ap.add_argument('um_per_px', nargs='?', type=float, help='micrometres per pixel, if known')
ap.add_argument('--roi', choices=['auto', 'field', 'well'], default='auto')
ap.add_argument('--well-diameter-mm', type=float,
                help='inner diameter of the well floor; sets um_per_px from the detected well')
args = ap.parse_args()
IMG = args.image
OUT = os.path.join(args.output_dir, '')
UM_PER_PX = args.um_per_px
if UM_PER_PX and args.well_diameter_mm:
    sys.exit('give either um_per_px or --well-diameter-mm, not both')

RIM_DEPTH_MIN, RIM_COV_MIN, CIRC_MIN, AREA_MIN = 6.0, 0.60, 0.55, 40
WELL_VS_FIELD = 0.85     # auto ROI: use the well when its area is below this fraction of the lit field
R_MAX_FRAC = 0.14        # largest organoid radius searched in well mode, as a fraction of the well radius
# Small organoids in whole-well photos (well mode only):
SMALL_R_MAX = 15         # px; "small" = Hough radius up to this
SMALL_DEPTH_FLOOR = 6.0  # their rim must be at least this many grey levels darker than the surroundings
                         # (checked on the test photos: below ~6 the extra detections were mostly empty
                         # background, above it mostly real small rings)...
SMALL_DEPTH_K = 3.5      # ...and at least this many times the photo's own noise (grainy photos need more)

if not os.path.isfile(IMG):
    sys.exit('image not found: %s' % IMG)
g = cv2.imdecode(np.fromfile(IMG, np.uint8), cv2.IMREAD_GRAYSCALE)    # also handles accented paths
if g is None:
    sys.exit('could not read %s as an image' % IMG)
os.makedirs(OUT, exist_ok=True)
H, W = g.shape
gf = g.astype(np.float32)
gs = cv2.GaussianBlur(gf, (0, 0), 1.5)

# ---------- 1. ROI: illuminated field, or the well floor ----------
_, roi = cv2.threshold(g, 30, 255, cv2.THRESH_BINARY)
roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
n, l, st, _ = cv2.connectedComponentsWithStats(roi)
field = (l == 1 + np.argmax(st[1:, 4])).astype(np.uint8) * 255
roi = cv2.erode(field, np.ones((31, 31), np.uint8)) > 0
roi_mode, well = 'field', None
if args.roi != 'field':
    well = find_well(g)
    if well is None and args.roi == 'well':
        sys.exit('no well found in %s; run with --roi field' % IMG)
    if well is not None:
        wm = well_mask(g.shape, well)
        if args.roi == 'well' or wm.sum() < WELL_VS_FIELD * (field > 0).sum():
            roi, roi_mode = wm & (field > 0), 'well'
if args.well_diameter_mm:
    if roi_mode != 'well':
        sys.exit('--well-diameter-mm needs a detected well (ROI mode is "%s")' % roi_mode)
    UM_PER_PX = args.well_diameter_mm * 1000 / (2 * well['r'])

# ---------- 2. multi-scale circle candidates ----------
cl = cv2.GaussianBlur(cv2.createCLAHE(3.0, (16, 16)).apply(g), (0, 0), 1.2)
BANDS = [(4,8),(7,13),(12,20),(18,30),(28,45),(42,70),(65,95)]
if roi_mode == 'well':                       # whole-well photos: organoids can be larger than 95 px
    r_max = R_MAX_FRAC * well['r']
    for band in [(90,135),(128,190),(180,270)]:
        if band[0] < r_max: BANDS.append(band)
cands = []
for rmin, rmax in BANDS:
    for p2 in (0.45, 0.60):
        c = cv2.HoughCircles(cl, cv2.HOUGH_GRADIENT_ALT, dp=1.5,
                             minDist=max(5, int(rmin*1.2)), param1=200,
                             param2=p2, minRadius=rmin, maxRadius=rmax)
        if c is not None: cands.append(c[0])
if roi_mode == 'well':
    # small organoids in whole-well photos are only ~6-30 px across with soft rims, below what the
    # pass above picks up: search again on a 2x enlarged copy of the well, more sensitively
    ys, xs = np.nonzero(roi)
    y0b, y1b, x0b, x1b = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    up = cv2.resize(cl[y0b:y1b, x0b:x1b], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    for rmin, rmax in [(6, 12), (10, 18), (16, 26), (24, 34)]:          # radii in enlarged pixels
        c = cv2.HoughCircles(up, cv2.HOUGH_GRADIENT_ALT, dp=1, minDist=max(5, rmin),
                             param1=90, param2=0.35, minRadius=rmin, maxRadius=rmax)
        if c is not None:
            c = c[0].copy()
            c[:, :2] = (c[:, :2] - 0.5) / 2 + [x0b, y0b]
            c[:, 2] /= 2
            cands.append(c)
C = np.vstack(cands) if cands else np.zeros((0, 3), np.float32)
C = np.array([c for c in C
              if roi[int(np.clip(c[1],0,H-1)), int(np.clip(c[0],0,W-1))]]).reshape(-1, 3)

NA = 180
ang = np.linspace(0, 2*np.pi, NA, endpoint=False)
ca, sa = np.cos(ang), np.sin(ang)

def sample(X, Y):
    return cv2.remap(gs, np.clip(X,0,W-1).astype(np.float32),
                     np.clip(Y,0,H-1).astype(np.float32),
                     cv2.INTER_LINEAR).reshape(X.shape)

def rim_stats(x0, y0, r):
    """Dark-rim depth + angular coverage; organoids have a continuous dark rim."""
    rr = np.linspace(0.60*r, 1.45*r, max(12, int(0.9*r)))
    prof = sample(x0+np.outer(ca,rr), y0+np.outer(sa,rr))
    rb = np.linspace(1.5*r, 2.0*r, 8)
    bg = np.median(sample(x0+np.outer(ca,rb), y0+np.outer(sa,rb)), axis=1)
    d = bg - prof.min(axis=1)
    return float(np.median(d)), float((d > 4).mean())

def refine(x0, y0, r):
    """Per-angle radius of the dark rim -> true (non-circular) contour."""
    rr = np.linspace(0.55*r, 1.5*r, max(16, int(1.4*r)))
    prof = cv2.GaussianBlur(sample(x0+np.outer(ca,rr), y0+np.outer(sa,rr)), (1,5), 0)
    rad = rr[prof.argmin(axis=1)]
    if roi_mode == 'well':
        # side-lit photos: on the bright side the rim can vanish and the minimum lands on texture
        # inside the organoid. Where the rim is weak, use the organoid's typical radius instead.
        rb = np.linspace(1.5*r, 2.0*r, 8)
        bg = np.median(sample(x0+np.outer(ca,rb), y0+np.outer(sa,rb)), axis=1)
        depth = bg - prof.min(axis=1)
        strong = depth >= 0.5 * np.median(depth)
        m = np.median(rad[strong]) if strong.mean() > 0.3 else r
        # second pass: look for the rim only near that radius, so interior texture can't win
        win = (rr >= 0.85 * m) & (rr <= 1.15 * m)
        if win.sum() >= 3:
            pw = prof[:, win]
            rad = rr[win][pw.argmin(axis=1)]
            strong = (bg - pw.min(axis=1)) >= 0.5 * np.median(depth)
        rad = np.clip(np.where(strong, rad, m), 0.85 * m, 1.15 * m)
    pad = np.r_[rad[-9:], rad, rad[:9]]
    return np.array([np.median(pad[i:i+19]) for i in range(NA)])

scored = []
for x0, y0, r in C:
    d, cov = rim_stats(x0, y0, r)
    scored.append((x0, y0, r, d, cov))
S = np.array(scored).reshape(-1, 5)

# ---------- 3. NMS (strongest rim wins) + rim verification ----------
small_depth = None
if roi_mode == 'well':
    # small organoids may be fainter than RIM_DEPTH_MIN; how faint is safe depends on the photo's noise
    hp = (gs - cv2.GaussianBlur(gs, (0, 0), 3))[roi]
    noise = float(1.4826 * np.median(np.abs(hp - np.median(hp))))
    small_depth = max(SMALL_DEPTH_FLOOR, SMALL_DEPTH_K * noise)
    def valid(k):
        need = small_depth if k[2] <= SMALL_R_MAX else RIM_DEPTH_MIN
        return k[3] >= need and k[4] >= RIM_COV_MIN
    # many overlapping candidates here: verify first, so the best *valid* circle wins each spot
    S = S[[valid(k) for k in S]].reshape(-1, 5) if len(S) else S
kept = []
for i in np.argsort(-(S[:,3] * S[:,4])):
    x, y, r = S[i,0], S[i,1], S[i,2]
    if all(np.hypot(x-k[0], y-k[1]) > 0.55*max(r, k[2]) for k in kept):
        kept.append(tuple(S[i]))
n_nms = len(kept)
if roi_mode != 'well':
    kept = [k for k in kept if k[3] >= RIM_DEPTH_MIN and k[4] >= RIM_COV_MIN]
if roi_mode == 'well':
    # organoids don't sit inside each other: a small circle centred well inside a much larger one
    # is texture inside that organoid, and would otherwise carve a wedge out of it
    kept = [k for k in kept if not any(b[2] > 2 * k[2] and np.hypot(k[0]-b[0], k[1]-b[1]) < b[2] - 0.5*k[2]
                                       for b in kept)]

# ---------- 4. refine contours ----------
objs = []
for x0, y0, r0, d, cov in kept:
    rad = refine(x0, y0, r0)
    pts = np.stack([x0 + rad*ca, y0 + rad*sa], 1)
    A = 0.5*abs(np.dot(pts[:,0], np.roll(pts[:,1],-1)) -
                np.dot(np.roll(pts[:,0],-1), pts[:,1]))
    P = float(np.sum(np.hypot(*(np.roll(pts,-1,0)-pts).T)))
    if A < AREA_MIN or 4*np.pi*A/P**2 < CIRC_MIN: continue
    objs.append(dict(x=x0, y=y0, pts=pts, rim_depth=d, rim_cov=cov, r=float(rad.mean())))

# ---------- 5. rasterise with overlap resolution (nearest normalised centre) ----------
lab = np.zeros((H, W), np.int32)
claim = np.full((H, W), np.inf, np.float32)
for j, o in enumerate(objs, 1):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [o['pts'].round().astype(np.int32)], 1)
    m = (m > 0) & roi
    ys, xs = np.nonzero(m)
    if len(ys) == 0: continue
    dist = np.hypot(xs - o['x'], ys - o['y']) / o['r']      # normalised radius
    better = dist < claim[ys, xs]
    claim[ys[better], xs[better]] = dist[better]
    lab[ys[better], xs[better]] = j

# drop objects largely swallowed by neighbours, then relabel
counts = np.bincount(lab.ravel(), minlength=len(objs)+1)
keep_ids = [j for j in range(1, len(objs)+1) if counts[j] >= AREA_MIN]
remap = np.zeros(len(objs)+1, np.int32)
for new, old in enumerate(keep_ids, 1): remap[old] = new
lab = remap[lab]

# ---------- 6. morphometry from the final masks ----------
recs = []
for new, old in enumerate(keep_ids, 1):
    m = (lab == new).astype(np.uint8)
    area = int(m.sum())
    cnt = max(cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    per = float(cv2.arcLength(cnt, True))
    (cx, cy), (MA, ma), _ = cv2.fitEllipse(cnt) if len(cnt) >= 5 else ((0,0),(0,0),0)
    hull = cv2.convexHull(cnt)
    o = objs[old-1]
    r = dict(id=new, centroid_x=round(float(cx),1), centroid_y=round(float(cy),1),
             area_px2=area, eq_diameter_px=round(2*np.sqrt(area/np.pi),1),
             perimeter_px=round(per,1),
             circularity=round(min(1.0, 4*np.pi*area/per**2),3),
             solidity=round(area/max(cv2.contourArea(hull),1),3),
             minor_axis_px=round(min(MA,ma),1), major_axis_px=round(max(MA,ma),1),
             aspect_ratio=round(max(MA,ma)/max(min(MA,ma),1e-6),2),
             rim_depth_gray=round(o['rim_depth'],1), rim_coverage=round(o['rim_cov'],2))
    if UM_PER_PX:
        r['area_um2'] = round(area*UM_PER_PX**2, 1)
        r['eq_diameter_um'] = round(r['eq_diameter_px']*UM_PER_PX, 1)
    recs.append(r)

recs.sort(key=lambda r: -r['area_px2'])
A = np.array([r['area_px2'] for r in recs], float)
D = np.array([r['eq_diameter_px'] for r in recs], float)

sd = lambda v: round(float(v.std(ddof=1)), 1) if len(v) > 1 else 0.0
summary = dict(image=IMG, image_size=[W, H], roi_area_px2=int(roi.sum()),
               n_candidates=int(len(C)), n_after_nms=int(n_nms), n_organoids=len(recs),
               total_area_px2=float(A.sum()),
               pct_field_covered=round(100*A.sum()/max(roi.sum(), 1), 2),
               area_px2=dict(min=float(A.min()), q1=float(np.percentile(A,25)),
                             median=float(np.median(A)), q3=float(np.percentile(A,75)),
                             max=float(A.max()), mean=round(float(A.mean()),1),
                             sd=sd(A)) if len(A) else None,
               eq_diameter_px=dict(min=float(D.min()), median=float(np.median(D)),
                                   mean=round(float(D.mean()),1), max=float(D.max()),
                                   sd=sd(D)) if len(D) else None,
               um_per_px=UM_PER_PX)
if UM_PER_PX:
    summary['total_area_um2'] = round(float(A.sum())*UM_PER_PX**2, 1)
summary['roi_mode'] = roi_mode
if well is not None:
    summary['well'] = dict(cx=round(well['cx'], 1), cy=round(well['cy'], 1), r_floor_px=round(well['r'], 1),
                           r_wall_px=round(well['r_wall'], 1), used_as_roi=roi_mode == 'well')
if small_depth is not None:
    summary['small_organoids'] = dict(noise_grey=round(noise, 2), rim_depth_min=round(small_depth, 1),
                                      max_radius_px=SMALL_R_MAX)
if args.well_diameter_mm:
    summary['scale_source'] = 'well floor diameter %.3f mm = %.1f px' % (args.well_diameter_mm, 2 * well['r'])

COLS = ['id', 'centroid_x', 'centroid_y', 'area_px2', 'eq_diameter_px', 'perimeter_px', 'circularity',
        'solidity', 'minor_axis_px', 'major_axis_px', 'aspect_ratio', 'rim_depth_gray', 'rim_coverage'] + \
       (['area_um2', 'eq_diameter_um'] if UM_PER_PX else [])
with open(OUT+'organoids.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(recs)
with open(OUT+'summary.json','w') as f: json.dump(summary, f, indent=2)
def imwrite(path, img):                     # cv2.imwrite cannot handle accented paths on Windows
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    buf.tofile(path)
imwrite(OUT+'mask_labels.png', lab.astype(np.uint16))
imwrite(OUT+'mask_binary.png', ((lab>0)*255).astype(np.uint8))

# ---------- 7. overlays ----------
base = cv2.cvtColor(cv2.createCLAHE(3.0,(16,16)).apply(g), cv2.COLOR_GRAY2BGR)
rng = np.random.default_rng(0)
colmap = np.zeros((H,W,3), np.uint8)
for i in range(1, lab.max()+1): colmap[lab==i] = rng.integers(70,255,3)
ov = cv2.addWeighted(base, 1.0, colmap, 0.40, 0)
for i in range(1, lab.max()+1):
    c,_ = cv2.findContours((lab==i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, c, -1, (0,0,255), 2)
if roi_mode == 'well':                       # show which region was analysed
    cv2.circle(ov, (int(round(well['cx'])), int(round(well['cy']))), int(round(well['r'])), (0,220,255), 3)
imwrite(OUT+'overlay_mask.png', ov)

ann = base.copy()
for r in recs:
    c,_ = cv2.findContours((lab==r['id']).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ann, c, -1, (0,0,255), 2)
    cv2.putText(ann, str(r['id']), (int(r['centroid_x'])-8, int(r['centroid_y'])+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,30,0), 1, cv2.LINE_AA)
if roi_mode == 'well':
    cv2.circle(ann, (int(round(well['cx'])), int(round(well['cy']))), int(round(well['r'])), (0,220,255), 3)
imwrite(OUT+'overlay_numbered.png', ann)

print(json.dumps(summary, indent=2))
