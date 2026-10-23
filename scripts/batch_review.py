#!/usr/bin/env python3
"""Analyse a folder of well images, then review and correct them all in the browser.

Usage:
    python batch_review.py <images_folder> [options]

    1. Runs the pipeline on every image in the folder (images already analysed are skipped),
       writing one results folder per image into "<images_folder> results".
    2. Starts a local server (this computer only) and opens organoid_editor.html with every
       image and its mask loaded. Edits are saved straight back into the results folder;
       the pipeline's own output is kept in <image>/pipeline_original/ the first time.
    3. "Export" in the editor writes, into the results folder:
         organoids_all.csv      one row per organoid, all images
         images_summary.csv     one row per image
         organoid_results.xlsx  both tables as Excel sheets, plus the settings used

Options:
    --out DIR               results folder (default: "<images_folder> results")
    --pipeline NAME         opencv (default) or skimage
    --well-diameter-mm D    calibrate to micrometres from the detected well floor
    --roi MODE              auto (default), well or field
    --workers N             images analysed in parallel (default: up to 4)
    --force                 re-analyse images that already have results; their previous results
                            (including hand edits) are moved to _previous_runs/<date_time>/ first
    --no-analyse            only open the review (no pipeline runs)
    --port N                server port (default: a free one)
    --no-browser            don't open the browser automatically

Stop the server with Ctrl+C. Nothing is lost by stopping it: edits are saved as you go.
"""
import argparse, base64, concurrent.futures as cf, csv, datetime, io, json, os, re, shutil, socket
import subprocess, sys, threading, time, webbrowser, zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, quote
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
PIPELINES = {"opencv": "organoid_quant.py", "skimage": "organoid_quant_skimage.py"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
RESULT_FILES = ["mask_labels.png", "mask_binary.png", "organoids.csv", "summary.json",
                "overlay_mask.png", "overlay_numbered.png"]
SAVE_FILES = {"mask_labels.png", "mask_binary.png", "organoids.csv", "summary.json", "edits.json",
              "overlay_edited.png"}
EXPORTS = ["organoids_all.csv", "images_summary.csv", "organoid_results.xlsx"]


# ============================================================ analysis
def find_images(folder):
    imgs = sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT
                  and not p.name.startswith("."))
    keys, out = set(), []
    for p in imgs:                     # results folder per image; disambiguate a.jpg / a.png
        k = p.stem if p.stem not in keys else p.stem + "_" + p.suffix[1:]
        keys.add(k)
        out.append((k, p))
    return out


def analyse(images, out, args):
    todo = [(k, p) for k, p in images if args.force or not (out / k / "summary.json").exists()]
    if not todo:
        print("All %d images already analysed (use --force to redo)." % len(images))
        return
    script = HERE / PIPELINES[args.pipeline]
    extra = ["--roi", args.roi] + (["--well-diameter-mm", str(args.well_diameter_mm)] if args.well_diameter_mm else [])
    print("Analysing %d image(s) with the %s pipeline, %d at a time..." % (len(todo), args.pipeline, args.workers))

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    backed_up = []

    def run(item):
        k, p = item
        d = out / k
        if args.force and d.exists() and any(d.iterdir()):
            # never overwrite earlier results (and hand edits): move them aside first
            dest = out / "_previous_runs" / stamp / k
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(d), str(dest))
            backed_up.append(k)
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(script), str(p), str(d)] + extra,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        (d / "pipeline_log.txt").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
        return k, r.returncode, time.time() - t0, r.stderr.strip().splitlines()[-1:] if r.returncode else []

    done = 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for k, code, dt, err in ex.map(run, todo):
            done += 1
            if code == 0:
                n = json.loads((out / k / "summary.json").read_text(encoding="utf-8"))["n_organoids"]
                print("  [%d/%d] %-40s %4d organoids  (%.0f s)" % (done, len(todo), k, n, dt))
            else:
                print("  [%d/%d] %-40s FAILED: %s" % (done, len(todo), k, err[0] if err else "see pipeline_log.txt"))
    if backed_up:
        # re-analysed images have new masks: they need reviewing again
        sp = out / "review_state.json"
        try:
            state = json.loads(sp.read_text(encoding="utf-8"))
            for k in backed_up:
                state.pop(k, None)
            sp.write_text(json.dumps(state, indent=1), encoding="utf-8")
        except Exception:
            pass
        print("Previous results (including any hand edits) were moved to %s" % (out / "_previous_runs" / stamp))


# ============================================================ session state
class Session:
    def __init__(self, images, out, args):
        self.images = dict(images)          # key -> image path
        self.order = [k for k, _ in images]
        self.out = out
        self.args = args
        self.lock = threading.Lock()
        self.state_path = out / "review_state.json"
        try:
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            self.state = {}

    def save_state(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def info(self, k):
        d = self.out / k
        e = {"key": k, "name": self.images[k].name, "reviewed": bool(self.state.get(k, {}).get("reviewed")),
             "edited": (d / "pipeline_original").exists(), "saved": self.state.get(k, {}).get("saved")}
        try:
            s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            e["n_organoids"] = s.get("n_organoids")
        except Exception:
            log = d / "pipeline_log.txt"
            e["error"] = (log.read_text(encoding="utf-8", errors="replace").strip().splitlines() or ["not analysed"])[-1] \
                if log.exists() else "not analysed"
        return e

    def listing(self):
        return {"results_dir": str(self.out), "pipeline": self.args.pipeline,
                "images": [self.info(k) for k in self.order]}

    def save(self, k, payload):
        d = self.out / k
        d.mkdir(parents=True, exist_ok=True)
        with self.lock:
            orig = d / "pipeline_original"
            if not orig.exists() and (d / "summary.json").exists():
                orig.mkdir()
                for f in RESULT_FILES + ["edits.json"]:
                    if (d / f).exists():
                        shutil.copy2(d / f, orig / f)
            for name, b64 in payload.get("files", {}).items():
                if name not in SAVE_FILES:
                    raise ValueError("refusing to write " + name)
                data = base64.b64decode(b64)
                if name == "edits.json":
                    data = self._merge_edits(d / name, json.loads(data), payload.get("session_id", ""))
                tmp = d / (name + ".tmp")
                tmp.write_bytes(data)
                os.replace(tmp, d / name)
            st = self.state.setdefault(k, {})
            st["saved"] = datetime.datetime.now().isoformat(timespec="seconds")
            if "reviewed" in payload:
                st["reviewed"] = bool(payload["reviewed"])
            self.save_state()
        return self.info(k)

    @staticmethod
    def _merge_edits(path, new, session_id):
        """Keep every editing session's operations; a re-save of the same session replaces it."""
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            old = {}
        hist = [h for h in old.get("sessions", []) if h.get("session_id") != session_id]
        hist.append({"session_id": session_id, "saved": datetime.datetime.now().isoformat(timespec="seconds"),
                     "operations": new.get("operations", [])})
        merged = {"image": new.get("image"), "sessions": hist, "id_map": new.get("id_map")}
        return json.dumps(merged, indent=1).encode("utf-8")

    def set_reviewed(self, k, value):
        with self.lock:
            self.state.setdefault(k, {})["reviewed"] = bool(value)
            self.save_state()
        return self.info(k)

    # -------------------------------------------------------- export
    def export(self):
        organoid_rows, image_rows, cols = [], [], []
        for k in self.order:
            d = self.out / k
            try:
                summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
                with open(d / "organoids.csv", newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
            except Exception:
                image_rows.append({"image": self.images[k].name, "error": self.info(k).get("error", "no results")})
                continue
            tags = name_tags(self.images[k].stem)
            reviewed = bool(self.state.get(k, {}).get("reviewed"))
            for r in rows:
                if not r.get("status"):                  # never edited: pipeline output as is
                    r.update(status="original", source_id=r.get("id", ""), note="")
                for c in r:
                    if c not in cols:
                        cols.append(c)
                organoid_rows.append({"image": self.images[k].name, **tags, **r, "reviewed": reviewed})
            ed = summ.get("edited") or {}
            st = Counter(r.get("status", "original") for r in rows)
            area = [num(r["area_px2"]) for r in rows]
            diam = [num(r["eq_diameter_px"]) for r in rows]
            um = summ.get("um_per_px")
            well = summ.get("well") or {}
            image_rows.append({
                "image": self.images[k].name, **tags, "reviewed": reviewed,
                "edited": (d / "pipeline_original").exists(),
                "n_organoids": len(rows), "total_area_px2": sum(area),
                "mean_area_px2": round(sum(area) / len(area), 1) if area else "",
                "median_area_px2": median(area), "median_eq_diameter_px": median(diam),
                "pct_area_covered": summ.get("pct_field_covered", ""),
                "um_per_px": um or "",
                "total_area_um2": round(sum(area) * um * um, 1) if um else "",
                "median_eq_diameter_um": round(median(diam) * um, 1) if um and diam else "",
                "n_unchanged": st.get("original", 0), "n_modified": st.get("modified", 0),
                "n_added": st.get("added", 0), "n_deleted": ed.get("n_deleted", 0),
                "n_from_pipeline": ed.get("source_n_organoids", len(rows)),
                "roi_mode": summ.get("roi_mode", "field"), "well_radius_px": well.get("r_floor_px", ""),
                "pipeline": summ.get("method") or ("scikit-image" if "skimage" in self.args.pipeline else "opencv"),
            })
        lead = ["image"] + [t for t in TAG_NAMES if any(t in r for r in organoid_rows)]
        ocols = lead + [c for c in cols if c not in lead] + ["reviewed"]
        icols = []
        for r in sorted(image_rows, key=lambda r: "error" in r):     # images with results define the order
            for c in r:
                if c not in icols and c != "error":
                    icols.append(c)
        if any("error" in r for r in image_rows):
            icols.append("error")
        write_csv(self.out / "organoids_all.csv", ocols, organoid_rows)
        write_csv(self.out / "images_summary.csv", icols, image_rows)
        about = [{"item": k2, "value": v} for k2, v in [
            ("exported", datetime.datetime.now().isoformat(timespec="seconds")),
            ("images folder", str(Path(self.args.images).resolve())), ("results folder", str(self.out)),
            ("pipeline", self.args.pipeline), ("ROI mode option", self.args.roi),
            ("well diameter (mm)", self.args.well_diameter_mm or "not given (areas in pixels)"),
            ("images", len(self.order)), ("reviewed", sum(1 for r in image_rows if r.get("reviewed"))),
            ("organoids", len(organoid_rows)),
            ("status column", "original = pipeline result kept; modified = edited by hand; added = drawn by hand"),
            ("source_id column", "the organoid's ID in the pipeline's own output (empty for added ones)"),
            ("columns sample / SI / J", "read from the file name (C<number>, SI<number>, J<number>) when present")]]
        write_xlsx(self.out / "organoid_results.xlsx", [
            ("Organoids", ocols, organoid_rows), ("Per image", icols, image_rows),
            ("About", ["item", "value"], about)])
        return {"files": [{"name": n, "url": "exports/" + quote(n), "path": str(self.out / n)} for n in EXPORTS],
                "n_organoids": len(organoid_rows), "n_images": len(image_rows)}


TAG_NAMES = ["sample", "SI", "J"]


def name_tags(stem):
    """Pull C<number>, SI<number>, J<number> out of names like 'C55269_J7_SI 2' or 'C59481-SI1-J9'."""
    out = {}
    for key, pat in [("sample", r"(?i)\b(C\d+)"), ("SI", r"(?i)SI\s*(\d+)"), ("J", r"(?i)(?:^|[^A-Za-z])J\s*(\d+)")]:
        m = re.search(pat, stem)
        if m:
            out[key] = m.group(1).upper() if key == "sample" else int(m.group(1))
    return out


class Counter(dict):
    def __init__(self, it):
        super().__init__()
        for x in it:
            self[x] = self.get(x, 0) + 1


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def median(v):
    if not v:
        return ""
    s = sorted(v); n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def write_csv(path, cols, rows):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:      # BOM: Excel reads accents correctly
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


# ============================================================ minimal .xlsx writer (no dependencies)
def _col(i):
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _cell(ref, v, style=0):
    st = ' s="%d"' % style if style else ""
    if isinstance(v, bool):
        return '<c r="%s" t="b"%s><v>%d</v></c>' % (ref, st, v)
    if isinstance(v, (int, float)) and v == v:
        return '<c r="%s"%s><v>%s</v></c>' % (ref, st, repr(float(v)) if isinstance(v, float) else v)
    if isinstance(v, str):
        try:                                            # numbers stored as text in the CSVs
            if v.strip() and re.fullmatch(r"-?\d+(\.\d+)?([eE]-?\d+)?", v.strip()):
                return '<c r="%s"%s><v>%s</v></c>' % (ref, st, v.strip())
        except Exception:
            pass
    if v is None or v == "":
        return ""
    t = escape(str(v)).replace("\n", "&#10;")
    return '<c r="%s" t="inlineStr"%s><is><t xml:space="preserve">%s</t></is></c>' % (ref, st, t)


def write_xlsx(path, sheets):
    def sheet_xml(cols, rows):
        widths = [max([len(str(c))] + [len(str(r.get(c, ""))) for r in rows[:500]]) for c in cols]
        out = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
               '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
               '</sheetView></sheetViews><cols>']
        out += ['<col min="%d" max="%d" width="%.1f" customWidth="1"/>' % (i + 1, i + 1, min(60, max(8, w + 2)))
                for i, w in enumerate(widths)]
        out.append("</cols><sheetData>")
        out.append('<row r="1">' + "".join(_cell(_col(i) + "1", c, 1) for i, c in enumerate(cols)) + "</row>")
        for n, r in enumerate(rows, 2):
            out.append('<row r="%d">' % n + "".join(_cell(_col(i) + str(n), r.get(c, "")) for i, c in enumerate(cols)) + "</row>")
        out.append('</sheetData><autoFilter ref="A1:%s%d"/></worksheet>' % (_col(max(len(cols) - 1, 0)), max(len(rows) + 1, 1)))
        return "".join(out)

    names = [re.sub(r"[\[\]:*?/\\]", "_", s[0])[:31] for s in sheets]
    ct = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
          '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>']
    ct += ['<Override PartName="/xl/worksheets/sheet%d.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' % (i + 1)
           for i in range(len(sheets))]
    ct.append("</Types>")
    wb = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
          + "".join('<sheet name="%s" sheetId="%d" r:id="rId%d"/>' % (escape(n), i + 1, i + 1) for i, n in enumerate(names))
          + "</sheets></workbook>")
    wbrels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
              + "".join('<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet%d.xml"/>' % (i + 1, i + 1)
                        for i in range(len(sheets)))
              + '<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>' % (len(sheets) + 1)
              + "</Relationships>")
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>")
    styles = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
              '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
              '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
              '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
              '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
              '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
              '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>'
              '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>')
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "".join(ct))
        z.writestr("_rels/.rels", rels)
        z.writestr("xl/workbook.xml", wb)
        z.writestr("xl/_rels/workbook.xml.rels", wbrels)
        z.writestr("xl/styles.xml", styles)
        for i, (_, cols, rows) in enumerate(sheets):
            z.writestr("xl/worksheets/sheet%d.xml" % (i + 1), sheet_xml(cols, rows))
    os.replace(tmp, path)


# ============================================================ HTTP server (127.0.0.1 only)
CTYPES = {".html": "text/html; charset=utf-8", ".json": "application/json", ".csv": "text/csv; charset=utf-8",
          ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".tif": "image/tiff",
          ".tiff": "image/tiff", ".bmp": "image/bmp",
          ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}


def make_handler(session):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype="application/json", extra=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode("utf-8")
            elif isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def file(self, p, download=False):
            if not p.is_file():
                return self.send(404, {"error": "not found"})
            extra = {"Content-Disposition": 'attachment; filename="%s"' % p.name} if download else None
            self.send(200, p.read_bytes(), CTYPES.get(p.suffix.lower(), "application/octet-stream"), extra)

        def parts(self):
            return [unquote(x) for x in self.path.split("?")[0].strip("/").split("/")]

        def do_GET(self):
            p = self.parts()
            try:
                if p in ([""], ["index.html"]):
                    return self.file(HERE / "organoid_editor.html")
                if p == ["api", "session"]:
                    return self.send(200, session.listing())
                if len(p) == 3 and p[0] == "files" and p[1] in session.images:
                    if p[2] == "image":
                        return self.file(session.images[p[1]])
                    if p[2] in ("mask_labels.png", "organoids.csv", "summary.json"):
                        return self.file(session.out / p[1] / p[2])
                if len(p) == 2 and p[0] == "exports" and p[1] in EXPORTS:
                    return self.file(session.out / p[1], download=True)
                self.send(404, {"error": "not found"})
            except Exception as e:
                self.send(500, {"error": str(e)})

        def do_POST(self):
            p = self.parts()
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if len(p) == 3 and p[:2] == ["api", "save"] and p[2] in session.images:
                    return self.send(200, session.save(p[2], body))
                if len(p) == 3 and p[:2] == ["api", "reviewed"] and p[2] in session.images:
                    return self.send(200, session.set_reviewed(p[2], body.get("reviewed")))
                if p == ["api", "log"]:              # diagnostics from the editor (display problems)
                    with open(session.out / "_editor_log.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(dict(body, server_time=datetime.datetime.now().isoformat(timespec="seconds"))) + chr(10))
                    return self.send(200, {"ok": True})
                if p == ["api", "export"]:
                    return self.send(200, session.export())
                self.send(404, {"error": "not found"})
            except Exception as e:
                self.send(500, {"error": str(e)})
    return H


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("images", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--pipeline", choices=sorted(PIPELINES), default="opencv")
    ap.add_argument("--well-diameter-mm", type=float)
    ap.add_argument("--roi", choices=["auto", "well", "field"], default="auto")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-analyse", action="store_true")
    ap.add_argument("--port", type=int)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    # Asking here rather than in the .bat: Windows batch mishandles dragged paths
    # (they arrive quoted, and quotes inside `set` break the script).
    def ask(prompt, default=""):
        try:
            v = input(prompt).strip().strip('"').strip("'").strip()
        except EOFError:
            v = ""
        return v or default

    interactive = args.images is None
    if interactive:
        print("\n  ORGANOID ANALYSIS\n  -----------------\n")
        print("  Which folder holds the photos?")
        print("  Drag the folder into this window and press Enter,")
        print('  or just press Enter to use "images test".\n')
        while True:
            args.images = ask("  Folder: ", "images test")
            if Path(args.images).is_dir():
                break
            print("  No such folder: %s\n" % args.images)
        if not args.well_diameter_mm:
            print("\n  Well diameter in mm (optional): the inner diameter of the well floor.")
            print("  Enter it to also get results in micrometres; press Enter to keep pixels only.\n")
            mm = ask("  Well floor diameter in mm: ")
            if mm:
                try:
                    args.well_diameter_mm = float(mm.replace(",", "."))
                except ValueError:
                    print("  Not a number; keeping pixels only.")

    src = Path(args.images)
    if not src.is_dir():
        sys.exit("not a folder: %s" % src)

    if args.force and interactive:
        print('\n  Every photo in "%s" will be analysed again from scratch.' % src)
        print("  Current results and hand corrections are moved to _previous_runs\\ first,")
        print("  and every photo will need reviewing again.\n")
        if ask("  Type YES to continue: ").upper() != "YES":
            sys.exit("\n  Cancelled. Nothing was changed.")
    out = Path(args.out) if args.out else src.parent / (src.name + " results")
    out.mkdir(parents=True, exist_ok=True)
    images = find_images(src)
    if not images:
        sys.exit("no images (.jpg .png .tif .bmp) in %s" % src)
    (out / "session.json").write_text(json.dumps({
        "images_folder": str(src.resolve()), "pipeline": args.pipeline, "roi": args.roi,
        "well_diameter_mm": args.well_diameter_mm, "updated": datetime.datetime.now().isoformat(timespec="seconds")},
        indent=1), encoding="utf-8")
    if not args.no_analyse:
        analyse(images, out, args)

    session = Session(images, out, args)
    port = args.port or free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(session))
    url = "http://127.0.0.1:%d/" % port
    print("\nReview %d image(s) at %s" % (len(images), url))
    print("Results folder: %s" % out)
    print("Press Ctrl+C here to stop (your edits are already saved).")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
