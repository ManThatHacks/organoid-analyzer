# Organoid Analyzer

Counts and measures organoids in brightfield well photos, then lets you correct
the result by hand and export it to Excel.

Everything runs on this computer. No installation, no internet, nothing is
uploaded: Python and all the packages live in the `_runtime` folder.

---

## 1. Everyday use

1. **Put your photos in one folder.** JPG, PNG or TIFF; one well per photo.
2. **Drag that folder onto `Analyse organoids.bat`.**
   (Or double-click it and drag the folder into the black window when it asks.)
3. **Well diameter in mm** — type it to also get results in micrometres, or
   press Enter for pixels only. Use the same number every time, otherwise
   results from different photos cannot be compared.
4. **Wait.** Roughly 20–40 s per photo, four at a time.
5. **The review page opens in your browser.** Correct the masks there:
   `Ctrl+Enter` saves and moves to the next photo, `?` lists all shortcuts.
6. **Click "Export Excel / CSV."**
7. **Close the black window** when you are done. Edits are saved as you go.

`Check installation.bat` tells you whether everything is in place.
Right-click `Analyse organoids.bat` → *Send to* → *Desktop* for a shortcut.

### What you get

Results go into a new folder named `<your folder> results`, next to your
photos:

| File | Contents |
|---|---|
| `organoid_results.xlsx` | **Main output.** Sheets: *Organoids* (one row per organoid, all photos), *Per image* (counts, areas, edits), *About* (settings used) |
| `organoids_all.csv`, `images_summary.csv` | The same two tables as CSV, for R or Python |
| `<photo>\organoids.csv` | One photo's organoids: area, diameter, perimeter, circularity, solidity, axes, centroid |
| `<photo>\mask_labels.png` | 16-bit label mask (opens in Fiji, CellProfiler) |
| `<photo>\overlay_numbered.png` | Outlines with ID numbers |
| `<photo>\pipeline_original\` | The script's own output, kept the first time you save a correction |
| `<photo>\edits.json` | Every correction you made, in order |

In the tables, `status` says whether an organoid came from the script
(`original`), was reshaped by you (`modified`) or drawn by you (`added`).
`sample`, `SI` and `J` columns are read from file names such as
`C55269_J7_SI 2` or `C59481-SI1-J9`.

### Clearing out false positives quickly

In sparse or grainy wells the script detects specks, dust and out-of-focus
smudges alongside the real organoids, and nearly all of them are small. The
**Size & quality filter** in the review page removes them in a few seconds,
without re-analysing anything:

- **Hide smaller than** — hides every object below a diameter (shown in µm
  when a well diameter was given, otherwise in pixels).
- **Hide rim fainter than** — hides objects whose outline is washed out. The
  number is how many grey levels darker the rim is than its surroundings, so
  it separates a real organoid edge from a faint smudge.

Both act instantly as a preview: the count updates (`7 shown · 12 hidden`) and
the hidden objects disappear from the image, so you can watch where the
threshold should sit. Nothing is changed until you press **Delete hidden**,
which removes them in a single step that `Ctrl+Z` undoes. **Reset** brings
everything back.

Objects you drew by hand are never hidden.

**Keep setting for next images** (the tick box under the sliders) decides what
happens when you open the next photo:

- **off** (default) — the sliders reset to zero for every photo, so each well
  is judged on its own. Best when wells differ a lot, since a dense well needs
  a much lower threshold than a near-empty one.
- **on** — the sliders stay where you put them, so you can set a threshold once
  and work down the list. The tick box itself is remembered the next time you
  open the review page.

Why this is a filter rather than a stricter detector: in dense wells the small
detections are mostly real. In the two wells reviewed so far, the objects
deleted by hand were mostly *large* ones that had been merged or split wrongly,
and almost every small object was kept. Tightening the detector itself would
therefore lose real organoids in good wells to fix a problem that only shows up
in near-empty ones.

### Running the same photos again

A normal run **skips photos that already have results** and prints
`All N images already analysed`. This is deliberate: adding new photos to a
folder never redoes the old ones or discards your corrections. The review page
still opens with the existing results.

To redo them from scratch, use **`Re-analyse all photos.bat`**. It asks you to
type `YES`, then moves the current results *and your corrections* into
`<your folder> results\_previous_runs\<date_time>\`, analyses everything again,
and clears the "reviewed" ticks.

---

## 2. Moving this folder to another computer

Copy the **whole folder**, including `_runtime` (about 480 MB). It is
self-contained and does not care where it sits, so a USB stick or a network
drive is fine. Windows 64-bit only.

If `Check installation.bat` ever reports broken packages, rebuild the runtime
with [uv](https://docs.astral.sh/uv/):

```powershell
uv python install 3.12
uv venv --python 3.12 _runtime\env
uv pip install --python _runtime\env\Scripts\python.exe -r scripts\requirements.txt
```

then change `set "PY=..."` in the three `.bat` files to
`%~dp0_runtime\env\Scripts\python.exe`.

---

## 3. What is in this folder

```
Analyse organoids.bat        run the analysis + review
Re-analyse all photos.bat    redo photos that already have results
Check installation.bat       self-test
README.md                    this file
scripts\                     the actual programs (see below)
fiji\                        two optional ImageJ/Fiji macros
reference\control_well\      reference result used to validate the pipeline
_runtime\                    Python and all packages (do not rename or delete)
```

| Script | What it does |
|---|---|
| `batch_review.py` | Runs the analysis over a folder, serves the review page, exports Excel/CSV |
| `organoid_quant.py` | The analysis itself (OpenCV). This is what the launchers use |
| `well_roi.py` | Finds the well floor in a whole-well photo |
| `organoid_editor.html` | The review and correction interface |
| `organoid_quant_skimage.py` | A second, independent implementation (scikit-image) for cross-checking |
| `compare_pipelines.py` | Compares two result folders object by object |
| `plot_distribution.py` | Size-distribution figure from one `organoids.csv` |
| `verify.py` | Checks a control-well run against the published reference |

---

## 4. How the analysis works

1. **Region of interest.** For a photo showing the whole well and the plate
   around it, the well floor is detected and only its inside is analysed;
   otherwise the plate's rings, walls and moulded numbers get counted as
   organoids. For images where everything outside the well is already black
   (such as the control image), the lit field is used instead. `--roi auto`
   decides; `--roi well` or `--roi field` force one.
2. **Detection.** Candidate circles are found at seven size bands, plus an
   extra pass on a 2× enlarged copy of the well for small organoids.
3. **Verification.** A candidate is kept only if it has a dark rim that is
   deep enough and continuous enough. Small objects must clear 6 grey levels,
   or 3.5× the photo's own noise if that is higher.
4. **Outlines.** The rim is traced at 180 angles, so masks follow the real
   shape rather than a circle. On side-lit photos, where part of the rim
   disappears, the outline is held within ±15 % of the organoid's typical
   radius.
5. **Touching organoids** are separated by assigning contested pixels to the
   nearer centre.
6. **Measurement** is done on the final masks.

### Calibration

Without `--well-diameter-mm` everything is in pixels. The zoom varies between
hand-held photos — across 11 test photos the well radius ranged from 879 to
1068 px, a 26 % spread — so **pixel areas are not comparable between photos**
until you give the well diameter.

### Known limits

- **The faintest small organoids are still missed** (rims under ~4 grey
  levels). Nothing distinguishes them from background texture reliably; that
  is what the editor is for.
- **In near-empty, grainy wells** a fair share of small detections are specks
  rather than organoids. Check those wells in the editor.
- **Small-organoid counts are the least stable number** the script produces.
  For comparing conditions, prefer total organoid area, or counts above a size
  limit where detection is reliable.
- **Image quality drives the results.** A microscope camera, a fixed
  objective, and the same zoom and focus every time would help more than any
  further tuning of the script.

### Cross-checking

`organoid_quant_skimage.py` segments the same image with a completely
different method, and `compare_pipelines.py` pairs the two results object by
object. On the control image the two agree closely (median overlap 0.87, areas
correlated r = 0.96). On whole-well photos the scikit-image version
under-detects and should not be trusted without re-tuning.

### Fiji macros (optional)

- `fiji\Fiji_A_measure_label_mask.ijm` loads a `mask_labels.png` into Fiji's
  ROI Manager, so you can inspect and re-measure the pipeline's own objects.
- `fiji\Fiji_B_segment_organoids.ijm` segments from scratch using only
  built-in Fiji commands. It is an approximation and will not reproduce the
  Python result.

### Reproducibility

There is no randomness in the pipeline. The same input gives byte-identical
output, and the reference control well (125 organoids, 157,708 px²) is
reproduced exactly by this runtime — `reference\control_well\` holds that
result, and `verify.py` checks a fresh run against it.
