/*  B — Pure-Fiji organoid segmentation (no plugins required).
 *
 *  Open your well image, then run this macro.
 *
 *  This is an APPROXIMATION of the Python pipeline, not a port. Fiji's macro
 *  language has no multi-scale Hough detector and no polar rim refinement, so
 *  this uses the classic edge -> close -> fill -> watershed route instead.
 *  Expect a somewhat different count; EDGE_PCTL is the knob that matters.
 *
 *  Why edges and not a plain threshold: in-focus contrast in these images is
 *  only ~8 grey levels, so intensity thresholding is unstable. The dark rim
 *  around each organoid is the reliable signal.
 */

// ------------------------------------------------------------------ settings
EDGE_PCTL  = 92;     // edge-strength percentile -> binary threshold.
                     //   LOWER  (88-90) = more objects, more texture noise
                     //   HIGHER (94-96) = fewer objects, only strong rims
BLUR_SIGMA = 2;      // denoise before edge detection
CLOSE_ITER = 4;      // iterations to seal gaps in each rim
SHRINK_ROI = 20;     // px to pull back from the illuminated field rim.
                     //   Must exceed the rim blur, or Fill Holes floods the
                     //   whole field. Raise it if that happens.
MIN_AREA   = 60;     // px^2
MIN_CIRC   = 0.55;   // rejects irregular texture blobs
PIXEL_SIZE = 0;      // micrometres per pixel. 0 = report in pixels
UNIT       = "um";
SAVE_CSV   = true;

// --------------------------------------------------------------------- setup
if (nImages == 0) exit("Open your well image first, then run this macro.");
orig = getTitle();
dir  = getDirectory("image");
if (dir == "") dir = getDirectory("home");

run("Clear Results");
roiManager("reset");
run("Select None");
setOption("BlackBackground", true);
setBackgroundColor(0, 0, 0);

selectWindow(orig);
if (PIXEL_SIZE > 0)
    run("Set Scale...", "distance=1 known=" + PIXEL_SIZE + " unit=" + UNIT);
else
    run("Set Scale...", "distance=0 known=0 unit=pixel");

// ------------------------------------- 1. ROI = illuminated field, pulled in
selectWindow(orig);
run("Duplicate...", "title=field");
run("8-bit");
setThreshold(30, 255);
run("Convert to Mask");
run("Fill Holes");
run("Create Selection");
run("Enlarge...", "enlarge=-" + SHRINK_ROI + " pixel");
roiManager("Add");                 // ROI index 0 = field of view
roiManager("Select", 0);
roiManager("Rename", "field");
close();                           // close "field"

// -------------------------------------------- 2. rim (gradient) magnitude
selectWindow(orig);
run("Duplicate...", "title=work");
run("8-bit");
run("Gaussian Blur...", "sigma=" + BLUR_SIGMA);
run("Find Edges");                 // 3x3 Sobel magnitude

selectWindow("work");
roiManager("Select", 0);
run("Clear Outside");              // kill the bright field-rim edge

// ------------------------------- 3. percentile threshold inside the field
roiManager("Select", 0);
nBins = 256;
getHistogram(vals, counts, nBins);
total = 0;
for (i = 0; i < nBins; i++) total += counts[i];
target = total * EDGE_PCTL / 100;
cum = 0; thr = 255;
for (i = 0; i < nBins; i++) {
    cum += counts[i];
    if (cum >= target) { thr = i; i = nBins; }
}
print("\\Clear");
print("Edge threshold at p" + EDGE_PCTL + " = " + thr + " / 255");

// ------------------------------------ 4. binarise, seal rims, fill, split
selectWindow("work");
run("Select None");
setThreshold(thr, 255);
run("Convert to Mask");

roiManager("Select", 0);
run("Clear Outside");
run("Select None");

run("Options...", "iterations=" + CLOSE_ITER + " count=1 black do=Close");
run("Fill Holes");
run("Watershed");                  // splits touching organoids

// ------------------------------------------------- 5. measure and report
run("Set Measurements...",
    "area centroid perimeter shape feret's redirect=None decimal=3");
roiManager("Select", 0);
run("Analyze Particles...",
    "size=" + MIN_AREA + "-Infinity pixel circularity=" + MIN_CIRC +
    "-1.00 display clear add");

// drop the helper "field" ROI so the manager holds only organoids
if (roiManager("count") > 0) {
    roiManager("Select", 0);
    if (Roi.getName == "field") roiManager("Delete");
}
run("Select None");

totalArea = 0;
for (r = 0; r < nResults; r++) totalArea += getResult("Area", r);
u = "px^2";
if (PIXEL_SIZE > 0) u = UNIT + "^2";

print("Organoids detected : " + nResults);
print("Total surface      : " + d2s(totalArea, 1) + " " + u);
if (nResults > 0)
    print("Mean surface       : " + d2s(totalArea / nResults, 1) + " " + u);

// ------------------------------------------------------------- 6. overlay
selectWindow(orig);
run("Select None");
roiManager("Show All with labels");

if (SAVE_CSV) {
    saveAs("Results", dir + "fiji_organoids.csv");
    roiManager("Deselect");
    roiManager("Save", dir + "fiji_rois.zip");
    print("Saved to: " + dir);
}
