/*  A — Measure the label mask produced by organoid_quant.py, inside Fiji.
 *
 *  Reproduces the Python numbers exactly (same masks), but puts every object
 *  in Fiji's own Results table and ROI Manager, so you can inspect, correct
 *  and re-measure objects by hand.
 *
 *  Use this when you want the segmentation quality of the Python pipeline
 *  with the convenience of Fiji. Use macro B for a pure-Fiji segmentation.
 */

// ------------------------------------------------------------------ settings
maskPath   = "";      // leave "" to get a file chooser
PIXEL_SIZE = 0;       // micrometres per pixel. 0 = report in pixels
UNIT       = "um";
SAVE_CSV   = true;

// --------------------------------------------------------------------- setup
run("Close All");
run("Clear Results");
roiManager("reset");
setOption("BlackBackground", true);

if (maskPath == "") maskPath = File.openDialog("Select mask_labels.png");
open(maskPath);
rename("labels");

if (bitDepth() == 24)
    exit("mask_labels.png opened as RGB. Re-save it as 16-bit grayscale.");

if (PIXEL_SIZE > 0)
    run("Set Scale...", "distance=1 known=" + PIXEL_SIZE + " unit=" + UNIT);
else
    run("Set Scale...", "distance=0 known=0 unit=pixel");

run("Set Measurements...",
    "area centroid perimeter shape feret's redirect=None decimal=3");

// ------------------------------------------------- one ROI per label value
getStatistics(dummyArea, dummyMean, dummyMin, maxLabel);
nLab = maxLabel;
print("\\Clear");
print("Label image: " + nLab + " labels found");

for (i = 1; i <= nLab; i++) {
    selectWindow("labels");
    run("Select None");
    setThreshold(i, i);
    run("Create Selection");
    if (selectionType() >= 0) {
        getStatistics(a);
        if (a > 0) {
            roiManager("Add");
            roiManager("Select", roiManager("count") - 1);
            roiManager("Rename", "organoid_" + IJ.pad(i, 3));
            run("Measure");
        }
    }
    resetThreshold();
}
run("Select None");

// ------------------------------------------------------------------ summary
totalArea = 0;
for (r = 0; r < nResults; r++) totalArea += getResult("Area", r);
u = "px^2";
if (PIXEL_SIZE > 0) u = UNIT + "^2";

print("Organoids measured : " + nResults);
print("Total surface      : " + d2s(totalArea, 1) + " " + u);
if (nResults > 0)
    print("Mean surface       : " + d2s(totalArea / nResults, 1) + " " + u);

// ------------------------------------------------------------------- output
if (SAVE_CSV) {
    outDir = File.getParent(maskPath) + File.separator;
    saveAs("Results", outDir + "fiji_measurements.csv");
    roiManager("Deselect");
    roiManager("Save", outDir + "fiji_rois.zip");
    print("Saved: fiji_measurements.csv, fiji_rois.zip");
}

selectWindow("labels");
roiManager("Show All with labels");
