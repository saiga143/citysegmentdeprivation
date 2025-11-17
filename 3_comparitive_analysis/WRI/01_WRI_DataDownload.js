/***************************************************************
 * 01_WRI_DataDownload.js
 * -------------------------------------------------------------
 * Purpose:
 *   Download the WRI Urban Land Use V1 image collection as
 *   multi-band GeoTIFFs to Google Drive, in manageable batches,
 *   for later offline processing and comparison with CSD/RF.
 *
 * Data source:
 *   Guzder-Williams, B., Mackres, E., Angel, S., Blei, A. M.
 *   & Lamson-Hall, P. (2023).
 *   "Intra-urban land use maps for a global sample of cities
 *   from Sentinel-2 satellite imagery and computer vision."
 *   Computers, Environment and Urban Systems, 100, 101917.
 *   https://doi.org/10.1016/j.compenvurbsys.2022.101917
 *
 *   GEE asset:
 *     projects/wri-datalab/urban_land_use/V1
 *
 * Context in this workflow:
 *   - This script is only for DATA DOWNLOAD.
 *   - The exported GeoTIFFs (all 10 bands per image) are used
 *     later in the WRI vs CSD comparative analysis notebooks.
 *   - Download is done in batches using OFFSET and LIMIT so that
 *     the Earth Engine task list doesn’t get overwhelmed.
 *
 * What this script does:
 *   1. Reads the WRI Urban Land Use V1 image collection.
 *   2. Prints the total number of images.
 *   3. Selects a slice of the collection using OFFSET + LIMIT.
 *   4. For each image in that slice:
 *        - Builds a human-readable file name:
 *          "<region>_y<year>_<system:index>"
 *        - Exports the full 10-band image to Google Drive
 *          (one GeoTIFF per image).
 *
 * How to use (batching logic):
 *   - Set OFFSET and LIMIT before running.
 *   - Example:
 *       OFFSET = 0;   LIMIT = 30;    // first batch
 *       OFFSET = 30;  LIMIT = 30;    // second batch
 *       OFFSET = 60;  LIMIT = 30;    // third batch
 *     etc., until OFFSET + LIMIT covers the full collection.
 *   - After each run:
 *       - Check the Tasks tab in the Code Editor.
 *       - Make sure exports are running/completed.
 *       - Then increment OFFSET and re-run the script.
 *
 * Output:
 *   - Google Drive folder: DRIVE_FOLDER (e.g. "WRI_urban_land_use_V1")
 *   - Files: one multi-band GeoTIFF per image with filename:
 *       <region>_y<year>_<system:index>.tif
 *     where:
 *       region = ee.Image property "region" (spaces replaced by "_")
 *       year   = ee.Image property "year"   (spaces replaced by "_")
 *       system:index = underlying EE ID
 *
 * Notes:
 *   - SCALE_M is set to 5 m, matching the native resolution of V1.
 *   - MAX_PIXELS is set high (1e13) to allow large exports.
 *   - CRS is left as the native projection for each asset.
 *   - This script does not subset bands; it exports all 10 bands.
 *   - You will need to organize downloaded GeoTIFFs locally
 *     (e.g., group by region, year) before downstream processing.
 ***************************************************************/


// === Settings ===

// WRI Urban Land Use V1 collection
var IC_ID = 'projects/wri-datalab/urban_land_use/V1';

// Google Drive target folder (creates if it does not exist)
var DRIVE_FOLDER = 'WRI_urban_land_use_V1';

// Export resolution (meters). Native resolution is 5 m.
var SCALE_M = 5;

// Allow large rasters (avoid "Too many pixels" errors).
var MAX_PIXELS = 1e13;

// ---------------------------------------------------------------------
// Batching settings:
//   - Run ~25–40 tasks at a time depending on your quota.
//   - After each batch finishes, increase OFFSET and run again.
// ---------------------------------------------------------------------
var OFFSET = 210;  // <-- change to 0, 30, 60, 90, ... per batch
var LIMIT  = 30;   // number of images to export in this batch

// ---------------------------------------------------------------------
// 1. Load collection and inspect size
// ---------------------------------------------------------------------
var col = ee.ImageCollection(IC_ID);
var total = col.size().getInfo();
print('Total images in collection:', total);

// Select the current batch [OFFSET, OFFSET + LIMIT)
var batchList = col.toList(LIMIT, OFFSET);

// ---------------------------------------------------------------------
// 2. Helper: safe-get a string property from image
// ---------------------------------------------------------------------
function getStr(img, key, fallback) {
  // If property exists, use it; otherwise use fallback.
  var v = ee.String(
    ee.Algorithms.If(
      img.propertyNames().contains(key),
      img.get(key),
      fallback
    )
  );
  // Replace spaces with underscores for safe filenames.
  return v.replace(' ', '_');
}

// ---------------------------------------------------------------------
// 3. Loop over batch and create export tasks
// ---------------------------------------------------------------------
for (var i = 0; i < LIMIT && (OFFSET + i) < total; i++) {
  var img = ee.Image(batchList.get(i));

  // Build a readable filename: region_year_index
  var region = getStr(img, 'region', 'unknown');
  var year   = getStr(img, 'year',   'unknown');
  var idx    = ee.String(img.get('system:index'));  // usually present

  var fname = ee.String(region)
    .cat('_y')
    .cat(year)
    .cat('_')
    .cat(idx);

  // Create export task (multi-band GeoTIFF)
  Export.image.toDrive({
    image: img,                       // all 10 bands
    description: fname.getInfo(),     // task name in EE Tasks tab
    folder: DRIVE_FOLDER,             // Drive folder
    fileNamePrefix: fname.getInfo(),  // Drive filename prefix
    fileFormat: 'GeoTIFF',
    region: img.geometry(),           // full extent of the image
    scale: SCALE_M,
    maxPixels: MAX_PIXELS
    // crs: left unset → native projection for each asset
  });
}

// ---------------------------------------------------------------------
// 4. Log batch info
// ---------------------------------------------------------------------
print(
  'Started batch OFFSET =',
  OFFSET,
  'to',
  Math.min(OFFSET + LIMIT, total),
  '| LIMIT =',
  LIMIT,
  '| Check Tasks panel for export progress.'
);
