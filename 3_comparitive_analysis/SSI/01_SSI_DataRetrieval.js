/************************************************************
 * 01_SSI_export_reference.js
 *
 * PURPOSE
 * -------
 * Reference Google Earth Engine (GEE) script showing how
 * the Slum Severity Index (SSI) layers used in our paper
 * were exported as 100 m rasters per country.
 *
 * IMPORTANT
 * ---------
 * - This file does NOT re-implement the full SSI model.
 * - All modelling, feature engineering, and training logic
 *   are from:
 *
 *   Li, C., Yu, L., Ndugwa, R. et al.
 *   "Mapping urban slums and their inequality in
 *    sub-Saharan Africa."
 *   Nature Cities (2025). https://doi.org/10.1038/s44284-025-00276-0
 *
 * - We only rely on their published assets (e.g. WaterImg01,
 *   ToiletImg01, HousingImg01, RoomImg01, AfricaCluster, AfriShp)
 *   and export a 5-band raster for each country:
 *
 *   1) WaterDef     – binary water deprivation
 *   2) SanitationDef– binary sanitation deprivation
 *   3) HousingDef   – binary housing deprivation
 *   4) SpaceDef     – binary living space deprivation
 *   5) SSI          – integer 0–4 (sum of the 4 elements)
 *
 ************************************************************
 * DATA PROVENANCE
 * ---------------
 *
 * SSI dataset and methodology:
 * - All modelling steps (feature selection, training data,
 *   classifier choice, thresholds, etc.) are described in
 *   Li et al. (2025). We do not modify or extend this model.
 *
 * GEE environment assumptions:
 * - The following variables / assets are assumed to exist,
 *   as they are defined in the original Li et al. GEE code:
 *
 *   - WaterImg01    : binary water deprivation (0/1)
 *   - ToiletImg01   : binary sanitation deprivation (0/1)
 *   - HousingImg01  : binary housing deprivation (0/1)
 *   - RoomImg01     : binary living space inadequacy (0/1)
 *   - AfricaCluster : living-space slum cluster mask
 *   - AfriShp       : country boundary polygons for SSA
 *
 * We do NOT show how those are computed here; that logic
 * belongs to the original authors.
 *
 ************************************************************
 * HOW THESE RASTERS ARE USED IN OUR PROJECT
 * -----------------------------------------
 *
 * - For each country, we exported a 5-band raster:
 *
 *     WaterDef, SanitationDef, HousingDef, SpaceDef, SSI
 *
 *   at 100 m resolution, clipped to the country boundary.
 *
 * - In our own pipeline, these per-country rasters are then:
 *   1) Reprojected / aligned where needed.
 *   2) Aggregated to city-segment polygons.
 *   3) Compared with our CSD model predictions to compute
 *      precision/recall/F1 and alignment metrics.
 *
 * - All of those aggregation + comparison steps are done in
 *   Python and documented in the Jupyter notebooks under:
 *
 *     3_comparative_analysis/SSI/
 *
 ************************************************************
 * GITHUB + ZENODO ARRANGEMENT
 * ---------------------------
 *
 * - The SSI rasters are too large to include directly in
 *   the GitHub repository.
 *
 * - For the subset of countries used in our manuscript, we
 *   provide a single ZIP archive on Zenodo containing files.
 *
 *     That information can be found is '3_comparative_analysis/SSI/ssi_rasters_zenodo.txt'
 *
 *   Each file has 5 bands in the following order:
 *
 *     1. WaterDef
 *     2. SanitationDef
 *     3. HousingDef
 *     4. SpaceDef
 *     5. SSI
 *
 * - The Zenodo DOI / download URL is documented in:
 *
 *     3_comparative_analysis/SSI/ssi_rasters_zenodo.txt
 *
 *
 ************************************************************
 * REPRODUCIBILITY NOTES
 * ---------------------
 *
 * - This script is provided for transparency: it shows how
 *   the four deprivation elements + SSI index can be
 *   combined and exported in GEE, assuming the upstream
 *   assets are already available.
 *
 * - Scientific questions about:
 *     • SSI model design,
 *     • feature sets and training data,
 *     • choice of thresholds, and
 *     • validation procedures
 *   should be directed to the Li et al. (2025) paper and
 *   its associated code/data.
 *
 ************************************************************
 * MINIMAL SSI EXPORT SCRIPT
 * -------------------------
 * Below is the only "live" code in this file. Everything
 * above is documentation (comments) only.
 *
 * You must:
 *   - Have WaterImg01, ToiletImg01, HousingImg01, RoomImg01,
 *     AfricaCluster, AfriShp already defined (from original
 *     SSI workflow).
 *   - Optionally adjust:
 *       CODE_PROP  – country-name or code property in AfriShp
 *       SCALE_M    – output pixel size (meters)
 *       FOLDER     – Google Drive folder name
 *       TARGETS    – null for all countries, or a list to
 *                    export a subset, e.g. ['Kenya','Ghana']
 ************************************************************/


// ===== CONFIG =====
var CODE_PROP   = 'ADM0_NAME';         // Country name field in AfriShp (adjust if needed)
var SCALE_M     = 100;                 // 100 m output resolution
var FOLDER      = 'SSI_elements_100m'; // Drive folder for exports
var TARGETS     = null;                // null = all countries; or e.g. ['Kenya','Ghana'];

// Helper to sanitize country names for filenames
function sanitize(s) {
  return ee.String(s)
    .replace('[^A-Za-z0-9]+', '_')
    .slice(0, 40);
}

// 1) Living space deprivation = OR(living-space slum cluster, inadequate room space)
var SpaceDef = AfricaCluster.max(RoomImg01).rename('SpaceDef');

// 2) Stack four deprivation elements (binary)
var Elements4 = ee.Image.cat([
  WaterImg01.rename('WaterDef'),
  ToiletImg01.rename('SanitationDef'),
  HousingImg01.rename('HousingDef'),
  SpaceDef
]).toByte();

// 3) SSI index = sum of four binary elements → 0–4
var SSI = Elements4
  .reduce(ee.Reducer.sum())
  .rename('SSI')
  .toByte();

// 4) Pack everything into a 5-band image
var SSIpack = Elements4.addBands(SSI);

// 5) Country list from AfriShp
var countries = ee.FeatureCollection(AfriShp)
  .aggregate_array(CODE_PROP)
  .distinct()
  .sort();

// Optionally restrict to specific targets
if (TARGETS !== null) {
  countries = ee.List(TARGETS);
}

print('Exporting SSI packs for countries:', countries);

// 6) Export helper for a single country
function exportSSIpack(countryName) {
  countryName = ee.String(countryName);

  var fc   = ee.FeatureCollection(AfriShp)
                .filter(ee.Filter.eq(CODE_PROP, countryName));
  var geom = fc.geometry().dissolve(1);

  var imgClipped = SSIpack.clip(geom);
  var regionBox  = imgClipped.geometry().bounds(1);

  var baseName = sanitize(countryName).cat('_SSIpack100m');

  Export.image.toDrive({
    image: imgClipped,
    description: baseName.getInfo(),
    fileNamePrefix: baseName.getInfo(),
    folder: FOLDER,
    region: regionBox,
    scale: SCALE_M,
    maxPixels: 1e13
    // crs: 'EPSG:4326' // optional, uncomment for explicit CRS
  });
}

// 7) Launch exports (client-side loop over the list)
countries.evaluate(function(list) {
  print('Launching SSI exports for', list.length, 'countries');
  list.forEach(function(name) {
    exportSSIpack(name);
  });
});
