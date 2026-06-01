# Zenodo Deposit Contents

Zenodo DOI: 10.5281/zenodo.20486977

This file documents which files are stored on the project Zenodo record, which are
tracked in the GitHub repository, and which must be obtained from third-party
providers. It is intended to guide users reconstructing the full local data layout
needed to run the pipeline.

**GitHub** contains: code, documentation, small summary tables, figures, and
labeled training data (8 IDEABench cities).

**Zenodo** contains: the trained RF model binary, per-country CSMD prediction
GeoPackages, large comparative-analysis derived files, and the UCDB/GHS-POP
derived intermediate used in the revision2 coverage analysis.

**External providers** supply: raw rasters, registration-gated training data, and
third-party geospatial datasets that cannot be redistributed here.

Large rasters, prediction GeoPackages, model binaries, and large intermediate
outputs are excluded from GitHub by `.gitignore`.

---

## 1. Project Zenodo deposit

The following files and folders are deposited on the project Zenodo record and
should be placed locally under `data_external/zenodo/` after download.

---

### A. Final RF model

**Expected local path:**
```
data_external/zenodo/models/rf_final_model_full.joblib
```

The serialized final Random Forest model trained on all eight IDEABench cities
using the full labeled dataset (no holdout). Produced by
`2_modelling/01_training/train_rf_full.py`. Not stored in GitHub because model
binaries are large and not human-readable.

---

### B. Global CSMD prediction GeoPackages

**Expected local path:**
```
data_external/zenodo/predictions/
  {country}_rf_preds.gpkg    (one file per country)
```

Per-country prediction GeoPackages produced by applying the final RF model to all
City Segments v1 countries (`2_modelling/02_application/01_apply_rf_predictions.ipynb`).
Each file contains segment-level geometry, `rf_prob`, `rf_label` (threshold 0.40),
and `POP_SEG` (GHS-POP 2025 city-allocated population). Not stored in GitHub
because these are large geospatial outputs covering 100+ countries.

---

### C. SSI comparative-analysis derived data

**Expected local path:**
```
data_external/zenodo/ssi_clipped/
  {country}/
    SSIpack100m_clipped_to_city_blocks_SIGNAL.tif
```

Per-country SSI rasters clipped to the union of CSMD city-segment polygons,
produced by `3_comparitive_analysis/SSI/02_ssi_clip_to_cities.ipynb`. These are
large raster files and are excluded from GitHub.

Final small SSI summary CSVs and figures are tracked in the repository at:
`3_comparitive_analysis/SSI/Pooled_Results/`

---

### D. MN comparative-analysis derived data

**Expected local paths:**
```
data_external/zenodo/mn_blocks_by_country/
  {country}_mn_blocks.gpkg    (one file per country)

data_external/zenodo/mn_comparison_files/
  {country}/
    {country}_segments_mnlabels_k{K}_maj{TAG}.gpkg
    {country}_segments_mnlabels_k{K}_maj{TAG}.csv
```

Per-country Million Neighborhoods (MN) block extractions and segment-level
comparison files produced by `3_comparitive_analysis/MN/01_MN_Data_and_Labels.ipynb`.
Run twice per country (K = 3 and K = 5; TAG ∈ {10, 20, 30}). These files are
large and excluded from GitHub.

Final small MN summary CSVs and figures are tracked in the repository at:
`3_comparitive_analysis/MN/Outputs/`

---

### E. WRI comparative-analysis derived data

**Expected local path:**
```
data_external/zenodo/wri_per_country_outputs/
  {country}/
    {country}_wri_informal_per_block.csv
    {country}_wri_overlap_audit.csv
    {country}_wri_vs_rf_threshold_sweep_country.csv
    {country}_wri_vs_rf_per_block_with_preds.csv
    {country}_overall_deprived_counts_by_tau.csv
    {country}_per_city_deprived_counts_by_tau.csv
```

Per-country WRI Urban Land Use V1 segment extraction and comparison outputs
produced by `3_comparitive_analysis/WRI/03_WRI_PerCountry_Metrics.ipynb`. These
files can be large in aggregate and are excluded from GitHub.

Final small WRI summary CSVs, intersection reports, and figures are tracked in the
repository at:
```
3_comparitive_analysis/WRI/Outputs/
3_comparitive_analysis/WRI/intersect_reports/
```

---

### F. Revision2 UCDB/GHS-POP coverage analysis — large derived intermediate

**Expected local path:**
```
data_external/zenodo/
  GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2_with_GHSPOP2023.gpkg
```

The GHS Urban Centre Database 2019 V1.2 polygon file augmented with GHS-POP
R2023A 2025-epoch city population estimates (`GHSPOP2023` column). Produced by
`notebooks/revision2_coverage/01_GHSPOP2023toUCDB2019.ipynb`. Not stored in
GitHub due to file size.

Final small revision2 CSVs and figures are tracked in the repository at:
```
outputs/tables/revision2/
outputs/figures/revision2/
```

---

### G. Revision2 intermediate CSVs

The following intermediate files are small enough to be tracked in GitHub and are
committed under `outputs/tables/revision2/intermediate/`. They are also safe to
mirror on Zenodo for provenance:

- `city_deprivation_80pct_with_ids_regionL2.csv`
  CSMD city summary with UCDB `ID_HDC_G0` identifiers and Region L2 label added.

- `city_deprivation_80pct_qc_with_ucdb_regions.csv`
  CSMD city summary joined to UCDB region fields, population, and city-size class.

**Final CSV outputs** (tracked in GitHub at `outputs/tables/revision2/`):

- `city_size_comparison_QC80_Africa_Asia_LAC_millions.csv`
- `city_counts_ucdb_vs_citysegments_80pct.csv`
- `city_stats_ucdb_vs_citysegments_80pct_counts_and_pop.csv`
- `regionL1_regionL2_citysize_omission_80pct.csv`
- `country_level_omission_table.csv`

**Revised coverage figure** (tracked in GitHub at `outputs/figures/revision2/`):

- `population_totals_citysize_region_QC80.pdf`
- `population_totals_citysize_region_QC80.png`

---

## 2. Third-party data — not deposited by us

The following datasets must be obtained from their original providers. They are not
committed to GitHub and are not on the project Zenodo record.

---

### City Segments v1

**Purpose:** Primary input to preprocessing and the full RF prediction pipeline.
Each city's segments provide the morphological indices (i1–i10) used as RF
predictors.
**Provider:** Harvard Dataverse
**Access:** https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XLRSF0
**DOI:** 10.7910/DVN/XLRSF0
**Expected local placement:** `data_external/city_segments/`
Raw files are large and excluded from GitHub.

---

### IDEABench

**Purpose:** Ground-truth DUA (Deprived Urban Area) GeoPackages for the eight
training cities used to create labeled data for RF training.
**Provider:** DANS DataStation (access conditions apply)
**Access:** https://phys-techsciences.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/PT/X4NJII
**Dataset DOI:** 10.17026/PT/X4NJII
**Paper DOI:** 10.1016/j.rse.2026.115272
**Expected local placement:** `data_external/ideabench/`
Raw DUA GeoPackages must not be committed to GitHub (restricted dataset). The
derived labeled CSVs (`1_preprocessing/LabelledData_For_RF/`) are tracked.

---

### GHS Urban Centre Database 2019 V1.2 (UCDB)

**Purpose:** Urban centre polygons and attributes used in the revision2 coverage
and omission analysis.
**Provider:** JRC/GHSL — `GHS_STAT_UCDB2015MT_GLOBE_R2019A`
**Expected local placement:** `data_external/ucdb/` (raw); the derived GPK with
GHS-POP 2025 estimates is on Zenodo (see §1-F above).
Raw files are large and excluded from GitHub.

---

### GHS-POP R2023A 2025 epoch

**Purpose:** Global population rasters (100 m and 1 km) used to allocate city
population to UCDB urban centres in the revision2 coverage analysis.
**Provider:** JRC/GHSL — `GHS_POP_E2025_GLOBE_R2023A_54009_*`
**Expected local placement:** `data_external/ghspop/`
Raw raster tiles are very large and excluded from GitHub.

---

### Slum Severity Index (SSI) source data

**Purpose:** Slum Severity Index rasters used in the SSI comparative analysis.
Raw SSI rasters must be downloaded from the SSI Zenodo record and exported
via Google Earth Engine using the script `3_comparitive_analysis/SSI/01_SSI_DataRetrieval.js`.
The project Zenodo package (DOI: 10.5281/zenodo.20486977) contains only the
derived per-country clipped SSI rasters (`data_external/zenodo/ssi_clipped/`),
not the raw SSI source data.
**Provider:** Li, C., Yu, L., Ndugwa, R. et al. (Nature Cities 2025,
DOI: 10.1038/s44284-025-00276-0)
**Raw data access:** https://zenodo.org/records/14998570
**Raw data DOI:** 10.5281/zenodo.14998570
**Expected local placement:** `data_external/ssi_raw/`
(one `{Country}_SSIpack100m.tif` per country)
Raw rasters are large and excluded from GitHub.

---

### Million Neighborhoods raw GeoParquet

**Purpose:** Block-level informal settlement data for sub-Saharan Africa, used in
the MN comparative analysis.
**Provider:** MN portal — https://www.millionneighborhoods.africa/download
(Bettencourt & Marchio, Nature 2025, DOI: 10.1038/s41586-025-09465-2)
**Expected local placement:** `data_external/mn_raw/africa_geodata.parquet`
Raw file is large and excluded from GitHub.

---

### WRI Urban Land Use dataset rasters

**Purpose:** Urban land-use class rasters (5 m resolution) used in the WRI
comparative analysis. Informal subdivision and atomistic classes (`p_informal`)
are used for the WRI comparison.
**Provider:** WRI/Guzder-Williams et al. (CEUS 2023, DOI:
10.1016/j.compenvurbsys.2022.101917)
**Access:** Google Earth Engine asset — https://code.earthengine.google.com/?asset=projects/wri-datalab/urban_land_use/V1
**Export script:** `3_comparitive_analysis/WRI/01_WRI_DataDownload.js`
**Expected local placement:** `data_external/wri_raw/{country}/*.tif`
Raw rasters are large and excluded from GitHub.

---

## 3. Recommended local folder layout

After downloading from Zenodo and the external providers listed above, the
`data_external/` directory should have the following structure:

```
data_external/
├── city_segments/          # City Segments v1 (Harvard Dataverse)
├── ideabench/              # IDEABench DUA GPKGs (DANS DataStation, access conditions apply)
├── ucdb/                   # GHS UCDB 2019 raw (JRC/GHSL)
├── ghspop/                 # GHS-POP R2023A raw rasters (JRC/GHSL)
├── ssi_raw/                # SSI GEE exports (Li et al. 2025)
├── mn_raw/                 # MN GeoParquet (MN portal)
├── wri_raw/                # WRI Land Use V1 GeoTIFFs (GEE/WRI)
│   ├── {country}/
│   │   └── *.tif
└── zenodo/                 # Downloaded from project Zenodo record
    ├── models/
    │   └── rf_final_model_full.joblib
    ├── predictions/
    │   └── {country}_rf_preds.gpkg
    ├── ssi_clipped/
    │   └── {country}/
    │       └── SSIpack100m_clipped_to_city_blocks_SIGNAL.tif
    ├── mn_blocks_by_country/
    │   └── {country}_mn_blocks.gpkg
    ├── mn_comparison_files/
    │   └── {country}/
    │       └── {country}_segments_mnlabels_k{K}_maj{TAG}.gpkg/.csv
    ├── wri_per_country_outputs/
    │   └── {country}/
    │       └── {country}_wri_vs_rf_threshold_sweep_country.csv (etc.)
    └── GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2_with_GHSPOP2023.gpkg
```

---

## 4. GitHub-tracked outputs (small; committed to repository)

The following small outputs are intentionally tracked in the repository and do not
require downloading from Zenodo:

| Path | Contents |
|---|---|
| `1_preprocessing/LabelledData_For_RF/` | Labeled training CSVs (8 cities) |
| `2_modelling/01_training/rf_outputs_full/tables/` | Training metadata, hyperparameters, feature importance, CV results |
| `2_modelling/01_training/rf_outputs_loco/tables/` | LOCO validation metrics (per-city and summary) |
| `2_modelling/02_application/summary_statistics/` | City, country, region, and city-size CSMD deprivation summaries |
| `3_comparitive_analysis/SSI/Pooled_Results/` | SSI–RF per-country summaries, pooled metrics, and figure |
| `3_comparitive_analysis/MN/Outputs/` | MN–RF population statistics and figure |
| `3_comparitive_analysis/WRI/Outputs/` | WRI–RF summary tables and figure |
| `3_comparitive_analysis/WRI/intersect_reports/` | WRI raster × GPKG intersection CSVs |
| `4_Figures_Tables/` | All manuscript figures (PDF and PNG) and the country summary table |
| `outputs/tables/revision2/` | Revision2 omission and coverage CSVs |
| `outputs/tables/revision2/intermediate/` | Intermediate city-level joins with UCDB identifiers |
| `outputs/figures/revision2/` | Revised Figure 4 (coverage/omission stacked bar) |

---

## 5. Notes and limitations

- **Exact filenames** on Zenodo may differ slightly depending on how the deposit is
  packaged. If a filename does not match what a notebook expects, update the
  relevant path variable at the top of the notebook (each notebook has a
  `PATH CONFIGURATION` cell with named variables for all inputs and outputs).

- **Folder names** under `data_external/zenodo/` are assumed by the notebook path
  configurations. Do not rename them unless you also update the corresponding
  path variables.

- **`.gitignore`** excludes the entire `data_external/` directory as well as
  individual large file extensions (`*.tif`, `*.parquet`, `*.joblib`,
  `**/*_rf_preds.gpkg`, etc.). Review `.gitignore` before staging new files.

- **Third-party licenses:** Each external dataset carries its own license.
  Consult the provider links above before redistributing any raw files.
