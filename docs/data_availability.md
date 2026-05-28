# Data Availability

This document describes the availability and access routes for all datasets used
in the CSMD pipeline. The `.gitignore` intentionally excludes raw rasters, model
binaries, prediction GeoPackages, and large raw data folders. See the `.gitignore`
at the repository root for the full list of excluded patterns.

---

## 1. GitHub-tracked data (small, derived, or reference)

These files are committed to this repository because they are small, derived, and
do not contain restricted information.

| File / folder | Description |
|---|---|
| `1_preprocessing/LabelledData_For_RF/*.csv` | Labeled training CSVs for 8 IDEABench cities (segment-level morphological features + `slum_label1`) |
| `2_modelling/01_training/rf_outputs_full/tables/` | Training metadata: best hyperparameters, feature importance, CV results, predictor list |
| `2_modelling/01_training/rf_outputs_full/plots/` | Feature importance plot |
| `2_modelling/01_training/rf_outputs_loco/tables/` | LOCO validation metrics (summary and per-city) |
| `2_modelling/02_application/summary_statistics/` | City, country, region, and size-class CSMD deprivation summaries |
| `3_comparitive_analysis/SSI/Pooled_Results/` | SSI–RF per-country summaries and pooled metrics |
| `3_comparitive_analysis/MN/Outputs/*.csv` | MN–RF population statistics |
| `3_comparitive_analysis/WRI/*.csv` | WRI–RF alignment summaries and intersection reports |
| `4_Figures_Tables/Figures/` | All manuscript figures (PDF and PNG) |
| `4_Figures_Tables/Tables/` | Country-level summary table |
| `4_Figures_Tables/AllCities_Points.gpkg` | City-level point layer (~1 MB; used for Figure 2) |
| `outputs/figures/revision2/` | Revised coverage/omission figure |
| `outputs/tables/revision2/` | Revision2 omission and coverage CSVs |
| `outputs/tables/revision2/intermediate/` | Intermediate city-level joins with UCDB identifiers |

---

## 2. Zenodo — large processed outputs

Large processed files that cannot be committed to GitHub are deposited on the
project Zenodo record (DOI: 10.5281/zenodo.18788260).

| File | Description |
|---|---|
| `rf_full_model.joblib` (or similar) | Trained final Random Forest model binary |
| `*_rf_preds.gpkg` (per country) | Per-country prediction GeoPackages with `rf_prob`, `rf_label`, and `POP_SEG` |
| `GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2_with_GHSPOP2023.gpkg` | Derived UCDB 2019 polygon file with GHS-POP 2025 population estimates (`GHSPOP2023` column) |

See `zenodo/ZENODO_CONTENTS.md` for the full planned Zenodo deposit manifest.

---

## 3. External public datasets

These datasets are publicly available from their original sources. They are not
stored in this repository. Download them before running the relevant pipeline stages.

| Dataset | Source | Used in | License |
|---|---|---|---|
| **City Segments v1** | Harvard Dataverse — DOI: 10.7910/DVN/XLRSF0 | `1_preprocessing/01_preprocess_city_segments.ipynb` | See Dataverse record |
| **GHS Urban Centre Database (UCDB) 2019 V1.2** | JRC/GHSL — GHS_STAT_UCDB2015MT_GLOBE_R2019A | `GHSUCDB_Analysis/` notebooks; revision2 coverage analysis | EC Reuse and Copyright (Florczyk et al. 2019) |
| **GHS-POP R2023A (2025 epoch)** | JRC/GHSL — 100 m and 1 km resolutions | `GHSUCDB_Analysis/GHSPOP2023toUCDB2019.ipynb` | EC Reuse and Copyright |
| **Slum Severity Index (SSI)** | Li et al. (Nature Cities 2025); exported via GEE | `3_comparitive_analysis/SSI/` | See original publication |
| **Million Neighborhoods (MN)** | Public download; GeoParquet format | `3_comparitive_analysis/MN/` | See MN data source |
| **WRI Urban Land Use V1** | WRI; exported via Google Earth Engine (`01_WRI_DataDownload.js`) | `3_comparitive_analysis/WRI/` | See WRI data terms |

See `docs/data_sources/ucdb.md` and `docs/data_sources/ghspop.md` for
citation details on the GHSL datasets.

---

## 4. Restricted or registration-gated datasets

| Dataset | Access | Used in | Notes |
|---|---|---|---|
| **IDEABench v2** | Registration-gated; DANS DataStation — DOI: 10.17026/PT/X4NJII | `1_preprocessing/02_create_labeled_data.ipynb` | Raw DUA GPKGs (`data/private/`) are excluded from the repository and from Zenodo; the derived labeled CSVs (`LabelledData_For_RF/`) are included in the repo |

---

## 5. Data not included due to size

The following files are too large for GitHub and are not committed. They are either
on Zenodo or must be obtained from their original sources.

| Pattern | Reason excluded |
|---|---|
| `data/raw/CitySegments/` | Raw City Segments v1 GPKGs; many GB total |
| `2_modelling/02_application/predictions/` | Per-country prediction GPKGs; deposited on Zenodo |
| `GHS_POP_E2025_GLOBE_R2023A_54009_*/` | Raw GHS-POP rasters; several GB each |
| `GHS_STAT_UCDB2015MT_GLOBE_R2019A/` | Raw UCDB data; includes large GeoPackage |
| `3_comparitive_analysis/SSI/PerCountry_Outputs/` | Per-city clipped SSI rasters |
| `3_comparitive_analysis/MN/Outputs/MN_Comparison_Files/` | Per-city MN–RF comparison GeoPackages |
| `3_comparitive_analysis/WRI/Outputs/PerCountry_Outputs/` | Per-country WRI raster outputs |

---

## 6. Raw data that must never be committed

> **Warning:** The following must not be added to git under any circumstances.

- Raw IDEABench DUA GeoPackages (`data/private/`) — restricted dataset; do not commit
- Raw GHS-POP raster tiles (`.tif`, `.tif.ovr`) — too large; obtain from JRC/GHSL
- Any file matching `**/*_rf_preds.gpkg` — prediction GeoPackages belong on Zenodo
- Model binary `*.joblib` / `*.pkl` — belongs on Zenodo

The `.gitignore` at the repository root enforces most of these exclusions
automatically. Review it before staging new files with `git add`.
