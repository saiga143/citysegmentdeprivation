# Code-to-Figure and Code-to-Output Map

This table maps each script and notebook to the manuscript figures, extended data
figures, tables, or intermediate outputs it produces.

**Status key:**  
`tracked` — output file is in the repository  
`zenodo` — output is too large for GitHub; deposited on Zenodo  
`not-in-repo` — output depends on restricted or large external data not committed  
`console-only` — output is printed to the R/Python session; nothing saved to disk  

---

## Stage 1 — Preprocessing

| Code file | Outputs | Output location | Status | Notes |
|---|---|---|---|---|
| `1_preprocessing/01_preprocess_city_segments.ipynb` | City Segments CSVs with all 21 derived indices (`i1_` – `i10_`) | `data/raw/CitySegments/` (per-country) | not-in-repo | Reads raw City Segments v1 GPKGs from Harvard Dataverse; outputs large per-country CSVs |
| `1_preprocessing/02_create_labeled_data.ipynb` | `LabelledData_For_RF/*_labeled_thr030.csv` (8 cities) | `1_preprocessing/LabelledData_For_RF/` | tracked | Reads IDEABench GPKGs from `data/private/`; applies `slum_fraction ≥ 0.30` threshold |

---

## Stage 2 — Modelling

| Code file | Outputs | Output location | Status | Notes |
|---|---|---|---|---|
| `2_modelling/01_training/run_VSURF.R` | VSURF variable rankings (3 ntree settings) | Console only | console-only | **No disk output.** Results are printed to the R session. Hard-coded path at line 37 requires correction before re-running. |
| `2_modelling/01_training/train_rf_full.py` | Model joblib, `best_params_full.json`, `feature_importance_full.csv/png`, full-data predictions CSV | `2_modelling/01_training/rf_outputs_full/` | mixed (tables tracked; model joblib on Zenodo) | CLI script; run with `--input-folder` and `--output-folder` args |
| `2_modelling/01_training/Validation_rf_model_loco.py` | `loco_metrics_by_city.csv`, `loco_metrics_summary.csv`, per-city LOCO predictions | `2_modelling/01_training/rf_outputs_loco/tables/` | tracked | LOCO evaluation; independent of the trained model file; requires labeled CSVs |
| `2_modelling/02_application/01_apply_rf_predictions.ipynb` | `*_rf_preds.gpkg` (one GeoPackage per country) | `2_modelling/02_application/predictions/` | zenodo | Requires trained model joblib + per-country City Segments CSVs/GPKGs |
| `2_modelling/02_application/02_summary_statistics.ipynb` | `city_deprivation_80pct.csv`, `country_deprivation_80pct.csv`, `regional_deprivation_80pct.csv`, city-size and region-size breakdown CSVs | `2_modelling/02_application/summary_statistics/` | tracked | Reads prediction GPKGs; applies 80% QC filter |

---

## Stage 3 — Comparative Analysis

| Code file | Outputs | Output location | Status | Notes |
|---|---|---|---|---|
| `3_comparitive_analysis/SSI/01_SSI_DataRetrieval.js` | SSI raster tiles (GEE export) | GEE Drive / external | not-in-repo | Google Earth Engine JavaScript API |
| `3_comparitive_analysis/SSI/02_ssi_clip_to_cities.ipynb` | Per-city SSI raster clips | `3_comparitive_analysis/SSI/PerCountry_Outputs/` | not-in-repo | Requires GEE-exported SSI rasters |
| `3_comparitive_analysis/SSI/03_ssi_rf_comparison.ipynb` | Per-country `*_ssi_rf_summary*.csv`, `country_processing_audit.csv` | `3_comparitive_analysis/SSI/Pooled_Results/` | tracked (summaries) | Computes SSI–RF agreement metrics per country |
| `3_comparitive_analysis/SSI/04_ssi_rf_comparison_plots.ipynb` | `multipanel_ssi_rf_boxplots_gray_compact.pdf/png` | `3_comparitive_analysis/SSI/Pooled_Results/Figures/` | tracked | Panel used in Figure 5 |
| `3_comparitive_analysis/MN/01_MN_Data_and_Labels.ipynb` | Per-city MN–RF comparison GPKGs and CSVs | `3_comparitive_analysis/MN/Outputs/MN_Comparison_Files/` | not-in-repo | Requires MN GeoParquet + prediction GPKGs |
| `3_comparitive_analysis/MN/02_MN_RF_comparison.ipynb` | `multipanel_mn_rf_boxplots_gray_compact.pdf/png`, `mn_rf_summary_segments_population*.csv` | `3_comparitive_analysis/MN/Outputs/` | tracked | Panel used in Figure 5; global MN summary table |
| `3_comparitive_analysis/WRI/01_WRI_DataDownload.js` | WRI Urban Land Use rasters (GEE export) | GEE Drive / external | not-in-repo | Google Earth Engine JavaScript API |
| `3_comparitive_analysis/WRI/02_WRI_IntersectionReports.ipynb` | `raster_gpkg_pairs_all.csv`, `pairwise_intersection_fraction.csv`, `rasters_with_any_intersection.csv` | `3_comparitive_analysis/WRI/intersect_reports/` | tracked | Identifies which WRI rasters overlap which prediction GPKGs |
| `3_comparitive_analysis/WRI/03_WRI_PerCountry_Metrics.ipynb` | Per-country WRI–RF alignment stats, `wri_rf_population_stats.csv`, WRI boxplot figure | `3_comparitive_analysis/WRI/Outputs/` | mixed | Panel used in Figure 5; large per-country GPKGs not tracked |
| `3_comparitive_analysis/WRI/04_WRI_Tables.ipynb` | `wri_rf_population_summary_GLOBAL_rule_threshold_table_millions.csv` | `3_comparitive_analysis/WRI/` | tracked | Global WRI summary table |

---

## Stage 4 — Figures and Tables

| Code file | Manuscript output | Output file | Output location | Status | Notes |
|---|---|---|---|---|---|
| `4_Figures_Tables/Generate_AllCities_Points.ipynb` | (data layer) | `AllCities_Points.gpkg` | `4_Figures_Tables/` | tracked | City-level point layer used by Figure 2 |
| `4_Figures_Tables/01_Figure2_Global_DeprivedShare.ipynb` | **Figure 2** | `Figure2_Global_Deprived_Share.pdf/png` | `4_Figures_Tables/Figures/` | tracked | World map of CSMD deprived-population share by city |
| `4_Figures_Tables/02_Figure3_Lollipop_Citysize.ipynb` | **Figure 3** | `Figure3_Lollipop_Regional_Deprivation.pdf/png`, `Figure3_CitySize_2x2.pdf/png`, regional panels | `4_Figures_Tables/Figures/` | tracked | Lollipop plot + city-size bar panels by region |
| `4_Figures_Tables/03_Figure4_Deprivation_by_Citysizemix.ipynb` | **Figure 4 (original)** | `Figure4_Deprivation_by_CitySizeMix.pdf/png` | `4_Figures_Tables/Figures/` | tracked | **Note:** this notebook reflects a pre-revision version of Figure 4. The accepted/revised Figure 4 appears to be represented by `outputs/figures/revision2/population_totals_citysize_region_QC80.*` from the UCDB coverage analysis. Confirm against the final accepted manuscript. |
| `4_Figures_Tables/04_Figure5_ThreeComparison.ipynb` | **Figure 5** | `Figure5_ThreeComparisons_withreg.pdf/png` | `4_Figures_Tables/Figures/` | tracked | Three-panel comparison: SSI, MN, WRI |
| `4_Figures_Tables/05_GlobalSummaryTable.ipynb` | **Global summary table** | `CountrySummary_RF.csv`, `CountrySummary_RF_raw.csv` | `4_Figures_Tables/Tables/` | tracked | Country-level CSMD summary |

---

## Revision 2 — Coverage and Omission Analysis

These outputs were produced from notebooks in `../GHSUCDB_Analysis/` (outside the
GitHub repository) and copied into the repo as final outputs. The source notebooks
are not yet integrated; their paths will be updated in a later step.

| Source notebook | Manuscript output | Output file | Output location | Status |
|---|---|---|---|---|
| `globalsouthcomparison.ipynb` | (intermediate) | `city_deprivation_80pct_with_ids_regionL2.csv`, `city_deprivation_80pct_qc_with_ucdb_regions.csv` | `outputs/tables/revision2/intermediate/` | tracked |
| `revision2_omittedstatistics.ipynb` | **Coverage/omission figure** (revised Figure 4 or Extended Data) | `population_totals_citysize_region_QC80.pdf/png` | `outputs/figures/revision2/` | tracked |
| `revision2_omittedstatistics.ipynb` | **Coverage tables** | `city_size_comparison_QC80_Africa_Asia_LAC_millions.csv`, `city_counts_ucdb_vs_citysegments_80pct.csv`, `city_stats_ucdb_vs_citysegments_80pct_counts_and_pop.csv`, `regionL1_regionL2_citysize_omission_80pct.csv` | `outputs/tables/revision2/` | tracked |
| `revision2_extendedtable2.ipynb` | **Extended Data Table 2** | `country_level_omission_table.csv` | `outputs/tables/revision2/` | tracked |
