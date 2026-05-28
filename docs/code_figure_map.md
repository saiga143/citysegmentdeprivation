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
| `1_preprocessing/01_preprocess_city_segments.ipynb` | City Segments CSVs with all 21 derived indices (`i1_` – `i10_`) | `data_external/city_segments/` (per-country) | not-in-repo | Reads raw City Segments v1 shapefiles from Harvard Dataverse; outputs large per-country CSVs alongside source shapefiles |
| `1_preprocessing/02_create_labeled_data.ipynb` | `LabelledData_For_RF/*_labeled_thr030.csv` (8 cities) | `1_preprocessing/LabelledData_For_RF/` | tracked | Reads IDEABench GPKGs from `data_external/ideabench/`; applies `slum_fraction ≥ 0.30` threshold |

---

## Stage 2 — Modelling

| Code file | Outputs | Output location | Status | Notes |
|---|---|---|---|---|
| `2_modelling/01_training/run_VSURF.R` | `vsurf_selected_variables_by_ntree.csv`, `vsurf_intersection_variables.csv`, `vsurf_run_metadata.txt`, `vsurf_selected_variables_venn.png` | `2_modelling/01_training/rf_outputs_full/tables/` (CSVs + metadata), `/plots/` (PNG) | tracked | Uses portable repo-root detection; saves VSURF outputs to disk for auditability; stable intersection across ntree = 800, 1000, 1200 defines the final predictor set |
| `2_modelling/01_training/train_rf_full.py` | Model joblib, `best_params_full.json`, `feature_importance_full.csv/png`, full-data predictions CSV | `2_modelling/01_training/rf_outputs_full/` | mixed (tables tracked; model joblib on Zenodo) | CLI script; run with `--input-folder` and `--output-folder` args |
| `2_modelling/01_training/Validation_rf_model_loco.py` | `loco_metrics_by_city.csv`, `loco_metrics_summary.csv`, per-city LOCO predictions | `2_modelling/01_training/rf_outputs_loco/tables/` | tracked | LOCO evaluation; independent of the trained model file; requires labeled CSVs |
| `2_modelling/02_application/01_apply_rf_predictions.ipynb` | `*_rf_preds.gpkg` (one GeoPackage per country) | `data_external/zenodo/predictions/` | zenodo | Requires RF model joblib from `data_external/zenodo/models/` + City Segments in `data_external/city_segments/` |
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
| `4_Figures_Tables/03_Figure4_Deprivation_by_Citysizemix.ipynb` | **Figure 4 (pre-revision — retained for provenance)** | `Figure4_Deprivation_by_CitySizeMix.pdf/png` | `4_Figures_Tables/Figures/` | tracked | This notebook is retained for provenance. It reflects an earlier version of Figure 4 showing country-level deprived-population composition by city-size mix, based on CSMD segment-aggregated population. In the revised repository structure, the accepted Figure 4 is produced by `notebooks/revision2_coverage/04_globalsouthcomparison.ipynb`; see the Revision 2 section below. |
| `4_Figures_Tables/04_Figure5_ThreeComparison.ipynb` | **Figure 5** | `Figure5_ThreeComparisons_withreg.pdf/png` | `4_Figures_Tables/Figures/` | tracked | Three-panel comparison: SSI, MN, WRI |
| `4_Figures_Tables/05_GlobalSummaryTable.ipynb` | **Global summary table** | `CountrySummary_RF.csv`, `CountrySummary_RF_raw.csv` | `4_Figures_Tables/Tables/` | tracked | Country-level CSMD summary |

---

## Revision 2 — Coverage and Omission Analysis

Source notebooks are integrated in `notebooks/revision2_coverage/` with portable
path configuration. See `notebooks/revision2_coverage/README.md` for execution
order and `data_external/` setup instructions.

| Source notebook | Manuscript output | Output file | Output location | Status | Notes |
|---|---|---|---|---|---|
| `notebooks/revision2_coverage/01_GHSPOP2023toUCDB2019.ipynb` | (derived input) | `GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2_with_GHSPOP2023.gpkg` | `data_external/zenodo/` | zenodo | Large derived file; not committed to GitHub; used by notebooks 02–04 |
| `notebooks/revision2_coverage/02_revision2_omittedstatistics.ipynb` | (intermediate) | `city_deprivation_80pct_with_ids_regionL2.csv`, `city_deprivation_80pct_qc_with_ucdb_regions.csv` | `outputs/tables/revision2/intermediate/` | tracked | UCDB-matched city summaries with region labels |
| `notebooks/revision2_coverage/02_revision2_omittedstatistics.ipynb` | **Omission summary table** | `regionL1_regionL2_citysize_omission_80pct.csv` | `outputs/tables/revision2/` | tracked | Regional × city-size omission statistics |
| `notebooks/revision2_coverage/03_revision2_extendedtable2.ipynb` | **Extended Data Table 2** | `country_level_omission_table.csv` | `outputs/tables/revision2/` | tracked | 135 countries sorted by omitted population |
| `notebooks/revision2_coverage/04_globalsouthcomparison.ipynb` | **Revised Figure 4 (coverage/omission)** | `population_totals_citysize_region_QC80.pdf/png` | `outputs/figures/revision2/` | tracked | In the revised repository structure, the accepted Figure 4 is represented by this output; see note below |
| `notebooks/revision2_coverage/04_globalsouthcomparison.ipynb` | **Coverage summary CSVs** | `city_size_comparison_QC80_Africa_Asia_LAC_millions.csv`, `city_counts_ucdb_vs_citysegments_80pct.csv`, `city_stats_ucdb_vs_citysegments_80pct_counts_and_pop.csv` | `outputs/tables/revision2/` | tracked | |

> **Note on Figure 4:** In the revised repository structure, the accepted Figure 4
> is represented by `outputs/figures/revision2/population_totals_citysize_region_QC80.*`,
> produced by `notebooks/revision2_coverage/04_globalsouthcomparison.ipynb`.
> The earlier notebook `4_Figures_Tables/03_Figure4_Deprivation_by_Citysizemix.ipynb`
> is retained for provenance — it reflects a pre-revision version of Figure 4 showing
> country-level deprived-population composition by city-size mix, based on CSMD
> segment-aggregated population rather than UCDB/GHS-POP denominators.
> Confirm correspondence with the final accepted manuscript before citing either output
> as the published figure.
