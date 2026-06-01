# Revision 2 Coverage and Omission Analysis Notebooks

## Purpose

These notebooks document the post-revision UCDB/GHS-POP coverage and omission
analysis used to support the revised coverage figure and omission tables in the
accepted Nature Cities manuscript. They were developed after the initial submission
in response to reviewer requests to quantify what share of the global urban
population in Africa, Asia, and Latin America and the Caribbean is covered by the
CSMD application.

Final outputs from these notebooks are committed to the repository under:
- `outputs/figures/revision2/` — revised coverage/omission figure
- `outputs/tables/revision2/` — final coverage and omission CSVs
- `outputs/tables/revision2/intermediate/` — intermediate city-level joins

## External Data Directory Convention

Large input files that cannot be committed to GitHub are placed under `data_external/`
at the repository root. This directory is excluded from version control (listed in
`.gitignore`) — you must create it and populate it manually before running the
notebooks.

Expected layout:

```
citysegmentdeprivation/
└── data_external/
    ├── ucdb/
    │   └── GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg
    ├── ghspop/
    │   └── GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0.tif
    └── zenodo/
        ├── GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2_with_GHSPOP2023.gpkg
        └── predictions/
            ├── afghanistan_rf_preds.gpkg
            ├── angola_rf_preds.gpkg
            └── ...  (one GPKG per country)
```

Outputs intended for GitHub are written to tracked folders:
- `outputs/tables/revision2/` — final CSVs
- `outputs/tables/revision2/intermediate/` — intermediate joins
- `outputs/figures/revision2/` — final figures (PDF + PNG)

These output directories are created automatically by the path-configuration
cell at the top of each notebook.

---

## Required External Inputs

The following inputs are needed to re-run these notebooks end-to-end. Large
GHSL/UCDB raster and vector files are not stored in GitHub; obtain them from
their official sources before running.

| Input | Local path under `data_external/` | Source | Used by |
|---|---|---|---|
| GHS Urban Centre Database 2019 V1.2 polygons (`GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg`) | `ucdb/` | JRC/GHSL official download | `01`, `02`, `03`, `04` |
| GHS-POP R2023A 2025 epoch raster, 100 m (`GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0.tif`) | `ghspop/` | JRC/GHSL official download | `01` |
| Derived UCDB + population GeoPackage (`GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2_with_GHSPOP2023.gpkg`) | `zenodo/` | Output of `01`; also on Zenodo | `02`, `03`, `04` |
| CSMD RF prediction GeoPackages (`*_rf_preds.gpkg`, one per country) | `zenodo/predictions/` | Zenodo (DOI: 10.5281/zenodo.20486977) | `02`, `04` |
| CSMD city-level summary table (`city_deprivation_80pct.csv`) | tracked in repo at `2_modelling/02_application/summary_statistics/` | — | `04` |

> **Important:** The derived file
> `GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2_with_GHSPOP2023.gpkg` should be stored
> on Zenodo, not in GitHub. It is too large for GitHub and is excluded by
> `.gitignore`. Place it under `data_external/zenodo/` for local re-runs.

See `docs/data_sources/ucdb.md` and `docs/data_sources/ghspop.md` for dataset
citations and download guidance.

---

## Notebook Execution Order

Run notebooks in numerical order. `99_*` is exploratory and is not required for
final reproduction.

### 1. `01_GHSPOP2023toUCDB2019.ipynb`

Adds GHS-POP 2025 population estimates to each UCDB 2019 city polygon by summing
100 m GHS-POP raster pixel values that fall within each polygon. The resulting
column is named `GHSPOP2023`.

**Output:** `GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2_with_GHSPOP2023.gpkg`
(large derived file — store on Zenodo, not GitHub)

**Note:** Computationally expensive because it operates on the 100 m global
GHS-POP raster. Run on a machine with sufficient RAM.

---

### 2. `02_revision2_omittedstatistics.ipynb`

Re-derives CSMD-covered city summaries with UCDB identifiers (`ID_HDC_G0`) and
Region L2 labels. Joins UCDB city population from the derived GPKG. Computes
region L1 / L2 / city-size-class omission statistics.

**Outputs:**
- `outputs/tables/revision2/intermediate/city_deprivation_80pct_with_ids_regionL2.csv`
- `outputs/tables/revision2/intermediate/city_deprivation_80pct_qc_with_ucdb_regions.csv`
- `outputs/tables/revision2/regionL1_regionL2_citysize_omission_80pct.csv`
- `outputs/tables/revision2/city_counts_ucdb_vs_citysegments_80pct.csv`
- `outputs/tables/revision2/city_stats_ucdb_vs_citysegments_80pct_counts_and_pop.csv`

---

### 3. `03_revision2_extendedtable2.ipynb`

Produces the country-level omission table used for Extended Data Table 2: 135
countries sorted by omitted population, with UCDB city count, CSMD-covered city
count, and omitted population share.

**Output:**
- `outputs/tables/revision2/country_level_omission_table.csv`

---

### 4. `04_globalsouthcomparison.ipynb`

Produces the revised coverage/omission figure (stacked bar chart by city-size
class and region) and the companion summary CSVs.

**Outputs:**
- `outputs/figures/revision2/population_totals_citysize_region_QC80.pdf`
- `outputs/figures/revision2/population_totals_citysize_region_QC80.png`
- `outputs/tables/revision2/city_size_comparison_QC80_Africa_Asia_LAC_millions.csv`

---

### 5. `99_GHSUCDB_OverlapCheck_exploratory.ipynb`

Exploratory notebook used to inspect UCDB–City Segments spatial overlap during
analysis development. **Not required** for final reproduction of manuscript outputs.

---

## Known Issues — To Resolve in Later Cleanup

The following issues were present in the notebooks as copied from the original
analysis folder. They should be fixed before final Zenodo deposit.

| Issue | Notebooks affected |
|---|---|
| Missing output figure `reg2_population_omission_3panel.pdf/png` — this file is referenced in notebook code but was not present in the source folder; needs to be regenerated or excluded from final documentation | `02` |
| The accepted/revised manuscript should be checked to confirm whether `population_totals_citysize_region_QC80` is the revised Figure 4 or an extended/supplementary figure | `04` |
