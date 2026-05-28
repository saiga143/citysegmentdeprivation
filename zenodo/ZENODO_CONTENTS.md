# Zenodo Deposit Contents

This file tracks what is planned for the Zenodo deposit alongside this repository.
Zenodo DOI: 10.5281/zenodo.18788260

---

## Revision2 coverage and omission analysis

The following files from the revision2 coverage analysis are planned for Zenodo
deposit. They are either too large for GitHub or are derived intermediates that
support reproducibility.

### Large derived intermediate (Zenodo only)

- `GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2_with_GHSPOP2023.gpkg`
  UCDB 2019 V1.2 polygon file with GHS-POP 2025 population estimates added per
  city (`GHSPOP2023` column). Produced by `GHSPOP2023toUCDB2019.ipynb`. Not
  stored in GitHub due to file size.

### Final CSV outputs (also in GitHub at `outputs/tables/revision2/`)

- `city_size_comparison_QC80_Africa_Asia_LAC_millions.csv`
  City counts and population by city-size class: UCDB universe vs. CSMD-covered.

- `city_counts_ucdb_vs_citysegments_80pct.csv`
  City counts by region × city-size class: UCDB vs. CSMD-covered vs. omitted.

- `city_stats_ucdb_vs_citysegments_80pct_counts_and_pop.csv`
  City counts and population in millions by region × city-size class.

- `regionL1_regionL2_citysize_omission_80pct.csv`
  Omitted population by Region L1, Region L2, and city-size class (62 rows).

- `country_level_omission_table.csv`
  Omitted population by country, sorted descending (135 countries). Source for
  Extended Data Table 2.

### Intermediate CSVs (also in GitHub at `outputs/tables/revision2/intermediate/`)

- `city_deprivation_80pct_with_ids_regionL2.csv`
  CSMD city summary with UCDB `ID_HDC_G0` identifiers and Region L2 label added.

- `city_deprivation_80pct_qc_with_ucdb_regions.csv`
  CSMD city summary further joined to UCDB region fields, population, and
  city-size class.

### Revised coverage figure (also in GitHub at `outputs/figures/revision2/`)

- `population_totals_citysize_region_QC80.pdf`
- `population_totals_citysize_region_QC80.png`
  Stacked bar chart of CSMD-covered vs. omitted population by city-size class and
  region. Supports the revised Figure 4 / coverage-omission figure.

---

*Additional Zenodo content (code, model outputs, labeled training data) will be
documented here in subsequent steps.*
