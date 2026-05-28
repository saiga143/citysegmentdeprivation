# Coverage and Omissions Analysis

This document describes the revision2 coverage analysis that compares CSMD-covered
cities against the GHS Urban Centre Database (UCDB) 2019 reference universe.

## Purpose

The analysis quantifies what share of the global urban population in Africa, Asia,
and Latin America and the Caribbean (LAC) is covered by the CSMD application, and
characterises the cities and population that are omitted.

## Population denominator

GHS-POP R2023A (2025 epoch) raster data are used as the population denominator.
Population estimates are assigned to UCDB 2019 city polygons by summing GHS-POP
pixel values that fall within each polygon. The resulting column is named
`GHSPOP2023` in the derived UCDB file.

## Relationship to manuscript figures and tables

- The coverage figure (`outputs/figures/revision2/population_totals_citysize_region_QC80.*`)
  supports the revised Figure 4 / coverage-omission figure in the manuscript.
- `outputs/tables/revision2/country_level_omission_table.csv` is the source for
  Extended Data Table 2.
- The remaining CSVs in `outputs/tables/revision2/` provide regional and city-size
  breakdowns reported in the manuscript text.

## City-size classification

City-size classes in this coverage analysis are derived from the UCDB/GHS-POP city
population (`GHSPOP2023`), not from the segment-aggregated `TotalPop` column used
in earlier CSMD summary tables. The two approaches may assign a small number of
cities to different size classes.

Size classes follow UN WUP 2018 thresholds:
- Small: < 500,000
- Medium: 500,000 – 1,000,000
- Large: 1,000,000 – 5,000,000
- Very Large: 5,000,000 – 10,000,000
- Megacity: ≥ 10,000,000

## QC filter

Only CSMD-covered cities where ≥ 80% of street-block segments have a valid `rf_label`
and `POP_SEG` value are included in coverage comparisons (the "QC-80" filter).

## Large inputs not stored in GitHub

The following large input files are not stored in this repository:

- `GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0/` — GHS-POP 100 m raster
- `GHS_POP_E2025_GLOBE_R2023A_54009_1000_V1_0/` — GHS-POP 1 km raster
- `GHS_STAT_UCDB2015MT_GLOBE_R2019A/` — raw UCDB 2019 V1.2 files

These should be obtained from their official JRC/GHSL sources. See
`docs/data_sources/ucdb.md` and `docs/data_sources/ghspop.md` for details.

## Derived intermediate file on Zenodo

`GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2_with_GHSPOP2023.gpkg` — the UCDB polygon
file with GHS-POP 2025 population added per city — is not stored in GitHub due to
file size. It will be deposited on Zenodo alongside the code and final outputs.

## Notebook execution order

The exact notebook execution order and path corrections will be documented in a
subsequent step once all paths have been made relative.
