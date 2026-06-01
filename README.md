[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20486977.svg)](https://doi.org/10.5281/zenodo.20486977)

# City Segment Morphological Deprivation (CSMD) Model

This repository contains the analysis pipeline for the **City Segment Morphological Deprivation (CSMD)** model, associated with an accepted-in-principle paper at *Nature Cities*.

The CSMD model maps morphologically deprived city segments across cities in Africa, Asia, and Latin America and the Caribbean, using City Segments v1 (CIESIN), IDEABench v2 labels, and Random Forest modelling.

> **Repository status:** Accepted in principle at *Nature Cities*. The repository is being prepared for final publication. Large processed outputs (RF model, prediction GeoPackages, derived datasets) are hosted on Zenodo; raw source datasets remain with their original providers.

![Figure1_Overview_CSMD](https://github.com/user-attachments/assets/bf8eaca2-dc30-4c30-b48c-c3b6b8799b10)

---

## Workflow Overview

| Step | Description |
|---|---|
| A. Preprocessing & labelling | Standardise city-segment data; assign DUA-overlap training labels from IDEABench v2 |
| B. RF training & LOCO validation | VSURF variable selection; Random Forest training; leave-one-city-out validation |
| C. Global prediction / application | Apply final RF model to 5 000+ cities across the Global South |
| D. Comparative alignment | Contextual triangulation against SSI, Million Neighborhoods, and WRI Urban Land Use |
| E. Manuscript figures & tables | Generate all published figures and summary tables |
| F. Revision 2 coverage & omission analysis | UCDB / GHS-POP coverage quantification in response to reviewer requests |

---

## Repository Structure

```
citysegmentdeprivation/
├── 1_preprocessing/          # City-segment standardisation and label creation
├── 2_modelling/
│   ├── 01_training/          # VSURF, RF training, LOCO validation
│   └── 02_application/       # Global RF application and city-level summaries
├── 3_comparitive_analysis/   # SSI, MN, WRI comparison scripts and outputs
│                             #   (note: folder name retains historical spelling)
├── 4_Figures_Tables/         # Manuscript figure and summary-table notebooks
├── notebooks/
│   └── revision2_coverage/   # Post-revision UCDB/GHS-POP coverage notebooks
├── outputs/
│   ├── tables/revision2/     # Final and intermediate omission CSVs
│   └── figures/revision2/    # Revised coverage/omission figures
├── docs/                     # Extended documentation (model decisions, data sources, etc.)
├── environment/              # Conda environment, pip requirements, R packages
├── zenodo/                   # Zenodo deposit manifest
└── data/                     # Small committed reference files
```

### Key folders

**`1_preprocessing/`** — Prepares standardised city-segment features and benchmark-labelled training CSVs from IDEABench v2.

**`2_modelling/`** — RF training (`01_training/`) and global application to 5 000+ cities (`02_application/`). City-level summary statistics and the final city deprivation CSV are committed under `02_application/summary_statistics/`.

**`3_comparitive_analysis/`** — Contextual alignment with SSI, Million Neighborhoods (MN), and WRI Urban Land Use datasets. The folder name preserves the historical spelling used throughout the project.

**`4_Figures_Tables/`** — Notebooks that generate all manuscript figures and global summary tables from committed intermediate CSVs.

**`notebooks/revision2_coverage/`** — Five notebooks added in Revision 2 to quantify UCDB/GHS-POP coverage and omissions. See [`notebooks/revision2_coverage/README.md`](notebooks/revision2_coverage/README.md) for execution order and data requirements.

**`outputs/`** — Final committed outputs: figures (`outputs/figures/revision2/`) and tables (`outputs/tables/revision2/`). Intermediate joins are under `outputs/tables/revision2/intermediate/`.

**`docs/`** — Extended documentation including predictor definitions, model decisions, data-source citations, and a code-to-figure map.

**`environment/`** — Reproducible environment files (`environment.yml`, `requirements.txt`, `r_packages.R`).

**`zenodo/`** — Manifest of files deposited on Zenodo ([DOI: 10.5281/zenodo.20486977](https://doi.org/10.5281/zenodo.20486977)).

---

## Environment Setup

[Conda](https://docs.conda.io/) is recommended, particularly for geospatial dependencies (`geopandas`, `rasterio`, `fiona`).

```bash
conda env create -f environment/environment.yml
conda activate csmd
```

Optional pip fallback (may require manual installation of geospatial system libraries):

```bash
pip install -r environment/requirements.txt
```

R packages (for VSURF variable selection):

```r
Rscript environment/r_packages.R
```

---

## Key Model Details

| Property | Value |
|---|---|
| Training labels | IDEABench v2, 8 cities |
| Label rule | DUA spatial overlap ≥ 0.30 |
| Final predictors (8) | `i5_par_area`, `i1_pop_area`, `B_AVG_SEG`, `i9_roads_par`, `i6_paru_area`, `PARU_A_SEG`, `B_CV_SEG`, `REGION_CODE` / `REG1_GHSL` |
| Validation | Leave-one-city-out (LOCO) |
| Decision threshold | p(DUA) ≥ 0.40 |
| Comparative analyses | Contextual alignment / triangulation with SSI, MN, WRI — not strict ground-truth validation |

See [`docs/predictor_definitions.md`](docs/predictor_definitions.md) and [`docs/model_decisions.md`](docs/model_decisions.md) for full detail.

---

## Data Availability

| What | Where |
|---|---|
| Final CSVs, selected outputs, scripts, notebooks, documentation | This GitHub repository |
| RF model (`.joblib`), prediction GeoPackages, derived UCDB+GHS-POP GPKG, large comparison outputs | [Zenodo DOI: 10.5281/zenodo.20486977](https://doi.org/10.5281/zenodo.20486977) |
| City Segments v1 | CIESIN (original provider) |
| IDEABench v2 | IDEAtlas (original provider) |
| GHS Urban Centre Database (UCDB) 2019 V1.2 | JRC / GHSL official download |
| GHS-POP R2023A 2025 epoch, 100 m | JRC / GHSL official download |
| SSI, Million Neighborhoods, WRI Urban Land Use | Original providers |

Raw rasters, per-country prediction GeoPackages, model binaries, and other large files are excluded from the repository by `.gitignore`.

See [`docs/data_availability.md`](docs/data_availability.md) for full dataset citations and download links.

---

## Reproduction Guide

**Full raw-to-final reproduction** requires downloading external datasets from their original providers and running computationally intensive steps (rasterisation over the global 100 m GHS-POP grid, RF global application across 5 000+ cities). This is feasible but not a one-command process.

**Paper-output reproduction** is more accessible: download the Zenodo deposit, place processed files in the expected locations, and run the figure/table notebooks directly from committed intermediate CSVs.

**Revision 2 coverage notebooks** use a `data_external/` convention for large external inputs. See [`notebooks/revision2_coverage/README.md`](notebooks/revision2_coverage/README.md) for the directory layout and setup instructions.

### Key documentation

| Document | Contents |
|---|---|
| [`docs/code_figure_map.md`](docs/code_figure_map.md) | Maps each manuscript figure/table to the notebook or script that generates it |
| [`docs/model_decisions.md`](docs/model_decisions.md) | Rationale for modelling choices (threshold, predictors, validation) |
| [`docs/predictor_definitions.md`](docs/predictor_definitions.md) | Definition and units of all RF predictors |
| [`docs/data_availability.md`](docs/data_availability.md) | Dataset citations, download URLs, and licence notes |
| [`docs/coverage_and_omissions.md`](docs/coverage_and_omissions.md) | Coverage and omission methodology for the Revision 2 analysis |
| [`notebooks/revision2_coverage/README.md`](notebooks/revision2_coverage/README.md) | Execution order and data requirements for coverage notebooks |

---

## Citation

If you use or build upon this work, please cite the Zenodo deposit and, once published, the journal article.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20486977.svg)](https://doi.org/10.5281/zenodo.20486977)

**Data deposit:** Veeravalli, S. G. (2026). *A Global, Standardized City Segment Morphological Deprivation (CSMD) Model: Preprocessing, Training, Predictions, and Cross-Dataset Comparisons* (Version v3) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.20486977

**Paper:** _Citation will be added after publication._ Currently accepted in principle at *Nature Cities*.

---

## Acknowledgements

This work is supported by:

- **FORMAS** (Swedish Research Council for Sustainable Development), project DEPRIMAP (2023-01210) — <https://sola.kau.se/deprimap/>
- **NAISS** (National Academic Infrastructure for Supercomputing in Sweden), partially funded by the Swedish Research Council through grant agreement no. 2022-06725 — computation for model training
- **CIESIN** for City Segments v1 and **IDEAtlas** for IDEABench v2

<img width="373" height="110" alt="image" src="https://github.com/user-attachments/assets/a180a6e3-1b60-429d-b0b8-c14a45e4e190" />
