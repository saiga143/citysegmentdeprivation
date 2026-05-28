# Predictor and Variable Definitions

This document defines all City Segments variables used in the CSMD modelling
pipeline, including the 21 candidates passed to VSURF feature selection and the
region category. Variables marked **Final model** are among the 8 predictors
retained in the production Random Forest.

Variable domains:
- **Raw segment** — direct measurements from the City Segments v1 dataset
- **Derived index** — ratios computed from raw segment variables during preprocessing
- **Categorical** — region encoding derived from the GHS-UCDB `REG1_GHSL` field

---

## Raw Segment Variables

| Variable | Domain | Definition | Unit | Formula | Final model |
|---|---|---|---|---|---|
| `POP_SEG` | Raw segment | Ambient population count within the segment | Persons | — | No |
| `AREAHA_SEG` | Raw segment | Segment area | Hectares | — | No |
| `ROAD_SEG` | Raw segment | Total road centreline length within the segment | Metres | — | No |
| `PAR_N_SEG` | Raw segment | Number of land parcels within the segment | Count | — | No |
| `PARU_N_SEG` | Raw segment | Number of unbuilt (vacant / informal) parcels within the segment | Count | — | No |
| `PARU_P_SEG` | Raw segment | Proportion of parcels that are unbuilt | Fraction [0, 1] | `PARU_N_SEG / PAR_N_SEG` | No |
| `PARU_A_SEG` | Raw segment | Total area of unbuilt parcels within the segment | m² | — | **Yes** |
| `PAR_CV_SEG` | Raw segment | Coefficient of variation of parcel area within the segment | Dimensionless | — | No |
| `B_AREA_SEG` | Raw segment | Total building footprint area within the segment | m² | — | No |
| `B_AVG_SEG` | Raw segment | Mean building footprint area within the segment | m² | — | **Yes** |
| `B_CV_SEG` | Raw segment | Coefficient of variation of building footprint area within the segment | Dimensionless | — | **Yes** |

---

## Derived Index Variables

These indices are computed from raw segment variables during preprocessing
(`1_preprocessing/01_preprocess_city_segments.ipynb`). All divisions are guarded
against zero denominators in the preprocessing code.

| Variable | Domain | Definition | Unit | Formula | Final model |
|---|---|---|---|---|---|
| `i1_pop_area` | Derived index | Population density | Persons / ha | `POP_SEG / AREAHA_SEG` | **Yes** |
| `i2_pop_par` | Derived index | Population per parcel | Persons / parcel | `POP_SEG / PAR_N_SEG` | No |
| `i3_pop_paru` | Derived index | Population per unbuilt parcel | Persons / unbuilt parcel | `POP_SEG / PARU_N_SEG` | No |
| `i4_pop_roads` | Derived index | Population per unit road length | Persons / m | `POP_SEG / ROAD_SEG` | No |
| `i5_par_area` | Derived index | Mean parcel area | m² / parcel | `AREAHA_SEG / PAR_N_SEG × 10 000` | **Yes** |
| `i6_paru_area` | Derived index | Mean unbuilt parcel area | m² / parcel | `AREAHA_SEG / PARU_N_SEG × 10 000` | **Yes** |
| `i7_roads_area` | Derived index | Road density | m / ha | `ROAD_SEG / AREAHA_SEG` | No |
| `i8_paru_par` | Derived index | Unbuilt parcel fraction | Fraction [0, 1] | `PARU_N_SEG / PAR_N_SEG` | No |
| `i9_roads_par` | Derived index | Road length per parcel | m / parcel | `ROAD_SEG / PAR_N_SEG` | **Yes** |
| `i10_roads_paru` | Derived index | Road length per unbuilt parcel | m / unbuilt parcel | `ROAD_SEG / PARU_N_SEG` | No |

---

## Categorical Variable

| Variable | Domain | Definition | Encoding | Final model |
|---|---|---|---|---|
| `REG1_GHSL` / `REGION_CODE` | Categorical | Broad world region derived from the GHS-UCDB `REG1_GHSL` field. Encoded as an integer before model training. | Unknown = 0, Asia = 1, Africa = 2, LAC = 3 | **Yes** |

The encoding is applied in `train_rf_full.py` and stored in
`2_modelling/01_training/rf_outputs_full/region_mapping.json`.

---

## Summary: Final Model Predictors

The 8 predictors retained after VSURF feature selection, in the order listed in
`train_rf_full.py`:

1. `i5_par_area` — mean parcel area
2. `i1_pop_area` — population density
3. `i6_paru_area` — mean unbuilt parcel area
4. `B_AVG_SEG` — mean building area
5. `i9_roads_par` — road length per parcel
6. `PARU_A_SEG` — total unbuilt parcel area
7. `B_CV_SEG` — building area coefficient of variation
8. `REGION_CODE` — region category (encoded `REG1_GHSL`)
