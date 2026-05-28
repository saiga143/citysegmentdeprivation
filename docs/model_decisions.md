# Modelling and Analysis Decisions

This document records the principal design decisions made during the CSMD
modelling pipeline. Where a decision has a fuller justification in the manuscript
or supplementary text, that is noted explicitly.

---

## 1. Label Construction

### IDEABench-to-City Segments aggregation

Training labels are derived by spatially overlaying the IDEABench v2 Deprived
Urban Areas (DUA) polygons onto City Segments v1 street-block segments. For each
segment the fraction of its area that intersects a DUA polygon is computed
(`slum_fraction`). This is done separately for each of the 8 IDEABench cities.

### Deprivation threshold: slum_fraction ≥ 0.30

A segment is labelled as deprived (`slum_label1 = 1`) if its DUA overlap fraction
is at least 0.30 (30%). Segments with `slum_fraction < 0.30` are labelled
non-deprived (`slum_label1 = 0`). The threshold was chosen to balance sensitivity
and label precision. Rationale should be verified against the manuscript.

### Non-built-up segments

Segments without buildings or parcels, and open-space segments that fall outside
DUA polygons, are grouped with the non-deprived class. This decision reflects the
scope of the CSMD model (built urban morphology) rather than a judgment that
non-built areas are formally non-deprived. Rationale should be verified against
the manuscript.

---

## 2. Feature Selection

### VSURF with three ntree settings

VSURF (Variable Selection Using Random Forests) was run three times — with
`ntree = 800`, `1000`, and `1200` — to assess stability of variable selection
across the 21 candidate morphological predictors plus the region category.
`set.seed(42)` was used for reproducibility. See `run_VSURF.R`.

### Final 8-predictor set

The variables retained consistently across ntree settings by VSURF's
interpretation-step selection are:

- `i5_par_area` (mean parcel area)
- `i1_pop_area` (population density)
- `i6_paru_area` (mean unbuilt parcel area)
- `B_AVG_SEG` (mean building footprint area)
- `i9_roads_par` (road length per parcel)
- `PARU_A_SEG` (total unbuilt parcel area)
- `B_CV_SEG` (building footprint area coefficient of variation)
- `REGION_CODE` (encoded region category)

See `docs/predictor_definitions.md` for full variable descriptions.

### Inclusion of the regional category

`REGION_CODE` (encoded from `REG1_GHSL`) was included as a predictor to allow
the model to partially account for regional differences in urban morphology that
are not captured by the segment-level built-environment indicators. The encoding
is: Unknown = 0, Asia = 1, Africa = 2, LAC = 3. This categorical is treated as a
numeric input by the Random Forest. Rationale should be verified against the
manuscript.

---

## 3. Model Training

### LOCO validation design

Leave-One-City-Out (LOCO) cross-validation was used to estimate generalisation
performance: in each fold, the model is trained on 7 cities and tested on the
held-out city. This is done independently for each of the 8 IDEABench cities.
LOCO is used for *evaluation only*; the final production model is trained on all
8 cities. See `Validation_rf_model_loco.py`.

### HalvingRandomSearchCV hyperparameter tuning

Hyperparameter tuning used scikit-learn's `HalvingRandomSearchCV` with:
- Objective: ROC-AUC (`scoring="roc_auc"`)
- CV folds: 5 (`n_splits=5`)
- Resource range: `min_resources=150`, `max_resources=2000`, `factor=3`
- Random state: 42

Tuning was run on the full 8-city labelled dataset (no holdout set — evaluation
was handled separately via LOCO). See `train_rf_full.py`.

### Final production model hyperparameters

| Parameter | Value |
|---|---|
| `n_estimators` | 1350 |
| `max_depth` | 22 |
| `min_samples_leaf` | 8 |
| `min_samples_split` | 6 |
| `max_features` | `"sqrt"` |
| `bootstrap` | `True` |
| `class_weight` | `"balanced_subsample"` |
| `random_state` | 42 |

`class_weight = "balanced_subsample"` was used to handle class imbalance
(positive rate ≈ 13.2% in the training set). Parameters are stored in
`2_modelling/01_training/rf_outputs_full/tables/best_params_full.json`.

---

## 4. Prediction and Thresholding

### Probability outputs vs. binary labels

The trained model outputs a continuous probability `p(DUA)` for each segment.
This probability is the primary output and is used directly in comparative
analyses (e.g. SSI correlation) and stored in prediction files.

### Binary label threshold: p(DUA) ≥ 0.40

Segments are assigned a binary deprived label (`rf_label = 1`) if their predicted
probability meets or exceeds 0.40. This threshold is lower than the default 0.50
to improve recall on the minority (deprived) class given the training imbalance.
The threshold is applied in `01_apply_rf_predictions.ipynb` and in
`Validation_rf_model_loco.py`. Rationale should be verified against the
manuscript.

---

## 5. Application and Coverage

### 80% QC filter

A city is included in the published CSMD summaries only if ≥ 80% of its
street-block segments have a valid `rf_label` and `POP_SEG` value. Cities that
fail this filter are excluded. This threshold guards against spurious
city-level summaries where a large fraction of blocks could not be processed.
Rationale should be verified against the manuscript.

### City-size class definitions

City-size classes follow UN World Urbanisation Prospects 2018 thresholds:

| Class | Population threshold |
|---|---|
| Small | < 500,000 |
| Medium | 500,000 – 1,000,000 |
| Large | 1,000,000 – 5,000,000 |
| Very Large | 5,000,000 – 10,000,000 |
| Megacity | ≥ 10,000,000 |

### Difference between CSMD city-size summaries and revision2 coverage analysis

The original CSMD summary files (in `2_modelling/02_application/summary_statistics/`)
use **segment-aggregated CSMD population** (`TotalPop`) to classify cities by size.

The revision2 coverage and omission analysis (in `outputs/tables/revision2/`) uses
**UCDB/GHS-POP city population** (`GHSPOP2023`, derived from GHS-POP R2023A) to
classify cities. This difference means a small number of cities may fall into
different size classes depending on which population source is used. See
`docs/coverage_and_omissions.md` for more detail.

---

## 6. Comparative Analysis

### SSI, MN, and WRI comparisons as alignment/triangulation, not strict validation

The comparisons against the Slum Severity Index (SSI), Million Neighborhoods (MN),
and WRI Urban Land Use (`p_informal`) are designed as external alignment checks
and triangulation exercises — not as strict ground-truth validation of the CSMD
model. No single external dataset provides the same conceptual scope or spatial
coverage as the CSMD labels, so agreement statistics should be interpreted in
that context. Rationale and interpretation should be verified against the
manuscript.
