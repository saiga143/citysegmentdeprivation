#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train FINAL Random Forest model on 100% of labeled data (8 IDEABench cities),
with Successive Halving hyperparameter tuning.

Purpose
-------
- LOCO is used for *evaluation* (already computed separately).
- This script trains the *final deployed model* on all labeled data to maximize
  learning before applying it to large-scale inference (5,000+ cities).

What it does
------------
- Loads labeled CSVs from --input-folder
- Uses slum_label1 as target
- Encodes REG1_GHSL -> REGION_CODE
- Runs HalvingRandomSearchCV on full dataset (no holdout)
- Refits best estimator on full dataset (refit=True)
- Saves:
    * best model (joblib)
    * best params (json)
    * full CV results (csv)
    * feature importance (csv + png)
    * full-data predictions (proba + label at --threshold) (csv)
    * reproducibility artifacts (file list, region mapping, warnings)

Run example1 (from repo root):
    python 2_modelling/01_training/train_rf_full.py \
        --input-folder 1_preprocessing/LabelledData_For_RF \
        --output-dir 2_modelling/01_training/rf_outputs_full \
        --threshold 0.4

Run example2 (from repo root):
    python 2_modelling/01_training/train_rf_full.py \
        --input-folder 1_preprocessing/LabelledData_For_RF \
        --output-dir 2_modelling/01_training/rf_outputs_full \
        --threshold 0.4 \
        --exclude-region


Notes
-----
- This script does NOT report model "performance" because training uses 100% data.
  Performance is be reported via LOCO results.
"""

import os
import json
import time
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# Enable Successive Halving
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import randint

from sklearn.utils.validation import check_is_fitted


# ──────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Train FINAL RF on 100% labeled data with HalvingRandomSearchCV."
    )
    p.add_argument(
        "--input-folder",
        type=str,
        required=False,
        default="1_preprocessing/LabelledData_For_RF",
        help="Folder with labeled CSVs (e.g., *_labeled_thr030.csv), relative to repo root.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default="2_modelling/01_training/rf_outputs_full",
        help="Output directory (relative to repo root).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        required=False,
        default=0.4,
        help="Probability threshold for saving full-data class labels (not for performance reporting).",
    )
    p.add_argument(
        "--exclude-region",
        action="store_true",
        help="If set, REGION_CODE is computed but excluded from predictors (ablation-style final model).",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────
def get_outer_jobs() -> int:
    cpu_env = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpu_env is not None:
        try:
            n = int(cpu_env)
        except ValueError:
            n = os.cpu_count() or 1
    else:
        n = os.cpu_count() or 1
    return max(1, n)


def safe_mkdirs(output_dir: Path):
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    logs_dir = output_dir / "logs"
    models_dir = output_dir / "models"

    for d in [tables_dir, plots_dir, logs_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return tables_dir, plots_dir, logs_dir, models_dir


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    input_folder = Path(args.input_folder)
    output_dir = Path(args.output_dir)
    threshold = float(args.threshold)
    exclude_region = bool(args.exclude_region)

    tables_dir, plots_dir, logs_dir, models_dir = safe_mkdirs(output_dir)

    print(f"📁 Input folder:  {input_folder.resolve()}")
    print(f"📁 Output folder: {output_dir.resolve()}")
    print(f"🎯 Threshold (for saved labels): {threshold}")
    print(f"🧩 Exclude REGION_CODE from predictors: {exclude_region}")

    # ──────────────────────
    # Fixed region mapping (persisted)
    # ──────────────────────
    REGION_MAP = {
        "Unknown": 0,
        "Asia": 1,
        "Africa": 2,
        "Latin America and the Caribbean": 3,
    }
    REGION_COL = "REG1_GHSL"
    REGION_MAP_PATH = output_dir / "region_mapping.json"
    with REGION_MAP_PATH.open("w", encoding="utf-8") as f:
        json.dump(REGION_MAP, f, ensure_ascii=False, indent=2)

    # ──────────────────────
    # Schema
    # ──────────────────────
    target_col = "slum_label1"

    base_predictors = [
        "i5_par_area", "i1_pop_area", "i6_paru_area", "B_AVG_SEG",
        "i9_roads_par", "PARU_A_SEG", "B_CV_SEG",
    ]
    predictor_cols = list(base_predictors)
    if not exclude_region:
        predictor_cols.append("REGION_CODE")

    # For loading, we may still require REGION_COL even if exclude_region=True,
    # because we compute REGION_CODE for logging/diagnostics.
    required_cols_base = set([target_col, REGION_COL]) | set(base_predictors)

    # ──────────────────────
    # Load labeled CSVs
    # ──────────────────────
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    csv_files = sorted([p for p in input_folder.iterdir() if p.suffix.lower() == ".csv"])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_folder}")

    # deterministic file list
    pd.DataFrame({"file": [p.name for p in csv_files]}).to_csv(
        tables_dir / "file_list.csv", index=False
    )

    def _map_region(val):
        if pd.isna(val):
            return REGION_MAP["Unknown"]
        return REGION_MAP.get(str(val), REGION_MAP["Unknown"])

    dfs = []
    missing_schema = []
    region_value_issues = []

    for file_path in csv_files:
        df = pd.read_csv(file_path)

        missing = [c for c in required_cols_base if c not in df.columns]
        if missing:
            missing_schema.append({"file": file_path.name, "missing_cols": missing})
            continue

        df["REGION_CODE"] = df[REGION_COL].map(_map_region)

        unseen = set(df[REGION_COL].dropna().unique()) - set(REGION_MAP.keys())
        if unseen:
            region_value_issues.append({"file": file_path.name, "unseen_values": sorted(list(unseen))})

        # provenance
        df["SRC_FILE"] = file_path.name
        df["ROW_IN_FILE"] = np.arange(len(df), dtype=np.int64)

        # keep columns: predictors (base + maybe region), target, provenance, and REGION_CODE always kept for logging
        keep_cols = list(base_predictors) + [target_col, "REGION_CODE", "SRC_FILE", "ROW_IN_FILE"]
        df = df[keep_cols]
        dfs.append(df)

    if missing_schema:
        pd.DataFrame(missing_schema).to_csv(logs_dir / "missing_schema_files.csv", index=False)
    if region_value_issues:
        pd.DataFrame(region_value_issues).to_csv(logs_dir / "unseen_region_values.csv", index=False)

    if not dfs:
        raise RuntimeError("No valid CSVs after schema checks. See logs/.")

    full_data = pd.concat(dfs, ignore_index=True)

    # Drop NA across predictors + target (note: REGION_CODE may be excluded from predictors)
    clean_data = full_data.dropna(subset=base_predictors + [target_col]).reset_index(drop=True)
    if not exclude_region:
        clean_data = clean_data.dropna(subset=["REGION_CODE"]).reset_index(drop=True)

    print(f"✅ Loaded and cleaned data: {clean_data.shape}")

    # distribution tables
    clean_data.groupby(["REGION_CODE", target_col]).size().unstack(fill_value=0).to_csv(
        tables_dir / "region_class_distribution.csv"
    )

    # matrices
    X = clean_data[predictor_cols].to_numpy(dtype=float)
    y = clean_data[target_col].to_numpy(dtype=int)

    print(f"🧮 X shape: {X.shape} | y positives: {int(y.sum())}/{len(y)} ({y.mean():.3f})")
    pd.DataFrame({"predictor": predictor_cols}).to_csv(tables_dir / "predictor_list.csv", index=False)

    # ──────────────────────
    # HalvingRandomSearchCV on FULL dataset
    # ──────────────────────
    outer_jobs = get_outer_jobs()
    print(f"🧵 Using n_jobs={outer_jobs} for HalvingRandomSearchCV")

    rf_base = RandomForestClassifier(random_state=42, n_jobs=1, oob_score=False)

    param_dist = {
        "max_depth": randint(10, 25),
        "min_samples_leaf": randint(1, 11),
        "min_samples_split": randint(2, 12),
        "max_features": ["sqrt", 0.5],
        "bootstrap": [True, False],
        "class_weight": ["balanced", "balanced_subsample"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Save CV fold indices for reproducibility/diagnostics
    folds_payload = []
    for f_id, (tr, va) in enumerate(cv.split(X, y)):
        folds_payload.append({"fold": int(f_id), "train_idx": tr.tolist(), "val_idx": va.tolist()})
    with (tables_dir / "cv_fold_indices.json").open("w", encoding="utf-8") as f:
        json.dump(folds_payload, f)

    search = HalvingRandomSearchCV(
        estimator=rf_base,
        param_distributions=param_dist,
        resource="n_estimators",
        min_resources=150,
        max_resources=2000,
        factor=3,
        scoring="roc_auc",
        cv=cv,
        n_jobs=outer_jobs,
        random_state=42,
        verbose=2,
        refit=True,  # refit best on full data after search
    )

    t0 = time.time()
    search.fit(X, y)
    elapsed_min = (time.time() - t0) / 60.0
    print(f"✅ Halving search completed in {elapsed_min:.2f} minutes")

    # Save CV results
    cvres = pd.DataFrame(search.cv_results_)
    cvres.to_csv(tables_dir / "halving_cv_results_full.csv", index=False)

    # Best params
    best_params = search.best_params_
    best_n_trees = int(search.cv_results_["param_n_estimators"][search.best_index_])
    best_model = search.best_estimator_
    actual_fit_trees = getattr(best_model, "n_estimators", None)

    best_payload = {
        "predictors_used": predictor_cols,
        "exclude_region": exclude_region,
        "threshold_saved_labels": threshold,
        "best_params": best_params,
        "best_n_estimators_cv": best_n_trees,
        "best_n_estimators_refit": actual_fit_trees,
        "n_rows": int(len(clean_data)),
        "pos_rate": float(y.mean()),
        "random_state": 42,
    }

    with (tables_dir / "best_params_full.json").open("w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)

    print("✅ Best Params:", best_params)
    print("🌲 n_estimators (CV):", best_n_trees, "| n_estimators (refit):", actual_fit_trees)

    # Save final model
    model_name = "rf_final_model_full_noreg.joblib" if exclude_region else "rf_final_model_full.joblib"
    joblib.dump(best_model, models_dir / model_name)
    print(f"💾 Saved final model: {models_dir / model_name}")

    # ──────────────────────
    # Feature importance
    # ──────────────────────
    try:
        check_is_fitted(best_model)
        importances = best_model.feature_importances_
        feat_df = (
            pd.DataFrame({"feature": predictor_cols, "importance": importances})
            .sort_values("importance", ascending=False)
        )
        feat_df.to_csv(tables_dir / "feature_importance_full.csv", index=False)

        plt.figure(figsize=(8, 5))
        sns.barplot(data=feat_df, x="importance", y="feature")
        plt.title("Feature Importance (Final Full-Data Model)")
        plt.tight_layout()
        plt.savefig(plots_dir / "feature_importance_full.png", dpi=200)
        plt.close()
    except Exception as e:
        with (logs_dir / "warnings.txt").open("a", encoding="utf-8") as f:
            f.write(f"[FeatureImportance] {repr(e)}\n")

    # ──────────────────────
    # Save full-data predictions (for diagnostics / thresholding)
    # NOTE: not used for performance reporting
    # ──────────────────────
    y_proba = best_model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    preds = clean_data[["SRC_FILE", "ROW_IN_FILE", "REGION_CODE", target_col]].copy()
    preds["y_proba"] = y_proba
    preds["y_pred_thr"] = y_pred
    preds["threshold"] = threshold
    preds.to_csv(tables_dir / "full_data_predictions.csv", index=False)

    # quick label counts
    counts = {
        "n_total": int(len(y)),
        "n_pos_true": int(y.sum()),
        "pos_rate_true": float(y.mean()),
        "n_pos_pred_thr": int(y_pred.sum()),
        "pos_rate_pred_thr": float(y_pred.mean()),
        "threshold": threshold,
    }
    pd.DataFrame([counts]).to_csv(tables_dir / "full_data_prediction_counts.csv", index=False)

    print("✅ Saved full-data predictions + counts.")
    print("✅ Done. Outputs saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
