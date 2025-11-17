#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Random Forest training with Successive Halving (VSURF-selected predictors).

- Loads labeled training CSVs (8 IDEABench cities).
- Uses slum_label1 as the target.
- Encodes REG1_GHSL into a numeric REGION_CODE.
- Splits a true 80/20 holdout with saved indices.
- Runs HalvingRandomSearchCV over RF hyperparameters.
- Saves:
    * best model
    * CV results
    * feature importance
    * holdout metrics
    * ROC curve

Inputs (per manuscript pipeline):
    - Labeled CSVs *_labeled_thr030.csv produced in preprocessing step.

Outputs (for GitHub repo):
    - All outputs are saved under the directory passed via --output-dir.
    - In the repo, we recommend: 2_modelling/01_training/rf_outputs/

Note:
    The full-scale computation was originally run on the NAISS HPC cluster
    (National Academic Infrastructure for Supercomputing in Sweden),
    partially funded by the Swedish Research Council (grant no. 2022-06725).
    This script, however, runs on any standard Python environment
    with sufficient memory/cores.

Run example (from repo root):
    python 2_modelling/01_training/train_rf_model.py \
        --input-folder 1_preprocessing/LabelledData_For_RF \
        --output-dir 2_modelling/01_training/rf_outputs
"""

import os
import json
import argparse
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
from sklearn.utils.validation import check_is_fitted

# Enable Successive Halving
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import randint


# ──────────────────────
# Argument Parser
# ──────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Random Forest model on labeled City Segments data."
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        required=False,
        default="1_preprocessing/LabelledData_For_RF",
        help="Folder with *_labeled_thr030.csv files "
             "(relative to current working directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default="2_modelling/01_training/rf_outputs",
        help="Folder where all RF outputs (tables/plots/model) will be saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_folder = Path(args.input_folder)
    output_dir = Path(args.output_dir)

    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    logs_dir = output_dir / "logs"

    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Input folder:  {input_folder.resolve()}")
    print(f"📁 Output folder: {output_dir.resolve()}")

    # ──────────────────────
    # Fixed region mapping (persisted for reproducibility)
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
    # Feature schema
    # ──────────────────────
    target_col = "slum_label1"
    predictor_cols = [  ## these are the outputs (predictor variables consistent across three ntree runs) from VSURF step.
        "i5_par_area", "i1_pop_area", "i6_paru_area", "i8_paru_par", "B_AVG_SEG",
        "i9_roads_par", "PARU_A_SEG", "B_AREA_SEG", "B_CV_SEG",
        "REGION_CODE",  # numeric code of region
    ]

    # ──────────────────────
    # Load & validate data (DETERMINISTIC ORDER + PROVENANCE)
    # ──────────────────────
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    csv_files = sorted([f for f in input_folder.iterdir() if f.suffix == ".csv"])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_folder}")

    # Log exact file order for reproducibility
    pd.DataFrame({"file": [p.name for p in csv_files]}).to_csv(
        tables_dir / "file_list.csv", index=False
    )

    dfs = []
    missing_schema = []
    region_value_issues = []

    required_non_region = [c for c in predictor_cols if c != "REGION_CODE"]
    required_cols_base = set([target_col, REGION_COL]) | set(required_non_region)

    for file_path in csv_files:
        df = pd.read_csv(file_path)

        # schema check
        missing = [c for c in required_cols_base if c not in df.columns]
        if missing:
            missing_schema.append({"file": file_path.name, "missing_cols": missing})
            continue

        # region map
        def _map_region(val):
            if pd.isna(val):
                return REGION_MAP["Unknown"]
            return REGION_MAP.get(str(val), REGION_MAP["Unknown"])

        df["REGION_CODE"] = df[REGION_COL].map(_map_region)

        unseen = set(df[REGION_COL].dropna().unique()) - set(REGION_MAP.keys())
        if unseen:
            region_value_issues.append(
                {"file": file_path.name, "unseen_values": sorted(list(unseen))}
            )

        # provenance columns
        df["SRC_FILE"] = file_path.name
        df["ROW_IN_FILE"] = np.arange(len(df), dtype=np.int64)

        # keep predictors + target + provenance
        keep_cols = [
            c for c in predictor_cols if c != "REGION_CODE"
        ] + [target_col, "REGION_CODE", "SRC_FILE", "ROW_IN_FILE"]
        df = df[keep_cols]
        dfs.append(df)

    # log issues
    if missing_schema:
        pd.DataFrame(missing_schema).to_csv(
            logs_dir / "missing_schema_files.csv", index=False
        )
    if region_value_issues:
        pd.DataFrame(region_value_issues).to_csv(
            logs_dir / "unseen_region_values.csv", index=False
        )

    if not dfs:
        raise RuntimeError(
            "No valid dataframes after schema checks. See logs for details."
        )

    full_data = pd.concat(dfs, ignore_index=True)

    # Drop NA across predictors + target
    clean_data = full_data.dropna(subset=predictor_cols + [target_col]).reset_index(
        drop=True
    )
    print(f"✅ Loaded and cleaned data: {clean_data.shape}")

    # Class/region distribution
    (
        clean_data.groupby(["REGION_CODE", target_col])
        .size()
        .unstack(fill_value=0)
        .rename_axis(index="REGION_CODE")
    ).to_csv(tables_dir / "region_class_distribution.csv")

    # Build matrices
    X = clean_data[predictor_cols].to_numpy(dtype=float)
    y = clean_data[target_col].to_numpy()
    n = len(clean_data)

    # ──────────────────────
    # True holdout split with EXPLICIT INDICES (saved to disk)
    # ──────────────────────
    all_idx = np.arange(n, dtype=np.int64)
    train_idx, test_idx = train_test_split(
        all_idx, stratify=y, test_size=0.20, random_state=42
    )

    # Save holdout indices with provenance for exact reproducibility
    holdout_df = clean_data.loc[
        test_idx, ["SRC_FILE", "ROW_IN_FILE", "REGION_CODE", target_col]
    ].copy()
    holdout_df.insert(0, "global_index", test_idx)
    holdout_df["is_test"] = 1

    not_holdout_df = clean_data.loc[
        train_idx, ["SRC_FILE", "ROW_IN_FILE", "REGION_CODE", target_col]
    ].copy()
    not_holdout_df.insert(0, "global_index", train_idx)
    not_holdout_df["is_test"] = 0

    (
        pd.concat([not_holdout_df, holdout_df], ignore_index=True)
        .sort_values("global_index")
        .to_csv(tables_dir / "holdout_indices.csv", index=False)
    )

    # Slice arrays by indices
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ──────────────────────
    # Successive Halving Search (no nested parallelism issues)
    # ──────────────────────
    # Use available CPU cores; on HPC, SLURM_CPUS_PER_TASK can still be set.
    cpu_env = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpu_env is not None:
        try:
            outer_jobs = int(cpu_env)
        except ValueError:
            outer_jobs = os.cpu_count() or 1
    else:
        outer_jobs = os.cpu_count() or 1

    outer_jobs = max(1, outer_jobs)
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
    for f_id, (tr, va) in enumerate(cv.split(X_train, y_train)):
        folds_payload.append(
            {
                "fold": int(f_id),
                "train_idx": tr.tolist(),
                "val_idx": va.tolist(),
            }
        )
    with (tables_dir / "cv_fold_indices.json").open("w") as f:
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
        refit=True,
    )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed_min = (time.time() - start) / 60.0
    print(f"✅ Halving search completed in {elapsed_min:.2f} minutes")

    # Save CV results
    cvres = pd.DataFrame(search.cv_results_)
    cvres.to_csv(tables_dir / "halving_cv_results.csv", index=False)

    # Best params + trees
    best_params = search.best_params_
    best_n_trees = int(search.cv_results_["param_n_estimators"][search.best_index_])
    best_model = search.best_estimator_
    actual_fit_trees = getattr(best_model, "n_estimators", None)
    print(
        "✅ Best Params:",
        best_params,
        "| n_estimators (CV):",
        best_n_trees,
        "| n_estimators (refit):",
        actual_fit_trees,
    )

    with (tables_dir / "best_params.json").open("w") as f:
        json.dump(
            {
                "best_params": best_params,
                "best_n_estimators_cv": best_n_trees,
                "best_n_estimators_refit": actual_fit_trees,
            },
            f,
            indent=2,
        )

    # Save model
    joblib.dump(best_model, output_dir / "rf_best_model.joblib")

    # ──────────────────────
    # Feature Importance
    # ──────────────────────
    try:
        check_is_fitted(best_model)
        importances = best_model.feature_importances_
        feature_df = (
            pd.DataFrame({"feature": predictor_cols, "importance": importances})
            .sort_values(by="importance", ascending=False)
        )
        feature_df.to_csv(tables_dir / "feature_importance.csv", index=False)

        plt.figure(figsize=(8, 5))
        sns.barplot(data=feature_df, x="importance", y="feature")
        plt.title("Feature Importance (Best Model)")
        plt.tight_layout()
        plt.savefig(plots_dir / "feature_importance.png")
        plt.close()
    except Exception as e:
        with (logs_dir / "warnings.txt").open("a") as f:
            f.write(f"[FeatureImportance] {repr(e)}\n")

    # ──────────────────────
    # Final evaluation on TRUE holdout
    # ──────────────────────
    y_proba_test = best_model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= 0.5).astype(int)

    final_auc = roc_auc_score(y_test, y_proba_test)
    final_acc = accuracy_score(y_test, y_pred_test)
    final_f1 = f1_score(y_test, y_pred_test)

    pd.DataFrame(
        [
            {
                "AUC_holdout": final_auc,
                "Accuracy_holdout": final_acc,
                "F1_holdout": final_f1,
                "Best_n_estimators_CV": best_n_trees,
                "Best_n_estimators_refit": actual_fit_trees,
                **best_params,
            }
        ]
    ).to_csv(tables_dir / "final_holdout_metrics.csv", index=False)

    print(
        f"✅ Holdout AUC={final_auc:.4f}, "
        f"ACC={final_acc:.4f}, F1={final_f1:.4f}"
    )

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba_test)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {final_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.title("ROC Curve (Best Model on True Holdout)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "final_roc_holdout.png")
    plt.close()

    print("✅ All done. Outputs saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
