#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Random Forest LOCO (Leave-One-City-Out) evaluation with Successive Halving.

Purpose
-------
City-level generalization test: each fold holds out one city (CTR_MN_NM),
tunes RF on remaining cities, evaluates on the held-out city.

Metrics reported (UNBALANCED only; i.e., full held-out city distribution)
----------------------------------------------------------------------------
Threshold-free:
- ROC AUC (positive class = DUA=1)
- PR-AUC / Average Precision (AP) (positive class = DUA=1)

Threshold-based at --threshold (default 0.4):
- Precision_pos (DUA=1)
- Recall_pos (DUA=1)
- F1_pos (DUA=1)
- Specificity (TNR)
- Balanced Accuracy
- MCC (Matthews correlation coefficient)

Label-averaged (computed from predicted labels at the same threshold):
- F1_macro
- F1_micro
- F1_weighted
- Accuracy

Area-weighted (using AREAHA_SEG; hectares) at the same threshold:
- TP_area_km2, FP_area_km2, FN_area_km2, TN_area_km2
- Precision_area, Recall_area, F1_area  (DUA positive, using area weights)

Notes
-----
- All precision/recall/F1_pos are for DUA=1 (sklearn default pos_label=1).
- Area-weighted metrics use segment area as weights (AREAHA_SEG in hectares).

Run example:
    python 2_modelling/01_training/Validation_rf_model_loco.py \
        --input-folder 1_preprocessing/LabelledData_For_RF \
        --output-dir 2_modelling/01_training/rf_outputs_loco \
        --threshold 0.4 \
        --area-col AREAHA_SEG
"""

import os
import json
import time
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    precision_score, recall_score, f1_score,
    balanced_accuracy_score,
    accuracy_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)

# Enable Successive Halving
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import randint


# ============================================================
# CONFIG — Schema + region mapping
# ============================================================
REGION_MAP = {
    "Unknown": 0,
    "Asia": 1,
    "Africa": 2,
    "Latin America and the Caribbean": 3,
}
REGION_COL = "REG1_GHSL"
CITY_COL = "CTR_MN_NM"
TARGET_COL = "slum_label1"

# Keep consistent with your VSURF selection
PREDICTOR_COLS = [
    "i5_par_area", "i1_pop_area", "i6_paru_area", "B_AVG_SEG",
    "i9_roads_par", "PARU_A_SEG", "B_CV_SEG",
    "REGION_CODE",
]


# ============================================================
# Args + folders
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="LOCO RF training + evaluation (unbalanced metrics + area-weighted).")
    p.add_argument(
        "--input-folder",
        type=str,
        required=False,
        default="1_preprocessing/LabelledData_For_RF",
        help="Folder with labeled CSVs (e.g., *_labeled_thr030.csv).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default="2_modelling/01_training/rf_outputs_loco",
        help="Folder where LOCO outputs will be saved.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        required=False,
        default=0.4,
        help="Probability threshold for label-based metrics (precision/recall/F1 etc.).",
    )
    p.add_argument(
        "--save-fold-models",
        action="store_true",
        help="If set, saves each fold's best model as joblib (can be large).",
    )
    p.add_argument(
        "--area-col",
        type=str,
        required=False,
        default="AREAHA_SEG",
        help="Column name for segment area (in hectares). Used for area-weighted metrics.",
    )
    return p.parse_args()


def setup_output_dirs(output_dir: Path, save_models: bool):
    tables_dir = output_dir / "tables"
    logs_dir = output_dir / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    models_dir = output_dir / "models"
    if save_models:
        models_dir.mkdir(parents=True, exist_ok=True)

    return tables_dir, logs_dir, models_dir


# ============================================================
# Data loading
# ============================================================
def map_region(val) -> int:
    if pd.isna(val):
        return REGION_MAP["Unknown"]
    return REGION_MAP.get(str(val), REGION_MAP["Unknown"])


def load_training_data(input_folder: Path, tables_dir: Path, logs_dir: Path, output_dir: Path, area_col: str) -> pd.DataFrame:
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    csv_files = sorted([p for p in input_folder.iterdir() if p.suffix.lower() == ".csv"])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_folder}")

    # deterministic file list
    pd.DataFrame({"file": [p.name for p in csv_files]}).to_csv(
        tables_dir / "file_list.csv", index=False
    )

    # persist region mapping
    with (output_dir / "region_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(REGION_MAP, f, ensure_ascii=False, indent=2)

    required_non_region = [c for c in PREDICTOR_COLS if c != "REGION_CODE"]
    required_cols = set(required_non_region) | {TARGET_COL, REGION_COL, CITY_COL}

    # area is optional but recommended; we will warn if missing
    want_area = area_col

    dfs = []
    missing_schema = []
    region_value_issues = []
    missing_area_files = []

    for file_path in csv_files:
        df = pd.read_csv(file_path)

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            missing_schema.append({"file": file_path.name, "missing_cols": missing})
            continue

        df["REGION_CODE"] = df[REGION_COL].map(map_region)

        unseen = set(df[REGION_COL].dropna().unique()) - set(REGION_MAP.keys())
        if unseen:
            region_value_issues.append(
                {"file": file_path.name, "unseen_values": sorted(list(unseen))}
            )

        # provenance
        df["SRC_FILE"] = file_path.name
        df["ROW_IN_FILE"] = np.arange(len(df), dtype=np.int64)

        if want_area not in df.columns:
            missing_area_files.append(file_path.name)
            df[want_area] = np.nan  # keep column so concat works

        keep_cols = (
            [c for c in PREDICTOR_COLS if c != "REGION_CODE"]
            + [TARGET_COL, "REGION_CODE", CITY_COL, "SRC_FILE", "ROW_IN_FILE", want_area]
        )
        df = df[keep_cols]
        dfs.append(df)

    if missing_schema:
        pd.DataFrame(missing_schema).to_csv(logs_dir / "missing_schema_files.csv", index=False)
    if region_value_issues:
        pd.DataFrame(region_value_issues).to_csv(logs_dir / "unseen_region_values.csv", index=False)
    if missing_area_files:
        pd.DataFrame({"file_missing_area_col": missing_area_files}).to_csv(
            logs_dir / "missing_area_col_files.csv", index=False
        )

    if not dfs:
        raise RuntimeError("No valid CSVs after schema checks. See logs/.")

    full = pd.concat(dfs, ignore_index=True)

    # Drop NA in predictors + target + city
    # For area, we do NOT drop rows here; we will handle NA area by skipping area-weighted metrics per fold if needed
    clean = full.dropna(subset=PREDICTOR_COLS + [TARGET_COL, CITY_COL]).reset_index(drop=True)

    # distributions
    clean.groupby(["REGION_CODE", TARGET_COL]).size().unstack(fill_value=0).to_csv(
        tables_dir / "region_class_distribution.csv"
    )
    clean.groupby([CITY_COL, TARGET_COL]).size().unstack(fill_value=0).to_csv(
        tables_dir / "city_class_distribution.csv"
    )

    return clean


# ============================================================
# Halving search
# ============================================================
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


def build_halving_search(outer_jobs: int):
    rf_base = RandomForestClassifier(
        random_state=42,
        n_jobs=1,  # avoid nested parallelism
        oob_score=False
    )

    param_dist = {
        "max_depth": randint(10, 25),
        "min_samples_leaf": randint(1, 11),
        "min_samples_split": randint(2, 12),
        "max_features": ["sqrt", 0.5],
        "bootstrap": [True, False],
        "class_weight": ["balanced", "balanced_subsample"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    return HalvingRandomSearchCV(
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
        verbose=1,
        refit=True,
    )


# ============================================================
# Metrics helpers
# ============================================================
def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b and b > 0 else 0.0


def compute_metrics_unbalanced(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    """
    Unweighted metrics on sample counts.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.clip(np.asarray(y_proba).astype(float), 0.0, 1.0)
    y_pred = (y_proba >= float(threshold)).astype(int)

    out = {}

    # threshold-free
    try:
        out["AUC"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        out["AUC"] = np.nan

    try:
        out["AP"] = float(average_precision_score(y_true, y_proba))
    except Exception:
        out["AP"] = np.nan

    # confusion matrix components (counts)
    # order: [[tn, fp],[fn,tp]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["TN"] = int(tn); out["FP"] = int(fp); out["FN"] = int(fn); out["TP"] = int(tp)

    # DUA-positive metrics (pos_label=1 by default)
    out["Precision_pos"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["Recall_pos"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["F1_pos"] = float(f1_score(y_true, y_pred, zero_division=0))

    # Specificity (TNR) = TN / (TN+FP)
    out["Specificity"] = _safe_div(float(tn), float(tn + fp))

    # symmetric-ish
    out["BalAcc"] = float(balanced_accuracy_score(y_true, y_pred))

    # averaged F1 variants
    out["F1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["F1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    out["F1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # accuracy
    out["Accuracy"] = float(accuracy_score(y_true, y_pred))

    # MCC
    try:
        out["MCC"] = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        out["MCC"] = np.nan

    out["n_test"] = int(len(y_true))
    out["pos_rate_test"] = float(np.mean(y_true))

    return out


def compute_area_weighted_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    area_ha: np.ndarray,
) -> dict:
    """
    Area-weighted confusion matrix + precision/recall/F1 for positive class (DUA=1).

    area_ha is in hectares; we report km^2:
      1 ha = 0.01 km^2
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    area_ha = np.asarray(area_ha).astype(float)

    # If any area is NaN/negative, the clean choice is: drop those rows for area-weighted metrics
    valid = np.isfinite(area_ha) & (area_ha >= 0)
    if valid.sum() == 0:
        return {
            "TP_area_km2": np.nan, "FP_area_km2": np.nan, "FN_area_km2": np.nan, "TN_area_km2": np.nan,
            "Precision_area": np.nan, "Recall_area": np.nan, "F1_area": np.nan,
            "n_area_valid": 0,
            "area_total_km2": np.nan,
        }

    yt = y_true[valid]
    yp = y_pred[valid]
    a_km2 = area_ha[valid] * 0.01

    tp_area = float(a_km2[(yt == 1) & (yp == 1)].sum())
    fp_area = float(a_km2[(yt == 0) & (yp == 1)].sum())
    fn_area = float(a_km2[(yt == 1) & (yp == 0)].sum())
    tn_area = float(a_km2[(yt == 0) & (yp == 0)].sum())

    prec_area = _safe_div(tp_area, tp_area + fp_area)
    rec_area = _safe_div(tp_area, tp_area + fn_area)
    f1_area = _safe_div(2 * prec_area * rec_area, (prec_area + rec_area)) if (prec_area + rec_area) > 0 else 0.0

    return {
        "TP_area_km2": tp_area,
        "FP_area_km2": fp_area,
        "FN_area_km2": fn_area,
        "TN_area_km2": tn_area,
        "Precision_area": float(prec_area),
        "Recall_area": float(rec_area),
        "F1_area": float(f1_area),
        "n_area_valid": int(valid.sum()),
        "area_total_km2": float(a_km2.sum()),
    }


# ============================================================
# Main
# ============================================================
def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ["_", "-"] else "_" for ch in s)


def main():
    args = parse_args()

    input_folder = Path(args.input_folder)
    output_dir = Path(args.output_dir)
    threshold = float(args.threshold)
    area_col = str(args.area_col)

    tables_dir, logs_dir, models_dir = setup_output_dirs(output_dir, save_models=args.save_fold_models)

    print(f"📁 Input folder:  {input_folder.resolve()}")
    print(f"📁 Output folder: {output_dir.resolve()}")
    print(f"🎯 Threshold (label metrics): {threshold}")
    print(f"📐 Area column (ha): {area_col}")

    clean_data = load_training_data(input_folder, tables_dir, logs_dir, output_dir, area_col=area_col)
    print(f"✅ Loaded and cleaned data: {clean_data.shape}")

    X_all = clean_data[PREDICTOR_COLS].to_numpy(dtype=float)
    y_all = clean_data[TARGET_COL].to_numpy().astype(int)
    city_all = clean_data[CITY_COL].astype(str).to_numpy()
    area_all = clean_data[area_col].to_numpy() if area_col in clean_data.columns else np.full(len(clean_data), np.nan)

    cities = sorted(pd.unique(city_all))
    if len(cities) < 2:
        raise RuntimeError(f"Need >=2 unique cities in {CITY_COL} for LOCO; found {len(cities)}")

    outer_jobs = get_outer_jobs()
    print(f"🧵 Using n_jobs={outer_jobs} for HalvingRandomSearchCV")

    fold_rows = []
    best_params_rows = []
    all_preds = []

    start_all = time.time()

    for fold_i, test_city in enumerate(cities):
        print("\n" + "=" * 70)
        print(f"🔁 LOCO Fold {fold_i+1}/{len(cities)} — Held-out city: {test_city}")

        test_mask = (city_all == test_city)
        train_mask = ~test_mask

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]
        area_test = area_all[test_mask]

        if len(np.unique(y_train)) < 2:
            msg = f"[Fold {test_city}] Training has single class; skipping."
            print("⚠️", msg)
            with (logs_dir / "warnings.txt").open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
            continue

        search = build_halving_search(outer_jobs=outer_jobs)

        t0 = time.time()
        search.fit(X_train, y_train)
        elapsed_min = (time.time() - t0) / 60.0
        print(f"✅ Search fit done in {elapsed_min:.2f} minutes")

        best_model = search.best_estimator_
        best_params = search.best_params_
        best_n_trees = int(search.cv_results_["param_n_estimators"][search.best_index_])
        actual_fit_trees = getattr(best_model, "n_estimators", None)

        best_params_rows.append({
            "heldout_city": test_city,
            "best_n_estimators_cv": best_n_trees,
            "best_n_estimators_refit": actual_fit_trees,
            **best_params,
        })

        # Predict proba of positive class (DUA=1)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        # Unweighted metrics on the full (unbalanced) held-out city
        m = compute_metrics_unbalanced(y_test, y_proba, threshold=threshold)

        # Area-weighted metrics (uses y_pred)
        m_area = compute_area_weighted_metrics(y_test, y_pred, area_test)

        row = {
            "heldout_city": test_city,
            "n_train": int(len(y_train)),
            "pos_rate_train": float(np.mean(y_train)),
            "best_n_estimators_cv": best_n_trees,
            "best_n_estimators_refit": actual_fit_trees,
            **m,
            **m_area,
        }
        fold_rows.append(row)

        print(
            f"📌 {test_city} | "
            f"AUC={m['AUC']:.3f} AP={m['AP']:.3f} "
            f"F1_pos={m['F1_pos']:.3f} Prec_pos={m['Precision_pos']:.3f} Rec_pos={m['Recall_pos']:.3f} "
            f"Spec={m['Specificity']:.3f} MCC={m['MCC']:.3f} "
            f"F1_area={m_area.get('F1_area', np.nan):.3f}"
        )

        # Save fold model optionally
        if args.save_fold_models:
            joblib.dump(best_model, models_dir / f"rf_best_model_heldout_{_safe_name(test_city)}.joblib")

        # Stacked predictions for downstream analysis/plots
        test_df = clean_data.loc[
            test_mask, [CITY_COL, "SRC_FILE", "ROW_IN_FILE", "REGION_CODE", TARGET_COL, area_col]
        ].copy()
        test_df["y_proba"] = y_proba
        test_df["y_pred"] = y_pred
        test_df["threshold"] = threshold
        all_preds.append(test_df)

    # Save outputs
    metrics_df = pd.DataFrame(fold_rows).sort_values("heldout_city")
    metrics_df.to_csv(tables_dir / "loco_metrics_by_city.csv", index=False)

    params_df = pd.DataFrame(best_params_rows).sort_values("heldout_city")
    params_df.to_csv(tables_dir / "loco_best_params_by_city.csv", index=False)

    if all_preds:
        preds_df = pd.concat(all_preds, ignore_index=True)
        preds_df.to_csv(tables_dir / "loco_predictions_all.csv", index=False)

    # Summary
    def _summ(col):
        s = metrics_df[col].astype(float)
        return {
            f"{col}_mean": float(np.nanmean(s)),
            f"{col}_std": float(np.nanstd(s)),
            f"{col}_min": float(np.nanmin(s)),
            f"{col}_max": float(np.nanmax(s)),
            f"{col}_median": float(np.nanmedian(s)),
        }

    summary_cols = [
        "AUC", "AP",
        "Precision_pos", "Recall_pos", "F1_pos",
        "Specificity", "BalAcc", "MCC",
        "F1_macro", "F1_micro", "F1_weighted",
        "Accuracy",
        # area-weighted
        "Precision_area", "Recall_area", "F1_area",
        "TP_area_km2", "FP_area_km2", "FN_area_km2", "TN_area_km2",
        "area_total_km2",
    ]

    summary = {
        "n_cities": int(metrics_df.shape[0]),
        "threshold": threshold,
        "area_col": area_col,
        **{k: v for col in summary_cols for k, v in _summ(col).items()},
    }
    pd.DataFrame([summary]).to_csv(tables_dir / "loco_metrics_summary.csv", index=False)

    elapsed_all = (time.time() - start_all) / 60.0
    print("\n" + "=" * 70)
    print(f"✅ LOCO complete in {elapsed_all:.2f} minutes")
    print(f"✅ Saved metrics:  {tables_dir / 'loco_metrics_by_city.csv'}")
    print(f"✅ Saved summary:  {tables_dir / 'loco_metrics_summary.csv'}")
    print(f"✅ Saved preds:    {tables_dir / 'loco_predictions_all.csv'}")
    if args.save_fold_models:
        print(f"✅ Saved models:   {models_dir}")


if __name__ == "__main__":
    main()