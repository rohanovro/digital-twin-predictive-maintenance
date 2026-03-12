"""
utils.py
========
Shared helper functions: data loading, feature engineering,
preprocessing, and metric printing.
"""

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve,
)

try:
    from src.config import (
        DATA_PATH, ALL_FEATURES, FAILURE_COL,
        RANDOM_STATE, TEST_SIZE, RESULTS_DIR,
    )
except ModuleNotFoundError:
    from config import (
        DATA_PATH, ALL_FEATURES, FAILURE_COL,
        RANDOM_STATE, TEST_SIZE, RESULTS_DIR,
    )


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to a raw AI4I dataframe.
    Returns a new dataframe with added columns.
    """
    df = df.copy()
    le = LabelEncoder()
    df["Type_enc"]         = le.fit_transform(df["Type"])

    # Physics-based pairs
    df["temp_diff"]         = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["power"]             = df["Rotational speed [rpm]"]  * df["Torque [Nm]"]
    df["wear_torque"]       = df["Tool wear [min]"]         * df["Torque [Nm]"]
    df["speed_wear"]        = df["Rotational speed [rpm]"]  * df["Tool wear [min]"]

    # Higher-order interactions (new)
    df["power_temp"]        = df["power"] * df["temp_diff"]
    df["wear_speed_torque"] = df["Tool wear [min]"] * df["Rotational speed [rpm]"] * df["Torque [Nm]"]

    return df


# ── LOADING & SPLITTING ───────────────────────────────────────────────────────

def load_data(path: str = DATA_PATH):
    """Load raw CSV, engineer features, return full dataframe."""
    df = pd.read_csv(path)
    df = engineer_features(df)
    print(f"[DATA] Loaded {len(df):,} rows | "
          f"Failure rate: {df[FAILURE_COL].mean()*100:.2f}% "
          f"({df[FAILURE_COL].sum()} failures)")
    return df


def make_splits(df: pd.DataFrame):
    """
    Return (X_train, X_test, y_train, y_test, scaler)
    with StandardScaler fitted on training data only.
    """
    X = df[ALL_FEATURES]
    y = df[FAILURE_COL]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X_tr)
    Xte_sc = scaler.transform(X_te)
    print(f"[SPLIT] Train={len(y_tr):,} | Test={len(y_te):,} | "
          f"Failures in test={y_te.sum()}")
    return Xtr_sc, Xte_sc, y_tr.values, y_te.values, scaler, X_te


# ── EVALUATION ────────────────────────────────────────────────────────────────

def evaluate_model(model, X_te, y_te, name: str) -> dict:
    """Compute and return full evaluation metrics for a fitted model."""
    yp    = model.predict(X_te)
    yprob = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, yprob)
    prec_c, rec_c, _ = precision_recall_curve(y_te, yprob)

    metrics = dict(
        name      = name,
        acc       = float(accuracy_score(y_te, yp)),
        prec      = float(precision_score(y_te, yp, zero_division=0)),
        rec       = float(recall_score(y_te, yp, zero_division=0)),
        f1        = float(f1_score(y_te, yp, zero_division=0)),
        auc       = float(roc_auc_score(y_te, yprob)),
        fpr       = fpr.tolist(),
        tpr       = tpr.tolist(),
        prec_c    = prec_c.tolist(),
        rec_c     = rec_c.tolist(),
        yprob     = yprob.tolist(),
        cm        = confusion_matrix(y_te, yp).tolist(),
        imp       = getattr(model, "feature_importances_", np.zeros(len(ALL_FEATURES))).tolist(),
        report    = classification_report(y_te, yp, target_names=["No Fail", "Fail"]),
    )
    print(f"  [{name}] ACC={metrics['acc']:.4f}  "
          f"PREC={metrics['prec']:.4f}  REC={metrics['rec']:.4f}  "
          f"F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}")
    return metrics


def save_metrics(all_metrics: dict, path: str = None):
    """Persist metrics dict to JSON."""
    if path is None:
        path = f"{RESULTS_DIR}/metrics.json"
    serialisable = {
        k: {mk: mv for mk, mv in v.items()
            if mk not in ("fpr","tpr","prec_c","rec_c","yprob","cm","imp","report")}
        for k, v in all_metrics.items()
    }
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"[SAVE] Metrics → {path}")


def print_banner(text: str, width: int = 65):
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)
