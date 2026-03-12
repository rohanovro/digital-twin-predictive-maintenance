"""
train.py
========
Full training pipeline:
  1. Load & engineer features
  2. SMOTE oversampling on minority class
  3. Train RF, Gradient Boosting, Logistic Regression
  4. 5-fold stratified cross-validation
  5. Anomaly detection (Isolation Forest)
  6. RUL estimation
  7. Save all models + metrics

Run:  python src/train.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils import resample

from src.config import (
    RF_PARAMS, GBT_PARAMS, LR_PARAMS,
    MODEL_DIR, RESULTS_DIR, ALL_FEATURES,
    RANDOM_STATE, CONTAMINATION, FEAT_LABELS,
    SIM_MAX_WEAR,
)
from src.utils import (
    load_data, make_splits, evaluate_model,
    save_metrics, print_banner,
)


# ── SMOTE-STYLE OVERSAMPLING (no imbalanced-learn dependency) ─────────────────
def oversample_minority(X_tr, y_tr):
    """
    Manual SMOTE-style oversampling: duplicate minority class
    with slight Gaussian noise to simulate SMOTE behaviour.
    Works without imbalanced-learn installed.
    """
    majority_idx = np.where(y_tr == 0)[0]
    minority_idx = np.where(y_tr == 1)[0]
    n_to_add     = len(majority_idx) - len(minority_idx)

    rng       = np.random.default_rng(RANDOM_STATE)
    chosen    = rng.choice(minority_idx, size=n_to_add, replace=True)
    noise     = rng.normal(0, 0.05, size=(n_to_add, X_tr.shape[1]))
    X_synth   = X_tr[chosen] + noise
    y_synth   = np.ones(n_to_add, dtype=int)

    X_bal = np.vstack([X_tr, X_synth])
    y_bal = np.concatenate([y_tr, y_synth])

    shuffle = rng.permutation(len(y_bal))
    print(f"[SMOTE] Balanced: {len(majority_idx):,} normal | "
          f"{len(minority_idx)+n_to_add:,} failure (was {len(minority_idx)})")
    return X_bal[shuffle], y_bal[shuffle]


# ── TRAIN ─────────────────────────────────────────────────────────────────────
def train_all(Xtr, Xte, y_tr, y_te, use_smote=True):
    """Train RF, GBT, LR and a soft-voting ensemble. Return metrics dict."""

    if use_smote:
        Xtr_fit, y_tr_fit = oversample_minority(Xtr, y_tr)
    else:
        Xtr_fit, y_tr_fit = Xtr, y_tr

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Individual models
    rf  = RandomForestClassifier(**RF_PARAMS)
    gbt = GradientBoostingClassifier(**GBT_PARAMS)
    lr  = LogisticRegression(**LR_PARAMS)

    print_banner("Training classifiers …")
    rf.fit(Xtr_fit,  y_tr_fit)
    gbt.fit(Xtr_fit, y_tr_fit)
    lr.fit(Xtr_fit,  y_tr_fit)

    # Soft-voting ensemble (RF + GBT)
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gbt", gbt)],
        voting="soft",
    )
    ensemble.fit(Xtr_fit, y_tr_fit)

    models_map = {
        "Random Forest":   rf,
        "Gradient Boost":  gbt,
        "Logistic Reg":    lr,
        "Ensemble (RF+GBT)": ensemble,
    }

    print("\n[EVAL] Test-set performance:")
    results = {}
    for name, mdl in models_map.items():
        m = evaluate_model(mdl, Xte, y_te, name)
        # Add cross-val F1
        cv_scores = cross_val_score(mdl, Xtr_fit, y_tr_fit, cv=cv, scoring="f1")
        m["cv_f1"]  = float(cv_scores.mean())
        m["cv_std"] = float(cv_scores.std())
        print(f"           CV-F1={m['cv_f1']:.4f} ± {m['cv_std']:.4f}")
        results[name] = m

    return results, models_map


# ── ANOMALY DETECTION ─────────────────────────────────────────────────────────
def run_anomaly_detection(Xtr, Xte, y_te):
    """Fit Isolation Forest and return predictions + scores."""
    print_banner("Anomaly Detection — Isolation Forest")
    iso = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)
    iso.fit(Xtr)
    preds  = iso.predict(Xte)          # -1 = anomaly, 1 = normal
    scores = iso.decision_function(Xte)
    n_anom = (preds == -1).sum()
    # Overlap with actual failures
    overlap = ((preds == -1) & (y_te == 1)).sum()
    print(f"  Anomalies detected : {n_anom} / {len(preds)}")
    print(f"  Overlap with actual failures: {overlap} / {y_te.sum()}")
    return preds, scores, iso


# ── RUL ESTIMATION ────────────────────────────────────────────────────────────
def estimate_rul(model, scaler, wear_range=range(0, 255, 5)) -> pd.DataFrame:
    """
    Estimate Remaining Useful Life (RUL) across a range of tool-wear values,
    holding all other sensors at their nominal operating point.
    """
    rows = []
    for wear in wear_range:
        nominal = {
            "Type_enc": 1,
            "Air temperature [K]":    300.0,
            "Process temperature [K]":310.0,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]":            40.0,
            "Tool wear [min]":        wear,
            "temp_diff":              10.0,
            "power":                  60000,
            "wear_torque":            wear * 40.0,
            "speed_wear":             wear * 1500,
            "power_temp":             60000 * 10.0,
            "wear_speed_torque":      wear * 1500 * 40.0,
        }
        x = np.array([[nominal[f] for f in ALL_FEATURES]])
        prob = model.predict_proba(scaler.transform(x))[0][1]
        rul  = max(0.0, (1 - prob) * (SIM_MAX_WEAR - wear))
        rows.append({"wear": wear, "fail_prob": prob * 100, "rul": rul})
    df = pd.DataFrame(rows)
    print(f"  Maintenance trigger: wear ≈ "
          f"{df[df['fail_prob']>=50]['wear'].min()} min "
          f"(fail prob crosses 50%)")
    return df


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print_banner("AI-Driven Digital Twin — Full Training Pipeline")

    df = load_data()
    Xtr, Xte, y_tr, y_te, scaler, X_te_df = make_splits(df)

    # ── Models ────────────────────────────────────────────────────────────────
    results, models_map = train_all(Xtr, Xte, y_tr, y_te, use_smote=True)

    # ── Anomaly detection ─────────────────────────────────────────────────────
    anom_preds, anom_scores, iso = run_anomaly_detection(Xtr, Xte, y_te)
    joblib.dump(iso, f"{MODEL_DIR}/isolation_forest.pkl")

    # ── RUL ───────────────────────────────────────────────────────────────────
    print_banner("RUL Estimation")
    best_mdl = models_map["Random Forest"]
    rul_df   = estimate_rul(best_mdl, scaler)
    rul_df.to_csv(f"{RESULTS_DIR}/rul_estimates.csv", index=False)

    # ── Save models ───────────────────────────────────────────────────────────
    print_banner("Saving Artefacts")
    for name, mdl in models_map.items():
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
        joblib.dump(mdl, f"{MODEL_DIR}/{fname}.pkl")
        print(f"  Saved → {MODEL_DIR}/{fname}.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    # ── Persist metrics ───────────────────────────────────────────────────────
    save_metrics(results)

    # Save anomaly results for visualise.py to use
    import json
    anom_out = {
        "preds":  anom_preds.tolist(),
        "scores": anom_scores.tolist(),
        "n_anomalies": int((anom_preds == -1).sum()),
    }
    with open(f"{RESULTS_DIR}/anomaly_results.json", "w") as f:
        json.dump(anom_out, f)

    print_banner("Pipeline Complete ✓")
    print(f"  Models  → {MODEL_DIR}/")
    print(f"  Results → {RESULTS_DIR}/")
    print(f"\n  Next: python src/visualise.py")


if __name__ == "__main__":
    main()
