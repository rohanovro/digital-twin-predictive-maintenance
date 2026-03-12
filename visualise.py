"""
visualise.py
============
Generate all 9 publication-ready figures for the Digital Twin project.
Warm teal / coral / gold palette throughout — no black backgrounds.

Run:  python src/visualise.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

from src.config import (
    PALETTE, COLORS, RC, FIGURES_DIR, RESULTS_DIR,
    MODEL_DIR, ALL_FEATURES, FEAT_LABELS, FAILURE_MODES,
    RANDOM_STATE, RF_PARAMS, GBT_PARAMS,
)
from src.utils import load_data, make_splits, evaluate_model

plt.rcParams.update(RC)

P = PALETTE
C = COLORS


def _save(name: str):
    path = f"{FIGURES_DIR}/{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close("all")
    print(f"  [FIG] → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def fig_dataset_overview(df):
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(P["bg"])
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("AI4I 2020 — Dataset Overview & Sensor Distributions",
                 fontsize=20, fontweight="bold", color=P["text"], y=0.98)

    sensors = [
        ("Air temperature [K]",    "Air Temperature (K)",     C[0]),
        ("Process temperature [K]","Process Temperature (K)", C[1]),
        ("Rotational speed [rpm]", "Rotational Speed (rpm)",  C[2]),
        ("Torque [Nm]",            "Torque (Nm)",             C[3]),
        ("Tool wear [min]",        "Tool Wear (min)",         C[4]),
    ]
    positions = [(0,0),(0,1),(0,2),(0,3),(1,0)]
    for (col, label, color), pos in zip(sensors, positions):
        ax = fig.add_subplot(gs[pos])
        vals_ok   = df[df["Machine failure"]==0][col]
        vals_fail = df[df["Machine failure"]==1][col]
        ax.hist(vals_ok,   bins=35, color=color, alpha=0.65, edgecolor="white",
                linewidth=0.4, label="Normal")
        ax.hist(vals_fail, bins=20, color=P["coral"], alpha=0.85, edgecolor="white",
                linewidth=0.4, label="Failure")
        ax.set_title(label, fontweight="bold", fontsize=11, color=P["text"])
        ax.set_xlabel("Value", fontsize=9, color=P["dim"])
        ax.set_ylabel("Count", fontsize=9, color=P["dim"])
        ax.legend(fontsize=8, framealpha=0.6)

    # Class balance bar
    ax_bal = fig.add_subplot(gs[1,1])
    cats   = ["Normal\n(96.6%)", "Failure\n(3.4%)"]
    vals   = [9661, 339]
    bars   = ax_bal.bar(cats, vals, color=[C[0], P["coral"]], alpha=0.85,
                        edgecolor="white", width=0.5)
    for b in bars:
        ax_bal.text(b.get_x()+b.get_width()/2, b.get_height()+80,
                    f"{int(b.get_height()):,}", ha="center", fontweight="bold", fontsize=11)
    ax_bal.set_title("Class Balance", fontweight="bold", fontsize=11)
    ax_bal.set_ylabel("Sample Count")

    # Failure mode pie
    ax_pie = fig.add_subplot(gs[1,2])
    ft = {k: int(df[k].sum()) for k in FAILURE_MODES}
    wedges, _, autotexts = ax_pie.pie(
        ft.values(), labels=ft.keys(), autopct="%1.0f%%",
        colors=C, startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2))
    for at in autotexts: at.set_fontsize(9)
    ax_pie.set_title("Failure Modes", fontweight="bold", fontsize=11)

    # Correlation heatmap (simplified)
    ax_corr = fig.add_subplot(gs[1,3])
    cols = ["Air temperature [K]","Process temperature [K]",
            "Rotational speed [rpm]","Torque [Nm]","Tool wear [min]","Machine failure"]
    corr = df[cols].corr()
    short = ["Air T","Proc T","Speed","Torque","Wear","Fail"]
    im = ax_corr.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax_corr.set_xticks(range(len(short))); ax_corr.set_yticks(range(len(short)))
    ax_corr.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax_corr.set_yticklabels(short, fontsize=8)
    for i in range(len(short)):
        for j in range(len(short)):
            ax_corr.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                         fontsize=7, color="black")
    ax_corr.set_title("Sensor Correlation", fontweight="bold", fontsize=11)
    plt.colorbar(im, ax=ax_corr, fraction=0.04)

    _save("01_dataset_overview")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def fig_model_comparison(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(P["bg"])
    fig.suptitle("Model Comparison — Performance Metrics (with SMOTE)",
                 fontsize=18, fontweight="bold", color=P["text"])

    mnames  = list(results.keys())
    metrics = ["acc","prec","rec","f1","auc"]
    mlabels = ["Accuracy","Precision","Recall","F1","AUC"]
    x = np.arange(len(metrics)); w = 0.18

    ax = axes[0]
    for i,(name,color) in enumerate(zip(mnames,C)):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + i*w, vals, w, label=name, color=color, alpha=0.85, edgecolor="white")
    ax.set_xticks(x + w*1.5); ax.set_xticklabels(mlabels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0.0, 1.05); ax.set_title("All Metrics", fontweight="bold", fontsize=13)
    ax.legend(fontsize=8, loc="lower right"); ax.set_ylabel("Score")

    ax2 = axes[1]
    for name, color in zip(mnames, C):
        fpr = results[name]["fpr"][::4]; tpr = results[name]["tpr"][::4]
        ax2.plot(fpr, tpr, color=color, lw=2.5,
                 label=f"{name[:14]} (AUC={results[name]['auc']:.3f})")
    ax2.plot([0,1],[0,1],"--", color=P["dim"], lw=1.5, label="Random")
    ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curves", fontweight="bold", fontsize=13)
    ax2.legend(fontsize=7)
    ax2.fill_between([0,1],[0,1], alpha=0.05, color=P["dim"])

    ax3 = axes[2]
    cv_vals = [results[n]["cv_f1"] for n in mnames]
    cv_stds = [results[n]["cv_std"] for n in mnames]
    bars = ax3.barh(mnames, cv_vals, xerr=cv_stds, color=C[:len(mnames)],
                    alpha=0.85, edgecolor="white", height=0.5, capsize=5,
                    error_kw=dict(ecolor=P["text"], lw=1.5))
    for bar, val in zip(bars, cv_vals):
        ax3.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=10, fontweight="bold")
    ax3.set_xlim(0, 1.08)
    ax3.set_title("5-Fold CV F1 ± Std", fontweight="bold", fontsize=13)
    ax3.set_xlabel("Mean F1 Score")

    plt.tight_layout()
    _save("02_model_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — CONFUSION MATRICES (ALL MODELS)
# ══════════════════════════════════════════════════════════════════════════════
def fig_confusion_matrices(results):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    fig.patch.set_facecolor(P["bg"])
    fig.suptitle("Confusion Matrices — All Models", fontsize=18,
                 fontweight="bold", color=P["text"])

    cmaps = ["YlGnBu","PuRd","YlOrRd","BuGn"]
    for ax, (name, res), cmap in zip(axes, results.items(), cmaps):
        cm = np.array(res["cm"])
        im = ax.imshow(cm, cmap=cmap, aspect="auto")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["No Fail","Fail"], fontsize=11)
        ax.set_yticklabels(["No Fail","Fail"], fontsize=11)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                        fontsize=24, fontweight="bold",
                        color="white" if cm[i,j] > cm.max()*0.5 else P["text"])
        spec = cm[0,0] / (cm[0,0]+cm[0,1]+1e-9)
        sens = cm[1,1] / (cm[1,0]+cm[1,1]+1e-9)
        ax.set_title(f"{name}\nSpec={spec:.1%}  Sens={sens:.1%}",
                     fontweight="bold", fontsize=11, color=P["text"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    _save("03_confusion_matrices")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — PRECISION-RECALL + ROC COMBINED
# ══════════════════════════════════════════════════════════════════════════════
def fig_pr_roc(results, y_te):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(P["bg"])
    fig.suptitle("Precision-Recall & ROC Curves — All Models",
                 fontsize=18, fontweight="bold", color=P["text"])

    for name, color in zip(results.keys(), C):
        yprob = np.array(results[name]["yprob"])
        prec_c, rec_c, _ = precision_recall_curve(y_te, yprob)
        axes[0].plot(rec_c, prec_c, color=color, lw=2.5, label=name)

        fpr, tpr, _ = roc_curve(y_te, yprob)
        axes[1].plot(fpr, tpr, color=color, lw=2.5,
                     label=f"{name} (AUC={results[name]['auc']:.3f})")

    # Baseline
    pos_rate = y_te.mean()
    axes[0].axhline(pos_rate, ls="--", color=P["dim"], lw=1.5, label=f"Baseline ({pos_rate:.2f})")
    axes[1].plot([0,1],[0,1],"--", color=P["dim"], lw=1.5, label="Random")

    for ax, title, xl, yl in zip(
        axes,
        ["Precision-Recall Curve\n(Imbalanced class — lower baseline expected)",
         "ROC Curve"],
        ["Recall","False Positive Rate"],
        ["Precision","True Positive Rate"]):
        ax.set_xlabel(xl, fontsize=11); ax.set_ylabel(yl, fontsize=11)
        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.legend(fontsize=8)
        ax.set_xlim(0,1); ax.set_ylim(0,1.02)

    plt.tight_layout()
    _save("04_pr_roc_curves")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — XAI FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
def fig_xai(results):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor(P["bg"])
    fig.suptitle("Explainable AI — Feature Importance Analysis",
                 fontsize=18, fontweight="bold", color=P["text"])

    # RF feature importance (horizontal bar)
    ax1 = axes[0]
    fi     = np.array(results["Random Forest"]["imp"])
    fi_pct = fi * 100
    sidx   = np.argsort(fi_pct)
    bar_c  = [P["teal"] if fi_pct[i]>15
              else P["gold"] if fi_pct[i]>8
              else P["lavender"] for i in sidx]
    bars = ax1.barh([FEAT_LABELS[i] for i in sidx], [fi_pct[i] for i in sidx],
                    color=bar_c, alpha=0.88, edgecolor="white")
    for bar, val in zip(bars, [fi_pct[i] for i in sidx]):
        ax1.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=8, fontweight="bold", color=P["text"])
    ax1.set_title("Random Forest\nGini Feature Importance", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Importance (%)")
    patches = [mpatches.Patch(color=P["teal"],    label="High (>15%)"),
               mpatches.Patch(color=P["gold"],    label="Medium (8-15%)"),
               mpatches.Patch(color=P["lavender"],label="Low (<8%)")]
    ax1.legend(handles=patches, fontsize=8, loc="lower right")

    # GBT feature importance
    ax2 = axes[1]
    fi2    = np.array(results["Gradient Boost"]["imp"])
    fi2_pct = fi2 * 100
    sidx2  = np.argsort(fi2_pct)
    ax2.barh([FEAT_LABELS[i] for i in sidx2], [fi2_pct[i] for i in sidx2],
             color=C[1], alpha=0.85, edgecolor="white")
    ax2.set_title("Gradient Boosting\nFeature Importance", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Importance (%)")

    # Failure driver donut
    ax3 = axes[2]
    cats   = ["Mechanical\nLoad","Wear\nDegradation","Thermal\nEffects","Other"]
    # Mechanical: speed, torque, power; Wear: tool wear, wear_torque, speed_wear; Thermal: temp_diff, air, proc; Other: type
    vals   = [
        fi_pct[3]+fi_pct[4]+fi_pct[7],  # speed + torque + power
        fi_pct[5]+fi_pct[8]+fi_pct[9],  # wear + wear_torque + speed_wear
        fi_pct[6]+fi_pct[1]+fi_pct[2],  # temp_diff + air + proc
        fi_pct[0]+fi_pct[10]+fi_pct[11],
    ]
    wedges, texts, autotexts = ax3.pie(
        vals, labels=cats, autopct="%1.1f%%",
        colors=[P["coral"],P["teal"],P["sky"],P["lavender"]],
        startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2.5),
        pctdistance=0.76)
    for at in autotexts: at.set_fontsize(10); at.set_fontweight("bold")
    ax3.set_title("Failure Driver Categories\n(Feature Group Importance)",
                  fontweight="bold", fontsize=12)
    ax3.add_patch(plt.Circle((0,0), 0.55, fc=P["panel"]))
    ax3.text(0,0,"12\nFeatures", ha="center", va="center",
             fontsize=12, fontweight="bold", color=P["text"])

    plt.tight_layout()
    _save("05_xai_feature_importance")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — SMOTE IMPACT
# ══════════════════════════════════════════════════════════════════════════════
def fig_smote_impact(results_no_smote, results_smote):
    """Compare key metrics before and after SMOTE oversampling."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(P["bg"])
    fig.suptitle("SMOTE Impact — Recall Improvement on Minority Class",
                 fontsize=18, fontweight="bold", color=P["text"])

    models  = ["Random Forest", "Gradient Boost"]
    metrics = ["rec", "f1", "prec"]
    mlabels = ["Recall", "F1", "Precision"]

    for ax, model in zip(axes, models):
        x  = np.arange(len(metrics)); w = 0.35
        b1 = ax.bar(x - w/2,
                    [results_no_smote[model][m] for m in metrics],
                    w, label="Before SMOTE", color=P["lavender"], alpha=0.85, edgecolor="white")
        b2 = ax.bar(x + w/2,
                    [results_smote[model][m] for m in metrics],
                    w, label="After SMOTE",  color=P["teal"],     alpha=0.85, edgecolor="white")
        for b in list(b1)+list(b2):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                    f"{b.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(mlabels)
        ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
        ax.set_title(model, fontweight="bold", fontsize=13)
        ax.legend(fontsize=10)

    plt.tight_layout()
    _save("06_smote_impact")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def fig_anomaly(anom_preds, anom_scores, X_te_df, y_te):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor(P["bg"])
    fig.suptitle("Anomaly Detection — Isolation Forest",
                 fontsize=18, fontweight="bold", color=P["text"])

    # Score distribution
    ax1 = axes[0]
    n_s = anom_scores[anom_preds == 1]
    a_s = anom_scores[anom_preds == -1]
    ax1.hist(n_s, bins=40, color=P["teal"],  alpha=0.75,
             label=f"Normal ({(anom_preds==1).sum()})", edgecolor="white")
    ax1.hist(a_s, bins=15, color=P["coral"], alpha=0.88,
             label=f"Anomaly ({(anom_preds==-1).sum()})", edgecolor="white")
    ax1.axvline(-0.1, color=P["text"], lw=2, ls="--", label="Decision boundary")
    ax1.set_xlabel("Anomaly Score"); ax1.set_ylabel("Count")
    ax1.set_title("Score Distribution", fontweight="bold", fontsize=13)
    ax1.legend(fontsize=9)

    # Speed vs Torque scatter
    ax2 = axes[1]
    speed  = X_te_df["Rotational speed [rpm]"].values
    torque = X_te_df["Torque [Nm]"].values
    normal = anom_preds == 1
    ax2.scatter(speed[normal],  torque[normal],  c=P["teal"],  alpha=0.22, s=8,  label="Normal", zorder=1)
    ax2.scatter(speed[~normal], torque[~normal], c=P["coral"], alpha=0.90, s=55,
                edgecolors="white", linewidths=0.5, label="Anomaly", zorder=3)
    ax2.set_xlabel("Rotational Speed (rpm)"); ax2.set_ylabel("Torque (Nm)")
    ax2.set_title("Speed vs Torque\nAnomaly Map", fontweight="bold", fontsize=13)
    ax2.legend(fontsize=9)

    # Anomaly vs actual failure overlap
    ax3 = axes[2]
    overlap = np.zeros((2,2))
    for pred, actual in zip(anom_preds, y_te):
        r = 0 if actual == 0 else 1
        c = 0 if pred == 1  else 1
        overlap[r, c] += 1
    im = ax3.imshow(overlap, cmap="YlOrRd", aspect="auto")
    ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
    ax3.set_xticklabels(["Normal\n(IF)","Anomaly\n(IF)"])
    ax3.set_yticklabels(["No Fail\n(True)","Fail\n(True)"])
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, int(overlap[i,j]), ha="center", va="center",
                     fontsize=22, fontweight="bold",
                     color="white" if overlap[i,j]>overlap.max()*0.5 else P["text"])
    ax3.set_title("IF Predictions vs\nActual Failures", fontweight="bold", fontsize=13)
    plt.colorbar(im, ax=ax3, fraction=0.046)

    plt.tight_layout()
    _save("07_anomaly_detection")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — RUL PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def fig_rul(rul_df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(P["bg"])
    fig.suptitle("Remaining Useful Life (RUL) — AI-Based Estimation",
                 fontsize=18, fontweight="bold", color=P["text"])

    ax1 = axes[0]
    ax1.fill_between(rul_df["wear"], rul_df["fail_prob"], color=P["coral"], alpha=0.25)
    ax1.plot(rul_df["wear"], rul_df["fail_prob"], color=P["coral"], lw=3,  label="Failure Probability")
    ax1.axhline(30, color=P["gold"],    ls="--", lw=2, label="Warning  (30%)")
    ax1.axhline(60, color=P["crimson"], ls="--", lw=2, label="Critical (60%)")
    ax1.fill_between(rul_df["wear"], 0, rul_df["fail_prob"],
                     where=rul_df["fail_prob"]>30, color=P["gold"],    alpha=0.10)
    ax1.fill_between(rul_df["wear"], 0, rul_df["fail_prob"],
                     where=rul_df["fail_prob"]>60, color=P["crimson"], alpha=0.12)

    # Shade zones
    ax1.axvspan(0,  100, alpha=0.04, color=P["mint"],  label="Safe zone")
    ax1.axvspan(150,250, alpha=0.04, color=P["coral"], label="Danger zone")

    ax1.set_xlabel("Tool Wear (minutes)", fontsize=11)
    ax1.set_ylabel("Failure Probability (%)", fontsize=11)
    ax1.set_title("Failure Probability vs Tool Wear", fontweight="bold", fontsize=13)
    ax1.legend(fontsize=9, ncol=2); ax1.set_ylim(0, 100)

    ax2 = axes[1]
    ax2.fill_between(rul_df["wear"], rul_df["rul"], color=P["teal"], alpha=0.22)
    ax2.plot(rul_df["wear"], rul_df["rul"], color=P["teal"], lw=3, label="Estimated RUL")

    # Mark key points
    trigger_row = rul_df[rul_df["fail_prob"] >= 30].head(1)
    if len(trigger_row):
        tw = trigger_row["wear"].values[0]
        tr = trigger_row["rul"].values[0]
        ax2.axvline(tw, color=P["gold"], ls="--", lw=2, label=f"Maintenance at {tw} min")
        ax2.scatter([tw],[tr], c=P["gold"], s=120, zorder=5)

    ax2.set_xlabel("Current Tool Wear (minutes)", fontsize=11)
    ax2.set_ylabel("Estimated RUL (minutes)", fontsize=11)
    ax2.set_title("Remaining Useful Life Curve", fontweight="bold", fontsize=13)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    _save("08_rul_prediction")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 9 — RESEARCH SUMMARY DASHBOARD (all KPIs in one figure)
# ══════════════════════════════════════════════════════════════════════════════
def fig_summary_dashboard(results, rul_df, anom_preds, anom_scores, y_te):
    fig = plt.figure(figsize=(22, 12))
    fig.patch.set_facecolor(P["bg"])
    gs  = gridspec.GridSpec(3, 5, figure=fig, hspace=0.50, wspace=0.40)
    fig.suptitle("AI-Driven Digital Twin — Research Summary Dashboard",
                 fontsize=22, fontweight="bold", color=P["text"], y=0.99)

    # ── Row 0: KPI tiles ──────────────────────────────────────────────────────
    kpis = [
        (f"{results['Random Forest']['acc']*100:.1f}%",  "RF Accuracy",    P["teal"]),
        (f"{results['Random Forest']['auc']:.3f}",        "ROC-AUC",        P["coral"]),
        (f"{results['Ensemble (RF+GBT)']['f1']:.3f}",     "Ensemble F1",    P["gold"]),
        (f"{results['Random Forest']['rec']*100:.1f}%",   "Recall (+SMOTE)",P["mint"]),
        (f"{(anom_preds==-1).sum()}",                     "Anomalies Found",P["lavender"]),
    ]
    for i,(val,lbl,col) in enumerate(kpis):
        ax = fig.add_subplot(gs[0,i])
        ax.set_facecolor(col); ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.text(0.5,0.60,val, ha="center",va="center", fontsize=24,
                fontweight="bold", color="white", transform=ax.transAxes)
        ax.text(0.5,0.22,lbl, ha="center",va="center", fontsize=10,
                color="white", alpha=0.9,   transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)

    # ── Row 1 col 0-1: ROC curves ─────────────────────────────────────────────
    ax_roc = fig.add_subplot(gs[1, :2])
    for name, color in zip(results.keys(), C):
        fpr = results[name]["fpr"][::5]; tpr = results[name]["tpr"][::5]
        ax_roc.plot(fpr, tpr, color=color, lw=2,
                    label=f"{name[:12]} ({results[name]['auc']:.3f})")
    ax_roc.plot([0,1],[0,1],"--", color=P["dim"], lw=1)
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.set_title("ROC Curves", fontweight="bold", fontsize=12)
    ax_roc.legend(fontsize=7, ncol=2)

    # ── Row 1 col 2-3: Feature importance ────────────────────────────────────
    ax_fi = fig.add_subplot(gs[1, 2:4])
    fi     = np.array(results["Random Forest"]["imp"])*100
    sidx   = np.argsort(fi)
    ax_fi.barh([FEAT_LABELS[i] for i in sidx], [fi[i] for i in sidx],
               color=[P["teal"] if fi[i]>12 else P["gold"] if fi[i]>6 else P["lavender"] for i in sidx],
               alpha=0.85, edgecolor="white")
    ax_fi.set_xlabel("Importance (%)"); ax_fi.tick_params(labelsize=8)
    ax_fi.set_title("Feature Importance (RF)", fontweight="bold", fontsize=12)

    # ── Row 1 col 4: failure modes ────────────────────────────────────────────
    ax_pie = fig.add_subplot(gs[1, 4])
    ft_labels = FAILURE_MODES
    ft_vals   = [46,115,95,98,19]
    ax_pie.pie(ft_vals, labels=ft_labels, autopct="%1.0f%%", colors=C,
               startangle=90, wedgeprops=dict(edgecolor="white",linewidth=1.5))
    ax_pie.set_title("Failure Modes", fontweight="bold", fontsize=12)

    # ── Row 2 col 0-1: Anomaly scores ────────────────────────────────────────
    ax_anom = fig.add_subplot(gs[2, :2])
    ax_anom.hist(anom_scores[anom_preds==1],  bins=30, color=P["teal"],  alpha=0.7,
                 label="Normal", edgecolor="white")
    ax_anom.hist(anom_scores[anom_preds==-1], bins=10, color=P["coral"], alpha=0.85,
                 label="Anomaly", edgecolor="white")
    ax_anom.set_xlabel("Anomaly Score"); ax_anom.set_ylabel("Count")
    ax_anom.set_title("Isolation Forest — Score Distribution", fontweight="bold", fontsize=12)
    ax_anom.legend(fontsize=9)

    # ── Row 2 col 2-3: RUL ───────────────────────────────────────────────────
    ax_rul = fig.add_subplot(gs[2, 2:4])
    ax_rul.fill_between(rul_df["wear"], rul_df["fail_prob"], color=P["coral"], alpha=0.25)
    ax_rul.plot(rul_df["wear"], rul_df["fail_prob"], color=P["coral"], lw=2.5)
    ax_rul.axhline(30, color=P["gold"],    ls="--", lw=1.5, label="Warning 30%")
    ax_rul.axhline(60, color=P["crimson"], ls="--", lw=1.5, label="Critical 60%")
    ax_rul.set_xlabel("Tool Wear (min)"); ax_rul.set_ylabel("Failure Prob (%)")
    ax_rul.set_title("RUL — Failure Probability", fontweight="bold", fontsize=12)
    ax_rul.set_ylim(0,100); ax_rul.legend(fontsize=8)

    # ── Row 2 col 4: metrics table ────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2, 4])
    ax_tbl.axis("off")
    best = "Ensemble (RF+GBT)"
    rows = [
        ["Metric","Value"],
        ["Accuracy", f"{results[best]['acc']*100:.1f}%"],
        ["Precision",f"{results[best]['prec']*100:.1f}%"],
        ["Recall",   f"{results[best]['rec']*100:.1f}%"],
        ["F1",       f"{results[best]['f1']:.3f}"],
        ["ROC-AUC",  f"{results[best]['auc']:.3f}"],
        ["CV-F1",    f"{results[best]['cv_f1']:.3f}"],
    ]
    tbl = ax_tbl.table(cellText=rows[1:], colLabels=rows[0],
                       cellLoc="center", loc="center",
                       colColours=[P["teal"],P["teal"]],
                       cellColours=[[P["bg"],P["panel"]]]*6)
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.6)
    ax_tbl.set_title(f"{best}\nMetrics", fontweight="bold", fontsize=11)

    plt.savefig(f"{FIGURES_DIR}/09_research_summary_dashboard.png",
                dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close("all")
    print(f"  [FIG] → {FIGURES_DIR}/09_research_summary_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    from src.utils import print_banner
    print_banner("Generating All Figures …")

    # Load data and pre-trained artefacts
    df = load_data()
    Xtr, Xte, y_tr, y_te, scaler, X_te_df = make_splits(df)

    # Reload trained models
    from sklearn.ensemble import (RandomForestClassifier,
                                   GradientBoostingClassifier, VotingClassifier)
    from sklearn.linear_model import LogisticRegression

    rf       = joblib.load(f"{MODEL_DIR}/random_forest.pkl")
    gbt      = joblib.load(f"{MODEL_DIR}/gradient_boost.pkl")
    lr       = joblib.load(f"{MODEL_DIR}/logistic_reg.pkl")
    ensemble = joblib.load(f"{MODEL_DIR}/ensemble_rfgbt.pkl")

    models_map = {
        "Random Forest":     rf,
        "Gradient Boost":    gbt,
        "Logistic Reg":      lr,
        "Ensemble (RF+GBT)": ensemble,
    }

    results = {}
    print("\n[EVAL] Re-evaluating models …")
    for name, mdl in models_map.items():
        m = evaluate_model(mdl, Xte, y_te, name)
        m["cv_f1"]  = 0.0  # placeholder (CV already done in train.py)
        m["cv_std"] = 0.0
        results[name] = m

    # Load CV results from saved metrics
    with open(f"{RESULTS_DIR}/metrics.json") as f:
        saved = json.load(f)
    for name in results:
        if name in saved:
            results[name]["cv_f1"]  = saved[name].get("cv_f1", 0.0)
            results[name]["cv_std"] = saved[name].get("cv_std", 0.0)

    # Anomaly results
    with open(f"{RESULTS_DIR}/anomaly_results.json") as f:
        anom_data = json.load(f)
    anom_preds  = np.array(anom_data["preds"])
    anom_scores = np.array(anom_data["scores"])

    # RUL
    rul_df = pd.read_csv(f"{RESULTS_DIR}/rul_estimates.csv")

    # No-SMOTE results for comparison fig
    from sklearn.ensemble import RandomForestClassifier as RFC
    rf_no_smote  = RFC(**RF_PARAMS); rf_no_smote.fit(Xtr, y_tr)
    gbt_no_smote = GradientBoostingClassifier(**GBT_PARAMS); gbt_no_smote.fit(Xtr, y_tr)
    results_no_smote = {
        "Random Forest":  evaluate_model(rf_no_smote,  Xte, y_te, "RF (no SMOTE)"),
        "Gradient Boost": evaluate_model(gbt_no_smote, Xte, y_te, "GBT (no SMOTE)"),
    }
    results_smote = {
        "Random Forest":  results["Random Forest"],
        "Gradient Boost": results["Gradient Boost"],
    }

    print("\n[FIG] Generating figures:")
    fig_dataset_overview(df)
    fig_model_comparison(results)
    fig_confusion_matrices(results)
    fig_pr_roc(results, y_te)
    fig_xai(results)
    fig_smote_impact(results_no_smote, results_smote)
    fig_anomaly(anom_preds, anom_scores, X_te_df, y_te)
    fig_rul(rul_df)
    fig_summary_dashboard(results, rul_df, anom_preds, anom_scores, y_te)

    print_banner("All 9 figures saved ✓")
    print(f"  → {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
