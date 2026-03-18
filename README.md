<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=30&pause=1000&color=2EC4B6&center=true&vCenter=true&width=800&lines=AI-Driven+Digital+Twin;Predictive+Maintenance+%7C+Smart+Manufacturing;Industry+4.0+%7C+IEEE+Research+Project" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.6%2B-11557C?style=for-the-badge)](https://matplotlib.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-2EC4B6?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Ready-E76F51?style=for-the-badge)]()
[![Institution](https://img.shields.io/badge/JUST-Bangladesh-2EC4B6?style=for-the-badge)]()

<br/>

> ### *"AI-Enhanced Digital Twin Framework for Predictive Maintenance of Industrial Machinery Using Sensor Fusion"*
>
> **Mahmudul Hasan Rohan** · Jashore University of Science and Technology, Bangladesh
>
> **Dataset:** AI4I 2020 Predictive Maintenance (UCI) &nbsp;|&nbsp; **Target:** IEEE Industrial Electronics / IEOM 2025

<br/>

| Metric | Result |
|--------|--------|
| 🎯 ROC-AUC | **0.979** |
| ✅ Accuracy | **99.1%** |
| 🔁 Recall (post-SMOTE) | **85.3%** |
| 🤝 Ensemble F1 | **0.868** |
| 🔍 Anomalies Detected | **52 / 2000** |

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Research Questions](#-research-questions)
- [Architecture](#-architecture)
- [Dataset & Features](#-dataset--features)
- [Results](#-results)
- [Figures](#-figures)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Methodology](#-methodology)
- [Limitations & Future Work](#-limitations--future-work)
- [Research Contributions](#-research-contributions)
- [Research Status](#-research-status)

---

## 🌟 Overview

Predictive maintenance in industrial CNC machinery remains an open challenge: rule-based systems miss novel failure modes, while pure supervised models fail when labelled failure data is scarce. This project investigates whether a **dual-layer AI framework** — combining supervised classification with unsupervised anomaly detection — can improve failure detection coverage without sacrificing precision, and whether **physics-informed sensor fusion** can surface failure drivers that raw sensor readings alone cannot reveal.

The result is a full research pipeline built on the AI4I 2020 benchmark dataset, including a real-time digital twin simulation environment and an interactive diagnostic dashboard.

### What's implemented

| Layer | Technology | What it does |
|-------|-----------|-------------|
| **Data** | AI4I 2020 (UCI) | Real industrial benchmark, 10K samples |
| **Feature Engineering** | Physics-based sensor fusion | 12 features from 5 raw sensors |
| **Oversampling** | SMOTE (train set only) | Recall: 75% → 85% on minority class |
| **Supervised ML** | Random Forest + Gradient Boosting + Ensemble | Failure classification |
| **Unsupervised ML** | Isolation Forest | Novel anomaly detection |
| **Explainable AI** | Gini Feature Importance | Which sensors drive failures |
| **RUL Estimation** | Probability-based regression | Remaining useful life prediction |
| **Digital Twin** | Dataset replay simulation | Live inference on real sensor sequences |
| **Dashboard** | Interactive HTML | Visual machine twin with all KPIs |

---

## ❓ Research Questions

This project was designed around three open questions that motivated the design choices:

**RQ1:** Does combining supervised failure classification with unsupervised anomaly detection improve coverage of novel, previously-unseen failure patterns — and what is the precision trade-off?

**RQ2:** Do physics-informed engineered features (e.g. thermal gradient, mechanical power) outperform raw sensor readings as failure predictors, and does this generalise across machine types beyond CNC?

**RQ3:** Can a probability-based RUL formula derived from classifier output provide actionable maintenance scheduling, or does it require calibration against actual time-to-failure ground truth?

These questions remain partially open and motivate the future work described below.

---

## 🏗️ Architecture

```
╔══════════════════════════════════════════════════════════════╗
║                    PHYSICAL MACHINE                          ║
║   Motor · Spindle · Tool · Bearings · Thermal System        ║
╚══════════════════════╦═══════════════════════════════════════╝
                       ║  5 raw sensor channels
                       ▼
╔══════════════════════════════════════════════════════════════╗
║               SENSOR FUSION LAYER                           ║
║  Air Temp · Proc Temp · Speed · Torque · Tool Wear          ║
║  + temp_diff · power · wear_torque · speed_wear             ║
║  + power_temp · wear_speed_torque        (12 features total) ║
╚══════════════════════╦═══════════════════════════════════════╝
                       ║
          ┌────────────┴────────────┐
          ▼                         ▼
╔══════════════════╗     ╔═══════════════════════╗
║  SUPERVISED AI   ║     ║   UNSUPERVISED AI      ║
║  Random Forest   ║     ║   Isolation Forest     ║
║  Gradient Boost  ║     ║   (no labels needed)   ║
║  Soft Ensemble   ║     ╚═══════════╦═══════════╝
╚════════╦═════════╝                 ║
         ║  Failure Prob             ║  Anomaly Flag
         ╚═══════════╦══════════════╝
                     ▼
╔══════════════════════════════════════════════════════════════╗
║               DIGITAL TWIN LAYER                            ║
║  Health Score · RUL Estimate · Alert Level · XAI            ║
║  Dataset replay simulation · Scenario testing · CSV logging ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 📊 Dataset & Features

### AI4I 2020 Predictive Maintenance Dataset

| Property | Detail |
|----------|--------|
| Source | [UCI ML Repository — ID 601](https://archive.ics.uci.edu/dataset/601) |
| Samples | 10,000 |
| Failure rate | 3.39% (339 failures) |
| Missing values | 0 |
| Failure modes | TWF · HDF · PWF · OSF · RNF |

### Engineered Features (Sensor Fusion)

Six cross-domain features were constructed from the five raw sensor channels based on physical relationships between heat, mechanical load, and wear:

```python
df["temp_diff"]          = proc_temp - air_temp            # Thermal gradient
df["power"]              = speed * torque                   # Mechanical power
df["wear_torque"]        = tool_wear * torque               # Wear under load
df["speed_wear"]         = speed * tool_wear                # Fatigue proxy
df["power_temp"]         = power * temp_diff                # Thermal overload
df["wear_speed_torque"]  = tool_wear * speed * torque       # High-order fatigue
```

---

## 🎯 Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV-F1 |
|-------|----------|-----------|--------|----|---------|-------|
| **Random Forest** | 99.05% | 86.6% | **85.3%** | 0.859 | 0.978 | 0.993 |
| **Gradient Boost** | 99.20% | **96.4%** | 79.4% | 0.871 | 0.973 | 0.996 |
| Logistic Reg | 86.15% | 17.9% | 85.3% | 0.295 | 0.934 | 0.843 |
| **Ensemble (RF+GBT)** ⭐ | 99.15% | 91.8% | 82.4% | **0.868** | **0.979** | **0.996** |

### Feature Importance (Top 6)

```
Rotational Speed   ████████████████████  19.3%   ← Primary mechanical stress
Power (Spd×Tq)    ███████████████████   18.2%   ← Overloading indicator
Torque             ██████████████████    17.8%   ← Direct overstrain
Tool Wear          ████████████          11.4%   ← Wear accumulation
Wear × Torque      ██████████            10.3%   ← Non-linear wear under load
Temp Gradient      █████████              9.9%   ← Novel thermal finding ★
```

★ The engineered thermal gradient feature (`temp_diff`) ranked above the raw air and process temperature readings individually — a finding that motivates RQ2.

---

## 📈 Figures

All figures are publication-ready at 150 DPI using a consistent warm teal/coral/gold palette.

### Figure 01 — Dataset Overview
> Sensor distributions split by failure/normal · Failure mode pie · Correlation heatmap

![Dataset Overview](01_dataset_overview.png)

---

### Figure 02 — Model Comparison
> All-model metric bars · ROC curves · 5-Fold CV F1 with error bars

![Model Comparison](02_model_comparison.png)

---

### Figure 03 — Confusion Matrices
> Side-by-side: Random Forest · Gradient Boosting · Logistic Regression · Ensemble

![Confusion Matrices](03_confusion_matrices.png)

---

### Figure 04 — Precision-Recall & ROC Curves
> All models overlaid · Baseline comparison · Critical for imbalanced datasets

![PR ROC Curves](04_pr_roc_curves.png)

---

### Figure 05 — Explainable AI — Feature Importance
> RF + GBT ranked importance bars · Failure driver category donut chart

![XAI Feature Importance](05_xai_feature_importance.png)

---

### Figure 06 — SMOTE Impact
> Before vs After SMOTE: Recall +10pp · F1 improvement · Precision trade-off

![SMOTE Impact](06_smote_impact.png)

---

### Figure 07 — Anomaly Detection
> Score distribution · Speed-Torque anomaly scatter · Overlap with actual failures

![Anomaly Detection](07_anomaly_detection.png)

---

### Figure 08 — Remaining Useful Life (RUL)
> Failure probability vs tool wear · RUL estimate curve · Maintenance trigger zone

![RUL Prediction](08_rul_prediction.png)

---

### Figure 09 — Research Summary Dashboard
> All KPIs + all key charts in one publication-ready figure

![Research Summary Dashboard](09_research_summary_dashboard.png)

---

## 📁 Project Structure

```
digital-twin-predictive-maintenance/
│
├── 📂 data/
│   └── ai4i2020.csv                    # AI4I 2020 dataset (UCI ML Repo)
│
├── 📂 src/
│   ├── config.py                       # Central config: paths, palette, hyperparams
│   ├── utils.py                        # Data loading, feature engineering, evaluation
│   ├── train.py                        # Full training pipeline (SMOTE + 4 models + anomaly + RUL)
│   ├── visualise.py                    # All 9 publication-ready figures
│   ├── simulate.py                     # Digital Twin simulation (replay + synthetic modes)
│   └── predict.py                      # Single-sample inference + batch mode
│
├── 📂 models/                          # Generated by train.py — see Quick Start
│   ├── random_forest.pkl
│   ├── gradient_boost.pkl
│   ├── logistic_reg.pkl
│   ├── ensemble_rfgbt.pkl
│   ├── isolation_forest.pkl
│   └── scaler.pkl
│
├── 📂 results/
│   ├── metrics.json                    # All model metrics (JSON)
│   ├── rul_estimates.csv               # RUL across wear levels
│   ├── anomaly_results.json            # Isolation Forest output
│   └── 📂 figures/                     # All 9 research figures (150 DPI)
│
├── 📂 dashboard/
│   └── digital_twin_dashboard.html     # Interactive browser dashboard
│
├── 📂 docs/
│   ├── paper_outline.md                # Full paper structure + results tables
│   └── research_summary.md            # 1-page research overview for collaborators
│
├── 📂 tests/
│   ├── test_utils.py                   # Unit tests for feature engineering
│   └── test_predict.py                 # Inference validation tests
│
├── .github/workflows/test.yml          # CI: runs pytest on every push
├── requirements.txt
├── LICENSE
└── README.md
```

> **Note:** `.pkl` model files are excluded from version control. Run `python src/train.py` to regenerate all models locally (takes ~2 minutes).

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/rohanovro/digital-twin-predictive-maintenance.git
cd digital-twin-predictive-maintenance
pip install -r requirements.txt
```

### 2. Download Dataset

Download `ai4i2020.csv` from [UCI ML Repository (ID 601)](https://archive.ics.uci.edu/dataset/601) and place it in the `data/` folder.

### 3. Train the Pipeline

```bash
python src/train.py
```

### 4. Generate All Figures

```bash
python src/visualise.py
```

### 5. Run Simulation

```bash
# Replay real AI4I dataset rows (recommended)
python src/simulate.py --mode replay --ticks 100

# Synthetic random sensor stream
python src/simulate.py --mode synthetic --ticks 100 --interval 0.3

# Stress scenarios (synthetic mode)
python src/simulate.py --mode synthetic --scenario overheat
python src/simulate.py --mode synthetic --scenario overload
```

### 6. Single Prediction

```bash
python src/predict.py --air_temp 302.5 --proc_temp 314 --speed 1400 --torque 58 --wear 185

# Interactive mode
python src/predict.py --interactive
```

### 7. Open Dashboard

```bash
open dashboard/digital_twin_dashboard.html
```

### 8. Run Tests

```bash
pytest tests/
```

---

## 🔬 Methodology

### Dual-Layer Detection System

```
         ┌─ RF/GBT Classifier ─► P(fail) from known failure patterns
Reading ──┤
         └─ Isolation Forest  ─► Anomaly score from normal operating envelope
                    ↓
            Combined Alert Level
```

The two layers are complementary: the supervised classifier is high-precision on known failure modes; the Isolation Forest catches operating points that deviate from normal even when they don't match a known failure signature.

### SMOTE Strategy

Applied **only to the training set** — the test set retains the original 3.4% failure rate for honest evaluation. This prevents the recall improvement from being an artifact of an artificially balanced test set. Effect: Recall +10pp with a small precision trade-off.

### RUL Formula

```
RUL(t) = (1 − P_failure(t)) × (Max_wear − Current_wear(t))
```

This is a probability-weighted estimate, not a calibrated ground-truth prediction. See Limitations for the distinction.

### Alert Levels

| Level | Threshold | Action |
|-------|----------|--------|
| ✅ OK | P < 30% | Continue operation |
| ⚠️ WARNING | 30–60% | Increase monitoring frequency |
| 🔴 HIGH RISK | 60–80% | Schedule maintenance |
| 🚨 CRITICAL | > 80% | Immediate shutdown |

---

## ⚠️ Limitations & Future Work

Being explicit about limitations is part of honest research practice.

**Current limitations:**

- **Simulation uses dataset replay, not live sensors.** The digital twin simulation replays rows from the AI4I dataset with artificial timestamps. It is a prototype demonstrating the inference pipeline — not a live sensor integration. Real deployment would require an OPC-UA or MQTT data bridge.
- **RUL is probability-weighted, not ground-truth calibrated.** The RUL formula produces relative estimates useful for scheduling decisions, but is not validated against actual time-to-failure measurements. Calibration against field data is required before operational use.
- **Single dataset.** All results are on AI4I 2020. Generalisation to other machine types, manufacturers, or operating conditions has not been tested.
- **No hardware-in-the-loop testing.** The framework has not been validated on physical CNC equipment.

**Future work:**

1. Validate the thermal gradient finding (RQ2) on the PRONOSTIA bearing dataset and PHM 2012 challenge data to test cross-dataset generalisation.
2. Implement MQTT-based live sensor integration to replace dataset replay with a real data stream.
3. Calibrate the RUL estimator against actual time-to-failure labels using survival analysis (Cox proportional hazards or Weibull AFT).

---

## 🏆 Research Contributions

1. **Dual-layer detection** — supervised RF/GBT ensemble + unsupervised Isolation Forest, addressing the labelled-data scarcity problem
2. **Physics-informed sensor fusion** — 12 engineered features from 5 raw sensors, grounded in mechanical and thermal domain knowledge
3. **Novel XAI finding** — engineered thermal gradient (`temp_diff`) outperforms raw temperature readings individually for HDF prediction (motivates RQ2)
4. **Honest SMOTE evaluation** — SMOTE applied to training data only, preserving test set class distribution for unbiased recall measurement
5. **Four-scenario simulation** — stress-tests the Digital Twin under overheat, overload, wear, and normal operating conditions
6. **Open-source, reproducible** — single command (`python src/train.py`) reproduces all models and metrics

---

## 📝 Research Status

This work is currently being prepared for submission to an IEEE conference in the area of AI-driven predictive maintenance and digital twin systems for Industry 4.0 applications.

**Author:** Mahmudul Hasan Rohan  
**Institution:** Jashore University of Science and Technology, Bangladesh  
**Repository:** https://github.com/rohanovro/digital-twin-predictive-maintenance  
**Year:** 2025

If you use this work, please cite the repository until a formal publication is available.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

*

</div>
