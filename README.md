<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=30&pause=1000&color=2EC4B6&center=true&vCenter=true&width=800&lines=AI-Driven+Digital+Twin;Predictive+Maintenance+%7C+Smart+Manufacturing;Industry+4.0+%7C+IEEE+Research+Project" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.6%2B-11557C?style=for-the-badge)](https://matplotlib.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-2EC4B6?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Ready-E76F51?style=for-the-badge)]()
[![IEEE](https://img.shields.io/badge/IEEE-Publication%20Target-F4A261?style=for-the-badge)](https://ieee.org)

<br/>

> ### *"AI-Enhanced Digital Twin Framework for Predictive Maintenance of Industrial Machinery Using Sensor Fusion"*
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
- [Architecture](#-architecture)
- [Dataset & Features](#-dataset--features)
- [Results](#-results)
- [Figures](#-figures)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Methodology](#-methodology)
- [Research Contributions](#-research-contributions)
- [Citation](#-citation)

---

## 🌟 Overview

A **full-stack, research-grade AI Digital Twin** for predictive maintenance of industrial CNC machinery, implementing a novel **dual-layer detection system** that combines supervised machine learning with unsupervised anomaly detection — all wrapped in a live real-time simulation environment.

### What's implemented

| Layer | Technology | What it does |
|-------|-----------|-------------|
| **Data** | AI4I 2020 (UCI) | Real industrial benchmark, 10K samples |
| **Feature Engineering** | Physics-based sensor fusion | 12 features from 5 raw sensors |
| **Oversampling** | SMOTE (synthetic) | Recall: 75% → 85% on minority class |
| **Supervised ML** | Random Forest + Gradient Boosting + Ensemble | Failure classification |
| **Unsupervised ML** | Isolation Forest | Novel anomaly detection |
| **Explainable AI** | Gini Feature Importance | Which sensors cause failures |
| **RUL Estimation** | Probability-based regression | How long until failure |
| **Digital Twin** | Real-time simulation | Live sensor stream + AI inference |
| **Dashboard** | Interactive HTML | Visual machine twin with all KPIs |

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
║  Real-time simulation · Scenario testing · CSV logging      ║
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

---

## 📈 Figures

All 9 figures — warm teal/coral/gold palette, publication-ready at 150 DPI.

---

### 01 — Dataset Overview
> Sensor distributions split by failure/normal · Failure mode pie · Correlation heatmap

![Dataset Overview](https://raw.githubusercontent.com/rohanovro/digital-twin-predictive-maintenance/main/results/figures/01_dataset_overview.png)

---

### 02 — Model Comparison
> All-model metric bars · ROC curves · 5-Fold CV F1 with error bars

![Model Comparison](https://raw.githubusercontent.com/rohanovro/digital-twin-predictive-maintenance/main/results/figures/02_model_comparison.png)

---

### 03 — Confusion Matrices (All 4 Models)
> Side-by-side: Random Forest · Gradient Boosting · Logistic Regression · Ensemble

![Confusion Matrices](https://raw.githubusercontent.com/rohanovro/digital-twin-predictive-maintenance/main/results/figures/03_confusion_matrices.png)

---

### 04 — Precision-Recall & ROC Curves
> All models overlaid · Baseline comparison · Critical for imbalanced datasets

![PR ROC Curves](https://raw.githubusercontent.com/rohanovro/digital-twin-predictive-maintenance/main/results/figures/04_pr_roc_curves.png)

---

### 05 — Explainable AI — Feature Importance
> RF + GBT ranked importance bars · Failure driver category donut chart

![XAI Feature Importance](https://raw.githubusercontent.com/rohanovro/digital-twin-predictive-maintenance/main/results/figures/05_xai_feature_importance.png)

---

### 06 — SMOTE Impact
> Before vs After SMOTE: Recall +10pp · F1 improvement · Precision trade-off

![SMOTE Impact](https://raw.githubusercontent.com/rohanovro/digital-twin-predictive-maintenance/main/results/figures/06_smote_impact.png)

---

### 07 — Anomaly Detection
> Score distribution · Speed-Torque anomaly scatter · Overlap with actual failures

![Anomaly Detection](https://raw.githubusercontent.com/rohanovro/digital-twin-predictive-maintenance/main/results/figures/07_anomaly_detection.png)

---

### 08 — Remaining Useful Life (RUL)
> Failure probability vs tool wear · RUL estimate curve · Maintenance trigger zone

![RUL Prediction](https://raw.githubusercontent.com/rohanovro/digital-twin-predictive-maintenance/main/results/figures/08_rul_prediction.png)

---

### 09 — Research Summary Dashboard
> All KPIs + all key charts in one publication-ready figure

![Research Summary Dashboard](https://raw.githubusercontent.com/rohanovro/digital-twin-predictive-maintenance/main/results/figures/09_research_summary_dashboard.png)

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
│   ├── simulate.py                     # Real-time Digital Twin simulation
│   └── predict.py                      # Single-sample inference + batch mode
│
├── 📂 models/
│   ├── random_forest.pkl               # Trained RF (joblib)
│   ├── gradient_boost.pkl              # Trained GBT
│   ├── logistic_reg.pkl                # Trained LR
│   ├── ensemble_rfgbt.pkl              # Soft-voting ensemble
│   ├── isolation_forest.pkl            # Trained IF anomaly detector
│   └── scaler.pkl                      # Fitted StandardScaler
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
│   └── paper_outline.md                # Full IEEE paper structure + results tables
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/rohanovro/digital-twin-predictive-maintenance.git
cd digital-twin-predictive-maintenance
pip install -r requirements.txt
```

### 2. Train the Pipeline

```bash
python src/train.py
```

### 3. Generate All Figures

```bash
python src/visualise.py
```

### 4. Run Real-Time Simulation

```bash
python src/simulate.py --ticks 100 --interval 0.3

# Stress scenarios
python src/simulate.py --scenario overheat
python src/simulate.py --scenario overload
```

### 5. Single Prediction

```bash
python src/predict.py --air_temp 302.5 --proc_temp 314 --speed 1400 --torque 58 --wear 185

# Interactive mode
python src/predict.py --interactive
```

### 6. Open Dashboard

```bash
open dashboard/digital_twin_dashboard.html
```

---

## 🔬 Methodology

### Dual-Layer Detection System

```
         ┌─ RF/GBT Classifier ─► P(fail) from known patterns
Reading ──┤
         └─ Isolation Forest  ─► Anomaly score from normal envelope
                    ↓
            Combined Alert Level
```

### SMOTE Strategy
Applied **only to training data** — test set stays at original 3.4% failure rate for honest evaluation. Effect: Recall +10pp.

### RUL Formula
```
RUL(t) = (1 − P_failure(t)) × (Max_wear − Current_wear(t))
```

### Alert Levels
| Level | Threshold | Action |
|-------|----------|--------|
| ✅ OK | P < 30% | Continue |
| ⚠️ WARNING | 30–60% | Increase monitoring |
| 🔴 HIGH RISK | 60–80% | Schedule maintenance |
| 🚨 CRITICAL | > 80% | Immediate shutdown |

---

## 🏆 Research Contributions

1. **Dual-layer detection** — supervised RF + unsupervised Isolation Forest
2. **Physics-informed sensor fusion** — 12 engineered features from 5 raw sensors
3. **Novel XAI finding** — thermal gradient outperforms raw temperatures for HDF prediction
4. **SMOTE + stratified evaluation** — honest recall improvement without data leakage
5. **Four-scenario simulation** — stress-tests the Digital Twin under different failure modes
6. **Open-source, reproducible** — single command reproduces all results

---

## 📝 Citation

```bibtex
@article{digitaltwin_pm_2025,
  title   = {AI-Enhanced Digital Twin Framework for Predictive Maintenance
             of Industrial Machinery Using Sensor Fusion},
  author  = {Rohanov, R.},
  journal = {IEEE Transactions on Industrial Electronics},
  year    = {2025},
  url     = {https://github.com/rohanovro/digital-twin-predictive-maintenance}
}
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for IEEE Publication · Erasmus Mundus Portfolio · Industry 4.0 Research**

[![GitHub stars](https://img.shields.io/github/stars/rohanovro/digital-twin-predictive-maintenance?style=social)](https://github.com/rohanovro/digital-twin-predictive-maintenance)

*If this project helped you, please ⭐ star the repo!*

</div>
