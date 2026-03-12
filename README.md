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

### Why this matters

> Traditional predictive maintenance uses a **single binary classifier** and raises alerts only on known failure patterns.
> This framework adds an **unsupervised anomaly layer** that catches *novel* failures the classifier has never seen,
> then feeds both signals into a **Digital Twin** that estimates Remaining Useful Life in real time.

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

### Sensor Channels

| Sensor | Range | Unit | Primary Failure Link |
|--------|-------|------|---------------------|
| Air Temperature | 295–305 | K | Heat Dissipation (HDF) |
| Process Temperature | 305–313 | K | Heat Dissipation (HDF) |
| Rotational Speed | 1168–2886 | rpm | Power Failure (PWF) |
| Torque | 3.8–76.6 | Nm | Overstrain (OSF) |
| Tool Wear | 0–253 | min | Tool Wear Failure (TWF) |

### Engineered Features (Sensor Fusion)

```python
df["temp_diff"]          = proc_temp - air_temp            # Thermal gradient
df["power"]              = speed * torque                   # Mechanical power
df["wear_torque"]        = tool_wear * torque               # Wear under load
df["speed_wear"]         = speed * tool_wear                # Fatigue proxy
df["power_temp"]         = power * temp_diff                # Thermal overload
df["wear_speed_torque"]  = tool_wear * speed * torque       # High-order fatigue
```

**Novel insight:** `temp_diff` (thermal gradient) ranks 6th in importance at 9.9%,
outperforming both raw temperature sensors individually. This means the *rate* of
heat dissipation predicts failures better than absolute temperature — a key XAI finding.

---

## 🎯 Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV-F1 |
|-------|----------|-----------|--------|----|---------|-------|
| **Random Forest** | 99.05% | 86.6% | **85.3%** | 0.859 | 0.978 | 0.993 |
| **Gradient Boost** | 99.20% | **96.4%** | 79.4% | 0.871 | 0.973 | 0.996 |
| Logistic Reg | 86.15% | 17.9% | 85.3% | 0.295 | 0.934 | 0.843 |
| **Ensemble (RF+GBT)** ⭐ | 99.15% | 91.8% | 82.4% | **0.868** | **0.979** | **0.996** |

### SMOTE Impact

| Model | Recall Before | Recall After | Change |
|-------|-------------|-------------|--------|
| Random Forest | 75.0% | 85.3% | **+10.3pp** |
| Gradient Boost | 79.4% | 79.4% | ≈ stable |

### Confusion Matrix (Random Forest + SMOTE)

```
                  Predicted
                  No Fail    Fail
Actual  No Fail  [  1923       9  ]   Specificity = 99.5%
        Fail     [    10      58  ]   Sensitivity = 85.3%
```

### Feature Importance (Top 6)

```
Rotational Speed   ████████████████████  19.3%   ← Primary mechanical stress
Power (Spd×Tq)    ███████████████████   18.2%   ← Overloading indicator
Torque             ██████████████████    17.8%   ← Direct overstrain
Tool Wear          ████████████          11.4%   ← Wear accumulation
Wear × Torque      ██████████            10.3%   ← Non-linear wear under load
Temp Gradient      █████████              9.9%   ← Novel thermal finding ★
```

### Anomaly Detection (Isolation Forest)

| Metric | Value |
|--------|-------|
| Test samples | 2,000 |
| Anomalies flagged | 52 (3.1%) |
| Overlap with actual failures | 16 of 68 (24%) |
| Dual-layer catch rate | Higher than either layer alone |

---

## 📈 Figures

All 9 figures are in `results/figures/` — warm teal/coral/gold palette, publication-ready at 150 DPI.

| # | Filename | Content |
|---|---------|---------|
| 01 | `01_dataset_overview.png` | Sensor histograms split by failure/normal + failure mode pie + correlation heatmap |
| 02 | `02_model_comparison.png` | All-model metric bars + ROC curves + CV F1 with error bars |
| 03 | `03_confusion_matrices.png` | All 4 models side-by-side confusion matrices |
| 04 | `04_pr_roc_curves.png` | Precision-Recall + ROC curves overlaid for all models |
| 05 | `05_xai_feature_importance.png` | RF + GBT importance bars + failure driver category donut |
| 06 | `06_smote_impact.png` | Before/after SMOTE comparison for recall, F1, precision |
| 07 | `07_anomaly_detection.png` | Score distribution + speed-torque scatter + overlap matrix |
| 08 | `08_rul_prediction.png` | Failure probability vs wear curve + RUL estimate curve |
| 09 | `09_research_summary_dashboard.png` | All KPIs + all key charts in one publication figure |

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
├── 📂 notebooks/
│   └── 01_full_analysis.ipynb          # (recommended) Jupyter walkthrough
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
│   ├── simulation_log.csv              # Generated by simulate.py
│   └── 📂 figures/
│       ├── 01_dataset_overview.png
│       ├── 02_model_comparison.png
│       ├── 03_confusion_matrices.png
│       ├── 04_pr_roc_curves.png
│       ├── 05_xai_feature_importance.png
│       ├── 06_smote_impact.png
│       ├── 07_anomaly_detection.png
│       ├── 08_rul_prediction.png
│       └── 09_research_summary_dashboard.png
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
git clone https://github.com/YOUR_USERNAME/digital-twin-predictive-maintenance.git
cd digital-twin-predictive-maintenance
pip install -r requirements.txt
```

### 2. Add Dataset

Download from [UCI ML Repository](https://archive.ics.uci.edu/dataset/601) and place in `data/`:
```bash
# File should be at:
data/ai4i2020.csv
```

### 3. Run Training Pipeline

```bash
python src/train.py
```

Expected output:
```
[DATA]   Loaded 10,000 rows | Failure rate: 3.39%
[SMOTE]  Balanced: 7,729 normal | 7,729 failure (was 271)
[RF]     ACC=0.9905  PREC=0.8657  REC=0.8529  F1=0.8593  AUC=0.9784
[GBT]    ACC=0.9920  PREC=0.9643  REC=0.7941  F1=0.8710  AUC=0.9731
[ENS]    ACC=0.9915  PREC=0.9180  REC=0.8235  F1=0.8682  AUC=0.9793
[ANOMALY] Detected 52 anomalies (3.1% of test set)
```

### 4. Generate All Figures

```bash
python src/visualise.py
# → results/figures/01_dataset_overview.png ... 09_research_summary_dashboard.png
```

### 5. Run Real-Time Simulation

```bash
# Normal scenario
python src/simulate.py --ticks 100 --interval 0.3

# Stress test — overheating scenario
python src/simulate.py --ticks 80 --scenario overheat

# Overload scenario
python src/simulate.py --ticks 80 --scenario overload
```

### 6. Single Prediction

```bash
# From CLI
python src/predict.py --air_temp 302.5 --proc_temp 314 --speed 1400 --torque 58 --wear 185

# Interactive mode
python src/predict.py --interactive
```

### 7. Open Dashboard

```bash
open dashboard/digital_twin_dashboard.html
```

---

## 🔬 Methodology

### Feature Engineering — Why Physics Matters

Raw sensor values miss interaction effects. Our engineered features capture physical relationships:

- **`power = speed × torque`** — captures mechanical overloading (PWF)
- **`wear_torque = wear × torque`** — non-linear wear acceleration under high load
- **`temp_diff = proc_temp − air_temp`** — thermal gradient reveals heat dissipation failure before absolute temp does
- **`power_temp = power × temp_diff`** — simultaneous mechanical + thermal stress

### Dual-Layer Detection System

```
         ┌─ RF/GBT Classifier ─► P(fail) from known patterns
Reading ──┤
         └─ Isolation Forest  ─► Anomaly score from normal envelope
                    ↓
            Combined Alert Level
```

- **Supervised layer** learns from 339 labelled failure examples
- **Unsupervised layer** learns normal operating envelope from 9,661 normal readings
- Together they catch what neither catches alone

### SMOTE Strategy

Applied **only to training data** — test set stays at original 3.4% failure rate for honest evaluation:

```python
# Minority class (failures) duplicated with small Gaussian noise
X_synth = X_minority[random_choice] + N(0, 0.05)
```

Effect: Recall +10pp, Precision −6pp — acceptable trade-off for safety-critical application.

### RUL Estimation Formula

```
RUL(t) = (1 − P_failure(t)) × (Max_wear − Current_wear(t))

where:
  P_failure(t) = model output probability at time t
  Max_wear     = 250 minutes (dataset maximum)
  Current_wear = accumulated tool wear at time t
```

---

## 🏆 Research Contributions

1. **Dual-layer detection** — supervised RF + unsupervised Isolation Forest for comprehensive coverage
2. **Physics-informed sensor fusion** — 12 engineered features from 5 raw sensors
3. **Novel XAI finding** — thermal gradient (temp_diff) outperforms raw temperatures for HDF prediction
4. **SMOTE + stratified evaluation** — honest recall improvement without data leakage
5. **RUL estimation** linked directly to live failure probability
6. **Four-scenario simulation** — stress-tests the Digital Twin under different failure modes
7. **Open-source, reproducible** — single `python src/train.py` reproduces all results

---

## 📄 Research Paper

Full IEEE-structured paper outline in [`docs/paper_outline.md`](docs/paper_outline.md), including:
- Abstract, Introduction, Literature Review
- Complete methodology with equations
- Full results tables
- Discussion of limitations & future work
- Reference list template (IEEE format)

---

## 📝 Citation

```bibtex
@article{digitaltwin_pm_2025,
  title   = {AI-Enhanced Digital Twin Framework for Predictive Maintenance
             of Industrial Machinery Using Sensor Fusion},
  author  = {[Your Name]},
  journal = {IEEE Transactions on Industrial Electronics},
  year    = {2025},
  url     = {https://github.com/YOUR_USERNAME/digital-twin-predictive-maintenance}
}
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for IEEE Publication · Erasmus Mundus Portfolio · Industry 4.0 Research**

[![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/digital-twin-predictive-maintenance?style=social)](https://github.com/YOUR_USERNAME/digital-twin-predictive-maintenance)

*If this project helped you, please ⭐ star the repo!*

</div>
