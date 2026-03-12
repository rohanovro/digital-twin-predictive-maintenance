"""
config.py
=========
Central configuration for the entire Digital Twin project.
Edit paths and hyperparameters here — all other scripts import from this file.
"""

import os

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, "data", "ai4i2020.csv")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
FIGURES_DIR  = os.path.join(BASE_DIR, "results", "figures")

for d in [MODEL_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ── DATASET ───────────────────────────────────────────────────────────────────
RANDOM_STATE     = 42
TEST_SIZE        = 0.20
FAILURE_COL      = "Machine failure"
FAILURE_MODES    = ["TWF", "HDF", "PWF", "OSF", "RNF"]
CONTAMINATION    = 0.034   # Isolation Forest — matches real failure rate

RAW_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

ALL_FEATURES = [
    "Type_enc",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temp_diff",
    "power",
    "wear_torque",
    "speed_wear",
    "power_temp",           # NEW: power × temp_diff
    "wear_speed_torque",    # NEW: high-order fatigue proxy
]

FEAT_LABELS = [
    "Machine Type", "Air Temp", "Proc Temp", "Rot Speed",
    "Torque", "Tool Wear", "Temp Diff", "Power",
    "Wear×Torque", "Speed×Wear", "Power×TempDiff", "Wear×Speed×Torque",
]

# ── COLOUR PALETTE (warm teal / coral / gold — not black) ────────────────────
PALETTE = {
    "teal":     "#2EC4B6",
    "coral":    "#E76F51",
    "gold":     "#F4A261",
    "mint":     "#52B788",
    "lavender": "#9B89C4",
    "sky":      "#48CAE4",
    "rose":     "#E9C46A",
    "crimson":  "#E63946",
    "bg":       "#F8F9FB",
    "panel":    "#FFFFFF",
    "text":     "#2D3142",
    "dim":      "#8892A4",
    "border":   "#E2E8F0",
}
COLORS = [
    PALETTE["teal"], PALETTE["coral"], PALETTE["gold"],
    PALETTE["mint"], PALETTE["lavender"], PALETTE["sky"], PALETTE["rose"],
]

# ── MATPLOTLIB RC ─────────────────────────────────────────────────────────────
RC = {
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor":   PALETTE["panel"],
    "axes.edgecolor":   PALETTE["border"],
    "axes.labelcolor":  PALETTE["text"],
    "xtick.color":      PALETTE["dim"],
    "ytick.color":      PALETTE["dim"],
    "text.color":       PALETTE["text"],
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       PALETTE["border"],
    "grid.linewidth":   0.8,
    "grid.alpha":       0.6,
    "figure.dpi":       120,
}

# ── MODEL HYPERPARAMETERS ─────────────────────────────────────────────────────
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
GBT_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=4,
    subsample=0.85,
    random_state=RANDOM_STATE,
)
LR_PARAMS = dict(
    class_weight="balanced",
    max_iter=1000,
    random_state=RANDOM_STATE,
)

# ── SIMULATION ────────────────────────────────────────────────────────────────
SIM_WEAR_RATE   = 2.5     # tool wear minutes per simulation tick
SIM_MAX_WEAR    = 250     # maximum tool wear before forced stop
ALERT_WARN      = 0.30    # failure prob threshold for WARNING
ALERT_HIGH      = 0.60    # failure prob threshold for HIGH RISK
ALERT_CRITICAL  = 0.80    # failure prob threshold for CRITICAL
