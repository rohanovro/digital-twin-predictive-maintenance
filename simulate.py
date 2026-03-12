"""
simulate.py
===========
Real-time Digital Twin simulation.
Streams synthetic sensor readings through the trained model,
prints a live colour-coded terminal feed, and logs all ticks to CSV.

Run:
    python src/simulate.py                         # default 100 ticks
    python src/simulate.py --ticks 200 --interval 0.2
    python src/simulate.py --scenario overheat     # stress scenario
"""

import argparse, os, sys, time, csv
import numpy as np
import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    MODEL_DIR, RESULTS_DIR, ALL_FEATURES,
    SIM_WEAR_RATE, SIM_MAX_WEAR,
    ALERT_WARN, ALERT_HIGH, ALERT_CRITICAL,
    RANDOM_STATE,
)

# ── TERMINAL COLOURS ──────────────────────────────────────────────────────────
CLR = {
    "green":  "\033[92m", "yellow": "\033[93m",
    "red":    "\033[91m", "cyan":   "\033[96m",
    "white":  "\033[97m", "dim":    "\033[2m",
    "bold":   "\033[1m",  "reset":  "\033[0m",
}

def c(text, *codes):
    return "".join(CLR[k] for k in codes) + str(text) + CLR["reset"]


# ── SENSOR GENERATOR ──────────────────────────────────────────────────────────
SENSOR_NOMINAL = {
    "Air temperature [K]":    (300.0, 2.0),
    "Process temperature [K]":(310.0, 1.5),
    "Rotational speed [rpm]": (1538,  180),
    "Torque [Nm]":            (39.9,  10.0),
}

SCENARIOS = {
    "normal":   {"degrade_speed": 1.0,  "anomaly_rate": 0.03},
    "overheat": {"degrade_speed": 1.8,  "anomaly_rate": 0.08,  "temp_bias": 8.0},
    "overload": {"degrade_speed": 2.0,  "anomaly_rate": 0.06,  "torque_bias": 15.0},
    "random":   {"degrade_speed": 1.0,  "anomaly_rate": 0.15},
}


def generate_reading(wear: float, tick: int, scenario: dict) -> dict:
    rng    = np.random
    degrade = min(1.0, wear / SIM_MAX_WEAR)

    air_temp  = (SENSOR_NOMINAL["Air temperature [K]"][0]
                 + rng.randn() * SENSOR_NOMINAL["Air temperature [K]"][1]
                 + degrade * 3
                 + scenario.get("temp_bias", 0))
    proc_temp = (SENSOR_NOMINAL["Process temperature [K]"][0]
                 + rng.randn() * 1.5
                 + degrade * 10
                 + scenario.get("temp_bias", 0) * 1.2)
    speed     = max(500,
                    SENSOR_NOMINAL["Rotational speed [rpm]"][0]
                    + rng.randn() * 180
                    - degrade * 250 * scenario.get("degrade_speed", 1))
    torque    = (SENSOR_NOMINAL["Torque [Nm]"][0]
                 + rng.randn() * 10
                 + degrade * 25 * scenario.get("degrade_speed", 1)
                 + scenario.get("torque_bias", 0))

    is_anomaly = rng.random() < scenario.get("anomaly_rate", 0.03)
    if is_anomaly:
        choice = rng.choice(["speed_spike","torque_spike","temp_spike"])
        if choice == "speed_spike":   speed  *= rng.choice([0.5, 2.0])
        elif choice == "torque_spike": torque *= rng.choice([0.4, 2.2])
        else:                          proc_temp += rng.uniform(8, 18)

    temp_diff        = proc_temp - air_temp
    power            = speed * torque
    wear_torque      = wear  * torque
    speed_wear       = speed * wear
    power_temp       = power * temp_diff
    wear_speed_torque= wear  * speed * torque

    return {
        "tick":                   tick,
        "wear":                   round(wear, 1),
        "Type_enc":               1,
        "Air temperature [K]":    round(air_temp, 2),
        "Process temperature [K]":round(proc_temp, 2),
        "Rotational speed [rpm]": round(speed),
        "Torque [Nm]":            round(torque, 2),
        "Tool wear [min]":        round(wear),
        "temp_diff":              round(temp_diff, 2),
        "power":                  round(power, 1),
        "wear_torque":            round(wear_torque, 1),
        "speed_wear":             round(speed_wear, 1),
        "power_temp":             round(power_temp, 1),
        "wear_speed_torque":      round(wear_speed_torque, 1),
        "is_anomaly":             is_anomaly,
    }


# ── HEALTH BAR ────────────────────────────────────────────────────────────────
def health_bar(prob: float, width: int = 25) -> str:
    health = 1 - prob
    filled = int(health * width)
    bar    = "█" * filled + "░" * (width - filled)
    col    = "green" if prob < ALERT_WARN else "yellow" if prob < ALERT_HIGH else "red"
    return c(f"[{bar}]", col)


def risk_tag(prob: float) -> str:
    if   prob < ALERT_WARN:     return c("● LOW RISK",      "green",  "bold")
    elif prob < ALERT_HIGH:     return c("▲ ELEVATED",       "yellow", "bold")
    elif prob < ALERT_CRITICAL: return c("■ HIGH RISK",      "red",    "bold")
    else:                       return c("✖ CRITICAL",       "red",    "bold")


# ── MAIN SIMULATION LOOP ──────────────────────────────────────────────────────
def run_simulation(model, iso_model, scaler,
                   n_ticks=100, interval=0.4,
                   wear_rate=SIM_WEAR_RATE, scenario_name="normal"):

    scenario = SCENARIOS.get(scenario_name, SCENARIOS["normal"])
    rng      = np.random.default_rng(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    log_rows      = []
    total_anom    = 0
    total_alerts  = 0
    prob_history  = []

    # CSV logger
    log_path = f"{RESULTS_DIR}/simulation_log.csv"
    log_cols = ["tick","wear","air_temp","proc_temp","speed","torque",
                "fail_prob","anomaly_flag","alert_level"]

    print(c("═"*70, "cyan"))
    print(c(f"  AI DIGITAL TWIN — REAL-TIME SIMULATION", "bold","white"))
    print(c(f"  Scenario: {scenario_name.upper()} | Ticks: {n_ticks} | Wear rate: {wear_rate} min/tick", "dim"))
    print(c("═"*70, "cyan"))
    print()

    header = (f"{'Tick':>5}  {'Wear':>6}  {'AirT':>7}  {'Speed':>6}  "
              f"{'Torque':>7}  {'FailProb':>9}  Health          Status")
    print(c(header, "dim"))
    print(c("─"*70, "dim"))

    for tick in range(1, n_ticks + 1):
        wear = min(SIM_MAX_WEAR, tick * wear_rate + rng.standard_normal() * 1.5)
        rd   = generate_reading(wear, tick, scenario)

        X     = np.array([[rd[f] for f in ALL_FEATURES]])
        Xsc   = scaler.transform(X)
        prob  = float(model.predict_proba(Xsc)[0][1])
        is_if = iso_model.predict(Xsc)[0] == -1

        if rd["is_anomaly"] or is_if:
            total_anom += 1
            prob = min(0.98, prob + 0.20)   # amplify prob when anomaly flagged

        alert = ("CRITICAL" if prob >= ALERT_CRITICAL else
                 "HIGH"     if prob >= ALERT_HIGH else
                 "WARNING"  if prob >= ALERT_WARN else "OK")
        if alert != "OK": total_alerts += 1
        prob_history.append(prob)

        # Console row
        anom_flag = c(" [IF-ANOM]", "red") if is_if else ""
        row_str = (
            f"{c(str(tick).rjust(5),'bold')}  "
            f"{c(f'{wear:6.1f}','yellow' if wear>150 else 'white')}  "
            f"{c(f'{rd[\"Air temperature [K]\"]:7.1f}','white')}  "
            f"{c(f'{rd[\"Rotational speed [rpm]\"]:6.0f}','white')}  "
            f"{c(f'{rd[\"Torque [Nm]\"]:7.1f}','white')}  "
            f"{c(f'{prob*100:8.1f}%','red' if prob>=ALERT_HIGH else 'yellow' if prob>=ALERT_WARN else 'green')}  "
            f"{health_bar(prob)}  {risk_tag(prob)}{anom_flag}"
        )
        print(row_str)

        # Log
        log_rows.append({
            "tick": tick, "wear": round(wear,1),
            "air_temp": rd["Air temperature [K]"],
            "proc_temp": rd["Process temperature [K]"],
            "speed": rd["Rotational speed [rpm]"],
            "torque": rd["Torque [Nm]"],
            "fail_prob": round(prob,4),
            "anomaly_flag": int(is_if),
            "alert_level": alert,
        })

        # Summary every 25 ticks
        if tick % 25 == 0:
            avg = np.mean(prob_history[-25:])
            print(c(f"\n  ┌── 25-tick summary: avg_prob={avg*100:.1f}%  "
                    f"anomalies={total_anom}  alerts={total_alerts} ──┐", "cyan"))
            print()

        time.sleep(interval)

    # Save log
    pd.DataFrame(log_rows).to_csv(log_path, index=False)

    print()
    print(c("═"*70, "cyan"))
    print(c("  SIMULATION COMPLETE", "bold","white"))
    print(f"  Total ticks    : {c(n_ticks, 'bold')}")
    print(f"  Total anomalies: {c(total_anom,  'red' if total_anom>5  else 'green')}")
    print(f"  Total alerts   : {c(total_alerts,'red' if total_alerts>10 else 'green')}")
    print(f"  Final wear     : {c(f'{wear:.0f} min', 'yellow')}")
    print(f"  Log saved      : {c(log_path, 'dim')}")
    print(c("═"*70, "cyan"))


def main():
    parser = argparse.ArgumentParser(description="Digital Twin Real-Time Simulation")
    parser.add_argument("--ticks",    type=int,   default=100,     help="Number of ticks")
    parser.add_argument("--interval", type=float, default=0.3,     help="Seconds per tick")
    parser.add_argument("--wearrate", type=float, default=SIM_WEAR_RATE, help="Wear per tick")
    parser.add_argument("--scenario", type=str,   default="normal",
                        choices=["normal","overheat","overload","random"],
                        help="Simulation scenario")
    args = parser.parse_args()

    if not os.path.exists(f"{MODEL_DIR}/random_forest.pkl"):
        print("Models not found — run `python src/train.py` first.")
        sys.exit(1)

    model     = joblib.load(f"{MODEL_DIR}/random_forest.pkl")
    iso_model = joblib.load(f"{MODEL_DIR}/isolation_forest.pkl")
    scaler    = joblib.load(f"{MODEL_DIR}/scaler.pkl")

    run_simulation(model, iso_model, scaler,
                   n_ticks=args.ticks, interval=args.interval,
                   wear_rate=args.wearrate, scenario_name=args.scenario)


if __name__ == "__main__":
    main()
