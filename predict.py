"""
predict.py
==========
Run a single-sample prediction through the trained Digital Twin model.
Useful for demos, testing, and integration with external sensor APIs.

Usage examples:
    # Interactive prompt
    python src/predict.py --interactive

    # Single CLI prediction
    python src/predict.py --air_temp 302.5 --proc_temp 313.0 \
                          --speed 1450 --torque 55.0 --wear 180

    # Batch predictions from CSV
    python src/predict.py --batch data/new_readings.csv
"""

import argparse, os, sys, json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODEL_DIR, ALL_FEATURES, ALERT_WARN, ALERT_HIGH, ALERT_CRITICAL


def build_feature_row(air_temp, proc_temp, speed, torque, wear, machine_type=1):
    """Construct the full feature vector from raw sensor values."""
    temp_diff        = proc_temp - air_temp
    power            = speed * torque
    wear_torque      = wear  * torque
    speed_wear       = speed * wear
    power_temp       = power * temp_diff
    wear_speed_torque= wear  * speed * torque

    row = {
        "Type_enc":                machine_type,
        "Air temperature [K]":     air_temp,
        "Process temperature [K]": proc_temp,
        "Rotational speed [rpm]":  speed,
        "Torque [Nm]":             torque,
        "Tool wear [min]":         wear,
        "temp_diff":               temp_diff,
        "power":                   power,
        "wear_torque":             wear_torque,
        "speed_wear":              speed_wear,
        "power_temp":              power_temp,
        "wear_speed_torque":       wear_speed_torque,
    }
    return np.array([[row[f] for f in ALL_FEATURES]])


def predict_single(model, iso, scaler, X_raw):
    """Return prediction dict for a single feature row."""
    Xsc      = scaler.transform(X_raw)
    prob     = float(model.predict_proba(Xsc)[0][1])
    pred     = int(model.predict(Xsc)[0])
    is_anom  = iso.predict(Xsc)[0] == -1

    if is_anom:
        prob = min(0.98, prob + 0.15)   # anomaly boosts failure probability

    wear     = float(X_raw[0, ALL_FEATURES.index("Tool wear [min]")])
    rul      = max(0.0, (1 - prob) * (250 - wear))
    alert    = ("CRITICAL" if prob >= ALERT_CRITICAL else
                "HIGH"     if prob >= ALERT_HIGH else
                "WARNING"  if prob >= ALERT_WARN else "OK")

    return {
        "failure_prediction":  pred,
        "failure_probability": round(prob, 4),
        "failure_probability_pct": f"{prob*100:.1f}%",
        "rul_estimate_min":    round(rul, 1),
        "anomaly_detected":    bool(is_anom),
        "alert_level":         alert,
        "health_score":        f"{(1-prob)*100:.1f}%",
    }


def print_result(result: dict, sensor_vals: dict):
    clr = {
        "OK":       "\033[92m", "WARNING":  "\033[93m",
        "HIGH":     "\033[91m", "CRITICAL": "\033[91m",
        "reset":    "\033[0m",  "bold":     "\033[1m",
        "dim":      "\033[2m",  "cyan":     "\033[96m",
    }
    al  = result["alert_level"]
    col = clr.get(al, clr["reset"])

    print(f"\n{clr['cyan']}{'─'*50}{clr['reset']}")
    print(f"{clr['bold']}  DIGITAL TWIN — PREDICTION RESULT{clr['reset']}")
    print(f"{clr['cyan']}{'─'*50}{clr['reset']}")
    print(f"  Air Temp      : {sensor_vals.get('air_temp',  '?')} K")
    print(f"  Process Temp  : {sensor_vals.get('proc_temp', '?')} K")
    print(f"  Speed         : {sensor_vals.get('speed',     '?')} rpm")
    print(f"  Torque        : {sensor_vals.get('torque',    '?')} Nm")
    print(f"  Tool Wear     : {sensor_vals.get('wear',      '?')} min")
    print(f"{clr['dim']}{'─'*50}{clr['reset']}")
    print(f"  Failure Prob  : {clr['bold']}{result['failure_probability_pct']}{clr['reset']}")
    print(f"  Health Score  : {result['health_score']}")
    print(f"  RUL Estimate  : {result['rul_estimate_min']} minutes")
    print(f"  Anomaly Flag  : {'YES ⚠' if result['anomaly_detected'] else 'No'}")
    print(f"  Alert Level   : {col}{clr['bold']}{result['alert_level']}{clr['reset']}")
    print(f"{clr['cyan']}{'─'*50}{clr['reset']}\n")


def interactive_mode(model, iso, scaler):
    print("\n\033[96m  Digital Twin — Interactive Prediction Mode\033[0m")
    print("  Enter sensor readings (press Ctrl-C to exit)\n")
    while True:
        try:
            at = float(input("  Air temperature [K]    (e.g. 300): "))
            pt = float(input("  Process temperature [K](e.g. 310): "))
            sp = float(input("  Rotational speed [rpm] (e.g. 1500): "))
            tq = float(input("  Torque [Nm]            (e.g. 40): "))
            wr = float(input("  Tool wear [min]        (e.g. 120): "))
            X  = build_feature_row(at, pt, sp, tq, wr)
            r  = predict_single(model, iso, scaler, X)
            print_result(r, {"air_temp":at,"proc_temp":pt,"speed":sp,"torque":tq,"wear":wr})
            again = input("  Another reading? [y/N]: ").strip().lower()
            if again != "y": break
        except KeyboardInterrupt:
            print("\n  Exiting."); break


def batch_mode(model, iso, scaler, csv_path: str):
    df = pd.read_csv(csv_path)
    required = ["air_temp","proc_temp","speed","torque","wear"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV missing column: {col}")

    records = []
    for _, row in df.iterrows():
        X = build_feature_row(row.air_temp, row.proc_temp,
                              row.speed, row.torque, row.wear)
        r = predict_single(model, iso, scaler, X)
        records.append({**row.to_dict(), **r})

    out = pd.DataFrame(records)
    out_path = csv_path.replace(".csv","_predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"  Batch predictions saved → {out_path}")
    print(out[["air_temp","speed","wear","failure_probability_pct","alert_level"]].to_string())


def main():
    parser = argparse.ArgumentParser(description="Digital Twin — Single Sample Predictor")
    parser.add_argument("--interactive",action="store_true")
    parser.add_argument("--batch",       type=str, default=None)
    parser.add_argument("--air_temp",    type=float, default=300.0)
    parser.add_argument("--proc_temp",   type=float, default=310.0)
    parser.add_argument("--speed",       type=float, default=1538.0)
    parser.add_argument("--torque",      type=float, default=39.9)
    parser.add_argument("--wear",        type=float, default=100.0)
    args = parser.parse_args()

    model  = joblib.load(f"{MODEL_DIR}/random_forest.pkl")
    iso    = joblib.load(f"{MODEL_DIR}/isolation_forest.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")

    if args.interactive:
        interactive_mode(model, iso, scaler)
    elif args.batch:
        batch_mode(model, iso, scaler, args.batch)
    else:
        X = build_feature_row(args.air_temp, args.proc_temp,
                              args.speed, args.torque, args.wear)
        r = predict_single(model, iso, scaler, X)
        print_result(r, {"air_temp":args.air_temp,"proc_temp":args.proc_temp,
                          "speed":args.speed,"torque":args.torque,"wear":args.wear})
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
