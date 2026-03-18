"""
tests/test_predict.py

Unit tests for inference functions in predict.py.
Tests build_feature_row and predict_single without requiring
trained model files — uses lightweight mock models.

Run with: pytest tests/
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import build_feature_row, predict_single


# --- Mock model and scaler so tests don't need .pkl files ---

class MockClassifier:
    """Returns a fixed failure probability for testing."""
    def __init__(self, prob=0.1):
        self.prob = prob

    def predict_proba(self, X):
        return np.array([[1 - self.prob, self.prob]])

    def predict(self, X):
        return np.array([1 if self.prob >= 0.5 else 0])


class MockIsolationForest:
    """Returns normal (1) or anomaly (-1) for testing."""
    def __init__(self, is_anomaly=False):
        self.is_anomaly = is_anomaly

    def predict(self, X):
        return np.array([-1 if self.is_anomaly else 1])


class MockScaler:
    def transform(self, X):
        return X  # pass-through for testing


# --- build_feature_row ---

def test_build_feature_row_returns_2d_array():
    X = build_feature_row(300.0, 310.0, 1500.0, 40.0, 100.0)
    assert X.ndim == 2
    assert X.shape[0] == 1


def test_build_feature_row_correct_column_count():
    from config import ALL_FEATURES
    X = build_feature_row(300.0, 310.0, 1500.0, 40.0, 100.0)
    assert X.shape[1] == len(ALL_FEATURES)


def test_temp_diff_in_feature_row():
    # temp_diff = proc_temp - air_temp = 310 - 300 = 10
    from config import ALL_FEATURES
    X = build_feature_row(300.0, 310.0, 1500.0, 40.0, 100.0)
    idx = ALL_FEATURES.index("temp_diff")
    assert X[0, idx] == pytest.approx(10.0)


def test_power_in_feature_row():
    # power = speed * torque = 1500 * 40 = 60000
    from config import ALL_FEATURES
    X = build_feature_row(300.0, 310.0, 1500.0, 40.0, 100.0)
    idx = ALL_FEATURES.index("power")
    assert X[0, idx] == pytest.approx(60000.0)


def test_wear_torque_in_feature_row():
    # wear_torque = wear * torque = 100 * 40 = 4000
    from config import ALL_FEATURES
    X = build_feature_row(300.0, 310.0, 1500.0, 40.0, 100.0)
    idx = ALL_FEATURES.index("wear_torque")
    assert X[0, idx] == pytest.approx(4000.0)


def test_feature_row_no_nan():
    X = build_feature_row(300.0, 310.0, 1500.0, 40.0, 100.0)
    assert not np.isnan(X).any()


# --- predict_single: output structure ---

def test_predict_single_returns_expected_keys():
    X = build_feature_row(300.0, 310.0, 1500.0, 40.0, 100.0)
    result = predict_single(MockClassifier(0.1), MockIsolationForest(), MockScaler(), X)
    expected_keys = [
        "failure_prediction", "failure_probability",
        "failure_probability_pct", "rul_estimate_min",
        "anomaly_detected", "alert_level", "health_score"
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"


def test_predict_single_prob_in_range():
    X = build_feature_row(300.0, 310.0, 1500.0, 40.0, 100.0)
    result = predict_single(MockClassifier(0.4), MockIsolationForest(), MockScaler(), X)
    assert 0.0 <= result["failure_probability"] <= 1.0


def test_predict_single_rul_non_negative():
    X = build_feature_row(300.0, 310.0, 1500.0, 40.0, 100.0)
    result = predict_single(MockClassifier(0.1), MockIsolationForest(), MockScaler(), X)
    assert result["rul_estimate_min"] >= 0.0


# --- alert levels ---

def test_alert_ok():
    X = build_feature_row(300.0, 310.0, 1500.0, 40.0, 100.0)
    result = predict_single(MockClassifier(0.1), MockIsolationForest(), MockScaler(), X)
    assert result["alert_level"] == "OK"


def test_alert_warning():
    X = build_feature_row(300.0, 310.0, 1500.0, 40.0, 100.0)
    result = predict_single(MockClassifier(0.45), MockIsolationForest(), MockScaler(), X)
    assert result["alert_level"] == "WARNING"


def test_alert_high():
    X = build_feature_row(300.0, 310.0, 1500.0, 40.0, 100.0)
    result = predict_single(MockClassifier(0.7), MockIsolationForest(), MockScaler(), X)
    assert result["alert_level"] == "HIGH"


def test_alert_critical():
    X = build_feature_row(300.0, 310.0, 1500.0, 40.0, 100.0)
    result = predict_single(MockClassifier(0.9), MockIsolationForest(), MockScaler(), X)
    assert result["alert_level"] == "CRITICAL"


# --- anomaly detection ---

def test_anomaly_flag_true_when_anomaly():
    X = build_feature_row(300.0, 310.0, 1500.0, 40.0, 100.0)
    result = predict_single(MockClassifier(0.1), MockIsolationForest(is_anomaly=True), MockScaler(), X)
    assert result["anomaly_detected"] is True


def test_anomaly_flag_false_when_normal():
    X = build_feature_row(300.0, 310.0, 1500.0, 40.0, 100.0)
    result = predict_single(MockClassifier(0.1), MockIsolationForest(is_anomaly=False), MockScaler(), X)
    assert result["anomaly_detected"] is False


def test_anomaly_boosts_probability():
    X = build_feature_row(300.0, 310.0, 1500.0, 40.0, 100.0)
    base_prob = 0.2
    normal = predict_single(MockClassifier(base_prob), MockIsolationForest(False), MockScaler(), X)
    anomaly = predict_single(MockClassifier(base_prob), MockIsolationForest(True), MockScaler(), X)
    assert anomaly["failure_probability"] > normal["failure_probability"]


# --- RUL behaviour ---

def test_rul_decreases_with_higher_wear():
    # higher tool wear should yield lower RUL
    X_low_wear = build_feature_row(300.0, 310.0, 1500.0, 40.0, 50.0)
    X_high_wear = build_feature_row(300.0, 310.0, 1500.0, 40.0, 200.0)
    model = MockClassifier(0.1)
    iso = MockIsolationForest()
    scaler = MockScaler()
    r_low = predict_single(model, iso, scaler, X_low_wear)
    r_high = predict_single(model, iso, scaler, X_high_wear)
    assert r_low["rul_estimate_min"] > r_high["rul_estimate_min"]


def test_rul_zero_at_max_wear():
    X = build_feature_row(300.0, 310.0, 1500.0, 40.0, 250.0)
    result = predict_single(MockClassifier(0.1), MockIsolationForest(), MockScaler(), X)
    assert result["rul_estimate_min"] == pytest.approx(0.0)
