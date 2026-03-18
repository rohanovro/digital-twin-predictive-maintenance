"""
tests/test_utils.py

Unit tests for feature engineering functions in utils.py.
Run with: pytest tests/
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import engineer_features


def make_sample_df(**overrides):
    """Return a minimal one-row dataframe with default sensor values."""
    defaults = {
        "Type": ["M"],
        "Air temperature [K]": [300.0],
        "Process temperature [K]": [310.0],
        "Rotational speed [rpm]": [1500.0],
        "Torque [Nm]": [40.0],
        "Tool wear [min]": [100.0],
        "Machine failure": [0],
        "TWF": [0], "HDF": [0], "PWF": [0], "OSF": [0], "RNF": [0],
    }
    defaults.update(overrides)
    return pd.DataFrame(defaults)


# --- temp_diff ---

def test_temp_diff_basic():
    df = make_sample_df(**{
        "Air temperature [K]": [300.0],
        "Process temperature [K]": [310.0],
    })
    result = engineer_features(df)
    assert result["temp_diff"].iloc[0] == pytest.approx(10.0)


def test_temp_diff_zero_when_equal():
    df = make_sample_df(**{
        "Air temperature [K]": [305.0],
        "Process temperature [K]": [305.0],
    })
    result = engineer_features(df)
    assert result["temp_diff"].iloc[0] == pytest.approx(0.0)


def test_temp_diff_negative():
    # process temp below air temp — unusual but should not error
    df = make_sample_df(**{
        "Air temperature [K]": [315.0],
        "Process temperature [K]": [310.0],
    })
    result = engineer_features(df)
    assert result["temp_diff"].iloc[0] == pytest.approx(-5.0)


# --- power (speed * torque) ---

def test_power_calculation():
    df = make_sample_df(**{
        "Rotational speed [rpm]": [1500.0],
        "Torque [Nm]": [40.0],
    })
    result = engineer_features(df)
    assert result["power"].iloc[0] == pytest.approx(60000.0)


def test_power_zero_torque():
    df = make_sample_df(**{"Torque [Nm]": [0.0]})
    result = engineer_features(df)
    assert result["power"].iloc[0] == pytest.approx(0.0)


def test_power_zero_speed():
    df = make_sample_df(**{"Rotational speed [rpm]": [0.0]})
    result = engineer_features(df)
    assert result["power"].iloc[0] == pytest.approx(0.0)


# --- wear_torque ---

def test_wear_torque():
    df = make_sample_df(**{
        "Tool wear [min]": [100.0],
        "Torque [Nm]": [40.0],
    })
    result = engineer_features(df)
    assert result["wear_torque"].iloc[0] == pytest.approx(4000.0)


# --- speed_wear ---

def test_speed_wear():
    df = make_sample_df(**{
        "Rotational speed [rpm]": [1500.0],
        "Tool wear [min]": [100.0],
    })
    result = engineer_features(df)
    assert result["speed_wear"].iloc[0] == pytest.approx(150000.0)


# --- power_temp ---

def test_power_temp():
    df = make_sample_df(**{
        "Rotational speed [rpm]": [1500.0],
        "Torque [Nm]": [40.0],
        "Air temperature [K]": [300.0],
        "Process temperature [K]": [310.0],
    })
    result = engineer_features(df)
    # power = 1500 * 40 = 60000, temp_diff = 10 → power_temp = 600000
    assert result["power_temp"].iloc[0] == pytest.approx(600000.0)


# --- wear_speed_torque ---

def test_wear_speed_torque():
    df = make_sample_df(**{
        "Tool wear [min]": [100.0],
        "Rotational speed [rpm]": [1500.0],
        "Torque [Nm]": [40.0],
    })
    result = engineer_features(df)
    assert result["wear_speed_torque"].iloc[0] == pytest.approx(6_000_000.0)


# --- output shape and columns ---

def test_all_engineered_columns_present():
    df = make_sample_df()
    result = engineer_features(df)
    expected_cols = [
        "temp_diff", "power", "wear_torque",
        "speed_wear", "power_temp", "wear_speed_torque", "Type_enc"
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_engineer_features_preserves_row_count():
    df = pd.concat([make_sample_df()] * 5, ignore_index=True)
    result = engineer_features(df)
    assert len(result) == 5


def test_engineer_features_does_not_mutate_input():
    df = make_sample_df()
    original_cols = list(df.columns)
    engineer_features(df)
    assert list(df.columns) == original_cols


# --- Type encoding ---

def test_type_encoding_is_numeric():
    df = make_sample_df(**{"Type": ["M"]})
    result = engineer_features(df)
    assert pd.api.types.is_numeric_dtype(result["Type_enc"])


def test_multiple_type_values_encoded():
    df = pd.DataFrame({
        "Type": ["M", "L", "H"],
        "Air temperature [K]": [300.0, 300.0, 300.0],
        "Process temperature [K]": [310.0, 310.0, 310.0],
        "Rotational speed [rpm]": [1500.0, 1500.0, 1500.0],
        "Torque [Nm]": [40.0, 40.0, 40.0],
        "Tool wear [min]": [100.0, 100.0, 100.0],
        "Machine failure": [0, 0, 0],
        "TWF": [0, 0, 0], "HDF": [0, 0, 0],
        "PWF": [0, 0, 0], "OSF": [0, 0, 0], "RNF": [0, 0, 0],
    })
    result = engineer_features(df)
    # all three rows should have different integer encodings
    assert result["Type_enc"].nunique() == 3
