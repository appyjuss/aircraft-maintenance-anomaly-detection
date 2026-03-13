"""Data loading and preprocessing utilities for aircraft landing gear datasets."""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

DATASET_INFO = {
    1: {
        "file": "LGD_S_DATASET_1_BogiePitchTrimmer_PressureDrop_OverFlightCycles.csv",
        "name": "Bogie Pitch Trimmer - Pressure Drop",
        "system": "Hydraulics",
        "application": "RUL Estimation",
        "description": "Hydraulic pressure decline over flight cycles due to fluid leakage.",
    },
    2: {
        "file": "LGD_S_DATASET_2_TyrePressureIndicator_PressureDrop_OverFlightCycles.csv",
        "name": "Tyre Pressure - Pressure Drop",
        "system": "Tyre Pressure",
        "application": "RUL Estimation",
        "description": "Tyre pressure decline over flight cycles due to leakage.",
    },
    3: {
        "file": "LGD_S_DATASET_3_TyrePressureIndicator_TemperatureRise_DuringLanding.csv",
        "name": "Tyre Pressure - Temperature Rise During Landing",
        "system": "Landing Gear Brake",
        "application": "Anomaly Classification",
        "description": "Brake temperature and tyre pressure during landing events.",
    },
    4: {
        "file": "LGD_S_DATASET_4_LandingGearBrakes_MaxTemperature_OverFlightCycles.csv",
        "name": "Landing Gear Brakes - Max Temperature",
        "system": "Brake Temperature",
        "application": "Anomaly Detection (Clustering)",
        "description": "Brake temperatures across 8 wheels over flight cycles.",
    },
    5: {
        "file": "LGD_S_DATASET_5_LandingGearBrakes_Deceleration_DuringTakeoff.csv",
        "name": "Landing Gear Brakes - Deceleration During Takeoff",
        "system": "Brakes & Wheels",
        "application": "Anomaly Detection (Autoencoder)",
        "description": "Brake temperature, pressure, and wheel speed during takeoff.",
    },
    6: {
        "file": "LGD_S_DATASET_6_LandingGearBrakes_Deceleration_OverFlightCycles.csv",
        "name": "Landing Gear Brakes - Deceleration Over Flight Cycles",
        "system": "Brakes & Wheels",
        "application": "Anomaly Detection (Autoencoder)",
        "description": "Max brake temp, pressure, and deceleration per flight cycle.",
    },
}


def load_dataset(dataset_id: int) -> pd.DataFrame:
    """Load a dataset by its ID (1-6)."""
    info = DATASET_INFO[dataset_id]
    path = os.path.join(DATA_DIR, info["file"])
    df = pd.read_csv(path)
    df["set_id"] = df["set_id"].astype(int)
    df["row_id"] = df["row_id"].astype(int)
    return df


def get_sensor_columns(df: pd.DataFrame) -> list[str]:
    """Return all columns except set_id and row_id."""
    return [c for c in df.columns if c not in ("set_id", "row_id")]


def get_single_series(df: pd.DataFrame, set_id: int) -> pd.DataFrame:
    """Extract a single time series by set_id."""
    return df[df["set_id"] == set_id].sort_values("row_id").reset_index(drop=True)


def normalize_features(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Min-max normalize sensor columns."""
    result = df.copy()
    cols = columns or get_sensor_columns(df)
    for col in cols:
        mn, mx = result[col].min(), result[col].max()
        if mx - mn > 0:
            result[col] = (result[col] - mn) / (mx - mn)
        else:
            result[col] = 0.0
    return result


def compute_series_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-set_id statistics for each sensor column."""
    sensor_cols = get_sensor_columns(df)
    records = []
    for sid, grp in df.groupby("set_id"):
        row = {"set_id": sid}
        for col in sensor_cols:
            vals = grp[col]
            row[f"{col}_mean"] = vals.mean()
            row[f"{col}_std"] = vals.std()
            row[f"{col}_min"] = vals.min()
            row[f"{col}_max"] = vals.max()
            row[f"{col}_trend"] = np.polyfit(grp["row_id"], vals, 1)[0]
        records.append(row)
    return pd.DataFrame(records)
