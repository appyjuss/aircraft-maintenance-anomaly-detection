"""Anomaly detection models for aircraft maintenance data."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from scipy import stats


# ---------------------------------------------------------------------------
# Statistical anomaly detection
# ---------------------------------------------------------------------------

def zscore_anomalies(df: pd.DataFrame, columns: list[str], threshold: float = 3.0) -> pd.DataFrame:
    """Flag rows where any sensor column exceeds the Z-score threshold."""
    result = df.copy()
    z_scores = np.abs(stats.zscore(df[columns], nan_policy="omit"))
    result["is_anomaly"] = (z_scores > threshold).any(axis=1)
    result["max_zscore"] = z_scores.max(axis=1)
    return result


def rolling_zscore_anomalies(
    series: pd.Series, window: int = 20, threshold: float = 3.0
) -> pd.Series:
    """Detect anomalies using a rolling window Z-score."""
    rolling_mean = series.rolling(window=window, center=True).mean()
    rolling_std = series.rolling(window=window, center=True).std()
    z = np.abs((series - rolling_mean) / rolling_std.replace(0, np.nan))
    return z > threshold


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------

def isolation_forest_detect(
    df: pd.DataFrame, columns: list[str], contamination: float = 0.05
) -> pd.DataFrame:
    """Run Isolation Forest on the given columns and return labels."""
    result = df.copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(df[columns])
    model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    preds = model.fit_predict(X)
    result["is_anomaly"] = preds == -1
    result["anomaly_score"] = model.decision_function(X)
    return result


# ---------------------------------------------------------------------------
# Clustering-based anomaly detection
# ---------------------------------------------------------------------------

def kmeans_anomaly_detect(
    df: pd.DataFrame, columns: list[str], n_clusters: int = 3, percentile: float = 95
) -> pd.DataFrame:
    """K-Means clustering: flag points far from their cluster center as anomalies."""
    result = df.copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(df[columns])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    distances = np.min(km.transform(X), axis=1)
    threshold = np.percentile(distances, percentile)
    result["cluster"] = labels
    result["distance_to_center"] = distances
    result["is_anomaly"] = distances > threshold
    return result


def dbscan_anomaly_detect(
    df: pd.DataFrame, columns: list[str], eps: float = 0.5, min_samples: int = 5
) -> pd.DataFrame:
    """DBSCAN: label noise points (-1) as anomalies."""
    result = df.copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(df[columns])
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    result["cluster"] = labels
    result["is_anomaly"] = labels == -1
    return result


# ---------------------------------------------------------------------------
# One-Class SVM
# ---------------------------------------------------------------------------

def ocsvm_detect(
    df: pd.DataFrame, columns: list[str], nu: float = 0.05
) -> pd.DataFrame:
    """One-Class SVM anomaly detection."""
    result = df.copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(df[columns])
    model = OneClassSVM(kernel="rbf", nu=nu)
    preds = model.fit_predict(X)
    result["is_anomaly"] = preds == -1
    result["anomaly_score"] = model.decision_function(X)
    return result


# ---------------------------------------------------------------------------
# Trend-based degradation detection
# ---------------------------------------------------------------------------

def detect_degradation_trend(
    series: pd.Series, threshold_slope: float | None = None
) -> dict:
    """Fit a linear trend to a series and report degradation info."""
    x = np.arange(len(series))
    slope, intercept = np.polyfit(x, series, 1)
    trend_line = slope * x + intercept
    residuals = series.values - trend_line
    if threshold_slope is None:
        threshold_slope = -np.abs(slope) * 0.5  # auto-threshold at half the slope
    return {
        "slope": slope,
        "intercept": intercept,
        "trend_line": trend_line,
        "residuals": residuals,
        "is_degrading": slope < threshold_slope,
        "r_squared": 1 - np.sum(residuals**2) / np.sum((series - series.mean()) ** 2),
    }


# ---------------------------------------------------------------------------
# RUL estimation
# ---------------------------------------------------------------------------

def estimate_rul(series: pd.Series, critical_threshold: float) -> dict:
    """Estimate Remaining Useful Life via linear extrapolation."""
    x = np.arange(len(series))
    slope, intercept = np.polyfit(x, series, 1)
    if slope >= 0:
        return {"rul": None, "slope": slope, "message": "No decline detected"}
    rul_step = (critical_threshold - intercept) / slope
    remaining = max(0, rul_step - len(series))
    return {
        "rul": remaining,
        "slope": slope,
        "intercept": intercept,
        "critical_threshold": critical_threshold,
        "predicted_failure_step": rul_step,
    }
