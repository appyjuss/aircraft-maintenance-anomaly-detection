"""Visualization utilities for the anomaly detection dashboard."""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


def plot_time_series(df: pd.DataFrame, column: str, set_id: int, anomaly_col: str | None = None) -> go.Figure:
    """Plot a single time series with optional anomaly highlighting."""
    subset = df[df["set_id"] == set_id].sort_values("row_id")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=subset["row_id"], y=subset[column],
        mode="lines", name=column, line=dict(color="#1f77b4"),
    ))
    if anomaly_col and anomaly_col in subset.columns:
        anom = subset[subset[anomaly_col]]
        fig.add_trace(go.Scatter(
            x=anom["row_id"], y=anom[column],
            mode="markers", name="Anomaly",
            marker=dict(color="red", size=8, symbol="x"),
        ))
    fig.update_layout(
        title=f"{column} — Set {set_id}",
        xaxis_title="Flight Cycle / Time Step",
        yaxis_title=column,
        template="plotly_white",
        height=400,
    )
    return fig


def plot_trend_with_rul(series: pd.Series, trend_info: dict, title: str = "") -> go.Figure:
    """Plot series with fitted trend line and RUL projection."""
    x = np.arange(len(series))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=series, mode="lines", name="Observed"))
    fig.add_trace(go.Scatter(
        x=x, y=trend_info["trend_line"], mode="lines",
        name=f"Trend (slope={trend_info['slope']:.4f})",
        line=dict(dash="dash", color="orange"),
    ))
    if "critical_threshold" in trend_info:
        fig.add_hline(
            y=trend_info["critical_threshold"], line_dash="dot",
            line_color="red", annotation_text="Critical Threshold",
        )
    fig.update_layout(
        title=title or "Trend & RUL Estimation",
        xaxis_title="Flight Cycle",
        yaxis_title="Value",
        template="plotly_white",
        height=400,
    )
    return fig


def plot_anomaly_scores(scores: np.ndarray, threshold: float, labels: list | None = None) -> go.Figure:
    """Bar chart of anomaly scores with threshold line."""
    x = labels or list(range(len(scores)))
    colors = ["red" if s > threshold else "#1f77b4" for s in scores]
    fig = go.Figure(go.Bar(x=x, y=scores, marker_color=colors))
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
    fig.update_layout(
        title="Anomaly Scores per Set",
        xaxis_title="Set ID",
        yaxis_title="Reconstruction Error / Score",
        template="plotly_white",
        height=400,
    )
    return fig


def plot_cluster_scatter(df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    """Scatter plot colored by cluster, with anomalies marked."""
    fig = go.Figure()
    normal = df[~df["is_anomaly"]]
    anomalous = df[df["is_anomaly"]]
    fig.add_trace(go.Scatter(
        x=normal[x_col], y=normal[y_col], mode="markers",
        marker=dict(color=normal["cluster"], colorscale="Viridis", size=5),
        name="Normal",
    ))
    fig.add_trace(go.Scatter(
        x=anomalous[x_col], y=anomalous[y_col], mode="markers",
        marker=dict(color="red", size=9, symbol="x"),
        name="Anomaly",
    ))
    fig.update_layout(
        title=f"Cluster Analysis: {x_col} vs {y_col}",
        xaxis_title=x_col, yaxis_title=y_col,
        template="plotly_white", height=450,
    )
    return fig


def plot_multi_sensor(df: pd.DataFrame, set_id: int, columns: list[str]) -> go.Figure:
    """Plot multiple sensor readings for a single set on the same chart."""
    subset = df[df["set_id"] == set_id].sort_values("row_id")
    fig = go.Figure()
    for col in columns:
        fig.add_trace(go.Scatter(
            x=subset["row_id"], y=subset[col], mode="lines", name=col,
        ))
    fig.update_layout(
        title=f"Multi-Sensor View — Set {set_id}",
        xaxis_title="Time Step",
        yaxis_title="Value",
        template="plotly_white",
        height=450,
    )
    return fig


def plot_heatmap(df: pd.DataFrame, columns: list[str], title: str = "Correlation Heatmap") -> go.Figure:
    """Correlation heatmap for sensor columns."""
    corr = df[columns].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu_r", zmin=-1, zmax=1,
    ))
    fig.update_layout(title=title, template="plotly_white", height=500, width=600)
    return fig
