"""
Aircraft Maintenance Anomaly Detection System
CSC 430 Senior Design Project
Authors: Pratima Rajbanshi, Samarpan Thapa
Instructor: Dr. Zhaoxian Zhou
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import (
    load_dataset, DATASET_INFO, get_sensor_columns,
    get_single_series, normalize_features, compute_series_stats,
)
from src.anomaly_detection import (
    zscore_anomalies, rolling_zscore_anomalies,
    isolation_forest_detect, kmeans_anomaly_detect,
    dbscan_anomaly_detect, ocsvm_detect,
    detect_degradation_trend, estimate_rul,
)
from src.visualization import (
    plot_time_series, plot_trend_with_rul, plot_anomaly_scores,
    plot_cluster_scatter, plot_multi_sensor, plot_heatmap,
)

st.set_page_config(
    page_title="Aircraft Maintenance Anomaly Detection",
    page_icon="✈",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("Aircraft Maintenance\nAnomaly Detection System")
st.sidebar.markdown("CSC 430 Senior Design Project")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "Overview",
    "Data Explorer",
    "RUL Estimation",
    "Anomaly Detection",
    "LSTM Autoencoder",
])

# ── Cache data loading ───────────────────────────────────────────────────────
@st.cache_data
def cached_load(ds_id):
    return load_dataset(ds_id)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Aircraft Maintenance Anomaly Detection System")
    st.markdown("""
    This system analyzes synthetic aircraft landing gear sensor data to detect
    anomalies and estimate remaining useful life (RUL) of critical components.
    It is designed as a **decision-support tool** for maintenance personnel —
    not for autonomous decision-making.

    ### Datasets
    The system uses 6 synthetic datasets generated from real Airbus landing gear
    systems, covering hydraulics, tyre pressure, brakes, and wheel deceleration.
    """)

    cols = st.columns(3)
    for i, (ds_id, info) in enumerate(DATASET_INFO.items()):
        with cols[i % 3]:
            st.markdown(f"**Dataset {ds_id}:** {info['name']}")
            st.caption(f"System: {info['system']} | Application: {info['application']}")
            st.markdown(f"_{info['description']}_")

    st.markdown("---")
    st.markdown("""
    ### Methods
    | Method | Use Case |
    |--------|----------|
    | Z-Score / Rolling Z-Score | Point-level anomaly flagging |
    | Isolation Forest | Multivariate outlier detection |
    | K-Means / DBSCAN | Cluster-based anomaly identification |
    | One-Class SVM | Boundary-based novelty detection |
    | Linear Regression | RUL estimation via trend extrapolation |
    | LSTM Autoencoder | Sequence-level anomaly detection |

    ### Disclaimer
    > This system is intended as a decision-support aid. All flagged anomalies
    > should be reviewed by qualified maintenance personnel before action is taken.
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Data Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.title("Data Explorer")

    ds_id = st.selectbox(
        "Select Dataset",
        list(DATASET_INFO.keys()),
        format_func=lambda x: f"Dataset {x}: {DATASET_INFO[x]['name']}",
    )
    df = cached_load(ds_id)
    sensor_cols = get_sensor_columns(df)

    st.markdown(f"**{DATASET_INFO[ds_id]['description']}**")
    st.markdown(f"Shape: `{df.shape}` — Sets: `{df['set_id'].nunique()}` — "
                f"Sensors: `{len(sensor_cols)}`")

    tab1, tab2, tab3 = st.tabs(["Time Series", "Statistics", "Correlation"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            set_id = st.number_input("Set ID", min_value=0,
                                     max_value=int(df["set_id"].max()), value=0)
        with col2:
            selected_sensors = st.multiselect("Sensors", sensor_cols,
                                              default=sensor_cols[:2])
        if selected_sensors:
            fig = plot_multi_sensor(df, set_id, selected_sensors)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

    with tab3:
        fig = plot_heatmap(df, sensor_cols,
                          title=f"Dataset {ds_id} Sensor Correlation")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RUL Estimation
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "RUL Estimation":
    st.title("Remaining Useful Life Estimation")
    st.markdown("""
    Uses linear trend extrapolation to estimate when a sensor value will cross
    a critical threshold. Best suited for Datasets 1 (hydraulic pressure) and
    2 (tyre pressure), which exhibit gradual decline.
    """)

    ds_id = st.selectbox("Dataset", [1, 2],
                         format_func=lambda x: f"Dataset {x}: {DATASET_INFO[x]['name']}")
    df = cached_load(ds_id)
    sensor_cols = get_sensor_columns(df)

    col1, col2, col3 = st.columns(3)
    with col1:
        set_id = st.number_input("Set ID", min_value=0,
                                 max_value=int(df["set_id"].max()), value=0)
    with col2:
        sensor = st.selectbox("Sensor", sensor_cols)
    with col3:
        threshold = st.number_input("Critical Threshold", value=0.0,
                                    help="Value at which the component is considered failed")

    series_df = get_single_series(df, set_id)
    series = series_df[sensor]

    # Auto-suggest threshold: 70% of starting value (for declining signals)
    if threshold == 0.0:
        suggested = series.iloc[:5].mean() * 0.7
        st.info(f"Suggested threshold (70% of initial): **{suggested:.2f}**. "
                "Set it in the input above.")
        threshold = suggested

    trend = detect_degradation_trend(series)
    rul = estimate_rul(series, critical_threshold=threshold)

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Trend Slope", f"{trend['slope']:.4f}")
        st.metric("R-squared", f"{trend['r_squared']:.4f}")
    with col_b:
        if rul["rul"] is not None:
            st.metric("Estimated RUL (cycles)", f"{rul['rul']:.0f}")
            st.metric("Predicted Failure Step", f"{rul['predicted_failure_step']:.0f}")
        else:
            st.warning("No decline detected — RUL is infinite.")

    trend["critical_threshold"] = threshold
    fig = plot_trend_with_rul(series, trend,
                              title=f"{sensor} — Set {set_id}: Trend & RUL")
    st.plotly_chart(fig, use_container_width=True)

    # Batch RUL for all sets
    with st.expander("Batch RUL across all sets"):
        results = []
        for sid in sorted(df["set_id"].unique()):
            s = get_single_series(df, sid)[sensor]
            r = estimate_rul(s, critical_threshold=threshold)
            results.append({"set_id": sid, "rul": r.get("rul"), "slope": r["slope"]})
        rul_df = pd.DataFrame(results)
        st.dataframe(rul_df.style.format({"rul": "{:.0f}", "slope": "{:.4f}"}),
                      use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Anomaly Detection
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Anomaly Detection":
    st.title("Anomaly Detection")

    ds_id = st.selectbox(
        "Select Dataset",
        list(DATASET_INFO.keys()),
        format_func=lambda x: f"Dataset {x}: {DATASET_INFO[x]['name']}",
    )
    df = cached_load(ds_id)
    sensor_cols = get_sensor_columns(df)

    method = st.selectbox("Detection Method", [
        "Z-Score",
        "Isolation Forest",
        "K-Means Clustering",
        "DBSCAN",
        "One-Class SVM",
    ])

    st.markdown("---")

    # ── Method-specific parameters and execution ──
    if method == "Z-Score":
        threshold = st.slider("Z-Score Threshold", 1.5, 5.0, 3.0, 0.1)
        result = zscore_anomalies(df, sensor_cols, threshold)

    elif method == "Isolation Forest":
        contamination = st.slider("Contamination", 0.01, 0.20, 0.05, 0.01)
        result = isolation_forest_detect(df, sensor_cols, contamination)

    elif method == "K-Means Clustering":
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        with col2:
            pct = st.slider("Anomaly Percentile", 80, 99, 95)
        result = kmeans_anomaly_detect(df, sensor_cols, n_clusters, pct)

    elif method == "DBSCAN":
        col1, col2 = st.columns(2)
        with col1:
            eps = st.slider("Epsilon (eps)", 0.1, 5.0, 0.5, 0.1)
        with col2:
            min_samples = st.slider("Min Samples", 2, 20, 5)
        result = dbscan_anomaly_detect(df, sensor_cols, eps, min_samples)

    else:  # One-Class SVM
        nu = st.slider("Nu (outlier fraction)", 0.01, 0.20, 0.05, 0.01)
        result = ocsvm_detect(df, sensor_cols, nu)

    n_anom = result["is_anomaly"].sum()
    n_total = len(result)
    st.markdown(f"**Anomalies found:** {n_anom} / {n_total} "
                f"({100 * n_anom / n_total:.1f}%)")

    # Visualization
    tab1, tab2 = st.tabs(["Time Series View", "Scatter / Cluster View"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            view_set = st.number_input("View Set ID", min_value=0,
                                       max_value=int(df["set_id"].max()), value=0,
                                       key="anom_set")
        with col2:
            view_sensor = st.selectbox("Sensor", sensor_cols, key="anom_sensor")
        fig = plot_time_series(result, view_sensor, view_set, anomaly_col="is_anomaly")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if len(sensor_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", sensor_cols, index=0, key="sc_x")
            with col2:
                y_col = st.selectbox("Y-axis", sensor_cols,
                                     index=min(1, len(sensor_cols) - 1), key="sc_y")
            if "cluster" not in result.columns:
                result["cluster"] = 0
            fig = plot_cluster_scatter(result, x_col, y_col)
            st.plotly_chart(fig, use_container_width=True)

    # Show anomalous rows
    with st.expander("View Anomalous Data Points"):
        st.dataframe(result[result["is_anomaly"]].head(200), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: LSTM Autoencoder
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "LSTM Autoencoder":
    st.title("LSTM Autoencoder Anomaly Detection")
    st.markdown("""
    Trains an LSTM autoencoder on the time-series sequences. Sequences with high
    reconstruction error are flagged as anomalous. Best for Datasets 3, 5, and 6.
    """)

    ds_id = st.selectbox("Dataset", [3, 5, 6],
                         format_func=lambda x: f"Dataset {x}: {DATASET_INFO[x]['name']}")
    df = cached_load(ds_id)
    sensor_cols = get_sensor_columns(df)

    col1, col2, col3 = st.columns(3)
    with col1:
        epochs = st.slider("Epochs", 5, 100, 20)
    with col2:
        batch_size = st.slider("Batch Size", 8, 64, 32)
    with col3:
        anomaly_pct = st.slider("Anomaly Percentile", 80, 99, 95)

    if st.button("Train Autoencoder", type="primary"):
        with st.spinner("Preparing data..."):
            from src.autoencoder import (
                build_lstm_autoencoder, prepare_sequences,
                compute_reconstruction_errors, detect_autoencoder_anomalies,
            )
            X, scaler = prepare_sequences(df, sensor_cols)
            n_sets, seq_len, n_feat = X.shape
            st.write(f"Sequences: {n_sets}, Length: {seq_len}, Features: {n_feat}")

        with st.spinner(f"Training LSTM autoencoder ({epochs} epochs)..."):
            model = build_lstm_autoencoder(n_feat, seq_len)
            history = model.fit(X, X, epochs=epochs, batch_size=batch_size,
                                validation_split=0.1, verbose=0)

        with st.spinner("Computing reconstruction errors..."):
            errors = compute_reconstruction_errors(model, X)
            anomalies, threshold = detect_autoencoder_anomalies(errors, anomaly_pct)

        st.success(f"Training complete! Anomalies: {anomalies.sum()} / {len(anomalies)}")

        # Training loss
        import plotly.graph_objects as go
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=history.history["loss"], name="Train Loss"))
        if "val_loss" in history.history:
            fig_loss.add_trace(go.Scatter(y=history.history["val_loss"], name="Val Loss"))
        fig_loss.update_layout(title="Training Loss", xaxis_title="Epoch",
                               yaxis_title="MSE", template="plotly_white", height=350)
        st.plotly_chart(fig_loss, use_container_width=True)

        # Anomaly scores
        set_ids = sorted(df["set_id"].unique())
        fig_scores = plot_anomaly_scores(errors, threshold, labels=set_ids)
        st.plotly_chart(fig_scores, use_container_width=True)

        # List anomalous sets
        anom_sets = [set_ids[i] for i in range(len(set_ids)) if anomalies[i]]
        st.markdown(f"**Anomalous sets:** {anom_sets}")

        # Let user inspect a specific anomalous sequence
        if anom_sets:
            inspect_set = st.selectbox("Inspect anomalous set", anom_sets)
            fig = plot_multi_sensor(df, inspect_set, sensor_cols[:4])
            st.plotly_chart(fig, use_container_width=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("University of Southern Mississippi\n"
                   "CSC 430: Senior Design Project\n"
                   "Pratima Rajbanshi & Samarpan Thapa")
