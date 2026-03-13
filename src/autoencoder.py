"""LSTM Autoencoder for time-series anomaly detection on landing gear data."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_lstm_autoencoder(n_features: int, seq_length: int, encoding_dim: int = 32):
    """Build and return an LSTM autoencoder model."""
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

    inputs = Input(shape=(seq_length, n_features))
    # Encoder
    encoded = LSTM(64, activation="relu", return_sequences=True)(inputs)
    encoded = LSTM(encoding_dim, activation="relu", return_sequences=False)(encoded)
    # Decoder
    decoded = RepeatVector(seq_length)(encoded)
    decoded = LSTM(encoding_dim, activation="relu", return_sequences=True)(decoded)
    decoded = LSTM(64, activation="relu", return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(n_features))(decoded)

    model = Model(inputs, decoded)
    model.compile(optimizer="adam", loss="mse")
    return model


def prepare_sequences(df: pd.DataFrame, sensor_cols: list[str]) -> tuple[np.ndarray, StandardScaler]:
    """Reshape per-set_id data into 3D array (n_sets, seq_length, n_features)."""
    scaler = StandardScaler()
    set_ids = sorted(df["set_id"].unique())
    sequences = []
    for sid in set_ids:
        grp = df[df["set_id"] == sid].sort_values("row_id")
        sequences.append(grp[sensor_cols].values)
    X = np.array(sequences)
    n_sets, seq_len, n_feat = X.shape
    flat = X.reshape(-1, n_feat)
    flat_scaled = scaler.fit_transform(flat)
    X_scaled = flat_scaled.reshape(n_sets, seq_len, n_feat)
    return X_scaled, scaler


def compute_reconstruction_errors(model, X: np.ndarray) -> np.ndarray:
    """Compute per-sequence mean squared reconstruction error."""
    X_pred = model.predict(X, verbose=0)
    mse = np.mean((X - X_pred) ** 2, axis=(1, 2))
    return mse


def detect_autoencoder_anomalies(
    errors: np.ndarray, percentile: float = 95
) -> tuple[np.ndarray, float]:
    """Flag sequences with reconstruction error above the percentile threshold."""
    threshold = np.percentile(errors, percentile)
    anomalies = errors > threshold
    return anomalies, threshold
