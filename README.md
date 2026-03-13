# Aircraft Maintenance Anomaly Detection System

**CSC 430 Senior Design Project** — University of Southern Mississippi
**Authors:** Pratima Rajbanshi, Samarpan Thapa
**Instructor:** Dr. Zhaoxian Zhou

An anomaly detection and Remaining Useful Life (RUL) estimation system for aircraft landing gear sensor data. The system applies multiple machine learning and statistical methods to synthetic Airbus landing gear datasets and provides an interactive Streamlit dashboard for maintenance decision support.

## Overview

Modern aircraft generate massive volumes of sensor data every flight cycle. Small anomalies in hydraulic pressure, brake temperature, or tyre pressure can signal early-stage mechanical degradation — but they're nearly impossible for human operators to catch manually across thousands of records.

This system automatically analyzes landing gear sensor data using six detection methods and flags abnormal patterns for human review. It is designed as a **decision-support tool**, not for autonomous maintenance decisions.

## Features

- **Six anomaly detection methods:** Z-Score, Isolation Forest, K-Means Clustering, DBSCAN, One-Class SVM, LSTM Autoencoder
- **RUL estimation:** Linear trend extrapolation to predict when components will reach critical thresholds
- **Interactive dashboard:** Streamlit-based UI with adjustable parameters, multi-sensor visualization, and anomaly highlighting
- **Six datasets:** Covering hydraulics, tyre pressure, brake temperature, and wheel deceleration from synthetic Airbus landing gear systems

## Datasets

| # | System | Sets | Time Steps | Sensors | Application |
|---|--------|------|------------|---------|-------------|
| 1 | Bogie Pitch Trimmer (Hydraulics) | 200 | 200 | 8 | RUL Estimation |
| 2 | Tyre Pressure (Pressure Drop) | 800 | 50 | 8 | RUL Estimation |
| 3 | Tyre Pressure (Temp During Landing) | 600 | 180 | 2 | Anomaly Classification |
| 4 | Brakes (Max Temperature) | 100 | 500 | 8 | Clustering Anomaly Detection |
| 5 | Brakes (Deceleration During Takeoff) | 200 | 100 | 24 | Autoencoder Anomaly Detection |
| 6 | Brakes (Deceleration Over Cycles) | 400 | 50 | 24 | Autoencoder Anomaly Detection |

Source: [UWE Bristol Research Data Repository](https://researchdata.uwe.ac.uk/id/eprint/717/) (Creative Commons licensed)

## Quick Start

```bash
# Clone the repo
git clone https://github.com/appyjuss/aircraft-maintenance-anomaly-detection.git
cd aircraft-maintenance-anomaly-detection

# Set up virtual environment
python -m venv venv
source venv/Scripts/activate   # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Download datasets (~70 MB)
python download_data.py

# Launch the dashboard
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

## Project Structure

```
├── app.py                  # Streamlit dashboard (main entry point)
├── download_data.py        # Script to fetch datasets from UWE Bristol
├── requirements.txt        # Python dependencies
├── data/                   # CSV datasets (downloaded via script)
│   └── README.txt          # Dataset documentation
├── src/
│   ├── data_loader.py      # Data loading, preprocessing, statistics
│   ├── anomaly_detection.py # Z-Score, Isolation Forest, K-Means, DBSCAN, SVM, RUL
│   ├── autoencoder.py      # LSTM Autoencoder for sequence-level detection
│   └── visualization.py    # Plotly interactive charts
└── report/
    ├── report430.tex       # LaTeX source for the final report
    ├── report430.pdf       # Compiled report (36 pages)
    ├── references.bib      # Bibliography
    └── figs/               # Report figures
```

## Methods & Results

| Method | Detection Rate (DS1) | Time (40K rows) | Best For |
|--------|---------------------|-----------------|----------|
| Z-Score | 3.32% | 0.01s | Fast univariate screening |
| Isolation Forest | 5.00% | 0.56s | General-purpose multivariate detection |
| K-Means | 5.00% | 2.16s | Cluster-based anomaly identification |
| DBSCAN | 0.01% | 11.98s | Dense cluster noise detection (needs tuning) |
| One-Class SVM | 5.00% | 13.45s | Boundary-based novelty detection |
| LSTM Autoencoder | Sequence-level | ~60s (training) | Temporal pattern anomalies |

## Tech Stack

- **Python 3.13** with pandas, NumPy, SciPy
- **scikit-learn** — Isolation Forest, K-Means, DBSCAN, One-Class SVM
- **TensorFlow/Keras** — LSTM Autoencoder
- **Plotly** — Interactive visualizations
- **Streamlit** — Web dashboard

## License

The datasets are used under Creative Commons Attribution-ShareAlike 4.0 and Attribution-NonCommercial-ShareAlike 4.0 licenses from UWE Bristol.
