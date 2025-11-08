# Logistics ML Case Study — End-to-End (Regression, Classification, Time Series, Anomalies, Geo)

This repository demonstrates an end-to-end ML workflow over a synthetic logistics dataset:
- **Regression**: shipping costs, lead time, fuel consumption (incl. quantile P50/P90 band for ETA).
- **Classification**: delay probability (calibration, PR curve, score bands), risk/order/cargo status.
- **Time series**: weekly nowcast/backtest with holiday features for PT.
- **Anomalies**: IsolationForest for fuel rate & cold chain temperature.
- **Geospatial**: DBSCAN (haversine) clusters + hexbin heatmap.

## How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Adjust `DATA_PATH` and `OUTPUT_DIR` inside the script.
3. Run: `python logi.py`
4. Outputs land in `outputs/` and a curated subset for BI in `outputs/BI/`.
5. Selected visuals are copied to `outputs/gallery/` for sharing.

## Key folders
- `outputs/regression/<target>/` — metrics, CV, residuals, scatter, feature importance/SHAP.
- `outputs/classification/<model>/` — classification report, PR curve, calibration, confusion matrices.
- `outputs/time_series/` — nowcast/backtest metrics & plots.
- `outputs/anomalies/` — scored anomalies + histograms (and maps if GPS exists).
- `outputs/geospatial/` — clustered points, centroids, hexbin heatmap.
- `outputs/BI/` — flat files ready for Power BI refresh.
- `outputs/gallery/` — **best PNGs** for LinkedIn/GitHub (auto-generated) + `_GALLERY_index.csv`.

## Notes
- Models use cross-validation, probability calibration checks (Brier/ECE) and quantile band for ETA.
- The dataset is synthetic; patterns and performance are illustrative.
