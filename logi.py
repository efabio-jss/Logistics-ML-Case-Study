import os, math, warnings, datetime as dt
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix,
    brier_score_loss, precision_recall_curve, average_precision_score,
    mean_pinball_loss
)
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN


import shutil, glob

warnings.filterwarnings("ignore", category=FutureWarning)


HAS_XGB, HAS_LGBM, HAS_SHAP = False, False, False
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    pass
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    pass
try:
    import shap  
    HAS_SHAP = True
except Exception:
    pass


@dataclass
class Config:
    DATA_PATH: str = r"#your file path"
    OUTPUT_DIR: str = r"#your output file path"
    RANDOM_STATE: int = 42


def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def derive_distance_feature(df, lat_col="vehicle_gps_latitude", lon_col="vehicle_gps_longitude", time_col="timestamp"):
    if not all(c in df.columns for c in [lat_col, lon_col, time_col]):
        df["distance_km"] = np.nan
        return df
    dfx = df.copy()
    dfx[time_col] = pd.to_datetime(dfx[time_col], errors="coerce")
    dfx = dfx.sort_values(time_col)
    lat_prev = dfx[lat_col].shift(1)
    lon_prev = dfx[lon_col].shift(1)
    dists = []
    for (lat, lon, la, lo) in zip(dfx[lat_col], dfx[lon_col], lat_prev, lon_prev):
        if np.any(pd.isna([lat, lon, la, lo])):
            dists.append(np.nan)
        else:
            dists.append(haversine_km(la, lo, lat, lon))
    dfx["distance_km"] = dists
    return dfx


def add_time_parts(df):
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["dow"] = df["timestamp"].dt.dayofweek
        df["weekofyear"] = df["timestamp"].dt.isocalendar().week.astype(int)
        df["hour"] = df["timestamp"].dt.hour
        df["month"] = df["timestamp"].dt.month
    return df



def _easter_date(year: int) -> dt.date:
    
    a = year % 19
    b = year // 100; c = year % 100
    d = b // 4; e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19*a + b - d - g + 15) % 30
    i = c // 4; k = c % 4
    l = (32 + 2*e + 2*i - h - k) % 7
    m = (a + 11*h + 22*l) // 451
    month = (h + l - 7*m + 114) // 31
    day = ((h + l - 7*m + 114) % 31) + 1
    return dt.date(year, month, day)

def portugal_holidays(year: int) -> set:
    easter = _easter_date(year)
    good_friday = easter - dt.timedelta(days=2)
    corpus_christi = easter + dt.timedelta(days=60)
    fixed = {
        dt.date(year,1,1),   
        dt.date(year,4,25),  
        dt.date(year,5,1),   
        dt.date(year,6,10),  
        dt.date(year,8,15),  
        dt.date(year,10,5),  
        dt.date(year,11,1),  
        dt.date(year,12,1),  
        dt.date(year,12,8),  
        dt.date(year,12,25), 
    }
    moveable = {good_friday, corpus_christi}
    return fixed | moveable

def add_pt_holiday_features(df):
    if "timestamp" not in df.columns:
        return df
    dfx = df.copy()
    dfx["timestamp"] = pd.to_datetime(dfx["timestamp"], errors="coerce")
    dfx["date"] = dfx["timestamp"].dt.date
    years = dfx["timestamp"].dropna().dt.year.unique().tolist()
    hol = set()
    for y in years:
        hol |= portugal_holidays(int(y))
    dfx["is_holiday_pt"] = dfx["date"].apply(lambda d: 1 if (isinstance(d, dt.date) and d in hol) else 0)
    dfx["weekday"] = dfx["timestamp"].dt.weekday
    prev_day = dfx["timestamp"].dt.date - pd.to_timedelta(1, unit="D")
    next_day = dfx["timestamp"].dt.date + pd.to_timedelta(1, unit="D")
    dfx["is_bridge_pt"] = ((dfx["weekday"] == 4) & prev_day.isin(hol)) | ((dfx["weekday"] == 0) & next_day.isin(hol))
    dfx["is_bridge_pt"] = dfx["is_bridge_pt"].astype(int)
    dfx.drop(columns=["date","weekday"], inplace=True)
    return dfx


def add_interactions(df):
    def g(c): return df[c] if c in df.columns else np.nan
    if "traffic_congestion_level" in df.columns and "weather_condition_severity" in df.columns:
        df["traffic_x_weather"] = g("traffic_congestion_level") * g("weather_condition_severity")
    if "distance_km" in df.columns and "loading_unloading_time" in df.columns:
        df["distance_x_loading"] = g("distance_km") * g("loading_unloading_time")
    if "port_congestion_level" in df.columns and "customs_clearance_time" in df.columns:
        df["port_x_customs"] = g("port_congestion_level") * g("customs_clearance_time")
    if "supplier_reliability_score" in df.columns and "route_risk_level" in df.columns:
        df["supplier_x_risk"] = g("supplier_reliability_score") * g("route_risk_level")
    return df


def add_telemetry_lags(df, cols, group_col=None, time_col="timestamp"):
    dfx = df.copy()
    if time_col in dfx.columns:
        dfx[time_col] = pd.to_datetime(dfx[time_col], errors="coerce")
    if group_col and group_col in dfx.columns:
        dfx = dfx.sort_values([group_col, time_col])
        grp = dfx.groupby(group_col, sort=False)
    else:
        dfx = dfx.sort_values(time_col)
        grp = dfx.groupby(lambda _: 0, sort=False)
    for c in cols:
        if c in dfx.columns:
            dfx[f"{c}_lag1"] = grp[c].shift(1)
            dfx[f"{c}_lag2"] = grp[c].shift(2)
            dfx[f"{c}_roll_mean3"] = grp[c].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
            dfx[f"{c}_roll_mean6"] = grp[c].rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)
    return dfx


def add_cumulative_distance(df, group_col=None, time_col="timestamp"):
    dfx = df.copy()
    if "distance_km" not in dfx.columns:
        dfx["distance_km_cum"] = np.nan
        return dfx
    if time_col in dfx.columns:
        dfx[time_col] = pd.to_datetime(dfx[time_col], errors="coerce")
    if group_col and group_col in dfx.columns:
        dfx = dfx.sort_values([group_col, time_col])
        dfx["distance_km_cum"] = dfx.groupby(group_col, sort=False)["distance_km"].cumsum()
    else:
        dfx = dfx.sort_values(time_col)
        dfx["distance_km_cum"] = dfx["distance_km"].cumsum()
    return dfx


def load_data(cfg: Config) -> pd.DataFrame:
    path = cfg.DATA_PATH
    if not os.path.exists(path) and os.path.exists("/mnt/data/dynamic_supply_chain_logistics_dataset.csv"):
        path = "/mnt/data/dynamic_supply_chain_logistics_dataset.csv"
    return pd.read_csv(path)


def preprocess_base(df):
    df = derive_distance_feature(df)
    df = add_time_parts(df)
    df = add_pt_holiday_features(df)
    df = add_interactions(df)
    lag_cols = [c for c in ["iot_temperature", "driver_behavior_score", "fatigue_monitoring_score"] if c in df.columns]
    df = add_telemetry_lags(df, lag_cols, group_col="vehicle_id" if "vehicle_id" in df.columns else None)
    df = add_cumulative_distance(df, group_col="vehicle_id" if "vehicle_id" in df.columns else None)
    return df


def build_preprocessor(df, x_cols):
    num_cols = [c for c in x_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in x_cols if not pd.api.types.is_numeric_dtype(df[c])]
    numeric = Pipeline([("scaler", StandardScaler())])
    categorical = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)], remainder="drop")


class ToPandas(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.feature_names_ = None
    def fit(self, X, y=None):
        self.preprocessor.fit(X, y)
        self.feature_names_ = self.preprocessor.get_feature_names_out()
        return self
    def transform(self, X):
        Xt = self.preprocessor.transform(X)
        import pandas as pd
        return pd.DataFrame(Xt, columns=self.feature_names_)


def metrics_regression(y_true, y_pred):
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred))}


def expected_calibration_error(y_true_binary, y_prob, n_bins=20):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true_binary)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins-1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = (y_true_binary[mask] == 1).mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / total) * abs(acc - conf)
    return float(ece)


def plot_scatter(y_true, y_pred, title, path):
    plt.figure()
    plt.scatter(y_true, y_pred, s=10)
    plt.title(title); plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.tight_layout(); plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close()


def plot_hist(values, title, xlabel, path):
    plt.figure()
    pd.Series(values).dropna().plot(kind="hist", bins=40)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("Frequency")
    plt.tight_layout(); plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close()


def plot_confusion(cm, classes, title, path):
    plt.figure()
    plt.imshow(cm, interpolation='nearest'); plt.title(title)
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right'); plt.yticks(range(len(classes)), classes)
    plt.colorbar(); plt.tight_layout(); plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close()


def plot_confusion_normalized(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    plt.figure()
    plt.imshow(cm, interpolation='nearest'); plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right'); plt.yticks(range(len(labels)), labels)
    plt.colorbar(); plt.tight_layout(); plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close()


def plot_calibration_curve(y_true, y_prob, title, path, n_bins=20):
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o"); plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(title + f" (bins={n_bins})"); plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.tight_layout(); plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close()


def crossval_regression(pipe, X, y, out_dir, random_state=42):
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2"
    }
    cv = cross_validate(pipe, X, y, cv=kf, scoring=scoring, n_jobs=None, return_train_score=False)
    df = pd.DataFrame({"fold": range(1, 6), "MAE": -cv["test_MAE"], "RMSE": -cv["test_RMSE"], "R2": cv["test_R2"]})
    df.loc["mean"] = ["mean", df["MAE"].mean(), df["RMSE"].mean(), df["R2"].mean()]
    df.loc["std"] = ["std", df["MAE"].std(ddof=1), df["RMSE"].std(ddof=1), df["R2"].std(ddof=1)]
    df.to_csv(os.path.join(out_dir, "cv_metrics.csv"), index=False)


def _maybe_shap_summary_for_shipping(pipe, X_va, out_dir):
    if not HAS_SHAP:
        return
    try:
        if "prep" in pipe.named_steps:
            Xp = pipe.named_steps["prep"].transform(X_va)
            model = pipe.named_steps["model"]
            if HAS_LGBM and isinstance(model, LGBMRegressor) or HAS_XGB and isinstance(model, XGBRegressor):
                sample = Xp.sample(n=min(2000, len(Xp)), random_state=42) if hasattr(Xp, "sample") else Xp
                expl = shap.TreeExplainer(model)
                shap_values = expl.shap_values(sample)
                plt.figure()
                shap.summary_plot(shap_values, sample, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "shap_summary.png"), dpi=160, bbox_inches="tight")
                plt.close()
    except Exception:
        pass


def train_regressor(df, y_col, x_cols, model_name, cfg: Config, prefer_booster=True):
    out_dir = os.path.join(cfg.OUTPUT_DIR, "regression", model_name)
    ensure_dir(out_dir)
    X = df[x_cols].copy(); y = df[y_col].copy()
    m = X.notna().all(axis=1) & y.notna(); X, y = X[m], y[m]
    if len(X) < 100:
        with open(os.path.join(out_dir, "note.txt"), "w") as f:
            f.write("Not enough rows after NA filtering.")
        return None

    pre = build_preprocessor(df, x_cols)

    if prefer_booster and HAS_LGBM:
        model = LGBMRegressor(random_state=cfg.RANDOM_STATE, n_estimators=500, learning_rate=0.05,
                              subsample=0.9, force_col_wise=True, verbose=-1)
        pipe = Pipeline([("prep", ToPandas(preprocessor=pre)), ("model", model)])
    elif prefer_booster and HAS_XGB:
        model = XGBRegressor(random_state=cfg.RANDOM_STATE, n_estimators=600, learning_rate=0.05,
                             max_depth=6, subsample=0.9, colsample_bytree=0.8, tree_method="hist")
        pipe = Pipeline([("pre", pre), ("model", model)])
    else:
        model = ElasticNet(alpha=0.001, l1_ratio=0.2, random_state=cfg.RANDOM_STATE, max_iter=4000)
        pipe = Pipeline([("pre", pre), ("model", model)])

    crossval_regression(pipe, X, y, out_dir, cfg.RANDOM_STATE)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=cfg.RANDOM_STATE)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_va)
    pd.Series(metrics_regression(y_va, y_pred)).to_csv(os.path.join(out_dir, "metrics.csv"))

    plot_scatter(y_va, y_pred, f"{model_name}: Actual vs Predicted", os.path.join(out_dir, "scatter.png"))
    plot_hist(y_va - y_pred, f"{model_name}: Error distribution", "Error", os.path.join(out_dir, "residuals.png"))

    try:
        if isinstance(model, ElasticNet):
            feat_names = pipe.named_steps["pre"].get_feature_names_out()
            coefs = np.ravel(pipe.named_steps["model"].coef_)
            dfc = pd.DataFrame({"feature": feat_names, "coef": coefs}).sort_values("coef", ascending=False)
            dfc.to_csv(os.path.join(out_dir, "business_rules_linear_coefs.csv"), index=False)
            dfc.head(10).to_csv(os.path.join(out_dir, "business_rules_top10.csv"), index=False)
        else:
            feat_names = pipe.named_steps.get("prep", pipe.named_steps.get("pre")).get_feature_names_out()
            importances = pipe.named_steps["model"].feature_importances_
            imp = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
            imp.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)
            imp.head(10).to_csv(os.path.join(out_dir, "feature_importance_top10.csv"), index=False)
    except Exception:
        pass

    if model_name == "shipping_costs":
        try:
            _maybe_shap_summary_for_shipping(pipe, X_va, out_dir)
        except Exception:
            pass

    return metrics_regression(y_va, y_pred)


def quantile_regression_delivery(df, cfg: Config, x_cols: List[str], alphas=(0.5, 0.9)):
    if "delivery_time_deviation" not in df.columns or not HAS_LGBM:
        return
    out_dir = os.path.join(cfg.OUTPUT_DIR, "regression", "delivery_time_deviation_quantiles")
    ensure_dir(out_dir)
    y_col = "delivery_time_deviation"
    X = df[x_cols].copy(); y = df[y_col].copy()
    m = X.notna().all(axis=1) & y.notna(); X, y = X[m], y[m]
    if len(X) < 200:
        with open(os.path.join(out_dir, "note.txt"), "w") as f:
            f.write("Not enough rows for quantile training.")
        return

    pre = build_preprocessor(df, x_cols)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=cfg.RANDOM_STATE)

    preds = {}; metrics_rows = []
    for q in alphas:
        model = LGBMRegressor(objective="quantile", alpha=q, random_state=cfg.RANDOM_STATE, n_estimators=700,
                              learning_rate=0.05, subsample=0.9, force_col_wise=True, verbose=-1)
        pipe = Pipeline([("prep", ToPandas(preprocessor=pre)), ("model", model)])
        pipe.fit(X_tr, y_tr)
        y_hat = pipe.predict(X_va)
        preds[q] = y_hat
        pin = mean_pinball_loss(y_va, y_hat, alpha=q)
        metrics_rows.append({"quantile": q, "mean_pinball_loss": pin})

        if q == 0.5:
            try:
                feat_names = pipe.named_steps["prep"].feature_names_
                imp = pd.DataFrame({"feature": feat_names, "importance": pipe.named_steps["model"].feature_importances_}) \
                        .sort_values("importance", ascending=False)
                imp.to_csv(os.path.join(out_dir, "feature_importance_q50.csv"), index=False)
            except Exception:
                pass

    pd.DataFrame(metrics_rows).to_csv(os.path.join(out_dir, "quantile_metrics.csv"), index=False)

    if 0.5 in preds and 0.9 in preds:
        plt.figure()
        idx = np.arange(len(y_va))
        plt.plot(idx, y_va.values, label="Actual", linewidth=1)
        plt.plot(idx, preds[0.5], label="Pred P50", linewidth=1)
        plt.plot(idx, preds[0.9], label="Pred P90", linewidth=1)
        plt.fill_between(idx, preds[0.5], preds[0.9], alpha=0.2, label="P50–P90 band")
        plt.title("Delivery time deviation — Quantile band (P50/P90)")
        plt.xlabel("Validation index"); plt.ylabel("Deviation"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "quantile_fan.png"), dpi=160, bbox_inches="tight"); plt.close()


def regression_suite(df, cfg: Config):
    results = {}
    x_ship = [c for c in [
        "distance_km","distance_km_cum","port_congestion_level","traffic_congestion_level",
        "handling_equipment_availability","loading_unloading_time","supplier_reliability_score",
        "lead_time_days","weather_condition_severity","route_risk_level","customs_clearance_time",
        "historical_demand","iot_temperature","driver_behavior_score","dow","weekofyear","hour","month",
        "is_holiday_pt","is_bridge_pt",
        "traffic_x_weather","distance_x_loading","port_x_customs","supplier_x_risk",
        "iot_temperature_lag1","iot_temperature_lag2","iot_temperature_roll_mean3","iot_temperature_roll_mean6"
    ] if c in df.columns]
    if "shipping_costs" in df.columns and len(x_ship) >= 5:
        results["shipping_costs"] = train_regressor(df, "shipping_costs", x_ship, "shipping_costs", cfg, prefer_booster=True)

    x_dev = [c for c in [
        "distance_km","distance_km_cum","traffic_congestion_level","port_congestion_level",
        "handling_equipment_availability","loading_unloading_time","supplier_reliability_score",
        "weather_condition_severity","route_risk_level","customs_clearance_time","historical_demand",
        "iot_temperature","driver_behavior_score","dow","weekofyear","hour","month",
        "is_holiday_pt","is_bridge_pt",
        "traffic_x_weather","distance_x_loading","port_x_customs","supplier_x_risk",
        "driver_behavior_score_lag1","driver_behavior_score_lag2","driver_behavior_score_roll_mean3","driver_behavior_score_roll_mean6"
    ] if c in df.columns]
    if "delivery_time_deviation" in df.columns and len(x_dev) >= 5:
        results["delivery_time_deviation"] = train_regressor(df, "delivery_time_deviation", x_dev, "delivery_time_deviation", cfg, prefer_booster=True)
        quantile_regression_delivery(df, cfg, x_dev, alphas=(0.5, 0.9))

    x_fuel = [c for c in [
        "distance_km","distance_km_cum","traffic_congestion_level","weather_condition_severity",
        "driver_behavior_score","fatigue_monitoring_score","iot_temperature","route_risk_level",
        "dow","weekofyear","hour","month","is_holiday_pt","is_bridge_pt","traffic_x_weather",
        "iot_temperature_lag1","iot_temperature_lag2","iot_temperature_roll_mean3","iot_temperature_roll_mean6",
        "driver_behavior_score_lag1","driver_behavior_score_lag2","driver_behavior_score_roll_mean3","driver_behavior_score_roll_mean6"
    ] if c in df.columns]
    if "fuel_consumption_rate" in df.columns and len(x_fuel) >= 5:
        results["fuel_consumption_rate"] = train_regressor(df, "fuel_consumption_rate", x_fuel, "fuel_consumption_rate", cfg, prefer_booster=True)

    x_lead = [c for c in [
        "distance_km","distance_km_cum","port_congestion_level","traffic_congestion_level",
        "handling_equipment_availability","loading_unloading_time","supplier_reliability_score",
        "weather_condition_severity","route_risk_level","customs_clearance_time","historical_demand",
        "iot_temperature","driver_behavior_score","dow","weekofyear","hour","month","is_holiday_pt","is_bridge_pt",
        "traffic_x_weather","distance_x_loading"
    ] if c in df.columns]
    if "lead_time_days" in df.columns and len(x_lead) >= 5:
        results["lead_time_days"] = train_regressor(df, "lead_time_days", x_lead, "lead_time_days", cfg, prefer_booster=True)

    return results


def _score_bands_table(y_true_bin, y_prob, out_csv):
    bins = np.linspace(0.0, 1.0, 6)  
    rows = []
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < len(bins)-2 else (y_prob >= lo) & (y_prob <= hi)
        n = mask.sum()
        if n == 0:
            rows.append({"band": f"[{lo:.1f},{hi:.1f}]", "n": 0, "delay_rate": np.nan})
        else:
            rate = float((y_true_bin[mask] == 1).mean())
            rows.append({"band": f"[{lo:.1f},{hi:.1f}]", "n": int(n), "delay_rate": rate})
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def classify_column(df, y_col: str, x_cols: List[str], cfg: Config, model_name: str):
    out_dir = os.path.join(cfg.OUTPUT_DIR, "classification", model_name); ensure_dir(out_dir)
    X = df[x_cols].copy(); y = df[y_col].copy()
    m = X.notna().all(axis=1) & y.notna(); X, y = X[m], y[m]
    if y.nunique() < 2:
        with open(os.path.join(out_dir, "notes.txt"), "a") as f: f.write("Single class — skipped.\n")
        return

    vc = y.value_counts(); rare = vc[vc < 2].index.tolist()
    if rare:
        keep = ~y.isin(rare); X, y = X[keep], y[keep]
        with open(os.path.join(out_dir, "notes.txt"), "a") as f: f.write(f"Removed rare classes: {rare}\n")
        if y.nunique() < 2:
            with open(os.path.join(out_dir, "notes.txt"), "a") as f: f.write("After filtering, single class — skipped.\n")
            return

    use_strat = (y.value_counts().min() >= 2)
    pre = build_preprocessor(df, x_cols)
    clf = LogisticRegression(max_iter=2000, multi_class="auto", class_weight="balanced")
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=cfg.RANDOM_STATE, stratify=y if use_strat else None)
    if not use_strat:
        with open(os.path.join(out_dir, "notes.txt"), "a") as f: f.write("Stratify disabled due to low counts.\n")

    pipe = Pipeline([("pre", pre), ("model", clf)]).fit(X_tr, y_tr)
    y_pred = pipe.predict(X_va)

    rep = classification_report(y_va, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(rep).to_csv(os.path.join(out_dir, "classification_report.csv"))

    labels = sorted(pd.Series(y_va).unique().tolist())
    cm = confusion_matrix(y_va, y_pred, labels=labels)
    plot_confusion(cm, labels, f"{model_name}: Confusion matrix", os.path.join(out_dir, "confusion_matrix.png"))
    plot_confusion_normalized(y_va, y_pred, labels, f"{model_name}: Confusion matrix (normalized)", os.path.join(out_dir, "confusion_matrix_normalized.png"))

    if y.nunique() == 2:
        y_prob = pipe.predict_proba(X_va)[:, 1]
        y_true_bin = (y_va == labels[1]).astype(int).values

        bs = brier_score_loss(y_true_bin, y_prob)
        with open(os.path.join(out_dir, "brier_score.txt"), "w") as f: f.write(f"Brier score: {bs:.6f}")

        plot_calibration_curve(y_true_bin, y_prob, f"{model_name}: Calibration", os.path.join(out_dir, "calibration_curve.png"), n_bins=20)

        precision, recall, thr = precision_recall_curve(y_true_bin, y_prob)
        ap = average_precision_score(y_true_bin, y_prob)
        plt.figure(); plt.step(recall, precision, where="post"); plt.title(f"{model_name}: PR curve\nAP={ap:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.tight_layout(); plt.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=160, bbox_inches="tight"); plt.close()

        f1_vals = 2 * (precision * recall) / (precision + recall + 1e-12)
        best_idx = int(np.nanargmax(f1_vals))
        best_thr = thr[max(best_idx - 1, 0)] if 0 < best_idx <= len(thr) else 0.5
        with open(os.path.join(out_dir, "best_threshold.txt"), "w") as f: f.write(f"{best_thr:.6f}")

        ece = expected_calibration_error(y_true_bin, y_prob, n_bins=20)
        with open(os.path.join(out_dir, "ece.txt"), "w") as f: f.write(f"ECE (20 bins): {ece:.6f}")

        _score_bands_table(y_true_bin, y_prob, os.path.join(out_dir, "score_bands.csv"))


def classification_suite(df, cfg: Config):
    x_base = [c for c in [
        "traffic_congestion_level","port_congestion_level","route_risk_level",
        "supplier_reliability_score","weather_condition_severity","disruption_likelihood_score",
        "dow","weekofyear","hour","month","is_holiday_pt","is_bridge_pt","traffic_x_weather"
    ] if c in df.columns]

    if "risk_classification" in df.columns and len(x_base) >= 2:
        classify_column(df, "risk_classification", x_base, cfg, "risk_classification")

    if "order_fulfillment_status" in df.columns and len(x_base) >= 2:
        classify_column(df, "order_fulfillment_status", x_base, cfg, "order_fulfillment_status")

    if "cargo_condition_status" in df.columns and len(x_base) >= 2:
        classify_column(df, "cargo_condition_status", x_base, cfg, "cargo_condition_status")

    if "delay_probability" in df.columns and len(x_base) >= 2 and pd.api.types.is_numeric_dtype(df["delay_probability"]):
        d2 = df.copy(); d2["delay_binary"] = (d2["delay_probability"] >= 0.5).astype(int)
        classify_column(d2, "delay_binary", x_base, cfg, "delay_probability_binary")


def time_series_suite(df, cfg: Config):
    out_dir = os.path.join(cfg.OUTPUT_DIR, "time_series"); ensure_dir(out_dir)
    if "timestamp" not in df.columns:
        with open(os.path.join(out_dir, "note.txt"), "w") as f: f.write("No timestamp column — forecasting skipped.")
        return

    dfx = df.copy(); dfx["timestamp"] = pd.to_datetime(dfx["timestamp"], errors="coerce")
    dfx = dfx.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    target = "lead_time_days"
    if target in dfx.columns:
        cols = [target] + [c for c in ["traffic_congestion_level","port_congestion_level","weather_condition_severity"] if c in dfx.columns]
        dfw = dfx[cols].resample("W").mean()

        years = dfw.index.year.unique().tolist()
        hol = set()
        for y in years:
            hol |= portugal_holidays(int(y))
        daily = pd.DataFrame(
            {"is_holiday_pt": [1 if ts.date() in hol else 0
                               for ts in pd.date_range(dfw.index.min()-pd.Timedelta(days=6),
                                                       dfw.index.max(), freq="D")]},
            index=pd.date_range(dfw.index.min()-pd.Timedelta(days=6), dfw.index.max(), freq="D")
        )
        hol_week = daily["is_holiday_pt"].resample("W").max()
        dfw["holidays_in_week"] = hol_week.reindex(dfw.index).fillna(0)

        for col in dfw.columns:
            if col == target:
                continue
            dfw[f"{col}_lag1"] = dfw[col].shift(1)
            dfw[f"{col}_lag2"] = dfw[col].shift(2)
        dfw = dfw.dropna()
        if len(dfw) > 20:
            n = len(dfw); fold_sizes = [int(n*0.6), int(n*0.8), n]
            rows = []
            for i, end in enumerate(fold_sizes[1:], start=1):
                train_end = fold_sizes[i-1]
                train_df = dfw.iloc[:train_end]; valid_df = dfw.iloc[train_end:end]
                if len(valid_df) < 4:
                    continue
                y_tr = train_df[target]; X_tr = train_df.drop(columns=[target])
                y_va = valid_df[target]; X_va = valid_df.drop(columns=[target])
                model = Ridge(alpha=0.1, random_state=cfg.RANDOM_STATE).fit(X_tr, y_tr)
                y_pred = model.predict(X_va)
                rows.append({"fold": i, **metrics_regression(y_va, y_pred)})
            if rows:
                pd.DataFrame(rows).to_csv(os.path.join(out_dir, "backtest_metrics.csv"), index=False)

            X_tr, X_va, y_tr, y_va = train_test_split(dfw.drop(columns=[target]), dfw[target],
                                                      test_size=0.2, random_state=cfg.RANDOM_STATE, shuffle=False)
            model = Ridge(alpha=0.1, random_state=cfg.RANDOM_STATE).fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            pd.Series(metrics_regression(y_va, y_pred)).to_csv(os.path.join(out_dir, "nowcast_metrics.csv"))
            plot_scatter(y_va, y_pred, "Nowcast lead_time_days", os.path.join(out_dir, "nowcast_scatter.png"))


def anomaly_suite(df, cfg: Config):
    out_dir = os.path.join(cfg.OUTPUT_DIR, "anomalies"); ensure_dir(out_dir)
    for col in ["fuel_consumption_rate","iot_temperature"]:
        if col not in df.columns:
            continue
        base = df.dropna(subset=[col]).copy()
        if len(base) < 100:
            continue
        clf = IsolationForest(random_state=cfg.RANDOM_STATE, contamination=0.01)
        clf.fit(base[[col]])
        pred = clf.predict(base[[col]])
        score = clf.decision_function(base[[col]])
        anom = base.loc[pred == -1, [col]].copy()
        anom["anomaly_score"] = score[pred == -1]
        for c in ["timestamp", "vehicle_gps_latitude", "vehicle_gps_longitude", "driver_behavior_score"]:
            if c in base.columns:
                anom[c] = base.loc[pred == -1, c].values
        anom.sort_values("anomaly_score", inplace=True)
        anom.to_csv(os.path.join(out_dir, f"anomalies_{col}_scored.csv"), index=False)
        plot_hist(base[col].values, f"Distribution — {col}", col, os.path.join(out_dir, f"hist_{col}.png"))

        if {"vehicle_gps_latitude","vehicle_gps_longitude"}.issubset(anom.columns):
            top = anom.head(min(200, len(anom)))
            if len(top) > 0:
                plt.figure(figsize=(7,6))
                plt.scatter(top["vehicle_gps_longitude"], top["vehicle_gps_latitude"], s=12)
                plt.title(f"Top anomalies — {col} (scatter)"); plt.xlabel("Longitude"); plt.ylabel("Latitude")
                plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"map_anomalies_{col}.png"), dpi=160, bbox_inches="tight"); plt.close()


def geospatial_suite(df, cfg: Config):
    out_dir = os.path.join(cfg.OUTPUT_DIR, "geospatial"); ensure_dir(out_dir)
    if not all(c in df.columns for c in ["vehicle_gps_latitude","vehicle_gps_longitude"]):
        with open(os.path.join(out_dir, "note.txt"), "w") as f:
            f.write("No GPS columns — geospatial skipped.")
        return
    dfx = df.dropna(subset=["vehicle_gps_latitude","vehicle_gps_longitude"]).copy()
    coords_rad = np.radians(dfx[["vehicle_gps_latitude","vehicle_gps_longitude"]].values)
    eps = 1.0/6371.0088
    clustering = DBSCAN(eps=eps, min_samples=50, metric="haversine").fit(coords_rad)
    dfx["cluster"] = clustering.labels_
    dfx[["vehicle_gps_latitude","vehicle_gps_longitude","cluster"]].to_csv(os.path.join(out_dir, "points_with_clusters.csv"), index=False)
    cent = dfx.groupby("cluster")[["vehicle_gps_latitude","vehicle_gps_longitude"]].mean().reset_index()
    cent.to_csv(os.path.join(out_dir, "clusters_centroids.csv"), index=False)

    plt.figure(figsize=(7,6))
    plt.hexbin(dfx["vehicle_gps_longitude"], dfx["vehicle_gps_latitude"], gridsize=60, bins='log')
    if "vehicle_gps_longitude" in cent.columns and "vehicle_gps_latitude" in cent.columns:
        plt.scatter(cent["vehicle_gps_longitude"], cent["vehicle_gps_latitude"], s=30, edgecolor='k')
    plt.title("GPS density (hexbin) with cluster centroids"); plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "heatmap_hexbin_centroids.png"), dpi=160, bbox_inches="tight"); plt.close()



def export_for_powerbi(cfg: Config):
    src_root = cfg.OUTPUT_DIR
    bi_root = os.path.join(src_root, "BI")
    ensure_dir(bi_root)

    patterns = [
        
        "regression/*/metrics.csv",
        "regression/*/cv_metrics.csv",
        "time_series/nowcast_metrics.csv",
        "time_series/backtest_metrics.csv",

        
        "classification/*/classification_report.csv",
        "classification/delay_probability_binary/score_bands.csv",

        
        "regression/*/feature_importance_top10.csv",
        "regression/*/feature_importance.csv",
        "regression/*/business_rules_top10.csv",

        
        "regression/delivery_time_deviation_quantiles/feature_importance_q50.csv",
        "regression/delivery_time_deviation_quantiles/quantile_metrics.csv",

        
        "anomalies/anomalies_fuel_consumption_rate_scored.csv",
        "anomalies/anomalies_iot_temperature_scored.csv",

        
        "geospatial/points_with_clusters.csv",
        "geospatial/clusters_centroids.csv",
    ]

    copied = []
    for pat in patterns:
        for src in glob.glob(os.path.join(src_root, pat)):
            if not os.path.isfile(src):
                continue
            rel = os.path.relpath(src, src_root).replace("\\", "/")
            flat = rel.replace("/", "__")  
            dst = os.path.join(bi_root, flat)
            ensure_dir(os.path.dirname(dst))
            shutil.copy2(src, dst)
            copied.append({"source": rel, "destination": f"BI/{flat}"})

    if copied:
        pd.DataFrame(copied).to_csv(os.path.join(bi_root, "_BI_index.csv"), index=False)


def main():
    cfg = Config(); ensure_dir(cfg.OUTPUT_DIR)
    df = load_data(cfg); df = preprocess_base(df)

    regression_suite(df, cfg)
    classification_suite(df, cfg)
    time_series_suite(df, cfg)
    anomaly_suite(df, cfg)
    geospatial_suite(df, cfg)

    
    export_for_powerbi(cfg)

    print("Done:", cfg.OUTPUT_DIR)


if __name__ == "__main__":
    main()
