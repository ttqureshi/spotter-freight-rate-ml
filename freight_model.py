"""Shared feature engineering and model helpers for freight rate prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
TARGET = "posted_rate"
CAT_COLS = ["equipment", "pickup", "delivery", "lane"]

FEATURE_COLS = [
    "distance",
    "log_distance",
    "weight",
    "log_weight",
    "weight_missing",
    "market_index",
    "market_index_missing",
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "mid_lat",
    "mid_lon",
    "bearing",
    "month",
    "day_of_week",
    "is_weekend",
    "month_sin",
    "month_cos",
    "equipment",
    "pickup",
    "delivery",
    "lane",
]

MODEL_DIR = Path(__file__).resolve().parent / "models"
DATA_DIR = Path(__file__).resolve().parent / "data"


def fit_preprocessing_stats(dev_df: pd.DataFrame, labeled_df: pd.DataFrame) -> dict:
    valid_weight = dev_df[dev_df["weight"] > 0]
    return {
        "weight_by_equipment": valid_weight.groupby("equipment")["weight"].median(),
        "market_by_month": labeled_df.groupby(labeled_df["date"].dt.month)[
            "market_index"
        ].median(),
        "market_index_median": labeled_df["market_index"].median(),
        "quote_signal_median": labeled_df["quote_signal"].median(),
        "pickup_coords": labeled_df.groupby("pickup")[["pickup_lat", "pickup_lon"]].first(),
        "delivery_coords": labeled_df.groupby("delivery")[
            ["delivery_lat", "delivery_lon"]
        ].first(),
    }


def enrich_raw_frame(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = df.copy()
    if "pickup_lat" not in out.columns:
        out = out.join(stats["pickup_coords"], on="pickup")
    if "delivery_lat" not in out.columns:
        out = out.join(stats["delivery_coords"], on="delivery")
    if "market_index" not in out.columns:
        out["market_index"] = np.nan
    if "quote_signal" not in out.columns:
        out["quote_signal"] = np.nan
    if "date" not in out.columns or out["date"].dtype == "object":
        out["date"] = pd.to_datetime(out["date"])
    return out


def bearing_degrees(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y))


def engineer_features(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = enrich_raw_frame(df, stats)

    out["weight_missing"] = out["weight"].isna() | (out["weight"] < 0)
    out.loc[out["weight"] < 0, "weight"] = np.nan
    out["weight"] = out["weight"].fillna(out["equipment"].map(stats["weight_by_equipment"]))

    out["market_index_missing"] = out["market_index"].isna().astype(int)
    out["market_index"] = (
        out["market_index"]
        .fillna(out["date"].dt.month.map(stats["market_by_month"]))
        .fillna(stats["market_index_median"])
    )
    out["quote_signal"] = out["quote_signal"].fillna(stats["quote_signal_median"])

    out["month"] = out["date"].dt.month
    out["day_of_week"] = out["date"].dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    out["log_distance"] = np.log1p(out["distance"])
    out["log_weight"] = np.log1p(out["weight"])
    out["lane"] = out["pickup"] + " -> " + out["delivery"]

    out["mid_lat"] = (out["pickup_lat"] + out["delivery_lat"]) / 2
    out["mid_lon"] = (out["pickup_lon"] + out["delivery_lon"]) / 2
    out["bearing"] = bearing_degrees(
        out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"]
    )

    return out


def make_estimator(name: str):
    if name == "random forest":
        return RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "hist gradient boosting":
        return HistGradientBoostingRegressor(
            max_depth=10,
            learning_rate=0.08,
            max_iter=250,
            random_state=RANDOM_STATE,
        )
    if name == "hist gradient boosting (deeper)":
        return HistGradientBoostingRegressor(
            max_depth=14,
            learning_rate=0.05,
            max_iter=350,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model name: {name}")


def make_pipeline(feature_list: list[str], model, dense_output: bool = False) -> Pipeline:
    num_cols = [col for col in feature_list if col not in CAT_COLS]
    preprocessor = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                    sparse_output=not dense_output,
                ),
                CAT_COLS,
            ),
        ]
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


def predict_rate(model, feature_df: pd.DataFrame, log_target: bool = False) -> np.ndarray:
    preds = model.predict(feature_df[FEATURE_COLS])
    if log_target:
        preds = np.expm1(preds)
    return np.maximum(preds, 1)


def save_artifacts(
    model,
    feature_stats: dict,
    *,
    log_target: bool,
    best_name: str,
    model_dir: Path | None = None,
) -> Path:
    out_dir = model_dir or MODEL_DIR
    out_dir.mkdir(exist_ok=True)

    joblib.dump(model, out_dir / "final_model.joblib")
    joblib.dump(feature_stats, out_dir / "feature_stats.joblib")
    joblib.dump(
        {
            "feature_cols": FEATURE_COLS,
            "log_target": log_target,
            "best_name": best_name,
            "random_state": RANDOM_STATE,
        },
        out_dir / "model_config.joblib",
    )
    return out_dir


def load_artifacts(model_dir: Path | None = None):
    root = model_dir or MODEL_DIR
    model = joblib.load(root / "final_model.joblib")
    feature_stats = joblib.load(root / "feature_stats.joblib")
    config = joblib.load(root / "model_config.joblib")
    return model, feature_stats, config
