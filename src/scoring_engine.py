from __future__ import annotations

from math import atan2, cos, radians, sin, sqrt

import pandas as pd
import streamlit as st

from src.cleaning import fill_missing_values
from src.config import GU_CENTERS, SCORING_CACHE_TTL, WORKPLACE_HUBS


def _minmax(series: pd.Series, reverse: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0, index=series.index, dtype="float64")
    denom = s.max() - s.min()
    scaled = pd.Series(50, index=series.index, dtype="float64") if denom == 0 else (s - s.min()) / denom * 100
    return 100 - scaled if reverse else scaled


def _zscore(series: pd.Series, reverse: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0, index=series.index, dtype="float64")
    std = s.std(ddof=0)
    scaled = pd.Series(0, index=series.index, dtype="float64") if std == 0 else (s - s.mean()) / std
    if scaled.max() == scaled.min():
        scaled = pd.Series(50, index=series.index, dtype="float64")
    else:
        scaled = (scaled - scaled.min()) / (scaled.max() - scaled.min()) * 100
    return 100 - scaled if reverse else scaled


def _scale(series: pd.Series, method: str, reverse: bool = False) -> pd.Series:
    if method == "MinMax":
        return _minmax(series, reverse=reverse)
    if method == "Z-score":
        return _zscore(series, reverse=reverse)
    ranked = pd.to_numeric(series, errors="coerce").rank(pct=True) * 100
    ranked = ranked.fillna(0)
    return 100 - ranked if reverse else ranked


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return r * c


def prepare_commute_frame(
    primary_workplace_name: str,
    feature_table: pd.DataFrame,
    commute_models: pd.DataFrame,
    household_type: str = "2인(맞벌이)",
    secondary_workplace_name: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    dual_income = str(household_type).startswith("2")
    primary = WORKPLACE_HUBS[primary_workplace_name]
    secondary = WORKPLACE_HUBS.get(secondary_workplace_name) if secondary_workplace_name else None
    model_lookup = commute_models.set_index("hub_name").to_dict("index")
    primary_model = model_lookup[primary_workplace_name]
    secondary_model = model_lookup.get(secondary_workplace_name) if secondary_workplace_name else None
    rows: list[dict[str, object]] = []
    for gu in feature_table["gu"].dropna().unique():
        center = GU_CENTERS.get(gu)
        if not center:
            continue
        primary_distance_km = _haversine_km(center["lat"], center["lon"], primary["lat"], primary["lon"])
        primary_minutes = (
            primary_model["intercept"]
            + primary_distance_km * primary_model["distance_coef"]
            + primary_model["avg_transfer"] * primary_model["transfer_coef"]
        )
        secondary_distance_km = None
        secondary_minutes = None
        if dual_income and secondary is not None:
            secondary_distance_km = _haversine_km(center["lat"], center["lon"], secondary["lat"], secondary["lon"])
            secondary_minutes = (
                secondary_model["intercept"]
                + secondary_distance_km * secondary_model["distance_coef"]
                + secondary_model["avg_transfer"] * secondary_model["transfer_coef"]
            )
            combined_minutes = primary_minutes * 0.55 + secondary_minutes * 0.45
            worst_minutes = max(primary_minutes, secondary_minutes)
        else:
            combined_minutes = primary_minutes
            worst_minutes = primary_minutes
        rows.append(
            {
                "gu": gu,
                "commute_minutes": round(combined_minutes, 1),
                "primary_commute_minutes": round(primary_minutes, 1),
                "secondary_commute_minutes": round(secondary_minutes, 1) if secondary_minutes is not None else None,
                "worst_commute_minutes": round(worst_minutes, 1),
                "distance_km": round(primary_distance_km, 2),
                "secondary_distance_km": round(secondary_distance_km, 2) if secondary_distance_km is not None else None,
                "workplace_name": primary_workplace_name,
                "secondary_workplace_name": secondary_workplace_name,
            }
        )
    return pd.DataFrame(rows), {
        "workplace_name": primary_workplace_name,
        "secondary_workplace_name": secondary_workplace_name,
        "household_type": household_type,
    }


@st.cache_data(ttl=SCORING_CACHE_TTL, show_spinner=False)
def score_recommendations(
    feature_table: pd.DataFrame,
    selected_gus: list[str],
    commute_frame: pd.DataFrame,
    weights: dict[str, float],
    scaling_method: str,
    missing_strategy: str,
    score_formula: str = "가중 합산",
    household_type: str = "2인(맞벌이)",
) -> tuple[pd.DataFrame, dict[str, float]]:
    dual_income = str(household_type).startswith("2")
    frame = feature_table.copy()
    frame = frame.merge(commute_frame, on="gu", how="left")
    frame = fill_missing_values(frame, missing_strategy)
    if selected_gus:
        frame = frame.loc[frame["gu"].isin(selected_gus)].copy()

    base_monthly = frame["monthly_rent_active_krw"].fillna(frame["monthly_rent_krw"]).fillna(0)
    frame["deposit_score"] = _scale(frame["deposit_price_krw"], scaling_method, reverse=True)
    frame["monthly_score"] = _scale(base_monthly, scaling_method, reverse=True)
    if str(household_type).startswith("1"):
        frame["price_score"] = frame["deposit_score"] * 0.45 + frame["monthly_score"] * 0.55
    else:
        frame["price_score"] = frame["deposit_score"] * 0.6 + frame["monthly_score"] * 0.4
    frame["infra_score"] = _scale(frame["infra_score_raw"], scaling_method)
    frame["safety_score"] = _scale(frame["safety_score_raw"], scaling_method)
    frame["redevelopment_score"] = _scale(frame["redevelopment_score_raw"], scaling_method)

    commute_available = frame["commute_minutes"].notna().sum() > 0
    applied_commute_weight = weights["commute"] if commute_available else 0.0
    if commute_available:
        if dual_income and "worst_commute_minutes" in frame.columns:
            avg_commute_score = _scale(frame["commute_minutes"], scaling_method, reverse=True)
            worst_commute_score = _scale(frame["worst_commute_minutes"], scaling_method, reverse=True)
            frame["commute_score"] = avg_commute_score * 0.6 + worst_commute_score * 0.4
        else:
            frame["commute_score"] = _scale(frame["commute_minutes"], scaling_method, reverse=True)
    else:
        frame["commute_score"] = 0.0
    frame["budget_score"] = (
        frame["housing_budget_fit"] * 70
        + _scale(frame["price_burden_index"] + frame["monthly_burden_index"], scaling_method, reverse=True) * 0.3
    )
    total_weight = weights["budget"] + weights["infra"] + weights["safety"] + applied_commute_weight
    total_weight = total_weight if total_weight > 0 else 1.0
    weighted_sum = (
        frame["budget_score"] * weights["budget"]
        + frame["infra_score"] * weights["infra"]
        + frame["safety_score"] * weights["safety"]
        + frame["commute_score"] * applied_commute_weight
    ) / total_weight
    if score_formula == "병목 기준":
        frame["total_score"] = (
            frame[["budget_score", "infra_score", "safety_score", "commute_score"]].min(axis=1) * 0.55
            + weighted_sum * 0.45
        )
    elif score_formula == "균형 보정":
        dispersion = frame[["budget_score", "infra_score", "safety_score", "commute_score"]].std(axis=1).fillna(0)
        frame["total_score"] = weighted_sum - dispersion * 0.15
    else:
        frame["total_score"] = weighted_sum
    frame["score_rank"] = frame["total_score"].rank(method="dense", ascending=False).astype(int)
    frame = frame.sort_values(["total_score", "housing_budget_fit"], ascending=[False, False]).reset_index(drop=True)
    return frame, {"commute_weight_applied": applied_commute_weight}
