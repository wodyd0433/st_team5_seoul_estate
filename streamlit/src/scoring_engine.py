from __future__ import annotations

from math import atan2, cos, radians, sin, sqrt

import numpy as np
import pandas as pd
import streamlit as st

from src.cleaning import fill_missing_values
from src.config import GU_CENTERS, SCORING_CACHE_TTL, WORKPLACE_HUBS


FIXED_WEIGHTS = {
    "price": 0.35,
    "commute": 0.30,
    "infra": 0.15,
    "safety": 0.10,
    "risk": 0.10,
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return r * c


def _star_string(stars: float | int | None) -> str:
    if stars is None or pd.isna(stars):
        return "-"
    return "★" * int(stars) + "☆" * (5 - int(stars))


def _grade_from_total_score(score: float) -> tuple[int, str, str]:
    if pd.isna(score):
        return 1, "D", "비추천"
    if score >= 90:
        return 5, "S", "강력 추천"
    if score >= 75:
        return 4, "A", "우수"
    if score >= 55:
        return 3, "B", "무난"
    if score >= 35:
        return 2, "C", "주의"
    return 1, "D", "비추천"


def _score_from_star(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(1).clip(1, 5) * 20


def _build_std_thresholds(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce")
    mean = float(values.mean()) if values.notna().any() else 0.0
    std = float(values.std(ddof=0)) if values.notna().any() else 0.0
    return {
        "mean": mean,
        "std": std,
        "lower_1_5_std": mean - 1.5 * std,
        "lower_0_5_std": mean - 0.5 * std,
        "upper_0_5_std": mean + 0.5 * std,
        "upper_1_5_std": mean + 1.5 * std,
    }


def _calculate_percentile_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().all():
        return pd.Series(0.0, index=series.index)
    
    # rank(pct=True) returns [0, 1]
    # To avoid 0.0 for the absolute minimum, we can use (rank - 0.5) / count or just stick to standard pct
    pct = values.rank(pct=True, method="min") * 100
    if not higher_is_better:
        pct = 100 - pct
    return pct.fillna(0.0)


def _score_star_from_thresholds(series: pd.Series, thresholds: dict[str, float], higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if higher_is_better:
        stars = np.select(
            [
                values.ge(thresholds["upper_1_5_std"]),
                values.ge(thresholds["upper_0_5_std"]),
                values.ge(thresholds["lower_0_5_std"]),
                values.ge(thresholds["lower_1_5_std"]),
            ],
            [5, 4, 3, 2],
            default=1,
        )
    else:
        stars = np.select(
            [
                values.le(thresholds["lower_1_5_std"]),
                values.le(thresholds["lower_0_5_std"]),
                values.le(thresholds["upper_0_5_std"]),
                values.le(thresholds["upper_1_5_std"]),
            ],
            [5, 4, 3, 2],
            default=1,
        )
    return pd.Series(stars, index=series.index, dtype="int64")


def prepare_commute_frame(
    primary_workplace_name: str,
    feature_table: pd.DataFrame,
    commute_models: pd.DataFrame,
    commute_weighted_avg: pd.DataFrame | None = None,
    household_type: str = "2인 맞벌이",
    secondary_workplace_name: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    dual_income = str(household_type).startswith("2")
    primary = WORKPLACE_HUBS[primary_workplace_name]
    secondary = WORKPLACE_HUBS.get(secondary_workplace_name) if secondary_workplace_name else None
    model_lookup = commute_models.set_index("hub_name").to_dict("index")
    primary_model = model_lookup[primary_workplace_name]
    secondary_model = model_lookup.get(secondary_workplace_name) if secondary_workplace_name else None
    weighted_lookup = {}
    if isinstance(commute_weighted_avg, pd.DataFrame) and not commute_weighted_avg.empty:
        weighted_lookup = commute_weighted_avg.set_index(["hub_name", "gu"])["avg_commute_minutes"].to_dict()

    rows: list[dict[str, object]] = []
    for gu in feature_table["gu"].dropna().unique():
        center = GU_CENTERS.get(gu)
        if not center:
            continue
        primary_distance_km = _haversine_km(center["lat"], center["lon"], primary["lat"], primary["lon"])
        primary_minutes = weighted_lookup.get((primary_workplace_name, gu))
        if primary_minutes is None or pd.isna(primary_minutes):
            primary_minutes = (
                primary_model["intercept"]
                + primary_distance_km * primary_model["distance_coef"]
                + primary_model["avg_transfer"] * primary_model["transfer_coef"]
            )
        secondary_distance_km = None
        secondary_minutes = None
        if dual_income and secondary is not None and secondary_model is not None:
            secondary_distance_km = _haversine_km(center["lat"], center["lon"], secondary["lat"], secondary["lon"])
            secondary_minutes = weighted_lookup.get((secondary_workplace_name, gu))
            if secondary_minutes is None or pd.isna(secondary_minutes):
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
    weights: dict[str, float] | None = None,
    threshold_overrides: dict[str, dict[str, float | str | bool]] | None = None,
    scaling_method: str | None = None,
    missing_strategy: str = "mean",
    score_formula: str | None = None,
    household_type: str = "2인 맞벌이",
    desired_contract_type: str = "전세",
) -> tuple[pd.DataFrame, dict[str, float]]:
    del scaling_method, score_formula, household_type

    frame = feature_table.copy().merge(commute_frame, on="gu", how="left")
    frame = fill_missing_values(frame, missing_strategy)
    if selected_gus:
        frame = frame.loc[frame["gu"].isin(selected_gus)].copy()

    frame["infra_index"] = (
        pd.to_numeric(frame.get("mart_count"), errors="coerce").fillna(0) * 2
        + pd.to_numeric(frame.get("hospital_count"), errors="coerce").fillna(0) * 1.5
        + pd.to_numeric(frame.get("park_count"), errors="coerce").fillna(0)
    )

    # 계약 방식에 따라 가격 지표 선택 (전세 보증금 vs 표준화 월세)
    price_col = "standardized_monthly_rent_krw" if desired_contract_type == "월세" else "deposit_price_krw"
    # 만약 해당 컬럼이 없으면 (구버전 데이터 등) 기본 보증금 사용
    if price_col not in frame.columns:
        price_col = "deposit_price_krw"

    metric_specs = {
        "price": {"series": frame[price_col], "higher_is_better": False, "source_col": price_col},
        "commute": {"series": frame["commute_minutes"], "higher_is_better": False, "source_col": "commute_minutes"},
        "infra": {"series": frame["infra_index"], "higher_is_better": True, "source_col": "infra_index"},
        "safety": {"series": frame["crime_total_count"], "higher_is_better": False, "source_col": "crime_total_count"},
        "risk": {"series": frame["jeonse_ratio_pct"], "higher_is_better": False, "source_col": "jeonse_ratio_pct"},
    }
    threshold_meta: dict[str, dict[str, float | str | bool]] = {}
    for metric_name, spec in metric_specs.items():
        thresholds = _build_std_thresholds(spec["series"])
        override = (threshold_overrides or {}).get(metric_name, {})
        for key in ["mean", "std", "lower_1_5_std", "lower_0_5_std", "upper_0_5_std", "upper_1_5_std"]:
            if key in override and pd.notna(override[key]):
                thresholds[key] = float(override[key])
        threshold_meta[metric_name] = {
            **thresholds,
            "higher_is_better": spec["higher_is_better"],
            "source_col": spec["source_col"],
        }
        # 백분위수 기반 세부 점수 계산 (0~100)
        score = _calculate_percentile_score(spec["series"], spec["higher_is_better"])
        frame[f"{metric_name}_score"] = score
        # 점수에 따른 별점 환산 (0~20: 1성, ..., 80~100: 5성)
        # 100점인 경우 5성을 유지하기 위해 20으로 나눈 뒤 올림 처리하거나 버림 후 1더함
        stars = (score / 20).apply(lambda x: int(np.floor(x)) if x < 100 else 4) + 1
        frame[f"{metric_name}_star"] = stars

    frame["budget_score"] = frame["price_score"]
    frame["jeonse_risk_score"] = frame["risk_score"]

    active_weights = weights or FIXED_WEIGHTS
    # 종합 점수는 각 세부 점수(0~100)의 가중 평균
    frame["total_score"] = (
        frame["price_score"] * active_weights["price"]
        + frame["commute_score"] * active_weights["commute"]
        + frame["infra_score"] * active_weights["infra"]
        + frame["safety_score"] * active_weights["safety"]
        + frame["risk_score"] * active_weights["risk"]
    )

    total_meta = frame["total_score"].apply(_grade_from_total_score)
    frame["total_star"] = total_meta.map(lambda x: x[0]).astype(int)
    frame["total_grade"] = total_meta.map(lambda x: x[1])
    frame["total_grade_label"] = total_meta.map(lambda x: x[2])

    for column in ["price_star", "commute_star", "infra_star", "safety_star", "risk_star", "total_star"]:
        label_column = column.replace("_star", "_star_label")
        frame[label_column] = frame[column].map(_star_string)

    frame["risk_critical"] = frame["risk_star"].eq(1)
    frame["risk_warning"] = np.where(
        frame["risk_critical"],
        "전세가율 80% 이상으로 역전세 위험이 높아 계약 회피 권고",
        "",
    )
    frame["score_rank"] = (
        frame.sort_values(["risk_critical", "total_score", "risk_score"], ascending=[True, False, False])
        .reset_index(drop=True)
        .index
        + 1
    )
    frame = frame.sort_values(
        ["risk_critical", "total_score", "risk_score", "price_score"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    frame["score_rank"] = frame.index + 1

    return frame, {"weights": active_weights.copy(), "thresholds": threshold_meta}
