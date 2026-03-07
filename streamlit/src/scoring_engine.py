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


def _price_star(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select(
            [
                values.lt(350_000_000),
                values.lt(500_000_000),
                values.lt(700_000_000),
                values.lt(950_000_000),
            ],
            [5, 4, 3, 2],
            default=1,
        ),
        index=series.index,
        dtype="int64",
    )


def _commute_star(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select(
            [
                values.le(20),
                values.le(30),
                values.le(45),
                values.le(60),
            ],
            [5, 4, 3, 2],
            default=1,
        ),
        index=series.index,
        dtype="int64",
    )


def _risk_star(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select(
            [
                values.lt(50),
                values.lt(60),
                values.lt(70),
                values.lt(80),
            ],
            [5, 4, 3, 2],
            default=1,
        ),
        index=series.index,
        dtype="int64",
    )


def _infra_star(frame: pd.DataFrame) -> pd.Series:
    mart = pd.to_numeric(frame.get("mart_count"), errors="coerce").fillna(0)
    hospital = pd.to_numeric(frame.get("hospital_count"), errors="coerce").fillna(0)
    park = pd.to_numeric(frame.get("park_count"), errors="coerce").fillna(0)

    hospital_q70 = hospital.quantile(0.7) if hospital.notna().any() else 0
    park_q70 = park.quantile(0.7) if park.notna().any() else 0
    infra_index = mart * 2 + hospital * 1.5 + park
    infra_q50 = infra_index.quantile(0.5) if infra_index.notna().any() else 0
    infra_q20 = infra_index.quantile(0.2) if infra_index.notna().any() else 0

    stars = np.select(
        [
            (mart >= 3) & (hospital >= 2) & (park >= park_q70),
            (mart >= 2) & (hospital >= hospital_q70) & (park >= park_q70),
            (mart >= 1) & (infra_index >= infra_q50),
            infra_index >= infra_q20,
        ],
        [5, 4, 3, 2],
        default=1,
    )
    return pd.Series(stars, index=frame.index, dtype="int64")


def _safety_star(frame: pd.DataFrame) -> pd.Series:
    crime = pd.to_numeric(frame.get("crime_total_count"), errors="coerce")
    police = pd.to_numeric(frame.get("police_satisfaction_score"), errors="coerce")

    crime = crime.fillna(crime.median() if crime.notna().any() else 0)
    police = police.fillna(police.median() if police.notna().any() else 0)

    crime_q10 = crime.quantile(0.10)
    crime_q25 = crime.quantile(0.25)
    crime_q75 = crime.quantile(0.75)
    crime_q90 = crime.quantile(0.90)
    police_q90 = police.quantile(0.90)

    stars = np.select(
        [
            (crime <= crime_q10) & (police >= police_q90),
            crime <= crime_q25,
            crime <= crime_q75,
            crime <= crime_q90,
        ],
        [5, 4, 3, 2],
        default=1,
    )
    return pd.Series(stars, index=frame.index, dtype="int64")


def prepare_commute_frame(
    primary_workplace_name: str,
    feature_table: pd.DataFrame,
    commute_models: pd.DataFrame,
    household_type: str = "2인 맞벌이",
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
        if dual_income and secondary is not None and secondary_model is not None:
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
    weights: dict[str, float] | None = None,
    scaling_method: str | None = None,
    missing_strategy: str = "mean",
    score_formula: str | None = None,
    household_type: str = "2인 맞벌이",
) -> tuple[pd.DataFrame, dict[str, float]]:
    del scaling_method, score_formula, household_type

    frame = feature_table.copy().merge(commute_frame, on="gu", how="left")
    frame = fill_missing_values(frame, missing_strategy)
    if selected_gus:
        frame = frame.loc[frame["gu"].isin(selected_gus)].copy()

    frame["price_star"] = _price_star(frame["deposit_price_krw"])
    frame["commute_star"] = _commute_star(frame["commute_minutes"])
    frame["infra_star"] = _infra_star(frame)
    frame["safety_star"] = _safety_star(frame)
    frame["risk_star"] = _risk_star(frame["jeonse_ratio_pct"])

    frame["price_score"] = _score_from_star(frame["price_star"])
    frame["commute_score"] = _score_from_star(frame["commute_star"])
    frame["infra_score"] = _score_from_star(frame["infra_star"])
    frame["safety_score"] = _score_from_star(frame["safety_star"])
    frame["risk_score"] = _score_from_star(frame["risk_star"])

    frame["budget_score"] = frame["price_score"]
    frame["jeonse_risk_score"] = frame["risk_score"]

    active_weights = weights or FIXED_WEIGHTS
    frame["total_score"] = (
        frame["price_star"] * active_weights["price"]
        + frame["commute_star"] * active_weights["commute"]
        + frame["infra_star"] * active_weights["infra"]
        + frame["safety_star"] * active_weights["safety"]
        + frame["risk_star"] * active_weights["risk"]
    ) * 20

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

    return frame, active_weights.copy()
