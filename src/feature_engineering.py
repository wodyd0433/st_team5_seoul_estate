from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.cleaning import remove_iqr_outliers
from src.config import FEATURE_CACHE_TTL, SEOUL_GUS


def _filter_by_area_range(frame: pd.DataFrame, area_column: str, min_area_pyeong: int, max_area_pyeong: int) -> pd.DataFrame:
    if area_column not in frame.columns:
        return frame
    min_m2 = min_area_pyeong * 3.3058
    max_m2 = max_area_pyeong * 3.3058
    area_series = pd.to_numeric(frame[area_column], errors="coerce")
    filtered = frame.loc[area_series.between(min_m2, max_m2, inclusive="both")].copy()
    return filtered if not filtered.empty else frame


def _aggregate_rent(rent: pd.DataFrame) -> pd.DataFrame:
    frame = rent.copy()
    frame["year"] = frame["년월"].astype(str).str[:4].astype(int)
    positive_monthly = frame.loc[frame["월세_만원_krw"].fillna(0) > 0].copy()
    grouped = frame.groupby(["gu", "year"], dropna=False).agg(
        deposit_price_krw=("보증금_만원_krw", "median"),
        monthly_rent_krw=("월세_만원_krw", "median"),
        rent_area_m2=("전용면적_m2", "median"),
        rent_build_year=("건축년도", lambda s: pd.to_numeric(s, errors="coerce").median()),
        rent_txn_count=("gu", "size"),
    )
    positive_grouped = (
        positive_monthly.groupby(["gu", "year"], dropna=False)
        .agg(
            monthly_rent_active_krw=("월세_만원_krw", "median"),
            monthly_rent_positive_ratio=("월세_만원_krw", lambda s: (s.fillna(0) > 0).mean()),
        )
        .reset_index()
    )
    return grouped.reset_index().merge(positive_grouped, on=["gu", "year"], how="left")


def _aggregate_sale(sale: pd.DataFrame) -> pd.DataFrame:
    frame = sale.copy()
    frame["sale_price_krw"] = pd.to_numeric(frame["dealAmount_krw"], errors="coerce")
    grouped = frame.groupby(["gu", "dealYear"], dropna=False).agg(
        sale_price_krw=("sale_price_krw", "median"),
        sale_area_m2=("excluUseAr", "median"),
        sale_build_year=("buildYear", lambda s: pd.to_numeric(s, errors="coerce").median()),
        sale_txn_count=("gu", "size"),
    )
    return grouped.reset_index().rename(columns={"dealYear": "year"})


def _aggregate_yearly_rent(yearly_rent: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for year, df in yearly_rent.items():
        temp = df.copy()
        temp["year"] = year
        if "gu" not in temp.columns:
            gu_candidates = [col for col in temp.columns if "구" in str(col)]
            if gu_candidates:
                temp["gu"] = temp[gu_candidates[0]]
        numeric_cols = [col for col in temp.columns if col.endswith("_krw")]
        if not numeric_cols or "gu" not in temp.columns:
            continue
        agg_row = temp.groupby("gu")[numeric_cols].median().reset_index()
        agg_row["year"] = year
        rows.append(agg_row)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["gu", "year"])


def _aggregate_yearly_rent_for_area_range(
    yearly_rent: dict[int, pd.DataFrame],
    min_area_pyeong: int,
    max_area_pyeong: int,
) -> pd.DataFrame:
    rows = []
    for year, df in yearly_rent.items():
        temp = df.copy()
        temp["year"] = year
        if "gu" not in temp.columns:
            gu_candidates = [col for col in temp.columns if "구" in str(col)]
            if gu_candidates:
                temp["gu"] = temp[gu_candidates[0]]
        if "면적구간" in temp.columns:
            temp["면적구간"] = pd.to_numeric(temp["면적구간"], errors="coerce")
            in_range = temp["면적구간"].between(min_area_pyeong, max_area_pyeong, inclusive="both")
            filtered = temp.loc[in_range].copy()
            if not filtered.empty:
                temp = filtered
            else:
                center = (min_area_pyeong + max_area_pyeong) / 2
                temp["area_gap"] = (temp["면적구간"] - center).abs()
                temp = temp.sort_values(["gu", "area_gap"]).drop_duplicates("gu")
        numeric_cols = [col for col in temp.columns if col.endswith("_krw")]
        if not numeric_cols or "gu" not in temp.columns:
            continue
        agg_row = temp.groupby("gu")[numeric_cols].median().reset_index()
        agg_row["year"] = year
        rows.append(agg_row)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["gu", "year"])


def _aggregate_infra(bundle: dict[str, object]) -> pd.DataFrame:
    infra = bundle["infra"][["gu", "hospital_count", "park_count"]].copy()
    infra["hospital_count"] = pd.to_numeric(infra["hospital_count"], errors="coerce")
    infra["park_count"] = pd.to_numeric(infra["park_count"], errors="coerce")
    infra = infra.groupby("gu", dropna=False)[["hospital_count", "park_count"]].mean().reset_index()
    hospital = bundle["hospital"].groupby("gu", dropna=False).size().rename("hospital_points").reset_index()
    park_stats = bundle["park_stats"].copy()
    park_stats["park_count_detail"] = pd.to_numeric(park_stats.get("park_count"), errors="coerce")
    park_stats = park_stats.groupby("gu", dropna=False)["park_count_detail"].mean().reset_index()
    dist = bundle["distribution"].groupby("gu", dropna=False).size().rename("retail_license_count").reset_index()
    return infra.merge(hospital, on="gu", how="outer").merge(park_stats, on="gu", how="outer").merge(dist, on="gu", how="outer")


def _aggregate_safety(bundle: dict[str, object]) -> pd.DataFrame:
    crime = bundle["crime"].copy()
    for column in crime.columns:
        if column != "gu":
            crime[column] = pd.to_numeric(crime[column], errors="coerce")
    crime_numeric = [col for col in crime.select_dtypes(include=["number"]).columns if col != "gu"]
    crime_grouped = crime.groupby("gu")[crime_numeric].mean(numeric_only=True).reset_index() if crime_numeric and "gu" in crime.columns else pd.DataFrame({"gu": SEOUL_GUS})
    crime_grouped["crime_score_proxy"] = crime_grouped.select_dtypes(include=["number"]).sum(axis=1) if not crime_grouped.empty else 0

    police = bundle["police"].copy()
    for column in police.columns:
        if column != "gu":
            police[column] = pd.to_numeric(police[column], errors="coerce")
    police_numeric = [col for col in police.select_dtypes(include=["number"]).columns if col != "gu"]
    police_grouped = police.groupby("gu")[police_numeric].mean(numeric_only=True).reset_index() if police_numeric and "gu" in police.columns else pd.DataFrame({"gu": SEOUL_GUS})
    police_grouped["police_satisfaction_score"] = police_grouped.select_dtypes(include=["number"]).mean(axis=1) if not police_grouped.empty else 0
    base = pd.DataFrame({"gu": SEOUL_GUS})
    return base.merge(crime_grouped, on="gu", how="left").merge(police_grouped, on="gu", how="left")


def _aggregate_redevelopment(bundle: dict[str, object]) -> pd.DataFrame:
    frame = bundle["redevelopment"].copy()
    stage_col = "사업추진단계" if "사업추진단계" in frame.columns else next((col for col in frame.columns if "단계" in str(col)), None)
    if "gu" not in frame.columns or frame["gu"].notna().sum() == 0:
        return pd.DataFrame({"gu": SEOUL_GUS, "redevelopment_count": 0, "active_stage_count": 0})
    base = frame.groupby("gu").size().rename("redevelopment_count").reset_index()
    if stage_col and stage_col in frame.columns:
        active = frame.groupby("gu")[stage_col].apply(lambda s: s.notna().sum()).rename("active_stage_count").reset_index()
        return pd.DataFrame({"gu": SEOUL_GUS}).merge(base, on="gu", how="left").merge(active, on="gu", how="left").fillna(0)
    base["active_stage_count"] = 0
    return pd.DataFrame({"gu": SEOUL_GUS}).merge(base, on="gu", how="left").fillna(0)


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0)
    mask = numeric_values.notna() & numeric_weights.gt(0)
    if not mask.any():
        return float(numeric_values.dropna().median()) if numeric_values.notna().any() else np.nan
    return float(np.average(numeric_values.loc[mask], weights=numeric_weights.loc[mask]))


def _build_feature_table_from_compact(
    bundle: dict[str, object],
    year: int,
    budget_cap: int,
    monthly_budget_cap: int,
    min_area_pyeong: int,
    max_area_pyeong: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    housing = bundle["compact_feature_base"].copy()
    district_metrics = bundle["compact_district_metrics"].copy()
    housing["year"] = pd.to_numeric(housing["year"], errors="coerce")
    housing["area_pyeong_bucket"] = pd.to_numeric(housing["area_pyeong_bucket"], errors="coerce")
    filtered = housing.loc[
        housing["year"].eq(year)
        & housing["area_pyeong_bucket"].between(min_area_pyeong, max_area_pyeong, inclusive="both")
    ].copy()
    if filtered.empty:
        fallback = housing.loc[housing["year"].eq(year)].copy()
        if fallback.empty:
            fallback = housing.copy()
        fallback["range_gap"] = (
            (fallback["area_pyeong_bucket"] - min_area_pyeong).abs()
            + (fallback["area_pyeong_bucket"] - max_area_pyeong).abs()
        )
        filtered = fallback.sort_values(["gu", "range_gap"]).groupby("gu", as_index=False).head(3).copy()

    def _summarize_group(group: pd.DataFrame) -> pd.Series:
        rent_weight = group["rent_txn_count"].fillna(0)
        sale_weight = group["sale_txn_count"].fillna(0)
        return pd.Series(
            {
                "deposit_price_krw": _weighted_average(group["deposit_price_krw"], rent_weight),
                "monthly_rent_krw": _weighted_average(group["monthly_rent_krw"], rent_weight),
                "monthly_rent_active_krw": _weighted_average(group["monthly_rent_active_krw"], rent_weight),
                "monthly_rent_positive_ratio": _weighted_average(group["monthly_rent_positive_ratio"], rent_weight),
                "rent_area_m2": _weighted_average(group["rent_area_m2"], rent_weight),
                "rent_build_year": _weighted_average(group["rent_build_year"], rent_weight),
                "rent_txn_count": rent_weight.sum(),
                "sale_price_krw": _weighted_average(group["sale_price_krw"], sale_weight),
                "sale_area_m2": _weighted_average(group["sale_area_m2"], sale_weight),
                "sale_build_year": _weighted_average(group["sale_build_year"], sale_weight),
                "sale_txn_count": sale_weight.sum(),
            }
        )

    housing_summary = filtered.groupby("gu", dropna=False).apply(_summarize_group, include_groups=False).reset_index()
    feature = pd.DataFrame({"gu": SEOUL_GUS}).merge(housing_summary, on="gu", how="left").merge(
        district_metrics,
        on="gu",
        how="left",
    )
    feature["year"] = year
    feature["budget_fit"] = ((feature["deposit_price_krw"].fillna(feature["deposit_price_krw"].median()) <= budget_cap)).astype(int)
    feature["monthly_budget_fit"] = (
        feature["monthly_rent_active_krw"].fillna(feature["monthly_rent_active_krw"].median()).fillna(0).le(monthly_budget_cap).astype(int)
    )
    feature["housing_budget_fit"] = ((feature["budget_fit"] + feature["monthly_budget_fit"]) >= 2).astype(int)
    feature["price_burden_index"] = feature["deposit_price_krw"].fillna(feature["deposit_price_krw"].median()) / max(budget_cap, 1)
    feature["monthly_burden_index"] = (
        feature["monthly_rent_active_krw"].fillna(feature["monthly_rent_active_krw"].median()).fillna(0) / max(monthly_budget_cap, 1)
    )
    feature["infra_score_raw"] = (
        feature["hospital_count"].fillna(0)
        + feature["park_count"].fillna(0)
        + feature["hospital_points"].fillna(0) / 10
        + feature["retail_license_count"].fillna(0) / 10
    )
    feature["safety_score_raw"] = feature["police_satisfaction_score"].fillna(0) - feature["crime_score_proxy"].fillna(0) / 10
    feature["redevelopment_score_raw"] = feature["redevelopment_count"].fillna(0) + feature["active_stage_count"].fillna(0) / 5
    feature["sale_rent_gap_krw"] = feature["sale_price_krw"] - feature["deposit_price_krw"]
    feature["age_proxy"] = year - feature["rent_build_year"].fillna(feature["sale_build_year"]).fillna(year)
    feature["selected_area_min_pyeong"] = min_area_pyeong
    feature["selected_area_max_pyeong"] = max_area_pyeong
    feature["selected_area_min_m2"] = round(min_area_pyeong * 3.3058, 1)
    feature["selected_area_max_m2"] = round(max_area_pyeong * 3.3058, 1)
    feature = feature.sort_values(["housing_budget_fit", "infra_score_raw"], ascending=[False, False]).reset_index(drop=True)
    return feature, {"memory_mb": round(feature.memory_usage(deep=True).sum() / 1024 / 1024, 2), "data_mode": "compact"}


@st.cache_data(ttl=FEATURE_CACHE_TTL, show_spinner=False)
def build_feature_table(
    bundle: dict[str, object],
    year: int,
    sampling_rate: float,
    budget_cap: int,
    remove_outliers: bool,
    monthly_budget_cap: int = 2_000_000,
    min_area_pyeong: int = 18,
    max_area_pyeong: int = 22,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if bundle.get("is_compact"):
        return _build_feature_table_from_compact(
            bundle=bundle,
            year=year,
            budget_cap=budget_cap,
            monthly_budget_cap=monthly_budget_cap,
            min_area_pyeong=min_area_pyeong,
            max_area_pyeong=max_area_pyeong,
        )

    rent = bundle["rent"].copy()
    sale = bundle["sale"].copy()
    if sampling_rate < 1.0:
        rent = rent.sample(frac=sampling_rate, random_state=42)
        sale = sale.sample(frac=sampling_rate, random_state=42)

    if remove_outliers:
        rent = remove_iqr_outliers(rent, ["보증금_만원_krw", "월세_만원_krw", "전용면적_m2"])
        sale = remove_iqr_outliers(sale, ["dealAmount_krw", "excluUseAr"])

    rent = _filter_by_area_range(rent, "전용면적_m2", min_area_pyeong, max_area_pyeong)
    sale = _filter_by_area_range(sale, "excluUseAr", min_area_pyeong, max_area_pyeong)

    rent_agg = _aggregate_rent(rent)
    sale_agg = _aggregate_sale(sale)
    infra_agg = _aggregate_infra(bundle)
    safety_agg = _aggregate_safety(bundle)
    redevelopment_agg = _aggregate_redevelopment(bundle)
    yearly_avg = _aggregate_yearly_rent_for_area_range(bundle["yearly_rent"], min_area_pyeong, max_area_pyeong)

    feature = (
        pd.DataFrame({"gu": SEOUL_GUS})
        .merge(rent_agg.query("year == @year"), on="gu", how="left")
        .merge(sale_agg.query("year == @year"), on=["gu", "year"], how="left")
        .merge(yearly_avg.query("year == @year"), on=["gu", "year"], how="left")
        .merge(infra_agg, on="gu", how="left")
        .merge(safety_agg, on="gu", how="left")
        .merge(redevelopment_agg, on="gu", how="left")
    )
    feature["year"] = feature["year"].fillna(year).astype(int)
    feature["budget_fit"] = ((feature["deposit_price_krw"].fillna(feature["deposit_price_krw"].median()) <= budget_cap)).astype(int)
    feature["monthly_budget_fit"] = (
        feature["monthly_rent_active_krw"]
        .fillna(feature["monthly_rent_active_krw"].median())
        .fillna(0)
        .le(monthly_budget_cap)
        .astype(int)
    )
    feature["housing_budget_fit"] = ((feature["budget_fit"] + feature["monthly_budget_fit"]) >= 2).astype(int)
    feature["price_burden_index"] = feature["deposit_price_krw"].fillna(feature["deposit_price_krw"].median()) / max(budget_cap, 1)
    feature["monthly_burden_index"] = (
        feature["monthly_rent_active_krw"].fillna(feature["monthly_rent_active_krw"].median()).fillna(0) / max(monthly_budget_cap, 1)
    )
    feature["infra_score_raw"] = (
        feature["hospital_count"].fillna(0)
        + feature["park_count"].fillna(0)
        + feature["hospital_points"].fillna(0) / 10
        + feature["retail_license_count"].fillna(0) / 10
    )
    feature["safety_score_raw"] = feature["police_satisfaction_score"].fillna(0) - feature["crime_score_proxy"].fillna(0) / 10
    feature["redevelopment_score_raw"] = feature["redevelopment_count"].fillna(0) + feature["active_stage_count"].fillna(0) / 5
    feature["sale_rent_gap_krw"] = feature["sale_price_krw"] - feature["deposit_price_krw"]
    feature["age_proxy"] = year - feature["rent_build_year"].fillna(feature["sale_build_year"]).fillna(year)
    feature["selected_area_min_pyeong"] = min_area_pyeong
    feature["selected_area_max_pyeong"] = max_area_pyeong
    feature["selected_area_min_m2"] = round(min_area_pyeong * 3.3058, 1)
    feature["selected_area_max_m2"] = round(max_area_pyeong * 3.3058, 1)
    feature = feature.sort_values(["housing_budget_fit", "infra_score_raw"], ascending=[False, False]).reset_index(drop=True)
    return feature, {"memory_mb": round(feature.memory_usage(deep=True).sum() / 1024 / 1024, 2)}
