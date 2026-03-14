from __future__ import annotations

import math

import numpy as np
import pandas as pd


PERSONA_WEIGHT_PRESETS = {
    "high_income_low_debt_dual": {"budget": 0.20, "commute": 0.35, "safety": 0.15, "infra": 0.30},
    "high_income_high_debt_dual": {"budget": 0.30, "commute": 0.30, "safety": 0.15, "infra": 0.25},
    "mid_income_low_debt_growth": {"budget": 0.30, "commute": 0.25, "safety": 0.15, "infra": 0.30},
    "mid_income_high_debt_defense": {"budget": 0.40, "commute": 0.20, "safety": 0.20, "infra": 0.20},
    "low_income_low_debt_stable": {"budget": 0.45, "commute": 0.20, "safety": 0.15, "infra": 0.20},
    "low_income_high_debt_risk": {"budget": 0.50, "commute": 0.15, "safety": 0.20, "infra": 0.15},
}


def _income_midpoint_annual_krw(label: str) -> float:
    mapping = {
        "1000만원미만": 8_000_000,
        "1000~3000만원 미만": 20_000_000,
        "3000~5000만원 미만": 40_000_000,
        "5000~7000만원 미만": 60_000_000,
        "7000~10000만원 미만": 85_000_000,
        "10000만원 이상": 120_000_000,
    }
    return mapping.get(str(label).strip(), math.nan)


def _debt_band_midpoint_krw(column_name: str) -> float:
    mapping = {
        "debt_lt_10m_pct": 5_000_000,
        "debt_10m_30m_pct": 20_000_000,
        "debt_30m_50m_pct": 40_000_000,
        "debt_50m_70m_pct": 60_000_000,
        "debt_70m_100m_pct": 85_000_000,
        "debt_100m_200m_pct": 150_000_000,
        "debt_200m_300m_pct": 250_000_000,
        "debt_ge_300m_pct": 350_000_000,
    }
    return mapping.get(column_name, 0.0)


def normalize_debt_newlyweds(df: pd.DataFrame) -> pd.DataFrame:
    raw = df.copy()
    header_top = raw.iloc[1].fillna("").tolist()
    header_bottom = raw.iloc[2].fillna("").tolist()
    rows = raw.iloc[3:].copy()
    rows.columns = [
        "feature_major",
        "feature_minor",
        "debt_holding_ratio_pct",
        "debt_band_total_pct",
        "debt_lt_10m_pct",
        "debt_10m_30m_pct",
        "debt_30m_50m_pct",
        "debt_50m_70m_pct",
        "debt_70m_100m_pct",
        "debt_100m_200m_pct",
        "debt_200m_300m_pct",
        "debt_ge_300m_pct",
        "debt_median_manwon",
    ]
    rows = rows.loc[~rows["feature_major"].astype(str).str.contains("신혼부부 특성별", na=False)].copy()
    numeric_columns = [col for col in rows.columns if col not in {"feature_major", "feature_minor"}]
    for column in numeric_columns:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    band_columns = [
        "debt_lt_10m_pct",
        "debt_10m_30m_pct",
        "debt_30m_50m_pct",
        "debt_50m_70m_pct",
        "debt_70m_100m_pct",
        "debt_100m_200m_pct",
        "debt_200m_300m_pct",
        "debt_ge_300m_pct",
    ]
    weights = rows[band_columns].fillna(0)
    weighted_midpoint = sum(weights[col] * _debt_band_midpoint_krw(col) for col in band_columns) / 100
    rows["debt_weighted_midpoint_krw"] = weighted_midpoint
    rows["debt_median_krw"] = rows["debt_median_manwon"] * 10_000
    rows["debt_band_mode"] = weights.idxmax(axis=1)
    rows["debt_risk_level"] = pd.cut(
        rows["debt_median_krw"],
        bins=[-1, 50_000_000, 150_000_000, float("inf")],
        labels=["저부채", "중부채", "고부채"],
    ).astype("string")
    rows.attrs["header_top"] = header_top
    rows.attrs["header_bottom"] = header_bottom
    return rows.reset_index(drop=True)


def summarize_income_distribution(df: pd.DataFrame, group_label: str = "전국") -> dict[str, float | str]:
    latest_year = int(pd.to_numeric(df["PRD_DE"], errors="coerce").max())
    frame = df.loc[df["PRD_DE"].eq(latest_year) & df["C2_NM"].eq(group_label)].copy()
    band_rows = frame.loc[frame["C1_NM"].map(_income_midpoint_annual_krw).notna()].copy()
    band_rows["midpoint_annual_krw"] = band_rows["C1_NM"].map(_income_midpoint_annual_krw)
    band_rows["DT"] = pd.to_numeric(band_rows["DT"], errors="coerce")
    weighted_annual_income_krw = (
        (band_rows["midpoint_annual_krw"] * band_rows["DT"].fillna(0)).sum() / 100 if not band_rows.empty else math.nan
    )
    top_band = band_rows.sort_values("DT", ascending=False)["C1_NM"].iloc[0] if not band_rows.empty else None
    return {
        "group_label": group_label,
        "latest_year": latest_year,
        "weighted_annual_income_krw": weighted_annual_income_krw,
        "weighted_monthly_income_krw": weighted_annual_income_krw / 12 if pd.notna(weighted_annual_income_krw) else math.nan,
        "top_income_band": top_band,
    }


def _weighted_percentile(values: pd.Series, weights: pd.Series, percentile: float) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return math.nan

    sorted_frame = (
        pd.DataFrame({"value": values.loc[valid].astype(float), "weight": weights.loc[valid].astype(float)})
        .sort_values("value")
        .reset_index(drop=True)
    )
    cumulative = sorted_frame["weight"].cumsum() / sorted_frame["weight"].sum()
    idx = int(np.searchsorted(cumulative.to_numpy(), percentile, side="left"))
    idx = min(idx, len(sorted_frame) - 1)
    return float(sorted_frame.loc[idx, "value"])


def build_income_percentile_reference(
    income_df: pd.DataFrame,
    group_label: str = "서울특별시",
) -> dict[str, float | str]:
    latest_year = int(pd.to_numeric(income_df["PRD_DE"], errors="coerce").max())
    frame = income_df.loc[income_df["PRD_DE"].eq(latest_year) & income_df["C2_NM"].eq(group_label)].copy()
    band_rows = frame.loc[frame["C1_NM"].map(_income_midpoint_annual_krw).notna()].copy()
    band_rows["midpoint_annual_krw"] = band_rows["C1_NM"].map(_income_midpoint_annual_krw)
    band_rows["DT"] = pd.to_numeric(band_rows["DT"], errors="coerce")

    p25 = _weighted_percentile(band_rows["midpoint_annual_krw"], band_rows["DT"], 0.25)
    p50 = _weighted_percentile(band_rows["midpoint_annual_krw"], band_rows["DT"], 0.50)
    p75 = _weighted_percentile(band_rows["midpoint_annual_krw"], band_rows["DT"], 0.75)

    return {
        "group_label": group_label,
        "latest_year": latest_year,
        "p25_annual_krw": p25,
        "p50_annual_krw": p50,
        "p75_annual_krw": p75,
        "p25_monthly_krw": p25 / 12 if pd.notna(p25) else math.nan,
        "p50_monthly_krw": p50 / 12 if pd.notna(p50) else math.nan,
        "p75_monthly_krw": p75 / 12 if pd.notna(p75) else math.nan,
    }


def build_persona_profiles(income_df: pd.DataFrame, debt_df: pd.DataFrame) -> pd.DataFrame:
    debt = normalize_debt_newlyweds(debt_df)
    income_national = summarize_income_distribution(income_df, "전국")
    income_seoul = summarize_income_distribution(income_df, "서울특별시")
    base_income = income_seoul["weighted_monthly_income_krw"]
    if pd.isna(base_income):
        base_income = income_national["weighted_monthly_income_krw"]

    debt_national = debt.loc[debt["feature_major"].eq("전국")].head(1)
    if debt_national.empty:
        debt_national = debt.head(1)
    base_debt = float(debt_national["debt_median_krw"].iloc[0]) if not debt_national.empty else 150_000_000

    persona_specs = [
        ("high_income_low_debt_dual", "고소득 저부채 맞벌이", 1.35, 0.55, 0.28, 900_000_000, 1_200_000, "통근과 미래 매수 전환을 중시하는 유형"),
        ("high_income_high_debt_dual", "고소득 고부채 맞벌이", 1.30, 1.35, 0.20, 850_000_000, 1_300_000, "소득은 높지만 기존 대출 부담을 관리해야 하는 유형"),
        ("mid_income_low_debt_growth", "중간소득 저부채 성장형", 1.00, 0.70, 0.22, 700_000_000, 1_000_000, "현재 전세 거주 후 2~3년 내 매수 전환을 노리는 유형"),
        ("mid_income_high_debt_defense", "중간소득 고부채 방어형", 0.95, 1.20, 0.14, 550_000_000, 800_000, "부채 방어와 월 부담 관리가 우선인 유형"),
        ("low_income_low_debt_stable", "저소득 저부채 안정형", 0.72, 0.55, 0.12, 420_000_000, 600_000, "생활비와 통근 안정성을 우선하는 유형"),
        ("low_income_high_debt_risk", "저소득 고부채 취약형", 0.68, 1.10, 0.06, 320_000_000, 450_000, "주거비와 금융부담을 가장 보수적으로 관리해야 하는 유형"),
    ]

    rows: list[dict[str, object]] = []
    for key, name, income_multiplier, debt_multiplier, saving_rate, deposit_budget, monthly_budget, summary in persona_specs:
        monthly_income = float(base_income) * income_multiplier
        debt_balance = base_debt * debt_multiplier
        monthly_debt_service = debt_balance * 0.0045
        monthly_living_cost = monthly_income * (0.52 if "고소득" in name else 0.58 if "중간소득" in name else 0.62)
        monthly_saving = max(monthly_income * saving_rate - monthly_debt_service, monthly_income * 0.04)
        current_seed = monthly_income * (3.2 if "고소득" in name else 2.4 if "중간소득" in name else 1.8)
        seed_money_2y = current_seed + monthly_saving * 24
        seed_money_3y = current_seed + monthly_saving * 36
        additional_loan_capacity = max(monthly_income * 36 - debt_balance * 0.35, 0)
        buying_power_2y = seed_money_2y + additional_loan_capacity
        buying_power_3y = seed_money_3y + additional_loan_capacity
        weights = PERSONA_WEIGHT_PRESETS[key]
        rows.append(
            {
                "persona_key": key,
                "persona_name": name,
                "persona_summary": summary,
                "income_basis": income_seoul["group_label"] if pd.notna(income_seoul["weighted_monthly_income_krw"]) else income_national["group_label"],
                "income_top_band": income_seoul["top_income_band"] if pd.notna(income_seoul["weighted_monthly_income_krw"]) else income_national["top_income_band"],
                "monthly_income_estimate_krw": round(monthly_income),
                "debt_basis": "전국",
                "debt_balance_estimate_krw": round(debt_balance),
                "monthly_debt_service_estimate_krw": round(monthly_debt_service),
                "monthly_living_cost_estimate_krw": round(monthly_living_cost),
                "monthly_saving_estimate_krw": round(monthly_saving),
                "current_seed_estimate_krw": round(current_seed),
                "seed_money_2y_krw": round(seed_money_2y),
                "seed_money_3y_krw": round(seed_money_3y),
                "buying_power_2y_krw": round(buying_power_2y),
                "buying_power_3y_krw": round(buying_power_3y),
                "deposit_budget_cap_krw": int(deposit_budget),
                "monthly_budget_cap_krw": int(monthly_budget),
                "weight_budget": weights["budget"],
                "weight_commute": weights["commute"],
                "weight_safety": weights["safety"],
                "weight_infra": weights["infra"],
            }
        )
    return pd.DataFrame(rows)


def build_persona_simulation(feature_table: pd.DataFrame, persona_row: pd.Series) -> pd.DataFrame:
    frame = feature_table.copy()
    frame["current_rent_fit"] = frame["deposit_price_krw"].fillna(float("inf")) <= persona_row["deposit_budget_cap_krw"]
    frame["current_monthly_fit"] = frame["monthly_rent_active_krw"].fillna(float("inf")) <= persona_row["monthly_budget_cap_krw"]
    frame["buy_2y_gap_krw"] = persona_row["buying_power_2y_krw"] - frame["sale_price_krw"].fillna(float("inf"))
    frame["buy_3y_gap_krw"] = persona_row["buying_power_3y_krw"] - frame["sale_price_krw"].fillna(float("inf"))
    frame["buying_status_2y"] = pd.cut(
        frame["buy_2y_gap_krw"],
        bins=[-float("inf"), -50_000_000, 0, float("inf")],
        labels=["어려움", "도전 가능", "매수 가능"],
    ).astype("string")
    frame["buying_status_3y"] = pd.cut(
        frame["buy_3y_gap_krw"],
        bins=[-float("inf"), -50_000_000, 0, float("inf")],
        labels=["어려움", "도전 가능", "매수 가능"],
    ).astype("string")
    return frame.sort_values(["buy_3y_gap_krw", "buy_2y_gap_krw"], ascending=False).reset_index(drop=True)
