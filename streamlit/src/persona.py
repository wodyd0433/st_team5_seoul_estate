from __future__ import annotations

import math

import numpy as np
import pandas as pd


PERSONA_WEIGHT_PRESETS = {
    "high_income_low_debt": {"budget": 0.20, "commute": 0.35, "safety": 0.15, "infra": 0.30},
    "high_income_mid_debt": {"budget": 0.25, "commute": 0.30, "safety": 0.20, "infra": 0.25},
    "high_income_high_debt": {"budget": 0.30, "commute": 0.28, "safety": 0.22, "infra": 0.20},
    "mid_income_low_debt": {"budget": 0.30, "commute": 0.25, "safety": 0.20, "infra": 0.25},
    "mid_income_mid_debt": {"budget": 0.35, "commute": 0.25, "safety": 0.20, "infra": 0.20},
    "mid_income_high_debt": {"budget": 0.40, "commute": 0.20, "safety": 0.20, "infra": 0.20},
    "low_income_low_debt": {"budget": 0.42, "commute": 0.20, "safety": 0.18, "infra": 0.20},
    "low_income_mid_debt": {"budget": 0.47, "commute": 0.18, "safety": 0.20, "infra": 0.15},
    "low_income_high_debt": {"budget": 0.52, "commute": 0.15, "safety": 0.20, "infra": 0.13},
}


def _income_midpoint_annual_krw(label: str) -> float:
    mapping = {
        "1000만원미만": 8_000_000,
        "1000~3000만원 미만": 20_000_000,
        "3000~5000만원 미만": 40_000_000,
        "5000~7000만원 미만": 60_000_000,
        "7000~10000만원 미만": 85_000_000,
        "10000만원 이상": 120_000_000,
        "1천만원 미만": 8_000_000,
        "1천만원~3천만원 미만": 20_000_000,
        "3천만원~5천만원 미만": 40_000_000,
        "5천만원~7천만원 미만": 60_000_000,
        "7천만원~1억원 미만": 85_000_000,
        "1억원 이상": 120_000_000,
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


def _debt_label_midpoint_krw(label: str) -> float:
    mapping = {
        "대출잔액 없음": 0.0,
        "1천만원 미만": 5_000_000,
        "1천만원~3천만원 미만": 20_000_000,
        "3천만원~5천만원 미만": 40_000_000,
        "5천만원~7천만원 미만": 60_000_000,
        "7천만원~1억원 미만": 85_000_000,
        "1억원~2억원 미만": 150_000_000,
        "2억원~3억원 미만": 250_000_000,
        "3억원 이상": 350_000_000,
    }
    return mapping.get(str(label).strip(), math.nan)


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


def _normalize_income_debt_distribution(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["PRD_DE"] = pd.to_numeric(frame["PRD_DE"], errors="coerce")
    frame["DT"] = pd.to_numeric(frame["DT"], errors="coerce")
    frame["C1"] = pd.to_numeric(frame["C1"], errors="coerce")
    frame["C2"] = pd.to_numeric(frame["C2"], errors="coerce")
    latest_year = int(frame["PRD_DE"].max())
    return frame.loc[frame["PRD_DE"].eq(latest_year)].copy()


def _weighted_percentile_label(frame: pd.DataFrame, value_col: str, percentile: float, label_col: str) -> tuple[float, str | None]:
    valid = frame[value_col].notna() & frame["DT"].notna() & frame["DT"].gt(0)
    if not valid.any():
        return math.nan, None

    sorted_frame = frame.loc[valid, [value_col, "DT", label_col]].sort_values(value_col).reset_index(drop=True)
    cumulative = sorted_frame["DT"].cumsum() / sorted_frame["DT"].sum()
    idx = int(np.searchsorted(cumulative.to_numpy(), percentile, side="left"))
    idx = min(idx, len(sorted_frame) - 1)
    row = sorted_frame.loc[idx]
    return float(row[value_col]), row[label_col]


def build_income_percentile_reference(income_debt_df: pd.DataFrame, group_label: str = "전국") -> dict[str, float | str]:
    del group_label
    frame = _normalize_income_debt_distribution(income_debt_df)
    income_rows = frame.loc[frame["C2"].eq(0) & frame["C1"].gt(0)].copy()
    income_rows["income_midpoint_annual_krw"] = income_rows["C1_NM"].map(_income_midpoint_annual_krw)

    p25, p25_label = _weighted_percentile_label(income_rows, "income_midpoint_annual_krw", 0.25, "C1_NM")
    p50, p50_label = _weighted_percentile_label(income_rows, "income_midpoint_annual_krw", 0.50, "C1_NM")
    p75, p75_label = _weighted_percentile_label(income_rows, "income_midpoint_annual_krw", 0.75, "C1_NM")

    return {
        "group_label": "전국",
        "latest_year": int(frame["PRD_DE"].max()),
        "p25_annual_krw": p25,
        "p50_annual_krw": p50,
        "p75_annual_krw": p75,
        "p25_monthly_krw": p25 / 12 if pd.notna(p25) else math.nan,
        "p50_monthly_krw": p50 / 12 if pd.notna(p50) else math.nan,
        "p75_monthly_krw": p75 / 12 if pd.notna(p75) else math.nan,
        "p25_income_label": p25_label,
        "p50_income_label": p50_label,
        "p75_income_label": p75_label,
    }


def build_debt_percentile_reference(income_debt_df: pd.DataFrame, income_label: str) -> dict[str, float | str]:
    frame = _normalize_income_debt_distribution(income_debt_df)
    debt_rows = frame.loc[frame["C1_NM"].eq(income_label) & frame["C2"].gt(0)].copy()
    debt_rows["debt_midpoint_krw"] = debt_rows["C2_NM"].map(_debt_label_midpoint_krw)

    p25, p25_label = _weighted_percentile_label(debt_rows, "debt_midpoint_krw", 0.25, "C2_NM")
    p50, p50_label = _weighted_percentile_label(debt_rows, "debt_midpoint_krw", 0.50, "C2_NM")
    p75, p75_label = _weighted_percentile_label(debt_rows, "debt_midpoint_krw", 0.75, "C2_NM")

    return {
        "latest_year": int(frame["PRD_DE"].max()),
        "income_label": income_label,
        "p25_krw": p25,
        "p50_krw": p50,
        "p75_krw": p75,
        "p25_label": p25_label,
        "p50_label": p50_label,
        "p75_label": p75_label,
    }


def _build_budget_caps(monthly_income: float, income_key: str, debt_key: str) -> tuple[int, int]:
    deposit_months = {"low_income": 110, "mid_income": 105, "high_income": 120}[income_key]
    debt_adjustment = {"low_debt": 1.00, "mid_debt": 0.90, "high_debt": 0.80}[debt_key]
    monthly_ratio = {"low_income": 0.18, "mid_income": 0.20, "high_income": 0.22}[income_key]
    rent_adjustment = {"low_debt": 1.00, "mid_debt": 0.92, "high_debt": 0.84}[debt_key]

    deposit_budget = int(min(max(monthly_income * deposit_months * debt_adjustment, 250_000_000), 1_200_000_000))
    monthly_budget = int(min(max(monthly_income * monthly_ratio * rent_adjustment, 450_000), 1_800_000))
    return deposit_budget, monthly_budget


def _build_saving_profile(monthly_income: float, debt_balance: float, income_label: str, debt_label: str) -> tuple[float, float, float]:
    saving_rate = {
        "저소득": {"저부채": 0.14, "중간부채": 0.10, "고부채": 0.07},
        "중간소득": {"저부채": 0.20, "중간부채": 0.16, "고부채": 0.12},
        "고소득": {"저부채": 0.26, "중간부채": 0.22, "고부채": 0.18},
    }[income_label][debt_label]
    living_cost_ratio = {"저소득": 0.62, "중간소득": 0.58, "고소득": 0.52}[income_label]
    seed_factor = {"저소득": 1.8, "중간소득": 2.4, "고소득": 3.2}[income_label]

    monthly_debt_service = debt_balance * 0.0045
    monthly_living_cost = monthly_income * living_cost_ratio
    monthly_saving = max(monthly_income * saving_rate - monthly_debt_service, monthly_income * 0.04)
    current_seed = monthly_income * seed_factor
    return monthly_debt_service, monthly_living_cost, max(monthly_saving, 0.0), current_seed


def build_persona_profiles(income_debt_df: pd.DataFrame) -> pd.DataFrame:
    income_reference = build_income_percentile_reference(income_debt_df)
    income_specs = [
        ("high_income", "고소득", 75, income_reference["p75_annual_krw"], income_reference["p75_income_label"]),
        ("mid_income", "중간소득", 50, income_reference["p50_annual_krw"], income_reference["p50_income_label"]),
        ("low_income", "저소득", 25, income_reference["p25_annual_krw"], income_reference["p25_income_label"]),
    ]
    debt_specs = [
        ("high_debt", "고부채", 75, "p75_krw", "p75_label"),
        ("mid_debt", "중간부채", 50, "p50_krw", "p50_label"),
        ("low_debt", "저부채", 25, "p25_krw", "p25_label"),
    ]

    rows: list[dict[str, object]] = []
    latest_year = int(income_reference["latest_year"])
    for income_key, income_label, income_percentile, annual_income, income_source_label in income_specs:
        monthly_income = float(annual_income) / 12 if pd.notna(annual_income) else math.nan
        debt_reference = build_debt_percentile_reference(income_debt_df, str(income_source_label))

        for debt_key, debt_label, debt_percentile, debt_value_key, debt_label_key in debt_specs:
            debt_balance = float(debt_reference[debt_value_key])
            monthly_debt_service, monthly_living_cost, monthly_saving, current_seed = _build_saving_profile(
                monthly_income,
                debt_balance,
                income_label,
                debt_label,
            )
            seed_money_2y = current_seed + monthly_saving * 24
            seed_money_3y = current_seed + monthly_saving * 36
            additional_loan_capacity = max(monthly_income * 36 - debt_balance * 0.35, 0)
            buying_power_2y = seed_money_2y + additional_loan_capacity
            buying_power_3y = seed_money_3y + additional_loan_capacity
            deposit_budget, monthly_budget = _build_budget_caps(monthly_income, income_key, debt_key)
            weights = PERSONA_WEIGHT_PRESETS[f"{income_key}_{debt_key}"]

            rows.append(
                {
                    "persona_key": f"{income_key}_{debt_key}",
                    "persona_name": f"{income_label} {debt_label}",
                    "persona_summary": (
                        f"{latest_year}년 기준 소득 P{income_percentile}와 해당 소득구간 내 금융권 대출 P{debt_percentile}를 반영한 유형"
                    ),
                    "income_basis": "전국 신혼부부 소득 분포",
                    "income_top_band": income_source_label,
                    "income_segment_label": income_label,
                    "income_percentile": income_percentile,
                    "income_source_label": income_source_label,
                    "monthly_income_estimate_krw": round(monthly_income),
                    "annual_income_estimate_krw": round(annual_income),
                    "debt_basis": f"{income_source_label} 소득구간 내 금융권 대출 분포",
                    "debt_segment_label": debt_label,
                    "debt_percentile": debt_percentile,
                    "debt_source_label": debt_reference[debt_label_key],
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
                    "income_p25_annual_krw": income_reference["p25_annual_krw"],
                    "income_p50_annual_krw": income_reference["p50_annual_krw"],
                    "income_p75_annual_krw": income_reference["p75_annual_krw"],
                    "income_p25_label": income_reference["p25_income_label"],
                    "income_p50_label": income_reference["p50_income_label"],
                    "income_p75_label": income_reference["p75_income_label"],
                    "debt_p25_krw": debt_reference["p25_krw"],
                    "debt_p50_krw": debt_reference["p50_krw"],
                    "debt_p75_krw": debt_reference["p75_krw"],
                    "debt_p25_label": debt_reference["p25_label"],
                    "debt_p50_label": debt_reference["p50_label"],
                    "debt_p75_label": debt_reference["p75_label"],
                    "reference_year": latest_year,
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

def build_persona_simulation(
    feature_table: pd.DataFrame,
    persona_row: pd.Series,
    cash_assets_krw: float,
    saving_ratio_pct: float,
) -> pd.DataFrame:
    frame = feature_table.copy()
    monthly_income = float(persona_row.get("monthly_income_estimate_krw", 0) or 0)
    current_total_debt_krw = float(persona_row.get("debt_balance_estimate_krw", 0) or 0)
    annual_savings_krw = max(monthly_income * (float(saving_ratio_pct) / 100.0) * 12, 0.0)

    frame["cash_assets_krw"] = float(cash_assets_krw)
    frame["current_total_debt_krw"] = current_total_debt_krw
    frame["annual_savings_krw"] = annual_savings_krw
    frame["available_loan_krw"] = frame["sale_price_krw"].fillna(0) * 0.70
    frame["available_funds_krw"] = frame["cash_assets_krw"] + frame["available_loan_krw"] + frame["annual_savings_krw"]
    frame["purchase_gap_krw"] = frame["sale_price_krw"].fillna(float("inf")) - frame["available_funds_krw"]
    frame["expected_total_debt_krw"] = frame["current_total_debt_krw"] + frame["available_loan_krw"]
    frame["monthly_interest_cost_krw"] = frame["expected_total_debt_krw"] * 0.04 / 12
    frame["interest_burden_rate_pct"] = np.where(
        monthly_income > 0,
        frame["monthly_interest_cost_krw"] / monthly_income * 100,
        np.nan,
    )
    return frame.sort_values(
        ["purchase_gap_krw", "interest_burden_rate_pct", "sale_price_krw"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
