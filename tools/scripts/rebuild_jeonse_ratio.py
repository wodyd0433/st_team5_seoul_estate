from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
CLEANED_HOUSING_DIR = ROOT_DIR / "datasets" / "cleaned" / "housing"
RAW_DIR = ROOT_DIR / "datasets" / "raw"
RATIO_PATH = CLEANED_HOUSING_DIR / "jeonse_ratio_by_gu_year_area.csv"
RANKING_PATH = CLEANED_HOUSING_DIR / "jeonse_ratio_ranking_by_gu.csv"
SALE_PATH = CLEANED_HOUSING_DIR / "apartment_sale_transactions_cleaned.csv"
SUPPORTED_AREA_BANDS = ["33㎡", "59㎡", "84㎡"]


def _risk_band(value: float) -> str:
    if pd.isna(value):
        return ""
    if value < 60:
        return "안정(<60)"
    if value < 80:
        return "주의(60~80)"
    return "위험(80+)"


def _score_desc(series: pd.Series) -> pd.Series:
    minimum = series.min()
    maximum = series.max()
    if pd.isna(minimum) or pd.isna(maximum) or minimum == maximum:
        return pd.Series(50.0, index=series.index)
    return (maximum - series) / (maximum - minimum) * 100


def _assign_area_band(series: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(series, errors="coerce"),
        bins=[float("-inf"), 40, 70, 100, float("inf")],
        labels=["33㎡", "59㎡", "84㎡", "기타"],
        right=False,
    )


def _load_rent_medians() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(RAW_DIR.glob("apartment_rent_transactions_*.csv")):
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame.columns = [
            "gu_std",
            "gu_code",
            "yearmonth",
            "apt_name",
            "dong",
            "area_m2",
            "floor",
            "deposit",
            "monthly_rent",
            "contract_type",
            "contract_date",
            "build_year",
        ]
        frame["year"] = pd.to_numeric(frame["yearmonth"].astype(str).str[:4], errors="coerce").astype("Int64")
        frame["deposit"] = pd.to_numeric(frame["deposit"], errors="coerce")
        frame["monthly_rent"] = pd.to_numeric(frame["monthly_rent"], errors="coerce").fillna(0)
        frame["area_band"] = _assign_area_band(frame["area_m2"])
        frame = frame.loc[frame["monthly_rent"].eq(0) & frame["area_band"].isin(SUPPORTED_AREA_BANDS)].copy()
        frames.append(frame[["gu_std", "year", "area_band", "deposit"]])

    rent = pd.concat(frames, ignore_index=True)
    return (
        rent.groupby(["gu_std", "year", "area_band"], as_index=False, observed=True)
        .agg(jeonse_price=("deposit", "median"), jeonse_count=("deposit", "size"))
        .assign(jeonse_count=lambda df: df["jeonse_count"].astype("Int64"))
    )


def _load_sale_medians() -> pd.DataFrame:
    sale = pd.read_csv(SALE_PATH, encoding="utf-8-sig", low_memory=False)
    sale["year"] = pd.to_numeric(sale["year"], errors="coerce").astype("Int64")
    sale["area_m2"] = pd.to_numeric(sale["area_m2"], errors="coerce")
    sale["sale_price_num"] = pd.to_numeric(sale["sale_price_num"], errors="coerce")
    sale["area_band"] = _assign_area_band(sale["area_m2"])
    sale = sale.loc[sale["area_band"].isin(SUPPORTED_AREA_BANDS)].copy()
    return (
        sale.groupby(["gu_std", "year", "area_band"], as_index=False, observed=True)
        .agg(sale_price=("sale_price_num", "median"), sale_count=("sale_price_num", "size"))
        .assign(sale_count=lambda df: df["sale_count"].astype("Int64"))
    )


def build_jeonse_ratio() -> tuple[pd.DataFrame, pd.DataFrame]:
    rent_frame = _load_rent_medians()
    sale_frame = _load_sale_medians()
    ratio = (
        rent_frame.merge(sale_frame, on=["gu_std", "year", "area_band"], how="inner")
        .dropna(subset=["jeonse_price", "sale_price"])
        .sort_values(["gu_std", "year", "area_band"], ignore_index=True)
    )
    ratio["jeonse_ratio_pct"] = ratio["jeonse_price"] / ratio["sale_price"] * 100
    ratio = ratio.loc[ratio["jeonse_ratio_pct"].le(100)].copy()
    ratio["risk_band"] = ratio["jeonse_ratio_pct"].map(_risk_band)
    ratio = ratio[
        [
            "gu_std",
            "year",
            "area_band",
            "jeonse_price",
            "jeonse_count",
            "sale_price",
            "sale_count",
            "jeonse_ratio_pct",
            "risk_band",
        ]
    ]

    ranking = (
        ratio.groupby("gu_std", as_index=False)
        .agg(jeonse_ratio_pct=("jeonse_ratio_pct", "mean"), matched_rows=("jeonse_ratio_pct", "size"))
        .sort_values(["jeonse_ratio_pct", "gu_std"], ascending=[False, True], ignore_index=True)
    )
    ranking["jeonse_ratio_score"] = _score_desc(ranking["jeonse_ratio_pct"])
    ranking = ranking[["gu_std", "jeonse_ratio_pct", "matched_rows", "jeonse_ratio_score"]]

    return ratio, ranking


def main() -> None:
    ratio, ranking = build_jeonse_ratio()
    ratio.to_csv(RATIO_PATH, index=False, encoding="utf-8-sig")
    ranking.to_csv(RANKING_PATH, index=False, encoding="utf-8-sig")
    print(f"Wrote {RATIO_PATH}")
    print(f"Wrote {RANKING_PATH}")
    print(f"Rows: ratio={len(ratio)}, ranking={len(ranking)}")


if __name__ == "__main__":
    main()
