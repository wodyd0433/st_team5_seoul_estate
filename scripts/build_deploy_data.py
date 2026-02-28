from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DEPLOY_DATA_DIR
from src.feature_engineering import _aggregate_infra, _aggregate_redevelopment, _aggregate_safety
from src.io_utils import load_dataset_bundle


def _prepare_rent_compact(rent: pd.DataFrame) -> pd.DataFrame:
    frame = rent.copy()
    frame["year"] = frame["년월"].astype(str).str[:4].astype(int)
    frame["area_pyeong_bucket"] = (pd.to_numeric(frame["전용면적_m2"], errors="coerce") / 3.3058).round().clip(10, 45)
    grouped = (
        frame.groupby(["gu", "year", "area_pyeong_bucket"], dropna=False)
        .agg(
            deposit_price_krw=("보증금_만원_krw", "median"),
            monthly_rent_krw=("월세_만원_krw", "median"),
            rent_area_m2=("전용면적_m2", "median"),
            rent_build_year=("건축년도", lambda s: pd.to_numeric(s, errors="coerce").median()),
            rent_txn_count=("gu", "size"),
        )
        .reset_index()
    )
    positive = frame.loc[frame["월세_만원_krw"].fillna(0) > 0].copy()
    positive_grouped = (
        positive.groupby(["gu", "year", "area_pyeong_bucket"], dropna=False)
        .agg(
            monthly_rent_active_krw=("월세_만원_krw", "median"),
            monthly_rent_positive_ratio=("월세_만원_krw", lambda s: (s.fillna(0) > 0).mean()),
        )
        .reset_index()
    )
    return grouped.merge(positive_grouped, on=["gu", "year", "area_pyeong_bucket"], how="left")


def _prepare_sale_compact(sale: pd.DataFrame) -> pd.DataFrame:
    frame = sale.copy()
    frame["area_pyeong_bucket"] = (pd.to_numeric(frame["excluUseAr"], errors="coerce") / 3.3058).round().clip(10, 45)
    frame["sale_price_krw"] = pd.to_numeric(frame["dealAmount_krw"], errors="coerce")
    return (
        frame.groupby(["gu", "dealYear", "area_pyeong_bucket"], dropna=False)
        .agg(
            sale_price_krw=("sale_price_krw", "median"),
            sale_area_m2=("excluUseAr", "median"),
            sale_build_year=("buildYear", lambda s: pd.to_numeric(s, errors="coerce").median()),
            sale_txn_count=("gu", "size"),
        )
        .reset_index()
        .rename(columns={"dealYear": "year"})
    )


def build_deploy_data(output_dir: Path) -> None:
    bundle = load_dataset_bundle()
    if bundle.get("is_compact"):
        raise RuntimeError("이미 경량 데이터 모드입니다. 원본 data_all 이 있는 환경에서 실행하세요.")

    output_dir.mkdir(parents=True, exist_ok=True)

    rent_compact = _prepare_rent_compact(bundle["rent"])
    sale_compact = _prepare_sale_compact(bundle["sale"])
    district_metrics = (
        pd.DataFrame({"gu": rent_compact["gu"].dropna().unique()})
        .merge(_aggregate_infra(bundle), on="gu", how="left")
        .merge(_aggregate_safety(bundle), on="gu", how="left")
        .merge(_aggregate_redevelopment(bundle), on="gu", how="left")
    )
    housing = rent_compact.merge(sale_compact, on=["gu", "year", "area_pyeong_bucket"], how="outer")
    housing = housing.sort_values(["year", "gu", "area_pyeong_bucket"]).reset_index(drop=True)

    housing.to_csv(output_dir / "compact_housing.csv", index=False, encoding="utf-8-sig")
    district_metrics.to_csv(output_dir / "compact_district_metrics.csv", index=False, encoding="utf-8-sig")
    bundle["commute_models"].to_csv(output_dir / "commute_models.csv", index=False, encoding="utf-8-sig")

    print("Deploy data generated:")
    print(output_dir / "compact_housing.csv")
    print(output_dir / "compact_district_metrics.csv")
    print(output_dir / "commute_models.csv")


if __name__ == "__main__":
    build_deploy_data(DEPLOY_DATA_DIR)
