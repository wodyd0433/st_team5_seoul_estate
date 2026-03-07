from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
APP_DIR = ROOT_DIR / "streamlit"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.config import DEPLOY_DATA_DIR
from src.feature_engineering import _aggregate_infra, _aggregate_redevelopment, _aggregate_safety
from src.io_utils import load_dataset_bundle


def _prepare_rent_compact(rent: pd.DataFrame) -> pd.DataFrame:
    frame = rent.copy()
    frame["year"] = frame["?꾩썡"].astype(str).str[:4].astype(int)
    frame["area_pyeong_bucket"] = (pd.to_numeric(frame["?꾩슜硫댁쟻_m2"], errors="coerce") / 3.3058).round().clip(10, 45)
    grouped = (
        frame.groupby(["gu", "year", "area_pyeong_bucket"], dropna=False)
        .agg(
            deposit_price_krw=("蹂댁쬆湲?留뚯썝_krw", "median"),
            monthly_rent_krw=("?붿꽭_留뚯썝_krw", "median"),
            rent_area_m2=("?꾩슜硫댁쟻_m2", "median"),
            rent_build_year=("嫄댁텞?꾨룄", lambda s: pd.to_numeric(s, errors="coerce").median()),
            rent_txn_count=("gu", "size"),
        )
        .reset_index()
    )

    positive = frame.loc[frame["?붿꽭_留뚯썝_krw"].fillna(0) > 0].copy()
    positive_grouped = (
        positive.groupby(["gu", "year", "area_pyeong_bucket"], dropna=False)
        .agg(
            monthly_rent_active_krw=("?붿꽭_留뚯썝_krw", "median"),
            monthly_rent_positive_ratio=("?붿꽭_留뚯썝_krw", lambda s: (s.fillna(0) > 0).mean()),
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
        raise RuntimeError("?꾩옱 寃쎈웾 ?곗씠??紐⑤뱶?낅땲?? ?먮낯 datasets/raw ???덈뒗 ?섍꼍?먯꽌 ?ㅽ뻾?댁빞 ?⑸땲??")

    output_dir.mkdir(parents=True, exist_ok=True)

    rent_compact = _prepare_rent_compact(bundle["rent"])
    sale_compact = _prepare_sale_compact(bundle["sale"])
    district_metrics = (
        pd.DataFrame({"gu": sorted(rent_compact["gu"].dropna().unique())})
        .merge(_aggregate_infra(bundle), on="gu", how="left")
        .merge(_aggregate_safety(bundle), on="gu", how="left")
        .merge(_aggregate_redevelopment(bundle), on="gu", how="left")
    )
    housing = rent_compact.merge(sale_compact, on=["gu", "year", "area_pyeong_bucket"], how="outer")
    housing = housing.sort_values(["year", "gu", "area_pyeong_bucket"]).reset_index(drop=True)
    persona_profiles = bundle["persona_profiles"].copy()

    housing.to_csv(output_dir / "compact_housing.csv", index=False, encoding="utf-8-sig")
    district_metrics.to_csv(output_dir / "compact_district_metrics.csv", index=False, encoding="utf-8-sig")
    bundle["commute_models"].to_csv(output_dir / "commute_models.csv", index=False, encoding="utf-8-sig")
    persona_profiles.to_csv(output_dir / "persona_profiles.csv", index=False, encoding="utf-8-sig")

    print("Deploy data generated:")
    print(output_dir / "compact_housing.csv")
    print(output_dir / "compact_district_metrics.csv")
    print(output_dir / "commute_models.csv")
    print(output_dir / "persona_profiles.csv")


if __name__ == "__main__":
    build_deploy_data(DEPLOY_DATA_DIR)
