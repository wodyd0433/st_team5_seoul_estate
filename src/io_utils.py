from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    COMMUTE_ZONE_PATHS,
    DATASET_PATHS,
    DATA_DIR,
    DATA_DIR_CANDIDATES,
    ENCODING_CANDIDATES,
    POLICE_STATION_TO_GU,
    RAW_CACHE_TTL,
)
from src.gu_standardizer import add_standard_gu, gu_match_report
from src.unit_detection import candidate_price_columns, standardize_price_columns


def _read_csv_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    last_error: Exception | None = None
    for encoding in ENCODING_CANDIDATES:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, **kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"{path.name} 로딩 실패: {last_error}")


def get_missing_dataset_report() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset_name, path in DATASET_PATHS.items():
        if not path.exists():
            rows.append({"dataset": dataset_name, "expected_path": str(path)})
    for hub_name, path in COMMUTE_ZONE_PATHS.items():
        if not path.exists():
            rows.append({"dataset": f"commute_{hub_name}", "expected_path": str(path)})
    return rows


def build_data_setup_message() -> str:
    search_roots = "\n".join(f"- `{candidate}`" for candidate in DATA_DIR_CANDIDATES)
    return (
        "필수 데이터 파일을 찾을 수 없습니다.\n\n"
        f"현재 탐색 중인 데이터 폴더: `{DATA_DIR}`\n\n"
        "다음 위치 중 하나에 `data_all` 폴더를 두거나 `DATA_DIR` 환경변수를 설정해야 합니다.\n"
        f"{search_roots}"
    )


def _load_redevelopment(path: Path) -> pd.DataFrame:
    try:
        df = _read_csv_with_fallback(path, header=2)
    except Exception:
        df = _read_csv_with_fallback(path, header=1)
    df = df.dropna(how="all")
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _load_police(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    for encoding in ["cp949", "euc-kr", "utf-8-sig", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception:
            continue
    raise RuntimeError(f"{path.name} 로딩 실패")


def _reshape_crime(df: pd.DataFrame) -> pd.DataFrame:
    district_columns = [col for col in df.columns if str(col).startswith("서울 ")]
    if not district_columns:
        return add_standard_gu(df, list(df.columns))
    melted = df.melt(
        id_vars=[col for col in ["범죄대분류", "범죄중분류"] if col in df.columns],
        value_vars=district_columns,
        var_name="district_name",
        value_name="crime_count",
    )
    melted = add_standard_gu(melted, ["district_name"])
    melted["crime_count"] = pd.to_numeric(melted["crime_count"], errors="coerce")
    return melted


def _reshape_police(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "응답자특성" in result.columns:
        result["gu"] = result["응답자특성"].map(POLICE_STATION_TO_GU)
    if "gu" not in result.columns or result["gu"].isna().all():
        result = add_standard_gu(result, ["응답자특성"] + list(result.columns))
    return result


def _reshape_redevelopment(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    rename_map = {}
    if "Unnamed: 1" in result.columns:
        rename_map["Unnamed: 1"] = "자치구"
    if "Unnamed: 2" in result.columns:
        rename_map["Unnamed: 2"] = "구역명"
    if "Unnamed: 3" in result.columns:
        rename_map["Unnamed: 3"] = "위치1"
    if "Unnamed: 4" in result.columns:
        rename_map["Unnamed: 4"] = "위치2"
    if "Unnamed: 7" in result.columns:
        rename_map["Unnamed: 7"] = "사업유형"
    if "Unnamed: 8" in result.columns:
        rename_map["Unnamed: 8"] = "사업추진단계"
    result = result.rename(columns=rename_map)
    result = result.dropna(how="all")
    result = result.loc[~(result.get("자치구").isna() & result.get("구역명").isna())].copy()
    result = add_standard_gu(result, ["자치구", "위치1", "위치2", "구역명"])
    return result


def _load_commute_zone_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for hub_name, path in COMMUTE_ZONE_PATHS.items():
        frames[hub_name] = _read_csv_with_fallback(path)
    return frames


def _fit_commute_models(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for hub_name, df in frames.items():
        frame = df.copy()
        frame["이동거리(km)"] = pd.to_numeric(frame["이동거리(km)"], errors="coerce")
        frame["소요시간(분)"] = pd.to_numeric(frame["소요시간(분)"], errors="coerce")
        frame["환승구간"] = pd.to_numeric(frame["환승구간"], errors="coerce").fillna(0)
        train = frame.dropna(subset=["이동거리(km)", "소요시간(분)"]).copy()
        x = np.column_stack(
            [
                np.ones(len(train)),
                train["이동거리(km)"].to_numpy(),
                train["환승구간"].to_numpy(),
            ]
        )
        y = train["소요시간(분)"].to_numpy()
        coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
        rows.append(
            {
                "hub_name": hub_name,
                "intercept": float(coeffs[0]),
                "distance_coef": float(coeffs[1]),
                "transfer_coef": float(coeffs[2]),
                "avg_transfer": float(train["환승구간"].median()),
                "avg_fare": float(pd.to_numeric(train["기본운임(원)"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=RAW_CACHE_TTL, show_spinner=False)
def load_dataset_bundle() -> dict[str, object]:
    missing_files = get_missing_dataset_report()
    if missing_files:
        missing_text = "\n".join(f"- {row['dataset']}: `{row['expected_path']}`" for row in missing_files[:12])
        if len(missing_files) > 12:
            missing_text += f"\n- 외 {len(missing_files) - 12}개"
        raise RuntimeError(f"{build_data_setup_message()}\n\n누락 파일:\n{missing_text}")

    rent = _read_csv_with_fallback(
        DATASET_PATHS["apt_rent"],
        usecols=["구", "구코드", "년월", "전용면적_m2", "보증금_만원", "월세_만원", "계약일", "건축년도"],
        dtype={
            "구": "string",
            "구코드": "string",
            "년월": "string",
            "전용면적_m2": "float32",
            "보증금_만원": "string",
            "월세_만원": "string",
            "계약일": "string",
            "건축년도": "string",
        },
    )
    sale = _read_csv_with_fallback(
        DATASET_PATHS["apt_deal"],
        usecols=["dealAmount", "dealYear", "dealMonth", "excluUseAr", "buildYear", "estateAgentSggNm", "region_name"],
        dtype={
            "dealAmount": "string",
            "dealYear": "Int64",
            "dealMonth": "Int64",
            "excluUseAr": "float32",
            "buildYear": "string",
            "estateAgentSggNm": "string",
            "region_name": "string",
        },
    )
    infra = _read_csv_with_fallback(DATASET_PATHS["infra_summary"])
    dist = _read_csv_with_fallback(DATASET_PATHS["distribution_license"])
    parks = _read_csv_with_fallback(DATASET_PATHS["seoul_parks"])
    park_stats = _read_csv_with_fallback(DATASET_PATHS["seoul_parks_stats"])
    hospital = _read_csv_with_fallback(DATASET_PATHS["hospital_data"])
    crime = _reshape_crime(_read_csv_with_fallback(DATASET_PATHS["crime"]))
    police = _reshape_police(_load_police(DATASET_PATHS["police"]))
    redevelopment = _reshape_redevelopment(_load_redevelopment(DATASET_PATHS["redevelopment"]))
    commute_zone_frames = _load_commute_zone_frames()
    commute_models = _fit_commute_models(commute_zone_frames)

    yearly_rent = {}
    for year in [2021, 2022, 2023, 2024, 2025]:
        yearly_rent[year] = _read_csv_with_fallback(DATASET_PATHS[f"rent_avg_{year}"])

    rent = add_standard_gu(rent, ["구"])
    sale = add_standard_gu(sale, ["region_name", "estateAgentSggNm"])
    infra = add_standard_gu(infra, ["rgn"])
    dist = add_standard_gu(dist, ["gu", "자치구", "시군구", "site_addr", "road_addr"])
    parks = add_standard_gu(parks, ["gu", "자치구", "park_addr"])
    park_stats = add_standard_gu(park_stats, ["rgn"])
    hospital = add_standard_gu(hospital, ["sgguCdNm", "addr"])
    if "gu" not in crime.columns or crime["gu"].isna().all():
        crime = add_standard_gu(crime, list(crime.columns))
    if "gu" not in police.columns or police["gu"].isna().all():
        police = add_standard_gu(police, list(police.columns))
    if "gu" not in redevelopment.columns or redevelopment["gu"].isna().all():
        redevelopment = add_standard_gu(redevelopment, ["자치구", "위치1", "위치2", "구역명"])

    rent, rent_unit_report = standardize_price_columns(rent, ["보증금_만원", "월세_만원"])
    sale, sale_unit_report = standardize_price_columns(sale, ["dealAmount"])
    year_unit_reports: list[dict[str, object]] = []
    for year, frame in yearly_rent.items():
        yearly_rent[year], unit_report = standardize_price_columns(frame, candidate_price_columns(frame).values())
        year_unit_reports.extend([{**row, "dataset": f"rent_avg_{year}"} for row in unit_report])

    with sqlite3.connect(DATASET_PATHS["hospital_db"]) as conn:
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn)

    return {
        "rent": rent,
        "sale": sale,
        "infra": infra,
        "distribution": dist,
        "parks": parks,
        "park_stats": park_stats,
        "hospital": hospital,
        "crime": crime,
        "police": police,
        "redevelopment": redevelopment,
        "yearly_rent": yearly_rent,
        "hospital_db_tables": tables,
        "commute_zone_frames": commute_zone_frames,
        "commute_models": commute_models,
        "raw_frames": {
            "apt_deal": sale,
            "apt_rent": rent,
            "infra_summary": infra,
            "distribution_license": dist,
            "seoul_parks": parks,
            "seoul_parks_stats": park_stats,
            "hospital_data": hospital,
            "crime": crime,
            "police": police,
            "redevelopment": redevelopment,
            **{f"commute_{k}": v for k, v in commute_zone_frames.items()},
        },
        "unit_report": pd.DataFrame(
            [{"dataset": "apt_rent", **row} for row in rent_unit_report]
            + [{"dataset": "apt_deal", **row} for row in sale_unit_report]
            + year_unit_reports
        ),
    }


def collect_data_quality_report(bundle: dict[str, object]) -> dict[str, pd.DataFrame]:
    quality_rows = []
    match_rows = []
    for name, df in bundle["raw_frames"].items():
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        quality_rows.append(
            {
                "dataset": name,
                "rows": len(df),
                "cols": df.shape[1],
                "memory_mb": round(memory_mb, 2),
                "missing_pct": round(df.isna().mean().mean() * 100, 2),
            }
        )
        match_rows.append(gu_match_report(df, name))
    return {
        "dataset_quality": pd.DataFrame(quality_rows).sort_values("memory_mb", ascending=False),
        "gu_match_report": pd.DataFrame(match_rows).sort_values("match_rate_pct"),
        "unit_detection_report": bundle["unit_report"],
    }
