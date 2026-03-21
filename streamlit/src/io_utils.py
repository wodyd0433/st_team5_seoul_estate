from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    APT_RENT_FILE_GLOB,
    APT_SALE_FILE_GLOB,
    COMPACT_DATA_PATHS,
    COMMUTE_ZONE_PATHS,
    DATASET_PATHS,
    DATA_DIR,
    DATA_DIR_CANDIDATES,
    DEPLOY_DATA_DIR,
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


def _read_csv_with_custom_encodings(path: Path, encodings: list[str], **kwargs) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, **kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"{path.name} 로딩 실패: {last_error}")
def _read_csv_parts(directory: Path, pattern: str, **kwargs) -> pd.DataFrame:
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"Missing file pattern: {directory / pattern}")
    frames = [_read_csv_with_fallback(path, **kwargs) for path in paths]
    return pd.concat(frames, ignore_index=True)




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
        "다음 위치 중 하나에 `datasets/raw` 폴더를 두거나 `DATA_DIR` 환경변수를 설정해야 합니다.\n"
        f"{search_roots}"
    )


def compact_data_available() -> bool:
    return all(path.exists() for path in COMPACT_DATA_PATHS.values())


def _load_compact_bundle() -> dict[str, object]:
    housing = _read_csv_with_fallback(COMPACT_DATA_PATHS["housing"])
    district_metrics = _read_csv_with_fallback(COMPACT_DATA_PATHS["district_metrics"])
    commute_models = _read_csv_with_fallback(COMPACT_DATA_PATHS["commute_models"])
    return {
        "compact_feature_base": housing,
        "compact_district_metrics": district_metrics,
        "commute_models": commute_models,
        "raw_frames": {
            "compact_housing": housing,
            "compact_district_metrics": district_metrics,
            "commute_models": commute_models,
        },
        "unit_report": pd.DataFrame(),
        "is_compact": True,
        "data_mode": "compact",
    }


def _load_redevelopment(path: Path) -> pd.DataFrame:
    try:
        df = _read_csv_with_fallback(path, header=2)
    except Exception:
        df = _read_csv_with_fallback(path, header=1)
    df = df.dropna(how="all")
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _load_police(path: Path) -> pd.DataFrame:
    return _read_csv_with_fallback(path)


def _load_hospital_frame(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(path) as conn:
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn)
        hospitals = pd.read_sql_query("SELECT * FROM hospitals", conn)
    hospitals.columns = [str(col).strip() for col in hospitals.columns]
    return hospitals, tables


def _load_debt_newlyweds(path: Path) -> pd.DataFrame:
    return _read_csv_with_custom_encodings(path, ["cp949", "euc-kr", "utf-8-sig"], header=None)


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


def _build_park_stats(parks: pd.DataFrame) -> pd.DataFrame:
    frame = parks.copy()
    frame = frame.loc[frame.get("rgn").notna()].copy()
    frame = add_standard_gu(frame, ["rgn", "park_addr"])
    stats = frame.groupby("gu", dropna=False).size().rename("park_count").reset_index()
    return stats


def _build_infra_summary(hospital: pd.DataFrame, park_stats: pd.DataFrame, mart: pd.DataFrame) -> pd.DataFrame:
    hospital_count = hospital.groupby("gu", dropna=False).size().rename("hospital_count").reset_index()
    park_count = park_stats.rename(columns={"park_count": "park_count"})
    mart_count = mart.groupby("gu", dropna=False).size().rename("mart_count").reset_index()
    return (
        hospital_count.merge(park_count, on="gu", how="outer")
        .merge(mart_count, on="gu", how="outer")
        .fillna(0)
    )


def _reshape_mart(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame = add_standard_gu(frame, ["도로명주소", "지번주소"])
    if "상세영업상태명" in frame.columns:
        frame = frame.loc[frame["상세영업상태명"].astype(str).str.contains("정상영업", na=False)].copy()
    elif "영업상태명" in frame.columns:
        frame = frame.loc[frame["영업상태명"].astype(str).str.contains("영업/정상", na=False)].copy()
    return frame


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


def _load_commute_average_frames() -> pd.DataFrame:
    source_specs = [
        ("광화문역", DATASET_PATHS["commute_avg_gwanghwamun"], "광화문"),
        ("강남역", DATASET_PATHS["commute_avg_gangnam"], "역삼동"),
        ("여의도역", DATASET_PATHS["commute_avg_yeouido"], "여의도동"),
        ("성수역", DATASET_PATHS["commute_avg_seongsu_1"], "성수동"),
        ("성수역", DATASET_PATHS["commute_avg_seongsu_2"], "성수동"),
    ]

    frames: list[pd.DataFrame] = []
    for hub_name, path, destination_name in source_specs:
        frame = _read_csv_with_fallback(path)
        frame["hub_name"] = hub_name
        frame["destination_name"] = destination_name
        frame["gu"] = frame["stgSggNm"].astype("string")
        frame["tzon"] = pd.to_numeric(frame["tzon"], errors="coerce")
        frame["quater"] = pd.to_numeric(frame["quater"], errors="coerce")
        frame["useStf"] = pd.to_numeric(frame["useStf"], errors="coerce")
        frame["useTm"] = pd.to_numeric(frame["useTm"], errors="coerce")
        frame["time_order"] = frame["tzon"] * 100 + frame["quater"]
        frame["time_label"] = frame["tzon"].fillna(0).astype(int).map(lambda value: f"{value:02d}") + ":" + frame["quater"].fillna(0).astype(int).map(lambda value: f"{value:02d}")
        frame["avg_minutes"] = frame["useTm"] / 60.0
        frames.append(frame[["hub_name", "destination_name", "gu", "time_order", "time_label", "useStf", "useTm", "avg_minutes"]])

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _remove_commute_outliers(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    bounds = (
        frame.groupby(["hub_name", "gu"], as_index=False)
        .agg(
            q1=("avg_minutes", lambda s: s.quantile(0.25)),
            q3=("avg_minutes", lambda s: s.quantile(0.75)),
        )
    )
    bounds["iqr"] = bounds["q3"] - bounds["q1"]
    bounds["lower_bound"] = bounds["q1"] - 1.5 * bounds["iqr"]
    bounds["upper_bound"] = bounds["q3"] + 1.5 * bounds["iqr"]
    merged = frame.merge(
        bounds[["hub_name", "gu", "lower_bound", "upper_bound"]],
        on=["hub_name", "gu"],
        how="left",
    )
    return merged.loc[
        merged["avg_minutes"].ge(merged["lower_bound"]) & merged["avg_minutes"].le(merged["upper_bound"])
    ].copy()


def _build_commute_average_bundle() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = _load_commute_average_frames()
    filtered = _remove_commute_outliers(raw)
    filtered["weighted_use_tm"] = pd.to_numeric(filtered["useTm"], errors="coerce") * pd.to_numeric(filtered["useStf"], errors="coerce")
    timeseries = (
        filtered.groupby(["hub_name", "destination_name", "gu", "time_order", "time_label"], as_index=False)
        .agg(
            avg_minutes=("avg_minutes", "mean"),
            use_stf_sum=("useStf", "sum"),
        )
        .sort_values(["hub_name", "gu", "time_order"])
        .reset_index(drop=True)
    )
    weighted = (
        filtered.groupby(["hub_name", "gu"], as_index=False)
        .agg(
            weighted_time_sum=("weighted_use_tm", "sum"),
            traffic_sum=("useStf", "sum"),
        )
    )
    weighted["avg_commute_minutes"] = weighted["weighted_time_sum"] / weighted["traffic_sum"].replace(0, np.nan) / 60.0
    return timeseries, weighted[["hub_name", "gu", "avg_commute_minutes"]]


@st.cache_data(ttl=RAW_CACHE_TTL, show_spinner=False)
def load_dataset_bundle() -> dict[str, object]:
    missing_files = get_missing_dataset_report()
    if missing_files:
        if compact_data_available():
            return _load_compact_bundle()
        missing_text = "\n".join(f"- {row['dataset']}: `{row['expected_path']}`" for row in missing_files[:12])
        if len(missing_files) > 12:
            missing_text += f"\n- 외 {len(missing_files) - 12}개"
        compact_text = "\n".join(f"- `{path}`" for path in COMPACT_DATA_PATHS.values())
        raise RuntimeError(
            f"{build_data_setup_message()}\n\n"
            f"누락 파일:\n{missing_text}\n\n"
            f"배포용 경량 데이터가 있다면 아래 파일을 `{DEPLOY_DATA_DIR}` 에 두면 됩니다.\n{compact_text}"
        )

    rent = _read_csv_parts(
        DATA_DIR,
        APT_RENT_FILE_GLOB,
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
    sale = _read_csv_parts(
        DATA_DIR,
        APT_SALE_FILE_GLOB,
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
    parks = _read_csv_with_fallback(DATASET_PATHS["seoul_parks"])
    mart = _reshape_mart(_read_csv_with_fallback(DATASET_PATHS["seoul_mart"]))
    hospital, tables = _load_hospital_frame(DATASET_PATHS["hospital_db"])
    crime = _reshape_crime(_read_csv_with_fallback(DATASET_PATHS["crime"]))
    police = _reshape_police(_load_police(DATASET_PATHS["police"]))
    redevelopment = _reshape_redevelopment(_load_redevelopment(DATASET_PATHS["redevelopment"]))
    commute_zone_frames = _load_commute_zone_frames()
    commute_models = _fit_commute_models(commute_zone_frames)
    commute_timeseries, commute_weighted_avg = _build_commute_average_bundle()

    yearly_rent_all = _read_csv_with_fallback(DATASET_PATHS["rent_avg_all"])
    year_column = next(
        (col for col in yearly_rent_all.columns if "연도" in str(col) or "year" in str(col).lower()),
        yearly_rent_all.columns[1],
    )
    yearly_rent = {}
    for year in [2021, 2022, 2023, 2024, 2025]:
        yearly_rent[year] = yearly_rent_all.loc[
            pd.to_numeric(yearly_rent_all[year_column], errors="coerce").eq(year)
        ].copy()

    rent = add_standard_gu(rent, ["구"])
    sale = add_standard_gu(sale, ["region_name", "estateAgentSggNm"])
    parks = add_standard_gu(parks, ["rgn", "park_addr"])
    mart = add_standard_gu(mart, ["도로명주소", "지번주소"])
    hospital = add_standard_gu(hospital, ["sgguCdNm", "addr"])
    if "gu" not in crime.columns or crime["gu"].isna().all():
        crime = add_standard_gu(crime, list(crime.columns))
    if "gu" not in police.columns or police["gu"].isna().all():
        police = add_standard_gu(police, list(police.columns))
    if "gu" not in redevelopment.columns or redevelopment["gu"].isna().all():
        redevelopment = add_standard_gu(redevelopment, ["자치구", "위치1", "위치2", "구역명"])

    park_stats = _build_park_stats(parks)
    infra = _build_infra_summary(hospital, park_stats, mart)

    rent, rent_unit_report = standardize_price_columns(rent, ["보증금_만원", "월세_만원"])
    sale, sale_unit_report = standardize_price_columns(sale, ["dealAmount"])
    year_unit_reports: list[dict[str, object]] = []
    for year, frame in yearly_rent.items():
        yearly_rent[year], unit_report = standardize_price_columns(frame, candidate_price_columns(frame).values())
        year_unit_reports.extend([{**row, "dataset": f"rent_avg_{year}"} for row in unit_report])

    return {
        "rent": rent,
        "sale": sale,
        "infra": infra,
        "parks": parks,
        "park_stats": park_stats,
        "mart": mart,
        "hospital": hospital,
        "crime": crime,
        "police": police,
        "redevelopment": redevelopment,
        "yearly_rent": yearly_rent,
        "raw_frames": {
            "apt_deal": sale,
            "apt_rent": rent,
            "infra_summary": infra,
            "seoul_parks": parks,
            "seoul_mart": mart,
            "park_stats": park_stats,
            "hospital_db": hospital,
            "crime": crime,
            "police": police,
            "redevelopment": redevelopment,
            **{f"commute_{k}": v for k, v in commute_zone_frames.items()},
            "commute_timeseries": commute_timeseries,
            "commute_weighted_avg": commute_weighted_avg,
        },
        "unit_report": pd.DataFrame(
            [{"dataset": "apt_rent", **row} for row in rent_unit_report]
            + [{"dataset": "apt_deal", **row} for row in sale_unit_report]
            + year_unit_reports
        ),
        "is_compact": False,
        "data_mode": "raw",
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
