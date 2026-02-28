from __future__ import annotations

import re
import sqlite3
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 설정 / 경로 / 옵션
# ============================================================

BASE_DIR = Path(r"C:\Users\wodyd\OneDrive\PythonWorkspace\icb7\korea_estate\gpt_analysis")
DATA_DIR = Path(r"C:\Users\wodyd\OneDrive\PythonWorkspace\icb7\korea_estate\data_all")
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"
CLEAN_DIR = OUTPUT_DIR / "cleaned"

CSV_ENCODINGS = ["utf-8", "utf-8-sig", "cp949", "euc-kr"]
AREA_TARGETS = [33, 59, 84]
AREA_TOLERANCE = 3

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

FILE_CANDIDATES = {
    "rent_raw": ["서울_아파트_전월세_실거래_5개년_2021_2026.csv", "seoul_apt_rent_5y.csv"],
    "sale_raw": ["apt_deal_total.csv"],
    "rent_avg_2021": ["자치구별_아파트_전월세_평균2021.csv"],
    "rent_avg_2022": ["자치구별_아파트_전월세_평균2022.csv"],
    "rent_avg_2023": ["자치구별_아파트_전월세_평균2023.csv"],
    "rent_avg_2024": ["자치구별_아파트_전월세_평균2024.csv"],
    "rent_avg_2025": ["자치구별_아파트_전월세_평균2025.csv"],
    "crime": ["crime_2024.csv"],
    "police": ["police_satisfaction_2025.csv"],
    "hospital_csv": ["hospital_data.csv"],
    "distribution": ["distribution_license.csv"],
    "parks": ["seoul_parks.csv"],
    "parks_stats": ["seoul_parks_stats.csv"],
    "infra_summary": ["seoul_infra_summary.csv"],
    "redevelopment": ["25.12기준.서울시정비사업추진현황.csv"],
    "hospital_db": ["hospital.db"],
}

COLUMN_CANDIDATES = {
    "gu": ["자치구", "구", "gu", "district", "region_name", "SIG_KOR_NM", "rgn", "sgguCdNm", "estateAgentSggNm", "소재지구", "자치구명"],
    "address": ["주소", "addr", "site_addr", "road_addr", "도로명주소", "지번주소", "park_addr"],
    "date": ["계약일", "date", "deal_date", "license_date", "last_modified_ts", "update_dt", "rgstDate"],
    "ym": ["년월", "dealYear", "dealMonth"],
    "deposit": ["보증금", "보증금_만원", "전세금", "평균 전세", "평균 월세보증금"],
    "monthly_rent": ["월세", "월세_만원", "평균 월세"],
    "sale_price": ["거래금액", "dealAmount", "매매가", "실거래가"],
    "area": ["전용면적_m2", "전용면적", "면적", "excluUseAr", "area", "site_area", "facility_total_scale", "면적구간"],
    "apt_name": ["아파트명", "aptNm", "단지명"],
    "dong": ["법정동", "umdNm", "동"],
    "hospital_name": ["yadmNm", "기관명", "병원명", "business_nm", "park_nm"],
    "hospital_type": ["clCdNm", "종별", "종별명", "business_type_nm", "sanitary_business_type_nm"],
    "lat": ["yaxsWgs84Cordnt", "coord_y", "ycrd", "위도"],
    "lon": ["xaxsWgs84Cordnt", "coord_x", "xcrd", "경도"],
}


def log(message: str) -> None:
    print(f"[INFO] {message}")


def log_skip(section: str, reason: str, candidates: Optional[Iterable[str]] = None, alternative: Optional[str] = None) -> None:
    print(f"[SKIP] {section}: {reason}")
    if candidates:
        print(f"       candidate columns: {list(candidates)}")
    if alternative:
        print(f"       alternative suggestion: {alternative}")


def ensure_dirs() -> None:
    for path in [BASE_DIR, OUTPUT_DIR, PLOTS_DIR, TABLES_DIR, CLEAN_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def resolve_file(key: str) -> Optional[Path]:
    for candidate in FILE_CANDIDATES.get(key, []):
        path = DATA_DIR / candidate
        if path.exists():
            return path
    return None


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in tup if str(x) != ""]).strip("_") for tup in df.columns]
    return df


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_money(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True).str.strip()
    s = s.replace({"": np.nan, "-": np.nan, ".": np.nan})
    return pd.to_numeric(s, errors="coerce")


def clean_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True).str.strip()
    s = s.replace({"": np.nan, "-": np.nan, ".": np.nan})
    return pd.to_numeric(s, errors="coerce")


def infer_column(df: pd.DataFrame, semantic_key: str) -> Optional[str]:
    candidates = COLUMN_CANDIDATES.get(semantic_key, [])
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for col in df.columns:
        col_lower = str(col).lower()
        for candidate in candidates:
            cand_lower = candidate.lower()
            if cand_lower in col_lower or col_lower in cand_lower:
                return col
    return None


def infer_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping = {key: infer_column(df, key) for key in COLUMN_CANDIDATES}
    print(f"[COLUMNS] {list(df.columns)}")
    print(f"[MAPPING] {mapping}")
    return mapping


def robust_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    last_error = None
    for enc in CSV_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
            df = flatten_columns(df)
            df.attrs["source_encoding"] = enc
            return df
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"CSV load failed for {path.name}: {last_error}")


def load_csv(path: Path, name: Optional[str] = None) -> pd.DataFrame:
    df = robust_read_csv(path)
    dataset_name = name or path.stem
    log(f"loaded CSV: {dataset_name} ({path.name}) encoding={df.attrs.get('source_encoding')} shape={df.shape}")
    infer_mapping(df)
    return df


def save_csv(df: Optional[pd.DataFrame], filename: str) -> None:
    if df is not None and not df.empty:
        df.to_csv(TABLES_DIR / filename, index=False, encoding="utf-8-sig")


def save_clean_csv(df: Optional[pd.DataFrame], filename: str) -> None:
    if df is not None and not df.empty:
        df.to_csv(CLEAN_DIR / filename, index=False, encoding="utf-8-sig")


def save_plot(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_dates(df: pd.DataFrame, date_col: Optional[str] = None, year_col: Optional[str] = None, month_col: Optional[str] = None) -> pd.Series:
    if date_col and date_col in df.columns:
        s = df[date_col].astype(str).str.strip()
        parsed = pd.to_datetime(s, errors="coerce")
        if parsed.notna().any():
            return parsed
        yy_mm_dd = s.str.extract(r"(?P<yy>\d{2})[./-]?(?P<mm>\d{2})[./-]?(?P<dd>\d{2})")
        if not yy_mm_dd.empty:
            yy = yy_mm_dd["yy"].where(yy_mm_dd["yy"].isna(), "20" + yy_mm_dd["yy"].fillna(""))
            parsed = pd.to_datetime(yy + "-" + yy_mm_dd["mm"].fillna("01") + "-" + yy_mm_dd["dd"].fillna("01"), errors="coerce")
            if parsed.notna().any():
                return parsed
    if year_col and month_col and year_col in df.columns and month_col in df.columns:
        y = clean_numeric(df[year_col]).fillna(0).astype("Int64").astype(str)
        m = clean_numeric(df[month_col]).fillna(1).astype("Int64").astype(str).str.zfill(2)
        parsed = pd.to_datetime(y + "-" + m + "-01", errors="coerce")
        if parsed.notna().any():
            return parsed
    ym_col = infer_column(df, "ym")
    if ym_col:
        s = df[ym_col].astype(str).str.extract(r"(?P<y>\d{4})(?P<m>\d{2})")
        parsed = pd.to_datetime(s["y"] + "-" + s["m"] + "-01", errors="coerce")
        if parsed.notna().any():
            return parsed
    return pd.Series(pd.NaT, index=df.index)


def assign_area_band(area_value) -> str:
    if pd.isna(area_value):
        return "기타"
    try:
        area = float(area_value)
    except Exception:
        return "기타"
    nearest = min(AREA_TARGETS, key=lambda x: abs(area - x))
    return f"{nearest}㎡" if abs(area - nearest) <= AREA_TOLERANCE else "기타"


def find_gu_in_text(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"([가-힣]+구)", str(text))
    return match.group(1) if match else None


def standardize_gu(series: pd.Series) -> pd.Series:
    def _std(value) -> Optional[str]:
        text = normalize_text(value)
        if not text:
            return np.nan
        text = text.replace("서울 ", "").replace("서울특별시 ", "").replace("서울특별시", "").strip()
        match = re.search(r"([가-힣]+구)", text)
        return match.group(1) if match else np.nan
    return series.map(_std)


def minmax_scale(series: pd.Series, reverse: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    min_v, max_v = s.min(), s.max()
    if pd.isna(min_v) or pd.isna(max_v) or min_v == max_v:
        scaled = pd.Series(50.0, index=series.index)
    else:
        scaled = (s - min_v) / (max_v - min_v) * 100
    return 100 - scaled if reverse else scaled


def profile_df(df: pd.DataFrame, name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    missing_rate = (df.isna().mean() * 100).sort_values(ascending=False).head(10).reset_index()
    missing_rate.columns = ["column", "missing_rate_pct"]
    dtype_summary = df.dtypes.astype(str).reset_index()
    dtype_summary.columns = ["column", "dtype"]
    summary = pd.DataFrame(
        {
            "dataset": [name],
            "rows": [len(df)],
            "cols": [df.shape[1]],
            "duplicate_rows": [int(df.duplicated().sum())],
        }
    )
    log(f"profile {name}: rows={len(df):,}, cols={df.shape[1]}, duplicates={int(df.duplicated().sum()):,}")
    print(missing_rate)
    return summary, missing_rate.merge(dtype_summary, on="column", how="left")


def dedupe_key_candidates(df: pd.DataFrame) -> List[str]:
    keys = []
    for semantic in ["gu", "address", "date", "area", "deposit", "monthly_rent", "sale_price", "apt_name", "hospital_name"]:
        col = infer_column(df, semantic)
        if col and col not in keys:
            keys.append(col)
    return keys


def load_sqlite_tables(db_path: Path) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name", conn)
    table_data: Dict[str, pd.DataFrame] = {}
    inventory_rows: List[Dict[str, object]] = []
    schema_rows: List[Dict[str, object]] = []
    for table_name in tables["name"].tolist():
        df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
        table_data[table_name] = df
        mapping = infer_mapping(df)
        row_count = int(pd.read_sql_query(f'SELECT COUNT(*) AS cnt FROM "{table_name}"', conn)["cnt"].iloc[0])
        inventory_rows.append(
            {
                "table": table_name,
                "rows": row_count,
                "cols": df.shape[1],
                "gu_col": mapping.get("gu"),
                "address_col": mapping.get("address"),
                "lat_col": mapping.get("lat"),
                "lon_col": mapping.get("lon"),
                "name_col": mapping.get("hospital_name"),
                "type_col": mapping.get("hospital_type"),
            }
        )
        pragma = pd.read_sql_query(f'PRAGMA table_info("{table_name}")', conn)
        for _, row in pragma.iterrows():
            schema_rows.append({"table": table_name, "cid": row["cid"], "column": row["name"], "dtype": row["type"], "notnull": row["notnull"], "pk": row["pk"]})
    conn.close()
    return table_data, pd.DataFrame(inventory_rows), pd.DataFrame(schema_rows)


def enrich_with_gu(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.copy()
    mapping = infer_mapping(df)
    gu_col = mapping.get("gu")
    addr_col = mapping.get("address")
    df["gu_std"] = standardize_gu(df[gu_col]) if gu_col else np.nan
    if df["gu_std"].isna().all() and addr_col:
        df["gu_std"] = df[addr_col].astype(str).map(find_gu_in_text)
    if gu_col and df["gu_std"].isna().any() and addr_col:
        df["gu_std"] = df["gu_std"].fillna(df[addr_col].astype(str).map(find_gu_in_text))
    unmatched = sorted([x for x in df["gu_std"].dropna().astype(str).unique() if not x.endswith("구")])
    if unmatched:
        print(f"[WARN] {name} unmatched gu values: {unmatched}")
    return df


def preprocess_rent_df(df: pd.DataFrame) -> pd.DataFrame:
    df = enrich_with_gu(df, "rent_raw")
    mapping = infer_mapping(df)
    deposit_col, monthly_col, area_col = mapping.get("deposit"), mapping.get("monthly_rent"), mapping.get("area")
    date_col, ym_col = mapping.get("date"), infer_column(df, "ym")
    df["deposit_num"] = clean_money(df[deposit_col]) if deposit_col else np.nan
    df["monthly_rent_num"] = clean_money(df[monthly_col]) if monthly_col else np.nan
    df["area_m2"] = clean_numeric(df[area_col]) if area_col else np.nan
    if date_col:
        df["contract_date"] = parse_dates(df, date_col=date_col)
    elif ym_col and ym_col in df.columns:
        df["contract_date"] = parse_dates(df)
    else:
        df["contract_date"] = parse_dates(df, year_col="dealYear", month_col="dealMonth")
    df["rent_type"] = np.where(df["monthly_rent_num"].fillna(0) > 0, "월세", np.where(df["deposit_num"].fillna(0) > 0, "전세", "미상"))
    df["year"] = df["contract_date"].dt.year
    df["month"] = df["contract_date"].dt.to_period("M").astype(str)
    df["area_band"] = df["area_m2"].apply(assign_area_band)
    df["is_invalid_amount"] = (df["deposit_num"].fillna(0) < 0) | (df["monthly_rent_num"].fillna(0) < 0)
    df["is_invalid_area"] = df["area_m2"].fillna(0) <= 0
    q1, q3 = df["deposit_num"].quantile(0.25), df["deposit_num"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 3 * iqr if pd.notna(iqr) else np.nan
    df["deposit_outlier"] = df["deposit_num"] > upper if pd.notna(upper) else False
    return df


def preprocess_sale_df(df: pd.DataFrame) -> pd.DataFrame:
    df = enrich_with_gu(df, "sale_raw")
    mapping = infer_mapping(df)
    area_col, price_col = mapping.get("area"), mapping.get("sale_price")
    df["area_m2"] = clean_numeric(df[area_col]) if area_col else np.nan
    df["sale_price_num"] = clean_money(df[price_col]) if price_col else np.nan
    if "dealYear" in df.columns and "dealMonth" in df.columns:
        df["deal_date"] = parse_dates(df, year_col="dealYear", month_col="dealMonth")
    else:
        df["deal_date"] = parse_dates(df, date_col=mapping.get("date"))
    df["year"] = df["deal_date"].dt.year
    df["area_band"] = df["area_m2"].apply(assign_area_band)
    return df


def preprocess_simple_yearly_avg(df: pd.DataFrame, year_hint: int) -> pd.DataFrame:
    df = enrich_with_gu(df, f"rent_avg_{year_hint}")
    mapping = infer_mapping(df)
    area_col = infer_column(df, "area")
    df["year"] = year_hint if "연도" not in df.columns else clean_numeric(df["연도"])
    df["avg_jeonse"] = clean_money(df[mapping["deposit"]]) if mapping.get("deposit") else np.nan
    df["avg_monthly_rent"] = clean_money(df[mapping["monthly_rent"]]) if mapping.get("monthly_rent") else np.nan
    df["tx_count"] = clean_numeric(df["거래건수"]) if "거래건수" in df.columns else np.nan
    raw_area = clean_numeric(df[area_col]) if area_col else pd.Series(np.nan, index=df.index)
    df["area_band"] = raw_area.apply(lambda x: f"{int(x)}㎡" if pd.notna(x) else "기타")
    return df


def preprocess_hospital_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = enrich_with_gu(df, name)
    mapping = infer_mapping(df)
    if mapping.get("lat"):
        df["lat_num"] = clean_numeric(df[mapping["lat"]])
    if mapping.get("lon"):
        df["lon_num"] = clean_numeric(df[mapping["lon"]])
    return df


def preprocess_distribution_df(df: pd.DataFrame) -> pd.DataFrame:
    df = enrich_with_gu(df, "distribution")
    if "business_status_nm" in df.columns:
        df = df[df["business_status_nm"].astype(str).str.contains("영업", na=False)].copy()
    return df


def preprocess_parks_df(df: pd.DataFrame) -> pd.DataFrame:
    df = enrich_with_gu(df, "parks")
    if df["gu_std"].isna().all() and "rgn" in df.columns:
        df["gu_std"] = standardize_gu(df["rgn"])
    if "area" in df.columns:
        df["park_area_num"] = clean_numeric(df["area"])
    return df


def preprocess_crime_df(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = [c for c in df.columns if c in ["범죄대분류", "범죄중분류"]]
    value_cols = [c for c in df.columns if c not in id_cols]
    long_df = df.melt(id_vars=id_cols, value_vars=value_cols, var_name="gu_raw", value_name="crime_count")
    long_df["crime_count"] = clean_numeric(long_df["crime_count"])
    long_df["gu_std"] = standardize_gu(long_df["gu_raw"])
    return long_df


def preprocess_police_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    first_col = df.columns[0]
    df["gu_std"] = df[first_col].astype(str).str.extract(r"서울([가-힣]+)경찰서")
    df["gu_std"] = df["gu_std"].fillna(df[first_col].astype(str).map(find_gu_in_text))
    if "서울중부경찰서" in df[first_col].astype(str).values:
        df.loc[df[first_col] == "서울중부경찰서", "gu_std"] = "중구"
    return df


def preprocess_redevelopment_df(df: pd.DataFrame) -> pd.DataFrame:
    df = enrich_with_gu(df, "redevelopment")
    if df["gu_std"].isna().all():
        for col in df.columns:
            extracted = df[col].astype(str).map(find_gu_in_text)
            if extracted.notna().sum() > 0:
                df["gu_std"] = extracted
                break
    return df


def eda_inventory(all_frames: Dict[str, pd.DataFrame], sqlite_inventory: Optional[pd.DataFrame], sqlite_schema: Optional[pd.DataFrame]) -> None:
    summary_rows = []
    quality_rows = []
    for name, df in all_frames.items():
        if df is None or df.empty:
            continue
        summary, quality = profile_df(df, name)
        summary["dedupe_key_candidates"] = ", ".join(dedupe_key_candidates(df))
        summary_rows.append(summary)
        quality["dataset"] = name
        quality_rows.append(quality)
    if summary_rows:
        save_csv(pd.concat(summary_rows, ignore_index=True), "data_inventory_summary.csv")
    if quality_rows:
        save_csv(pd.concat(quality_rows, ignore_index=True), "data_quality_top_missing.csv")
    if sqlite_inventory is not None and not sqlite_inventory.empty:
        save_csv(sqlite_inventory, "sqlite_inventory.csv")
    if sqlite_schema is not None and not sqlite_schema.empty:
        save_csv(sqlite_schema, "sqlite_schema.csv")


def eda_market_trend(rent_df: Optional[pd.DataFrame], yearly_avg_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if rent_df is not None and {"gu_std", "year", "rent_type", "deposit_num"}.issubset(rent_df.columns):
        tmp = rent_df.dropna(subset=["gu_std", "year"]).copy()
        tmp = tmp[tmp["rent_type"].isin(["전세", "월세"])]
        trend = tmp.groupby(["gu_std", "year", "rent_type"]).agg(avg_deposit=("deposit_num", "median"), avg_monthly_rent=("monthly_rent_num", "median"), tx_count=("deposit_num", "size")).reset_index()
    elif yearly_avg_df is not None and not yearly_avg_df.empty:
        trend = yearly_avg_df.groupby(["gu_std", "year", "area_band"]).agg(avg_jeonse=("avg_jeonse", "mean"), avg_monthly_rent=("avg_monthly_rent", "mean"), tx_count=("tx_count", "sum")).reset_index()
        save_csv(trend, "market_trend_from_yearly_avg.csv")
        return trend
    else:
        log_skip("eda_market_trend", "rent transaction dataset unavailable", COLUMN_CANDIDATES["gu"], "use yearly average files")
        return None

    save_csv(trend, "market_trend_by_gu_year_rent_type.csv")
    top_gus = trend.groupby("gu_std")["tx_count"].sum().sort_values(ascending=False).head(8).index.tolist()
    plot_df = trend[trend["gu_std"].isin(top_gus)].copy()
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        for gu in plot_df["gu_std"].dropna().unique():
            sub = plot_df[(plot_df["gu_std"] == gu) & (plot_df["rent_type"] == "전세")].sort_values("year")
            if not sub.empty:
                ax.plot(sub["year"], sub["avg_deposit"], marker="o", label=gu)
        ax.set_title("자치구별 전세 보증금 추세")
        ax.set_xlabel("연도")
        ax.set_ylabel("중앙 보증금(만원)")
        ax.legend(ncol=2, fontsize=8)
        save_plot(fig, "market_trend_jeonse.png")
    return trend


def eda_rent_ratio(rent_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if rent_df is None or rent_df.empty or "rent_type" not in rent_df.columns or "year" not in rent_df.columns:
        log_skip("eda_rent_ratio", "required columns missing", ["rent_type", "year"], "derive from monthly rent > 0")
        return None
    tmp = rent_df.dropna(subset=["year"]).copy()
    ratio = tmp[tmp["rent_type"].isin(["전세", "월세"])].groupby(["gu_std", "year", "rent_type"]).size().reset_index(name="count")
    total = ratio.groupby(["gu_std", "year"])["count"].transform("sum")
    ratio["ratio_pct"] = ratio["count"] / total * 100
    save_csv(ratio, "rent_ratio_by_gu_year.csv")
    yearly = ratio.groupby(["year", "rent_type"])["count"].sum().reset_index()
    yearly["ratio_pct"] = yearly["count"] / yearly.groupby("year")["count"].transform("sum") * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    for rent_type in yearly["rent_type"].unique():
        sub = yearly[yearly["rent_type"] == rent_type].sort_values("year")
        ax.plot(sub["year"], sub["ratio_pct"], marker="o", label=rent_type)
    ax.set_title("전세 vs 월세 비율 변화")
    ax.set_xlabel("연도")
    ax.set_ylabel("비중(%)")
    ax.legend()
    save_plot(fig, "rent_ratio_yearly.png")
    return ratio


def eda_area_band(rent_df: Optional[pd.DataFrame], yearly_avg_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if rent_df is not None and not rent_df.empty and {"area_band", "deposit_num", "gu_std"}.issubset(rent_df.columns):
        tmp = rent_df[(rent_df["area_band"] != "기타") & (rent_df["rent_type"].isin(["전세", "월세"]))].copy()
        area_stats = tmp.groupby(["gu_std", "area_band", "rent_type"]).agg(deposit_median=("deposit_num", "median"), deposit_mean=("deposit_num", "mean"), monthly_rent_median=("monthly_rent_num", "median"), count=("deposit_num", "size")).reset_index()
    elif yearly_avg_df is not None and not yearly_avg_df.empty:
        area_stats = yearly_avg_df.groupby(["gu_std", "area_band"]).agg(avg_jeonse=("avg_jeonse", "mean"), avg_monthly_rent=("avg_monthly_rent", "mean"), count=("tx_count", "sum")).reset_index()
    else:
        log_skip("eda_area_band", "area-based price data unavailable", COLUMN_CANDIDATES["area"], "use 면적구간 yearly avg csv")
        return None
    save_csv(area_stats, "area_band_price_distribution.csv")
    plot_df = area_stats.copy()
    y_col = "avg_jeonse"
    if "rent_type" in plot_df.columns:
        plot_df = plot_df[plot_df["rent_type"] == "전세"]
        y_col = "deposit_median"
    if not plot_df.empty:
        top_gus = plot_df.groupby("gu_std")[y_col].mean().sort_values(ascending=False).head(8).index
        pivot = plot_df[plot_df["gu_std"].isin(top_gus)].pivot(index="gu_std", columns="area_band", values=y_col).fillna(0)
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("자치구 TOP 전세가격 면적대 비교")
        ax.set_xlabel("자치구")
        ax.set_ylabel("가격(만원)")
        save_plot(fig, "area_band_jeonse_top_gu.png")
    return area_stats


def eda_budget_candidates(rent_df: Optional[pd.DataFrame], yearly_avg_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if rent_df is not None and not rent_df.empty:
        budget_df = rent_df[rent_df["rent_type"] == "전세"].groupby("gu_std")["deposit_num"].median().reset_index(name="jeonse_median")
    elif yearly_avg_df is not None and not yearly_avg_df.empty:
        budget_df = yearly_avg_df.groupby("gu_std")["avg_jeonse"].mean().reset_index(name="jeonse_median")
    else:
        log_skip("eda_budget_candidates", "jeonse price data unavailable", COLUMN_CANDIDATES["deposit"], "use rent avg files")
        return None
    budget_df["under_3eok"] = budget_df["jeonse_median"] <= 30000
    budget_df["under_5eok"] = budget_df["jeonse_median"] <= 50000
    budget_df["over_5eok"] = budget_df["jeonse_median"] > 50000
    save_csv(budget_df.sort_values("jeonse_median"), "budget_candidates_by_gu.csv")
    return budget_df


def build_hospital_source_options(hospital_csv_df: Optional[pd.DataFrame], hospital_db_tables: Dict[str, pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], Dict[str, pd.DataFrame]]:
    options: Dict[str, pd.DataFrame] = {}
    if hospital_csv_df is not None and not hospital_csv_df.empty:
        csv_pre = preprocess_hospital_df(hospital_csv_df, "hospital_csv")
        csv_pre["source"] = "csv"
        options["csv"] = csv_pre
    db_best = None
    for table_name, table_df in hospital_db_tables.items():
        pre = preprocess_hospital_df(table_df, f"hospital_db_{table_name}")
        options[f"db_{table_name}"] = pre
        if db_best is None or len(pre) > len(db_best):
            db_best = pre.copy()
    if db_best is not None:
        db_best["source"] = "db"
        options["db_priority"] = db_best
    if hospital_csv_df is not None and db_best is not None:
        union_df = pd.concat([options["csv"], db_best], ignore_index=True, sort=False)
        name_col = infer_column(union_df, "hospital_name") or "yadmNm"
        options["union"] = union_df.drop_duplicates(subset=["gu_std", "source", name_col], keep="first")
        csv_name = infer_column(options["csv"], "hospital_name") or "yadmNm"
        db_name = infer_column(db_best, "hospital_name") or "yadmNm"
        priority_df = pd.concat(
            [
                db_best.assign(priority=0).rename(columns={db_name: "_name"}),
                options["csv"].assign(priority=1).rename(columns={csv_name: "_name"}),
            ],
            ignore_index=True,
            sort=False,
        )
        options["priority_merge"] = priority_df.sort_values(["priority"]).drop_duplicates(subset=["gu_std", "_name"], keep="first")
    preferred = options.get("db_priority") if options.get("db_priority") is not None else options.get("csv")
    return preferred, options


def eda_infra_hospital_db_vs_csv(hospital_csv_df: Optional[pd.DataFrame], hospital_db_tables: Dict[str, pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], Dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
    preferred, options = build_hospital_source_options(hospital_csv_df, hospital_db_tables)
    comparison = None
    csv_source = options.get("csv")
    db_source = options.get("db_priority")
    if csv_source is not None and db_source is not None:
        csv_count = csv_source.groupby("gu_std").size().reset_index(name="hospital_count_csv")
        db_count = db_source.groupby("gu_std").size().reset_index(name="hospital_count_db")
        comparison = csv_count.merge(db_count, on="gu_std", how="outer")
        comparison["diff_abs"] = comparison["hospital_count_db"].fillna(0) - comparison["hospital_count_csv"].fillna(0)
        comparison["diff_pct_vs_csv"] = np.where(comparison["hospital_count_csv"].fillna(0) > 0, comparison["diff_abs"] / comparison["hospital_count_csv"] * 100, np.nan)
        save_csv(comparison.sort_values("diff_abs", ascending=False), "hospital_db_vs_csv_comparison.csv")
    if preferred is None or preferred.empty:
        log_skip("eda_infra_hospital_db_vs_csv", "hospital sources unavailable", ["hospital_data.csv", "hospital.db"], "use DB if richer")
        return None, options, comparison
    type_col = infer_column(preferred, "hospital_type")
    group_cols = ["gu_std"] + ([type_col] if type_col else [])
    hospital_stats = preferred.groupby(group_cols).size().reset_index(name="hospital_count")
    save_csv(hospital_stats, "hospital_count_by_gu_type.csv")
    if type_col:
        pivot = hospital_stats.pivot(index="gu_std", columns=type_col, values="hospital_count").fillna(0)
        fig, ax = plt.subplots(figsize=(14, 7))
        pivot.head(25).plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("자치구별 병원 종별 분포")
        ax.set_xlabel("자치구")
        ax.set_ylabel("기관 수")
        save_plot(fig, "hospital_type_distribution.png")
    for option_name, option_df in options.items():
        if option_df is not None and not option_df.empty:
            save_clean_csv(option_df, f"hospital_option_{safe_filename(option_name)}.csv")
    return hospital_stats, options, comparison


def eda_infra_total(hospital_preferred: Optional[pd.DataFrame], distribution_df: Optional[pd.DataFrame], parks_df: Optional[pd.DataFrame], parks_stats_df: Optional[pd.DataFrame], infra_summary_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    frames = []
    if hospital_preferred is not None and not hospital_preferred.empty:
        frames.append(hospital_preferred.groupby("gu_std").size().reset_index(name="hospital_count"))
    if distribution_df is not None and not distribution_df.empty:
        frames.append(distribution_df.groupby("gu_std").size().reset_index(name="distribution_count"))
    if parks_df is not None and not parks_df.empty:
        frames.append(parks_df.groupby("gu_std").agg(park_count=("gu_std", "size"), park_area_total=("park_area_num", "sum")).reset_index())
    elif parks_stats_df is not None and not parks_stats_df.empty:
        ps = enrich_with_gu(parks_stats_df, "parks_stats")
        if "park_count" in ps.columns:
            ps["park_count"] = clean_numeric(ps["park_count"])
            frames.append(ps.groupby("gu_std").agg(park_count=("park_count", "sum")).reset_index())
    if infra_summary_df is not None and not infra_summary_df.empty:
        base = enrich_with_gu(infra_summary_df, "infra_summary")
        cols = [c for c in ["hospital_count", "park_count"] if c in base.columns]
        if cols:
            frames.append(base[["gu_std"] + cols].groupby("gu_std", as_index=False).sum())
    if not frames:
        log_skip("eda_infra_total", "infra datasets unavailable", ["hospital", "distribution", "parks"], "aggregate available source counts")
        return None
    infra = None
    for frame in frames:
        infra = frame if infra is None else infra.merge(frame, on="gu_std", how="outer")
    infra = infra.groupby("gu_std", as_index=False).sum(numeric_only=True)
    numeric_cols = [c for c in infra.columns if c != "gu_std"]
    for col in numeric_cols:
        infra[f"{col}_score"] = minmax_scale(infra[col])
    score_cols = [c for c in infra.columns if c.endswith("_score")]
    infra["infra_index"] = infra[score_cols].mean(axis=1, skipna=True)
    infra = infra.sort_values("infra_index", ascending=False)
    save_csv(infra, "infra_composite_index.csv")
    fig, ax = plt.subplots(figsize=(10, 6))
    infra.head(10).plot(x="gu_std", y="infra_index", kind="bar", ax=ax, legend=False)
    ax.set_title("생활 인프라 종합지수 상위 10개 자치구")
    ax.set_xlabel("자치구")
    ax.set_ylabel("인프라 지수")
    save_plot(fig, "infra_index_top10.png")
    return infra


def eda_safety(crime_df: Optional[pd.DataFrame], police_df: Optional[pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    crime_summary = None
    safety_summary = None
    if crime_df is not None and not crime_df.empty:
        crime_summary = crime_df.groupby("gu_std")["crime_count"].sum().reset_index(name="crime_total")
        crime_summary["crime_score"] = minmax_scale(crime_summary["crime_total"], reverse=True)
        save_csv(crime_summary.sort_values("crime_total", ascending=False), "crime_summary_by_gu.csv")
        fig, ax = plt.subplots(figsize=(10, 6))
        crime_summary.sort_values("crime_total", ascending=False).head(10).plot(x="gu_std", y="crime_total", kind="bar", ax=ax, legend=False)
        ax.set_title("범죄 건수 상위 10개 자치구")
        ax.set_xlabel("자치구")
        ax.set_ylabel("범죄 건수")
        save_plot(fig, "crime_top10.png")
    if police_df is not None and not police_df.empty:
        numeric_cols = [c for c in police_df.columns if c not in [police_df.columns[0], "gu_std"]]
        tmp = police_df.copy()
        for col in numeric_cols:
            tmp[col] = clean_numeric(tmp[col])
        tmp["police_satisfaction"] = tmp[numeric_cols].mean(axis=1, skipna=True)
        safety_summary = tmp.groupby("gu_std", as_index=False)["police_satisfaction"].mean()
        safety_summary["police_score"] = minmax_scale(safety_summary["police_satisfaction"])
        save_csv(safety_summary.sort_values("police_satisfaction", ascending=False), "police_satisfaction_by_gu.csv")
    if crime_summary is not None or safety_summary is not None:
        merged = crime_summary if crime_summary is not None else None
        if safety_summary is not None:
            merged = safety_summary if merged is None else merged.merge(safety_summary, on="gu_std", how="outer")
        if merged is not None and {"crime_total", "police_satisfaction"}.issubset(merged.columns):
            merged["crime_police_corr"] = merged[["crime_total", "police_satisfaction"]].corr().iloc[0, 1]
        if merged is not None:
            save_csv(merged, "safety_merged_summary.csv")
        return crime_summary, merged
    log_skip("eda_safety", "crime or police satisfaction data insufficient", ["crime_2024.csv", "police_satisfaction_2025.csv"], "report each separately")
    return crime_summary, safety_summary


def compute_jeonse_ratio(rent_df: Optional[pd.DataFrame], sale_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if rent_df is None or sale_df is None or rent_df.empty or sale_df.empty:
        log_skip("compute_jeonse_ratio", "rent or sale source unavailable", ["deposit", "sale_price"], "join by gu/year/area_band")
        return None
    jeonse = rent_df[rent_df["rent_type"] == "전세"].dropna(subset=["gu_std", "year"]).copy()
    sale = sale_df.dropna(subset=["gu_std", "year"]).copy()
    jeonse_group = jeonse.groupby(["gu_std", "year", "area_band"]).agg(jeonse_price=("deposit_num", "median"), jeonse_count=("deposit_num", "size")).reset_index()
    sale_group = sale.groupby(["gu_std", "year", "area_band"]).agg(sale_price=("sale_price_num", "median"), sale_count=("sale_price_num", "size")).reset_index()
    merged = jeonse_group.merge(sale_group, on=["gu_std", "year", "area_band"], how="inner")
    merged = merged[(merged["sale_price"] > 0) & (merged["jeonse_price"] > 0)]
    if merged.empty:
        log_skip("compute_jeonse_ratio", "no joinable keys after gu/year/area_band merge", ["gu_std", "year", "area_band"], "consider dong-level or monthly match later")
        return None
    merged["jeonse_ratio_pct"] = merged["jeonse_price"] / merged["sale_price"] * 100
    merged["risk_band"] = pd.cut(merged["jeonse_ratio_pct"], bins=[-np.inf, 60, 80, np.inf], labels=["안정(<60)", "주의(60~80)", "위험(80+)"])
    save_csv(merged.sort_values("jeonse_ratio_pct", ascending=False), "jeonse_ratio_by_gu_year_area.csv")
    gu_ratio = merged.groupby("gu_std").agg(jeonse_ratio_pct=("jeonse_ratio_pct", "mean"), matched_rows=("jeonse_ratio_pct", "size")).reset_index().sort_values("jeonse_ratio_pct", ascending=False)
    gu_ratio["jeonse_ratio_score"] = minmax_scale(gu_ratio["jeonse_ratio_pct"], reverse=True)
    save_csv(gu_ratio, "jeonse_ratio_ranking_by_gu.csv")
    fig, ax = plt.subplots(figsize=(10, 6))
    gu_ratio.head(10).plot(x="gu_std", y="jeonse_ratio_pct", kind="bar", ax=ax, legend=False)
    ax.set_title("전세가율 상위 10개 자치구")
    ax.set_xlabel("자치구")
    ax.set_ylabel("전세가율(%)")
    save_plot(fig, "jeonse_ratio_top10.png")
    return gu_ratio


def eda_redevelopment(redev_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if redev_df is None or redev_df.empty:
        log_skip("eda_redevelopment", "redevelopment dataset unavailable", ["25.12기준.서울시정비사업추진현황.csv"])
        return None
    stage_col = None
    for col in redev_df.columns:
        if "단계" in str(col) or "추진" in str(col):
            stage_col = col
            break
    if stage_col:
        summary = redev_df.groupby(["gu_std", stage_col]).size().reset_index(name="project_count")
    else:
        summary = redev_df.groupby("gu_std").size().reset_index(name="project_count")
    save_csv(summary, "redevelopment_summary.csv")
    return summary


def compute_eda_score(budget_df: Optional[pd.DataFrame], infra_df: Optional[pd.DataFrame], safety_merged_df: Optional[pd.DataFrame], jeonse_ratio_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    frames = []
    if budget_df is not None and not budget_df.empty:
        price = budget_df[["gu_std", "jeonse_median"]].copy()
        price["price_score"] = minmax_scale(price["jeonse_median"], reverse=True)
        frames.append(price[["gu_std", "price_score"]])
    if infra_df is not None and not infra_df.empty:
        frames.append(infra_df[["gu_std", "infra_index"]].rename(columns={"infra_index": "infra_score"}))
    if safety_merged_df is not None and not safety_merged_df.empty:
        tmp = safety_merged_df.copy()
        score_candidates = [c for c in ["crime_score", "police_score"] if c in tmp.columns]
        if score_candidates:
            tmp["safety_score"] = tmp[score_candidates].mean(axis=1, skipna=True)
            frames.append(tmp[["gu_std", "safety_score"]])
    if jeonse_ratio_df is not None and not jeonse_ratio_df.empty:
        frames.append(jeonse_ratio_df[["gu_std", "jeonse_ratio_score"]])
    if not frames:
        log_skip("compute_eda_score", "no input metrics available", ["budget", "infra", "safety", "jeonse_ratio"])
        return None
    score_df = None
    for frame in frames:
        score_df = frame if score_df is None else score_df.merge(frame, on="gu_std", how="outer")
    weight_map = {"price_score": 35, "infra_score": 15, "safety_score": 10, "jeonse_ratio_score": 10}
    active_weights = {k: v for k, v in weight_map.items() if k in score_df.columns}
    total_weight = sum(active_weights.values())
    if total_weight == 0:
        return None
    score_df["eda_total_score"] = 0.0
    for col, weight in active_weights.items():
        score_df["eda_total_score"] += pd.to_numeric(score_df[col], errors="coerce").fillna(pd.to_numeric(score_df[col], errors="coerce").median()) * (weight / total_weight)
    score_df["commute_excluded_reason"] = "통근시간 데이터 부재로 점수에서 제외"
    score_df = score_df.sort_values("eda_total_score", ascending=False)
    save_csv(score_df, "eda_total_score_top_gu.csv")
    return score_df


def build_summary_report(budget_df: Optional[pd.DataFrame], jeonse_ratio_df: Optional[pd.DataFrame], infra_df: Optional[pd.DataFrame], score_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    if budget_df is not None and not budget_df.empty:
        rows.append({"category": "예산 3억 이하 가능 구 수", "value": int(budget_df["under_3eok"].sum())})
        rows.append({"category": "예산 5억 이하 가능 구 수", "value": int(budget_df["under_5eok"].sum())})
        rows.append({"category": "예산 3억 이하 후보", "value": ", ".join(budget_df.loc[budget_df["under_3eok"], "gu_std"].dropna().sort_values().tolist())})
        rows.append({"category": "예산 5억 이하 후보", "value": ", ".join(budget_df.loc[budget_df["under_5eok"], "gu_std"].dropna().sort_values().tolist())})
    if jeonse_ratio_df is not None and not jeonse_ratio_df.empty:
        rows.append({"category": "전세가율 위험 상위 5개 구", "value": ", ".join(jeonse_ratio_df.sort_values("jeonse_ratio_pct", ascending=False).head(5)["gu_std"].tolist())})
    if infra_df is not None and not infra_df.empty:
        rows.append({"category": "인프라 상위 5개 구", "value": ", ".join(infra_df.head(5)["gu_std"].tolist())})
    if score_df is not None and not score_df.empty:
        rows.append({"category": "EDA 종합점수 상위 10개 구", "value": ", ".join(score_df.head(10)["gu_std"].tolist())})
    summary_df = pd.DataFrame(rows)
    save_csv(summary_df, "summary_report.csv")
    return summary_df


def main() -> None:
    ensure_dirs()
    log(f"base_dir={BASE_DIR}")
    log(f"data_dir={DATA_DIR}")

    raw_frames: Dict[str, pd.DataFrame] = {}
    processed_frames: Dict[str, pd.DataFrame] = {}
    for key, candidates in FILE_CANDIDATES.items():
        path = resolve_file(key)
        if path is None:
            print(f"[WARN] file not found for key={key}, candidates={candidates}")
            continue
        if path.suffix.lower() == ".csv":
            try:
                raw_frames[key] = load_csv(path, key)
            except Exception as exc:
                print(f"[WARN] failed to load {path.name}: {exc}")

    sqlite_tables = {}
    sqlite_inventory = None
    sqlite_schema = None
    db_path = resolve_file("hospital_db")
    if db_path and db_path.exists():
        try:
            sqlite_tables, sqlite_inventory, sqlite_schema = load_sqlite_tables(db_path)
            log(f"loaded sqlite db tables: {list(sqlite_tables.keys())}")
        except Exception as exc:
            print(f"[WARN] failed to inspect sqlite db: {exc}")

    eda_inventory({**raw_frames, **{f"sqlite_{k}": v for k, v in sqlite_tables.items()}}, sqlite_inventory, sqlite_schema)

    rent_df = preprocess_rent_df(raw_frames["rent_raw"]) if "rent_raw" in raw_frames else None
    sale_df = preprocess_sale_df(raw_frames["sale_raw"]) if "sale_raw" in raw_frames else None
    yearly_avg_frames = [preprocess_simple_yearly_avg(raw_frames[f"rent_avg_{year}"], year) for year in [2021, 2022, 2023, 2024, 2025] if f"rent_avg_{year}" in raw_frames]
    yearly_avg_df = pd.concat(yearly_avg_frames, ignore_index=True, sort=False) if yearly_avg_frames else None
    hospital_csv_df = preprocess_hospital_df(raw_frames["hospital_csv"], "hospital_csv") if "hospital_csv" in raw_frames else None
    distribution_df = preprocess_distribution_df(raw_frames["distribution"]) if "distribution" in raw_frames else None
    parks_df = preprocess_parks_df(raw_frames["parks"]) if "parks" in raw_frames else None
    parks_stats_df = raw_frames.get("parks_stats")
    if parks_stats_df is not None and not parks_stats_df.empty and "park_count" in parks_stats_df.columns:
        parks_stats_df = parks_stats_df.copy()
        parks_stats_df["park_count"] = clean_numeric(parks_stats_df["park_count"])
    infra_summary_df = raw_frames.get("infra_summary")
    crime_df = preprocess_crime_df(raw_frames["crime"]) if "crime" in raw_frames else None
    police_df = preprocess_police_df(raw_frames["police"]) if "police" in raw_frames else None
    redevelopment_df = preprocess_redevelopment_df(raw_frames["redevelopment"]) if "redevelopment" in raw_frames else None

    processed_frames.update({"rent_clean": rent_df, "sale_clean": sale_df, "yearly_avg_clean": yearly_avg_df, "hospital_csv_clean": hospital_csv_df, "distribution_clean": distribution_df, "parks_clean": parks_df, "crime_clean": crime_df, "police_clean": police_df, "redevelopment_clean": redevelopment_df})
    for name, df in processed_frames.items():
        if df is not None and not df.empty:
            save_clean_csv(df, f"{safe_filename(name)}.csv")

    market_trend = eda_market_trend(rent_df, yearly_avg_df)
    rent_ratio = eda_rent_ratio(rent_df)
    area_band = eda_area_band(rent_df, yearly_avg_df)
    budget_df = eda_budget_candidates(rent_df, yearly_avg_df)
    hospital_stats, hospital_options, hospital_compare = eda_infra_hospital_db_vs_csv(hospital_csv_df, sqlite_tables)
    hospital_preferred = hospital_options.get("db_priority") if hospital_options.get("db_priority") is not None else hospital_options.get("csv")
    infra_df = eda_infra_total(hospital_preferred, distribution_df, parks_df, parks_stats_df, infra_summary_df)
    crime_summary, safety_merged_df = eda_safety(crime_df, police_df)
    jeonse_ratio_df = compute_jeonse_ratio(rent_df, sale_df)
    redevelopment_summary = eda_redevelopment(redevelopment_df)
    score_df = compute_eda_score(budget_df, infra_df, safety_merged_df, jeonse_ratio_df)
    summary_df = build_summary_report(budget_df, jeonse_ratio_df, infra_df, score_df)

    print("\n" + "=" * 80)
    print("요약 리포트")
    print("=" * 80)
    if budget_df is not None and not budget_df.empty:
        print("[예산 3억 이하 후보]")
        print(", ".join(budget_df.loc[budget_df["under_3eok"], "gu_std"].dropna().sort_values().tolist()) or "없음")
        print("[예산 5억 이하 후보]")
        print(", ".join(budget_df.loc[budget_df["under_5eok"], "gu_std"].dropna().sort_values().tolist()) or "없음")
    if jeonse_ratio_df is not None and not jeonse_ratio_df.empty:
        print("[전세가율 위험 상위]")
        print(jeonse_ratio_df[["gu_std", "jeonse_ratio_pct"]].head(5).to_string(index=False))
    if infra_df is not None and not infra_df.empty:
        print("[인프라 상위]")
        print(infra_df[["gu_std", "infra_index"]].head(5).to_string(index=False))
    if score_df is not None and not score_df.empty:
        print("[임시 종합점수 상위]")
        print(score_df[["gu_std", "eda_total_score"]].head(10).to_string(index=False))
    print("=" * 80)
    print(f"outputs saved to: {OUTPUT_DIR}")
    print(f"tables: {TABLES_DIR}")
    print(f"plots: {PLOTS_DIR}")
    print(f"cleaned data: {CLEAN_DIR}")
    _ = market_trend, rent_ratio, area_band, hospital_stats, hospital_compare, crime_summary, redevelopment_summary, summary_df


if __name__ == "__main__":
    main()
