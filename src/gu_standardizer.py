from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import GU_ALIASES, SEOUL_GUS


GU_PATTERN = re.compile(r"(강남구|강동구|강북구|강서구|관악구|광진구|구로구|금천구|노원구|도봉구|동대문구|동작구|마포구|서대문구|서초구|성동구|성북구|송파구|양천구|영등포구|용산구|은평구|종로구|중구|중랑구)")


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def standardize_gu_value(value: object) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    if text in GU_ALIASES:
        return GU_ALIASES[text]
    if text in SEOUL_GUS:
        return text
    text = text.replace("서울특별시", "").replace("서울시", "").replace("서울", "").strip()
    if text in GU_ALIASES:
        return GU_ALIASES[text]
    match = GU_PATTERN.search(text)
    return match.group(1) if match else None


def add_standard_gu(df: pd.DataFrame, candidate_columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    result["gu"] = None
    for column in candidate_columns:
        if column in result.columns:
            normalized = result[column].map(standardize_gu_value)
            result["gu"] = result["gu"].fillna(normalized)
    return result


def gu_match_report(df: pd.DataFrame, dataset_name: str) -> dict[str, object]:
    total_rows = len(df)
    matched_rows = int(df["gu"].notna().sum()) if "gu" in df.columns else 0
    unmatched_rows = total_rows - matched_rows
    samples = ""
    if "gu" in df.columns and unmatched_rows:
        samples = ", ".join(df.loc[df["gu"].isna()].astype(str).head(3).sum(axis=1).tolist())
    return {
        "dataset": dataset_name,
        "rows": total_rows,
        "matched_rows": matched_rows,
        "unmatched_rows": unmatched_rows,
        "match_rate_pct": round(matched_rows / total_rows * 100, 2) if total_rows else 0.0,
        "unmatched_samples": samples,
    }
