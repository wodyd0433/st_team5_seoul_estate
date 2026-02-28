from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.config import PRICE_COLUMN_CANDIDATES


def to_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace({"": np.nan, "-": np.nan, ".": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def infer_price_unit(series: pd.Series) -> tuple[str, float]:
    numeric = to_numeric(series).dropna()
    if numeric.empty:
        return "unknown", 1.0
    median = float(numeric.median())
    if median > 100_000_000:
        return "원", 1.0
    if median < 100_000:
        return "만원", 10_000.0
    return "원 추정", 1.0


def candidate_price_columns(df: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    lowered = {str(col).lower(): col for col in df.columns}
    for semantic, names in PRICE_COLUMN_CANDIDATES.items():
        for name in names:
            if name in df.columns:
                mapping[semantic] = name
                break
            if name.lower() in lowered:
                mapping[semantic] = lowered[name.lower()]
                break
    return mapping


def standardize_price_columns(df: pd.DataFrame, columns: Iterable[str]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    result = df.copy()
    report: list[dict[str, object]] = []
    for column in columns:
        if column not in result.columns:
            continue
        unit, multiplier = infer_price_unit(result[column])
        numeric = to_numeric(result[column])
        result[f"{column}_krw"] = numeric * multiplier
        report.append(
            {
                "column": column,
                "detected_unit": unit,
                "multiplier_to_krw": multiplier,
                "median_raw_value": round(float(numeric.median()), 2) if numeric.notna().any() else None,
            }
        )
    return result, report
