from __future__ import annotations

import numpy as np
import pandas as pd


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def remove_iqr_outliers(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    mask = pd.Series(True, index=result.index)
    for column in columns:
        if column not in result.columns:
            continue
        series = safe_numeric(result[column]).dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask &= result[column].between(lower, upper) | result[column].isna()
    return result.loc[mask].copy()


def fill_missing_values(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    result = df.copy()
    numeric_cols = result.select_dtypes(include=["number"]).columns
    for column in numeric_cols:
        filler = result[column].mean() if strategy == "mean" else result[column].min()
        result[column] = result[column].fillna(filler if not np.isnan(filler) else 0)
    return result
