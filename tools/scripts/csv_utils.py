from __future__ import annotations

from pathlib import Path

import pandas as pd


CSV_ENCODING_CANDIDATES = ("utf-8-sig", "utf-8", "cp949", "euc-kr")


def read_csv_auto(path: str | Path, **kwargs) -> pd.DataFrame:
    csv_path = Path(path)
    last_error: Exception | None = None
    for encoding in CSV_ENCODING_CANDIDATES:
        try:
            return pd.read_csv(csv_path, encoding=encoding, **kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to read {csv_path.name}: {last_error}")
