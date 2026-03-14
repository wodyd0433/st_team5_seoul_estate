from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


API_ROOT = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
DEFAULT_PARAMS = {
    "method": "getList",
    "format": "json",
    "jsonVD": "Y",
    "prdSe": "Y",
    "newEstPrdCnt": "3",
    "orgId": "101",
    "itmId": "T10",
    "objL1": "ALL",
    "objL2": "ALL",
    "objL3": "",
    "objL4": "",
    "objL5": "",
    "objL6": "",
    "objL7": "",
    "objL8": "",
}


def build_url(api_key: str, table_id: str) -> str:
    params = {**DEFAULT_PARAMS, "apiKey": api_key, "tblId": table_id}
    return f"{API_ROOT}?{urlencode(params)}"


def fetch_kosis_rows(api_key: str, table_id: str) -> list[dict[str, object]]:
    url = build_url(api_key, table_id)
    with urlopen(url) as response:
        payload = response.read().decode("utf-8")
    rows = json.loads(payload)
    if not isinstance(rows, list):
        raise RuntimeError(f"Unexpected KOSIS response type: {type(rows)!r}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a KOSIS table and save it as CSV.")
    parser.add_argument("--api-key", required=True, help="KOSIS OpenAPI key")
    parser.add_argument("--table-id", required=True, help="KOSIS table id, e.g. DT_1NW1036")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = fetch_kosis_rows(args.api_key, args.table_id)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No rows returned for {args.table_id}")

    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(frame):,} rows to {output_path}")


if __name__ == "__main__":
    main()
