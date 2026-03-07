from __future__ import annotations

from pathlib import Path

from csv_utils import CSV_ENCODING_CANDIDATES, read_csv_auto


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "datasets" / "raw"


def detect_encoding(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in CSV_ENCODING_CANDIDATES:
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "unreadable"


def main() -> int:
    csv_paths = sorted(DATA_DIR.glob("*.csv"))
    if not csv_paths:
        print(f"No CSV files found in {DATA_DIR}")
        return 1

    failures = 0
    for path in csv_paths:
        encoding = detect_encoding(path)
        if encoding == "unreadable":
            print(f"{path.name}\tunreadable\t-\t-")
            failures += 1
            continue

        df = read_csv_auto(path, nrows=3)
        print(f"{path.name}\t{encoding}\trows={len(df)}\tcols={len(df.columns)}")

    print(f"\nCSV files: {len(csv_paths)}")
    print(f"Unreadable: {failures}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
