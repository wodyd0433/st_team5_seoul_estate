from __future__ import annotations

import argparse
from pathlib import Path

from csv_utils import CSV_ENCODING_CANDIDATES

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "datasets" / "raw"
TARGET_ENCODING = "utf-8-sig"


def detect_encoding(raw: bytes) -> str:
    for encoding in CSV_ENCODING_CANDIDATES:
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", raw, 0, 1, "no supported encoding matched")


def normalize_file(path: Path, apply_changes: bool) -> tuple[str, bool]:
    raw = path.read_bytes()
    source_encoding = detect_encoding(raw)
    text = raw.decode(source_encoding)
    normalized = text.encode(TARGET_ENCODING)
    changed = raw != normalized

    if apply_changes and changed:
        path.write_bytes(normalized)

    return source_encoding, changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect CSV encodings in datasets/raw and normalize them to UTF-8 BOM."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write normalized UTF-8 BOM files in place. Default is dry run.",
    )
    args = parser.parse_args()

    csv_paths = sorted(DATA_DIR.glob("*.csv"))
    if not csv_paths:
        print(f"No CSV files found in {DATA_DIR}")
        return 1

    changed_count = 0
    unreadable: list[Path] = []

    for path in csv_paths:
        try:
            source_encoding, changed = normalize_file(path, args.apply)
            status = "rewrite" if changed else "ok"
            print(f"{path.name}\t{source_encoding}\t{status}")
            if changed:
                changed_count += 1
        except UnicodeDecodeError:
            unreadable.append(path)
            print(f"{path.name}\tunreadable\tfailed")

    print(f"\nCSV files: {len(csv_paths)}")
    print(f"Needs rewrite: {changed_count}")
    print(f"Unreadable: {len(unreadable)}")

    if unreadable:
        print("\nUnreadable files:")
        for path in unreadable:
            print(path.name)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
