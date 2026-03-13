from __future__ import annotations

import argparse

from collect_stcis_quarterod_common import run_collection


DESTINATION_EMD_CD = "1156011000"
DESTINATION_NAME = "여의도"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260225")
    parser.add_argument("--hour-from", type=int, default=7)
    parser.add_argument("--hour-to", type=int, default=11)
    parser.add_argument("--pause-seconds", type=float, default=0.05)
    parser.add_argument("--lookback-days", type=int, default=14)
    args = parser.parse_args()

    run_collection(
        requested_date=args.date,
        destination_emd_cd=DESTINATION_EMD_CD,
        destination_name=DESTINATION_NAME,
        hour_from=args.hour_from,
        hour_to=args.hour_to,
        pause_seconds=args.pause_seconds,
        lookback_days=args.lookback_days,
    )


if __name__ == "__main__":
    main()
