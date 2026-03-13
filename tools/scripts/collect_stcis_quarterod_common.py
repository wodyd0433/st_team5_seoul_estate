from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep

import pandas as pd
import requests


# STCIS API 및 프로젝트 기준 경로를 한곳에 모아 재사용한다.
API_ROOT = "https://stcis.go.kr/openapi"
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "datasets" / "raw"
ENV_PATH = ROOT_DIR / ".env"


@dataclass
class RequestStat:
    # API 호출 결과를 유형별로 집계해 메타 정보에 남긴다.
    ok: int = 0
    not_found: int = 0
    error: int = 0
    exception: int = 0


def load_env(path: Path = ENV_PATH) -> None:
    # python-dotenv 없이도 .env 값을 현재 프로세스 환경 변수로 적재한다.
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def call_api(session: requests.Session, endpoint: str, params: dict[str, str], timeout: int = 30) -> dict:
    # 공통 GET 호출 래퍼로 HTTP 오류를 즉시 노출하고 JSON만 반환한다.
    response = session.get(f"{API_ROOT}/{endpoint}", params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def make_session() -> requests.Session:
    # 재사용 가능한 세션을 만들고 서비스 식별용 User-Agent를 설정한다.
    session = requests.Session()
    session.headers.update({"User-Agent": "korea-estate-data-collector/1.0"})
    return session


def fetch_area_codes(session: requests.Session, apikey: str) -> pd.DataFrame:
    # 시도 -> 시군구 -> 읍면동 순으로 전체 행정구역 코드를 펼쳐서 테이블로 만든다.
    province_data = call_api(session, "areacode.json", {"apikey": apikey})
    provinces = province_data.get("result", []) or []
    rows: list[dict[str, str | None]] = []

    for province in provinces:
        sd_cd = str(province.get("sdCd") or "")
        if not sd_cd:
            continue
        # 각 시도 아래 시군구 목록을 조회한다.
        sgg_data = call_api(session, "areacode.json", {"apikey": apikey, "sdCd": sd_cd})
        for sgg in sgg_data.get("result", []) or []:
            sgg_cd = str(sgg.get("sggCd") or "")
            if not sgg_cd:
                continue
            # 각 시군구 아래 읍면동 목록을 조회해 최종 수집 대상 행으로 저장한다.
            emd_data = call_api(session, "areacode.json", {"apikey": apikey, "sdCd": sd_cd, "sggCd": sgg_cd})
            for emd in emd_data.get("result", []) or []:
                emd_cd = str(emd.get("emdCd") or "")
                if not emd_cd:
                    continue
                rows.append(
                    {
                        "sdCd": str(emd.get("sdCd") or ""),
                        "sdNm": emd.get("sdNm"),
                        "sggCd": str(emd.get("sggCd") or ""),
                        "sggNm": emd.get("sggNm"),
                        "emdCd": emd_cd,
                        "emdNm": emd.get("emdNm"),
                    }
                )
            sleep(0.05)
        sleep(0.05)

    frame = pd.DataFrame(rows)
    # 중복 코드 제거 후 행정구역 코드 순서대로 정렬한다.
    return frame.drop_duplicates(subset=["emdCd"]).sort_values(["sdCd", "sggCd", "emdCd"]).reset_index(drop=True)


def is_available_date(
    session: requests.Session,
    apikey: str,
    oprat_date: str,
    destination_emd_cd: str,
    sample_origin_emd_cd: str = "1168010100",
) -> bool:
    # 대표 출발지 하나로 호출 가능 여부만 확인해 날짜 유효성을 빠르게 판정한다.
    payload = call_api(
        session,
        "quarterod.json",
        {
            "apikey": apikey,
            "opratDate": oprat_date,
            "stgEmdCd": sample_origin_emd_cd,
            "arrEmdCd": destination_emd_cd,
        },
    )
    return str(payload.get("status") or "") == "OK"


def find_latest_available_date(
    session: requests.Session,
    apikey: str,
    requested_date: str,
    destination_emd_cd: str,
    lookback_days: int,
) -> str:
    # 요청일이 비어 있으면 과거로 하루씩 이동하며 실제 조회 가능한 최신 일자를 찾는다.
    current = datetime.strptime(requested_date, "%Y%m%d")
    for _ in range(lookback_days + 1):
        candidate = current.strftime("%Y%m%d")
        if is_available_date(session, apikey, candidate, destination_emd_cd):
            return candidate
        current -= timedelta(days=1)
    raise RuntimeError(
        f"No available quarter OD date found within {lookback_days} days of {requested_date} for destination {destination_emd_cd}"
    )


def build_gu_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    # 읍면동 출발 데이터를 구 단위로 집계해 요약 테이블을 만든다.
    if frame.empty:
        return pd.DataFrame()

    working = frame.copy()
    # 합계 및 가중평균 계산 전에 수치형 컬럼을 명시적으로 변환한다.
    working["useStf"] = pd.to_numeric(working["useStf"], errors="coerce")
    working["useTm"] = pd.to_numeric(working["useTm"], errors="coerce")
    grouped = (
        working.groupby(
            [
                "opratDate",
                "stgSdCd",
                "stgSdNm",
                "stgSggCd",
                "stgSggNm",
                "arrSdCd",
                "arrSdNm",
                "arrSggCd",
                "arrSggNm",
                "arrEmdCd",
                "arrEmdNm",
                "tzon",
                "quater",
            ],
            dropna=False,
        )
        # 출발 읍면동 수, 이동 인원 합계, 이용 시간 가중평균을 함께 계산한다.
        .apply(
            lambda group: pd.Series(
                {
                    "origin_emd_count": group["stgEmdCd"].nunique(),
                    "useStf": group["useStf"].sum(min_count=1),
                    "useTm": (group["useTm"] * group["useStf"]).sum() / group["useStf"].sum() if group["useStf"].sum() else pd.NA,
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return grouped


def collect_quarter_od(
    session: requests.Session,
    apikey: str,
    oprat_date: str,
    destination_emd_cd: str,
    origins: pd.DataFrame,
    hour_from: int,
    hour_to: int,
    pause_seconds: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    # 목적지 하나를 기준으로 모든 출발 읍면동의 분기별 OD 데이터를 수집한다.
    rows: list[dict[str, object]] = []
    stats = RequestStat()
    sample_errors: list[dict[str, object]] = []

    for idx, origin in enumerate(origins.itertuples(index=False), start=1):
        # 각 출발지 코드별로 동일한 목적지에 대한 quarter OD를 조회한다.
        params = {
            "apikey": apikey,
            "opratDate": oprat_date,
            "stgEmdCd": str(origin.emdCd),
            "arrEmdCd": destination_emd_cd,
        }
        try:
            payload = call_api(session, "quarterod.json", params)
            status = str(payload.get("status") or "")
            result = payload.get("result") or []
            if status == "OK":
                stats.ok += 1
                # 요청된 시간대 범위에 해당하는 레코드만 남기고 출발지 메타를 덧붙인다.
                for item in result:
                    tzon = pd.to_numeric(item.get("tzon"), errors="coerce")
                    if pd.isna(tzon) or not (hour_from <= int(tzon) <= hour_to):
                        continue
                    row = dict(item)
                    row["origin_sdCd"] = origin.sdCd
                    row["origin_sdNm"] = origin.sdNm
                    row["origin_sggCd"] = origin.sggCd
                    row["origin_sggNm"] = origin.sggNm
                    row["origin_emdCd"] = origin.emdCd
                    row["origin_emdNm"] = origin.emdNm
                    rows.append(row)
            elif status == "NOT_FOUND":
                # 데이터가 없는 출발지는 정상적인 빈 결과로 집계만 한다.
                stats.not_found += 1
            else:
                # 예상하지 못한 응답은 일부 샘플을 남겨 사후 점검이 가능하게 한다.
                stats.error += 1
                if len(sample_errors) < 20:
                    sample_errors.append({"origin_emdCd": origin.emdCd, "status": status, "payload": payload})
        except Exception as exc:  # pragma: no cover
            # 예외도 수집을 중단하지 않고 샘플만 기록한 뒤 다음 출발지로 진행한다.
            stats.exception += 1
            if len(sample_errors) < 20:
                sample_errors.append({"origin_emdCd": origin.emdCd, "status": "EXCEPTION", "error": str(exc)})

        if pause_seconds > 0:
            sleep(pause_seconds)
        # 장시간 실행 중 진행 상황을 주기적으로 출력한다.
        if idx % 250 == 0:
            print(f"processed={idx} ok={stats.ok} not_found={stats.not_found} error={stats.error} exception={stats.exception}")

    frame = pd.DataFrame(rows)
    # 산출물과 호출 상태를 함께 저장할 메타 정보를 구성한다.
    meta = {
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "oprat_date": oprat_date,
        "time_filter": {"hour_from": hour_from, "hour_to": hour_to},
        "destination_emd_cd": destination_emd_cd,
        "origin_emd_count": int(len(origins)),
        "request_stats": asdict(stats),
        "row_count": int(len(frame)),
        "sample_errors": sample_errors,
    }
    return frame, meta


def run_collection(
    requested_date: str,
    destination_emd_cd: str,
    destination_name: str,
    hour_from: int,
    hour_to: int,
    pause_seconds: float,
    lookback_days: int,
) -> None:
    # 환경 변수, 출력 경로, 실제 조회 가능 날짜를 먼저 준비한다.
    load_env()
    apikey = os.getenv("STCIS_QUARTEROD_API_KEY")
    if not apikey:
        raise RuntimeError("STCIS_QUARTEROD_API_KEY not found in environment or .env")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    session = make_session()
    resolved_date = find_latest_available_date(
        session=session,
        apikey=apikey,
        requested_date=requested_date,
        destination_emd_cd=destination_emd_cd,
        lookback_days=lookback_days,
    )
    if resolved_date != requested_date:
        print(f"requested date {requested_date} unavailable; using latest available date {resolved_date}")

    # 원본 데이터, 메타데이터, 구 단위 집계 파일 경로를 미리 계산한다.
    out_csv = (
        RAW_DIR
        / f"od_15m_all_emd_to_{destination_name}_{destination_emd_cd}_{resolved_date}_{hour_from:02d}_{hour_to:02d}.csv"
    )
    out_meta = (
        RAW_DIR
        / f"od_15m_all_emd_to_{destination_name}_{destination_emd_cd}_{resolved_date}_{hour_from:02d}_{hour_to:02d}_meta.json"
    )
    out_gu_csv = (
        RAW_DIR
        / f"od_15m_seoul_gu_to_{destination_name}_{destination_emd_cd}_{resolved_date}_{hour_from:02d}_{hour_to:02d}.csv"
    )
    out_area = RAW_DIR / "stcis_areacode_seoul.csv"

    # 전체 행정구역 코드 중 서울만 필터링해 출발지 목록으로 사용한다.
    print("fetching area codes...")
    area_codes = fetch_area_codes(session, apikey)
    area_codes = area_codes.loc[area_codes["sdCd"].eq("11")].copy()
    area_codes.to_csv(out_area, index=False, encoding="utf-8-sig")
    print(f"area codes saved: {out_area} rows={len(area_codes)}")

    # 서울 전체 읍면동을 출발지로 하여 지정 목적지 OD 데이터를 수집한다.
    print("collecting quarter OD...")
    frame, meta = collect_quarter_od(
        session=session,
        apikey=apikey,
        oprat_date=resolved_date,
        destination_emd_cd=destination_emd_cd,
        origins=area_codes,
        hour_from=hour_from,
        hour_to=hour_to,
        pause_seconds=pause_seconds,
    )

    if not frame.empty:
        # 원본은 상세 기준으로 정렬 저장하고, 별도로 구 단위 집계본도 생성한다.
        frame = frame.sort_values(["opratDate", "stgSggCd", "stgEmdCd", "tzon", "quater"]).reset_index(drop=True)
        frame.to_csv(out_csv, index=False, encoding="utf-8-sig")
        gu_frame = build_gu_aggregate(frame)
        gu_frame.to_csv(out_gu_csv, index=False, encoding="utf-8-sig")
    else:
        gu_frame = pd.DataFrame()

    # 실행 조건과 산출물 위치를 메타 파일에 기록해 재현성을 확보한다.
    meta["output_csv"] = str(out_csv)
    meta["output_gu_csv"] = str(out_gu_csv)
    meta["output_area_codes_csv"] = str(out_area)
    meta["scope"] = "seoul_only"
    meta["requested_date"] = requested_date
    meta["resolved_date"] = resolved_date
    meta["destination_name"] = destination_name
    meta["gu_row_count"] = int(len(gu_frame))
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"od saved: {out_csv} rows={len(frame)}")
    print(f"gu od saved: {out_gu_csv} rows={len(gu_frame)}")
    print(f"meta saved: {out_meta}")
