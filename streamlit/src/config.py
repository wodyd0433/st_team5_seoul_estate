from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
DEPLOY_DATA_DIR = PROJECT_ROOT / "datasets" / "deploy"
PRIMARY_DATA_DIR = PROJECT_ROOT / "datasets" / "raw"
REQUIRED_DATA_FILENAMES = {
    "apartment_rent_avg_by_gu_area_2021_2025.csv",
    "seoul_crime_report_2024.csv",
    "police_satisfaction_by_station_2025.csv",
    "seoul_hospital_registry.db",
}
REQUIRED_DATA_GLOBS = {
    "apartment_sale_transactions_*.csv",
    "apartment_rent_transactions_*.csv",
}


def _candidate_score(path: Path) -> tuple[int, int]:
    if not path.exists():
        return (-1, -1)
    present = sum((path / name).exists() for name in REQUIRED_DATA_FILENAMES)
    present += sum(any(path.glob(pattern)) for pattern in REQUIRED_DATA_GLOBS)
    total = len(list(path.glob("*")))
    return (present, total)


def resolve_data_dir() -> Path:
    env_dir = os.getenv("DATA_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(PRIMARY_DATA_DIR)
    ranked = sorted(
        ((candidate, _candidate_score(candidate)) for candidate in candidates),
        key=lambda item: item[1],
        reverse=True,
    )
    for candidate, score in ranked:
        if score[0] >= 0:
            return candidate
    return candidates[0] if candidates else PRIMARY_DATA_DIR


DATA_DIR = resolve_data_dir()
DATA_DIR_CANDIDATES = [
    path
    for path in [
        os.getenv("DATA_DIR"),
        str(PRIMARY_DATA_DIR),
    ]
    if path
]

COMPACT_DATA_PATHS = {
    "housing": DEPLOY_DATA_DIR / "compact_housing.csv",
    "district_metrics": DEPLOY_DATA_DIR / "compact_district_metrics.csv",
    "commute_models": DEPLOY_DATA_DIR / "commute_models.csv",
    "persona_profiles": DEPLOY_DATA_DIR / "persona_profiles.csv",
}

APT_SALE_FILE_GLOB = "apartment_sale_transactions_*.csv"
APT_RENT_FILE_GLOB = "apartment_rent_transactions_*.csv"

DATASET_PATHS = {
    "rent_avg_all": DATA_DIR / "apartment_rent_avg_by_gu_area_2021_2025.csv",
    "seoul_parks": DATA_DIR / "seoul_park_locations.csv",
    "seoul_mart": DATA_DIR / "seoul_large_mart_locations.csv",
    "hospital_db": DATA_DIR / "seoul_hospital_registry.db",
    "crime": DATA_DIR / "seoul_crime_report_2024.csv",
    "police": DATA_DIR / "police_satisfaction_by_station_2025.csv",
    "redevelopment": DATA_DIR / "seoul_redevelopment_projects_202512.csv",
    "income_newlyweds": DATA_DIR / "newlywed_income_by_region.csv",
    "debt_newlyweds": DATA_DIR / "newlywed_debt_by_region.csv",
}

SEOUL_GUS = [
    "강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구", "노원구", "도봉구",
    "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구", "성북구", "송파구", "양천구", "영등포구",
    "용산구", "은평구", "종로구", "중구", "중랑구",
]

GU_ALIASES = {f"서울 {gu}": gu for gu in SEOUL_GUS}
GU_ALIASES.update({f"서울특별시 {gu}": gu for gu in SEOUL_GUS})
GU_ALIASES.update({
    "종로구청": "종로구",
    "중구청": "중구",
    "용산구청": "용산구",
    "영등포구청": "영등포구",
})

ENCODING_CANDIDATES = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
RAW_CACHE_TTL = 3600
FEATURE_CACHE_TTL = 3600
SCORING_CACHE_TTL = 3600

PRICE_COLUMN_CANDIDATES = {
    "deposit": ["보증금_만원", "보증금", "평균 전세", "평균 보증금", "deposit", "jeonse"],
    "monthly": ["월세_만원", "월세", "평균 월세", "monthly_rent"],
    "sale": ["dealAmount", "매매가", "거래금액", "sale_price"],
}

POLICE_STATION_TO_GU = {
    "서울중부경찰서": "중구",
    "서울종로경찰서": "종로구",
    "서울남대문경찰서": "중구",
    "서울서대문경찰서": "서대문구",
    "서울혜화경찰서": "종로구",
    "서울용산경찰서": "용산구",
    "서울성북경찰서": "성북구",
    "서울동대문경찰서": "동대문구",
    "서울마포경찰서": "마포구",
    "서울영등포경찰서": "영등포구",
    "서울성동경찰서": "성동구",
    "서울동작경찰서": "동작구",
    "서울광진경찰서": "광진구",
    "서울서부경찰서": "은평구",
    "서울강북경찰서": "강북구",
    "서울금천경찰서": "금천구",
    "서울중랑경찰서": "중랑구",
    "서울강남경찰서": "강남구",
    "서울관악경찰서": "관악구",
    "서울강서경찰서": "강서구",
    "서울강동경찰서": "강동구",
    "서울종암경찰서": "성북구",
    "서울구로경찰서": "구로구",
    "서울서초경찰서": "서초구",
    "서울양천경찰서": "양천구",
    "서울송파경찰서": "송파구",
    "서울노원경찰서": "노원구",
    "서울방배경찰서": "서초구",
    "서울수서경찰서": "강남구",
    "서울도봉경찰서": "도봉구",
    "서울은평경찰서": "은평구",
}

GU_CENTERS = {
    "강남구": {"lat": 37.5172, "lon": 127.0473},
    "강동구": {"lat": 37.5301, "lon": 127.1238},
    "강북구": {"lat": 37.6396, "lon": 127.0257},
    "강서구": {"lat": 37.5509, "lon": 126.8495},
    "관악구": {"lat": 37.4784, "lon": 126.9516},
    "광진구": {"lat": 37.5385, "lon": 127.0823},
    "구로구": {"lat": 37.4954, "lon": 126.8874},
    "금천구": {"lat": 37.4569, "lon": 126.8956},
    "노원구": {"lat": 37.6542, "lon": 127.0568},
    "도봉구": {"lat": 37.6688, "lon": 127.0471},
    "동대문구": {"lat": 37.5744, "lon": 127.0396},
    "동작구": {"lat": 37.5124, "lon": 126.9393},
    "마포구": {"lat": 37.5663, "lon": 126.9019},
    "서대문구": {"lat": 37.5792, "lon": 126.9368},
    "서초구": {"lat": 37.4837, "lon": 127.0324},
    "성동구": {"lat": 37.5633, "lon": 127.0364},
    "성북구": {"lat": 37.5894, "lon": 127.0167},
    "송파구": {"lat": 37.5145, "lon": 127.1059},
    "양천구": {"lat": 37.5169, "lon": 126.8664},
    "영등포구": {"lat": 37.5264, "lon": 126.8962},
    "용산구": {"lat": 37.5323, "lon": 126.9900},
    "은평구": {"lat": 37.6176, "lon": 126.9227},
    "종로구": {"lat": 37.5735, "lon": 126.9790},
    "중구": {"lat": 37.5641, "lon": 126.9979},
    "중랑구": {"lat": 37.6063, "lon": 127.0927},
}

WORKPLACE_HUBS = {
    "광화문역": {"lat": 37.5714, "lon": 126.9768, "label": "광화문역", "base_station": "광화문"},
    "강남역": {"lat": 37.4979, "lon": 127.0276, "label": "강남역", "base_station": "강남"},
    "성수역": {"lat": 37.5446, "lon": 127.0557, "label": "성수역", "base_station": "성수"},
    "여의도역": {"lat": 37.5219, "lon": 126.9245, "label": "여의도역", "base_station": "여의도"},
}

COMMUTE_ZONE_PATHS = {
    "광화문역": DATA_DIR / "commute_zone_gwanghwamun.csv",
    "강남역": DATA_DIR / "commute_zone_gangnam_20260227.csv",
    "성수역": DATA_DIR / "commute_zone_seongsu_20260227.csv",
    "여의도역": DATA_DIR / "commute_zone_yeouido_20260224.csv",
}
