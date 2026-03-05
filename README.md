# Seoul Newlywed Housing Recommendation Dashboard

서울 신혼부부 전월세·매수 추천을 위한 Streamlit 대시보드입니다. 예산, 통근, 치안, 인프라(병원·공원·대형마트) 지표를 결합해 자치구 추천을 제공합니다.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository Scope

저장소에는 애플리케이션 코드와 경량 배포 데이터만 포함합니다.

- `app.py`
- `src/`
- `scripts/`
- `deploy_data/`
- `requirements.txt`
- `README.md`

## Data Setup

앱은 아래 순서로 데이터 폴더를 탐색합니다.

1. `DATA_DIR` 환경변수
2. `gpt_analysis/data_all/`
3. 상위 루트 `data_all/`
4. `gpt_analysis/deploy_data/` (경량 배포 데이터)

원본 `data_all`이 없는 환경에서는 아래 스크립트로 경량 데이터를 생성해 사용합니다.

```bash
python scripts/build_deploy_data.py
```

생성 파일:

- `deploy_data/compact_housing.csv`
- `deploy_data/compact_district_metrics.csv`
- `deploy_data/commute_models.csv`
- `deploy_data/persona_profiles.csv`

## Main Data Files (Raw)

- `apt_deal_total.csv`
- `seoul_apt_rent_5y.csv`
- `자치구별_아파트_전월세_평균2021~2025.csv` 계열
- `seoul_parks.csv`
- `seoul_mart.csv`
- `hospital.db`
- `crime_2024.csv`
- `police_satisfaction_2025.csv`
- `25.12기준.서울시정비사업추진현황.csv`
- `DT_1NW1027.csv`
- `debt_newlyweds.csv`
- `*_time_zones*.csv`

참고: `distribution_license.csv`는 원천에서 제거되어 현재 점수 계산과 대시보드에 사용하지 않습니다.
