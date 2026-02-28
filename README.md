# Seoul Newlywed Housing Recommendation Dashboard

서울 신혼부부 전월세 입지 추천을 위한 Streamlit 대시보드입니다. 예산, 통근, 치안, 인프라를 종합해 자치구 추천 결과를 제공합니다.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository Scope

이 저장소는 기본적으로 애플리케이션 코드 중심으로 관리합니다.

포함 권장:

- `app.py`
- `src/`
- `requirements.txt`
- `README.md`
- `.gitignore`

## Data Setup

앱은 아래 순서로 데이터 폴더를 탐색합니다.

1. `DATA_DIR` 환경변수
2. `gpt_analysis/data_all/`
3. 상위 폴더의 `data_all/`
4. `gpt_analysis/deploy_data/` 경량 배포 파일

즉 Streamlit Cloud에 배포할 때는 아래 두 방식 중 하나를 쓰면 됩니다.

### Option 1. 저장소 내부에 data_all 포함

```text
gpt_analysis/
├── app.py
├── requirements.txt
├── src/
└── data_all/
```

### Option 2. 환경변수로 데이터 경로 지정

`DATA_DIR` 를 실제 데이터 폴더 경로로 설정합니다.

### Option 3. 경량 배포 데이터 사용

원본 `data_all` 이 너무 커서 GitHub/Streamlit Cloud에 올릴 수 없으면 아래 스크립트로 경량 파일을 생성합니다.

```bash
python scripts/build_deploy_data.py
```

생성 파일:

- `deploy_data/compact_housing.csv`
- `deploy_data/compact_district_metrics.csv`
- `deploy_data/commute_models.csv`

이 세 파일만 저장소에 포함하면 Streamlit Cloud에서 추천 대시보드를 실행할 수 있습니다.

## Main Data Files

- `apt_deal_total.csv`
- `seoul_apt_rent_5y.csv`
- `자치구별_아파트_전월세_평균2021.csv`
- `자치구별_아파트_전월세_평균2022.csv`
- `자치구별_아파트_전월세_평균2023.csv`
- `자치구별_아파트_전월세_평균2024.csv`
- `자치구별_아파트_전월세_평균2025.csv`
- `seoul_infra_summary.csv`
- `distribution_license.csv`
- `seoul_parks.csv`
- `seoul_parks_stats.csv`
- `hospital_data.csv`
- `hospital.db`
- `crime_2024.csv`
- `police_satisfaction_2025.csv`
- `25.12기준.서울시정비사업추진현황.csv`
- `Gwanghwamun_time_zones.csv`
- `gangnam_time_zones_20260227.csv`
- `seongsu_time_zones_20260227.csv`
- `yeouido_time_zones_20260224.csv`
