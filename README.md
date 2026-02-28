# Seoul Newlywed Housing Recommendation Dashboard

Streamlit dashboard for recommending Seoul residential districts for newlyweds based on budget, commute, safety, and local infrastructure.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository Scope

This repository is intended to store application code only.

Included:

- `app.py`
- `src/`
- `requirements.txt`
- `README.md`
- `.gitignore`

Excluded:

- raw datasets
- local output files
- logs
- virtual environments

## Data Location

Raw data files are not included in this repository because of size limits.

Place the required files in the sibling directory:

```text
../data_all/
```

Expected structure:

```text
project_root/
├── data_all/
└── gpt_analysis/
    ├── app.py
    ├── requirements.txt
    └── src/
```

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

## GitHub Upload Recommendation

Upload only:

- `app.py`
- `requirements.txt`
- `README.md`
- `.gitignore`
- `src/`

Do not upload:

- `outputs/`
- `data_all/`
- `.venv/`
- log files
