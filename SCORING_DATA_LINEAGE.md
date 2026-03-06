# 점수 계산 로직 및 데이터 라인리지

## 1) 스케일링(0~100) 계산식

점수 엔진(`src/scoring_engine.py`)은 기본적으로 모든 지표를 0~100으로 정규화한 뒤 가중합한다.

### MinMax
- 정방향: `scaled = (x - min(x)) / (max(x)-min(x)) * 100`
- 역방향(값이 작을수록 좋은 지표): `scaled_rev = 100 - scaled`
- 분모가 0이면 50점 고정

### Z-score
- `z = (x - mean(x)) / std(x)` 후 다시 MinMax로 0~100 변환
- 역방향은 `100 - scaled`
- 표준편차 0이면 50점 고정

### Percentile
- `scaled = rank_pct(x) * 100`
- 역방향은 `100 - scaled`

## 2) 영역별 점수식

### 예산 점수
- `deposit_score = scale(deposit_price_krw, reverse=True)`
- `monthly_score = scale(monthly_rent_active_krw, reverse=True)`
- 1인가구: `price_score = 0.45*deposit_score + 0.55*monthly_score`
- 2인 맞벌이: `price_score = 0.60*deposit_score + 0.40*monthly_score`
- 최종 예산점수:
  - `budget_score = housing_budget_fit * 70 + scale(price_burden_index + monthly_burden_index, reverse=True) * 0.3`

### 인프라 점수
- `infra_score = 0.4*scale(hospital_count) + 0.3*scale(park_count) + 0.3*scale(mart_count)`
- `retail_license_count`는 원천 제거에 따라 계산에서 제외됨.

### 치안 점수
- `safety_score = 0.55*scale(crime_total_count, reverse=True) + 0.45*scale(police_satisfaction_score)`

### 통근 점수
- 1인가구: `commute_score = scale(commute_minutes, reverse=True)`
- 2인 맞벌이:
  - `avg_score = scale(commute_minutes, reverse=True)`
  - `worst_score = scale(worst_commute_minutes, reverse=True)`
  - `commute_score = 0.6*avg_score + 0.4*worst_score`

### 종합 점수
- 기본: `weighted_sum = Σ(영역점수 * 영역가중치) / Σ가중치`
- 옵션식:
  - 균형 보정: `total = weighted_sum - std(영역점수)*0.15`
  - 병목 기준: `total = min(영역점수)*0.55 + weighted_sum*0.45`

## 3) 데이터 라인리지(원천→집계→점수)

| 점수 영역 | 원천 데이터 | 파생/조인 방식 |
|---|---|---|
| 예산 | `seoul_apt_rent_5y.csv`, `apt_deal_total.csv` | 자치구·연도·면적구간 집계(중앙값), 예산적합/부담지수 계산 |
| 인프라 | `hospital.db`, `seoul_parks.csv`, `seoul_mart.csv` | 자치구 단위 건수 집계 후 가중합 |
| 치안 | `crime_2024.csv`, `police_satisfaction_2025.csv` | 자치구 평균/합 집계 후 정규화 |
| 통근 | `commute_models.csv`, `GU_CENTERS` | 자치구 중심 좌표와 허브 회귀계수로 통근시간 추정 |
| 개발(비교지표) | `25.12기준.서울시정비사업추진현황.csv` | 구역 수/진행단계 수 집계, 보조 분석에 활용 |

## 4) 점수 분포(배포 데이터 기준)

기준: `year=2025`, `20~29평`, 자치구 25개, MinMax.

### 분위수 요약
- 예산 점수: min 0.34 / Q1 77.53 / median 83.73 / Q3 86.37 / max 100.00
- 인프라 점수: min 5.71 / Q1 14.21 / median 19.83 / Q3 36.34 / max 76.43
- 치안 점수: min 17.17 / Q1 57.74 / median 77.93 / Q3 83.05 / max 90.32
- 통근 점수: min 0.00 / Q1 29.69 / median 50.98 / Q3 77.08 / max 100.00
- 종합 점수: min 39.92 / Q1 57.44 / median 61.73 / Q3 65.09 / max 72.49

### 구간 분포(0~20 / 20~40 / 40~60 / 60~80 / 80~100)

```text
예산  : 2 / 0 / 0 / 7 / 16
인프라: 13 / 7 / 4 / 1 / 0
치안  : 1 / 1 / 5 / 9 / 9
통근  : 3 / 5 / 5 / 7 / 5
종합  : 0 / 1 / 7 / 17 / 0
```

해석: 현재 배포 데이터에서는 예산 점수가 상위 구간으로 몰리고, 인프라는 저중간 구간 집중이 뚜렷하다. 따라서 가중치 조정 시 인프라/통근 가중치 변화가 순위에 상대적으로 크게 작동한다.
