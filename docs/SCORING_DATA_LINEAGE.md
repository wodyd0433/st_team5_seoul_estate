# 점수 계산 로직 및 데이터 라인리지

## 1) 스케일링(0~100) 계산식

점수 엔진(`src/scoring_engine.py`)은 현재 사용자 경험 최적화를 위해 **별점 시스템(Stars)** 기반 점수화를 수행한다.

### 별점 기반 점수 (Stars to Score)
- 각 지표를 1~5단계 별점으로 변환 후 20을 곱해 0~100점 사이의 점수로 환산한다.
- `Score = Star * 20`
- 별점 기준:
  - **가격**: 3.5억 미만(5), 5억 미만(4), 7억 미만(3), 9.5억 미만(2), 그 이상(1)
  - **통근**: 20분 이하(5), 30분 이하(4), 45분 이하(3), 60분 이하(2), 그 이상(1)
  - **전세가율**: 50% 미만(5), 60% 미만(4), 70% 미만(3), 80% 미만(2), 그 이상(1)
  - **인프라/치안**: 마트/병원/공원/범죄율 등 복합 규칙 기반 산정

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
- `Weighted Star = Σ(영역별 별점 * 영역 가중치)`
- `Total Score = Weighted Star * 20`
- 등급 기준: 90점 이상(S), 75점 이상(A), 55점 이상(B), 35점 이상(C), 그 미만(D)

## 3) 데이터 라인리지(원천→집계→점수)

| 점수 영역 | 원천 데이터 출처 | 파생/조인 방식 |
|---|---|---|
| 예산 | 국토교통부 실거래가 (`seoul_apt_rent_5y.csv`) | 자치구·연도·면적구간 집계(중앙값) |
| 인프라 | 서울시 열린데이터광장 (`hospital.db`, `seoul_parks.csv`, `seoul_mart.csv`) | 자치구 단위 건수 집계 후 규칙 기반 산정 |
| 치안 | 경찰청/공공데이터포털 (`crime_2024.csv`, `police_satisfaction_2025.csv`) | 자치구별 범죄 발생량 및 만족도 집계 |
| 통근 | 카카오맵 API/좌표 기반 시뮬레이션 (`commute_models.csv`) | 자치구 중심-직장 간 최단거리 회귀모델 추정 |
| 개발 | 서울시 정비사업 추진현황 (상시 업데이트) | 구역 수/진행단계 집계 |

## 4) 점수 분포(배포 데이터 기준)

기준: `year=2025`, `20~29평`, 자치구 25개, 별점 환산 방식.

### 분위수 요약
- 가격 점수: min 40 / Q1 60 / median 60 / Q3 80 / max 100
- 인프라 점수: min 20 / Q1 40 / median 60 / Q3 100 / max 100
- 치안 점수: min 20 / Q1 60 / median 60 / Q3 80 / max 80
- 통근 점수: min 60 / Q1 80 / median 80 / Q3 100 / max 100
- 종합 점수: min 64.0 / Q1 69.0 / median 72.0 / Q3 75.0 / max 84.0

### 구간 분포(0~20 / 20~40 / 40~60 / 60~80 / 80~100)

```text
가격  : 0 / 0 / 3 / 14 / 8
인프라: 0 / 4 / 5 / 7 / 9
치안  : 0 / 3 / 3 / 12 / 7
통근  : 0 / 0 / 0 / 5 / 20
종합  : 0 / 0 / 0 / 22 / 3
```

해석: 별점 기반 점수화 시스템으로 인해 점수가 특정 구간(20점 단위)에 이산적으로 분포하며, 통근 점수의 경우 대체로 상향 평준화되어 있다. 종합 점수는 대부분 60~80점 사이(D~B등급)의 안정적인 분포를 보인다.
