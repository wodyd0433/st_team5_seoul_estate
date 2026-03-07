# cleaned

`cleaned/` 는 `raw/` 원천 데이터를 정제하거나 분석용으로 요약한 CSV를 모아둔 디렉터리다.

## Housing

- `housing/apartment_rent_transactions_cleaned.csv`
  아파트 임대차 원천 거래를 정리한 정제본이다.
- `housing/apartment_sale_transactions_cleaned.csv`
  아파트 매매 원천 거래를 정리한 정제본이다.
- `housing/apartment_rent_avg_by_gu_area_cleaned.csv`
  자치구+연도+면적구간 기준 전세/월세 평균 집계본이다.
- `housing/budget_candidates_by_gu.csv`
  예산 조건을 기준으로 후보 자치구를 요약한 파일이다.
- `housing/area_band_price_distribution.csv`
  자치구+면적구간+임대유형 기준 가격 수준 요약본이다. 전세보증금 중앙값/평균, 월세 중앙값, 거래건수를 담고 있으며 연도 축은 없다.
- `housing/jeonse_ratio_by_gu_year_area.csv`
  자치구+연도+면적구간 기준 전세가율 요약본이다. 현재는 전세/매매 모두 중앙값 기준으로 계산하며 `33㎡`, `59㎡`, `84㎡` 구간만 사용한다.
- `housing/jeonse_ratio_ranking_by_gu.csv`
  자치구별 전세가율 비교용 랭킹 요약본이다.
- `housing/market_trend_by_gu_year_rent_type.csv`
  자치구+연도+임대유형 기준 시장 추세 요약본이다. `median_deposit`, `median_monthly_rent`, `tx_count` 컬럼으로 전세보증금 중앙값, 월세 중앙값, 거래건수를 담는다.
- `housing/rent_ratio_by_gu_year.csv`
  자치구+연도 기준 전세/월세 거래 비중 요약본이다.

## Safety

- `safety/crime_clean.csv`
  범죄 원천 데이터를 정리한 정제본이다.
- `safety/crime_summary_by_gu.csv`
  범죄 데이터를 자치구 단위로 요약한 파일이다.
- `safety/police_clean.csv`
  경찰 만족도 원천 데이터를 정리한 정제본이다.
- `safety/police_satisfaction_by_gu.csv`
  경찰 만족도를 자치구 단위 평균으로 집계한 파일이다.
- `safety/safety_merged_summary.csv`
  범죄와 경찰 만족도 요약을 합친 통합 파일이다.

## Infra

- `infra/hospital_csv_clean.csv`
  병원 CSV 원천을 정리한 정제본이다.
- `infra/hospital_count_by_gu_type.csv`
  자치구+기관유형 기준 병원 수를 집계한 파일이다.
- `infra/hospital_reference_union.csv`
  병원 관련 참조 테이블을 합친 비교용 파일이다.
- `infra/parks_clean.csv`
  공원 원천 데이터를 정리한 정제본이다.
- `infra/infra_composite_index.csv`
  병원, 공원, 생활 편의 집계를 합쳐 만든 인프라 종합 지표다.

## Redevelopment

- `redevelopment/redevelopment_clean.csv`
  정비사업 원천을 정리한 정제본이다.
- `redevelopment/redevelopment_summary.csv`
  정비사업 현황을 자치구 단위로 요약한 파일이다.

## Transform Rules

- 공통 정제 단계에서는 인코딩, 컬럼명, 자치구명, 숫자형 금액 컬럼을 표준화한다.
- 거래 데이터는 원천 row 단위 정제본과 집계 요약본을 분리해 보관한다.
- 비교/요약 데이터는 자치구, 연도, 면적구간, 임대유형 같은 분석 축별로 별도 집계한다.
