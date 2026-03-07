# cleaned

`cleaned/`는 `raw/` 원천을 정제, 표준화, 집계, 요약해서 만든 파생 CSV를 보관한다. 현재 파일은 하위 폴더로 나뉘지만, 설명 문서는 이 README 하나에서 통합 관리한다.

## Housing

- `housing/apartment_rent_transactions_cleaned.csv`
  아파트 전월세 원천을 컬럼 표준화, 자치구 표준화, 금액 단위 정규화한 정제본이다.
- `housing/apartment_sale_transactions_cleaned.csv`
  아파트 매매 원천을 컬럼 표준화, 자치구 표준화, 거래금액 숫자화한 정제본이다.
- `housing/apartment_rent_avg_by_gu_area_cleaned.csv`
  구별·연도별·면적구간별 평균 전세/월세 자료를 하나로 합친 뒤 컬럼과 금액 단위를 정리한 정제본이다.
- `housing/budget_candidates_by_gu.csv`
  예산 상한 조건을 적용해 전세 예산 충족 가능 구를 요약한 표다.
- `housing/area_band_price_distribution.csv`
  면적구간별 전세/월세/매매 가격 분포를 요약한 표다.
- `housing/jeonse_ratio_by_gu_year_area.csv`
  구·연도·면적구간 기준으로 전세가율을 계산한 표다.
- `housing/jeonse_ratio_ranking_by_gu.csv`
  구별 전세가율을 정렬해 비교하기 위한 요약표다.
- `housing/market_trend_by_gu_year_rent_type.csv`
  구별 연도 추세를 전세/월세 관점에서 요약한 표다.
- `housing/rent_ratio_by_gu_year.csv`
  구별 연도별 전세·월세 비중을 요약한 표다.

## Safety

- `safety/crime_clean.csv`
  범죄 원천 데이터를 long 형태로 변환하고 구명을 표준화한 정제본이다.
- `safety/crime_summary_by_gu.csv`
  `crime_clean.csv`를 구별 합계 기준으로 집계한 요약표다.
- `safety/police_clean.csv`
  경찰 만족도 원천에서 경찰서명을 구명으로 매핑한 정제본이다.
- `safety/police_satisfaction_by_gu.csv`
  `police_clean.csv`를 구별 평균 만족도로 집계한 표다.
- `safety/safety_merged_summary.csv`
  범죄 건수와 경찰 만족도 요약을 결합한 치안 통합표다.

## Infra

- `infra/hospital_csv_clean.csv`
  병원 CSV 계열 원천을 정리하고 구명을 표준화한 정제본이다.
- `infra/hospital_count_by_gu_type.csv`
  병원 데이터를 구·기관유형 기준으로 집계한 표다.
- `infra/hospital_reference_union.csv`
  병원 관련 여러 참조 소스를 합친 비교용 통합표다.
- `infra/parks_clean.csv`
  공원 원천에서 주소/권역 기준 구명을 표준화한 정제본이다.
- `infra/infra_composite_index.csv`
  병원, 공원, 대형마트 관련 집계를 조합해 인프라 비교용 지수 형태로 만든 표다.

## Redevelopment

- `redevelopment/redevelopment_clean.csv`
  정비사업 원천 CSV의 헤더/구명/단계를 정리한 정제본이다.
- `redevelopment/redevelopment_summary.csv`
  정비사업 건수와 단계 수를 구별로 요약한 표다.

## Transform Rules

- 공통 정제 단계
  인코딩 정규화, 컬럼명 정리, 구명 표준화, 금액 숫자화가 먼저 수행된다.
- 거래성 데이터
  원천 row를 유지한 정제본과, 구·연도·면적구간 기준 집계표를 분리해 보관한다.
- 비교/요약 데이터
  시각화와 감사 문서에 쓰기 쉽게 구 단위 집계, 비율 계산, 순위 정렬을 적용한다.
