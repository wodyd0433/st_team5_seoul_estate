# korea_estate Workspace

서울 부동산 데이터와 추천 대시보드를 유형별로 분리해 관리하는 작업 공간이다.

## Directory Layout

- `streamlit/`
  Streamlit 앱 소스만 둔다.
- `datasets/raw/`
  운영 기준 원천 데이터를 둔다.
- `datasets/cleaned/`
  정제·요약·비교용 파생 CSV를 둔다.
- `datasets/deploy/`
  대시보드 배포용 경량 데이터를 둔다.
- `docs/`
  계획서, 감사 문서, 참고 문서를 한 곳에서 관리한다.
- `tools/scripts/`
  데이터 점검, 인코딩 정리, 배포용 데이터 생성 스크립트가 있다.
- `artifacts/`
  분석 중간 산출물이나 임시 결과물을 둔다.

## Rules

- 앱 코드는 `streamlit/` 아래에만 둔다.
- 운영 기준 데이터는 루트 `datasets/raw/`를 우선 사용한다.
- 일회성 결과물은 `artifacts/`에만 두고, 운영 코드가 직접 참조하지 않게 한다.
- 데이터 파일은 `datasets/raw`, `datasets/cleaned`, `datasets/deploy` 아래로만 정리한다.
- 작업 스크립트는 앱 내부에 흩뿌리지 않고 `tools/scripts/`로 모은다.
