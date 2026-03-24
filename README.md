# korea_estate Workspace
aaa
서울 부동산 데이터와 Streamlit 추천 대시보드를 한 저장소에서 관리한다.

## Directory Layout

- `streamlit/`
  Streamlit 앱 소스와 실행 의존성이 있다.
- `datasets/`
  원천, 정제, 배포용 데이터를 용도별로 나눠 관리한다.
- `docs/`
  감사 문서, 점수 산식 문서, 외부 참고 문서를 둔다.
- `tools/scripts/`
  데이터 점검, 인코딩 정리, 배포 데이터 생성 스크립트를 둔다.

## Rules

- 앱 코드는 `streamlit/` 아래에만 둔다.
- 데이터 파일은 `datasets/raw`, `datasets/cleaned`, `datasets/deploy` 아래로만 정리한다.
- 문서는 `docs/`에서 통합 관리한다.
- 대용량 파일은 GitHub 제한을 넘기지 않도록 필요 시 Git LFS로 관리한다.
