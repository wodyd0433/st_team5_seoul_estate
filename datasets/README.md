# datasets

이 디렉터리는 앱에서 사용하는 데이터 자산을 단계별로 관리한다.

## Structure

- `raw/`
  수집 또는 원본 기준 데이터를 둔다.
  거래 데이터는 연도별 파일로 분할되어 있다.
- `cleaned/`
  정제, 비교, 요약 목적의 파생 데이터를 둔다.
  현재는 `housing`, `safety`, `infra`, `redevelopment`로 나뉜다.
- `deploy/`
  Streamlit 앱이 경량 모드에서 바로 읽는 배포용 CSV를 둔다.

## Notes

- `raw/`는 앱의 주 원천 데이터다.
- `cleaned/`는 분석 보조용 산출물이며 앱이 직접 필수로 읽는 구조는 아니다.
- `deploy/`는 `tools/scripts/build_deploy_data.py`로 갱신한다.
- 대용량 파일은 GitHub 제한 때문에 Git LFS 추적 대상이 될 수 있다.
