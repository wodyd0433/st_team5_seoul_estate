# datasets

이 디렉터리는 앱에서 사용하는 데이터 자산을 단계별로 관리한다.

## Structure

- `raw/`
  수집 또는 원본 기준 데이터를 둔다.
  거래 데이터는 연도별 파일로 분할되어 있다.
- `cleaned/`
  원천 데이터를 정제하거나 요약해 만든 파생 CSV를 둔다.
- `deploy/`
  Streamlit 앱이 경량 모드에서 직접 읽는 배포용 CSV를 둔다.

## Notes

- `raw/`는 앱의 주 원천 데이터다.
- `cleaned/`는 분석 보조용 산출물이며, 어떤 CSV를 어떻게 만들었는지는 `cleaned/README.md`에 정리한다.
- `deploy/`는 `tools/scripts/build_deploy_data.py`로 갱신하며, 변환 방식은 `deploy/README.md`에 정리한다.
- 대용량 파일은 GitHub 제한 때문에 Git LFS 추적 대상이 될 수 있다.
