# tools/scripts

루트 `tools/scripts`는 작업성 스크립트를 한 곳에 모아 관리하는 디렉터리다.

## Files

- `build_deploy_data.py`
  `streamlit` 앱이 읽는 원천 데이터를 집계해 `datasets/deploy/` 경량 배포본을 생성한다.
- `audit_data_all.py`
  `datasets/raw` 아래 CSV를 순회하면서 인코딩, 행 수, 열 수를 점검한다.
- `normalize_data_all_encoding.py`
  CSV 인코딩을 감지해서 필요 시 UTF-8 BOM으로 정규화한다.
- `csv_utils.py`
  `utf-8-sig`, `utf-8`, `cp949`, `euc-kr` 순으로 CSV 인코딩을 시도하는 공용 헬퍼다.
