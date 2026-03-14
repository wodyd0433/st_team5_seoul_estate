# tools/scripts

루트 `tools/scripts/`는 데이터 관리용 작업 스크립트를 모아 두는 디렉터리다.

## Files

- `build_deploy_data.py`
  `datasets/raw` 원천 데이터를 읽어 `datasets/deploy` 경량 배포본을 생성한다.
- `audit_data_all.py`
  `datasets/raw` 아래 파일의 인코딩, 행 수, 열 수를 점검한다.
- `normalize_data_all_encoding.py`
  `datasets/raw` CSV 인코딩을 감지하고 UTF-8 BOM 기준으로 정규화한다.
- `csv_utils.py`
  CSV 인코딩 후보를 순차적으로 시도하는 공용 헬퍼다.
- `collect_kosis_table.py`
  KOSIS OpenAPI 표를 JSON으로 받아 `datasets/raw`용 CSV로 저장한다.
