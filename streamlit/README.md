# Seoul Newlywed Housing Recommendation Dashboard

`streamlit/`은 서울 신혼부부 주거 추천 대시보드 앱 소스 디렉터리다.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Scope

- `app.py`
- `src/`
- `requirements.txt`

## Data Resolution Order

앱은 아래 순서로 데이터를 찾는다.

1. `DATA_DIR` 환경변수
2. 루트 `datasets/raw/`
3. 루트 `datasets/deploy/`

운영 원천 데이터가 없더라도 `datasets/deploy/`만 있으면 경량 모드로 실행할 수 있다.

## Related Directories

- 문서: 루트 `docs/`
- 원천 데이터: 루트 `datasets/raw/`
- 정제 데이터: 루트 `datasets/cleaned/`
- 배포 데이터: 루트 `datasets/deploy/`
- 유틸리티 스크립트: 루트 `tools/scripts/`
