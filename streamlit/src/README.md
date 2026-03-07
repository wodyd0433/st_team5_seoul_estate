# streamlit/src

이 디렉터리는 Streamlit 앱의 내부 로직 모듈을 둔다.

## Files

- `config.py`
  데이터 경로, 구 목록, 허브 좌표, 상수 설정을 관리한다.
- `io_utils.py`
  raw/deploy 데이터를 읽고 기본 전처리를 수행한다.
- `feature_engineering.py`
  구 단위 피처와 지표를 계산한다.
- `scoring_engine.py`
  예산, 통근, 치안, 인프라 점수를 결합해 추천 점수를 계산한다.
- `persona.py`
  페르소나 생성과 시뮬레이션 로직을 관리한다.
- `visualization.py`
  차트와 요약 시각화 출력을 만든다.
- `gu_standardizer.py`
  서울 자치구 명칭을 표준화한다.
- `unit_detection.py`
  가격 단위와 금액 컬럼을 감지하고 정규화한다.
- `cleaning.py`
  이상치 제거 등 보조 정제 함수를 제공한다.
