# docs

이 디렉터리는 프로젝트 문서를 한 곳에 모아 관리한다.

## Structure

- `RECOMMENDATION_VALUE_AUDIT.md`
  추천 요약에 쓰이는 값이 원천인지, 파생인지, 시뮬레이션인지 점검한 문서다.
- `SCORING_DATA_LINEAGE.md`
  점수 계산 로직, 스케일링 방식, 데이터 계보를 정리한 문서다.
- `reference/`
  외부 API 가이드, 기술문서 원본을 둔다.

## Rules

- 설계 메모, 감사 문서, 참고 문서는 `docs/` 아래에서만 관리한다.
- 앱 화면에 노출하는 문서는 루트 `docs/` 기준 경로를 사용한다.
