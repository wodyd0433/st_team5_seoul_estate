from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import DATA_DIR, DATA_DIR_CANDIDATES, PROJECT_ROOT, WORKPLACE_HUBS
from src.feature_engineering import build_feature_table
from src.io_utils import load_dataset_bundle
from src.persona import build_income_percentile_reference, build_persona_simulation
from src.scoring_engine import prepare_commute_frame, score_recommendations
from src.visualization import (
    build_commute_timeseries_chart,
    build_recommendation_map,
    build_recommendation_summary,
    build_short_reco_label,
    build_top_rank_chart,
    build_visualization_gallery,
    format_korean_money,
)


PAGE_TITLE = "서울 신혼부부 전월세·매매 추천 대시보드"
DOC_DIR = PROJECT_ROOT / "docs"
PROJECT_README_PATH = PROJECT_ROOT / "README.md"
DOCS_README_PATH = DOC_DIR / "README.md"
RECOMMENDATION_AUDIT_PATH = DOC_DIR / "RECOMMENDATION_VALUE_AUDIT.md"
SCORING_LINEAGE_PATH = DOC_DIR / "SCORING_DATA_LINEAGE.md"
HOUSEHOLD_OPTIONS = ["1인", "2인 맞벌이"]
AREA_BAND_OPTIONS = {
    "10평대": (10, 19),
    "20평대": (20, 29),
    "30평대": (30, 39),
    "40평대+": (40, 45),
}
WEIGHT_PRESET_OPTIONS = {
    "균형 모드": {"price": 30, "commute": 25, "infra": 20, "safety": 15, "risk": 10},
    "통근 최우선": {"price": 20, "commute": 40, "infra": 15, "safety": 15, "risk": 10},
    "치안/안전 중심": {"price": 20, "commute": 20, "infra": 15, "safety": 35, "risk": 10},
    "인프라 중심": {"price": 20, "commute": 20, "infra": 35, "safety": 15, "risk": 10},
}
WEIGHT_MODE_OPTIONS = [*WEIGHT_PRESET_OPTIONS.keys(), "상세 설정"]


st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #020814 0%, #04111f 100%);
        color: #f7f4ed;
    }
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
    }
    .section-title {
        font-size: 1.7rem;
        font-weight: 800;
        margin: 0.25rem 0 1rem 0;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.12rem 0.42rem;
        margin-right: 0.28rem;
        border-radius: 0.4rem;
        background: rgba(34, 197, 94, 0.18);
        color: #86efac;
        font-size: 0.86rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_markdown(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        return f"문서를 읽지 못했습니다: `{path.name}`\n\n```text\n{exc}\n```"


def _apply_persona_defaults(persona_row: pd.Series) -> None:
    deposit_slider_min = 100_000_000
    deposit_slider_max = 1_500_000_000
    monthly_slider_min = 300_000
    monthly_slider_max = 4_000_000

    deposit_max = min(int(persona_row["deposit_budget_cap_krw"]), deposit_slider_max)
    monthly_max = min(int(persona_row["monthly_budget_cap_krw"]), monthly_slider_max)
    deposit_min = max(deposit_slider_min, min(int(deposit_max * 0.6), deposit_max))
    monthly_min = max(monthly_slider_min, min(int(monthly_max * 0.6), monthly_max))

    st.session_state["budget_cap_range"] = (deposit_min, deposit_max)
    st.session_state["monthly_budget_range"] = (monthly_min, monthly_max)
    st.session_state["cash_assets_krw"] = 100_000_000
    st.session_state["saving_ratio_pct"] = 50.0
    st.session_state["use_deposit_budget_filter"] = False
    st.session_state["use_monthly_budget_filter"] = False


def _get_applied_weights() -> dict[str, int]:
    if "applied_weights_pct" not in st.session_state:
        st.session_state["weight_mode_select"] = "균형 모드"
        st.session_state["last_weight_preset_name"] = "균형 모드"
        st.session_state["applied_weights_pct"] = dict(WEIGHT_PRESET_OPTIONS["균형 모드"])
    return dict(st.session_state["applied_weights_pct"])


def _sync_weight_preset(selected_mode: str) -> dict[str, int]:
    applied_weights_pct = _get_applied_weights()
    previous_mode = st.session_state.get("last_weight_preset_name")

    if selected_mode != previous_mode and selected_mode in WEIGHT_PRESET_OPTIONS:
        st.session_state["applied_weights_pct"] = dict(WEIGHT_PRESET_OPTIONS[selected_mode])
        applied_weights_pct = dict(st.session_state["applied_weights_pct"])

    st.session_state["last_weight_preset_name"] = selected_mode
    return applied_weights_pct


def _build_income_reference(bundle: dict[str, object]) -> dict[str, object] | None:
    income_df = bundle.get("income_debt_distribution")
    if not isinstance(income_df, pd.DataFrame) or income_df.empty:
        return None
    try:
        return build_income_percentile_reference(income_df)
    except Exception:
        return None


def _persona_scale_factor(household_type: str) -> int:
    return 2 if str(household_type).startswith("2") else 1


def _scale_persona_row(persona_row: pd.Series | None, household_type: str) -> pd.Series | None:
    if persona_row is None:
        return None
    factor = _persona_scale_factor(household_type)
    if factor == 1:
        return persona_row

    scaled = persona_row.copy()
    scale_columns = [
        "monthly_income_estimate_krw",
        "annual_income_estimate_krw",
        "debt_balance_estimate_krw",
        "monthly_debt_service_estimate_krw",
        "monthly_living_cost_estimate_krw",
        "monthly_saving_estimate_krw",
        "current_seed_estimate_krw",
        "seed_money_2y_krw",
        "seed_money_3y_krw",
        "buying_power_2y_krw",
        "buying_power_3y_krw",
        "deposit_budget_cap_krw",
        "monthly_budget_cap_krw",
        "income_p25_annual_krw",
        "income_p50_annual_krw",
        "income_p75_annual_krw",
        "debt_p25_krw",
        "debt_p50_krw",
        "debt_p75_krw",
    ]
    for column in scale_columns:
        if column in scaled.index and pd.notna(scaled[column]):
            scaled[column] = float(scaled[column]) * factor

    scaled["persona_summary"] = f"{scaled['persona_summary']} ({household_type} 기준 {factor}배 적용)"
    return scaled


def _scale_income_reference(income_reference: dict[str, object] | None, household_type: str) -> dict[str, object] | None:
    if income_reference is None:
        return None
    factor = _persona_scale_factor(household_type)
    if factor == 1:
        return income_reference

    scaled = dict(income_reference)
    for key in [
        "p25_annual_krw",
        "p50_annual_krw",
        "p75_annual_krw",
        "p25_monthly_krw",
        "p50_monthly_krw",
        "p75_monthly_krw",
    ]:
        if key in scaled and pd.notna(scaled[key]):
            scaled[key] = float(scaled[key]) * factor
    return scaled


def _resolve_persona_income_band(persona_row: pd.Series | None, income_reference: dict[str, object] | None) -> str | None:
    if persona_row is None:
        return None
    segment = persona_row.get("income_segment_label")
    percentile = persona_row.get("income_percentile")
    if pd.notna(segment) and pd.notna(percentile):
        return f"{segment} (P{int(percentile)})"
    return None


def _ensure_persona_state(persona_profiles: pd.DataFrame, household_type: str) -> pd.Series | None:
    if persona_profiles.empty:
        return None

    st.session_state.setdefault("persona_name", "중간소득 고부채")
    persona_options = persona_profiles["persona_name"].tolist()
    selected_persona = st.sidebar.selectbox("페르소나 선택", persona_options, key="persona_name")
    persona_row = persona_profiles.loc[persona_profiles["persona_name"].eq(selected_persona)].iloc[0]
    persona_row = _scale_persona_row(persona_row, household_type)
    persona_state_key = f"{selected_persona}|{household_type}"

    if st.session_state.get("last_persona_state_key") != persona_state_key:
        _apply_persona_defaults(persona_row)
        st.session_state["last_persona_name"] = selected_persona
        st.session_state["last_persona_state_key"] = persona_state_key

    if st.sidebar.button("페르소나 기본값 다시 적용", use_container_width=True):
        _apply_persona_defaults(persona_row)

    return persona_row


def _resolve_available_years(bundle: dict[str, object]) -> list[int]:
    years: set[int] = set()

    compact_housing = bundle.get("compact_feature_base")
    if isinstance(compact_housing, pd.DataFrame) and "year" in compact_housing.columns:
        years.update(pd.to_numeric(compact_housing["year"], errors="coerce").dropna().astype(int).tolist())

    rent = bundle.get("rent")
    if isinstance(rent, pd.DataFrame) and "년월" in rent.columns:
        rent_years = pd.to_numeric(rent["년월"].astype(str).str[:4], errors="coerce").dropna().astype(int)
        years.update(rent_years.tolist())

    sale = bundle.get("sale")
    if isinstance(sale, pd.DataFrame) and "dealYear" in sale.columns:
        years.update(pd.to_numeric(sale["dealYear"], errors="coerce").dropna().astype(int).tolist())

    yearly_rent = bundle.get("yearly_rent")
    if isinstance(yearly_rent, dict):
        years.update(int(year) for year in yearly_rent.keys())

    return sorted(years, reverse=True) or [2025]


def _build_rent_distribution_chart(feature_table: pd.DataFrame):
    rent_cols = ["deposit_price_krw", "monthly_rent_active_krw"]
    available = [col for col in rent_cols if col in feature_table.columns]
    if not available:
        return None

    rent_view = feature_table[["gu", *available]].melt(
        id_vars="gu",
        value_vars=available,
        var_name="value_type",
        value_name="value_krw",
    )
    rent_view["value_krw"] = pd.to_numeric(rent_view["value_krw"], errors="coerce")
    rent_view = rent_view.dropna(subset=["value_krw"])
    if rent_view.empty:
        return None

    rent_view["value_type"] = rent_view["value_type"].map(
        {
            "deposit_price_krw": "전세보증금",
            "monthly_rent_active_krw": "월세",
        }
    )
    fig = px.histogram(
        rent_view,
        x="value_krw",
        color="value_type",
        facet_col="value_type",
        nbins=16,
        barmode="overlay",
        color_discrete_map={"전세보증금": "#8ec5fc", "월세": "#ffb3c1"},
        title="전세보증금·월세 분포",
        labels={"value_krw": "금액(원)", "count": "자치구 수", "value_type": "항목"},
    )
    fig.update_xaxes(tickformat=",")
    fig.update_layout(showlegend=False, margin={"l": 20, "r": 20, "t": 48, "b": 20})
    return fig


def _build_score_distribution_chart(recommendations: pd.DataFrame):
    score_cols = ["price_score", "commute_score", "infra_score", "safety_score", "risk_score"]
    available = [col for col in score_cols if col in recommendations.columns]
    if not available:
        return None

    score_view = recommendations[["gu", *available]].melt(
        id_vars="gu",
        value_vars=available,
        var_name="score_type",
        value_name="score",
    )
    score_view["score"] = pd.to_numeric(score_view["score"], errors="coerce")
    score_view = score_view.dropna(subset=["score"])
    if score_view.empty:
        return None

    score_view["score_type"] = score_view["score_type"].map(
        {
            "price_score": "가격 점수",
            "commute_score": "통근 점수",
            "infra_score": "인프라 점수",
            "safety_score": "치안 점수",
            "risk_score": "전세가율 점수",
        }
    )
    fig = px.box(
        score_view,
        x="score_type",
        y="score",
        color="score_type",
        points="all",
        title="영역별 점수 분포(0~100)",
        labels={"score_type": "점수 영역", "score": "점수"},
    )
    fig.update_layout(showlegend=False, margin={"l": 20, "r": 20, "t": 48, "b": 20})
    return fig


def _get_commute_timeseries(bundle: dict[str, object]) -> pd.DataFrame:
    frame = bundle.get("commute_timeseries")
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    return pd.DataFrame(columns=["destination_name", "gu", "time_order", "time_label", "avg_minutes"])


def _select_top_commute_gus(frame: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    if frame.empty:
        return frame
    top_gus = (
        frame.groupby("gu", as_index=False)
        .agg(overall_avg_minutes=("avg_minutes", "mean"))
        .sort_values("overall_avg_minutes", ascending=False)
        .head(top_n)["gu"]
        .tolist()
    )
    return frame.loc[frame["gu"].isin(top_gus)].copy()


def _show_intro(bundle: dict[str, object]) -> None:
    st.title(PAGE_TITLE)
    with st.popover("대시보드 기준 설명"):
        st.markdown(
            """
            - `가격 점수`: 전세가 중위값을 5단계 별점으로 환산합니다.
            - `통근 점수`: 예측 통근시간을 20/30/45/60분 기준으로 평가합니다.
            - `인프라 점수`: 대형마트, 병원, 공원 조합을 규칙 기반으로 평가합니다.
            - `치안 점수`: 범죄 발생량과 경찰 만족도를 함께 봅니다.
            - `전세가율 점수`: 전세가율이 높을수록 리스크가 커집니다.
            - `가중치`: 가격, 통근, 인프라, 치안, 전세가율 비율을 직접 입력하고 100% 합계로 확정합니다.
            - `페르소나`: 예산 한도와 향후 매수 시뮬레이션에만 사용합니다.
            
            **데이터 출처:**
            - `원천 거래`: 국토교통부 실거래가 오픈 API
            - `생활 인프라`: 서울시 열린데이터광장 (병원, 공원, 마트)
            - `치안/안전`: 경찰청 범죄통계 및 치안고객만족도 조사 결과
            - `통근/입지`: 카카오맵 API 기반 거리 및 예상 소요 시간 시뮬레이션
            """
        )


    if bundle.get("data_mode") == "compact":
        st.caption("현재 `datasets/deploy` 기반 경량 배포 모드로 실행 중입니다.")


def _show_data_load_error(exc: Exception) -> None:
    st.error("데이터 파일을 불러오지 못했습니다.")
    st.markdown(
        "\n".join(
            [
                f"- 현재 데이터 루트: `{DATA_DIR}`",
                "- 우선 탐색 경로:",
                *[f"  - `{candidate}`" for candidate in DATA_DIR_CANDIDATES],
            ]
        )
    )
    st.code(str(exc))


def _collect_sidebar_inputs(bundle: dict[str, object]) -> tuple[pd.Series | None, dict[str, object]]:
    available_years = _resolve_available_years(bundle)
    applied_weights_pct = _get_applied_weights()
    st.session_state.setdefault("budget_cap_range", (400_000_000, 700_000_000))
    st.session_state.setdefault("monthly_budget_range", (500_000, 1_000_000))
    st.session_state.setdefault("weight_mode_select", "균형 모드")
    st.session_state.setdefault("cash_assets_krw", 100_000_000)
    st.session_state.setdefault("saving_ratio_pct", 50.0)
    st.session_state.setdefault("use_deposit_budget_filter", False)
    st.session_state.setdefault("use_monthly_budget_filter", False)

    year_index = available_years.index(2025) if 2025 in available_years else 0
    selected_year = st.sidebar.selectbox("기준 연도", available_years, index=year_index)
    household_type = st.sidebar.selectbox("가구 유형", HOUSEHOLD_OPTIONS, index=1)
    persona_row = _ensure_persona_state(bundle.get("persona_profiles", pd.DataFrame()), household_type)
    workplace_options = list(WORKPLACE_HUBS.keys())
    primary_workplace_index = workplace_options.index("강남역") if "강남역" in workplace_options else 0
    workplace_name = st.sidebar.selectbox("직장 위치 1", workplace_options, index=primary_workplace_index)

    secondary_workplace_name = None
    if household_type.startswith("2"):
        secondary_workplace_index = workplace_options.index("여의도역") if "여의도역" in workplace_options else 0
        secondary_workplace_name = st.sidebar.selectbox(
            "직장 위치 2",
            workplace_options,
            index=secondary_workplace_index,
        )

    area_band = st.sidebar.segmented_control("평형대", list(AREA_BAND_OPTIONS.keys()), default="20평대")
    min_area_pyeong, max_area_pyeong = st.sidebar.slider(
        "희망 평수 구간",
        10,
        45,
        AREA_BAND_OPTIONS[area_band],
        step=1,
    )

    use_deposit_budget_filter = st.sidebar.checkbox(
        "전세보증금 예산 구간 설정",
        value=bool(st.session_state.get("use_deposit_budget_filter", False)),
        key="use_deposit_budget_filter",
    )
    if use_deposit_budget_filter:
        deposit_budget_min, deposit_budget_max = st.sidebar.slider(
            "전세보증금 예산 구간",
            100_000_000,
            1_500_000_000,
            key="budget_cap_range",
            step=50_000_000,
        )
        st.sidebar.caption(
            f"전세 선택 구간: {format_korean_money(deposit_budget_min)} ~ {format_korean_money(deposit_budget_max)}"
        )
    else:
        deposit_budget_min, deposit_budget_max = 100_000_000, 1_500_000_000

    use_monthly_budget_filter = st.sidebar.checkbox(
        "월세 예산 구간 설정",
        value=bool(st.session_state.get("use_monthly_budget_filter", False)),
        key="use_monthly_budget_filter",
    )
    if use_monthly_budget_filter:
        monthly_budget_min, monthly_budget_max = st.sidebar.slider(
            "월세 예산 구간",
            300_000,
            4_000_000,
            key="monthly_budget_range",
            step=100_000,
        )
        st.sidebar.caption(
            f"월세 선택 구간: {format_korean_money(monthly_budget_min)} ~ {format_korean_money(monthly_budget_max)}"
        )
    else:
        monthly_budget_min, monthly_budget_max = 300_000, 4_000_000
    cash_assets_krw = st.sidebar.number_input(
        "현금자산",
        min_value=0,
        value=int(st.session_state.get("cash_assets_krw", 0)),
        step=10_000_000,
        key="cash_assets_krw",
    )
    saving_ratio_pct = st.sidebar.number_input(
        "저축비율(소득 대비, %)",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state.get("saving_ratio_pct", 10.0)),
        step=1.0,
        key="saving_ratio_pct",
    )
    st.sidebar.markdown("#### 가중치 설정")
    selected_weight_mode = st.sidebar.selectbox("모드 선택", WEIGHT_MODE_OPTIONS, key="weight_mode_select")
    applied_weights_pct = _sync_weight_preset(selected_weight_mode)

    price_weight = applied_weights_pct["price"]
    commute_weight = applied_weights_pct["commute"]
    infra_weight = applied_weights_pct["infra"]
    safety_weight = applied_weights_pct["safety"]
    risk_weight = applied_weights_pct["risk"]
    submitted = False
    if selected_weight_mode == "상세 설정":
        with st.sidebar.form("weight_form"):
            price_weight = st.number_input("가격(%)", min_value=0, max_value=100, value=applied_weights_pct["price"], step=1)
            commute_weight = st.number_input("통근(%)", min_value=0, max_value=100, value=applied_weights_pct["commute"], step=1)
            infra_weight = st.number_input("인프라(%)", min_value=0, max_value=100, value=applied_weights_pct["infra"], step=1)
            safety_weight = st.number_input("치안(%)", min_value=0, max_value=100, value=applied_weights_pct["safety"], step=1)
            risk_weight = st.number_input("전세가율(%)", min_value=0, max_value=100, value=applied_weights_pct["risk"], step=1)
            submitted = st.form_submit_button("상세 가중치 적용", use_container_width=True)

    pending_total = int(price_weight + commute_weight + infra_weight + safety_weight + risk_weight)
    if selected_weight_mode == "상세 설정" and submitted:
        if pending_total == 100:
            st.session_state["applied_weights_pct"] = {
                "price": int(price_weight),
                "commute": int(commute_weight),
                "infra": int(infra_weight),
                "safety": int(safety_weight),
                "risk": int(risk_weight),
            }
            applied_weights_pct = dict(st.session_state["applied_weights_pct"])
            st.sidebar.success("가중치를 반영했습니다.")
        else:
            st.sidebar.error(f"합계가 {pending_total}%입니다. 100%로 맞춘 뒤 확정해야 합니다.")

    st.sidebar.caption(
        f"적용 중: 가격 {applied_weights_pct['price']} / 통근 {applied_weights_pct['commute']} / "
        f"인프라 {applied_weights_pct['infra']} / 치안 {applied_weights_pct['safety']} / 전세가율 {applied_weights_pct['risk']} (%)"
    )

    state = {
        "selected_year": selected_year,
        "household_type": household_type,
        "workplace_name": workplace_name,
        "secondary_workplace_name": secondary_workplace_name,
        "min_area_pyeong": min_area_pyeong,
        "max_area_pyeong": max_area_pyeong,
        "min_area_m2": round(min_area_pyeong * 3.3058, 1),
        "max_area_m2": round(max_area_pyeong * 3.3058, 1),
        "deposit_budget_min_krw": deposit_budget_min,
        "deposit_budget_max_krw": deposit_budget_max,
        "monthly_budget_min_krw": monthly_budget_min,
        "monthly_budget_max_krw": monthly_budget_max,
        "use_deposit_budget_filter": use_deposit_budget_filter,
        "use_monthly_budget_filter": use_monthly_budget_filter,
        "cash_assets_krw": cash_assets_krw,
        "saving_ratio_pct": saving_ratio_pct,
        "budget_cap": deposit_budget_max,
        "monthly_budget_cap": monthly_budget_max,
        "weights": {key: value / 100 for key, value in applied_weights_pct.items()},
        "weights_pct": applied_weights_pct,
        "weight_mode": selected_weight_mode,
    }
    return persona_row, state


def _filter_budget_ranges(feature_table: pd.DataFrame, ui: dict[str, object]) -> tuple[pd.DataFrame, str | None]:
    if not ui.get("use_deposit_budget_filter") and not ui.get("use_monthly_budget_filter"):
        return feature_table, None

    deposit_mask = pd.to_numeric(feature_table["deposit_price_krw"], errors="coerce").between(
        ui["deposit_budget_min_krw"],
        ui["deposit_budget_max_krw"],
        inclusive="both",
    )
    monthly_mask = pd.to_numeric(feature_table["monthly_rent_active_krw"], errors="coerce").between(
        ui["monthly_budget_min_krw"],
        ui["monthly_budget_max_krw"],
        inclusive="both",
    )
    combined_mask = pd.Series(True, index=feature_table.index)
    if ui.get("use_deposit_budget_filter"):
        combined_mask &= deposit_mask
    if ui.get("use_monthly_budget_filter"):
        combined_mask &= monthly_mask

    filtered = feature_table.loc[combined_mask].copy()
    if filtered.empty:
        return (
            feature_table,
            "선택한 전세보증금/월세 예산 구간에 정확히 맞는 자치구가 없어 전체 후보를 그대로 표시합니다.",
        )
    return filtered, None


def _render_scoring_thresholds_guide() -> None:
    with st.expander("점수 산정 기준 및 Threshold", expanded=False):
        st.markdown(
            """
            **0~100점 환산 방식:**

            - 각 항목은 `5점 척도`로 먼저 평가한 뒤, `별점 x 20`으로 `0~100점`으로 환산합니다.
            - 예: 5점=100점, 4점=80점, 3점=60점, 2점=40점, 1점=20점
            """
        )
        st.markdown(
            """
            **항목별 Threshold:**

            - <span class="metric-badge">가격 점수</span> 전세보증금 기준
              3.5억 미만=100점, 5억 미만=80점, 7억 미만=60점, 9.5억 미만=40점, 그 외=20점
            - <span class="metric-badge">통근 점수</span> 통근시간 기준
              20분 이하=100점, 30분 이하=80점, 45분 이하=60점, 60분 이하=40점, 초과=20점
            - <span class="metric-badge">인프라 점수</span> `마트*2 + 병원*1.5 + 공원` 지수 기준
              3개 이상 마트+2개 이상 병원+공원 상위 30%=100점
              2개 이상 마트+병원 상위 30%+공원 상위 30%=80점
              1개 이상 마트+인프라지수 중앙값 이상=60점
              인프라지수 하위 20% 초과=40점, 그 외=20점
            - <span class="metric-badge">치안 점수</span> 범죄건수와 경찰만족도 기준
              범죄 하위 10% 이내이면서 경찰만족도 상위 10%=100점
              범죄 하위 25%=80점
              범죄 하위 75%=60점
              범죄 하위 90%=40점, 그 외=20점
            - <span class="metric-badge">전세가율 점수</span> 전세가율 기준
              50% 미만=100점, 60% 미만=80점, 70% 미만=60점, 80% 미만=40점, 그 외=20점
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            **종합점수 산정식:**

            - `종합점수 = (가격별점×가중치 + 통근별점×가중치 + 인프라별점×가중치 + 치안별점×가중치 + 전세가율별점×가중치) × 20`
            - 가중치는 사이드바 `가중치 설정`에서 선택한 프리셋 또는 상세 설정값을 그대로 사용합니다.
            """
        )
        st.markdown(
            """
            **종합별점 / 등급 Threshold:**

            - <span class="metric-badge">S / ★★★★★</span> 90점 이상, 강력 추천
            - <span class="metric-badge">A / ★★★★☆</span> 75점 이상 90점 미만, 우수
            - <span class="metric-badge">B / ★★★☆☆</span> 55점 이상 75점 미만, 무난
            - <span class="metric-badge">C / ★★☆☆☆</span> 35점 이상 55점 미만, 주의
            - <span class="metric-badge">D / ★☆☆☆☆</span> 35점 미만, 비추천
            """,
            unsafe_allow_html=True,
        )


def _compute_outputs(bundle: dict[str, object], ui: dict[str, object]) -> dict[str, object]:
    feature_table, feature_meta = build_feature_table(
        bundle=bundle,
        year=ui["selected_year"],
        sampling_rate=1.0,
        budget_cap=ui["budget_cap"],
        remove_outliers=True,
        monthly_budget_cap=ui["monthly_budget_cap"],
        min_area_pyeong=ui["min_area_pyeong"],
        max_area_pyeong=ui["max_area_pyeong"],
    )
    filtered_feature_table, filter_notice = _filter_budget_ranges(feature_table, ui)
    commute_frame, commute_meta = prepare_commute_frame(
        ui["workplace_name"],
        filtered_feature_table,
        bundle["commute_models"],
        bundle.get("commute_weighted_avg"),
        household_type=ui["household_type"],
        secondary_workplace_name=ui["secondary_workplace_name"],
    )
    recommendations, scoring_meta = score_recommendations(
        feature_table=filtered_feature_table,
        selected_gus=[],
        commute_frame=commute_frame,
        weights=ui["weights"],
        missing_strategy="mean",
        household_type=ui["household_type"],
    )

    return {
        "feature_table": filtered_feature_table,
        "feature_meta": feature_meta,
        "commute_frame": commute_frame,
        "commute_meta": commute_meta,
        "recommendations": recommendations,
        "scoring_meta": scoring_meta,
        "filter_notice": filter_notice,
    }


def _render_summary_tab(
    recommendations: pd.DataFrame,
    recommendation_summary: pd.DataFrame,
    recommendation_map,
    rank_chart,
    persona_row: pd.Series | None,
    ui: dict[str, object],
    income_reference: dict[str, object] | None,
    filter_notice: str | None,
) -> None:
    st.markdown('<div class="section-title">추천 요약</div>', unsafe_allow_html=True)
    if filter_notice:
        st.info(filter_notice)
    if persona_row is not None:
        persona_income_band = _resolve_persona_income_band(persona_row, income_reference)
        st.caption(
            f"선택 페르소나: {persona_row['persona_name']} | "
            f"월소득 추정 {format_korean_money(persona_row['monthly_income_estimate_krw'])} | "
            f"부채 추정 {format_korean_money(persona_row['debt_balance_estimate_krw'])}"
        )
        if persona_income_band:
            debt_segment_label = persona_row.get("debt_segment_label", "-")
            debt_percentile = persona_row.get("debt_percentile")
            debt_percentile_text = f"P{int(debt_percentile)}" if pd.notna(debt_percentile) else "-"
            st.caption(
                f"소득 구간 판정: {persona_income_band} / "
                f"금융권 대출 구간 판정: {debt_segment_label} ({debt_percentile_text})"
            )
        with st.expander("페르소나 분류 기준 및 출처", expanded=False):
            if income_reference is not None:
                debt_p25 = persona_row.get("debt_p25_krw")
                debt_p50 = persona_row.get("debt_p50_krw")
                debt_p75 = persona_row.get("debt_p75_krw")
                debt_p25_label = persona_row.get("debt_p25_label", "-")
                debt_p50_label = persona_row.get("debt_p50_label", "-")
                debt_p75_label = persona_row.get("debt_p75_label", "-")
                income_segment_label = persona_row.get("income_segment_label", "-")
                reference_year = persona_row.get("reference_year", income_reference["latest_year"])
                st.markdown(
                    f"""
                    **소득 구간 기준 (전국, {income_reference['latest_year']}년 연소득 분포):**
                    - `저소득`: {format_korean_money(income_reference['p25_annual_krw'])}, `P25`, `{income_reference['p25_income_label']}`
                    - `중간소득`: {format_korean_money(income_reference['p50_annual_krw'])}, `P50`, `{income_reference['p50_income_label']}`
                    - `고소득`: {format_korean_money(income_reference['p75_annual_krw'])}, `P75`, `{income_reference['p75_income_label']}`

                    **금융권 대출 기준 ({income_segment_label}, {reference_year}년):**
                    - `저부채`: {format_korean_money(debt_p25)}, `P25`, `{debt_p25_label}`
                    - `중간부채`: {format_korean_money(debt_p50)}, `P50`, `{debt_p50_label}`
                    - `고부채`: {format_korean_money(debt_p75)}, `P75`, `{debt_p75_label}`

                    **표시 기준:**
                    - `소득 저/중/고`: `DT_1NW1036` 전체 소득 분포의 `P25 / P50 / P75` 대표값
                    - `금융권 대출 저/중/고`: 선택된 소득구간 내부 금융권 대출 분포의 `P25 / P50 / P75` 대표값
                    - `페르소나 판정`: 소득 percentile 3단계 x 해당 소득구간 내 금융권 대출 percentile 3단계 조합
                    """
                )
            st.markdown(
                """
                **데이터 출처:**
                - `원천 데이터`: 통계청(kosis) `DT_1NW1036`
                - `항목 상세`: 소득(근로·사업소득) 구간과 금융권 대출잔액 구간별 신혼부부 분포
                """
            )


    st.caption(
        f"적용 가중치: 가격 {ui['weights_pct']['price']}% / 통근 {ui['weights_pct']['commute']}% / "
        f"인프라 {ui['weights_pct']['infra']}% / 치안 {ui['weights_pct']['safety']}% / 전세가율 {ui['weights_pct']['risk']}%"
    )
    deposit_caption = (
        f"{format_korean_money(ui['deposit_budget_min_krw'])} ~ {format_korean_money(ui['deposit_budget_max_krw'])}"
        if ui.get("use_deposit_budget_filter")
        else "미설정"
    )
    monthly_caption = (
        f"{format_korean_money(ui['monthly_budget_min_krw'])} ~ {format_korean_money(ui['monthly_budget_max_krw'])}"
        if ui.get("use_monthly_budget_filter")
        else "미설정"
    )
    st.caption(f"예산 구간: 전세 {deposit_caption} / 월세 {monthly_caption}")

    if recommendations.empty:
        st.warning("현재 조건에 맞는 추천 결과가 없습니다.")
        return

    top_cards = recommendations.head(5).copy()
    card_cols = st.columns(5)
    for idx, (_, row) in enumerate(top_cards.iterrows()):
        with card_cols[idx]:
            with st.container(border=True):
                st.markdown(f"**TOP {idx + 1}**")
                st.markdown(f"### {row['gu']}")
                st.markdown(f"## {row['total_score']:.1f}점")
                st.caption(f"{row['total_grade']}등급 · {row['total_star_label']}")
                label_persona = persona_row["persona_name"] if persona_row is not None else "기본"
                st.write(build_short_reco_label(row, label_persona))
                st.text(
                    f"{ui['min_area_m2']}~{ui['max_area_m2']}㎡ / "
                    f"{ui['min_area_pyeong']}~{ui['max_area_pyeong']}평 기준"
                )
                st.write(f"전세 보증금 {format_korean_money(row['deposit_price_krw'])}")
                st.write(f"월세 {format_korean_money(row['monthly_rent_active_krw'])}")
                st.write(f"전세가율 {row['jeonse_ratio_pct']:.1f}%")
                if ui["household_type"].startswith("2") and pd.notna(row.get("secondary_commute_minutes")):
                    st.write(f"직장1 통근 {row['primary_commute_minutes']:.1f}분")
                    st.write(f"직장2 통근 {row['secondary_commute_minutes']:.1f}분")
                else:
                    st.write(f"통근 {row['commute_minutes']:.1f}분")
                if row.get("risk_warning"):
                    st.warning(row["risk_warning"])

    left, right = st.columns([1.05, 1])
    with left:
        st.pydeck_chart(recommendation_map, width="stretch")
    with right:
        st.plotly_chart(rank_chart, width="stretch")
    st.dataframe(recommendation_summary, width="stretch", height=320)


def _render_compare_tab(recommendations: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">구별 상세 비교</div>', unsafe_allow_html=True)
    if recommendations.empty:
        st.warning("현재 조건에 맞는 비교 대상이 없습니다.")
        return
    _render_scoring_thresholds_guide()
    compare_cols = [
        "gu",
        "total_score",
        "total_grade",
        "total_star_label",
        "selected_area_min_m2",
        "selected_area_max_m2",
        "selected_area_min_pyeong",
        "selected_area_max_pyeong",
        "deposit_price_krw",
        "monthly_rent_active_krw",
        "jeonse_ratio_pct",
        "price_score",
        "commute_score",
        "infra_score",
        "safety_score",
        "risk_score",
        "primary_commute_minutes",
        "secondary_commute_minutes",
        "worst_commute_minutes",
    ]
    compare = recommendations[compare_cols].copy().head(15)
    compare["면적"] = (
        compare["selected_area_min_m2"].round(1).astype(str)
        + "~"
        + compare["selected_area_max_m2"].round(1).astype(str)
        + "㎡ / "
        + compare["selected_area_min_pyeong"].astype(int).astype(str)
        + "~"
        + compare["selected_area_max_pyeong"].astype(int).astype(str)
        + "평"
    )
    compare["deposit_price_krw"] = compare["deposit_price_krw"].map(format_korean_money)
    compare["monthly_rent_active_krw"] = compare["monthly_rent_active_krw"].map(format_korean_money)
    compare["jeonse_ratio_pct"] = compare["jeonse_ratio_pct"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
    compare = compare.drop(
        columns=["selected_area_min_m2", "selected_area_max_m2", "selected_area_min_pyeong", "selected_area_max_pyeong"]
    )
    compare = compare.rename(
        columns={
            "gu": "자치구",
            "total_score": "종합점수",
            "total_grade": "등급",
            "total_star_label": "종합별점",
            "deposit_price_krw": "전세 보증금",
            "monthly_rent_active_krw": "월세",
            "jeonse_ratio_pct": "전세가율",
            "price_score": "가격 점수",
            "commute_score": "통근 점수",
            "infra_score": "인프라 점수",
            "safety_score": "치안 점수",
            "risk_score": "전세가율 점수",
            "primary_commute_minutes": "직장1 통근시간",
            "secondary_commute_minutes": "직장2 통근시간",
            "worst_commute_minutes": "최장 통근시간",
        }
    )
    for column in ["직장1 통근시간", "직장2 통근시간", "최장 통근시간"]:
        if column in compare.columns:
            compare[column] = compare[column].map(lambda x: f"{x:.1f}분" if pd.notna(x) else "-")
    st.dataframe(compare, width="stretch", height=430)


def _render_persona_tab(persona_row: pd.Series | None, persona_simulation: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">페르소나 구매 시뮬레이션</div>', unsafe_allow_html=True)
    if persona_row is None:
        st.info("페르소나 프로필 데이터가 없어 시뮬레이션을 표시할 수 없습니다.")
        return
    if persona_simulation.empty:
        st.warning("현재 조건에 맞는 시뮬레이션 대상이 없습니다.")
        return

    metrics = st.columns(4)
    metrics[0].metric("월소득 추정", format_korean_money(persona_row["monthly_income_estimate_krw"]))
    metrics[1].metric("부채 추정 잔액", format_korean_money(persona_row["debt_balance_estimate_krw"]))
    metrics[2].metric("2년 후 종잣돈", format_korean_money(persona_row["seed_money_2y_krw"]))
    metrics[3].metric("3년 후 추정 매수여력", format_korean_money(persona_row["buying_power_3y_krw"]))
    st.caption(persona_row["persona_summary"])

    sim_cols = [
        "gu",
        "deposit_price_krw",
        "monthly_rent_active_krw",
        "sale_price_krw",
        "buying_status_2y",
        "buying_status_3y",
        "buy_2y_gap_krw",
        "buy_3y_gap_krw",
    ]
    sim_table = persona_simulation[sim_cols].copy().head(15)
    sim_table["deposit_price_krw"] = sim_table["deposit_price_krw"].map(format_korean_money)
    sim_table["monthly_rent_active_krw"] = sim_table["monthly_rent_active_krw"].map(format_korean_money)
    sim_table["sale_price_krw"] = sim_table["sale_price_krw"].map(format_korean_money)
    sim_table["buy_2y_gap_krw"] = sim_table["buy_2y_gap_krw"].map(format_korean_money)
    sim_table["buy_3y_gap_krw"] = sim_table["buy_3y_gap_krw"].map(format_korean_money)
    sim_table = sim_table.rename(
        columns={
            "gu": "자치구",
            "deposit_price_krw": "전세 보증금",
            "monthly_rent_active_krw": "월세",
            "sale_price_krw": "추정 매매가",
            "buying_status_2y": "2년 후 매수 가능성",
            "buying_status_3y": "3년 후 매수 가능성",
            "buy_2y_gap_krw": "2년 후 매매가 차이",
            "buy_3y_gap_krw": "3년 후 매매가 차이",
        }
    )
    st.dataframe(sim_table, width="stretch", height=420)


def _render_data_tab(
    feature_table: pd.DataFrame,
    recommendations: pd.DataFrame,
    rent_distribution,
    score_distribution,
    recommendation_audit_md: str,
    scoring_lineage_md: str,
) -> None:
    st.markdown('<div class="section-title">데이터 근거</div>', unsafe_allow_html=True)
    if feature_table.empty or recommendations.empty:
        st.warning("현재 조건에 맞는 데이터가 없어 분포를 표시할 수 없습니다.")
        return
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("전세 중앙값", format_korean_money(feature_table["deposit_price_krw"].median()))
    k2.metric("월세 중앙값", format_korean_money(feature_table["monthly_rent_active_krw"].median()))
    k3.metric("종합점수 중앙값", f"{recommendations['total_score'].median():.1f}점")
    k4.metric("종합점수 최고", f"{recommendations['total_score'].max():.1f}점")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("#### 추천 값 검증")
            st.caption("추천과 시뮬레이션에 사용한 전세·월세 분포를 점검합니다.")
            st.write("- 자치구 집계값과 추천값을 분리해 확인")
            st.write("- 전세와 월세 분포를 동일 축에서 비교")
        if rent_distribution is not None:
            st.plotly_chart(rent_distribution, width="stretch")

    with c2:
        with st.container(border=True):
            st.markdown("#### 점수 라인리지")
            st.caption("5개 축 점수 분포와 종합점수 산식을 검토합니다.")
            st.write("- 가격, 통근, 인프라, 치안, 전세가율 점수 공개")
            st.write("- 고정 가중합 기반 종합점수 계산")
        if score_distribution is not None:
            st.plotly_chart(score_distribution, width="stretch")

    with st.expander("추천 값 검증 문서 보기", expanded=False):
        st.markdown(recommendation_audit_md)
    with st.expander("점수 라인리지 문서 보기", expanded=False):
        st.markdown(scoring_lineage_md)


def _render_landing_tab(
    project_readme_md: str,
    docs_readme_md: str,
    recommendation_audit_md: str,
    scoring_lineage_md: str,
) -> None:
    st.markdown('<div class="section-title">대시보드 안내</div>', unsafe_allow_html=True)
    st.markdown(
        """
        서울 자치구 추천 대시보드를 시작하기 전에 아래 문서에서 데이터 범위, 페르소나 기준, 점수 계산 방식을 먼저 확인할 수 있습니다.
        사이드바에서 필터를 설정하면 나머지 탭이 같은 조건으로 동시에 갱신됩니다.
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("기본 현금자산", format_korean_money(100_000_000))
    c2.metric("기본 저축비율", "50%")
    c3.metric("예산 구간 필터", "기본 해제")

    with st.expander("대시보드 README", expanded=True):
        st.markdown(project_readme_md)

    with st.expander("문서 구조 안내", expanded=False):
        st.markdown(docs_readme_md)

    with st.expander("데이터 출처", expanded=False):
        st.markdown(
            """
            - `매매/전월세`: 국토교통부 실거래가 공개 데이터와 집계 파일
            - `통근시간`: 자치구별 목적지 평균 소요시간 원천 CSV, 시간대별 통행량 가중평균 적용
            - `인프라`: 서울시 공원, 대형마트, 병원 위치 데이터
            - `치안`: 서울시 범죄 통계, 경찰 만족도 데이터
            - `페르소나`: 통계청 KOSIS `DT_1NW1036` 신혼부부 소득 및 금융권 대출 잔액 분포
            """
        )

    with st.expander("페르소나 선정 기준", expanded=False):
        st.markdown(
            """
            - 소득 구간은 전체 분포 기준 `P25 / P50 / P75` 대표값으로 `저소득 / 중간소득 / 고소득`을 구분합니다.
            - 부채 구간은 선택된 소득구간 내부의 금융권 대출 잔액 분포 기준 `P25 / P50 / P75` 대표값으로 `저부채 / 중간부채 / 고부채`를 구분합니다.
            - 가구 유형이 `2인 맞벌이`이면 소득과 부채, 구매 시뮬레이션 기준값을 `1인` 대비 2배로 적용합니다.
            """
        )

    _render_scoring_thresholds_guide()

    with st.expander("점수 라인리지 문서", expanded=False):
        st.markdown(scoring_lineage_md)

    with st.expander("추천 값 검증 자료", expanded=False):
        st.markdown(recommendation_audit_md)


def main() -> None:
    try:
        bundle = load_dataset_bundle()
    except Exception as exc:
        _show_data_load_error(exc)
        st.stop()

    _show_intro(bundle)
    persona_row, ui = _collect_sidebar_inputs(bundle)
    outputs = _compute_outputs(bundle, ui)
    income_reference = _scale_income_reference(_build_income_reference(bundle), ui["household_type"])

    recommendations = outputs["recommendations"]
    feature_table = outputs["feature_table"]
    persona_simulation = (
        build_persona_simulation(
            recommendations,
            persona_row,
            ui["cash_assets_krw"],
            ui["saving_ratio_pct"],
        )
        if persona_row is not None
        else recommendations.copy()
    )

    gallery = build_visualization_gallery(feature_table, recommendations, bundle, ui["selected_year"])
    recommendation_summary = build_recommendation_summary(recommendations, ui["household_type"])
    recommendation_map = build_recommendation_map(
        recommendations,
        ui["workplace_name"],
        ui["secondary_workplace_name"],
    )
    rank_chart = build_top_rank_chart(recommendations)
    project_readme_md = _read_markdown(PROJECT_README_PATH)
    docs_readme_md = _read_markdown(DOCS_README_PATH)
    recommendation_audit_md = _read_markdown(RECOMMENDATION_AUDIT_PATH)
    scoring_lineage_md = _read_markdown(SCORING_LINEAGE_PATH)

    tabs = st.tabs(
        [
            "랜딩",
            "추천 요약",
            "구별 상세 비교",
            "인프라·입지 분석",
            "치안·재개발 분석",
            "페르소나 구매 시뮬레이션",
        ]
    )

    with tabs[0]:
        _render_landing_tab(
            project_readme_md,
            docs_readme_md,
            recommendation_audit_md,
            scoring_lineage_md,
        )

    with tabs[1]:
        _render_summary_tab(
            recommendations,
            recommendation_summary,
            recommendation_map,
            rank_chart,
            persona_row,
            ui,
            income_reference,
            outputs["filter_notice"],
        )

    with tabs[2]:
        _render_compare_tab(recommendations)
        st.plotly_chart(gallery["score_stacked_bar"], width="stretch")

    with tabs[3]:
        st.markdown('<div class="section-title">인프라·입지 분석</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(gallery["infra_bar"], width="stretch")
            st.plotly_chart(gallery["infra_scatter"], width="stretch")
        with c2:
            st.plotly_chart(gallery["infra_score_bar"], width="stretch")
            st.plotly_chart(gallery["recommendation_bubble"], width="stretch")
        commute_timeseries = _get_commute_timeseries(bundle)
        if not commute_timeseries.empty:
            destination_options = commute_timeseries["destination_name"].drop_duplicates().tolist()
            selected_destination = st.selectbox(
                "시간대별 평균 소요시간 목적지",
                destination_options,
                index=0,
                key="commute_timeseries_destination",
            )
            destination_frame = commute_timeseries.loc[
                commute_timeseries["destination_name"].eq(selected_destination)
            ].copy()
            destination_frame = _select_top_commute_gus(destination_frame, top_n=5)
            st.caption("평균 통근 시간이 가장 긴 자치구 TOP 5만 표시합니다.")
            st.plotly_chart(build_commute_timeseries_chart(destination_frame, selected_destination), width="stretch")

    with tabs[4]:
        st.markdown('<div class="section-title">치안·재개발 분석</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(gallery["safety_dual_axis"], width="stretch")
            st.plotly_chart(gallery["crime_vs_police"], width="stretch")
        with c2:
            st.plotly_chart(gallery["redevelopment_stage_bar"], width="stretch")
            st.plotly_chart(gallery["redevelopment_vs_score"], width="stretch")

    with tabs[5]:
        _render_persona_tab(persona_row, persona_simulation)

def _render_persona_tab(persona_row: pd.Series | None, persona_simulation: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">페르소나 구매 시뮬레이션</div>', unsafe_allow_html=True)
    if persona_row is None:
        st.info("페르소나 프로필 데이터가 없어 시뮬레이션을 표시할 수 없습니다.")
        return
    if persona_simulation.empty:
        st.warning("현재 조건에 맞는 시뮬레이션 대상이 없습니다.")
        return

    summary_row = persona_simulation.iloc[0]
    annual_income = max(float(persona_row.get("monthly_income_estimate_krw", 0) or 0) * 12, 1)
    saving_ratio_pct = float(summary_row.get("annual_savings_krw", 0) or 0) / annual_income * 100

    metrics = st.columns(5)
    metrics[0].metric("월소득 추정", format_korean_money(persona_row["monthly_income_estimate_krw"]))
    metrics[1].metric("현금자산", format_korean_money(summary_row["cash_assets_krw"]))
    metrics[2].metric("현재 총 부채", format_korean_money(persona_row["debt_balance_estimate_krw"]))
    metrics[3].metric("저축비율", f"{saving_ratio_pct:.1f}%")
    metrics[4].metric("예상 저축액(1년)", format_korean_money(summary_row["annual_savings_krw"]))
    st.caption(persona_row["persona_summary"])
    st.caption("가용금액 = 현금자산 + 대출가능금액(LTV 70%) + 예상 저축액(1년), 이자율은 연 4% 가정입니다.")

    sim_cols = [
        "gu",
        "sale_price_krw",
        "cash_assets_krw",
        "current_total_debt_krw",
        "available_loan_krw",
        "annual_savings_krw",
        "interest_burden_rate_pct",
        "purchase_gap_krw",
    ]
    sim_table = persona_simulation[sim_cols].copy().head(15)
    sim_table["sale_price_krw"] = sim_table["sale_price_krw"].map(format_korean_money)
    sim_table["cash_assets_krw"] = sim_table["cash_assets_krw"].map(format_korean_money)
    sim_table["current_total_debt_krw"] = sim_table["current_total_debt_krw"].map(format_korean_money)
    sim_table["available_loan_krw"] = sim_table["available_loan_krw"].map(format_korean_money)
    sim_table["annual_savings_krw"] = sim_table["annual_savings_krw"].map(format_korean_money)
    sim_table["interest_burden_rate_pct"] = sim_table["interest_burden_rate_pct"].map(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "-"
    )
    sim_table["purchase_gap_krw"] = sim_table["purchase_gap_krw"].map(format_korean_money)
    sim_table = sim_table.rename(
        columns={
            "gu": "자치구",
            "sale_price_krw": "자치구별 매매가",
            "cash_assets_krw": "현금자산",
            "current_total_debt_krw": "현재 총 부채",
            "available_loan_krw": "대출가능금액(LTV 70%)",
            "annual_savings_krw": "예상 저축액(1년)",
            "interest_burden_rate_pct": "예상 총 부채이자부담률",
            "purchase_gap_krw": "매매가-가용금액",
        }
    )
    st.dataframe(sim_table, width="stretch", height=420)


if __name__ == "__main__":
    main()
