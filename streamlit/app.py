from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.cleaning import remove_iqr_outliers
from src.config import DATA_DIR, DATA_DIR_CANDIDATES, PROJECT_ROOT, WORKPLACE_HUBS
from src.feature_engineering import build_feature_table
from src.io_utils import load_dataset_bundle
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
            - `가격 점수(전세)`: 전세 보증금 중위값의 분포(+/- 1.5 시그마)를 기준으로 상대적 순위(Percentile)를 점수화합니다.
            - `가격 점수(월세)`: 표준화된 월세(보증금 20:월세 1 일할)의 분포를 기준으로 점수화합니다.
            - `통근 점수`: 선택한 직장들까지의 예측 통근시간을 기반으로 평가한 점수입니다.
            - `인프라 점수`: 자치구별 병원, 공원, 대형마트 수를 표준화(Min-Max)하여 합산한 결과입니다.
            - `치안 점수`: 자치구별 5대 범죄 발생 건수와 경찰 서비스 만족도를 종합하여 절대적 안정성을 평가합니다.
            - `전세가율 점수`: 자치구별 매매가 대비 전세가 비중을 계산하여 리스크가 클수록 낮은 점수를 부여합니다.
            - `가중치`: 위 5가지 지표의 반영 비율을 직접 설정하여 합계 100%로 가중치를 부여합니다.
            
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


def _collect_sidebar_inputs(bundle: dict[str, object]) -> dict[str, object]:
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
    household_type = "2인 맞벌이"
    

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

    # 예산 구간 및 개별 자산 입력 제거 (EDA 탭으로 이동됨)
    deposit_budget_min, deposit_budget_max = 0, 2_000_000_000
    monthly_budget_min, monthly_budget_max = 0, 5_000_000
    use_deposit_budget_filter = False
    use_monthly_budget_filter = False
    
    # 자산/소득 정보는 st.session_state (EDA 탭) 에서 가져오므로 사이드바에서는 제거
    cash_assets_krw = st.session_state.get("cash_assets_krw_eda", 100_000_000)
    saving_ratio_pct = st.session_state.get("saving_ratio_pct_eda", 50.0)

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
        "desired_contract_type_eda": st.session_state.get("desired_contract_type_eda", "전세"),
    }
    return state


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
    raw_eda_series, raw_thresholds = _build_raw_eda_inputs(bundle, ui)
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
    # 추천 점수용 임계값 조정 (월세 선택 시 가격 임계값을 월세 기준으로 교체)
    desired_type = ui.get("desired_contract_type_eda", "전세")
    scoring_thresholds = raw_thresholds.copy()
    if desired_type == "월세":
        scoring_thresholds["price"] = raw_thresholds.get("monthly_price", {})

    recommendations, scoring_meta = score_recommendations(
        feature_table=filtered_feature_table,
        selected_gus=[],
        commute_frame=commute_frame,
        weights=ui["weights"],
        threshold_overrides=scoring_thresholds,
        missing_strategy="mean",
        household_type=ui["household_type"],
        desired_contract_type=desired_type,
    )

    # 1. 사용자 정보 기반 추가 필터링 (Limit A & 부담률)
    # 현재 시점(today)은 메타데이터(2026-03-21) 기준
    today = datetime.date(2026, 3, 21)
    move_in = st.session_state.get("move_in_date_eda", datetime.date(2026, 10, 1))
    months = max(0, (move_in.year - today.year) * 12 + (move_in.month - today.month))
    
    cash = st.session_state.get("cash_assets_krw_eda", 100_000_000)
    income_val = st.session_state.get("user_income_eda", 3_400_000) + st.session_state.get("spouse_income_eda", 3_400_000)
    income_val = max(income_val, 1) # 0으로 나누기 방지
    saving_ratio = st.session_state.get("saving_ratio_pct_eda", 50) / 100
    burden_range = st.session_state.get("financial_burden_rate_eda", (10.0, 40.0))
    
    if desired_type == "전세":
        loan_ratio = st.session_state.get("jeonse_loan_ratio_eda", 80.0) / 100
        limit_a = (cash + (income_val * saving_ratio * months)) / max(0.01, 1 - loan_ratio)
        # C-2: 최악의 경우(6% 이자율) 기준 부담률
        recommendations["burden_rate"] = ((recommendations["deposit_price_krw"] * 0.06) / 12) / income_val * 100
    else:
        limit_a = (cash + (income_val * saving_ratio * months))
        # B: 표준화 월세 대비 부담률
        recommendations["burden_rate"] = (recommendations["standardized_monthly_rent_krw"] / income_val) * 100

    # 필터 적용: 가격(보증금) A 이하 및 부담률 범위 내
    mask = (recommendations["deposit_price_krw"] <= limit_a) & \
           (recommendations["burden_rate"] >= burden_range[0]) & \
           (recommendations["burden_rate"] <= burden_range[1])
    
    recommendations = recommendations[mask].copy()
    # 필터 결과 안내 문구 추가
    if recommendations.empty:
        filter_notice = f"입력하신 가용 자산({format_korean_money(limit_a)}) 및 부담률 범위({burden_range[0]}~{burden_range[1]}%) 내에 해당하는 지역이 없어 추천 결과가 비어있습니다."
    else:
        filter_notice = (filter_notice + " | " if filter_notice else "") + f"가용 자산 {format_korean_money(limit_a)} 이하 필터 적용됨"

    return {
        "feature_table": filtered_feature_table,
        "feature_meta": feature_meta,
        "commute_frame": commute_frame,
        "commute_meta": commute_meta,
        "recommendations": recommendations,
        "scoring_meta": scoring_meta,
        "raw_eda_series": raw_eda_series,
        "raw_thresholds": raw_thresholds,
        "filter_notice": filter_notice,
    }


def _build_raw_eda_inputs(bundle: dict[str, object], ui: dict[str, object]) -> tuple[dict[str, pd.Series], dict[str, dict[str, float]]]:
    rent = bundle["rent"].copy()
    sale = bundle["sale"].copy()
    crime = bundle["crime"].copy()
    commute_timeseries = bundle.get("commute_timeseries", pd.DataFrame()).copy()
    hospital = bundle["hospital"].copy()
    mart = bundle["mart"].copy()
    parks = bundle["parks"].copy()

    rent_year_col = "년월" if "년월" in rent.columns else next((col for col in ["?꾩썡"] if col in rent.columns), None)
    rent_area_col = "전용면적_m2" if "전용면적_m2" in rent.columns else next((col for col in ["?꾩슜硫댁쟻_m2"] if col in rent.columns), None)
    rent_deposit_col = "보증금_만원_krw" if "보증금_만원_krw" in rent.columns else next((col for col in ["蹂댁쬆湲?留뚯썝_krw", "보증금_만원"] if col in rent.columns), "보증금_만원_krw")
    rent_monthly_col = "월세_만원_krw" if "월세_만원_krw" in rent.columns else next((col for col in ["?붿꽭_留뚯썝_krw", "월세_만원"] if col in rent.columns), "월세_만원_krw")

    if rent_year_col is None or rent_area_col is None or rent_deposit_col is None or rent_monthly_col is None:
        raise KeyError("Rent source columns are missing required fields for raw EDA.")

    rent["year"] = pd.to_numeric(rent[rent_year_col].astype(str).str[:4], errors="coerce")
    sale["dealYear"] = pd.to_numeric(sale["dealYear"], errors="coerce")
    rent = remove_iqr_outliers(rent, [rent_deposit_col, rent_monthly_col, rent_area_col])
    sale = remove_iqr_outliers(sale, ["dealAmount_krw", "excluUseAr"])

    min_m2 = ui["min_area_pyeong"] * 3.3058
    max_m2 = ui["max_area_pyeong"] * 3.3058
    
    # 2025년 기준 및 선택 연도 기준 필터링 (EDA용)
    rent_price_all = rent.loc[
        pd.to_numeric(rent[rent_area_col], errors="coerce").between(min_m2, max_m2, inclusive="both")
        & pd.to_numeric(rent["year"], errors="coerce").eq(2025)
    ].copy()
    rent_curr = rent.loc[
        pd.to_numeric(rent[rent_area_col], errors="coerce").between(min_m2, max_m2, inclusive="both")
        & pd.to_numeric(rent["year"], errors="coerce").eq(ui["selected_year"])
    ].copy()
    sale_curr = sale.loc[
        pd.to_numeric(sale["excluUseAr"], errors="coerce").between(min_m2, max_m2, inclusive="both")
        & pd.to_numeric(sale["dealYear"], errors="coerce").eq(ui["selected_year"])
    ].copy()

    # 자치구별 매매가 중위값
    sale_gu_median = sale_curr.groupby("gu", as_index=False)["dealAmount_krw"].median().rename(columns={"dealAmount_krw": "sale_median_krw"})
    
    # 1. EDA용 전체 Population 리스크 계산 (2025년 기준)
    rent_with_sale_all = rent_price_all.merge(sale_gu_median, on="gu", how="left")
    risk_series_all = (
        pd.to_numeric(rent_with_sale_all[rent_deposit_col], errors="coerce")
        / pd.to_numeric(rent_with_sale_all["sale_median_krw"], errors="coerce").replace(0, pd.NA)
        * 100
    )

    # 통근/인프라/치안
    primary_commute = commute_timeseries.loc[commute_timeseries["hub_name"].eq(ui["workplace_name"])].copy()
    if ui.get("secondary_workplace_name"):
        secondary_commute = commute_timeseries.loc[commute_timeseries["hub_name"].eq(ui["secondary_workplace_name"])].copy()
        commute_source = primary_commute.merge(
            secondary_commute[["gu", "time_order", "avg_minutes"]].rename(columns={"avg_minutes": "secondary_avg_minutes"}),
            on=["gu", "time_order"], how="inner",
        )
        commute_series = commute_source["avg_minutes"] * 0.55 + commute_source["secondary_avg_minutes"] * 0.45
    else:
        commute_series = pd.to_numeric(primary_commute["avg_minutes"], errors="coerce")

    infra_base = bundle["infra"][["gu", "hospital_count", "park_count", "mart_count"]].copy()
    for col in ["hospital_count", "park_count", "mart_count"]:
        infra_base[col] = pd.to_numeric(infra_base[col], errors="coerce").fillna(0)
        c_min, c_max = float(infra_base[col].min()), float(infra_base[col].max())
        infra_base[f"{col}_norm"] = (infra_base[col] - c_min) / (c_max - c_min) if c_max > c_min else 0.0
    infra_series = infra_base["hospital_count_norm"] + infra_base["park_count_norm"] + infra_base["mart_count_norm"]
    
    crime_series = crime.assign(crime_count=pd.to_numeric(crime.get("crime_count"), errors="coerce")).groupby("gu", dropna=False)["crime_count"].sum(min_count=1).reset_index(drop=True)

    def standardize_wolse(row):
        dep, ren = row[rent_deposit_col], row[rent_monthly_col]
        return (ren + dep * 0.005) / 1.1 if ren > 0 else None

    # EDA 시리즈 (전체 모수)
    raw_series_all = {
        "price": pd.to_numeric(rent_price_all[rent_deposit_col], errors="coerce"),
        "monthly_price": rent_price_all.apply(standardize_wolse, axis=1),
        "commute": pd.to_numeric(commute_series, errors="coerce"),
        "infra": infra_series,
        "safety": pd.to_numeric(crime_series, errors="coerce"),
        "risk": pd.to_numeric(risk_series_all, errors="coerce"),
    }

    # 2. 임계값 계산용 데이터셋 분리 (전세 vs 월세)
    rent_jeonse = rent_price_all[rent_price_all[rent_monthly_col] == 0]
    rent_wolse = rent_price_all[rent_price_all[rent_monthly_col] > 0]

    # 임계값 계산용 시리즈 구성
    scoring_series = {
        "price": pd.to_numeric(rent_jeonse[rent_deposit_col], errors="coerce"),
        "monthly_price": rent_wolse.apply(standardize_wolse, axis=1),
        "commute": pd.to_numeric(commute_series, errors="coerce"),
        "infra": infra_series,
        "safety": pd.to_numeric(crime_series, errors="coerce"),
        "risk": pd.to_numeric(risk_series_all, errors="coerce"), # 리스크는 전체 모수 기준
    }

    thresholds = {}
    for metric_name, series in scoring_series.items():
        clean = pd.to_numeric(series, errors="coerce").dropna()
        mean = float(clean.mean()) if not clean.empty else 0.0
        std = float(clean.std(ddof=0)) if not clean.empty else 0.0
        thresholds[metric_name] = {
            "mean": mean, "std": std,
            "lower_1_5_std": mean - 1.5 * std,
            "lower_0_5_std": mean - 0.5 * std,
            "upper_0_5_std": mean + 0.5 * std,
            "upper_1_5_std": mean + 1.5 * std,
        }
    return raw_series_all, thresholds


def _render_summary_tab(
    recommendations: pd.DataFrame,
    recommendation_summary: pd.DataFrame,
    recommendation_map,
    rank_chart,
    ui: dict[str, object],
    filter_notice: str | None = None,
    radar_chart: go.Figure | None = None,
) -> None:
    st.markdown('<div class="section-title">추천 요약</div>', unsafe_allow_html=True)
    
    # 사용자 개인화 요약 문구 생성
    today = datetime.date(2026, 3, 21)
    move_in = st.session_state.get("move_in_date_eda", datetime.date(2026, 10, 1))
    months = max(0, (move_in.year - today.year) * 12 + (move_in.month - today.month))
    cash = st.session_state.get("cash_assets_krw_eda", 100_000_000)
    income_val = st.session_state.get("user_income_eda", 3_400_000) + st.session_state.get("spouse_income_eda", 3_400_000)
    income_val = max(income_val, 1)
    saving_ratio = st.session_state.get("saving_ratio_pct_eda", 50) / 100
    desired_type = st.session_state.get("desired_contract_type_eda", "전세")

    if desired_type == "전세":
        loan_ratio_pct = st.session_state.get("jeonse_loan_ratio_eda", 80.0)
        loan_ratio = loan_ratio_pct / 100
        limit_a = (cash + (income_val * saving_ratio * months)) / max(0.01, 1 - loan_ratio)
        b1 = (limit_a * 0.04) / 12
        b2 = (limit_a * 0.06) / 12
        c1 = (b1 / income_val) * 100
        c2 = (b2 / income_val) * 100
        summary_text = (
            f"현재 현금자산 {format_korean_money(cash)}, 합산소득 월 {format_korean_money(income_val)}, "
            f"저축비율 {int(saving_ratio*100)}%, 전세자금대출 {int(loan_ratio_pct)}%를 고려했을 때, "
            f"**가능한 전세 구간은 {format_korean_money(limit_a)} 이하** 입니다. "
            f"이 경우 예상 월 발생 금융 비용은 4~6% 가정할 경우 {format_korean_money(b1)} ~ {format_korean_money(b2)}이고 "
            f"합산소득 대비 {c1:.1f}% ~ {c2:.1f}% 입니다."
        )
    else:
        limit_a = (cash + (income_val * saving_ratio * months))
        summary_text = (
            f"현재 현금자산 {format_korean_money(cash)}, 합산소득 월 {format_korean_money(income_val)}, "
            f"저축비율 {int(saving_ratio*100)}%를 고려했을 때 **가능한 월세 보증금 구간은 {format_korean_money(limit_a)} 이하**입니다."
        )
    
    st.info(summary_text)

    if filter_notice:
        st.caption(f"참고: {filter_notice}")


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
                st.write(build_short_reco_label(row, "기본"))
                st.text(
                    f"{ui['min_area_m2']}~{ui['max_area_m2']}㎡ / "
                    f"{ui['min_area_pyeong']}~{ui['max_area_pyeong']}평 기준"
                )
                
                # 계약 방식에 따른 동적 지표 표시 (요청사항 반영)
                if ui.get("desired_contract_type_eda") == "전세":
                    st.write(f"전세 보증금: {format_korean_money(row['deposit_price_krw'])}")
                    st.write(f"전세가율: {row['jeonse_ratio_pct']:.1f}%")
                else:
                    st.write(f"보증금: {format_korean_money(row['deposit_price_krw'])}")
                    # 월세(표준화) 표시를 위해 src.feature_engineering 의 계산 로직 사용 
                    # 또는 기 가공된 standardized_monthly_rent_krw 사용
                    st.write(f"월세 (표준화): {format_korean_money(row.get('standardized_monthly_rent_krw', row['monthly_rent_active_krw']))}")
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

    if radar_chart:
        st.plotly_chart(radar_chart, use_container_width=True)

    st.dataframe(recommendation_summary, width="stretch", height=320)


def _render_compare_tab(recommendations: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">구별 상세 비교</div>', unsafe_allow_html=True)
    if recommendations.empty:
        st.warning("현재 조건에 맞는 비교 대상이 없습니다.")
        return
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





def _render_landing_tab() -> None:
    st.markdown('<div class="section-title">프로젝트 안내</div>', unsafe_allow_html=True)
    st.markdown(
        """
        **팀명:** 데이터 그래비티  
        **프로젝트명:** 신혼부부를 위한 서울 아파트 전월세 최적 입지 추천 분석  
        **진행 기간:** 2026년 2월 20일 ~ 3월 27일  
        **핵심 질문:** 회사 위치와 예산을 고려했을 때, 어디에 신혼집을 구하는 게 가장 합리적인가?
        """
    )
    st.markdown(
        """
        가격, 통근, 생활환경, 안전성을 함께 점수화해 서울 자치구 추천 결과를 보여줍니다.
        매물 검색보다 `내 조건에 맞는 입지 추천`에 초점을 둔 대시보드입니다.
        """
    )
    st.markdown(
        """
        **누가 쓰면 좋은가**
        - 예비 신혼부부
        - 서울 이사를 고민하는 1~2인 직장인 가구
        - 주거 데이터 분석에 관심 있는 사용자

        **해결하는 문제**
        - 입지 결정까지 걸리는 탐색 시간을 줄입니다.
        - 추천 결과 확인 이후 전월세 대출 비교와 상담 전환으로 이어지는 흐름을 돕습니다.

        **데이터 출처**
        - 국토교통부 실거래가 데이터
        - 서울시 공원, 병원, 대형마트 데이터
        - 서울시 범죄 통계와 경찰 만족도 데이터
        - 자치구별 목적지 평균 소요시간 원천 CSV
        - KOSIS `DT_1NW1036` 신혼부부 소득 및 금융권 대출 잔액 분포

        **점수 산정 기준**
        - 가격, 통근, 인프라, 치안, 전세가율을 0~100점으로 환산합니다.
        - 사용자가 입력한 예산 및 자산 정보를 기반으로 입지 적합성을 평가합니다.
        - 가중치를 반영한 종합점수로 추천 지역 TOP 5를 제시합니다.
        """
    )


def _build_eda_threshold_table(scoring_meta: dict[str, object]) -> pd.DataFrame:
    thresholds = scoring_meta.get("thresholds", {}) if isinstance(scoring_meta, dict) else {}
    rows: list[dict[str, object]] = []
    label_map = {
        "price": "가격(전세)",
        "commute": "통근",
        "infra": "인프라",
        "safety": "치안(범죄건수)",
        "risk": "전세가율",
    }
    for metric_key, label in label_map.items():
        metric_meta = thresholds.get(metric_key, {})
        rows.append(
            {
                "지표": label,
                "평균": metric_meta.get("mean"),
                "표준편차": metric_meta.get("std"),
                "-1.5σ": metric_meta.get("lower_1_5_std"),
                "-0.5σ": metric_meta.get("lower_0_5_std"),
                "0.5σ": metric_meta.get("upper_0_5_std"),
                "1.5σ": metric_meta.get("upper_1_5_std"),
                "방향": "높을수록 좋음" if metric_meta.get("higher_is_better") else "낮을수록 좋음",
            }
        )
    return pd.DataFrame(rows)


def _render_eda_tab(
    recommendations: pd.DataFrame,
    raw_eda_series: dict[str, pd.Series],
    raw_thresholds: dict[str, dict[str, float]],
    scoring_meta: dict[str, object],
    bundle: dict[str, object],
    ui: dict[str, object],
) -> None:

    with st.container(border=True):
        st.markdown("#### 사용자 정보 입력")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("현금자산(원)", min_value=0, step=10_000_000, key="cash_assets_krw_eda")
            st.number_input("본인 소득(월/원)", min_value=0, step=100_000, key="user_income_eda")
            st.number_input("배우자 소득(월/원)", min_value=0, step=100_000, key="spouse_income_eda")
        with c2:
            st.selectbox("희망 계약 방식", ["전세", "월세"], key="desired_contract_type_eda")
            st.number_input("저축 비율(%)", min_value=0, max_value=100, step=5, key="saving_ratio_pct_eda")
            st.date_input("예상 입주 시기", key="move_in_date_eda")
            if st.session_state.get("desired_contract_type_eda") == "전세":
                st.slider("희망 전세자금대출 비율 (%)", 0.0, 80.0, key="jeonse_loan_ratio_eda", step=5.0)
        with c3:
            st.slider(
                "금융비용 부담률 (%)",
                0.0, 100.0,
                key="financial_burden_rate_eda",
                step=1.0,
                help="합산 소득 대비 주거 금융 비용(이자 또는 월세)의 비중 범위를 설정합니다."
            )
            st.caption(f"설정 범위: {st.session_state.financial_burden_rate_eda[0]}% ~ {st.session_state.financial_burden_rate_eda[1]}%")



    st.markdown("#### 점수 산정 기준")
    st.markdown(
        """
        가격(전세), 통근, 인프라, 치안(범죄건수), 전세가율의 분포를 보고
        `평균 ± 0.5σ`, `평균 ± 1.5σ`를 5단계 점수 구간 threshold로 사용합니다.
        """
    )

    threshold_table = _build_eda_threshold_table(scoring_meta)
    numeric_cols = ["평균", "표준편차", "-1.5σ", "-0.5σ", "0.5σ", "1.5σ"]
    styled_thresholds = threshold_table.copy()
    for column in numeric_cols:
        styled_thresholds[column] = styled_thresholds[column].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")
    st.dataframe(styled_thresholds, width="stretch", height=240)

    # 지역구별 전세/월세 계약 건수 막대그래프
    st.markdown("#### 지역구별 전세/월세 계약 현황")
    rent_df = bundle.get("rent")
    if rent_df is not None:
        # 컬럼명 확인
        rent_deposit_col = "보증금_만원_krw" if "보증금_만원_krw" in rent_df.columns else next((col for col in ["蹂댁쬆湲?留뚯썝_krw", "보증금_만원"] if col in rent_df.columns), "보증금_만원_krw")
        rent_monthly_col = "월세_만원_krw" if "월세_만원_krw" in rent_df.columns else next((col for col in ["?붿꽭_留뚯썝_krw", "월세_만원"] if col in rent_df.columns), "월세_만원_krw")

        rent_temp = rent_df.copy()
        # 전세/월세 구분 (월세_만원_krw가 0이면 전세)
        rent_temp["계약유형"] = rent_temp[rent_monthly_col].apply(lambda x: "전세" if x == 0 else "월세")
        contract_counts = rent_temp.groupby(["gu", "계약유형"]).size().reset_index(name="건수")
        
        fig_contracts = px.bar(
            contract_counts, 
            x="gu", 
            y="건수", 
            color="계약유형", 
            barmode="group",
            title="지역구별 전세 vs 월세 계약 건수",
            labels={"gu": "자치구", "건수": "계약 건수", "계약유형": "유형"},
            color_discrete_map={"전세": "#8ec5fc", "월세": "#ffb3c1"}
        )
        st.plotly_chart(fig_contracts, width="stretch")

    # 월세 계약 보증금 vs 월세 스캐터 플롯
    st.markdown("#### 월세 계약 보증금 vs 월세 관계")
    if rent_df is not None:
        # 월세_만원_krw 대신 탐색된 컬럼 사용
        wolse_df = rent_df[rent_df[rent_monthly_col] > 0].copy()
        if not wolse_df.empty:
            fig_wolse_scatter = px.scatter(
                wolse_df,
                x=rent_deposit_col,
                y=rent_monthly_col,
                color="gu",
                title="월세 계약: 보증금 vs 월세 금액 (만원)",
                labels={rent_deposit_col: "보증금(만원)", rent_monthly_col: "월세(만원)", "gu": "자치구"},
                hover_data=["년월"]
            )
            st.plotly_chart(fig_wolse_scatter, width="stretch")
            st.markdown("**월세 계약의 경우 보증금 비율이 굉장히 상이하기 때문에 표준화가 필요함**")
        else:
            st.info("월세 계약 데이터가 없습니다.")

    # 표준화된 월세 스캐터 플롯 (전세/월세 통합)
    st.markdown("#### 표준화된 월세 분석 (보증금 20배 기준 통합)")
    if rent_df is not None:
        wolse_df = rent_df[rent_df[rent_monthly_col] > 0].copy()
        if not wolse_df.empty:
            # 모든 월세 계약을 '보증금 = 월세 * 20' 기준으로 표준화
            wolse_df["표준화_월세_만원"] = (wolse_df[rent_monthly_col] + wolse_df[rent_deposit_col] * 0.005) / 1.1
            # 표준화된 보증금 (월세의 20배)
            wolse_df["표준화_보증금_만원"] = wolse_df["표준화_월세_만원"] * 20
            
            fig_wolse_std = px.scatter(
                wolse_df,
                x="표준화_보증금_만원",
                y="표준화_월세_만원",
                color="gu",
                title="표준화 월세 vs 표준화 보증금 (보증금 20배 고정)",
                labels={"표준화_보증금_만원": "표준화 보증금(만원)", "표준화_월세_만원": "표준화 월세(만원)", "gu": "자치구"},
                hover_data=["년월", "보증금_만원_krw", "월세_만원_krw"]
            )
            st.plotly_chart(fig_wolse_std, width="stretch")
            
            st.markdown("**모든 월세 계약을 '보증금 = 월세 * 20' 기준으로 표준화(월세 조정)하여 가격을 비교합니다.** (환산율 6% 적용)")


    metric_defs = [
        ("가격(전세)", "price"),
        ("표준화 월세", "monthly_price"),
        ("통근", "commute"),
        ("인프라", "infra"),
        ("치안(범죄건수)", "safety"),
        ("전세가율", "risk"),
    ]
    descriptions = {
        "price": f"{ui['selected_year']}년 기준 필터링된 면적의 모든 전세 실거래 데이터를 기반으로 한 보증금 분포입니다.",
        "monthly_price": f"{ui['selected_year']}년 기준 필터링된 면적의 모든 월세 실거래 데이터를 대상으로, 보증금을 월세로 환산하여 표준화한(보증금=월세*20, 환산율 6%) 가격 분포입니다.",
        "commute": "직장 소지 지점(및 부 소지 지점)까지의 대중교통 이용 시간(분)에 대한 자치구별 가중 평균 분포입니다. (낮을수록 접근성 우수)",
        "infra": "자치구별 병원, 공원, 마트 수를 각각 0~1로 표준화하여 합산한 인프라 지수 분포입니다. (높을수록 편의시설 풍부)",
        "safety": "서울시 자치구별 연간 발생한 총 범죄 건수에 대한 분포입니다. (낮을수록 안전)",
        "risk": "자치구별 아파트 매매 중위가 대비 전세 보증금의 비율(%)에 대한 분포입니다. (낮을수록 역전세 위험 적음)",
    }

    chart_cols = st.columns(2)
    for idx, (label, metric_key) in enumerate(metric_defs):
        series = pd.to_numeric(raw_eda_series.get(metric_key), errors="coerce").dropna()
        if series.empty:
            continue
        
        col = chart_cols[idx % 2]
        # 지표별 설명 추가
        if metric_key in descriptions:
            col.caption(descriptions[metric_key])
            
        metric_meta = raw_thresholds.get(metric_key, {})
        fig = px.histogram(x=series, nbins=20, title=f"{label} 원본 분포")
        for threshold_key, color in [
            ("lower_1_5_std", "#22c55e"),
            ("lower_0_5_std", "#84cc16"),
            ("upper_0_5_std", "#f59e0b"),
            ("upper_1_5_std", "#ef4444"),
        ]:
            threshold_value = metric_meta.get(threshold_key)
            if pd.notna(threshold_value):
                fig.add_vline(x=float(threshold_value), line_dash="dash", line_color=color)
        col.plotly_chart(fig, width="stretch")









def main() -> None:
    # 사용자 정보 초기값 설정 (EDA 탭 및 필터링용)
    st.session_state.setdefault("cash_assets_krw_eda", 100_000_000)
    st.session_state.setdefault("user_income_eda", 3_400_000)
    st.session_state.setdefault("spouse_income_eda", 3_400_000)
    st.session_state.setdefault("desired_contract_type_eda", "전세")
    st.session_state.setdefault("saving_ratio_pct_eda", 50)
    st.session_state.setdefault("move_in_date_eda", datetime.date(2026, 10, 1))
    st.session_state.setdefault("financial_burden_rate_eda", (10.0, 40.0))
    st.session_state.setdefault("jeonse_loan_ratio_eda", 80.0)

    try:
        bundle = load_dataset_bundle()
    except Exception as exc:
        _show_data_load_error(exc)
        st.stop()

    try:
        _show_intro(bundle)
        ui = _collect_sidebar_inputs(bundle)
        outputs = _compute_outputs(bundle, ui)
        
        recommendations = outputs["recommendations"]
        feature_table = outputs["feature_table"]

        gallery = build_visualization_gallery(feature_table, recommendations, bundle, ui["selected_year"])
        recommendation_summary = build_recommendation_summary(
            recommendations, 
            ui["household_type"],
            desired_contract_type=st.session_state.get("desired_contract_type_eda", "전세")
        )
        recommendation_map = build_recommendation_map(
            recommendations,
            ui["workplace_name"],
            ui["secondary_workplace_name"],
        )
        rank_chart = build_top_rank_chart(recommendations)
        tabs = st.tabs(
            [
                "프로젝트 개요",
                "EDA 및 기준 선정",
                "추천 요약",
                "구별 상세 비교",
                "인프라·입지 분석",
                "치안·재개발 분석",
            ]
        )

        with tabs[0]:
            _render_landing_tab()

        with tabs[1]:
            _render_eda_tab(
                recommendations,
                outputs["raw_eda_series"],
                outputs["raw_thresholds"],
                outputs["scoring_meta"],
                bundle,
                ui,
            )

        with tabs[2]:
            _render_summary_tab(
                recommendations,
                recommendation_summary,
                recommendation_map,
                rank_chart,
                ui,
                outputs["filter_notice"],
                radar_chart=gallery.get("score_radar"),
            )

        with tabs[3]:
            _render_compare_tab(recommendations)
            st.plotly_chart(gallery["score_stacked_bar"], width="stretch")

        with tabs[4]:
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

        with tabs[5]:
            st.markdown('<div class="section-title">치안·재개발 분석</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(gallery["safety_dual_axis"], width="stretch")
                st.plotly_chart(gallery["crime_vs_police"], width="stretch")
            with c2:
                st.plotly_chart(gallery["redevelopment_stage_bar"], width="stretch")
                st.plotly_chart(gallery["redevelopment_vs_score"], width="stretch")
    except Exception as e:
        st.error(f"애플리케이션 실행 중 오류가 발생했습니다: {e}")
        st.exception(e)



if __name__ == "__main__":
    main()
