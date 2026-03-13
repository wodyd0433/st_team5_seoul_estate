from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import DATA_DIR, DATA_DIR_CANDIDATES, PROJECT_ROOT, WORKPLACE_HUBS
from src.feature_engineering import build_feature_table
from src.io_utils import load_dataset_bundle
from src.persona import build_persona_simulation
from src.scoring_engine import FIXED_WEIGHTS, prepare_commute_frame, score_recommendations
from src.visualization import (
    build_recommendation_map,
    build_recommendation_summary,
    build_short_reco_label,
    build_top_rank_chart,
    build_visualization_gallery,
    format_korean_money,
)


PAGE_TITLE = "서울 신혼부부 전월세·매매 추천 대시보드"
DOC_DIR = PROJECT_ROOT / "docs"
RECOMMENDATION_AUDIT_PATH = DOC_DIR / "RECOMMENDATION_VALUE_AUDIT.md"
SCORING_LINEAGE_PATH = DOC_DIR / "SCORING_DATA_LINEAGE.md"
HOUSEHOLD_OPTIONS = ["1인", "2인 맞벌이"]
AREA_BAND_OPTIONS = {
    "10평대": (10, 19),
    "20평대": (20, 29),
    "30평대": (30, 39),
    "40평대+": (40, 45),
}


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
    st.session_state["budget_cap"] = int(persona_row["deposit_budget_cap_krw"])
    st.session_state["monthly_budget_cap"] = int(persona_row["monthly_budget_cap_krw"])


def _get_applied_weights() -> dict[str, int]:
    if "applied_weights_pct" not in st.session_state:
        st.session_state["applied_weights_pct"] = {
            "price": int(FIXED_WEIGHTS["price"] * 100),
            "commute": int(FIXED_WEIGHTS["commute"] * 100),
            "infra": int(FIXED_WEIGHTS["infra"] * 100),
            "safety": int(FIXED_WEIGHTS["safety"] * 100),
            "risk": int(FIXED_WEIGHTS["risk"] * 100),
        }
    return dict(st.session_state["applied_weights_pct"])


def _ensure_persona_state(persona_profiles: pd.DataFrame) -> pd.Series | None:
    if persona_profiles.empty:
        return None

    persona_options = persona_profiles["persona_name"].tolist()
    selected_persona = st.sidebar.selectbox("페르소나 선택", persona_options, key="persona_name")
    persona_row = persona_profiles.loc[persona_profiles["persona_name"].eq(selected_persona)].iloc[0]

    if st.session_state.get("last_persona_name") != selected_persona:
        _apply_persona_defaults(persona_row)
        st.session_state["last_persona_name"] = selected_persona

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
    persona_row = _ensure_persona_state(bundle.get("persona_profiles", pd.DataFrame()))
    available_years = _resolve_available_years(bundle)
    applied_weights_pct = _get_applied_weights()

    selected_year = st.sidebar.selectbox("기준 연도", available_years, index=0)
    household_type = st.sidebar.selectbox("가구 유형", HOUSEHOLD_OPTIONS, index=1)
    workplace_name = st.sidebar.selectbox("직장 위치 1", list(WORKPLACE_HUBS.keys()), index=0)

    secondary_workplace_name = None
    if household_type.startswith("2"):
        secondary_workplace_name = st.sidebar.selectbox(
            "직장 위치 2",
            list(WORKPLACE_HUBS.keys()),
            index=1 if len(WORKPLACE_HUBS) > 1 else 0,
        )

    area_band = st.sidebar.segmented_control("평형대", list(AREA_BAND_OPTIONS.keys()), default="20평대")
    min_area_pyeong, max_area_pyeong = st.sidebar.slider(
        "희망 평수 구간",
        10,
        45,
        AREA_BAND_OPTIONS[area_band],
        step=1,
    )

    budget_cap = st.sidebar.slider("전세 보증금 예산", 100_000_000, 1_500_000_000, key="budget_cap", step=50_000_000)
    monthly_budget_cap = st.sidebar.slider("월세 예산", 300_000, 4_000_000, key="monthly_budget_cap", step=100_000)
    st.sidebar.caption(
        f"선택 금액: 전세 {format_korean_money(budget_cap)} / 월세 {format_korean_money(monthly_budget_cap)}"
    )
    st.sidebar.markdown("#### 점수 가중치")
    with st.sidebar.form("weight_form"):
        price_weight = st.number_input("가격(%)", min_value=0, max_value=100, value=applied_weights_pct["price"], step=1)
        commute_weight = st.number_input("통근(%)", min_value=0, max_value=100, value=applied_weights_pct["commute"], step=1)
        infra_weight = st.number_input("인프라(%)", min_value=0, max_value=100, value=applied_weights_pct["infra"], step=1)
        safety_weight = st.number_input("치안(%)", min_value=0, max_value=100, value=applied_weights_pct["safety"], step=1)
        risk_weight = st.number_input("전세가율(%)", min_value=0, max_value=100, value=applied_weights_pct["risk"], step=1)
        submitted = st.form_submit_button("가중치 확정", use_container_width=True)

    pending_total = int(price_weight + commute_weight + infra_weight + safety_weight + risk_weight)
    if submitted:
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
        "budget_cap": budget_cap,
        "monthly_budget_cap": monthly_budget_cap,
        "weights": {key: value / 100 for key, value in applied_weights_pct.items()},
        "weights_pct": applied_weights_pct,
    }
    return persona_row, state


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
    commute_frame, commute_meta = prepare_commute_frame(
        ui["workplace_name"],
        feature_table,
        bundle["commute_models"],
        household_type=ui["household_type"],
        secondary_workplace_name=ui["secondary_workplace_name"],
    )
    recommendations, scoring_meta = score_recommendations(
        feature_table=feature_table,
        selected_gus=[],
        commute_frame=commute_frame,
        weights=ui["weights"],
        missing_strategy="mean",
        household_type=ui["household_type"],
    )

    return {
        "feature_table": feature_table,
        "feature_meta": feature_meta,
        "commute_frame": commute_frame,
        "commute_meta": commute_meta,
        "recommendations": recommendations,
        "scoring_meta": scoring_meta,
    }


def _render_summary_tab(
    recommendations: pd.DataFrame,
    recommendation_summary: pd.DataFrame,
    recommendation_map,
    rank_chart,
    persona_row: pd.Series | None,
    ui: dict[str, object],
) -> None:
    st.markdown('<div class="section-title">추천 요약</div>', unsafe_allow_html=True)
    if persona_row is not None:
        st.caption(
            f"선택 페르소나: {persona_row['persona_name']} | "
            f"월소득 추정 {format_korean_money(persona_row['monthly_income_estimate_krw'])} | "
            f"부채 추정 {format_korean_money(persona_row['debt_balance_estimate_krw'])}"
        )
    st.caption(
        f"적용 가중치: 가격 {ui['weights_pct']['price']}% / 통근 {ui['weights_pct']['commute']}% / "
        f"인프라 {ui['weights_pct']['infra']}% / 치안 {ui['weights_pct']['safety']}% / 전세가율 {ui['weights_pct']['risk']}%"
    )

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
        st.plotly_chart(recommendation_map, width="stretch")
    with right:
        st.plotly_chart(rank_chart, width="stretch")
    st.dataframe(recommendation_summary, width="stretch", height=320)


def _render_compare_tab(recommendations: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">구별 상세 비교</div>', unsafe_allow_html=True)
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


def main() -> None:
    try:
        bundle = load_dataset_bundle()
    except Exception as exc:
        _show_data_load_error(exc)
        st.stop()

    _show_intro(bundle)
    persona_row, ui = _collect_sidebar_inputs(bundle)
    outputs = _compute_outputs(bundle, ui)

    recommendations = outputs["recommendations"]
    feature_table = outputs["feature_table"]
    persona_simulation = (
        build_persona_simulation(recommendations, persona_row) if persona_row is not None else recommendations.copy()
    )

    gallery = build_visualization_gallery(feature_table, recommendations, bundle, ui["selected_year"])
    recommendation_summary = build_recommendation_summary(recommendations, ui["household_type"])
    recommendation_map = build_recommendation_map(
        recommendations,
        ui["workplace_name"],
        ui["secondary_workplace_name"],
    )
    rank_chart = build_top_rank_chart(recommendations)
    rent_distribution = _build_rent_distribution_chart(feature_table)
    score_distribution = _build_score_distribution_chart(recommendations)
    recommendation_audit_md = _read_markdown(RECOMMENDATION_AUDIT_PATH)
    scoring_lineage_md = _read_markdown(SCORING_LINEAGE_PATH)

    tabs = st.tabs(
        [
            "추천 요약",
            "구별 상세 비교",
            "인프라·입지 분석",
            "치안·재개발 분석",
            "페르소나 구매 시뮬레이션",
            "데이터 근거",
        ]
    )

    with tabs[0]:
        _render_summary_tab(
            recommendations,
            recommendation_summary,
            recommendation_map,
            rank_chart,
            persona_row,
            ui,
        )

    with tabs[1]:
        _render_compare_tab(recommendations)
        st.plotly_chart(gallery["score_stacked_bar"], width="stretch")

    with tabs[2]:
        st.markdown('<div class="section-title">인프라·입지 분석</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(gallery["infra_bar"], width="stretch")
            st.plotly_chart(gallery["infra_scatter"], width="stretch")
        with c2:
            st.plotly_chart(gallery["infra_score_bar"], width="stretch")
            st.plotly_chart(gallery["recommendation_bubble"], width="stretch")

    with tabs[3]:
        st.markdown('<div class="section-title">치안·재개발 분석</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(gallery["safety_dual_axis"], width="stretch")
            st.plotly_chart(gallery["crime_vs_police"], width="stretch")
        with c2:
            st.plotly_chart(gallery["redevelopment_stage_bar"], width="stretch")
            st.plotly_chart(gallery["redevelopment_vs_score"], width="stretch")

    with tabs[4]:
        _render_persona_tab(persona_row, persona_simulation)

    with tabs[5]:
        _render_data_tab(
            feature_table,
            recommendations,
            rent_distribution,
            score_distribution,
            recommendation_audit_md,
            scoring_lineage_md,
        )


if __name__ == "__main__":
    main()
