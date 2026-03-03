from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import DATA_DIR, DATA_DIR_CANDIDATES, WORKPLACE_HUBS
from src.feature_engineering import build_feature_table
from src.io_utils import load_dataset_bundle
from src.persona import build_persona_simulation
from src.scoring_engine import prepare_commute_frame, score_recommendations
from src.visualization import (
    build_recommendation_map,
    build_recommendation_summary,
    build_short_reco_label,
    build_top_rank_chart,
    build_visualization_gallery,
    format_korean_money,
)


st.set_page_config(
    page_title="서울 신혼부부 전월세·매수 추천 대시보드",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


def _apply_persona_defaults(persona_row: pd.Series) -> None:
    st.session_state["budget_weight"] = float(persona_row["weight_budget"])
    st.session_state["commute_weight"] = float(persona_row["weight_commute"])
    st.session_state["safety_weight"] = float(persona_row["weight_safety"])
    st.session_state["infra_weight"] = float(persona_row["weight_infra"])
    st.session_state["budget_cap"] = int(persona_row["deposit_budget_cap_krw"])
    st.session_state["monthly_budget_cap"] = int(persona_row["monthly_budget_cap_krw"])


def _ensure_persona_state(persona_profiles: pd.DataFrame) -> pd.Series | None:
    if persona_profiles.empty:
        return None

    persona_options = persona_profiles["persona_name"].tolist()
    selected_persona = st.sidebar.selectbox("페르소나 선택", persona_options, key="persona_name")
    persona_row = persona_profiles.loc[persona_profiles["persona_name"].eq(selected_persona)].iloc[0]

    if st.session_state.get("last_persona_name") != selected_persona:
        _apply_persona_defaults(persona_row)
        st.session_state["last_persona_name"] = selected_persona

    if st.sidebar.button("페르소나 프리셋 다시 적용", use_container_width=True):
        _apply_persona_defaults(persona_row)

    return persona_row


def main() -> None:
    st.title("서울 신혼부부 전월세·매수 추천 대시보드")
    with st.popover("대시보드 기준 설명"):
        st.markdown(
            """
            - `예산 점수`: 전세 보증금과 월세 부담을 함께 반영합니다.
            - `통근 점수`: 주요 직장까지의 예상 통근시간을 기준으로 계산합니다.
            - `치안 점수`: 범죄 발생 건수와 경찰 만족도를 함께 반영합니다.
            - `인프라 점수`: 병원 수, 공원 수, 대형마트 수를 함께 반영합니다.
            - `페르소나`: 신혼부부 소득·부채 통계를 바탕으로 기본 가중치와 예산 한도를 제안합니다.
            - `사용자 직접 조정`: 페르소나를 불러온 뒤에도 가중치와 예산은 직접 수정할 수 있습니다.
            """
        )

    try:
        bundle = load_dataset_bundle()
    except Exception as exc:
        st.error("데이터 파일을 불러오지 못했습니다.")
        st.markdown(
            "\n".join(
                [
                    f"- 현재 데이터 폴더: `{DATA_DIR}`",
                    "- 우선 탐색 경로:",
                    *[f"  - `{candidate}`" for candidate in DATA_DIR_CANDIDATES],
                ]
            )
        )
        st.code(str(exc))
        st.stop()

    if bundle.get("data_mode") == "compact":
        st.caption("현재 경량 배포 데이터로 실행 중입니다. 원천 데이터 기반 상세 분석 대신 자치구 집계와 페르소나 프리셋을 사용합니다.")

    persona_row = _ensure_persona_state(bundle.get("persona_profiles", pd.DataFrame()))

    selected_year = 2025
    household_type = st.sidebar.selectbox("가구 유형", ["1인", "2인 맞벌이"], index=1)
    workplace_name = st.sidebar.selectbox("직장 위치 1", list(WORKPLACE_HUBS.keys()), index=0)
    secondary_workplace_name = None
    if household_type == "2인 맞벌이":
        secondary_workplace_name = st.sidebar.selectbox(
            "직장 위치 2",
            list(WORKPLACE_HUBS.keys()),
            index=1 if len(WORKPLACE_HUBS) > 1 else 0,
        )

    area_band = st.sidebar.segmented_control("평형대", ["10평대", "20평대", "30평대", "40평대+"], default="20평대")
    area_band_defaults = {"10평대": (10, 19), "20평대": (20, 29), "30평대": (30, 39), "40평대+": (40, 45)}
    min_area_pyeong, max_area_pyeong = st.sidebar.slider("희망 평수 구간", 10, 45, area_band_defaults[area_band], step=1)
    min_area_m2 = round(min_area_pyeong * 3.3058, 1)
    max_area_m2 = round(max_area_pyeong * 3.3058, 1)

    budget_cap = st.sidebar.slider("전세 보증금 예산", 100_000_000, 1_500_000_000, key="budget_cap", step=50_000_000)
    monthly_budget_cap = st.sidebar.slider("월세 예산", 300_000, 4_000_000, key="monthly_budget_cap", step=100_000)
    scaling_method = st.sidebar.segmented_control("스케일 방식", ["MinMax", "Z-score", "Percentile"], default="MinMax")
    score_formula = st.sidebar.selectbox("점수 합산 방식", ["가중 합산", "균형 보정", "병목 기준"], index=0)

    st.sidebar.markdown("#### 영역 가중치 조정")
    budget_weight = st.sidebar.slider("예산", 0.0, 1.0, key="budget_weight", step=0.01)
    commute_weight = st.sidebar.slider("통근", 0.0, 1.0, key="commute_weight", step=0.01)
    safety_weight = st.sidebar.slider("치안", 0.0, 1.0, key="safety_weight", step=0.01)
    infra_weight = st.sidebar.slider("인프라", 0.0, 1.0, key="infra_weight", step=0.01)
    weights = {"budget": budget_weight, "commute": commute_weight, "safety": safety_weight, "infra": infra_weight}

    feature_table, _ = build_feature_table(
        bundle=bundle,
        year=selected_year,
        sampling_rate=1.0,
        budget_cap=budget_cap,
        remove_outliers=True,
        monthly_budget_cap=monthly_budget_cap,
        min_area_pyeong=min_area_pyeong,
        max_area_pyeong=max_area_pyeong,
    )
    commute_frame, _ = prepare_commute_frame(
        workplace_name,
        feature_table,
        bundle["commute_models"],
        household_type=household_type,
        secondary_workplace_name=secondary_workplace_name,
    )
    recommendations, _ = score_recommendations(
        feature_table=feature_table,
        selected_gus=[],
        commute_frame=commute_frame,
        weights=weights,
        scaling_method=scaling_method,
        missing_strategy="mean",
        score_formula=score_formula,
        household_type=household_type,
    )

    persona_simulation = build_persona_simulation(recommendations, persona_row) if persona_row is not None else recommendations.copy()
    gallery = build_visualization_gallery(feature_table, recommendations, bundle, selected_year)
    recommendation_summary = build_recommendation_summary(recommendations, household_type)
    recommendation_map = build_recommendation_map(recommendations, workplace_name, secondary_workplace_name)
    rank_chart = build_top_rank_chart(recommendations)

    tabs = st.tabs(
        [
            "추천 요약",
            "구별 상세 비교",
            "인프라 심층 분석",
            "치안·개발 분석",
            "페르소나 구매 시뮬레이션",
        ]
    )

    with tabs[0]:
        st.markdown('<div class="section-title">추천 요약</div>', unsafe_allow_html=True)
        if persona_row is not None:
            st.caption(
                f"선택 페르소나: {persona_row['persona_name']} | "
                f"월소득 추정 {format_korean_money(persona_row['monthly_income_estimate_krw'])} | "
                f"추정부채 {format_korean_money(persona_row['debt_balance_estimate_krw'])}"
            )

        top5_cards = recommendations.head(5).copy()
        card_cols = st.columns(5)
        for idx, (_, row) in enumerate(top5_cards.iterrows()):
            with card_cols[idx]:
                with st.container(border=True):
                    st.markdown(f"**{idx + 1}위**")
                    st.markdown(f"### {row['gu']}")
                    st.markdown(f"## {row['total_score']:.1f}점")
                    st.caption(build_short_reco_label(row, persona_row["persona_name"] if persona_row is not None else "기본"))
                    st.text(f"{min_area_m2}~{max_area_m2}㎡ / {min_area_pyeong}~{max_area_pyeong}평 기준")
                    st.write(f"전세 보증금 {format_korean_money(row['deposit_price_krw'])}")
                    st.write(f"월세 {format_korean_money(row['monthly_rent_active_krw'])}")
                    if household_type == "2인 맞벌이" and pd.notna(row.get("secondary_commute_minutes")):
                        st.write(f"직장1 통근 {row['primary_commute_minutes']:.1f}분")
                        st.write(f"직장2 통근 {row['secondary_commute_minutes']:.1f}분")
                    else:
                        st.write(f"통근 {row['commute_minutes']:.1f}분")

        left, right = st.columns([1.05, 1])
        with left:
            st.plotly_chart(recommendation_map, width="stretch")
        with right:
            st.plotly_chart(rank_chart, width="stretch")
        st.dataframe(recommendation_summary, width="stretch", height=320)

    with tabs[1]:
        st.markdown('<div class="section-title">구별 상세 비교</div>', unsafe_allow_html=True)
        compare_cols = [
            "gu",
            "total_score",
            "selected_area_min_m2",
            "selected_area_max_m2",
            "selected_area_min_pyeong",
            "selected_area_max_pyeong",
            "deposit_price_krw",
            "monthly_rent_active_krw",
            "budget_score",
            "infra_score",
            "safety_score",
            "commute_score",
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
        compare = compare.drop(columns=["selected_area_min_m2", "selected_area_max_m2", "selected_area_min_pyeong", "selected_area_max_pyeong"])
        compare = compare.rename(
            columns={
                "gu": "자치구",
                "total_score": "종합점수",
                "deposit_price_krw": "전세 보증금",
                "monthly_rent_active_krw": "월세",
                "budget_score": "예산 점수",
                "infra_score": "인프라 점수",
                "safety_score": "치안 점수",
                "commute_score": "통근 점수",
                "primary_commute_minutes": "직장1 통근시간",
                "secondary_commute_minutes": "직장2 통근시간",
                "worst_commute_minutes": "최장 통근시간",
            }
        )
        for column in ["직장1 통근시간", "직장2 통근시간", "최장 통근시간"]:
            if column in compare.columns:
                compare[column] = compare[column].map(lambda x: f"{x:.1f}분" if pd.notna(x) else "-")
        st.dataframe(compare, width="stretch", height=430)
        st.plotly_chart(gallery["score_stacked_bar"], width="stretch")

    with tabs[2]:
        st.markdown('<div class="section-title">인프라 심층 분석</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(gallery["infra_bar"], width="stretch")
            st.plotly_chart(gallery["infra_scatter"], width="stretch")
        with c2:
            st.plotly_chart(gallery["infra_score_bar"], width="stretch")
            st.plotly_chart(gallery["recommendation_bubble"], width="stretch")

    with tabs[3]:
        st.markdown('<div class="section-title">치안·개발 분석</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(gallery["safety_dual_axis"], width="stretch")
            st.plotly_chart(gallery["crime_vs_police"], width="stretch")
        with c2:
            st.plotly_chart(gallery["redevelopment_stage_bar"], width="stretch")
            st.plotly_chart(gallery["redevelopment_vs_score"], width="stretch")

    with tabs[4]:
        st.markdown('<div class="section-title">페르소나 구매 시뮬레이션</div>', unsafe_allow_html=True)
        if persona_row is None:
            st.info("페르소나 프로필 데이터가 없어 시뮬레이션을 표시할 수 없습니다.")
        else:
            metrics = st.columns(4)
            metrics[0].metric("월소득 추정", format_korean_money(persona_row["monthly_income_estimate_krw"]))
            metrics[1].metric("추정 부채잔액", format_korean_money(persona_row["debt_balance_estimate_krw"]))
            metrics[2].metric("2년 후 종잣돈", format_korean_money(persona_row["seed_money_2y_krw"]))
            metrics[3].metric("3년 후 실질 매수예산", format_korean_money(persona_row["buying_power_3y_krw"]))
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
                    "sale_price_krw": "대표 매매가",
                    "buying_status_2y": "2년 후 매수 가능성",
                    "buying_status_3y": "3년 후 매수 가능성",
                    "buy_2y_gap_krw": "2년 후 매매가 차이",
                    "buy_3y_gap_krw": "3년 후 매매가 차이",
                }
            )
            st.dataframe(sim_table, width="stretch", height=420)


if __name__ == "__main__":
    main()
