from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import DATA_DIR, DATA_DIR_CANDIDATES, WORKPLACE_HUBS
from src.feature_engineering import build_feature_table
from src.io_utils import load_dataset_bundle
from src.scoring_engine import prepare_commute_frame, score_recommendations
from src.visualization import (
    build_short_reco_label,
    build_recommendation_map,
    build_recommendation_summary,
    build_top_rank_chart,
    build_visualization_gallery,
    format_korean_money,
    render_figure_grid,
)


st.set_page_config(
    page_title="서울 신혼부부 전월세 최적 입지 추천",
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
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .hero-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px 20px;
        min-height: 180px;
    }
    .hero-rank {
        color: #f9c74f;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .hero-gu {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .hero-score {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 12px;
    }
    .hero-meta {
        color: #d8d4cb;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .section-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0.4rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_krw(value: float | int | None) -> str:
    return format_korean_money(value)


def main() -> None:
    st.title("서울 신혼부부 전월세 최적 입지 추천 대시보드")
    with st.popover("대시보드 기준 설명"):
        st.markdown(
            """
            - `예산 점수`: 보증금 예산 적합도와 월세 부담을 함께 반영합니다.
            - `통근 점수`: 직장까지의 예상 통근시간을 기준으로 산정합니다.
            - `안전 점수`: 범죄 지표와 경찰 만족도 데이터를 결합합니다.
            - `인프라 점수`: 병원, 공원, 생활편의 시설 밀도를 반영합니다.
            - `가중 합산`: 선택한 가중치대로 단순 합산합니다.
            - `균형 보정`: 특정 지표만 높고 나머지가 낮은 지역을 일부 감점합니다.
            - `병목 기준`: 가장 약한 항목을 더 강하게 반영합니다.
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
                    "- Streamlit Cloud에서는 저장소 내부 `data_all/` 또는 `DATA_DIR` 환경변수가 필요합니다.",
                ]
            )
        )
        st.code(str(exc))
        st.stop()

    selected_year = 2025
    household_type = st.sidebar.selectbox("가구 유형", ["1인", "2인(맞벌이)"], index=1)
    workplace_name = st.sidebar.selectbox("직장 위치 1", list(WORKPLACE_HUBS.keys()), index=0)
    secondary_workplace_name = None
    if household_type == "2인(맞벌이)":
        secondary_options = list(WORKPLACE_HUBS.keys())
        secondary_workplace_name = st.sidebar.selectbox("직장 위치 2", secondary_options, index=1 if len(secondary_options) > 1 else 0)
    area_band = st.sidebar.segmented_control("평수대", ["10평대", "20평대", "30평대", "40평대+"], default="20평대")
    area_band_defaults = {
        "10평대": (10, 19),
        "20평대": (20, 29),
        "30평대": (30, 39),
        "40평대+": (40, 45),
    }
    default_area_range = area_band_defaults[area_band]
    area_range_pyeong = st.sidebar.slider("희망 평수 구간", 10, 45, default_area_range, step=1)
    min_area_pyeong, max_area_pyeong = area_range_pyeong
    min_area_m2 = round(min_area_pyeong * 3.3058, 1)
    max_area_m2 = round(max_area_pyeong * 3.3058, 1)
    budget_cap = st.sidebar.slider("보증금 예산", 100_000_000, 1_500_000_000, 700_000_000, step=50_000_000)
    monthly_budget_cap = st.sidebar.slider("월세 예산", 300_000, 4_000_000, 1_200_000, step=100_000)
    scaling_method = st.sidebar.segmented_control("스케일링", ["MinMax", "Z-score", "Percentile"], default="MinMax")
    score_formula = st.sidebar.selectbox("점수 산정 방식", ["가중 합산", "균형 보정", "병목 기준"], index=0)
    family_focus = st.sidebar.selectbox("추천 성향", ["균형형", "통근 우선", "예산 우선", "안전 우선", "인프라 우선"], index=0)

    weight_presets = {
        "균형형": {"budget": 0.32, "infra": 0.24, "safety": 0.22, "commute": 0.22},
        "통근 우선": {"budget": 0.22, "infra": 0.18, "safety": 0.15, "commute": 0.45},
        "예산 우선": {"budget": 0.48, "infra": 0.18, "safety": 0.14, "commute": 0.20},
        "안전 우선": {"budget": 0.22, "infra": 0.18, "safety": 0.40, "commute": 0.20},
        "인프라 우선": {"budget": 0.22, "infra": 0.40, "safety": 0.16, "commute": 0.22},
    }
    default_weights = weight_presets[family_focus]
    st.sidebar.markdown("#### 세부 가중치 조정")
    budget_weight = st.sidebar.slider("예산", 0.0, 1.0, float(default_weights["budget"]), 0.01)
    commute_weight = st.sidebar.slider("통근", 0.0, 1.0, float(default_weights["commute"]), 0.01)
    safety_weight = st.sidebar.slider("안전", 0.0, 1.0, float(default_weights["safety"]), 0.01)
    infra_weight = st.sidebar.slider("인프라", 0.0, 1.0, float(default_weights["infra"]), 0.01)
    weights = {"budget": budget_weight, "commute": commute_weight, "safety": safety_weight, "infra": infra_weight}

    feature_table, feature_meta = build_feature_table(
        bundle=bundle,
        year=selected_year,
        sampling_rate=1.0,
        budget_cap=budget_cap,
        remove_outliers=True,
        monthly_budget_cap=monthly_budget_cap,
        min_area_pyeong=min_area_pyeong,
        max_area_pyeong=max_area_pyeong,
    )
    commute_frame, commute_meta = prepare_commute_frame(
        workplace_name,
        feature_table,
        bundle["commute_models"],
        household_type=household_type,
        secondary_workplace_name=secondary_workplace_name,
    )
    recommendations, score_meta = score_recommendations(
        feature_table=feature_table,
        selected_gus=[],
        commute_frame=commute_frame,
        weights=weights,
        scaling_method=scaling_method,
        missing_strategy="mean",
        score_formula=score_formula,
        household_type=household_type,
    )
    recommendations = recommendations.merge(
        commute_frame[
            [
                "gu",
                "distance_km",
                "secondary_distance_km",
                "workplace_name",
                "secondary_workplace_name",
                "primary_commute_minutes",
                "secondary_commute_minutes",
                "worst_commute_minutes",
            ]
        ],
        on="gu",
        how="left",
    )

    tabs = st.tabs(
        [
            "추천 요약",
            "구별 상세 비교",
            "인프라 심층 분석",
            "안전/개발 가치",
            f"{family_focus} 인사이트",
        ]
    )

    gallery = build_visualization_gallery(feature_table, recommendations, bundle, selected_year)
    recommendation_summary = build_recommendation_summary(recommendations)
    recommendation_map = build_recommendation_map(recommendations, workplace_name, secondary_workplace_name)
    rank_chart = build_top_rank_chart(recommendations)

    with tabs[0]:
        st.markdown('<div class="section-title">신혼부부 추천 Top 5</div>', unsafe_allow_html=True)
        top5_cards = recommendations.head(5).copy()
        card_cols = st.columns(5)
        for idx, (_, row) in enumerate(top5_cards.iterrows()):
            with card_cols[idx]:
                with st.container(border=True):
                    st.markdown(f"**{idx + 1}위**")
                    st.markdown(f"### {row['gu']}")
                    st.markdown(f"## {row['total_score']:.1f}점")
                    st.caption(build_short_reco_label(row, family_focus))
                    st.text(f"{min_area_m2}~{max_area_m2}㎡ / {min_area_pyeong}~{max_area_pyeong}평 기준")
                    st.write(f"전세 {format_korean_money(row['deposit_price_krw'])}")
                    st.write(f"월세 {format_korean_money(row['monthly_rent_active_krw'])}")
                    st.write(f"통근 {row['commute_minutes']:.1f}분")
                    if household_type == "2인(맞벌이)" and pd.notna(row.get("secondary_commute_minutes")):
                        st.caption(f"직장1 {row['primary_commute_minutes']:.1f}분 / 직장2 {row['secondary_commute_minutes']:.1f}분")
        left, right = st.columns([1.05, 1])
        with left:
            st.plotly_chart(recommendation_map, width="stretch", key="recommendation_map")
        with right:
            st.plotly_chart(rank_chart, width="stretch", key="recommendation_rank_chart")
        st.dataframe(recommendation_summary, width="stretch", height=280)

    with tabs[1]:
        compare_cols = [
            "gu", "total_score", "selected_area_min_m2", "selected_area_max_m2", "selected_area_min_pyeong", "selected_area_max_pyeong", "deposit_price_krw", "monthly_rent_active_krw",
            "commute_minutes", "budget_score", "infra_score", "safety_score", "commute_score"
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
                "면적": "면적(㎡ / 평)",
                "deposit_price_krw": "전세",
                "monthly_rent_active_krw": "월세",
                "commute_minutes": "통근시간(분)",
                "budget_score": "예산 점수",
                "infra_score": "인프라 점수",
                "safety_score": "치안 점수",
                "commute_score": "통근 점수",
            }
        )
        st.dataframe(compare, width="stretch", height=420)
        st.plotly_chart(gallery["score_stacked_bar"], width="stretch", key="compare_score_stacked_bar")

    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(gallery["infra_bar"], width="stretch", key="infra_infra_bar")
            st.plotly_chart(gallery["hospital_park_scatter"], width="stretch", key="infra_hospital_park_scatter")
        with c2:
            st.plotly_chart(gallery["amenity_parallel"], width="stretch", key="infra_amenity_parallel")
            st.plotly_chart(gallery["infra_radar"], width="stretch", key="infra_radar")

    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(gallery["safety_bar"], width="stretch", key="safety_bar")
            st.plotly_chart(gallery["crime_vs_police"], width="stretch", key="crime_vs_police")
        with c2:
            st.plotly_chart(gallery["redevelopment_stage_bar"], width="stretch", key="redevelopment_stage_bar")
            st.plotly_chart(gallery["redevelopment_vs_score"], width="stretch", key="redevelopment_vs_score")

    with tabs[4]:
        st.subheader(f"{family_focus} 기준 해석")
        with st.popover("점수 산정 방식 설명"):
            st.markdown(
                f"""
                - 현재 방식: `{score_formula}`
                - 현재 가구 유형: `{household_type}`
                - 현재 평수 기준: `{min_area_pyeong}~{max_area_pyeong}평 ({min_area_m2}~{max_area_m2}㎡)`
                - 통근은 `{workplace_name}`{f" / `{secondary_workplace_name}`" if secondary_workplace_name else ""} 기준으로 계산합니다.
                - `가중 합산`: 선택한 가중치를 그대로 반영합니다.
                - `균형 보정`: 특정 항목 편중 지역을 일부 감점합니다.
                - `병목 기준`: 가장 취약한 항목이 총점에 더 크게 영향을 줍니다.
                """
            )
        insight_cols = st.columns(4)
        insight_cols[0].metric("가구 유형", household_type)
        insight_cols[1].metric("직장 위치", workplace_name if not secondary_workplace_name else f"{workplace_name} / {secondary_workplace_name}")
        insight_cols[2].metric("희망 평수", f"{min_area_pyeong}~{max_area_pyeong}평")
        insight_cols[3].metric("보증금 / 월세", f"{format_krw(budget_cap)} / {format_krw(monthly_budget_cap)}")
        st.dataframe(
            pd.DataFrame(
                [
                    {"기준": "예산", "가중치": weights["budget"]},
                    {"기준": "통근", "가중치": weights["commute"]},
                    {"기준": "안전", "가중치": weights["safety"]},
                    {"기준": "인프라", "가중치": weights["infra"]},
                ]
            ),
            width="stretch",
        )
        st.plotly_chart(gallery["commute_bar"], width="stretch", key="insight_commute_bar")
        st.markdown("### 가구 유형별 해석")
        if str(household_type).startswith("2"):
            st.write("맞벌이 가구는 두 직장의 평균 통근과 최악 통근 부담을 함께 반영합니다. 한 사람만 멀어지는 지역은 점수가 과도하게 높아지지 않도록 보정했습니다.")
        else:
            st.write("1인 가구는 월세 부담과 단일 직장 통근 효율을 더 직접적으로 반영합니다. 초기 현금흐름 부담을 줄이는 지역이 상대적으로 유리합니다.")


if __name__ == "__main__":
    main()
