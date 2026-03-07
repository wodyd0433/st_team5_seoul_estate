from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.config import GU_CENTERS, WORKPLACE_HUBS


def format_korean_money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(round(float(value))):,}원"


def build_short_reco_label(row: pd.Series, persona_name: str) -> str:
    status = row.get("buying_status_3y")
    if pd.notna(status):
        return f"{persona_name} 기준 3년 후 {status}"
    return f"{persona_name} 기준 상위 추천"


def _apply_common_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f7f4ed"},
        margin={"l": 30, "r": 20, "t": 50, "b": 30},
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    return fig


def build_visualization_gallery(
    feature_table: pd.DataFrame,
    recommendations: pd.DataFrame,
    bundle: dict[str, object],
    selected_year: int,
) -> dict[str, go.Figure]:
    del feature_table, bundle, selected_year

    figs: dict[str, go.Figure] = {}

    infra_top = recommendations.head(10)[["gu", "hospital_count", "park_count", "mart_count"]].copy()
    infra_long = infra_top.melt(id_vars="gu", var_name="지표", value_name="값")
    infra_long["지표"] = infra_long["지표"].map(
        {
            "hospital_count": "병원 수",
            "park_count": "공원 수",
            "mart_count": "대형마트 수",
        }
    )
    figs["infra_bar"] = _apply_common_layout(
        px.bar(infra_long, x="gu", y="값", color="지표", barmode="group", title="상위 추천 지역 인프라 비교")
    )
    figs["infra_bar"].update_xaxes(title="자치구")
    figs["infra_bar"].update_yaxes(title="시설 수")

    figs["infra_scatter"] = _apply_common_layout(
        px.scatter(
            recommendations.head(15),
            x="hospital_count",
            y="park_count",
            size="mart_count",
            color="total_score",
            hover_name="gu",
            color_continuous_scale="Blues",
            title="병원·공원·대형마트 분포",
            labels={"hospital_count": "병원 수", "park_count": "공원 수", "mart_count": "대형마트 수", "total_score": "종합점수"},
        )
    )

    infra_score_view = recommendations.head(15)[["gu", "infra_score"]].copy().sort_values("infra_score", ascending=False)
    figs["infra_score_bar"] = _apply_common_layout(
        px.bar(
            infra_score_view,
            x="gu",
            y="infra_score",
            color="infra_score",
            color_continuous_scale="Teal",
            title="인프라 종합점수",
            labels={"gu": "자치구", "infra_score": "인프라 점수"},
        )
    )
    figs["infra_score_bar"].update_layout(coloraxis_showscale=False)

    safety_top = recommendations.head(15).sort_values("crime_total_count", ascending=False)
    safety_fig = make_subplots(specs=[[{"secondary_y": True}]])
    safety_fig.add_trace(
        go.Bar(name="범죄 발생 건수", x=safety_top["gu"], y=safety_top["crime_total_count"], marker_color="#8ec5fc"),
        secondary_y=False,
    )
    safety_fig.add_trace(
        go.Scatter(
            name="경찰 만족도",
            x=safety_top["gu"],
            y=safety_top["police_satisfaction_score"],
            mode="lines+markers",
            line={"color": "#ffb3c1", "width": 3},
        ),
        secondary_y=True,
    )
    safety_fig.update_layout(title="치안 비교")
    safety_fig.update_xaxes(title="자치구")
    safety_fig.update_yaxes(title_text="범죄 발생 건수", secondary_y=False)
    safety_fig.update_yaxes(title_text="경찰 만족도", secondary_y=True)
    figs["safety_dual_axis"] = _apply_common_layout(safety_fig)

    figs["crime_vs_police"] = _apply_common_layout(
        px.scatter(
            recommendations.head(15),
            x="crime_total_count",
            y="police_satisfaction_score",
            color="total_score",
            size="redevelopment_count",
            hover_name="gu",
            title="범죄 발생 건수와 경찰 만족도",
            labels={
                "crime_total_count": "범죄 발생 건수",
                "police_satisfaction_score": "경찰 만족도",
                "redevelopment_count": "정비사업 구역 수",
                "total_score": "종합점수",
            },
        )
    )

    redevelopment_view = recommendations.head(15)[["gu", "redevelopment_count", "active_stage_count"]].copy()
    redevelopment_long = redevelopment_view.melt(id_vars="gu", var_name="지표", value_name="값")
    redevelopment_long["지표"] = redevelopment_long["지표"].map(
        {
            "redevelopment_count": "정비사업 구역 수",
            "active_stage_count": "진행 단계 수",
        }
    )
    figs["redevelopment_stage_bar"] = _apply_common_layout(
        px.bar(redevelopment_long, x="gu", y="값", color="지표", barmode="group", title="정비사업 단계별 현황")
    )
    figs["redevelopment_stage_bar"].update_xaxes(title="자치구")
    figs["redevelopment_stage_bar"].update_yaxes(title="건수")

    figs["redevelopment_vs_score"] = _apply_common_layout(
        px.scatter(
            recommendations,
            x="redevelopment_count",
            y="total_score",
            size="active_stage_count",
            color="gu",
            title="정비사업과 종합점수 관계",
            labels={"redevelopment_count": "정비사업 구역 수", "total_score": "종합점수", "active_stage_count": "진행 단계 수"},
        )
    )

    stacked = recommendations.head(10).melt(
        id_vars=["gu"],
        value_vars=["budget_score", "infra_score", "safety_score", "commute_score"],
        var_name="구성요소",
        value_name="점수",
    )
    stacked["구성요소"] = stacked["구성요소"].map(
        {
            "budget_score": "예산 점수",
            "infra_score": "인프라 점수",
            "safety_score": "치안 점수",
            "commute_score": "통근 점수",
        }
    )
    figs["score_stacked_bar"] = _apply_common_layout(
        px.bar(stacked, x="gu", y="점수", color="구성요소", title="종합점수 구성 비교")
    )
    figs["score_stacked_bar"].update_xaxes(title="자치구")

    commute_target = (
        ["primary_commute_minutes", "secondary_commute_minutes"]
        if "secondary_commute_minutes" in recommendations.columns
        else ["commute_minutes"]
    )
    figs["commute_bar"] = _apply_common_layout(
        px.bar(
            recommendations.head(15),
            x="gu",
            y=commute_target,
            barmode="group",
            title="통근시간 비교",
            labels={
                "value": "통근시간(분)",
                "variable": "항목",
                "primary_commute_minutes": "직장1 통근시간",
                "secondary_commute_minutes": "직장2 통근시간",
                "commute_minutes": "통근시간",
            },
        )
    )

    figs["recommendation_bubble"] = _apply_common_layout(
        px.scatter(
            recommendations,
            x="deposit_price_krw",
            y="monthly_rent_active_krw",
            size="total_score",
            color="infra_score",
            hover_name="gu",
            title="전세 보증금과 월세 부담 비교",
            labels={"deposit_price_krw": "전세 보증금", "monthly_rent_active_krw": "월세", "infra_score": "인프라 점수"},
        )
    )
    figs["recommendation_bubble"].update_xaxes(tickformat=",")
    figs["recommendation_bubble"].update_yaxes(tickformat=",")

    return figs


def build_recommendation_summary(recommendations: pd.DataFrame, household_type: str) -> pd.DataFrame:
    top = recommendations.head(5).copy()
    summary = pd.DataFrame(
        {
            "자치구": top["gu"],
            "종합점수": top["total_score"].round(1),
            "예산 점수": top["budget_score"].round(1),
            "인프라 점수": top["infra_score"].round(1),
            "치안 점수": top["safety_score"].round(1),
            "통근 점수": top["commute_score"].round(1),
            "면적": (
                top["selected_area_min_m2"].round(1).astype(str)
                + "~"
                + top["selected_area_max_m2"].round(1).astype(str)
                + "㎡ / "
                + top["selected_area_min_pyeong"].astype(int).astype(str)
                + "~"
                + top["selected_area_max_pyeong"].astype(int).astype(str)
                + "평"
            ),
            "전세 보증금": top["deposit_price_krw"].map(format_korean_money),
            "월세": top["monthly_rent_active_krw"].map(format_korean_money),
        }
    )
    if str(household_type).startswith("2"):
        summary["직장1 통근시간"] = top["primary_commute_minutes"].map(lambda x: f"{x:.1f}분" if pd.notna(x) else "-")
        summary["직장2 통근시간"] = top["secondary_commute_minutes"].map(lambda x: f"{x:.1f}분" if pd.notna(x) else "-")
        summary["최장 통근시간"] = top["worst_commute_minutes"].map(lambda x: f"{x:.1f}분" if pd.notna(x) else "-")
    else:
        summary["통근시간"] = top["commute_minutes"].map(lambda x: f"{x:.1f}분" if pd.notna(x) else "-")
    return summary


def build_recommendation_map(
    recommendations: pd.DataFrame,
    workplace_name: str,
    secondary_workplace_name: str | None = None,
) -> go.Figure:
    all_points = recommendations.copy()
    all_points["lat"] = all_points["gu"].map(lambda x: GU_CENTERS.get(x, {}).get("lat"))
    all_points["lon"] = all_points["gu"].map(lambda x: GU_CENTERS.get(x, {}).get("lon"))
    all_points = all_points.dropna(subset=["lat", "lon"])

    top = recommendations.head(5).copy()
    top["lat"] = top["gu"].map(lambda x: GU_CENTERS.get(x, {}).get("lat"))
    top["lon"] = top["gu"].map(lambda x: GU_CENTERS.get(x, {}).get("lon"))
    top = top.dropna(subset=["lat", "lon"])

    fig = go.Figure()
    fig.add_trace(
        go.Scattermapbox(
            lat=all_points["lat"],
            lon=all_points["lon"],
            mode="markers",
            marker={"size": 14, "color": all_points["total_score"], "colorscale": "Viridis", "opacity": 0.45},
            customdata=list(
                zip(
                    all_points["gu"],
                    all_points["total_score"].round(1),
                    all_points["deposit_price_krw"].map(format_korean_money),
                    all_points["monthly_rent_active_krw"].map(format_korean_money),
                )
            ),
            hovertemplate="<b>%{customdata[0]}</b><br>종합점수 %{customdata[1]}점<br>전세 %{customdata[2]}<br>월세 %{customdata[3]}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattermapbox(
            lat=top["lat"],
            lon=top["lon"],
            mode="markers+text",
            text=top["gu"],
            textposition="top center",
            marker={"size": 24, "color": "#f9c74f"},
            hovertemplate="<b>%{text}</b><extra></extra>",
            showlegend=False,
        )
    )

    primary = WORKPLACE_HUBS[workplace_name]
    fig.add_trace(
        go.Scattermapbox(
            lat=[primary["lat"]],
            lon=[primary["lon"]],
            mode="markers+text",
            marker={"size": 18, "color": "#ff595e"},
            text=[f"직장1: {primary['label']}"],
            textposition="top right",
            showlegend=False,
        )
    )

    if secondary_workplace_name:
        secondary = WORKPLACE_HUBS[secondary_workplace_name]
        fig.add_trace(
            go.Scattermapbox(
                lat=[secondary["lat"]],
                lon=[secondary["lon"]],
                mode="markers+text",
                marker={"size": 18, "color": "#1982c4"},
                text=[f"직장2: {secondary['label']}"],
                textposition="top right",
                showlegend=False,
            )
        )

    fig.update_layout(
        title="추천 자치구 지도",
        mapbox={"style": "carto-positron", "center": {"lat": 37.5665, "lon": 126.9780}, "zoom": 10},
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
    )
    return fig


def build_top_rank_chart(recommendations: pd.DataFrame) -> go.Figure:
    top = recommendations.head(15).sort_values("total_score", ascending=True)
    fig = px.bar(
        top,
        x="total_score",
        y="gu",
        orientation="h",
        color="total_score",
        color_continuous_scale="Viridis",
        title="종합점수 순위",
        text=top["total_score"].round(1),
    )
    fig.update_layout(yaxis_title="자치구", xaxis_title="종합점수", coloraxis_showscale=False)
    return _apply_common_layout(fig)


def render_figure_grid(figures: dict[str, go.Figure]) -> None:
    names = list(figures.keys())
    for idx in range(0, len(names), 2):
        cols = st.columns(2)
        for offset in range(2):
            if idx + offset >= len(names):
                continue
            key = names[idx + offset]
            with cols[offset]:
                st.plotly_chart(figures[key], width="stretch", key=f"gallery_{key}_{idx + offset}")
