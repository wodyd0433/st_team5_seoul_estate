from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from plotly.subplots import make_subplots

from src.config import GU_CENTERS, WORKPLACE_HUBS


def format_korean_money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(round(float(value))):,}원"


def build_short_reco_label(row: pd.Series, label: str) -> str:
    grade = row.get("total_grade")
    if pd.notna(grade):
        return f"{label} 기준 {grade}등급 추천"
    return f"{label} 기준 추천"


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
        px.bar(infra_long, x="gu", y="값", color="지표", barmode="group", title="상위 추천지 인프라 비교")
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
            title="병원·공원·마트 분포",
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
            title="인프라 점수",
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
        px.bar(redevelopment_long, x="gu", y="값", color="지표", barmode="group", title="재개발 단계별 현황")
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
        value_vars=["price_score", "commute_score", "infra_score", "safety_score", "risk_score"],
        var_name="구성요소",
        value_name="점수",
    )
    stacked["구성요소"] = stacked["구성요소"].map(
        {
            "price_score": "가격 점수",
            "commute_score": "통근 점수",
            "infra_score": "인프라 점수",
            "safety_score": "치안 점수",
            "risk_score": "전세가율 리스크 점수",
        }
    )
    figs["score_stacked_bar"] = _apply_common_layout(
        px.bar(stacked, x="gu", y="점수", color="구성요소", title="종합점수 구성 비교")
    )
    figs["score_stacked_bar"].update_xaxes(title="자치구")

    commute_target = [
        column
        for column in ["primary_commute_minutes", "secondary_commute_minutes", "commute_minutes"]
        if column in recommendations.columns
    ]
    commute_bar_frame = recommendations.head(15)[["gu", *commute_target]].copy()
    for column in commute_target:
        commute_bar_frame[column] = pd.to_numeric(commute_bar_frame[column], errors="coerce")
    commute_bar_frame = commute_bar_frame.melt(
        id_vars=["gu"],
        value_vars=commute_target,
        var_name="항목",
        value_name="통근시간(분)",
    ).dropna(subset=["통근시간(분)"])
    commute_bar_frame["항목"] = commute_bar_frame["항목"].map(
        {
            "primary_commute_minutes": "직장1 통근시간",
            "secondary_commute_minutes": "직장2 통근시간",
            "commute_minutes": "통근시간",
        }
    )
    figs["commute_bar"] = _apply_common_layout(
        px.bar(
            commute_bar_frame,
            x="gu",
            y="통근시간(분)",
            color="항목",
            barmode="group",
            title="통근 시간 비교",
        )
    )

    # 각 지표별 전체 순위 차트 생성 (사용자 요청: 모든 자치구 순위 확인)
    for metric_name, label in {
        "price_score": "가격 점수",
        "commute_score": "통근 점수",
        "infra_score": "인프라 점수",
        "safety_score": "치안 점수",
        "risk_score": "전세가율 점수"
    }.items():
        if metric_name in recommendations.columns:
            view = recommendations[["gu", metric_name]].copy().sort_values(metric_name, ascending=False)
            figs[f"ranking_{metric_name}"] = _apply_common_layout(
                px.bar(
                    view, 
                    x=metric_name, 
                    y="gu", 
                    orientation="h", 
                    title=f"{label} 순위 (전체)",
                    color=metric_name,
                    color_continuous_scale="Viridis",
                    labels={metric_name: "점수", "gu": "자치구"}
                )
            )
            figs[f"ranking_{metric_name}"].update_layout(coloraxis_showscale=False, yaxis={'categoryorder':'total ascending'})

    figs["recommendation_bubble"] = _apply_common_layout(
        px.scatter(
            recommendations,
            x="deposit_price_krw",
            y="monthly_rent_active_krw",
            size="total_score",
            color="risk_score",
            hover_name="gu",
            title="전세 보증금과 월세 부담 비교",
            labels={"deposit_price_krw": "전세 보증금", "monthly_rent_active_krw": "월세", "risk_score": "전세가율 리스크 점수"},
        )
    )
    figs["recommendation_bubble"].update_xaxes(tickformat=",")
    figs["recommendation_bubble"].update_yaxes(tickformat=",")

    return figs


    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="rgba(148, 163, 184, 0.2)",
                linecolor="rgba(148, 163, 184, 0.2)",
            ),
            angularaxis=dict(
                gridcolor="rgba(148, 163, 184, 0.2)",
                linecolor="rgba(148, 163, 184, 0.2)",
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        title="상위 추천 단지 5대 지표 비교 (0~100)",
        legend=dict(orientation="h", y=-0.2),
    )
    return _apply_common_layout(fig)


def build_commute_timeseries_chart(frame: pd.DataFrame, destination_name: str) -> go.Figure:
    view = frame.copy()
    if view.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{destination_name} 시간대별 구단위 평균 소요시간")
        return _apply_common_layout(fig)

    view = view.sort_values(["gu", "time_order"]).reset_index(drop=True)
    ordered_labels = (
        view[["time_order", "time_label"]]
        .drop_duplicates()
        .sort_values("time_order")["time_label"]
        .tolist()
    )

    fig = px.line(
        view,
        x="time_label",
        y="avg_minutes",
        color="gu",
        markers=True,
        title=f"{destination_name} 시간대별 구단위 평균 소요시간",
        labels={"time_label": "시간", "avg_minutes": "평균 소요시간(분)", "gu": "자치구"},
    )
    fig.update_traces(mode="lines+markers")
    fig.update_xaxes(categoryorder="array", categoryarray=ordered_labels)
    fig.update_yaxes(ticksuffix="분")
    return _apply_common_layout(fig)


def build_recommendation_summary(
    recommendations: pd.DataFrame, 
    household_type: str,
    desired_contract_type: str = "전세"
) -> pd.DataFrame:

    top = recommendations.head(5).copy()
    # 공통 항목 구성
    data = {
        "자치구": top["gu"],
        "종합점수": top["total_score"].round(1),
        "등급": top["total_grade"],
        "종합별점": top["total_star_label"],
        "가격 점수": top["price_score"].round(1),
        "통근 점수": top["commute_score"].round(1),
        "인프라 점수": top["infra_score"].round(1),
        "치안 점수": top["safety_score"].round(1),
        "전세가율 점수": top["risk_score"].round(1),
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
    }

    if desired_contract_type == "전세":
        data["전세 보증금"] = top["deposit_price_krw"].map(format_korean_money)
        data["전세가율 (%)"] = top["jeonse_ratio_pct"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
    else:
        data["보증금"] = top["deposit_price_krw"].map(format_korean_money)
        data["월세 (표준화)"] = top["standardized_monthly_rent_krw"].map(format_korean_money)

    summary = pd.DataFrame(data)
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
) -> pdk.Deck:
    def _score_to_rgba(score: float | int | None, alpha: int) -> list[int]:
        if score is None or pd.isna(score):
            return [148, 163, 184, alpha]
        normalized = max(0.0, min(float(score) / 100.0, 1.0))
        return [
            int(245 - normalized * 95),
            int(120 + normalized * 70),
            int(98 + normalized * 45),
            alpha,
        ]

    all_points = recommendations.copy()
    all_points["lat"] = all_points["gu"].map(lambda x: GU_CENTERS.get(x, {}).get("lat"))
    all_points["lon"] = all_points["gu"].map(lambda x: GU_CENTERS.get(x, {}).get("lon"))
    all_points = all_points.dropna(subset=["lat", "lon"])
    all_points["fill_color"] = all_points["total_score"].map(lambda value: _score_to_rgba(value, 180))
    all_points["deposit_label"] = all_points["deposit_price_krw"].map(format_korean_money)
    all_points["monthly_label"] = all_points["monthly_rent_active_krw"].map(format_korean_money)
    all_points["total_score"] = all_points["total_score"].round(1)

    top = recommendations.head(5).copy()
    top["lat"] = top["gu"].map(lambda x: GU_CENTERS.get(x, {}).get("lat"))
    top["lon"] = top["gu"].map(lambda x: GU_CENTERS.get(x, {}).get("lon"))
    top = top.dropna(subset=["lat", "lon"])
    top["fill_color"] = [[255, 159, 28, 235] for _ in range(len(top))]
    top["label"] = top["gu"]

    primary = WORKPLACE_HUBS[workplace_name]
    workplaces = [
        {
            "label": f"직장1: {primary['label']}",
            "lat": primary["lat"],
            "lon": primary["lon"],
            "fill_color": [229, 57, 53, 240],
        }
    ]

    if secondary_workplace_name:
        secondary = WORKPLACE_HUBS[secondary_workplace_name]
        workplaces.append(
            {
                "label": f"직장2: {secondary['label']}",
                "lat": secondary["lat"],
                "lon": secondary["lon"],
                "fill_color": [25, 130, 196, 240],
            }
        )

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=all_points,
            get_position="[lon, lat]",
            get_radius=900,
            get_fill_color="fill_color",
            get_line_color=[255, 255, 255, 40],
            line_width_min_pixels=1,
            stroked=True,
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=top,
            get_position="[lon, lat]",
            get_radius=1550,
            get_fill_color="fill_color",
            get_line_color=[255, 244, 214, 220],
            line_width_min_pixels=2,
            stroked=True,
            pickable=True,
        ),
        pdk.Layer(
            "TextLayer",
            data=top,
            get_position="[lon, lat]",
            get_text="label",
            get_size=14,
            get_color=[33, 37, 41, 230],
            get_alignment_baseline="'top'",
            get_pixel_offset=[0, 18],
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=workplaces,
            get_position="[lon, lat]",
            get_radius=1850,
            get_fill_color="fill_color",
            get_line_color=[255, 255, 255, 220],
            line_width_min_pixels=2,
            stroked=True,
            pickable=True,
        ),
        pdk.Layer(
            "TextLayer",
            data=workplaces,
            get_position="[lon, lat]",
            get_text="label",
            get_size=14,
            get_color=[34, 34, 34, 230],
            get_alignment_baseline="'top'",
            get_pixel_offset=[0, 18],
        ),
    ]

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=37.5665, longitude=126.9780, zoom=10.2, pitch=0),
        map_provider="carto",
        map_style="road",
        height=540,
        tooltip={
            "html": "<b>{gu}</b><br/>종합점수 {total_score}점<br/>등급 {total_grade}<br/>전세 {deposit_label}<br/>월세 {monthly_label}",
            "style": {"backgroundColor": "rgba(15,23,42,0.92)", "color": "white"},
        },
    )


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
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title="종합점수")
    fig.update_yaxes(title="자치구")
    return _apply_common_layout(fig)
