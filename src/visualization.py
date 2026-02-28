from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import GU_CENTERS, WORKPLACE_HUBS


def format_korean_money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    value = float(value)
    eok = int(value // 100_000_000)
    man = int((value % 100_000_000) // 10_000)
    if eok > 0 and man > 0:
        return f"{eok}억 {man:,}만원"
    if eok > 0:
        return f"{eok}억"
    return f"{man:,}만원"


def build_short_reco_label(row: pd.Series, family_focus: str) -> str:
    if family_focus == "통근 우선":
        return "출퇴근 효율 우수"
    if family_focus == "예산 우선":
        return "예산 부담 낮은 선택"
    if family_focus == "안전 우선":
        return "안전 체감이 강한 지역"
    if family_focus == "인프라 우선":
        return "생활 인프라 밀도 우수"
    return "균형감이 좋은 상위 후보"


def build_visualization_gallery(
    feature_table: pd.DataFrame,
    recommendations: pd.DataFrame,
    bundle: dict[str, object],
    selected_year: int,
) -> dict[str, go.Figure]:
    figs: dict[str, go.Figure] = {}
    if bundle.get("is_compact"):
        area_min = int(feature_table["selected_area_min_pyeong"].iloc[0])
        area_max = int(feature_table["selected_area_max_pyeong"].iloc[0])
        compact = bundle["compact_feature_base"].copy()
        compact["year"] = pd.to_numeric(compact["year"], errors="coerce")
        compact["area_pyeong_bucket"] = pd.to_numeric(compact["area_pyeong_bucket"], errors="coerce")
        compact = compact.loc[compact["area_pyeong_bucket"].between(area_min, area_max, inclusive="both")].copy()
        rent_trend = compact.groupby(["year", "gu"])["deposit_price_krw"].mean().reset_index()
        sale_trend = compact.groupby(["year", "gu"])["sale_price_krw"].mean().reset_index()
        current_compact = compact.loc[compact["year"].eq(selected_year)].copy()
        current_compact["display_area_m2"] = current_compact["area_pyeong_bucket"] * 3.3058
    else:
        rent = bundle["rent"].copy()
        rent["year"] = rent["년월"].astype(str).str[:4].astype(int)
        sale = bundle["sale"].copy()
        rent_trend = rent.groupby(["year", "gu"])["보증금_만원_krw"].median().reset_index()
        sale_trend = sale.groupby(["dealYear", "gu"])["dealAmount_krw"].median().reset_index().rename(columns={"dealYear": "year"})
    gap = feature_table[["gu", "sale_rent_gap_krw"]].sort_values("sale_rent_gap_krw", ascending=False)
    rank = recommendations[["gu", "score_rank", "total_score"]].copy()

    rent_trend_y = "deposit_price_krw" if "deposit_price_krw" in rent_trend.columns else "보증금_만원_krw"
    sale_trend_y = "sale_price_krw" if "sale_price_krw" in sale_trend.columns else "dealAmount_krw"
    figs["rent_trend_line"] = px.line(rent_trend, x="year", y=rent_trend_y, color="gu", title="자치구별 전세보증금 추세")
    figs["sale_trend_line"] = px.line(sale_trend, x="year", y=sale_trend_y, color="gu", title="자치구별 매매가 추세")
    figs["rent_sale_gap"] = px.bar(gap, x="gu", y="sale_rent_gap_krw", title="매매-전세 격차")
    if bundle.get("is_compact"):
        figs["price_box"] = px.bar(current_compact, x="gu", y="deposit_price_krw", title="전세보증금 수준")
        figs["area_price_scatter"] = px.scatter(current_compact, x="display_area_m2", y="deposit_price_krw", color="gu", title="면적 대비 전세보증금")
        figs["price_density"] = px.histogram(current_compact, x="deposit_price_krw", nbins=20, color="gu", title="전세보증금 분포")
    else:
        figs["price_box"] = px.box(rent.query("year == @selected_year"), x="gu", y="보증금_만원_krw", title="전세보증금 분포")
        figs["area_price_scatter"] = px.scatter(rent.query("year == @selected_year"), x="전용면적_m2", y="보증금_만원_krw", color="gu", title="면적 대비 전세보증금")
        figs["price_density"] = px.histogram(rent.query("year == @selected_year"), x="보증금_만원_krw", nbins=30, color="gu", title="전세보증금 밀도")
    figs["yearly_rank_change"] = px.scatter(rank, x="score_rank", y="total_score", text="gu", size="total_score", title="자치구 랭킹 변동 차트")
    figs["infra_bar"] = px.bar(feature_table, x="gu", y=["hospital_count", "park_count", "retail_license_count"], barmode="group", title="인프라 현황")
    figs["safety_bar"] = px.bar(feature_table, x="gu", y=["crime_score_proxy", "police_satisfaction_score"], barmode="group", title="치안 비교")
    figs["corr_heatmap"] = px.imshow(feature_table.select_dtypes(include="number").corr().round(2), text_auto=True, title="상관관계 히트맵")
    figs["hospital_park_scatter"] = px.scatter(feature_table, x="hospital_count", y="park_count", size="infra_score_raw", color="gu", title="병원 수 vs 공원 수")
    figs["crime_vs_police"] = px.scatter(feature_table, x="crime_score_proxy", y="police_satisfaction_score", text="gu", title="범죄 지표 vs 경찰 만족도")
    figs["infra_radar"] = go.Figure()
    for _, row in feature_table.head(5).iterrows():
        figs["infra_radar"].add_trace(go.Scatterpolar(r=[row["hospital_count"], row["park_count"], row["retail_license_count"], row["infra_score_raw"]], theta=["병원", "공원", "유통", "종합"], fill="toself", name=row["gu"]))
    figs["infra_radar"].update_layout(title="상위 자치구 인프라 레이더")
    figs["amenity_parallel"] = px.parallel_coordinates(feature_table, color="infra_score_raw", dimensions=["hospital_count", "park_count", "retail_license_count", "infra_score_raw"], title="인프라 평행좌표")
    figs["redevelopment_scatter"] = px.scatter(feature_table, x="redevelopment_count", y="deposit_price_krw", size="infra_score_raw", color="gu", title="정비사업 vs 가격")
    figs["redevelopment_stage_bar"] = px.bar(feature_table, x="gu", y=["redevelopment_count", "active_stage_count"], barmode="group", title="정비사업 단계 수")
    x_col = "redevelopment_score" if "redevelopment_score" in recommendations.columns else "infra_score"
    figs["redevelopment_vs_score"] = px.scatter(recommendations, x=x_col, y="total_score", text="gu", title="개발/생활 점수 vs 총점")
    figs["redevelopment_treemap"] = px.treemap(feature_table, path=["gu"], values="redevelopment_count", color="redevelopment_score_raw", title="정비사업 Treemap")
    figs["top5_score_bar"] = px.bar(recommendations.head(5), x="gu", y="total_score", color="total_score", title="추천 TOP5 총점")
    stacked = recommendations.head(10).melt(
        id_vars=["gu"],
        value_vars=["budget_score", "infra_score", "safety_score", "commute_score"],
        var_name="component",
        value_name="score",
    )
    figs["score_stacked_bar"] = px.bar(stacked, x="gu", y="score", color="component", title="점수 구성 비율 stacked bar")
    figs["recommendation_bubble"] = px.scatter(recommendations, x="deposit_price_krw", y="infra_score_raw", size="total_score", color="gu", hover_data=["commute_minutes"], title="추천 버블 차트")
    figs["budget_fit_bar"] = px.bar(feature_table, x="gu", y="budget_fit", title="예산 적합도")
    figs["monthly_rent_line"] = px.line(feature_table.sort_values("gu"), x="gu", y="monthly_rent_krw", title="월세 중앙값")
    figs["age_proxy_bar"] = px.bar(feature_table, x="gu", y="age_proxy", title="건축연식 프록시")
    figs["sale_count_bar"] = px.bar(feature_table, x="gu", y="sale_txn_count", title="매매 거래량")
    figs["rent_count_bar"] = px.bar(feature_table, x="gu", y="rent_txn_count", title="전월세 거래량")
    figs["commute_bar"] = px.bar(recommendations, x="gu", y="commute_minutes", title="통근 시간")
    figs["price_burden_bar"] = px.bar(feature_table, x="gu", y="price_burden_index", title="예산 부담 지수")
    figs["sale_area_scatter"] = px.scatter(feature_table, x="sale_area_m2", y="sale_price_krw", color="gu", title="매매 면적 vs 가격")
    figs["rent_area_scatter"] = px.scatter(feature_table, x="rent_area_m2", y="deposit_price_krw", color="gu", title="전세 면적 vs 가격")
    figs["score_histogram"] = px.histogram(recommendations, x="total_score", nbins=20, title="추천 점수 분포")
    return figs


def build_recommendation_summary(recommendations: pd.DataFrame) -> pd.DataFrame:
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
            "전세": top["deposit_price_krw"].map(format_korean_money),
            "월세": top["monthly_rent_active_krw"].map(format_korean_money),
            "통근": top["commute_minutes"].map(lambda x: f"{x:.1f}분" if pd.notna(x) else "-"),
        }
    )
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
    score_min = float(all_points["total_score"].min())
    score_max = float(all_points["total_score"].max())
    fig = go.Figure()
    fig.add_trace(
        go.Scattermap(
            lat=all_points["lat"],
            lon=all_points["lon"],
            mode="markers",
            marker={
                "size": 15,
                "color": all_points["total_score"],
                "colorscale": "Viridis",
                "cmin": score_min,
                "cmax": score_max,
                "opacity": 0.45,
                "colorbar": {"title": "점수", "x": 1.03},
            },
            customdata=list(
                zip(
                    all_points["gu"],
                    all_points["total_score"].round(1),
                    all_points["deposit_price_krw"].map(format_korean_money),
                    all_points["monthly_rent_active_krw"].map(format_korean_money),
                    all_points["commute_minutes"].round(1),
                )
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "종합점수 %{customdata[1]}점<br>"
                "전세 %{customdata[2]}<br>"
                "월세 %{customdata[3]}<br>"
                "통근 %{customdata[4]}분<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattermap(
            lat=top["lat"],
            lon=top["lon"],
            mode="markers",
            marker={
                "size": 28,
                "color": "#f4e7b2",
                "opacity": 0.95,
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattermap(
            lat=top["lat"],
            lon=top["lon"],
            mode="markers+text",
            text=top["gu"],
            textposition="top center",
            marker={
                "size": 24,
                "color": top["total_score"],
                "colorscale": "Viridis",
                "cmin": score_min,
                "cmax": score_max,
            },
            customdata=list(zip(top["gu"], top["total_score"].round(1))),
            hovertemplate="<b>%{customdata[0]}</b><br>종합점수 %{customdata[1]}점<extra></extra>",
            showlegend=False,
        )
    )
    workplace = WORKPLACE_HUBS[workplace_name]
    for _, row in top.iterrows():
        fig.add_trace(
            go.Scattermap(
                lat=[workplace["lat"], row["lat"]],
                lon=[workplace["lon"], row["lon"]],
                mode="lines",
                line={"width": 2, "color": "rgba(255,77,79,0.45)"},
                name=f"직장1-{row['gu']}",
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scattermap(
            lat=[workplace["lat"]],
            lon=[workplace["lon"]],
            mode="markers+text",
            marker={"size": 18, "color": "#ff4d4f"},
            text=[f"직장1: {workplace['label']}"],
            textposition="top right",
            name="직장1",
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    if secondary_workplace_name:
        secondary = WORKPLACE_HUBS[secondary_workplace_name]
        for _, row in top.iterrows():
            fig.add_trace(
                go.Scattermap(
                    lat=[secondary["lat"], row["lat"]],
                    lon=[secondary["lon"], row["lon"]],
                    mode="lines",
                    line={"width": 2, "color": "rgba(77,171,247,0.45)"},
                    name=f"직장2-{row['gu']}",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scattermap(
                lat=[secondary["lat"]],
                lon=[secondary["lon"]],
                mode="markers+text",
                marker={"size": 18, "color": "#4dabf7"},
                text=[f"직장2: {secondary['label']}"],
                textposition="top right",
                name="직장2",
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )
    fig.update_layout(
        title="추천 자치구 지도",
        map={"style": "carto-positron", "center": {"lat": 37.5665, "lon": 126.9780}, "zoom": 10},
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
        showlegend=False,
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
        title="종합 점수 순위 (Top 15)",
        text=top["total_score"].round(1),
    )
    fig.update_layout(yaxis_title="자치구", xaxis_title="종합점수", coloraxis_colorbar_title="점수")
    return fig


def render_figure_grid(figures: dict[str, go.Figure]) -> None:
    names = list(figures.keys())
    for idx in range(0, len(names), 2):
        cols = st.columns(2)
        for offset in range(2):
            if idx + offset >= len(names):
                continue
            key = names[idx + offset]
            with cols[offset]:
                st.plotly_chart(figures[key], width="stretch", key=f"gallery_{key}_{idx+offset}")
