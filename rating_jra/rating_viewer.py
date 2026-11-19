from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from viewer_data import EXPECTED_P7_HASH, SCALE, load_viewer_config, load_viewer_data

st.set_page_config(page_title="P7 JRAレーティング ビューワー", layout="wide")


GRADE_LABELS = {
    "A": "GI",
    "B": "GII",
    "C": "GIII",
    "D": "その他重賞",
    "E": "特別",
    "F": "J-GI",
    "G": "J-GII",
    "H": "J-GIII",
}
AGE_LABELS = {11: "2歳", 12: "3歳", 13: "3歳以上", 14: "4歳以上", 18: "障害3歳以上", 19: "障害4歳以上"}


def class_condition_label(row: pd.Series) -> str:
    age_value = row.get("race_age_code")
    age = AGE_LABELS.get(int(age_value), "") if pd.notna(age_value) else ""
    grade_value = row.get("grade_code")
    grade = str(grade_value).strip().upper() if pd.notna(grade_value) else ""
    if grade in GRADE_LABELS:
        return age + GRADE_LABELS[grade]
    code_value = row.get("condition_code")
    code = int(code_value) if pd.notna(code_value) else 0
    if code == 701:
        condition = "新馬"
    elif code == 702:
        condition = "未出走"
    elif code == 703:
        condition = "未勝利"
    elif code == 999:
        condition = "オープン"
    elif code == 5:
        condition = "1勝クラス"
    elif code == 10:
        condition = "2勝クラス"
    elif code == 16:
        condition = "3勝クラス"
    elif 1 <= code <= 700:
        condition = f"{code * 100}万円条件"
    else:
        condition = str(row.get("condition_name", "")).strip()
    return age + condition


def race_label(row: pd.Series) -> str:
    venue = row.get("venue_name", "")
    number = row.get("race_number", "")
    condition = class_condition_label(row)
    name_value = row.get("race_name", "")
    name = str(name_value).strip() if pd.notna(name_value) else ""
    if name == "名称未設定":
        name = ""
    label = f"{row['race_id']}｜{venue} {number}R"
    if condition:
        label += f"｜{condition}"
    if name:
        label += f" {name}"
    return label


def race_view(data: pd.DataFrame) -> None:
    st.header("レース別ビュー")
    dates = sorted(data["race_date"].dt.normalize().unique(), reverse=True)
    selected_date = st.selectbox("開催日", dates, format_func=lambda value: pd.Timestamp(value).strftime("%Y-%m-%d"))
    venues = ["すべて"] + sorted(data.loc[data["race_date"].dt.normalize().eq(selected_date), "venue_name"].dropna().unique().tolist())
    venue = st.selectbox("競馬場", venues)
    subset = data[data["race_date"].dt.normalize().eq(selected_date)]
    if venue != "すべて":
        subset = subset[subset["venue_name"].eq(venue)]
    race_ids = sorted(subset["race_id"].unique().tolist(), reverse=True)
    if not race_ids:
        st.info("該当レースがありません")
        return
    race_id = st.selectbox("レース", race_ids, format_func=lambda value: race_label(subset[subset["race_id"].eq(value)].iloc[0]))
    available = subset[subset["race_id"].eq(race_id)]
    modes = ["final", "predict"]
    labels = {"final": "確定", "predict": "発走前（未確定）"}
    available_modes = [mode for mode in modes if mode in set(available["record_type"].unique())]
    selected_mode = st.selectbox("表示区分", available_modes, format_func=lambda mode: labels[mode])
    race = available[available["record_type"].eq(selected_mode)].copy()
    race["rating_rank"] = race["elo_pre"].rank(method="min", ascending=False).astype(int)
    race = race.sort_values(["elo_pre", "horse_id"], ascending=[False, True])
    columns = [
        "rating_rank", "horse_name", "surface", "elo_pre", "elo_base_pre", "elo_surface_adjustment",
        "elo_sd", "elo_prior_boost", "official_finish", "elo_post", "elo_delta",
        "mu_pre", "var_pre_eff", "n_graph_starts", "days_since_start",
    ]
    display = race[columns].rename(columns={
        "rating_rank": "順位", "horse_name": "馬名", "surface": "馬場", "elo_pre": "発走前Elo",
        "elo_base_pre": "基礎体力Elo", "elo_surface_adjustment": "馬場適性補正Elo", "elo_sd": "±SD",
        "elo_prior_boost": "P0比", "official_finish": "確定着順", "elo_post": "更新後Elo",
        "elo_delta": "変動幅", "mu_pre": "生mu_pre", "var_pre_eff": "有効分散",
        "n_graph_starts": "累積出走数", "days_since_start": "休養日数",
    })
    st.dataframe(display, hide_index=True, width="stretch")
    chart = race.sort_values("elo_pre", ascending=True)
    fig = go.Figure(go.Bar(
        x=chart["elo_pre"],
        y=chart["horse_name"],
        orientation="h",
        error_x=dict(type="data", symmetric=False, array=chart["elo_sd"], arrayminus=chart["elo_sd"]),
        marker=dict(color=chart["elo_prior_boost"], colorscale="Viridis", colorbar=dict(title="P0比")),
        customdata=np.stack([
            chart["elo_prior_boost"], chart["official_finish"], chart["elo_post"], chart["elo_delta"],
            chart["elo_base_pre"], chart["elo_surface_adjustment"],
        ], axis=-1),
        hovertemplate="%{y}<br>発走前Elo: %{x:.3f}<br>基礎体力Elo: %{customdata[4]:.3f}<br>馬場適性補正: %{customdata[5]:.3f}<br>P0比: %{customdata[0]:.3f}<br>着順: %{customdata[1]}<br>更新後Elo: %{customdata[2]:.3f}<br>変動幅: %{customdata[3]:.3f}<extra></extra>",
    ))
    fig.update_layout(height=max(400, 28 * len(chart) + 180), yaxis_title="馬名", xaxis_title="発走前 Elo風レーティング ±1 SD")
    st.plotly_chart(fig, width="stretch")


def career_view(data: pd.DataFrame) -> None:
    st.header("馬別推移ビュー")
    search = st.text_input("馬名（部分一致検索）")
    horse_id_text = st.text_input("馬ID（指定時はこちらを優先）")
    horses = data[["horse_id", "horse_name"]].drop_duplicates().copy()
    if search:
        horses = horses[horses["horse_name"].fillna("").str.contains(search, case=False, regex=False)]
    if horse_id_text.strip():
        horses = horses[horses["horse_id"].astype(str).eq(horse_id_text.strip())]
    if horses.empty:
        st.info("該当する馬がいません")
        return
    horses = horses.sort_values(["horse_name", "horse_id"])
    labels = [f"{row.horse_name} ({int(row.horse_id)})" for row in horses.itertuples()]
    selected = st.selectbox("馬を選択", labels)
    selected_row = horses.iloc[labels.index(selected)]
    horse_id = int(selected_row["horse_id"])
    include_predict = st.checkbox("未確定の発走前レースも含める", value=False)
    career = data[data["horse_id"].eq(horse_id)].copy()
    if not include_predict:
        career = career[career["record_type"].eq("final")]
    career = career.sort_values("race_date")
    if career.empty:
        st.info("出走レースがありません")
        return
    custom = np.stack([
        career["race_name"], career["official_finish"], career["days_since_start"],
        career["elo_delta"], career["n_graph_starts"],
    ], axis=-1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=career["race_date"], y=career["elo_pre"], mode="lines+markers", name="発走前Elo",
        error_y=dict(type="data", symmetric=False, array=career["elo_sd"], arrayminus=career["elo_sd"]),
        customdata=custom,
        hovertemplate="%{x|%Y-%m-%d}<br>%{customdata[0]}<br>着順: %{customdata[1]}<br>休養: %{customdata[2]}日<br>ΔElo: %{customdata[3]:.3f}<br>通算出走: %{customdata[4]}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=career["race_date"], y=career["elo_post"], mode="lines+markers", name="更新後Elo",
        line=dict(dash="dash"),
        customdata=custom,
        hovertemplate="%{x|%Y-%m-%d}<br>%{customdata[0]}<br>着順: %{customdata[1]}<br>休養: %{customdata[2]}日<br>ΔElo: %{customdata[3]:.3f}<br>通算出走: %{customdata[4]}<extra></extra>",
    ))
    fig.update_layout(height=520, xaxis_title="開催日", yaxis_title="Elo風レーティング ±1 SD", hovermode="closest")
    st.plotly_chart(fig, width="stretch")
    columns = [
        "race_date", "venue_name", "race_name", "surface", "official_finish", "elo_pre", "elo_base_pre",
        "elo_surface_adjustment", "elo_post", "elo_delta", "elo_sd", "elo_prior_boost",
        "n_graph_starts", "days_since_start", "condition_name", "distance",
    ]
    display = career[columns].rename(columns={
        "race_date": "開催日", "venue_name": "競馬場", "race_name": "レース名", "surface": "馬場",
        "official_finish": "着順", "elo_pre": "発走前Elo", "elo_base_pre": "基礎体力Elo",
        "elo_surface_adjustment": "馬場適性補正Elo", "elo_post": "更新後Elo", "elo_delta": "変動幅",
        "elo_sd": "±SD", "elo_prior_boost": "P0比", "n_graph_starts": "累積出走数",
        "days_since_start": "休養日数", "condition_name": "条件", "distance": "距離",
    })
    st.dataframe(display, hide_index=True, width="stretch")


def ranking_view(data: pd.DataFrame) -> None:
    st.header("ランキングビュー（現役トップ100）")
    confirmed = data[data["record_type"].eq("final")].copy()
    dates = sorted(confirmed["race_date"].dt.normalize().unique(), reverse=True)
    asof = st.selectbox("基準日", dates, format_func=lambda value: pd.Timestamp(value).strftime("%Y-%m-%d"))
    surface = st.selectbox("コース", ["すべて", "芝", "ダート"])
    min_starts = st.number_input("最小出走数", min_value=1, max_value=100, value=1, step=1)
    recent = confirmed[confirmed["race_date"].dt.normalize().between(asof - pd.Timedelta(days=365), asof)]
    latest = recent.sort_values(["horse_id", "race_date", "record_type"], kind="stable").groupby("horse_id", as_index=False).tail(1).copy()
    latest["elo_now"] = latest["elo_post"].fillna(latest["elo_pre"])
    latest["current_starts"] = latest["n_graph_starts"] + 1
    if surface != "すべて":
        latest = latest[latest["surface_code"].eq(0 if surface == "芝" else 1)]
    latest = latest[latest["current_starts"].ge(min_starts)]
    top = latest.sort_values(["elo_now", "horse_id"], ascending=[False, True]).head(100).copy()
    top["rank"] = np.arange(1, len(top) + 1)
    columns = ["rank", "horse_name", "elo_now", "current_starts", "race_date", "venue_name", "race_name"]
    display = top[columns].rename(columns={
        "rank": "順位", "horse_name": "馬名", "elo_now": "Elo", "current_starts": "出走数",
        "race_date": "最終出走日", "venue_name": "最終競馬場", "race_name": "最終レース",
    })
    st.dataframe(display, hide_index=True, width="stretch")
    chart = top.head(20).sort_values("elo_now", ascending=True)
    fig = go.Figure(go.Bar(x=chart["elo_now"], y=chart["horse_name"], orientation="h"))
    fig.update_layout(height=600, xaxis_title="最新 Elo風レーティング", yaxis_title="馬名")
    st.plotly_chart(fig, width="stretch")


def pedigree_view(auxiliary: pd.DataFrame) -> None:
    st.header("血統評価ビュー")
    if auxiliary.empty:
        st.info("血統補助集計がありません")
        return
    side = st.selectbox("血統区分", ["sire", "bms"], format_func=lambda value: "種牡馬" if value == "sire" else "母父")
    frame = auxiliary[auxiliary["side"].eq(side)].copy()
    search = st.text_input("種牡馬・母父名（部分一致）", key="pedigree_search")
    if search:
        frame = frame[frame["pedigree_name"].fillna("").str.contains(search, case=False, regex=False)]
    minimum = st.number_input("最小産駒数", min_value=1, max_value=1000, value=1, step=1, key="min_offspring")
    frame = frame[frame["offspring"].ge(minimum)]
    metrics = {
        "新馬戦勝率": "new_horse_win_rate",
        "新馬戦3着内率": "new_horse_top3_rate",
        "3走以内勝ち上がり率": "offspring_win_within3_rate",
        "勝ち残差平均": "residual_win_mean",
        "3着内残差平均": "residual_top3_mean",
        "産駒数": "offspring",
    }
    label = st.selectbox("並べ替え指標", list(metrics))
    metric = metrics[label]
    frame = frame.sort_values([metric, "offspring"], ascending=[False, False]).head(100).copy()
    frame["pedigree_name"] = frame["pedigree_name"].fillna(frame[f"{side}_id"].astype(str))
    columns = [
        "pedigree_name", "offspring", "starts", "new_horse_win_rate", "new_horse_top3_rate",
        "offspring_win_within3_rate", "residual_win_mean", "residual_top3_mean", "residual_starts",
    ]
    display = frame[columns].rename(columns={
        "pedigree_name": "種牡馬・母父", "offspring": "産駒数", "starts": "集計走数",
        "new_horse_win_rate": "新馬戦勝率", "new_horse_top3_rate": "新馬戦3着内率",
        "offspring_win_within3_rate": "3走以内勝ち上がり率", "residual_win_mean": "勝ち残差平均",
        "residual_top3_mean": "3着内残差平均", "residual_starts": "残差出走数",
    })
    st.dataframe(display, hide_index=True, width="stretch")
    chart = frame.head(20).sort_values(metric, ascending=True)
    fig = go.Figure(go.Bar(x=chart[metric], y=chart["pedigree_name"], orientation="h"))
    fig.update_layout(height=600, xaxis_title=label, yaxis_title="種牡馬・母父")
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    st.title("P7 JRAレーティング ビューワー")
    config = load_viewer_config()
    with st.spinner("2017年以降のデータを読み込み・キャッシュしています…"):
        data, auxiliary, fingerprint = load_viewer_data(config)
    st.sidebar.markdown("**P7 config hash**")
    st.sidebar.code(EXPECTED_P7_HASH)
    st.sidebar.markdown(f"**件数**: {len(data):,} rows")
    st.sidebar.markdown(f"**最新開催日**: {data['race_date'].max().date()}")
    st.sidebar.markdown("**2017-01-01以降のみ表示**")
    tabs = st.tabs(["レース別", "馬別推移", "ランキング", "血統評価"])
    with tabs[0]:
        race_view(data)
    with tabs[1]:
        career_view(data)
    with tabs[2]:
        ranking_view(data)
    with tabs[3]:
        pedigree_view(auxiliary)


if __name__ == "__main__":
    main()
