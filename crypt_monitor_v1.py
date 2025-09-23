# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 13:10:44 2025

@author: my199
"""

# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Crypto Monitor (bitFlyer / GMO / Coincheck)", layout="wide")

# --------------------
# 設定パラメータ（UIで変更可）
# --------------------
st.title("暗号資産取引所モニター（BTC/JPY 等）")

col_ctrl, col_info = st.columns([3, 1])
with col_ctrl:
    symbol = st.selectbox("銘柄を選択してください", ["BTC/JPY", "ETH/JPY", "XRP/JPY"])
    update_interval = st.number_input("更新間隔（秒）", min_value=1, max_value=60, value=5, step=1)
    iterations = st.number_input("更新回数（ループ回数、0 = 無限）", min_value=0, value=0, step=1)
    z_threshold = st.number_input("Z閾値（σ）: 単純判定", min_value=0.5, max_value=10.0, value=3.0, step=0.1)
    pct_spread_threshold = st.number_input("取引所間スプレッド閾値（%）", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
    ml_enable = st.checkbox("IsolationForest による ML 異常検知を有効化", value=True)
    ml_contamination = st.number_input("ML contamination", min_value=0.001, max_value=0.2, value=0.02, step=0.001, format="%.3f")
    zero_base_checkbox = st.checkbox("縮小時に Y 軸を 0 ベースにする（チェックすると常に 0 下限）", value=False)
with col_info:
    st.markdown("**注意**")
    st.markdown("- 長時間の連続実行は Streamlit の実行時間制限や API レート制限に注意してください。")
    st.markdown("- 必要なら `iterations` を 0 にして手動停止（ページリロード）で止めて下さい。")

# --------------------
# 取引所 API 定義
# （シンプルに public ticker を叩いて last を抜く実装）
# --------------------
EXCHANGES = {
    "bitFlyer": {
        "get_url": lambda sym: f"https://api.bitflyer.com/v1/ticker?product_code={sym.replace('/','_')}",
        "parser": lambda j: j.get("ltp") if isinstance(j, dict) else None
    },
    "GMO Coin": {
        "get_url": lambda sym: f"https://api.coin.z.com/public/v1/ticker?symbol={sym.replace('/','_')}",
        "parser": lambda j: float(j.get("data")[0]["last"]) if isinstance(j, dict) and "data" in j else None
    },
    "Coincheck": {
        # Coincheck の /api/ticker は BTC/JPY 用で pair パラメータ未対応のケースがあるため、
        # BTC/JPY の場合は /api/ticker を使い、他ペアは ?pair=xxx を試す（失敗時 None）
        "get_url": lambda sym: (
            "https://coincheck.com/api/ticker" if sym.upper() == "BTC/JPY" else f"https://coincheck.com/api/ticker?pair={sym.replace('/','').lower()}"
        ),
        "parser": lambda j: float(j.get("last")) if isinstance(j, dict) and "last" in j else None
    }
}

# --------------------
# 価格取得関数（耐障害）
# --------------------
def fetch_price(exchange_name, symbol):
    cfg = EXCHANGES[exchange_name]
    url = cfg["get_url"](symbol)
    try:
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        j = r.json()
        price = cfg["parser"](j)
        if price is None:
            return None
        return float(price)
    except Exception as e:
        # ネットワークやパースエラーは None を返す（ログは呼び出し側で出す）
        return None

# --------------------
# セッションステート初期化（履歴保存）
# --------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {"time": pd.Timestamp, "bitFlyer":..., ...}

# UI プレースホルダ（更新時にその部分のみ上書き）
chart_placeholder = st.empty()
table_placeholder = st.empty()
alert_placeholder = st.empty()
download_placeholder = st.empty()

# スタートボタン
start = st.button("開始")
stop = st.button("停止（即時停止）")

# 一時停止フラグを session_state で管理
if "running" not in st.session_state:
    st.session_state.running = False
if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

# --------------------
# 実行ループ（Start が押されたらここで動く）
# --------------------
if st.session_state.running:
    st.info("モニタリング開始... (コントロールパネルで停止可能)")
    loop_count = 0
    # iterations==0 => 無限ループ（ユーザーが Stop またはページリロードで止める）
    while st.session_state.running and (iterations == 0 or loop_count < iterations):
        loop_count += 1
        # 取得時刻（JST）
        ts = pd.Timestamp.now(tz="Asia/Tokyo")
        row = {"time": ts}
        # 価格取得
        for ex in EXCHANGES.keys():
            price = fetch_price(ex, symbol)
            row[ex] = price

        # ログ追加（セッションに保存）
        st.session_state.history.append(row)

        # DataFrame 作成
        df = pd.DataFrame(st.session_state.history)
        # index を JST 時刻にしておく
        df = df.set_index("time")
        # 整理: float に変換（失敗は NaN）
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 単純異常検知（最新時点）
        latest = df.iloc[-1]
        valid_prices = latest.dropna().values
        alerts = []
        simple_flags = {col: False for col in df.columns}
        spread_flag = False
        if len(valid_prices) > 0:
            avg_price = np.nanmean(valid_prices)
            # 取引所間スプレッド(%)（最新）
            maxp = np.nanmax(valid_prices)
            minp = np.nanmin(valid_prices)
            if (minp is not None) and (minp > 0):
                pct_spread = (maxp - minp) / minp * 100.0
            else:
                pct_spread = 0.0
            if pct_spread >= pct_spread_threshold:
                spread_flag = True
                alerts.append(f"⚠️ 取引所間スプレッドが {pct_spread:.2f}% です（閾値: {pct_spread_threshold}％）")

            # Zスコア（最新行）: 過去履歴の時系列ごとの std を使うより、ここは直近値の分散を利用
            # ここでは列ごとの過去履歴標準偏差ではなく、時点ごとの中央値と std を利用（横比較）
            median_at_latest = np.nanmedian(valid_prices)
            std_at_latest = np.nanstd(valid_prices) if np.nanstd(valid_prices) > 0 else 1e-9
            for col in df.columns:
                val = latest[col]
                if pd.isna(val):
                    continue
                z = abs((val - median_at_latest) / std_at_latest)
                if z > z_threshold:
                    simple_flags[col] = True
                    alerts.append(f"⚠️ {col} が中央値から {z:.2f}σ 乖離しています (price={val:.0f})")

        # ML 異常検知（行レベル） - オプション
        ml_flag_row = False
        if ml_enable:
            try:
                # 最低学習点数チェック
                if len(df.dropna(how='all')) >= 10:
                    # 特徴量: 各取引所のパーセントリターン（時系列） + スプレッド + std
                    features = df.pct_change().fillna(0)
                    spread_series = (df.max(axis=1) - df.min(axis=1)).fillna(0)
                    std_series = df.std(axis=1).fillna(0)
                    features["spread_pct"] = spread_series
                    features["std"] = std_series
                    X = features.values
                    scaler = StandardScaler()
                    Xs = scaler.fit_transform(X)
                    clf = IsolationForest(contamination=float(ml_contamination), random_state=42)
                    preds = clf.fit_predict(Xs)  # -1 anomaly, 1 normal
                    if preds[-1] == -1:
                        ml_flag_row = True
                        alerts.append("🔴 ML (IsolationForest) が最新行を異常と判定しました")
                else:
                    # 学習データ不足（do nothing）
                    pass
            except Exception as e:
                # MLエラーは警告表示に留める
                alerts.append(f"⚠️ ML処理でエラー: {e}")

        # --------------------
        # 可視化（Plotly） — グラフだけを更新
        # --------------------
        fig = go.Figure()
        # 各取引所のライン描画
        for col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=col,
                hovertemplate="%{y:.0f} 円<br>%{x|%Y-%m-%d %H:%M:%S}<extra></extra>",
                line=dict(width=2)
            ))

        # 最新点マーカー（色は ML > simple > normal の優先で決定）
        marker_x = []
        marker_y = []
        marker_color = []
        marker_text = []
        for col in df.columns:
            last_val = df[col].iloc[-1]
            marker_x.append(df.index[-1])
            marker_y.append(last_val)
            if pd.isna(last_val):
                marker_color.append("gray")
                marker_text.append(f"{col}: no data")
                continue
            if ml_flag_row:
                marker_color.append("red")
                marker_text.append(f"{col}: ML異常 (行レベル)")
            elif simple_flags.get(col, False) or spread_flag:
                marker_color.append("orange")
                # priority: individual zflag or spread
                reason = []
                if simple_flags.get(col, False):
                    reason.append("σ乖離")
                if spread_flag:
                    reason.append("取引所間スプレッド")
                marker_text.append(f"{col}: {' & '.join(reason)}")
            else:
                marker_color.append("green")
                marker_text.append(f"{col}: 正常")

        fig.add_trace(go.Scatter(
            x=marker_x,
            y=marker_y,
            mode="markers",
            marker=dict(size=12, color=marker_color),
            showlegend=False,
            hovertext=marker_text
        ))

        # レイアウト
        yaxis_dict = dict(title="価格（円）")
        # 初期は通常スケール（差が見やすい）。チェックで常に 0 ベースに。
        if zero_base_checkbox:
            yaxis_dict["rangemode"] = "tozero"
        else:
            yaxis_dict["rangemode"] = "normal"

        fig.update_layout(
            title=f"{symbol} — 取引所別価格（JST） — 最終更新: {df.index[-1].strftime('%Y-%m-%d %H:%M:%S')}",
            xaxis_title="時間（JST）",
            yaxis=yaxis_dict,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=40, r=20, t=80, b=40),
            height=520
        )

        # x 軸の時刻表示を見やすく（時間:分:秒）
        fig.update_xaxes(tickformat="%H:%M:%S")

        # 描画（chart_placeholder を上書き）
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        # テーブル（最新行）を更新
        table_placeholder.dataframe(df.tail(1).T.rename(columns={df.index[-1]: "最新値"}))

        # アラート表示（上側にまとめて表示）
        if alerts:
            alert_placeholder.error("\n".join(alerts))
        else:
            alert_placeholder.success("現在異常は検知されていません")

        # CSV ダウンロード（最新履歴全て）
        csv = df.to_csv().encode("utf-8")
        download_placeholder.download_button("CSVダウンロード（履歴）", csv, file_name=f"prices_{symbol.replace('/','')}.csv", mime="text/csv")

        # 次ループまで待機（この間、ページはブロックされます）
        # 停止ボタンを押すと session_state.running が False になりますが、
        # ブロッキング中に押しても即時反映されない点に注意（Streamlitの動作仕様）。
        # 強制停止はページリロードや別タブで Stop ボタン押下を推奨します。
        for _ in range(int(update_interval * 10)):  # 0.1s単位でチェックして早く停止できるようにする
            time.sleep(0.1)
            # ユーザーが Stop を押した場合、the page will re-run and set running False,
            # but because we are blocked in this loop, we cannot detect that immediately in all environments.
            # To reduce latency we break early if running flag got cleared by a rerun (rare).
            if not st.session_state.get("running", False):
                break

    # ループを抜けたとき
    st.session_state.running = False
    st.success("モニタリングを停止しました。")

else:
    st.info("「開始」ボタンを押すと5秒ごとに価格を更新して可視化します。停止は「停止」ボタンかページリロードで行ってください。")
    # 小さく直近の履歴を表示しておく
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history).set_index("time")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        st.dataframe(df.tail(5).T)


