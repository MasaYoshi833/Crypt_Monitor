# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 10:43:18 2025

@author: my199
"""

# app.py
"""
Streamlit app: 直接取引所の公開APIを叩いて複数取引所の同一銘柄価格を比較し、
統計的異常（σ / 取引所間乖離）とIsolationForestによるML異常検知を行う。
-- Designed for Japanese domestic exchanges + Binance (global).
-- If an exchange API changes, update EXCHANGES dict (url / parser).
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import time
import math

st.set_page_config(layout="wide", page_title="Direct-API 暗号資産監視 (国内複数取引所)")

st.title("Direct-API 暗号資産売買審査システム（プロトタイプ）")
st.markdown(
    """
    直接各取引所の公開REST APIを叩いて価格を取得します。  
    - 国内取引所はシンボル表記が異なるため、**同じ建て（例: JPY）に揃えて選択してください**。  
    - 動かない取引所があれば `EXCHANGES` 辞書の URL / parser を編集してください。
    """
)

# ------------------------
# 取引所アダプタ定義（要修正箇所はコメントで示す）
# 各 entry: 'label' (表示), 'url' (GETするURL) または 'func' (カスタム取得関数)
# 'parser' は response.json() を受け取り float 価格 を返す関数
# ------------------------

EXCHANGES = {
    # Binance (global / USDT建て)
    "binance": {
        "label": "Binance (BTC/USDT 等)",
        "url_template": "https://api.binance.com/api/v3/ticker/price?symbol={symbol_no_slash}",
        "symbol_format": lambda s: s.replace("/", "").upper(),  # BTC/USDT -> BTCUSDT
        "parser": lambda j: float(j.get("price")) if isinstance(j, dict) and "price" in j else float(j),
        "note": "Binance の symbol は例: BTCUSDT"
    },
    # bitFlyer (JPY)
    "bitflyer": {
        "label": "bitFlyer (BTC/JPY 等)",
        "url_template": "https://api.bitflyer.com/v1/getticker?product_code={symbol_dot}",
        "symbol_format": lambda s: s.replace("/", "_").upper().replace("_", "_").replace("BTC_JPY", "BTC_JPY"),  # expects BTC_JPY etc.
        "parser": lambda j: float(j.get("ltp")) if isinstance(j, dict) and "ltp" in j else (float(j.get("last")) if isinstance(j, dict) and "last" in j else None),
        "note": "bitFlyer uses product_code like BTC_JPY. Example URL: ?product_code=BTC_JPY"
    },
    # Coincheck (JPY)
    "coincheck": {
        "label": "Coincheck (BTC/JPY 等)",
        "url_template": "https://coincheck.com/api/ticker",  # pair optional param
        "symbol_format": lambda s: s.split("/")[0].lower() + "_" + s.split("/")[1].lower(),  # BTC/JPY -> btc_jpy
        "parser": lambda j: float(j.get("last")) if isinstance(j, dict) and "last" in j else None,
        "note": "Coincheck: GET /api/ticker?pair=btc_jpy (pair optional)"
    },
    # bitbank (JPY) - public endpoint variants exist: public.bitbank.cc / api.bitbank.cc
    "bitbank": {
        "label": "bitbank (BTC/JPY 等)",
        # Two common endpoints; try the public.bitbank one first, fallback handled in code
        "url_template": "https://public.bitbank.cc/v1/ticker?pair={pair}",  # e.g. pair=btc_jpy
        "symbol_format": lambda s: s.split("/")[0].lower() + "_" + s.split("/")[1].lower(),  # btc_jpy
        "parser": lambda j: float(j.get("data", {}).get("last")) if isinstance(j, dict) and "data" in j else None,
        "note": "bitbank: public API returns JSON.data.last"
    },
    # GMO Coin (example; docs at api.coin.z.com) — exact public ticker endpoint may vary; placeholder here
    "gmo": {
        "label": "GMO Coin (public API - 要確認)",
        "url_template": "https://api.coin.z.com/public/v1/ticker?symbol={symbol_gmo}",
        "symbol_format": lambda s: s.replace("/", "_"),  # placeholder transform; confirm with GMO docs
        "parser": lambda j: float(j.get("data", [])[0].get("last")) if isinstance(j, dict) and "data" in j and len(j["data"])>0 else None,
        "note": "GMO Coin: API 仕様を公式で確認してください（例示）"
    },
    # Bitpoint, DMM, RakutenWallet, etc. -- placeholders. URL and parser need to be filled with official docs.
    "bitpoint": {
        "label": "Bitpoint (要設定)",
        "url_template": None,
        "symbol_format": lambda s: s,
        "parser": lambda j: None,
        "note": "Bitpoint の public ticker エンドポイントをここに記載してください"
    },
    "dmm": {
        "label": "DMM Bitcoin (要設定)",
        "url_template": None,
        "symbol_format": lambda s: s,
        "parser": lambda j: None,
        "note": "DMM Bitcoin の public ticker エンドポイントをここに記載してください"
    },
    "rakuten": {
        "label": "Rakuten Wallet (要設定)",
        "url_template": None,
        "symbol_format": lambda s: s,
        "parser": lambda j: None,
        "note": "Rakuten Wallet の public ticker エンドポイントをここに記載してください"
    }
}

# ------------------------
# ヘルパー: 単一取引所から価格を取得
# ------------------------
def fetch_price_from_exchange(entry, symbol):
    """
    entry: EXCHANGES[key]
    symbol: user input like 'BTC/JPY'
    returns: float price or None
    """
    url_template = entry.get("url_template")
    fmt = entry.get("symbol_format", lambda s: s)
    parser = entry.get("parser", lambda j: None)

    # if url_template missing -> can't fetch
    if not url_template:
        return None, f"No URL configured for {entry.get('label')}"

    # build symbol
    try:
        symbol_formatted = fmt(symbol)
        # some templates expect no slash, some expect underscore etc.
        url = url_template.format(
            symbol_no_slash=symbol.replace("/", "").upper(),
            symbol_dot=symbol.replace("/", ".").upper(),
            pair=symbol_formatted,
            symbol_gmo=symbol.replace("/", "").upper()  # placeholder
        )
    except Exception as e:
        return None, f"Symbol formatting error: {e}"

    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        j = resp.json()
        price = parser(j)
        if price is None:
            return None, f"Parser returned None for response: {j}"
        return float(price), None
    except requests.exceptions.RequestException as e:
        return None, f"RequestError: {e}"
    except Exception as e:
        return None, f"ParseError: {e}"

# ------------------------
# UI: 銘柄・取引所選択・パラメータ
# ------------------------
col1, col2 = st.columns([3,1])
with col1:
    st.subheader("設定")
    symbol = st.text_input("銘柄（例: BTC/JPY, BTC/USDT, ETH/JPY）", value="BTC/JPY")
    # Show available exchanges (labels)
    exch_keys = list(EXCHANGES.keys())
    exch_labels = [f"{k} — {EXCHANGES[k]['label']}" for k in exch_keys]
    selected = st.multiselect("参照する取引所（複数選択）", options=exch_keys,
                              format_func=lambda k: f"{k} — {EXCHANGES[k]['label']}",
                              default=["bitflyer", "bitbank", "coincheck", "binance"])
with col2:
    st.subheader("パラメータ")
    samples = st.number_input("サンプル数（ティッカーループ回数）", min_value=1, max_value=300, value=30)
    interval = st.number_input("サンプリング間隔（秒）", min_value=0.5, max_value=60.0, value=1.0, step=0.5)
    z_threshold = st.number_input("Z閾値（σ）", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
    pct_spread_threshold = st.number_input("取引所間スプレッド閾値（%）", min_value=0.1, max_value=50.0, value=2.0, step=0.1)
    ml_enable = st.checkbox("IsolationForest による ML 異常検知", value=True)
    ml_contamination = st.number_input("IsolationForest contamination", min_value=0.001, max_value=0.2, value=0.02, step=0.001, format="%.3f")

run = st.button("取得実行（直接API）")

# ------------------------
# 実行
# ------------------------
if run:
    if not selected:
        st.error("取引所を1つ以上選択してください。")
        st.stop()

    st.info(f"取得対象: {symbol} / 取引所: {', '.join(selected)} / サンプル数: {samples} / 間隔: {interval}s")

    # collect time series
    series_dict = {k: [] for k in selected}
    timestamps = []

    # loop to create time series (simple approach)
    for i in range(samples):
        ts = pd.Timestamp.now()
        timestamps.append(ts)
        for k in selected:
            price, err = fetch_price_from_exchange(EXCHANGES[k], symbol)
            if err:
                # keep NaN to preserve index alignment; log error in sidebar
                series_dict[k].append(np.nan)
            else:
                series_dict[k].append(price)
        # show progress
        st.experimental_rerun() if False else None  # no-op placeholder to avoid streamlit warnings
        time.sleep(interval)

    # build DataFrame
    df = pd.DataFrame(index=timestamps)
    for k in selected:
        df[k] = pd.Series(series_dict[k], index=timestamps)

    # forward/backward fill to mitigate sparse failures
    df = df.sort_index().ffill().bfill()

    st.subheader("取得結果（下位50行）")
    st.dataframe(df.tail(50))

    # simple statistical checks
    median_series = df.median(axis=1)
    std_series = df.std(axis=1).replace(0, 1e-9)
    zscores = (df.sub(median_series, axis=0)).div(std_series, axis=0).abs()
    simple_flags = zscores > z_threshold

    max_per_ts = df.max(axis=1)
    min_per_ts = df.min(axis=1)
    pct_spread = (max_per_ts - min_per_ts) / min_per_ts * 100.0
    spread_flags = pct_spread > pct_spread_threshold

    last_time_alert = (simple_flags.iloc[-1].any()) or spread_flags.iloc[-1]

    if last_time_alert:
        st.error("⚠️ シンプルアラート（最新時点）：統計基準で異常検知")
    else:
        st.success("最新時点: シンプル検知は正常")

    # ML detection
    ml_flags = pd.Series(False, index=df.index)
    if ml_enable:
        st.subheader("ML 異常検知（IsolationForest）")
        # features: percent returns + spread + std
        df_returns = df.pct_change().fillna(0)
        features = df_returns.copy()
        features['spread_pct'] = pct_spread.fillna(0)
        features['std'] = df.std(axis=1).fillna(0)
        X = features.values
        try:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            clf = IsolationForest(contamination=float(ml_contamination), random_state=42)
            clf.fit(Xs)
            preds = clf.predict(Xs)  # -1 anomalous, 1 normal
            ml_flags = pd.Series(preds == -1, index=features.index)
            if ml_flags.iloc[-1]:
                st.error("🔴 MLによる異常検知: 最新時点で異常と判定")
            else:
                st.success("🟢 ML: 最新時点は正常")
        except Exception as e:
            st.warning(f"ML処理でエラー: {e}")

    # Visualization
    st.subheader("可視化（Plotly）")
    fig = go.Figure()
    colors = ["blue", "green", "purple", "orange", "brown", "cyan", "magenta", "grey"]
    for i, col in enumerate(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col, line=dict(width=1.5)))
    # markers for latest point colored by severity
    marker_x = []
    marker_y = []
    marker_color = []
    marker_text = []
    for col in df.columns:
        val = df[col].iloc[-1]
        marker_x.append(df.index[-1])
        marker_y.append(val)
        is_simple = simple_flags[col].iloc[-1] if col in simple_flags else False
        is_ml = ml_flags.iloc[-1] if ml_enable else False
        if is_ml:
            marker_color.append("red")
            marker_text.append(f"{col}: ML異常")
        elif is_simple:
            marker_color.append("orange")
            marker_text.append(f"{col}: 統計異常")
        else:
            marker_color.append("green")
            marker_text.append(f"{col}: 正常")
    fig.add_trace(go.Scatter(x=marker_x, y=marker_y, mode="markers", marker=dict(size=12, color=marker_color), showlegend=False, hovertext=marker_text))
    # spread trace (secondary axis)
    fig.add_trace(go.Scatter(x=df.index, y=pct_spread, mode="lines", name="Pct Spread (%)", yaxis="y2", line=dict(width=1, dash="dash")))
    fig.update_layout(xaxis=dict(title="Time"), yaxis=dict(title=f"Price ({symbol})"), yaxis2=dict(title="Spread (%)", overlaying="y", side="right"), height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Log & Download
    st.subheader("診断ログ")
    log = {
        "timestamp": str(df.index[-1]),
        "simple_alert": bool(last_time_alert),
        "pct_spread": float(pct_spread.iloc[-1]),
        "ml_alert": bool(ml_flags.iloc[-1]) if ml_enable else False
    }
    st.json(log)

    csv = df.to_csv().encode("utf-8")
    st.download_button("価格データ(CSV)ダウンロード", csv, file_name=f"prices_{symbol.replace('/','')}.csv", mime="text/csv")

    # Show notes / missing endpoints
    st.subheader("注意 / 取引所設定")
    for k in selected:
        note = EXCHANGES[k].get("note", "")
        url_temp = EXCHANGES[k].get("url_template")
        if not url_temp:
            st.warning(f"{k}: APIエンドポイントが未設定です。公式ドキュメントで public ticker の URL を確認してください.")
        else:
            st.write(f"{k}: {EXCHANGES[k]['label']}  (Template: {url_temp}) — {note}")

    st.info("本番運用前に: APIレート制限 / シンボルマッピング / TLS 証明書・IP 制限等の確認を行ってください。")
