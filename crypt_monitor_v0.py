# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 10:43:18 2025

@author: my199
"""

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import datetime

# ==============================
# 利用可能な取引所（ccxtがサポートしているもの）
# ==============================
EXCHANGE_CHOICES = {
    "binance": ccxt.binance(),
    "bitflyer": ccxt.bitflyer(),
    "coinbase": ccxt.coinbase(),
}

# ==============================
# 価格データ取得関数
# ==============================
def fetch_price_history(exchange, symbol, limit=200):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1m", limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df[["datetime", "close"]]
    except Exception as e:
        st.warning(f"{exchange.id} でデータ取得失敗: {e}")
        return None

# ==============================
# 異常検知（単純版）
# ==============================
def detect_anomalies_simple(df_all):
    alerts = []
    if df_all is None or df_all.empty:
        return alerts
    latest = df_all.iloc[-1]
    mean = df_all.mean(axis=1).iloc[-1]
    std = df_all.std(axis=1).iloc[-1]
    for col in df_all.columns:
        if abs(latest[col] - mean) > 2 * std:  # 2σ乖離
            alerts.append(f"単純検知: {col} が異常値 (価格={latest[col]:.2f}, 平均={mean:.2f})")
    return alerts

# ==============================
# 異常検知（機械学習版: Isolation Forest）
# ==============================
def detect_anomalies_ml(df_all):
    alerts = []
    if df_all is None or df_all.empty:
        return alerts
    model = IsolationForest(contamination=0.05, random_state=42)
    preds = model.fit_predict(df_all.fillna(method="ffill"))
    if preds[-1] == -1:
        alerts.append("ML検知: 最新データに異常が検出されました！")
    return alerts

# ==============================
# Streamlit アプリ
# ==============================
st.title("暗号資産 売買監視システム（デモ）")

# 銘柄選択
symbol = st.selectbox("銘柄を選択してください", ["BTC/USDT", "ETH/USDT", "BTC/JPY"])

# 取引所選択
selected_exchanges_ui = st.multiselect(
    "取引所を複数選択", 
    options=list(EXCHANGE_CHOICES.keys()),
    default=["binance", "bitflyer"]
)

# 実行ボタン
if st.button("データ取得 & 可視化"):
    dfs = {}
    for ex_name in selected_exchanges_ui:
        df = fetch_price_history(EXCHANGE_CHOICES[ex_name], symbol)
        if df is not None:
            dfs[ex_name] = df.set_index("datetime")["close"]

    if dfs:
        df_all = pd.concat(dfs, axis=1)

        # グラフ表示
        st.subheader("価格推移")
        st.line_chart(df_all)

        # 異常検知
        st.subheader("異常検知結果")

        simple_alerts = detect_anomalies_simple(df_all)
        ml_alerts = detect_anomalies_ml(df_all)

        if simple_alerts:
            for a in simple_alerts:
                st.error(a)
        else:
            st.success("単純検知: 異常なし")

        if ml_alerts:
            for a in ml_alerts:
                st.warning(a)
        else:
            st.info("ML検知: 異常なし")

    else:
        st.warning("選択した取引所から有効なデータが取得できませんでした。")
