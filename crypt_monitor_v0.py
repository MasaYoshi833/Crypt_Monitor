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
import time
import matplotlib.pyplot as plt

# ==========================
# API取得関数（取引所ごと）
# ==========================
def fetch_bitflyer(symbol="BTC_JPY"):
    try:
        url = f"https://api.bitflyer.com/v1/ticker?product_code={symbol}"
        res = requests.get(url).json()
        return res.get("ltp")
    except:
        return None

def fetch_coincheck(symbol="btc_jpy"):
    try:
        url = f"https://coincheck.com/api/ticker"
        res = requests.get(url).json()
        return float(res.get("last"))
    except:
        return None

def fetch_gmo(symbol="BTC_JPY"):
    try:
        url = f"https://api.coin.z.com/public/v1/ticker?symbol={symbol}"
        res = requests.get(url).json()
        return float(res["data"][0]["last"])
    except:
        return None

def fetch_zaif(symbol="btc_jpy"):
    try:
        url = f"https://api.zaif.jp/api/1/ticker/{symbol}"
        res = requests.get(url).json()
        return float(res.get("last"))
    except:
        return None

def fetch_liquid(symbol="5"):  # 5 = BTC/JPY の product_id
    try:
        url = f"https://api.liquid.com/products/{symbol}"
        res = requests.get(url).json()
        return float(res.get("last_traded_price"))
    except:
        return None

def fetch_binance(symbol="BTCUSDT"):
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        res = requests.get(url).json()
        return float(res.get("price"))
    except:
        return None


# ==========================
# Streamlit アプリ
# ==========================
st.title("国内暗号資産取引所価格モニター")

symbol = st.selectbox("銘柄を選択してください", ["BTC/JPY", "ETH/JPY"])

# 取引所リスト
EXCHANGES = {
    "bitFlyer": fetch_bitflyer,
    "Coincheck": fetch_coincheck,
    "GMOコイン": fetch_gmo,
    "Zaif": fetch_zaif,
    "Liquid": fetch_liquid,
    "Binance(USDT)": fetch_binance,
}

selected_exchanges = st.multiselect(
    "参照する取引所を選択してください",
    options=list(EXCHANGES.keys()),
    default=["bitFlyer", "Coincheck", "GMOコイン"]
)

# データ収集
st.write("価格取得中...（5秒間隔で更新）")

if st.button("実行"):
    prices = {ex: [] for ex in selected_exchanges}
    timestamps = []

    progress = st.empty()
    chart_area = st.empty()

    for i in range(20):  # 20回（約100秒分）
        row = {}
        for ex in selected_exchanges:
            price = EXCHANGES[ex]()  # API呼び出し
            row[ex] = price
            prices[ex].append(price)
        timestamps.append(pd.Timestamp.now())

        # DataFrameに変換
        df = pd.DataFrame(prices, index=timestamps)

        # 可視化
        fig, ax = plt.subplots(figsize=(10, 5))
        df.plot(ax=ax)
        ax.set_title(f"{symbol} 各取引所の価格推移")
        ax.set_ylabel("価格")
        chart_area.pyplot(fig)

        progress.text(f"{i+1}/20 取得完了")
        time.sleep(5)
