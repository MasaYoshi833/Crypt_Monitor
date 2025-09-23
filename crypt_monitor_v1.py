# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 13:10:44 2025

@author: my199
"""

import streamlit as st
import requests
import pandas as pd
import time
import plotly.express as px

st.set_page_config(page_title="暗号資産取引所モニター", layout="wide")

# ==========================
# API取得関数
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
        url = "https://coincheck.com/api/ticker"
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

def fetch_liquid(symbol="5"):  # 5=BTC/JPY product_id
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

def fetch_bybit(symbol="BTCUSDT"):
    try:
        url = f"https://api.bybit.com/v5/market/tickers?category=spot&symbol={symbol}"
        res = requests.get(url).json()
        return float(res["result"]["list"][0]["lastPrice"])
    except:
        return None

def fetch_kraken(symbol="XXBTZUSD"):
    try:
        url = f"https://api.kraken.com/0/public/Ticker?pair={symbol}"
        res = requests.get(url).json()
        pair = list(res["result"].keys())[0]
        return float(res["result"][pair]["c"][0])
    except:
        return None

# ==========================
# Streamlit UI
# ==========================
st.title("暗号資産取引所モニター（リアルタイム）")

symbol = st.selectbox("銘柄を選択してください", ["BTC/JPY", "ETH/JPY", "XRP/JPY"])

EXCHANGES = {
    "bitFlyer": fetch_bitflyer,
    "Coincheck": fetch_coincheck,
    "GMOコイン": fetch_gmo,
    "Zaif": fetch_zaif,
    "Liquid": fetch_liquid,
    "Binance(USDT)": fetch_binance,
    "Bybit(USDT)": fetch_bybit,
    "Kraken(USD)": fetch_kraken,
}

selected_exchanges = st.multiselect(
    "参照する取引所を選択してください",
    options=list(EXCHANGES.keys()),
    default=["bitFlyer", "Coincheck", "GMOコイン"]
)

st.subheader("リアルタイム価格モニター（10秒更新）")

if st.button("開始"):
    prices = {ex: [] for ex in selected_exchanges}
    timestamps = []
    chart_area = st.empty()

    for i in range(60):  # 約10分間
        row = {}
        for ex in selected_exchanges:
            price = EXCHANGES[ex]()  # API呼び出し
            row[ex] = price
            prices[ex].append(price)
        timestamps.append(pd.Timestamp.now(tz="Asia/Tokyo"))

        df = pd.DataFrame(prices, index=timestamps)
        fig = px.line(df, x=df.index, y=df.columns, title=f"{symbol} 各取引所の価格推移")
        fig.update_layout(
            yaxis=dict(rangemode="normal"),  # 初期表示は各価格の差をそのまま
            xaxis_title="時間（日本時間）",
            yaxis_title="価格"
        )

        chart_area.plotly_chart(fig, use_container_width=True)
        time.sleep(10)
