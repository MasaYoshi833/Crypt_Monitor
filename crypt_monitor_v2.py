# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 13:50:23 2025

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


# ==========================
# Streamlit UI
# ==========================
st.title("暗号資産取引所モニター（リアルタイム）")

symbol = st.selectbox("銘柄を選択してください", ["BTC/JPY", "ETH/JPY", "XRP/JPY"])

EXCHANGES = {
    "bitFlyer": fetch_bitflyer,
    "Coincheck": fetch_coincheck,
    "GMOコイン": fetch_gmo,
}

selected_exchanges = st.multiselect(
    "参照する取引所を選択してください",
    options=list(EXCHANGES.keys()),
    default=["bitFlyer", "Coincheck", "GMOコイン"]
)

st.subheader("リアルタイム価格モニター（5秒更新）")

if st.button("開始"):
    prices = {ex: [] for ex in selected_exchanges}
    timestamps = []
    chart_area = st.empty()
    alert_area = st.empty()

    for i in range(120):  # 約10分間
        row = {}
        for ex in selected_exchanges:
            price = EXCHANGES[ex]()  # API呼び出し
            row[ex] = price
            prices[ex].append(price)
        timestamps.append(pd.Timestamp.now(tz="Asia/Tokyo"))

        df = pd.DataFrame(prices, index=timestamps)

        # ==== 異常検知 ====
        latest_prices = df.iloc[-1].dropna()
        if not latest_prices.empty:
            avg_price = latest_prices.mean()
            for ex, price in latest_prices.items():
                if price is not None and abs(price - avg_price) / avg_price > 0.03:
                    alert_area.warning(f"⚠️ {ex} の価格が平均から3%以上乖離: {price:,.0f} 円")

        # ==== グラフ表示 ====
        fig = px.line(df, x=df.index, y=df.columns, title=f"{symbol} 各取引所の価格推移")
        fig.update_layout(
            yaxis=dict(rangemode="tozero"),  # ズームアウト時に0円まで縮小可能
            xaxis_title="時間（日本時間）",
            yaxis_title="価格（円）"
        )

        chart_area.plotly_chart(fig, use_container_width=True)
        time.sleep(5)
