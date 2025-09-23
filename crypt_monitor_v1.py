# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 13:10:44 2025

@author: my199
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone, timedelta
import time

st.set_page_config(page_title="Crypto Monitor", layout="wide")

# ===== 各取引所のAPI設定 =====
EXCHANGES = {
    "bitFlyer": {
        "url": "https://api.bitflyer.com/v1/ticker?product_code=BTC_JPY",
        "field": "ltp",
    },
    "GMO Coin": {
        "url": "https://api.coin.z.com/public/v1/ticker?symbol=BTC_JPY",
        "field": "data",
    },
    "Coincheck": {
        "url": "https://coincheck.com/api/ticker",
        "field": "last",
    }
}

# ===== データ取得関数 =====
def fetch_price(exchange_name, config):
    try:
        r = requests.get(config["url"], timeout=5)
        data = r.json()
        if exchange_name == "bitFlyer":
            return data["ltp"]
        elif exchange_name == "GMO Coin":
            return float(data["data"][0]["last"])
        elif exchange_name == "Coincheck":
            return data["last"]
    except Exception as e:
        return None

# ===== メイン処理 =====
st.title("📊 暗号資産取引所モニター（BTC/JPY）")

# 保存用セッションステート
if "price_history" not in st.session_state:
    st.session_state.price_history = []

# 価格取得
prices = {}
for name, cfg in EXCHANGES.items():
    prices[name] = fetch_price(name, cfg)

# JSTタイムスタンプ
jst = timezone(timedelta(hours=9))
timestamp = datetime.now(jst).strftime("%Y-%m-%d %H:%M:%S")

# データを履歴に追加
st.session_state.price_history.append({"time": timestamp, **prices})

# ===== 異常検知 =====
alerts = []
valid_prices = [p for p in prices.values() if p is not None]
if valid_prices:
    avg_price = sum(valid_prices) / len(valid_prices)
    for name, price in prices.items():
        if price is not None:
            diff = (price - avg_price) / avg_price * 100
            if abs(diff) >= 5:  # ±5%以上の乖離をアラート
                alerts.append(f"⚠️ {name} の価格が平均比 {diff:.2f}% 乖離しています")

if alerts:
    st.error("\n".join(alerts))

# ===== 表示 =====
df = pd.DataFrame(st.session_state.price_history)

st.subheader("最新価格")
st.dataframe(df.tail(1).set_index("time"))

# グラフ
fig = px.line(df, x="time", y=df.columns[1:], title="取引所別 BTC/JPY 価格推移")
fig.update_layout(
    yaxis=dict(rangemode="tozero", title="価格（円）"),
    xaxis_title="時間（JST）",
)
st.plotly_chart(fig, use_container_width=True)

# ===== 自動更新（5秒） =====
st.write("⏳ 5秒ごとに自動更新します...")
time.sleep(5)
st.experimental_rerun()
