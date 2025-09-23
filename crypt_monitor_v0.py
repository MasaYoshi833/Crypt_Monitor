# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 10:43:18 2025

@author: my199
"""

# app.py
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide", page_title="暗号資産 売買審査システム（プロトタイプ）")

st.title("暗号資産 売買審査システム（価格比較 + 異常検知）")
st.markdown(
    """
    - 複数取引所の同一銘柄を重ねて可視化します。  
    - シンプルな統計（標準偏差 / 他取引所との乖離）でアラートを出します。  
    - 発展形としてIsolationForestを用いた機械学習異常検知も行います（軽量モデル）。
    """
)

# ------------------------------
# 取引所リストとユーザー入力
# ------------------------------
# 利用可能なccxt取引所IDの短いサンプル
EXCHANGE_CHOICES = {
    "Binance (binance)": "binance",
    "Coinbase Pro (coinbasepro)": "coinbasepro",
    "bitFlyer (bitflyer)": "bitflyer",
    "bitbank (bitbank)": "bitbank",
    "Coincheck (coincheck)": "coincheck",
    # 必要に応じてここに追記
}

col1, col2 = st.columns([2, 1])
with col1:
    symbol_input = st.text_input("銘柄（ccxt形式で入力）例: BTC/JPY, BTC/USDT, ETH/USD", value="BTC/JPY")
    selected_exchanges_ui = st.multiselect("取引所を複数選択", options=list(EXCHANGE_CHOICES.keys()),
                                           default=["bitFlyer (bitflyer)", "binance (binance)"])
with col2:
    samples = st.number_input("サンプリング回数（過去データ点数）", min_value=1, max_value=500, value=60)
    interval = st.number_input("サンプリング間隔（秒） — if using live tickers", min_value=0.5, max_value=60.0, value=1.0, step=0.5)
    z_threshold = st.number_input("単純統計の閾値 (σ)（超えたらアラート）", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
    pct_spread_threshold = st.number_input("取引所間差の閾値（%）", min_value=0.1, max_value=50.0, value=2.0, step=0.1)

ml_enable = st.checkbox("IsolationForest による機械学習異常検知を有効にする", value=True)
ml_contamination = st.number_input("IsolationForest contamination（異常想定割合）", min_value=0.001, max_value=0.5, value=0.01, step=0.001, format="%.3f")

fetch_mode = st.radio("データ取得モード", options=["OHLCV（可能なら）で一括取得（推奨）", "Tickerを繰り返し取得（ライブ風）"], index=0)

run_button = st.button("データ取得＆異常検知を実行")

# ------------------------------
# ヘルパー関数
# ------------------------------
def make_exchange_instance(exchange_id):
    try:
        exchange_cls = getattr(ccxt, exchange_id)
        ex = exchange_cls({'enableRateLimit': True})
        # 一部の取引所はOHLCVを使うためにload_marketsが必要
        try:
            ex.load_markets()
        except Exception:
            pass
        return ex
    except Exception as e:
        raise RuntimeError(f"Exchange `{exchange_id}` のインスタンス作成に失敗: {e}")

def try_fetch_ohlcv(ex, symbol, limit=200):
    # 万が一取引所がOHLCVをサポートしない場合は例外
    if not hasattr(ex, 'fetch_ohlcv'):
        raise RuntimeError("OHLCV非対応")
    # ccxtではシンボル表記が取引所ごとに違う場合がある -> ここは試行錯誤の余地あり
    return ex.fetch_ohlcv(symbol, timeframe='1m', limit=limit)

def try_fetch_ticker_last(ex, symbol):
    t = ex.fetch_ticker(symbol)
    # fetch_ticker の standard フィールド 'last' を返す
    return t.get('last', None)

# ------------------------------
# 実行部分
# ------------------------------
if run_button:
    if len(selected_exchanges_ui) < 1:
        st.error("少なくとも1つの取引所を選んでください。")
        st.stop()

    # map UI names to ccxt ids
    selected_exchanges = [EXCHANGE_CHOICES[name] for name in selected_exchanges_ui]

    st.info(f"取得対象: {symbol_input} / 取引所: {', '.join(selected_exchanges)} / サンプル数: {samples} / モード: {fetch_mode}")

    # DataFrame 構築用
    price_frames = {}
    timestamps = None
    ok_exchanges = []
    errors = {}

    # 優先してOHLCVで一括取得（履歴）を試みる
    if fetch_mode.startswith("OHLCV"):
        st.write("OHLCVで一括取得を試みます（サポートしない取引所はティッカー取得へフォールバック）...")
        for ex_id in selected_exchanges:
            try:
                ex = make_exchange_instance(ex_id)
                # OHLCVで過去 samples 分が取得できるか試す（1分足で取得）
                ohlcv = try_fetch_ohlcv(ex, symbol_input, limit=samples)
                # ohlcv: [ [ts, open, high, low, close, volume], ...]
                df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
                df = df.set_index('datetime')
                # 末尾 samples を使う（多く取れる場合）
                df = df.iloc[-samples:]['close']
                price_frames[ex_id] = df.rename(ex_id)
                ok_exchanges.append(ex_id)
                st.success(f"{ex_id}: OHLCV取得成功（{len(df)}点）")
            except Exception as e:
                errors[ex_id] = str(e)
                st.warning(f"{ex_id}: OHLCV取得失敗 -> {e} （ティッカー取得へフォールバック）")

        # 失敗したものをtickerで逐次取得（擬似的に時系列を作る）
        fallback = [ex for ex in selected_exchanges if ex not in ok_exchanges]
        if fallback:
            st.write("ティッカーから逐次取得（フォールバック）: ", fallback)
            # 初期化
            for ex_id in fallback:
                price_frames[ex_id] = []
            timestamps = []
            for i in range(samples):
                ts = pd.Timestamp.now()
                timestamps.append(ts)
                for ex_id in fallback:
                    try:
                        ex = make_exchange_instance(ex_id)
                        price = try_fetch_ticker_last(ex, symbol_input)
                        price_frames[ex_id].append(price)
                    except Exception as e:
                        price_frames[ex_id].append(np.nan)
                        errors[ex_id] = errors.get(ex_id, "") + f" | {e}"
                time.sleep(interval)
            # convert to pandas series
            for ex_id in fallback:
                price_frames[ex_id] = pd.Series(price_frames[ex_id], index=timestamps, name=ex_id)

    else:
        # tickerを繰り返し取得するモード（ライブ風）
        st.write("Ticker を繰り返し取得します（ライブ風）...")
        timestamps = []
        for ex_id in selected_exchanges:
            price_frames[ex_id] = []
        for i in range(samples):
            ts = pd.Timestamp.now()
            timestamps.append(ts)
            for ex_id in selected_exchanges:
                try:
                    ex = make_exchange_instance(ex_id)
                    price = try_fetch_ticker_last(ex, symbol_input)
                    price_frames[ex_id].append(price)
                except Exception as e:
                    price_frames[ex_id].append(np.nan)
                    errors[ex_id] = errors.get(ex_id, "") + f" | {e}"
            time.sleep(interval)
        for ex_id in selected_exchanges:
            price_frames[ex_id] = pd.Series(price_frames[ex_id], index=timestamps, name=ex_id)
        ok_exchanges = selected_exchanges

    # 集約
    try:
        df_all = pd.concat([price_frames[e] for e in price_frames], axis=1)
    except Exception as e:
        st.error(f"データ整形でエラー: {e}")
        st.stop()

    # 欠損値処理（前方/後方埋めで簡易処理）
    df_all = df_all.sort_index().ffill().bfill()

    st.subheader("取得した価格データ（表）")
    st.dataframe(df_all.tail(50))

    # ------------------------------
    # シンプルな異常検知（統計的）
    # ------------------------------
    st.subheader("シンプル異常検知（統計的アラート）")
    # 各時点での中央値との差（標準偏差）
    median_series = df_all.median(axis=1)
    std_series = df_all.std(axis=1).replace(0, 1e-9)  # 0除算回避
    # Zスコア（各セル毎に）
    zscores = (df_all.sub(median_series, axis=0)).div(std_series, axis=0).abs()

    # どの取引所が閾値超えか
    simple_flags = zscores > z_threshold

    # 取引所間の最大割合乖離
    max_per_ts = df_all.max(axis=1)
    min_per_ts = df_all.min(axis=1)
    pct_spread = (max_per_ts - min_per_ts) / min_per_ts * 100.0
    spread_flags = pct_spread > pct_spread_threshold

    # 合成シンプルアラート（時点ベース）
    simple_time_alert = spread_flags | (zscores.gt(z_threshold).any(axis=1))

    st.write(f"Z閾値: {z_threshold}σ / 取引所間差閾値: {pct_spread_threshold}%")
    st.write(f"最新時刻: {df_all.index[-1]}")
    if simple_time_alert.iloc[-1]:
        st.error("⚠️ シンプルアラート: 最新時点で異常が検知されました（統計基準）。")
    else:
        st.success("最新時点: シンプル検知は正常でした。")

    # ------------------------------
    # 機械学習（IsolationForest）
    # ------------------------------
    ml_flags = pd.Series(False, index=df_all.index)
    if ml_enable:
        st.subheader("機械学習による異常検知（IsolationForest）")
        # 特徴量作成: 各取引所のリターン（log return） と スプレッド
        df_returns = df_all.pct_change().fillna(0)
        features = df_returns.copy()
        # スプレッドとボラティリティ特徴量追加
        features['spread_pct'] = pct_spread.fillna(0)
        features['std'] = df_all.std(axis=1).fillna(0)
        # 学習用ウィンドウ: 最後の N-1 行を学習し、最新行を予測（オンライン的）
        if len(features) < 10:
            st.warning("データ点が少ないため、MLはうまく動かない可能性があります（最低でも10点推奨）。")
        try:
            X = features.values
            # 標準化
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            # 学習: 過去データ全体を使う（プロトタイプ）
            clf = IsolationForest(contamination=ml_contamination, random_state=42)
            clf.fit(Xs)

            # 予測（全点）: -1 が異常、1 が正常
            preds = clf.predict(Xs)
            ml_flags = pd.Series(preds == -1, index=features.index)

            # 最新点の結果
            if ml_flags.iloc[-1]:
                st.error("🔴 MLアラート（IsolationForest）: 最新時点で機械学習が異常と判定しました。")
            else:
                st.success("🟢 ML判定: 最新時点は正常です。")
        except Exception as e:
            st.error(f"ML処理でエラー: {e}")
    else:
        st.info("ML検知はオフです。")

    # ------------------------------
    # 可視化（Plotly）
    # ------------------------------
    st.subheader("可視化 — 複数取引所の価格推移とアラート表示")

    fig = go.Figure()

    # 価格ライン
    colors = ['blue', 'green', 'purple', 'orange', 'brown', 'cyan', 'magenta', 'grey']
    for i, col in enumerate(df_all.columns):
        fig.add_trace(go.Scatter(
            x=df_all.index, y=df_all[col],
            mode='lines',
            name=col,
            line=dict(width=1.5),
            hovertemplate="%{y:.2f}<br>%{x}<extra></extra>"
        ))

    # アラートの重ね描画: 最新時点を中心に散布
    latest_ts = df_all.index[-1]
    # prepare markers per exchange for latest point
    marker_x = []
    marker_y = []
    marker_color = []
    marker_text = []
    for col in df_all.columns:
        val = df_all[col].iloc[-1]
        marker_x.append(latest_ts)
        marker_y.append(val)
        # Determine color:
        # - if ML flagged -> red
        # - elif simple zscore for that exchange at latest > threshold -> orange
        # - else green
        col_z = zscores[col].iloc[-1] if col in zscores else 0
        is_simple = col_z > z_threshold or spread_flags.iloc[-1]
        is_ml = ml_flags.iloc[-1] if ml_enable else False
        if is_ml:
            c = "red"
            txt = f"{col} (ML異常)"
        elif is_simple:
            c = "orange"
            txt = f"{col} (統計異常)"
        else:
            c = "green"
            txt = f"{col} (正常)"
        marker_color.append(c)
        marker_text.append(txt)

    fig.add_trace(go.Scatter(
        x=marker_x, y=marker_y,
        mode='markers',
        marker=dict(size=12, color=marker_color, symbol='circle'),
        showlegend=False,
        hovertext=marker_text,
    ))

    # スプレッド（副軸）を折れ線で描画（任意）
    fig.add_trace(go.Scatter(
        x=df_all.index, y=pct_spread,
        mode='lines',
        name='Pct Spread (%)',
        yaxis='y2',
        line=dict(width=1, dash='dash'),
        hovertemplate="%{y:.2f}%<br>%{x}<extra></extra>"
    ))

    # レイアウト設定
    fig.update_layout(
        xaxis=dict(title="Time"),
        yaxis=dict(title=f"Price ({symbol_input})"),
        yaxis2=dict(title="Spread (%)", overlaying='y', side='right', showgrid=False, rangemode='auto'),
        legend=dict(orientation="h"),
        height=600,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # ログとダウンロード
    # ------------------------------
    st.subheader("診断ログ")
    last_row = df_all.iloc[-1].to_frame().T
    log = {
        "timestamp": df_all.index[-1],
        "simple_alert": bool(simple_time_alert.iloc[-1]),
        "pct_spread": float(pct_spread.iloc[-1]),
        "ml_alert": bool(ml_flags.iloc[-1]) if ml_enable else False
    }
    st.json(log)

    # エラー表示
    if errors:
        st.subheader("取得時の注意/エラー（参考）")
        for k, v in errors.items():
            st.write(f"- {k}: {v}")

    # CSV ダウンロード
    csv = df_all.to_csv().encode('utf-8')
    st.download_button("価格データをCSVダウンロード", csv, file_name=f"prices_{symbol_input.replace('/','')}.csv", mime="text/csv")

    st.info("プロトタイプです：本番導入前にAPIレート制限・シンボルマッピング・認証等を整備してください。")
