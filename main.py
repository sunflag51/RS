"""米国11セクターETFとSPYの相対パフォーマンスを確認するStreamlitアプリ。"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

APP_TITLE = "米国11セクター RS ダッシュボード"
SECTORS: Dict[str, str] = {
    "XLB": "素材", "XLC": "コミュニケーション・サービス", "XLE": "エネルギー",
    "XLF": "金融", "XLI": "資本財", "XLK": "情報技術",
    "XLP": "生活必需品", "XLRE": "不動産", "XLU": "公益事業",
    "XLV": "ヘルスケア", "XLY": "一般消費財",
}
SPY = "SPY"

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="centered", initial_sidebar_state="collapsed")
st.title("📊 米国11セクター RS ダッシュボード")
st.caption("Sector SPDR 11セクターETFをSPYと比較し、相対的な強さと市場の偏りを見やすくします。")

with st.sidebar:
    st.header("表示設定")
    period = st.selectbox("データ期間", ["6か月", "1年", "2年", "5年"], index=1)
    lookback = st.selectbox("ランキングの比較日数", [20, 60, 120], index=0, format_func=lambda x: f"{x}営業日")
    chart_mode = st.radio("チャート", ["SPYを100とした相対パフォーマンス", "SPY比率（RSライン）"], index=0)
    selected = st.multiselect("チャートに表示するセクター", list(SECTORS), default=list(SECTORS), format_func=lambda x: f"{x}  {SECTORS[x]}")
    st.divider()
    st.caption("データ: Yahoo Finance（yfinance）。iPhoneでは画面右上のメニューから表示期間などを変更できます。")

period_map = {"6か月": "6mo", "1年": "1y", "2年": "2y", "5年": "5y"}

@st.cache_data(ttl=900, show_spinner=False)
def load_market_data(period_value: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = list(SECTORS) + [SPY]
    raw = yf.download(symbols, period=period_value, interval="1d", auto_adjust=True,
                      group_by="column", progress=False, threads=True)
    if raw is None or raw.empty:
        raise RuntimeError("株価データを取得できませんでした。しばらくしてから再読み込みしてください。")
    if isinstance(raw.columns, pd.MultiIndex):
        # yfinanceのバージョンによって列階層の順番が違う場合に対応
        labels = [set(raw.columns.get_level_values(i)) for i in range(raw.columns.nlevels)]
        price_field = next((name for name in ("Close", "Adj Close") if any(name in level for level in labels)), None)
        if price_field is None:
            raise RuntimeError("取得データに終値の列が見つかりませんでした。")
        price_level = next(i for i, values in enumerate(labels) if price_field in values)
        volume_level = next((i for i, values in enumerate(labels) if "Volume" in values), None)
        close = raw.xs(price_field, axis=1, level=price_level, drop_level=True)
        volume = raw.xs("Volume", axis=1, level=volume_level, drop_level=True) if volume_level is not None else pd.DataFrame(index=raw.index)
    else:
        close = raw[["Close"]].rename(columns={"Close": SPY}) if "Close" in raw else raw[["Adj Close"]].rename(columns={"Adj Close": SPY})
        volume = raw[["Volume"]].rename(columns={"Volume": SPY})
    close.columns = [str(c) for c in close.columns]
    volume.columns = [str(c) for c in volume.columns]
    close = close.reindex(columns=symbols).apply(pd.to_numeric, errors="coerce").dropna(how="all")
    volume = volume.reindex(columns=symbols).apply(pd.to_numeric, errors="coerce")
    return close, volume

try:
    with st.spinner("株価と出来高を取得しています…"):
        prices, volumes = load_market_data(period_map[period])
except Exception as exc:
    st.error(str(exc))
    st.info("インターネット接続を確認し、画面右上の再実行を押してください。")
    st.stop()

available = [s for s in list(SECTORS) + [SPY] if s in prices and prices[s].notna().sum() > 1]
missing = [s for s in list(SECTORS) + [SPY] if s not in available]
if SPY not in available:
    st.error("基準となるSPYのデータがありません。時間をおいて再読み込みしてください。")
    st.stop()
if missing:
    st.warning("一部ETFのデータがありません: " + ", ".join(missing))

prices = prices[available].ffill()
volumes = volumes.reindex(columns=available)
valid = prices.dropna(subset=[SPY])
if len(valid) < 2:
    st.error("比較に必要な株価データが不足しています。")
    st.stop()

# 最新日に揃え、指定営業日前の値と比較（データ不足時は取得可能な最古日を使用）
end_date = valid.index[-1]
start_pos = max(0, len(valid) - 1 - lookback)
base_date = valid.index[start_pos]
start_prices = valid.iloc[start_pos]
end_prices = valid.iloc[-1]
actual_days = len(valid) - 1 - start_pos
returns = (end_prices / start_prices - 1) * 100
spy_return = float(returns[SPY])
sector_returns = returns.drop(labels=[SPY], errors="ignore")
excess = sector_returns - spy_return

# RS100 = 期間中のセクター騰落率とSPY騰落率を起点100で比較
rebased = prices.loc[base_date:end_date] / prices.loc[base_date] * 100
relative = (prices.loc[base_date:end_date].div(prices.loc[base_date], axis="columns")
            .div((prices.loc[base_date:end_date][SPY] / prices.loc[base_date][SPY]), axis=0) * 100)

leaders = int((excess > 0).sum())
col1, col2, col3, col4 = st.columns(4)
col1.metric(f"SPY 騰落率（{actual_days}営業日）", f"{spy_return:+.2f}%")
col2.metric("SPYを上回るセクター", f"{leaders} / {len(sector_returns)}")
if len(excess):
    best = excess.idxmax()
    worst = excess.idxmin()
    col3.metric("相対強度トップ", f"{best} {SECTORS.get(best, '')}", f"SPY比 {excess[best]:+.2f}pt")
    col4.metric("相対強度ワースト", f"{worst} {SECTORS.get(worst, '')}", f"SPY比 {excess[worst]:+.2f}pt")

st.subheader("セクター比較")
show = [s for s in selected if s in available and s != SPY]
if not show:
    st.info("左側から表示するセクターを選択してください。")
else:
    if chart_mode == "SPYを100とした相対パフォーマンス":
        plot_data = relative[show].rename(columns={s: f"{s} {SECTORS[s]}" for s in show}).reset_index(names="日付")
        fig = px.line(plot_data, x="日付", y=plot_data.columns[1:], title=f"SPY比 RSライン（比較起点 = 100、{base_date.date()} 起点）")
        fig.add_hline(y=100, line_dash="dash", line_color="#777", annotation_text="SPYと同じ動き")
        fig.update_yaxes(title="SPY比（100を上回るとSPYより強い）")
    else:
        plot_data = rebased[[SPY] + show].rename(columns={s: f"{s} {SECTORS.get(s, 'S&P 500')}" for s in [SPY] + show}).reset_index(names="日付")
        fig = px.line(plot_data, x="日付", y=plot_data.columns[1:], title=f"価格パフォーマンス（{base_date.date()} = 100）")
        fig.add_hline(y=100, line_dash="dash", line_color="#777")
        fig.update_yaxes(title="起点を100とした指数")
    fig.update_layout(height=440, hovermode="x unified", legend_title="ETF", margin=dict(l=10, r=10, t=65, b=10), font=dict(size=12))
    st.plotly_chart(fig, use_container_width=True)

st.subheader(f"{actual_days}営業日の相対強度ランキング")
rows = []
for ticker, ret in sector_returns.items():
    # 直近5日平均出来高 / それ以前20日平均出来高を活動量の参考値として表示
    vol = volumes[ticker].dropna() if ticker in volumes else pd.Series(dtype=float)
    recent_vol = vol.loc[:end_date].tail(5).mean() if len(vol) else np.nan
    prior_vol = vol.loc[:end_date].iloc[-25:-5].mean() if len(vol) >= 10 else np.nan
    vol_ratio = recent_vol / prior_vol if pd.notna(prior_vol) and prior_vol > 0 else np.nan
    rows.append({
        "順位": 0, "ETF": ticker, "セクター": SECTORS.get(ticker, ""),
        f"{actual_days}日騰落率": float(ret), "SPY騰落率": spy_return,
        "SPYとの差": float(excess[ticker]), "RSライン": float(relative[ticker].iloc[-1]),
        "直近出来高比*": float(vol_ratio) if pd.notna(vol_ratio) else np.nan,
    })
table = pd.DataFrame(rows).sort_values("SPYとの差", ascending=False).reset_index(drop=True)
table["順位"] = np.arange(1, len(table) + 1)

def color_excess(value):
    return "color: #087443; font-weight: bold" if value > 0 else ("color: #b42318; font-weight: bold" if value < 0 else "")
formatters = {f"{actual_days}日騰落率": "{:+.2f}%", "SPY騰落率": "{:+.2f}%", "SPYとの差": "{:+.2f} pt", "RSライン": "{:.2f}", "直近出来高比*": "{:.2f}倍"}
st.dataframe(table.style.format(formatters, na_rep="—").map(color_excess, subset=["SPYとの差"]), use_container_width=True, hide_index=True)
st.caption("* 直近5営業日の平均出来高 ÷ その前の最大20営業日の平均出来高。売買の活発さの目安で、資金流入額そのものではありません。")

st.subheader("相対強度マップ")
heat = table.set_index("ETF")[[f"{actual_days}日騰落率", "SPYとの差"]]
heat.index = [f"{t} {SECTORS.get(t, '')}" for t in heat.index]
heat_fig = px.imshow(heat, aspect="auto", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                     labels={"x": "指標", "y": "セクター", "color": "% / pt"}, text_auto=".2f", title="セクター別の騰落率とSPYとの差")
heat_fig.update_layout(height=390, margin=dict(l=5, r=5, t=55, b=5), font=dict(size=11))
st.plotly_chart(heat_fig, use_container_width=True)

with st.expander("この画面の見方・『資金が流れていそう』の解釈"):
    st.markdown("""
- **RSラインが100より上**: 選択した期間に、そのセクターETFがSPYより値上がりしています。100より下ならSPYより弱い動きです。
- **SPYとの差**: セクターETFの騰落率 − SPYの騰落率です。例: セクター +8%、SPY +5%なら +3ポイントです。
- **相対強度トップ**: 期間中にSPYを最も上回ったセクターです。将来の上昇を保証するものではありません。
- **出来高比**: 売買の活発さの変化を見る補助情報です。ETFの実際の純資金流入を示す数字ではありません。
- このアプリは価格と出来高で市場の強弱を比較します。資金流入額を直接確認するには、ETFの設定・解約（fund flows）データが必要です。上昇を「資金が入った」と断定しないでください。
""")

st.caption(f"取得データ最終日: {end_date.date()} ｜ 比較開始日: {base_date.date()} ｜ 数値は調整後終値を使用。投資判断の参考情報であり、売買推奨ではありません。")
