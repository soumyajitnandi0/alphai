import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import scipy.stats as stats
from datetime import datetime, timezone
import os

st.set_page_config(layout="wide")
st.title("🔮 Bitcoin Hourly Forecast – AlphaI × Polaris")
st.markdown("### 95% Prediction Interval for the next hour")

# ---------- Load backtest metrics ----------
try:
    with open("backtest_metrics.json", "r") as f:
        metrics = json.load(f)
    col1, col2, col3 = st.columns(3)
    col1.metric("Coverage (target 95%)", f"{metrics['coverage_95']:.2%}")
    col2.metric("Avg. Width (USD)", f"${metrics['avg_width_95']:.0f}")
    col3.metric("Winkler Score (lower is better)", f"{metrics['mean_winkler_95']:.2f}")
except FileNotFoundError:
    st.warning("Backtest metrics not found. Make sure backtest_metrics.json is in the same folder.")
except Exception as e:
    st.error(f"Error loading metrics: {e}")

# ---------- Fetch live Bitcoin data ----------
@st.cache_data(ttl=3600)
def get_live_data(limit=500):
    url = f"https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1h&limit={limit}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    df['close'] = df['close'].astype(float)
    # Drop the in-progress candle (last row) to ensure we only use fully closed bars
    df_closed = df.iloc[:-1].copy().reset_index(drop=True)
    return df_closed[['timestamp', 'close_time', 'close']]

# ---------- Student-t forecast (with dynamic fitting and scale inflation) ----------
def gbm_forecast(prices, n_sims=10000, scale_inflate=1.03):
    prices_array = prices.to_numpy(dtype=float)
    # Calculate log returns
    returns = np.diff(np.log(prices_array))
    if len(returns) < 30:
        return None, None
    
    last_price = prices_array[-1]
    
    # Fit Student-t distribution to log returns
    df_hat, loc, scale = stats.t.fit(returns)
    df_hat = float(np.clip(df_hat, 3.0, 30.0))
    scale = float(max(scale * scale_inflate, 1e-12))
    
    # Simulate future returns
    rng = np.random.default_rng()
    sim_returns = stats.t(df_hat, loc=loc, scale=scale).rvs(size=n_sims, random_state=rng)
    sim_prices = last_price * np.exp(sim_returns)
    
    lower = np.percentile(sim_prices, 2.5)
    upper = np.percentile(sim_prices, 97.5)
    return float(lower), float(upper)

# ---------- Part C: Persistence ----------
HISTORY_FILE = "prediction_history.jsonl"

def append_history(entry):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def read_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    rows = []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

# ---------- Dashboard UI ----------
with st.spinner("Fetching live Bitcoin data..."):
    df = get_live_data(limit=501) # fetch an extra to account for dropping the last forming candle
    if not df.empty:
        lower, upper = gbm_forecast(df['close'])
        last_price = df['close'].iloc[-1]
        last_close_time = df['close_time'].iloc[-1]

        st.subheader("📡 Current Forecast")
        st.write(f"Last **fully closed** 1h bar (UTC): `{last_close_time}`")
        c4, c5, c6 = st.columns(3)
        c4.metric("Latest Closed Price", f"${last_price:,.2f}")
        c5.metric("95% Prediction Interval", f"${lower:,.0f} – ${upper:,.0f}")
        c6.metric("Interval Width", f"${upper - lower:,.0f}")

        # Plot with shaded ribbon using Plotly
        st.subheader("📈 Recent Price Action + Forecast")
        import plotly.graph_objects as go
        plot_df = df.tail(50).copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df['timestamp'], y=plot_df['close'],
                                 mode='lines', name='Actual Price', line=dict(color="#f7931a", width=2)))
        # Shaded region for the predicted hour
        fig.add_hrect(
            y0=lower, y1=upper, line_width=0, fillcolor="rgba(0,128,255,0.15)", name="95% next-hour range"
        )
        fig.update_layout(
            xaxis_title="Time (UTC)", yaxis_title="BTC Price (USDT)",
            height=460, template="plotly_dark",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("Data: https://data-api.binance.vision — last kline excluded as it is still forming.")

        # Part C: Visit Logging
        entry = {
            "logged_at_utc": datetime.now(timezone.utc).isoformat(),
            "anchor_close_time_utc": last_close_time.isoformat(),
            "scale_inflate_used": 1.03,
            "s0_close": float(last_price),
            "predict_low": lower,
            "predict_high": upper,
        }
        append_history(entry)
        
        hist = read_history()
        if hist:
            st.divider()
            st.subheader("Visit Log (Part C)")
            st.dataframe(pd.DataFrame(hist).tail(200), use_container_width=True, height=280)
    else:
        st.error("Failed to fetch data from Binance.")