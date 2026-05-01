import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import scipy.stats as stats
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🔮 Bitcoin Hourly Forecast – AlphaI × Polaris")
st.markdown("### 95% Prediction Interval for the next hour")

# ---------- Load your backtest metrics ----------
try:
    with open("backtest_metrics.json", "r") as f:
        metrics = json.load(f)
    col1, col2, col3 = st.columns(3)
    col1.metric("Coverage (target 95%)", f"{metrics['coverage_95']:.2%}")
    col2.metric("Avg. Width (USD)", f"${metrics['avg_width_95']:.0f}")
    col3.metric("Winkler Score (lower is better)", f"{metrics['mean_winkler_95']:.2f}")
except:
    st.warning("Backtest metrics not found. Make sure backtest_metrics.json is in the same folder.")

# ---------- Fetch live Bitcoin data ----------
@st.cache_data(ttl=3600)
def get_live_data(limit=500):
    url = "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=" + str(limit)
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    return df[['timestamp', 'close']]

# ---------- GBM forecast (same as backtest) ----------
def gbm_forecast(prices, n_sims=10000, nu=4):
    returns = prices.pct_change().dropna()
    if len(returns) < 5:
        return None, None
    mu = returns.mean()
    sigma = returns.std()
    last_price = prices.iloc[-1]
    sim_returns = stats.t.rvs(df=nu, size=n_sims, loc=mu, scale=sigma)
    sim_prices = last_price * np.exp(sim_returns)
    lower = np.percentile(sim_prices, 2.5)
    upper = np.percentile(sim_prices, 97.5)
    return lower, upper

# ---------- Dashboard UI ----------
with st.spinner("Fetching live Bitcoin data..."):
    df = get_live_data(500)
    if not df.empty:
        lower, upper = gbm_forecast(df['close'])
        last_price = df['close'].iloc[-1]

        st.subheader("📡 Current Forecast")
        c4, c5, c6 = st.columns(3)
        c4.metric("Latest BTC Price", f"${last_price:,.2f}")
        c5.metric("95% Prediction Interval", f"${lower:,.0f} – ${upper:,.0f}")
        c6.metric("Interval Width", f"${upper - lower:,.0f}")

        # Plot with shaded ribbon using Plotly
        st.subheader("📈 Recent Price Action + Forecast")
        import plotly.graph_objects as go
        plot_df = df.tail(50).copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df['timestamp'], y=plot_df['close'],
                                 mode='lines', name='Actual Price'))
        # Shaded region for the predicted hour
        next_ts = plot_df['timestamp'].iloc[-1] + pd.Timedelta(hours=1)
        fig.add_vrect(x0=plot_df['timestamp'].iloc[-1], x1=next_ts,
                      fillcolor="rgba(0,100,255,0.2)", line_width=0,
                      annotation_text=f"Predicted {lower:.0f}–{upper:.0f}")
        fig.update_layout(xaxis_title="Time", yaxis_title="BTC Price (USD)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Failed to fetch data.")