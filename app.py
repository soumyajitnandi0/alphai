import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import scipy.stats as stats
from datetime import datetime, timezone
import os

st.set_page_config(layout="wide")
st.title("🔮 Bitcoin Hourly Forecast – Alphai")
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
@st.cache_data(ttl=120)
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
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    # Only drop the last candle if it is currently forming
    now_utc_naive = datetime.now(timezone.utc).replace(tzinfo=None)
    if df['close_time'].iloc[-1] > now_utc_naive:
        df_closed = df.iloc[:-1].copy().reset_index(drop=True)
    else:
        df_closed = df.copy()
        
    return df_closed[['timestamp', 'close_time', 'open', 'high', 'low', 'close', 'volume']]

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
        
        # The UI timer should always count down to the top of the next real-world hour
        now_utc = datetime.now(timezone.utc)
        next_top_of_hour = now_utc.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)
        ui_time_rem = next_top_of_hour - now_utc
        
        # For background refresh logic, we need to know if the data itself is lagging
        next_candle_time = last_close_time.replace(tzinfo=timezone.utc) + pd.Timedelta(hours=1)
        data_time_rem = next_candle_time - now_utc

        c4, c5, c6, c7 = st.columns(4)
        c4.metric("Latest Closed Price", f"${last_price:,.2f}")
        c5.metric("95% Prediction Interval", f"${lower:,.0f} – ${upper:,.0f}")
        c6.metric("Interval Width", f"${upper - lower:,.0f}")
        
        with c7:
            import streamlit.components.v1 as components
            timer_html = f"""
            <style>
                body {{ margin: 0; font-family: "Source Sans Pro", sans-serif; background-color: transparent; }}
                .metric-label {{ font-size: 14px; color: rgba(250, 250, 250, 0.6); margin-bottom: 4px; }}
                .metric-value {{ font-size: 1.8rem; font-weight: 400; color: rgb(250, 250, 250); }}
            </style>
            <div>
                <div class="metric-label">⏳ Next Prediction In</div>
                <div class="metric-value" id="countdown">--m --s</div>
            </div>
            <script>
                var distance = {int(ui_time_rem.total_seconds() * 1000)};
                function updateTimer() {{
                    if (distance < 0) {{
                        document.getElementById("countdown").innerHTML = "0m 0s";
                    }} else {{
                        var m = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
                        var s = Math.floor((distance % (1000 * 60)) / 1000);
                        document.getElementById("countdown").innerHTML = m + "m " + s + "s";
                    }}
                    distance -= 1000;
                }}
                updateTimer(); // run immediately
                setInterval(updateTimer, 1000);
            </script>
            """
            components.html(timer_html, height=80)

        # Plot with shaded ribbon using Plotly
        st.subheader("📈 Recent Price Action + Forecast")
        import plotly.graph_objects as go
        plot_df = df.tail(50).copy()
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=plot_df['timestamp'],
            open=plot_df['open'],
            high=plot_df['high'],
            low=plot_df['low'],
            close=plot_df['close'],
            name='Actual Price'
        ))
        fig.update_layout(xaxis_rangeslider_visible=False)
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
            hist_df = pd.DataFrame(hist)
            # Parse dates
            hist_df['logged_at_utc'] = pd.to_datetime(hist_df['logged_at_utc'])
            hist_df['anchor_close_time_utc'] = pd.to_datetime(hist_df['anchor_close_time_utc'])
            
            # Calculate range width
            hist_df['range_width'] = hist_df['predict_high'] - hist_df['predict_low']
            
            # Target close time is exactly 1 hour after the anchor close time
            hist_df['target_close_time'] = hist_df['anchor_close_time_utc'] + pd.Timedelta(hours=1)
            
            # Map actual close from the current live data (if available)
            close_map = df.set_index('close_time')['close'].to_dict()
            hist_df['actual_close'] = hist_df['target_close_time'].map(close_map)
            
            # Calculate Winkler Score (alpha = 0.05 for 95% interval)
            alpha = 0.05
            def calc_winkler(row):
                if pd.isna(row['actual_close']):
                    return np.nan
                width = row['range_width']
                actual = row['actual_close']
                lower = row['predict_low']
                upper = row['predict_high']
                
                if actual < lower:
                    return width + (2 / alpha) * (lower - actual)
                elif actual > upper:
                    return width + (2 / alpha) * (actual - upper)
                else:
                    return width
            
            hist_df['winkler_score'] = hist_df.apply(calc_winkler, axis=1)

            # Check if prediction was correct
            def check_hit(row):
                if pd.isna(row['actual_close']):
                    return "⏳ Pending"
                actual = row['actual_close']
                lower = row['predict_low']
                upper = row['predict_high']
                if actual < lower:
                    return "❌ Miss (Low)"
                elif actual > upper:
                    return "❌ Miss (High)"
                else:
                    return "✅ Hit"

            hist_df['result'] = hist_df.apply(check_hit, axis=1)
            
            # Format pending values nicely
            hist_df['actual_close_fmt'] = hist_df['actual_close'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "Pending")
            hist_df['winkler_score_fmt'] = hist_df['winkler_score'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "Pending")

            # Rename columns for proper display
            rename_dict = {
                'logged_at_utc': 'Logged At (UTC)',
                'anchor_close_time_utc': 'Anchor Time (UTC)',
                's0_close': 'Anchor Close ($)',
                'predict_low': 'Predict Low ($)',
                'predict_high': 'Predict High ($)',
                'range_width': 'Range Width ($)',
                'actual_close_fmt': 'Actual Close',
                'winkler_score_fmt': 'Winkler Score',
                'result': 'Result',
                'scale_inflate_used': 'Scale Inflate'
            }
            display_df = hist_df.rename(columns=rename_dict)
            
            # Order the columns nicely
            cols_order = [
                'Logged At (UTC)', 'Anchor Time (UTC)', 'Anchor Close ($)', 
                'Predict Low ($)', 'Predict High ($)', 'Range Width ($)', 
                'Actual Close', 'Result', 'Winkler Score', 'Scale Inflate'
            ]
            display_df = display_df[[c for c in cols_order if c in display_df.columns]]
            
            st.divider()
            st.subheader("Prediction History & Validation")
            st.dataframe(
                display_df, 
                use_container_width=True, 
                height=280,
                column_config={
                    'Logged At (UTC)': st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
                    'Anchor Time (UTC)': st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
                    'Anchor Close ($)': st.column_config.NumberColumn(format="$%.2f"),
                    'Predict Low ($)': st.column_config.NumberColumn(format="$%.2f"),
                    'Predict High ($)': st.column_config.NumberColumn(format="$%.2f"),
                    'Range Width ($)': st.column_config.NumberColumn(format="$%.2f"),
                    'Actual Close': st.column_config.TextColumn(),
                    'Result': st.column_config.TextColumn(),
                    'Winkler Score': st.column_config.TextColumn(),
                }
            )

        # Part D: Auto-Refresh using JS injection
        import streamlit.components.v1 as components
        if data_time_rem.total_seconds() > 0:
            # Refresh 15 seconds after the candle closes to ensure Binance API has the new candle
            refresh_ms = int(data_time_rem.total_seconds() * 1000) + 15000
            components.html(
                f"""
                <script>
                    setTimeout(function() {{
                        window.parent.location.reload();
                    }}, {refresh_ms});
                </script>
                """,
                height=0, width=0
            )
        else:
            # If the time has already passed but we haven't loaded the new bar yet, refresh every 60 seconds
            # This happens if the Binance Vision API is heavily cached and lagging behind real-time
            components.html(
                """
                <script>
                    setTimeout(function() {
                        window.parent.location.reload();
                    }, 60000);
                </script>
                """,
                height=0, width=0
            )

    else:
        st.error("Failed to fetch data from Binance.")