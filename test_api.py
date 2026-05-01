import requests
import pandas as pd
from datetime import datetime, timezone

url = "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=5"
resp = requests.get(url)
data = resp.json()
df = pd.DataFrame(data, columns=[
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

# drop last
df_closed = df.iloc[:-1].copy().reset_index(drop=True)

last_close_time = df_closed['close_time'].iloc[-1]
print("Last close time:", last_close_time)

now_utc = datetime.now(timezone.utc)
print("Now UTC:", now_utc)

next_candle_time = last_close_time.replace(tzinfo=timezone.utc) + pd.Timedelta(hours=1)
print("Next candle time:", next_candle_time)

time_rem = next_candle_time - now_utc
print("Time rem total seconds:", time_rem.total_seconds())

