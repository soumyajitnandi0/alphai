import pandas as pd
import json

# mock history
hist = [
    {"anchor_close_time_utc": "2026-05-01T12:59:59.999000"}
]
hist_df = pd.DataFrame(hist)
hist_df['anchor_close_time_utc'] = pd.to_datetime(hist_df['anchor_close_time_utc'])
hist_df['target_close_time'] = hist_df['anchor_close_time_utc'] + pd.Timedelta(hours=1)

# mock live data
# simulate Binance close_time timestamp for 13:59:59.999
# 13:59:59.999 on May 1 2026
ts = pd.Timestamp("2026-05-01 13:59:59.999000")
print("Target from history:", hist_df['target_close_time'][0])
print("Simulated ts:", ts)

# Let's check equality
print("Are they equal?", hist_df['target_close_time'][0] == ts)
