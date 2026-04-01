"""
data_prep.py

Downloads the four FRED yield series used for parameter calibration
and saves a cleaned Parquet file to data/processed/fred_yields.parquet.

Run with:
    python3 data_prep.py

Requires the FRED_API_KEY environment variable to be set.
"""

import os
import sys
import pathlib
import numpy as np
import pandas as pd
from fredapi import Fred

# -- Configuration ---------------------------------------------------------
START_DATE   = "2015-01-01"
END_DATE     = "2026-03-18"
TICKERS      = ["DGS3MO", "DGS2", "DGS10", "T10Y2Y"]
COLUMN_NAMES = ["yield_3m", "yield_2y", "yield_10y", "spread_10y2y"]
OUT_DIR      = pathlib.Path("data") / "processed"
OUT_FILE     = OUT_DIR / "fred_yields.parquet"

# -- API key ---------------------------------------------------------------
api_key = os.environ.get("FRED_API_KEY")
if not api_key:
    sys.exit(
        "Error: FRED_API_KEY environment variable is not set.\n"
        "Register for a free key at fred.stlouisfed.org/docs/api/api_key.html\n"
        "Then: export FRED_API_KEY='your_key_here'"
    )

# -- Download --------------------------------------------------------------
fred = Fred(api_key=api_key)

series_list = []
for ticker, col_name in zip(TICKERS, COLUMN_NAMES):
    s = fred.get_series(ticker, observation_start=START_DATE,
                        observation_end=END_DATE)
    s.name = col_name
    series_list.append(s)

df = pd.concat(series_list, axis=1)

# -- Clean -----------------------------------------------------------------
df = df.dropna()
df.index = pd.to_datetime(df.index)
df.index.name = "date"

# -- Daily first differences (delta r_t = r_{t+1} - r_t) ------------------
df["delta_3m"]  = df["yield_3m"].diff()
df["delta_2y"]  = df["yield_2y"].diff()
df["delta_10y"] = df["yield_10y"].diff()
df = df.dropna()

# -- Annualised rate volatility --------------------------------------------
TRADING_DAYS = 252
df.attrs["ann_vol_3m"]  = float(df["delta_3m"].std()  * np.sqrt(TRADING_DAYS))
df.attrs["ann_vol_2y"]  = float(df["delta_2y"].std()  * np.sqrt(TRADING_DAYS))
df.attrs["ann_vol_10y"] = float(df["delta_10y"].std() * np.sqrt(TRADING_DAYS))

# -- Save ------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_FILE)

# -- Report ----------------------------------------------------------------
print(f"Saved {len(df)} rows")
print(f"  Date range: {df.index[0].date()}  to  {df.index[-1].date()}")
print(f"  Columns:    {list(df.columns)}")
print(f"  Ann. vol (3M):  {df.attrs['ann_vol_3m']:.6f}")
print(f"  Ann. vol (2Y):  {df.attrs['ann_vol_2y']:.6f}")
print(f"  Ann. vol (10Y): {df.attrs['ann_vol_10y']:.6f}")
print("data_prep.py complete.")
