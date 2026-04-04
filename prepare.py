#!/usr/bin/env python3
"""One-time data prep: parquet → raw price tensors. Do not modify."""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

DATA = Path(__file__).parent / "data"
OUT = DATA

# ── Config ──
TICKERS = ["SPY", "QQQ", "IWM", "EFA", "VEA", "VWO", "EEM",
           "TLT", "IEF", "SHY", "TIP", "HYG",
           "GLD", "DBC", "USO", "VNQ", "UUP"]
N_ASSETS = len(TICKERS)
START_DATE = "2007-08-01"  # all 17 ETFs have data from here


def main():
    print("Loading parquet...")
    df = pd.read_parquet(DATA / "etf_daily.parquet")

    # Filter to common date range
    df = df.loc[START_DATE:]

    # Pivot to (date, ticker) → price matrices
    dates = df.index.get_level_values("date").unique().sort_values()
    dates = dates[dates >= START_DATE]

    # Build price matrix: (T, N_ASSETS) for Adj Close
    adj_close = df["Adj Close"].unstack("ticker")[TICKERS]
    adj_close = adj_close.loc[START_DATE:]
    adj_close = adj_close.ffill().bfill()  # fill any gaps

    # Log returns
    log_ret = np.log(adj_close / adj_close.shift(1))
    log_ret = log_ret.iloc[1:]  # drop first NaN row
    adj_close = adj_close.iloc[1:]

    # Also keep OHLCV for potential feature engineering in train.py
    open_px = df["Open"].unstack("ticker")[TICKERS].loc[START_DATE:].ffill().bfill().iloc[1:]
    high_px = df["High"].unstack("ticker")[TICKERS].loc[START_DATE:].ffill().bfill().iloc[1:]
    low_px = df["Low"].unstack("ticker")[TICKERS].loc[START_DATE:].ffill().bfill().iloc[1:]
    volume = df["Volume"].unstack("ticker")[TICKERS].loc[START_DATE:].ffill().bfill().iloc[1:]

    T = len(log_ret)
    print(f"Date range: {log_ret.index[0].date()} ~ {log_ret.index[-1].date()}")
    print(f"Shape: {T} days × {N_ASSETS} assets")

    # Save as tensors
    tensors = {
        "adj_close": torch.tensor(adj_close.values, dtype=torch.float32),  # (T, N)
        "log_return": torch.tensor(log_ret.values, dtype=torch.float32),   # (T, N)
        "open": torch.tensor(open_px.values, dtype=torch.float32),         # (T, N)
        "high": torch.tensor(high_px.values, dtype=torch.float32),         # (T, N)
        "low": torch.tensor(low_px.values, dtype=torch.float32),           # (T, N)
        "volume": torch.tensor(volume.values, dtype=torch.float32),        # (T, N)
        "dates": log_ret.index.strftime("%Y-%m-%d").tolist(),
        "tickers": TICKERS,
    }

    torch.save(tensors, OUT / "tensors.pt")
    print(f"Saved tensors.pt: {T} days × {N_ASSETS} assets")
    print(f"Keys: {[k for k in tensors if k not in ('dates', 'tickers')]}")
    print("Done.")


if __name__ == "__main__":
    main()
