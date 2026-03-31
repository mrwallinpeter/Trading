"""
Multi-timeframe MACD backtest using Polygon.io
Requirements:
    pip install pandas numpy matplotlib ta requests

Before running:
    1. Get a free API key from https://polygon.io
    2. Put your tickers in 'tickers.txt', one per line
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.trend import MACD
from datetime import datetime, timedelta

print("Starting backtest...")

# ===== USER SETTINGS =====
API_KEY = "fsJD6qUtx7RlzTtPJMHp1B2Hvw35JIJ4"
START_DATE = "2024-01-01"
END_DATE = "2024-03-01"
TIMEFRAMES = ["5", "15", "60", "240"]  # minutes
TICKER_FILE = "tickers.txt"
initial_cash = 10000

# ===== HELPER: Download OHLC data from polygon.io =====
def get_polygon_bars(symbol, timespan, start, end):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{timespan}/minute/{start}/{end}?adjusted=true&sort=asc&apiKey={API_KEY}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json().get("results", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df = df.rename(columns={"t": "date", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df[["date", "open", "high", "low", "close", "volume"]]

# ===== MACD computation =====
def macd(df, fast=12, slow=26, signal=9):
    macd_calc = MACD(close=df["close"], window_slow=slow, window_fast=fast, window_sign=signal)
    df["macd_line"] = macd_calc.macd()
    df["signal_line"] = macd_calc.macd_signal()
    df["hist"] = macd_calc.macd_diff()
    return df

# ===== Backtest core logic =====
def backtest(df5, df15, df1h, df4h):
    """Simple 1-position backtest"""
    position = 0
    cash = initial_cash
    equity_curve = []

    for i in range(1, len(df5)):
        t = df5.iloc[i]["date"]
        price = df5.iloc[i]["close"]

        # Align higher TF data to current time
        h1_row = df1h[df1h["date"] <= t].tail(1)
        h4_row = df4h[df4h["date"] <= t].tail(1)
        if h1_row.empty or h4_row.empty:
            continue

        # Conditions
        hist_prev, hist_now = df5.iloc[i-1]["hist"], df5.iloc[i]["hist"]
        buy_signal = (hist_prev < 0 and hist_now > 0 and
                      h1_row["macd_line"].values[0] > 0 and
                      h4_row["macd_line"].values[0] > 0)
        sell_signal = (hist_prev > 0 and hist_now < 0)

        # Buy
        if buy_signal and position == 0:
            position = cash / price
            cash = 0
        # Sell
        elif sell_signal and position > 0:
            cash = position * price
            position = 0

        total_equity = cash + position * price
        equity_curve.append({"date": t, "equity": total_equity})

    equity_df = pd.DataFrame(equity_curve)
    return equity_df

# ===== MAIN LOOP =====
with open(TICKER_FILE) as f:
    tickers = [t.strip().upper() for t in f if t.strip()]

for symbol in tickers:
    print(f"\n=== Backtesting {symbol} ===")
    try:
        # Download each timeframe
        data = {tf: get_polygon_bars(symbol, tf, START_DATE, END_DATE) for tf in TIMEFRAMES}
        # Add MACD
        for tf in data:
            if not data[tf].empty:
                data[tf] = macd(data[tf])

        if any(df.empty for df in data.values()):
            print("  Missing data for some timeframes, skipping.")
            continue

        # Run backtest (using 5-min chart as trading base)
        result = backtest(data["5"], data["15"], data["60"], data["240"])

        # Plot
        plt.figure(figsize=(10,4))
        plt.plot(result["date"], result["equity"])
        plt.title(f"{symbol} — Equity Curve")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.show()

        print(f"  Final equity: {result['equity'].iloc[-1]:.2f}")

    except Exception as e:
        print(f"Error for {symbol}: {e}")
