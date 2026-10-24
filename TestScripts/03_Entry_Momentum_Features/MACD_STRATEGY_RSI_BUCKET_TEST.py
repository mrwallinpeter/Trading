import os
import numpy as np
import pandas as pd

# ==================================================================================================
# SETTINGS
# ==================================================================================================

CSV_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "MACD_STRATEGY_RSI_BUCKET_TEST"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14

PRICE_MOVE_MIN = 0.35
PRICE_MOVE_MAX = 0.90

PRIOR_LOOKBACK = 20
PRIOR_MOVE_MIN = -2.0
PRIOR_MOVE_MAX = 2.0

RSI_LENGTH = 14

HOLD_MINUTES = 210

START_YEAR = 2022
END_YEAR = 2026

PINE_START = pd.Timestamp("2026-05-01 00:00:00")
PINE_END = pd.Timestamp("2026-09-18 23:59:00")

RSI_BINS = [
    -np.inf,
    30,
    40,
    50,
    60,
    70,
    np.inf
]

RSI_LABELS = [
    "<30",
    "30-40",
    "40-50",
    "50-60",
    "60-70",
    "70+"
]


# ==================================================================================================
# HELPERS
# ==================================================================================================

def section(title):
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def wilder_atr(df, length):
    prev_close = df["Close"].shift(1)

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1.0 / length,
        adjust=False
    ).mean()


def wilder_rsi(close, length=14):
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(
        alpha=1.0 / length,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1.0 / length,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    rsi = 100.0 - (
        100.0 / (1.0 + rs)
    )

    return rsi


def profit_factor(pnls):
    gross_profit = pnls[pnls > 0].sum()
    gross_loss = -pnls[pnls < 0].sum()

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else np.nan

    return gross_profit / gross_loss


def max_drawdown(pnls):
    if len(pnls) == 0:
        return np.nan

    equity = pnls.cumsum()
    peak = equity.cummax()
    dd = peak - equity

    return dd.max()


def summarize_trades(trades):
    if len(trades) == 0:
        return {
            "N": 0,
            "AvgPnL": np.nan,
            "NetPnL": np.nan,
            "PF": np.nan,
            "WinPct": np.nan,
            "MaxDD": np.nan
        }

    pnl = trades["PnL"]

    return {
        "N": len(trades),
        "AvgPnL": pnl.mean(),
        "NetPnL": pnl.sum(),
        "PF": profit_factor(pnl),
        "WinPct": (pnl > 0).mean() * 100,
        "MaxDD": max_drawdown(pnl)
    }


# ==================================================================================================
# LOAD DATA
# ==================================================================================================

section("LOADING DATA")

df1 = pd.read_csv(CSV_PATH)

possible_time_cols = [
    "DateTime",
    "Datetime",
    "datetime",
    "Timestamp",
    "timestamp",
    "Time",
    "time",
    "Date",
    "date"
]

time_col = None

for c in possible_time_cols:
    if c in df1.columns:
        time_col = c
        break

if time_col is None:
    raise ValueError("Could not find timestamp column.")

df1[time_col] = pd.to_datetime(df1[time_col])
df1 = df1.set_index(time_col).sort_index()

rename_map = {}

for c in df1.columns:
    lc = c.lower()

    if lc == "open":
        rename_map[c] = "Open"
    elif lc == "high":
        rename_map[c] = "High"
    elif lc == "low":
        rename_map[c] = "Low"
    elif lc == "close":
        rename_map[c] = "Close"
    elif lc == "volume":
        rename_map[c] = "Volume"

df1 = df1.rename(columns=rename_map)

print(f"1m rows: {len(df1):,}")
print(f"Start: {df1.index.min()}")
print(f"End:   {df1.index.max()}")


# ==================================================================================================
# BUILD 15-MINUTE BARS
# ==================================================================================================

section("BUILDING 15-MINUTE BARS")

agg = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last"
}

if "Volume" in df1.columns:
    agg["Volume"] = "sum"

df15 = df1.resample(
    "15min",
    label="right",
    closed="right"
).agg(agg)

df15 = df15.dropna(
    subset=["Open", "High", "Low", "Close"]
)

print(f"15m bars: {len(df15):,}")


# ==================================================================================================
# MACD
# ==================================================================================================

ema_fast = df15["Close"].ewm(
    span=MACD_FAST,
    adjust=False
).mean()

ema_slow = df15["Close"].ewm(
    span=MACD_SLOW,
    adjust=False
).mean()

macd = ema_fast - ema_slow

signal_line = macd.ewm(
    span=MACD_SIGNAL,
    adjust=False
).mean()

df15["MACDHist"] = macd - signal_line

df15["BullCross"] = (
    (df15["MACDHist"].shift(1) <= 0) &
    (df15["MACDHist"] > 0)
)


# ==================================================================================================
# EXISTING FILTERS
# ==================================================================================================

df15["ATR14"] = wilder_atr(
    df15,
    ATR_LENGTH
)

df15["PriceMoveATR"] = (
    (df15["Close"] - df15["Close"].shift(1)) /
    df15["ATR14"]
)

df15["PriceMovePass"] = (
    (df15["PriceMoveATR"] >= PRICE_MOVE_MIN) &
    (df15["PriceMoveATR"] <= PRICE_MOVE_MAX)
)

df15["PriorMoveATR"] = (
    (df15["Close"].shift(1) - df15["Close"].shift(PRIOR_LOOKBACK)) /
    df15["ATR14"]
)

df15["PriorMovePass"] = (
    (df15["PriorMoveATR"] >= PRIOR_MOVE_MIN) &
    (df15["PriorMoveATR"] <= PRIOR_MOVE_MAX)
)


# ==================================================================================================
# RSI
# ==================================================================================================

df15["RSI"] = wilder_rsi(
    df15["Close"],
    RSI_LENGTH
)


# ==================================================================================================
# BASE SIGNAL
# ==================================================================================================

df15["BaseSignal"] = (
    df15["BullCross"] &
    df15["PriceMovePass"] &
    df15["PriorMovePass"]
)

signals = df15[
    df15["BaseSignal"]
].copy()

signals = signals[
    (signals.index.year >= START_YEAR) &
    (signals.index.year <= END_YEAR)
].copy()

signals["Year"] = signals.index.year

signals = signals.dropna(
    subset=["RSI"]
)

section("BASE SIGNALS")

print(
    f"MACD + PriceMove + PriorMove signals: {len(signals):,}"
)


# ==================================================================================================
# 1-MINUTE ARRAYS
# ==================================================================================================

times = df1.index.values
opens = df1["Open"].to_numpy()


# ==================================================================================================
# ONE-POSITION-AT-A-TIME BACKTEST
# ==================================================================================================

def run_strategy(signal_df):

    trades = []

    position_exit_time = None
    blocked_signals = 0

    for signal_time, s in signal_df.iterrows():

        if position_exit_time is not None and signal_time < position_exit_time:
            blocked_signals += 1
            continue

        entry_pos = np.searchsorted(
            times,
            np.datetime64(signal_time),
            side="right"
        )

        if entry_pos >= len(df1):
            continue

        entry_time = df1.index[entry_pos]
        entry_price = opens[entry_pos]

        target_exit_time = (
            entry_time +
            pd.Timedelta(minutes=HOLD_MINUTES)
        )

        exit_pos = np.searchsorted(
            times,
            np.datetime64(target_exit_time),
            side="left"
        )

        if exit_pos >= len(df1):
            continue

        actual_exit_time = df1.index[exit_pos]

        if actual_exit_time > target_exit_time + pd.Timedelta(minutes=2):
            continue

        exit_price = opens[exit_pos]

        pnl = (
            exit_price -
            entry_price
        )

        trades.append({
            "SignalTime": signal_time,
            "EntryTime": entry_time,
            "ExitTime": actual_exit_time,
            "EntryPrice": entry_price,
            "ExitPrice": exit_price,
            "PnL": pnl,
            "Year": signal_time.year,
            "RSI": s["RSI"],
            "PriceMoveATR": s["PriceMoveATR"],
            "PriorMoveATR": s["PriorMoveATR"]
        })

        position_exit_time = actual_exit_time

    return pd.DataFrame(trades), blocked_signals


# ==================================================================================================
# BASE STRATEGY
# ==================================================================================================

section("RUNNING BASE STRATEGY")

trades, blocked = run_strategy(
    signals
)

print(f"Executed trades: {len(trades):,}")
print(f"Blocked signals: {blocked:,}")

base_summary = summarize_trades(
    trades
)

print()
print(
    f"Base PF:      {base_summary['PF']:.3f}"
)

print(
    f"Base AvgPnL:  {base_summary['AvgPnL']:.3f}"
)

print(
    f"Base NetPnL:  {base_summary['NetPnL']:.3f}"
)


# ==================================================================================================
# RSI BUCKETS
# ==================================================================================================

trades["RSIBucket"] = pd.cut(
    trades["RSI"],
    bins=RSI_BINS,
    labels=RSI_LABELS,
    right=False
)

bucket_rows = []

for bucket in RSI_LABELS:

    g = trades[
        trades["RSIBucket"].astype(str) == bucket
    ]

    row = {
        "RSIBucket": bucket
    }

    row.update(
        summarize_trades(g)
    )

    bucket_rows.append(row)

bucket_summary = pd.DataFrame(
    bucket_rows
)


# ==================================================================================================
# YEAR STABILITY BY RSI BUCKET
# ==================================================================================================

year_rows = []

for bucket in RSI_LABELS:

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        g = trades[
            (trades["RSIBucket"].astype(str) == bucket) &
            (trades["Year"] == year)
        ]

        row = {
            "RSIBucket": bucket,
            "Year": year
        }

        row.update(
            summarize_trades(g)
        )

        year_rows.append(row)

year_summary = pd.DataFrame(
    year_rows
)


# ==================================================================================================
# RSI CUMULATIVE TESTS
# ==================================================================================================

tests = {
    "NO RSI FILTER": pd.Series(True, index=trades.index),

    "RSI >= 40":
        trades["RSI"] >= 40,

    "RSI >= 45":
        trades["RSI"] >= 45,

    "RSI >= 50":
        trades["RSI"] >= 50,

    "RSI 40-60":
        (
            (trades["RSI"] >= 40) &
            (trades["RSI"] < 60)
        ),

    "RSI 45-65":
        (
            (trades["RSI"] >= 45) &
            (trades["RSI"] < 65)
        ),

    "RSI 50-70":
        (
            (trades["RSI"] >= 50) &
            (trades["RSI"] < 70)
        ),

    "RSI >= 60":
        trades["RSI"] >= 60
}

filter_rows = []

for name, mask in tests.items():

    g = trades[
        mask
    ]

    row = {
        "Test": name
    }

    row.update(
        summarize_trades(g)
    )

    filter_rows.append(row)

filter_summary = pd.DataFrame(
    filter_rows
)


# ==================================================================================================
# PINE WINDOW
# ==================================================================================================

pine_rows = []

for bucket in RSI_LABELS:

    g = trades[
        (trades["RSIBucket"].astype(str) == bucket) &
        (trades["SignalTime"] >= PINE_START) &
        (trades["SignalTime"] <= PINE_END)
    ]

    row = {
        "RSIBucket": bucket
    }

    row.update(
        summarize_trades(g)
    )

    pine_rows.append(row)

pine_bucket_summary = pd.DataFrame(
    pine_rows
)


# ==================================================================================================
# PRINT
# ==================================================================================================

pd.set_option(
    "display.max_columns",
    100
)

pd.set_option(
    "display.width",
    260
)

pd.set_option(
    "display.float_format",
    lambda x: f"{x:.3f}"
)


section("RSI BUCKET SUMMARY")

print(
    bucket_summary[
        [
            "RSIBucket",
            "N",
            "AvgPnL",
            "NetPnL",
            "PF",
            "WinPct",
            "MaxDD"
        ]
    ].to_string(
        index=False
    )
)


section("RSI FILTER COMPARISON")

print(
    filter_summary[
        [
            "Test",
            "N",
            "AvgPnL",
            "NetPnL",
            "PF",
            "WinPct",
            "MaxDD"
        ]
    ].to_string(
        index=False
    )
)


section("YEAR STABILITY BY RSI BUCKET")

print(
    year_summary[
        [
            "RSIBucket",
            "Year",
            "N",
            "AvgPnL",
            "NetPnL",
            "PF",
            "WinPct",
            "MaxDD"
        ]
    ].to_string(
        index=False
    )
)


section("PINE WINDOW — MAY 1 TO SEP 18 2026")

print(
    pine_bucket_summary[
        [
            "RSIBucket",
            "N",
            "AvgPnL",
            "NetPnL",
            "PF",
            "WinPct",
            "MaxDD"
        ]
    ].to_string(
        index=False
    )
)


# ==================================================================================================
# SAVE
# ==================================================================================================

trades.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "raw_trades.csv"
    ),
    index=False
)

bucket_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rsi_bucket_summary.csv"
    ),
    index=False
)

filter_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rsi_filter_summary.csv"
    ),
    index=False
)

year_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rsi_year_summary.csv"
    ),
    index=False
)

pine_bucket_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rsi_pine_window.csv"
    ),
    index=False
)


section("DONE")

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Main question:")
print("Does RSI identify a useful momentum regime for the exact strategy?")