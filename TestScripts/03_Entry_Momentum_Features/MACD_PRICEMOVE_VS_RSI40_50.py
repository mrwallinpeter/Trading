import os
import numpy as np
import pandas as pd

# ==================================================================================================
# SETTINGS
# ==================================================================================================

CSV_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "MACD_PRICEMOVE_VS_RSI40_50"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14

# PriceMove architecture
PRICE_MOVE_MIN = 0.35
PRICE_MOVE_MAX = 0.90

# PriorMove shared by both
PRIOR_LOOKBACK = 20
PRIOR_MOVE_MIN = -2.0
PRIOR_MOVE_MAX = 2.0

# RSI architecture
RSI_LENGTH = 14
RSI_MIN = 40.0
RSI_MAX = 50.0

HOLD_MINUTES = 210

START_YEAR = 2022
END_YEAR = 2026

PINE_START = pd.Timestamp("2026-05-01 00:00:00")
PINE_END = pd.Timestamp("2026-09-18 23:59:00")


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

    return 100.0 - (100.0 / (1.0 + rs))


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
    drawdown = peak - equity

    return drawdown.max()


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
# ATR
# ==================================================================================================

df15["ATR14"] = wilder_atr(
    df15,
    ATR_LENGTH
)


# ==================================================================================================
# PRICEMOVE
# ==================================================================================================

df15["PriceMoveATR"] = (
    (df15["Close"] - df15["Close"].shift(1)) /
    df15["ATR14"]
)

df15["PriceMovePass"] = (
    (df15["PriceMoveATR"] >= PRICE_MOVE_MIN) &
    (df15["PriceMoveATR"] <= PRICE_MOVE_MAX)
)


# ==================================================================================================
# PRIOR MOVE
# ==================================================================================================

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

df15["RSIPass"] = (
    (df15["RSI"] >= RSI_MIN) &
    (df15["RSI"] < RSI_MAX)
)


# ==================================================================================================
# TWO ARCHITECTURES
# ==================================================================================================

df15["Signal_PriceMove"] = (
    df15["BullCross"] &
    df15["PriceMovePass"] &
    df15["PriorMovePass"]
)

df15["Signal_RSI"] = (
    df15["BullCross"] &
    df15["RSIPass"] &
    df15["PriorMovePass"]
)


# ==================================================================================================
# SIGNAL TABLES
# ==================================================================================================

price_signals = df15[
    df15["Signal_PriceMove"]
].copy()

rsi_signals = df15[
    df15["Signal_RSI"]
].copy()

price_signals = price_signals[
    (price_signals.index.year >= START_YEAR) &
    (price_signals.index.year <= END_YEAR)
].copy()

rsi_signals = rsi_signals[
    (rsi_signals.index.year >= START_YEAR) &
    (rsi_signals.index.year <= END_YEAR)
].copy()

section("SIGNAL COUNTS")

print(f"PriceMove architecture raw signals: {len(price_signals):,}")
print(f"RSI 40-50 architecture raw signals: {len(rsi_signals):,}")


# ==================================================================================================
# 1-MINUTE ARRAYS
# ==================================================================================================

times = df1.index.values
opens = df1["Open"].to_numpy()


# ==================================================================================================
# BACKTEST FUNCTION
# ==================================================================================================

def run_strategy(signal_df, architecture_name):
    trades = []

    position_exit_time = None
    blocked = 0

    for signal_time, s in signal_df.iterrows():

        if position_exit_time is not None and signal_time < position_exit_time:
            blocked += 1
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

        pnl = exit_price - entry_price

        trades.append({
            "Architecture": architecture_name,
            "SignalTime": signal_time,
            "EntryTime": entry_time,
            "ExitTime": actual_exit_time,
            "EntryPrice": entry_price,
            "ExitPrice": exit_price,
            "PnL": pnl,
            "Year": signal_time.year,
            "PriceMoveATR": s["PriceMoveATR"],
            "PriorMoveATR": s["PriorMoveATR"],
            "RSI": s["RSI"]
        })

        position_exit_time = actual_exit_time

    return pd.DataFrame(trades), blocked


# ==================================================================================================
# RUN BOTH
# ==================================================================================================

section("RUNNING BOTH ARCHITECTURES")

price_trades, price_blocked = run_strategy(
    price_signals,
    "PriceMove"
)

rsi_trades, rsi_blocked = run_strategy(
    rsi_signals,
    "RSI40-50"
)

print(f"PriceMove executed trades: {len(price_trades):,}")
print(f"PriceMove blocked signals: {price_blocked:,}")

print()

print(f"RSI executed trades:       {len(rsi_trades):,}")
print(f"RSI blocked signals:       {rsi_blocked:,}")


# ==================================================================================================
# OVERALL COMPARISON
# ==================================================================================================

section("OVERALL HEAD-TO-HEAD")

rows = []

for name, trades in [
    ("PriceMove + PriorMove", price_trades),
    ("RSI40-50 + PriorMove", rsi_trades)
]:
    row = {
        "Architecture": name
    }

    row.update(
        summarize_trades(trades)
    )

    rows.append(row)

overall = pd.DataFrame(rows)

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 250)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")

print(
    overall[
        [
            "Architecture",
            "N",
            "AvgPnL",
            "NetPnL",
            "PF",
            "WinPct",
            "MaxDD"
        ]
    ].to_string(index=False)
)


# ==================================================================================================
# YEAR-BY-YEAR COMPARISON
# ==================================================================================================

section("YEAR-BY-YEAR HEAD-TO-HEAD")

year_rows = []

for name, trades in [
    ("PriceMove + PriorMove", price_trades),
    ("RSI40-50 + PriorMove", rsi_trades)
]:

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        g = trades[
            trades["Year"] == year
        ]

        row = {
            "Architecture": name,
            "Year": year
        }

        row.update(
            summarize_trades(g)
        )

        year_rows.append(row)

year_summary = pd.DataFrame(
    year_rows
)

print(
    year_summary[
        [
            "Architecture",
            "Year",
            "N",
            "AvgPnL",
            "NetPnL",
            "PF",
            "WinPct",
            "MaxDD"
        ]
    ].to_string(index=False)
)


# ==================================================================================================
# STABILITY SUMMARY
# ==================================================================================================

section("STABILITY SUMMARY")

stability_rows = []

for name, trades in [
    ("PriceMove + PriorMove", price_trades),
    ("RSI40-50 + PriorMove", rsi_trades)
]:

    yearly_avg = []
    yearly_pf = []

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        gy = trades[
            trades["Year"] == year
        ]

        if len(gy) == 0:
            continue

        yearly_avg.append(
            gy["PnL"].mean()
        )

        yearly_pf.append(
            profit_factor(
                gy["PnL"]
            )
        )

    s = summarize_trades(
        trades
    )

    stability_rows.append({
        "Architecture": name,
        "N": s["N"],
        "AvgPnL": s["AvgPnL"],
        "NetPnL": s["NetPnL"],
        "PF": s["PF"],
        "WinPct": s["WinPct"],
        "MaxDD": s["MaxDD"],
        "PositiveYears": sum(x > 0 for x in yearly_avg),
        "NegativeYears": sum(x < 0 for x in yearly_avg),
        "WorstYearAvg": min(yearly_avg),
        "WorstYearPF": min(yearly_pf)
    })

stability = pd.DataFrame(
    stability_rows
)

print(
    stability.to_string(
        index=False
    )
)


# ==================================================================================================
# PINE WINDOW
# ==================================================================================================

section("PINE WINDOW — MAY 1 TO SEP 18 2026")

pine_rows = []

for name, trades in [
    ("PriceMove + PriorMove", price_trades),
    ("RSI40-50 + PriorMove", rsi_trades)
]:

    g = trades[
        (trades["SignalTime"] >= PINE_START) &
        (trades["SignalTime"] <= PINE_END)
    ]

    row = {
        "Architecture": name
    }

    row.update(
        summarize_trades(g)
    )

    pine_rows.append(row)

pine_summary = pd.DataFrame(
    pine_rows
)

print(
    pine_summary[
        [
            "Architecture",
            "N",
            "AvgPnL",
            "NetPnL",
            "PF",
            "WinPct",
            "MaxDD"
        ]
    ].to_string(index=False)
)


# ==================================================================================================
# SIGNAL OVERLAP
# ==================================================================================================

section("SIGNAL OVERLAP")

price_times = set(
    price_signals.index
)

rsi_times = set(
    rsi_signals.index
)

overlap = (
    price_times &
    rsi_times
)

print(
    f"PriceMove raw signals: {len(price_times):,}"
)

print(
    f"RSI raw signals:       {len(rsi_times):,}"
)

print(
    f"Signals in both:       {len(overlap):,}"
)

if len(price_times) > 0:
    print(
        f"Overlap as % of PriceMove signals: "
        f"{len(overlap) / len(price_times) * 100:.2f}%"
    )

if len(rsi_times) > 0:
    print(
        f"Overlap as % of RSI signals:       "
        f"{len(overlap) / len(rsi_times) * 100:.2f}%"
    )


# ==================================================================================================
# SAVE
# ==================================================================================================

overall.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "overall_comparison.csv"
    ),
    index=False
)

year_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "year_comparison.csv"
    ),
    index=False
)

stability.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "stability_comparison.csv"
    ),
    index=False
)

pine_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "pine_window_comparison.csv"
    ),
    index=False
)

price_trades.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "trades_pricemove.csv"
    ),
    index=False
)

rsi_trades.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "trades_rsi40_50.csv"
    ),
    index=False
)


# ==================================================================================================
# DONE
# ==================================================================================================

section("DONE")

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("HEAD-TO-HEAD:")
print("A = MACD + PriceMove 0.35-0.90 + PriorMove")
print("B = MACD + RSI 40-50 + PriorMove")
print()
print("Same one-position-at-a-time execution and same 210-minute hold.")