import numpy as np
import pandas as pd


# ==================================================================================================
# SETTINGS
# ==================================================================================================

CSV_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14
RSI_LENGTH = 14

# Two filters
PRICEMOVE_MIN = 0.35
PRICEMOVE_MAX = 0.90

RSI_MIN = 40.0
RSI_MAX = 50.0

# Bollinger Bands
BB_LENGTH = 20
BB_MULT = 2.0
BB_WIDTH_SMA_LENGTH = 200

HOLD_MINUTES = 210

START_YEAR = 2022
END_YEAR = 2026


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

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1.0 / length,
        adjust=False
    ).mean()


def wilder_rsi(close, length):
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


def profit_factor(series):
    gross_profit = series[series > 0].sum()
    gross_loss = -series[series < 0].sum()

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else np.nan

    return gross_profit / gross_loss


def max_drawdown(series):
    if len(series) == 0:
        return np.nan

    equity = series.cumsum()
    peak = equity.cummax()

    return (peak - equity).max()


def stats(df):
    if len(df) == 0:
        return {
            "N": 0,
            "Winners": 0,
            "Losers": 0,
            "WinPct": np.nan,
            "AvgPnL": np.nan,
            "PF": np.nan,
            "NetPnL": np.nan,
            "MaxDD": np.nan
        }

    pnl = df["PnL"]

    winners = (pnl > 0).sum()
    losers = (pnl < 0).sum()

    return {
        "N": len(df),
        "Winners": winners,
        "Losers": losers,
        "WinPct": winners / len(df) * 100.0,
        "AvgPnL": pnl.mean(),
        "PF": profit_factor(pnl),
        "NetPnL": pnl.sum(),
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

section("CALCULATING INDICATORS")

ema_fast = df15["Close"].ewm(
    span=MACD_FAST,
    adjust=False
).mean()

ema_slow = df15["Close"].ewm(
    span=MACD_SLOW,
    adjust=False
).mean()

macd = ema_fast - ema_slow

macd_signal = macd.ewm(
    span=MACD_SIGNAL,
    adjust=False
).mean()

df15["MACDHist"] = macd - macd_signal

df15["BullCross"] = (
    (df15["MACDHist"].shift(1) <= 0) &
    (df15["MACDHist"] > 0)
)


# ==================================================================================================
# ATR + PRICE MOVE
# ==================================================================================================

df15["ATR14"] = wilder_atr(
    df15,
    ATR_LENGTH
)

df15["PriceMoveATR"] = (
    (df15["Close"] - df15["Close"].shift(1)) /
    df15["ATR14"]
)


# ==================================================================================================
# RSI
# ==================================================================================================

df15["RSI14"] = wilder_rsi(
    df15["Close"],
    RSI_LENGTH
)


# ==================================================================================================
# BOLLINGER BAND WIDTH
# ==================================================================================================

df15["BBBasis"] = df15["Close"].rolling(
    BB_LENGTH
).mean()

bb_std = df15["Close"].rolling(
    BB_LENGTH
).std(ddof=0)

df15["BBUpper"] = (
    df15["BBBasis"] +
    BB_MULT * bb_std
)

df15["BBLower"] = (
    df15["BBBasis"] -
    BB_MULT * bb_std
)

# Percentage width
df15["BBWidth"] = (
    (df15["BBUpper"] - df15["BBLower"]) /
    df15["BBBasis"]
) * 100.0

df15["BBWidthSMA200"] = df15["BBWidth"].rolling(
    BB_WIDTH_SMA_LENGTH
).mean()

df15["BBState"] = np.where(
    df15["BBWidth"] >= df15["BBWidthSMA200"],
    "ABOVE SMA200",
    "BELOW SMA200"
)


# ==================================================================================================
# SIGNALS
# ==================================================================================================

signals = df15[
    df15["BullCross"] &
    (df15["PriceMoveATR"] >= PRICEMOVE_MIN) &
    (df15["PriceMoveATR"] <= PRICEMOVE_MAX) &
    (df15["RSI14"] >= RSI_MIN) &
    (df15["RSI14"] <= RSI_MAX)
].copy()

signals = signals[
    (signals.index.year >= START_YEAR) &
    (signals.index.year <= END_YEAR)
]

signals = signals.dropna(
    subset=[
        "PriceMoveATR",
        "RSI14",
        "BBWidth",
        "BBWidthSMA200"
    ]
)

section("QUALIFYING SIGNALS")

print(f"Signals: {len(signals):,}")

print()
print(
    f"PriceMoveATR: {PRICEMOVE_MIN:.2f} to {PRICEMOVE_MAX:.2f}"
)

print(
    f"RSI14:        {RSI_MIN:.1f} to {RSI_MAX:.1f}"
)

print()
print("BB definition:")
print(f"Length:       {BB_LENGTH}")
print(f"Multiplier:   {BB_MULT}")
print(f"Width SMA:    {BB_WIDTH_SMA_LENGTH}")


# ==================================================================================================
# BUILD TRADES
# ==================================================================================================

section("BUILDING 210-MINUTE TRADES")

times = df1.index.values
opens = df1["Open"].to_numpy()

records = []

for signal_time, row in signals.iterrows():

    # First 1-minute open strictly after completed 15m signal bar
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

    exit_time = df1.index[exit_pos]

    if exit_time > target_exit_time + pd.Timedelta(minutes=2):
        continue

    exit_price = opens[exit_pos]

    pnl = exit_price - entry_price

    records.append({
        "SignalTime": signal_time,
        "EntryTime": entry_time,
        "ExitTime": exit_time,
        "Year": signal_time.year,

        "PriceMoveATR": row["PriceMoveATR"],
        "RSI14": row["RSI14"],

        "BBWidth": row["BBWidth"],
        "BBWidthSMA200": row["BBWidthSMA200"],
        "BBWidthRatio": row["BBWidth"] / row["BBWidthSMA200"],
        "BBState": row["BBState"],

        "EntryPrice": entry_price,
        "ExitPrice": exit_price,
        "PnL": pnl,

        "Result": (
            "WIN"
            if pnl > 0
            else "LOSS"
            if pnl < 0
            else "FLAT"
        )
    })


trades = pd.DataFrame(records)

print(f"Valid trades: {len(trades):,}")


# ==================================================================================================
# OVERALL
# ==================================================================================================

section("OVERALL RESULTS")

overall = stats(trades)

for key, value in overall.items():

    if isinstance(value, float):
        print(f"{key:12s}: {value:.3f}")
    else:
        print(f"{key:12s}: {value}")


# ==================================================================================================
# BB WIDTH ABOVE / BELOW SMA200
# ==================================================================================================

section("BB WIDTH STATE")

rows = []

for state in [
    "BELOW SMA200",
    "ABOVE SMA200"
]:

    g = trades[
        trades["BBState"] == state
    ]

    s = stats(g)

    rows.append({
        "BBState": state,
        **s
    })


state_stats = pd.DataFrame(rows)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")

print(
    state_stats.to_string(
        index=False
    )
)


# ==================================================================================================
# WINNERS VS LOSERS
# ==================================================================================================

section("WINNER / LOSER DISTRIBUTION BY BB WIDTH STATE")

winners = trades[
    trades["PnL"] > 0
]

losers = trades[
    trades["PnL"] < 0
]

winner_rows = []

for state in [
    "BELOW SMA200",
    "ABOVE SMA200"
]:

    w = winners[
        winners["BBState"] == state
    ]

    l = losers[
        losers["BBState"] == state
    ]

    winner_pct = (
        len(w) / len(winners) * 100
        if len(winners) > 0
        else np.nan
    )

    loser_pct = (
        len(l) / len(losers) * 100
        if len(losers) > 0
        else np.nan
    )

    winner_rows.append({
        "BBState": state,

        "WinnerCount": len(w),
        "WinnerSharePct": winner_pct,

        "LoserCount": len(l),
        "LoserSharePct": loser_pct,

        "AvgWinner": (
            w["PnL"].mean()
            if len(w) > 0
            else np.nan
        ),

        "AvgLoser": (
            l["PnL"].mean()
            if len(l) > 0
            else np.nan
        )
    })


winner_loser_stats = pd.DataFrame(
    winner_rows
)

print(
    winner_loser_stats.to_string(
        index=False
    )
)


# ==================================================================================================
# BB WIDTH NUMBERS FOR WINNERS / LOSERS
# ==================================================================================================

section("BB WIDTH CHARACTERISTICS — WINNERS VS LOSERS")

comparison_rows = []

for result, g in [
    ("WIN", winners),
    ("LOSS", losers)
]:

    comparison_rows.append({
        "Result": result,
        "N": len(g),

        "AvgBBWidth": g["BBWidth"].mean(),
        "MedianBBWidth": g["BBWidth"].median(),

        "AvgBBWidthSMA200": g["BBWidthSMA200"].mean(),

        "AvgWidthRatio": g["BBWidthRatio"].mean(),
        "MedianWidthRatio": g["BBWidthRatio"].median(),

        "PctBelowSMA200": (
            (g["BBState"] == "BELOW SMA200").mean() * 100
        ),

        "PctAboveSMA200": (
            (g["BBState"] == "ABOVE SMA200").mean() * 100
        )
    })


comparison = pd.DataFrame(
    comparison_rows
)

print(
    comparison.to_string(
        index=False
    )
)


# ==================================================================================================
# YEAR-BY-YEAR
# ==================================================================================================

section("YEAR-BY-YEAR — BB WIDTH STATE")

year_rows = []

for year in sorted(
    trades["Year"].unique()
):

    for state in [
        "BELOW SMA200",
        "ABOVE SMA200"
    ]:

        g = trades[
            (trades["Year"] == year) &
            (trades["BBState"] == state)
        ]

        s = stats(g)

        year_rows.append({
            "Year": year,
            "BBState": state,
            **s
        })


year_stats = pd.DataFrame(
    year_rows
)

print(
    year_stats.to_string(
        index=False
    )
)


# ==================================================================================================
# BB WIDTH RATIO BUCKETS
# Optional extra information around the SMA200 boundary
# ==================================================================================================

section("BB WIDTH / SMA200 RATIO BUCKETS")

bins = [
    -np.inf,
    0.70,
    0.85,
    1.00,
    1.15,
    1.30,
    np.inf
]

labels = [
    "<0.70",
    "0.70-0.85",
    "0.85-1.00",
    "1.00-1.15",
    "1.15-1.30",
    "1.30+"
]

trades["BBRatioBucket"] = pd.cut(
    trades["BBWidthRatio"],
    bins=bins,
    labels=labels,
    right=False
)

bucket_rows = []

for label in labels:

    g = trades[
        trades["BBRatioBucket"] == label
    ]

    s = stats(g)

    bucket_rows.append({
        "BBRatioBucket": label,
        **s
    })


bucket_stats = pd.DataFrame(
    bucket_rows
)

print(
    bucket_stats.to_string(
        index=False
    )
)


# ==================================================================================================
# SAVE RESULTS
# ==================================================================================================

output_path = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\MACD_2FILTER_BBWIDTH_RESULTS.csv"

trades.to_csv(
    output_path,
    index=False
)


# ==================================================================================================
# DONE
# ==================================================================================================

section("DONE")

print("Signal:")
print("  Bullish MACD(5,36,5) zero cross")

print()
print("Filters:")
print(
    f"  PriceMoveATR {PRICEMOVE_MIN:.2f} - {PRICEMOVE_MAX:.2f}"
)
print(
    f"  RSI14        {RSI_MIN:.1f} - {RSI_MAX:.1f}"
)

print()
print("Bollinger Bands:")
print(f"  BB({BB_LENGTH}, {BB_MULT})")
print(f"  BB width compared with SMA({BB_WIDTH_SMA_LENGTH}) of BB width")

print()
print(f"Forward horizon: {HOLD_MINUTES} minutes")

print()
print("Results CSV:")
print(output_path)