import os
import numpy as np
import pandas as pd

# ==================================================================================================
# SETTINGS
# ==================================================================================================

CSV_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "MACD_PRICEMOVE_RSI_OVERLAP_TEST"
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

    return (peak - equity).max()


def summarize(trades):
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
# LOAD 1-MINUTE DATA
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
# BUILD 15-MINUTE DATA
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
# BASE UNIVERSE
# MACD + PriorMove only
# ==================================================================================================

base = df15[
    df15["BullCross"] &
    df15["PriorMovePass"]
].copy()

base = base[
    (base.index.year >= START_YEAR) &
    (base.index.year <= END_YEAR)
].copy()

base = base.dropna(
    subset=["RSI", "PriceMoveATR"]
)


# ==================================================================================================
# MUTUALLY EXCLUSIVE GROUPS
# ==================================================================================================

conditions = [
    base["PriceMovePass"] & ~base["RSIPass"],
    ~base["PriceMovePass"] & base["RSIPass"],
    base["PriceMovePass"] & base["RSIPass"],
    ~base["PriceMovePass"] & ~base["RSIPass"]
]

labels = [
    "PriceMove ONLY",
    "RSI40-50 ONLY",
    "BOTH",
    "NEITHER"
]

base["Group"] = np.select(
    conditions,
    labels,
    default="UNKNOWN"
)


section("RAW SIGNAL GROUPS")

print(base["Group"].value_counts().reindex(labels).fillna(0).astype(int))

print()
print(f"Total MACD + PriorMove signals: {len(base):,}")


# ==================================================================================================
# 1-MINUTE ARRAYS
# ==================================================================================================

times = df1.index.values
opens = df1["Open"].to_numpy()


# ==================================================================================================
# BACKTEST
# Each group is tested independently with one position at a time.
# ==================================================================================================

def run_strategy(signal_df, group_name):

    trades = []
    blocked = 0
    position_exit_time = None

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

        exit_time = df1.index[exit_pos]

        if exit_time > target_exit_time + pd.Timedelta(minutes=2):
            continue

        exit_price = opens[exit_pos]

        pnl = exit_price - entry_price

        trades.append({
            "Group": group_name,
            "SignalTime": signal_time,
            "EntryTime": entry_time,
            "ExitTime": exit_time,
            "EntryPrice": entry_price,
            "ExitPrice": exit_price,
            "PnL": pnl,
            "Year": signal_time.year,
            "RSI": s["RSI"],
            "PriceMoveATR": s["PriceMoveATR"],
            "PriorMoveATR": s["PriorMoveATR"]
        })

        position_exit_time = exit_time

    return pd.DataFrame(trades), blocked


# ==================================================================================================
# RUN GROUPS
# ==================================================================================================

all_trades = {}
blocked_counts = {}

for group in labels:

    signals = base[
        base["Group"] == group
    ]

    trades, blocked = run_strategy(
        signals,
        group
    )

    all_trades[group] = trades
    blocked_counts[group] = blocked


# ==================================================================================================
# OVERALL GROUP COMPARISON
# ==================================================================================================

rows = []

for group in labels:

    s = summarize(
        all_trades[group]
    )

    row = {
        "Group": group,
        "RawSignals": int((base["Group"] == group).sum()),
        "Blocked": blocked_counts[group]
    }

    row.update(s)

    rows.append(row)

overall = pd.DataFrame(rows)


pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 260)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")


section("OVERALL GROUP COMPARISON")

print(
    overall[
        [
            "Group",
            "RawSignals",
            "N",
            "AvgPnL",
            "NetPnL",
            "PF",
            "WinPct",
            "MaxDD",
            "Blocked"
        ]
    ].to_string(index=False)
)


# ==================================================================================================
# YEAR-BY-YEAR
# ==================================================================================================

year_rows = []

for group in labels:

    trades = all_trades[group]

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        gy = trades[
            trades["Year"] == year
        ]

        row = {
            "Group": group,
            "Year": year
        }

        row.update(
            summarize(gy)
        )

        year_rows.append(row)

year_summary = pd.DataFrame(
    year_rows
)


section("YEAR-BY-YEAR")

print(
    year_summary[
        [
            "Group",
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
# STABILITY
# ==================================================================================================

stability_rows = []

for group in labels:

    trades = all_trades[group]

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
            profit_factor(gy["PnL"])
        )

    s = summarize(trades)

    stability_rows.append({
        "Group": group,
        "N": s["N"],
        "AvgPnL": s["AvgPnL"],
        "NetPnL": s["NetPnL"],
        "PF": s["PF"],
        "WinPct": s["WinPct"],
        "MaxDD": s["MaxDD"],
        "PositiveYears": sum(x > 0 for x in yearly_avg),
        "NegativeYears": sum(x < 0 for x in yearly_avg),
        "WorstYearAvg": min(yearly_avg) if yearly_avg else np.nan,
        "WorstYearPF": min(yearly_pf) if yearly_pf else np.nan
    })

stability = pd.DataFrame(
    stability_rows
)


section("STABILITY SUMMARY")

print(
    stability.sort_values(
        [
            "PositiveYears",
            "WorstYearPF",
            "PF"
        ],
        ascending=[
            False,
            False,
            False
        ]
    ).to_string(index=False)
)


# ==================================================================================================
# RECENT PINE WINDOW
# ==================================================================================================

pine_rows = []

for group in labels:

    trades = all_trades[group]

    g = trades[
        (trades["SignalTime"] >= PINE_START) &
        (trades["SignalTime"] <= PINE_END)
    ]

    row = {
        "Group": group
    }

    row.update(
        summarize(g)
    )

    pine_rows.append(row)

pine_summary = pd.DataFrame(
    pine_rows
)


section("PINE WINDOW — MAY 1 TO SEP 18 2026")

print(
    pine_summary[
        [
            "Group",
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
# SAVE
# ==================================================================================================

overall.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "overall_groups.csv"
    ),
    index=False
)

year_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "year_groups.csv"
    ),
    index=False
)

stability.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "stability_groups.csv"
    ),
    index=False
)

pine_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "pine_window_groups.csv"
    ),
    index=False
)

for group in labels:

    safe = (
        group
        .replace(" ", "_")
        .replace("-", "_")
    )

    all_trades[group].to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"trades_{safe}.csv"
        ),
        index=False
    )


# ==================================================================================================
# DONE
# ==================================================================================================

section("DONE")

print()
print("Groups:")
print("  PriceMove ONLY = PriceMove passes, RSI does not")
print("  RSI40-50 ONLY   = RSI 40-50 passes, PriceMove does not")
print("  BOTH            = both filters pass")
print("  NEITHER         = neither filter passes")

print()
print("Every group is:")
print("  bullish MACD zero-cross")
print("  PriorMove -2 to +2")
print("  one position at a time")
print("  210-minute hold")

print()
print("Output directory:")
print(OUTPUT_DIR)