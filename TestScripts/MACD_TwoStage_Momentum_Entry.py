import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ONE_MIN_FILE = Path(
    r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\TestScripts\MACD_TwoStage_Momentum"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14

START_HOUR = 10
END_HOUR = 18

MOVE_MIN = 0.35
MOVE_MAX = 0.90

HOLD_MINUTES = 240


# ============================================================
# LOAD 1-MIN DATA
# ============================================================

print()
print("=" * 140)
print("LOADING 1-MIN DATA")
print("=" * 140)

df1 = pd.read_csv(
    ONE_MIN_FILE
)

df1["DateTime"] = pd.to_datetime(
    df1["DateTime"]
)

df1 = (
    df1
    .sort_values("DateTime")
    .drop_duplicates("DateTime")
    .set_index("DateTime")
)

for col in [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]:
    df1[col] = pd.to_numeric(
        df1[col],
        errors="coerce"
    )

df1 = df1.dropna(
    subset=[
        "Open",
        "High",
        "Low",
        "Close"
    ]
)

print("Rows:", len(df1))
print("Start:", df1.index.min())
print("End:  ", df1.index.max())


# ============================================================
# BUILD 15-MIN DATA
# ============================================================

df15 = df1.resample(
    "15min",
    label="right",
    closed="right"
).agg({
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum"
})

df15 = df15.dropna(
    subset=[
        "Open",
        "High",
        "Low",
        "Close"
    ]
)


# ============================================================
# MACD
# ============================================================

df15["FastEMA"] = (
    df15["Close"]
    .ewm(
        span=MACD_FAST,
        adjust=False
    )
    .mean()
)

df15["SlowEMA"] = (
    df15["Close"]
    .ewm(
        span=MACD_SLOW,
        adjust=False
    )
    .mean()
)

df15["MACD"] = (
    df15["FastEMA"]
    -
    df15["SlowEMA"]
)

df15["Signal"] = (
    df15["MACD"]
    .ewm(
        span=MACD_SIGNAL,
        adjust=False
    )
    .mean()
)

df15["Hist"] = (
    df15["MACD"]
    -
    df15["Signal"]
)

df15["BullCross"] = (
    (df15["Hist"].shift(1) <= 0)
    &
    (df15["Hist"] > 0)
)


# ============================================================
# ATR
# ============================================================

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


df15["ATR"] = wilder_atr(
    df15,
    ATR_LENGTH
)


# ============================================================
# TIME FILTER
# ============================================================

df15["TimeValid"] = (
    (df15.index.hour >= START_HOUR)
    &
    (df15.index.hour < END_HOUR)
)


# ============================================================
# STATS
# ============================================================

def profit_factor(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    gp = pnl[
        pnl > 0
    ].sum()

    gl = -pnl[
        pnl < 0
    ].sum()

    if gl == 0:

        if gp > 0:
            return np.inf

        return np.nan

    return gp / gl


def sqn(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    if len(pnl) < 2:
        return np.nan

    std = np.std(
        pnl,
        ddof=1
    )

    if std == 0:
        return np.nan

    return (
        np.sqrt(len(pnl))
        *
        np.mean(pnl)
        /
        std
    )


def max_drawdown(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    if len(pnl) == 0:
        return np.nan

    equity = np.cumsum(
        pnl
    )

    equity = np.concatenate(
        [
            [0.0],
            equity
        ]
    )

    running_max = np.maximum.accumulate(
        equity
    )

    dd = (
        equity
        -
        running_max
    )

    return abs(
        dd.min()
    )


def calculate_stats(d):

    pnl = (
        d["PnL"]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if len(pnl) == 0:

        return {
            "N": 0,
            "WinRatePct": np.nan,
            "NetPoints": np.nan,
            "AveragePnL": np.nan,
            "MedianPnL": np.nan,
            "ProfitFactor": np.nan,
            "SQN": np.nan,
            "MaxDrawdownPoints": np.nan
        }

    return {

        "N":
            len(pnl),

        "WinRatePct":
            np.mean(
                pnl > 0
            ) * 100,

        "NetPoints":
            pnl.sum(),

        "AveragePnL":
            pnl.mean(),

        "MedianPnL":
            np.median(
                pnl
            ),

        "ProfitFactor":
            profit_factor(
                pnl
            ),

        "SQN":
            sqn(
                pnl
            ),

        "MaxDrawdownPoints":
            max_drawdown(
                pnl
            )
    }


# ============================================================
# BUILD RAW MACD CROSS LIST
# ============================================================

signals = df15[
    df15["BullCross"]
    &
    df15["TimeValid"]
].copy()


print()
print("=" * 140)
print("BUILDING TWO-STAGE ENTRIES")
print("=" * 140)

print("Raw bullish MACD crosses:", len(signals))


rows = []


# ============================================================
# PROCESS EVERY MACD CROSS
# ============================================================

for signal_time in signals.index:

    pos = df15.index.get_loc(
        signal_time
    )

    if pos < 1:
        continue

    if pos + 1 >= len(df15):
        continue


    # ========================================================
    # ATR FROM ORIGINAL CROSS BAR
    # ========================================================

    atr_cross = float(
        df15.iloc[pos]["ATR"]
    )

    if (
        not np.isfinite(atr_cross)
        or
        atr_cross <= 0
    ):
        continue


    # ========================================================
    # FIRST CANDLE STRENGTH
    #
    # Cross candle:
    #
    # Close[t] - Close[t-1]
    # ----------------------
    # ATR on cross bar
    # ========================================================

    close_prev = float(
        df15.iloc[pos - 1]["Close"]
    )

    close_cross = float(
        df15.iloc[pos]["Close"]
    )

    move1 = (
        close_cross
        -
        close_prev
    ) / atr_cross


    first_valid = (
        MOVE_MIN
        <=
        move1
        <=
        MOVE_MAX
    )


    # ========================================================
    # SECOND CANDLE STRENGTH
    #
    # Only matters if candle 1 did NOT qualify.
    #
    # Close[t+1] - Close[t]
    # ----------------------
    # ATR from original cross
    # ========================================================

    close_second = float(
        df15.iloc[pos + 1]["Close"]
    )

    move2 = (
        close_second
        -
        close_cross
    ) / atr_cross


    second_valid = (
        MOVE_MIN
        <=
        move2
        <=
        MOVE_MAX
    )


    # ========================================================
    # CLASSIFY ENTRY TYPE
    # ========================================================

    if first_valid:

        entry_type = "ENTER_FIRST"
        entry_signal_time = signal_time

    elif second_valid:

        entry_type = "ENTER_SECOND"
        entry_signal_time = df15.index[
            pos + 1
        ]

    else:

        entry_type = "NO_TRADE"
        entry_signal_time = pd.NaT


    # ========================================================
    # STORE NO-TRADE CASE
    # ========================================================

    if entry_type == "NO_TRADE":

        rows.append({

            "SignalTime":
                signal_time,

            "Year":
                signal_time.year,

            "EntryType":
                entry_type,

            "Move1":
                move1,

            "Move2":
                move2,

            "ATR_Cross":
                atr_cross,

            "EntryTime":
                pd.NaT,

            "ExitTime":
                pd.NaT,

            "EntryPrice":
                np.nan,

            "ExitPrice":
                np.nan,

            "PnL":
                np.nan
        })

        continue


    # ========================================================
    # ACTUAL ENTRY
    #
    # first 1-minute open AFTER the confirming 15m candle
    # ========================================================

    entry_pos = df1.index.searchsorted(
        entry_signal_time,
        side="right"
    )

    if entry_pos >= len(df1):
        continue

    entry_time = df1.index[
        entry_pos
    ]

    entry_price = float(
        df1.iloc[
            entry_pos
        ]["Open"]
    )


    # ========================================================
    # 240-MIN EXIT FROM ACTUAL ENTRY
    # ========================================================

    exit_target = (
        entry_time
        +
        pd.Timedelta(
            minutes=HOLD_MINUTES
        )
    )

    exit_pos = df1.index.searchsorted(
        exit_target,
        side="left"
    )

    if exit_pos >= len(df1):
        continue

    exit_time = df1.index[
        exit_pos
    ]

    if (
        exit_time
        >
        exit_target
        +
        pd.Timedelta(
            minutes=2
        )
    ):
        continue

    exit_price = float(
        df1.iloc[
            exit_pos
        ]["Close"]
    )

    pnl = (
        exit_price
        -
        entry_price
    )


    # ========================================================
    # STORE TRADE
    # ========================================================

    rows.append({

        "SignalTime":
            signal_time,

        "Year":
            signal_time.year,

        "EntryType":
            entry_type,

        "Move1":
            move1,

        "Move2":
            move2,

        "ATR_Cross":
            atr_cross,

        "EntryTime":
            entry_time,

        "ExitTime":
            exit_time,

        "EntryPrice":
            entry_price,

        "ExitPrice":
            exit_price,

        "PnL":
            pnl
    })


results = pd.DataFrame(
    rows
)


# ============================================================
# GROUPS
# ============================================================

first = results[
    results["EntryType"]
    ==
    "ENTER_FIRST"
].copy()

second = results[
    results["EntryType"]
    ==
    "ENTER_SECOND"
].copy()

combined = results[
    results["EntryType"].isin(
        [
            "ENTER_FIRST",
            "ENTER_SECOND"
        ]
    )
].copy()

no_trade = results[
    results["EntryType"]
    ==
    "NO_TRADE"
].copy()


# ============================================================
# COUNTS
# ============================================================

print()
print("=" * 140)
print("ENTRY COUNTS")
print("=" * 140)

print(
    "ENTER_FIRST:",
    len(first)
)

print(
    "ENTER_SECOND:",
    len(second)
)

print(
    "COMBINED:",
    len(combined)
)

print(
    "NO_TRADE:",
    len(no_trade)
)

print()

print(
    "Combined retention:",
    round(
        len(combined)
        /
        len(results)
        *
        100,
        2
    ),
    "%"
)


# ============================================================
# STATS BY GROUP AND YEAR
# ============================================================

def stats_table(d, strategy_name):

    rows = []


    rows.append({

        "Strategy":
            strategy_name,

        "Year":
            "ALL",

        **calculate_stats(
            d
        )
    })


    for year in sorted(
        results["Year"].unique()
    ):

        yd = d[
            d["Year"]
            ==
            year
        ]

        rows.append({

            "Strategy":
                strategy_name,

            "Year":
                str(year),

            **calculate_stats(
                yd
            )
        })


    return pd.DataFrame(
        rows
    )


first_stats = stats_table(
    first,
    "ENTER_FIRST"
)

second_stats = stats_table(
    second,
    "ENTER_SECOND"
)

combined_stats = stats_table(
    combined,
    "COMBINED"
)


stats_df = pd.concat(
    [
        first_stats,
        second_stats,
        combined_stats
    ],
    ignore_index=True
)


# ============================================================
# RAW MACD BASELINE
#
# Immediate entry after cross.
# Same 240-minute convention.
# ============================================================

raw_rows = []


for signal_time in signals.index:

    entry_pos = df1.index.searchsorted(
        signal_time,
        side="right"
    )

    if entry_pos >= len(df1):
        continue

    entry_time = df1.index[
        entry_pos
    ]

    entry_price = float(
        df1.iloc[
            entry_pos
        ]["Open"]
    )

    exit_target = (
        entry_time
        +
        pd.Timedelta(
            minutes=HOLD_MINUTES
        )
    )

    exit_pos = df1.index.searchsorted(
        exit_target,
        side="left"
    )

    if exit_pos >= len(df1):
        continue

    exit_time = df1.index[
        exit_pos
    ]

    if (
        exit_time
        >
        exit_target
        +
        pd.Timedelta(
            minutes=2
        )
    ):
        continue

    exit_price = float(
        df1.iloc[
            exit_pos
        ]["Close"]
    )

    pnl = (
        exit_price
        -
        entry_price
    )

    raw_rows.append({

        "SignalTime":
            signal_time,

        "Year":
            signal_time.year,

        "PnL":
            pnl
    })


raw = pd.DataFrame(
    raw_rows
)

raw_stats = stats_table(
    raw,
    "RAW_MACD"
)

stats_df = pd.concat(
    [
        raw_stats,
        stats_df
    ],
    ignore_index=True
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUTPUT_DIR
    /
    "all_two_stage_results.csv",
    index=False
)

first.to_csv(
    OUTPUT_DIR
    /
    "enter_first_trades.csv",
    index=False
)

second.to_csv(
    OUTPUT_DIR
    /
    "enter_second_trades.csv",
    index=False
)

combined.to_csv(
    OUTPUT_DIR
    /
    "combined_trades.csv",
    index=False
)

no_trade.to_csv(
    OUTPUT_DIR
    /
    "no_trade_crosses.csv",
    index=False
)

stats_df.to_csv(
    OUTPUT_DIR
    /
    "two_stage_statistics.csv",
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    300
)


print()
print("=" * 180)
print("TWO-STAGE MOMENTUM RESULTS")
print("=" * 180)

print(
    stats_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 180)
print("DONE")
print("=" * 180)

print()
print("Output folder:")
print(OUTPUT_DIR)