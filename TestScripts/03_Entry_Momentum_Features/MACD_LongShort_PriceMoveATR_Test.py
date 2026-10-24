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
    r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\TestScripts"
    r"\MACD_LongShort_PriceMoveATR_Test"
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

HOLD_MINUTES = 240

LONG_PM_MIN = 0.35
LONG_PM_MAX = 0.90

SHORT_PM_MIN = -0.90
SHORT_PM_MAX = -0.35

MAIN_YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026
]


# ============================================================
# LOAD 1-MIN DATA
# ============================================================

print()
print("=" * 160)
print("LOADING DATA")
print("=" * 160)

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

df15["BearCross"] = (
    (df15["Hist"].shift(1) >= 0)
    &
    (df15["Hist"] < 0)
)


# ============================================================
# ATR14
# ============================================================

prev_close = df15["Close"].shift(1)

tr = pd.concat(
    [
        df15["High"] - df15["Low"],
        (df15["High"] - prev_close).abs(),
        (df15["Low"] - prev_close).abs()
    ],
    axis=1
).max(axis=1)

df15["ATR14"] = tr.ewm(
    alpha=1.0 / ATR_LENGTH,
    adjust=False
).mean()


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

    gp = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()

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

    equity = np.concatenate(
        [
            [0.0],
            np.cumsum(pnl)
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
            "MaxDrawdown": np.nan
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

        "MaxDrawdown":
            max_drawdown(
                pnl
            )
    }


# ============================================================
# BUILD TRADE
# ============================================================

def build_trade(signal_time, direction):

    entry_pos = df1.index.searchsorted(
        signal_time,
        side="right"
    )

    if entry_pos >= len(df1):
        return None

    entry_time = df1.index[
        entry_pos
    ]

    entry_price = float(
        df1.iloc[
            entry_pos
        ]["Open"]
    )

    target = (
        entry_time
        +
        pd.Timedelta(
            minutes=HOLD_MINUTES
        )
    )

    exit_pos = df1.index.searchsorted(
        target,
        side="left"
    )

    if exit_pos >= len(df1):
        return None

    exit_time = df1.index[
        exit_pos
    ]

    if (
        exit_time
        >
        target
        +
        pd.Timedelta(
            minutes=2
        )
    ):
        return None

    exit_price = float(
        df1.iloc[
            exit_pos
        ]["Close"]
    )

    if direction == "LONG":

        pnl = (
            exit_price
            -
            entry_price
        )

    else:

        pnl = (
            entry_price
            -
            exit_price
        )

    return {

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
    }


# ============================================================
# BUILD SIGNAL DATASET
# ============================================================

print()
print("=" * 160)
print("BUILDING SIGNAL DATASET")
print("=" * 160)

rows = []


for signal_time, row in df15[
    df15["TimeValid"]
    &
    (
        df15["BullCross"]
        |
        df15["BearCross"]
    )
].iterrows():

    if row["BullCross"]:

        direction = "LONG"

    elif row["BearCross"]:

        direction = "SHORT"

    else:

        continue


    pos = df15.index.get_loc(
        signal_time
    )

    if pos < 1:
        continue


    trade = build_trade(
        signal_time,
        direction
    )

    if trade is None:
        continue


    atr14 = float(
        df15.iloc[
            pos
        ]["ATR14"]
    )

    if (
        not np.isfinite(atr14)
        or
        atr14 <= 0
    ):
        continue


    close_now = float(
        df15.iloc[
            pos
        ]["Close"]
    )

    close_prev = float(
        df15.iloc[
            pos - 1
        ]["Close"]
    )


    price_move_15_atr = (
        close_now
        -
        close_prev
    ) / atr14


    rows.append({

        "SignalTime":
            signal_time,

        "Year":
            signal_time.year,

        "Direction":
            direction,

        "ATR14":
            atr14,

        "ClosePrev":
            close_prev,

        "CloseCross":
            close_now,

        "PriceMove15ATR":
            price_move_15_atr,

        **trade
    })


df = pd.DataFrame(
    rows
)


long_raw = df[
    df["Direction"]
    ==
    "LONG"
].copy()

short_raw = df[
    df["Direction"]
    ==
    "SHORT"
].copy()


long_filtered = long_raw[
    (
        long_raw["PriceMove15ATR"]
        >=
        LONG_PM_MIN
    )
    &
    (
        long_raw["PriceMove15ATR"]
        <=
        LONG_PM_MAX
    )
].copy()


short_filtered = short_raw[
    (
        short_raw["PriceMove15ATR"]
        >=
        SHORT_PM_MIN
    )
    &
    (
        short_raw["PriceMove15ATR"]
        <=
        SHORT_PM_MAX
    )
].copy()


print(
    "Completed LONG trades:",
    len(long_raw)
)

print(
    "Completed SHORT trades:",
    len(short_raw)
)

print(
    "LONG PriceMove 0.35 to 0.90:",
    len(long_filtered)
)

print(
    "SHORT PriceMove -0.90 to -0.35:",
    len(short_filtered)
)


# ============================================================
# OVERALL SUMMARY
# ============================================================

groups = {

    "LONG_RAW":
        long_raw,

    "LONG_PM_035_090":
        long_filtered,

    "SHORT_RAW":
        short_raw,

    "SHORT_PM_M090_M035":
        short_filtered
}


summary_rows = []


for name, subset in groups.items():

    stats = calculate_stats(
        subset
    )

    year_pfs = []
    profitable_years = 0
    min_year_trades = np.inf

    for year in MAIN_YEARS:

        yd = subset[
            subset["Year"]
            ==
            year
        ]

        ys = calculate_stats(
            yd
        )

        if np.isfinite(
            ys["ProfitFactor"]
        ):

            year_pfs.append(
                ys["ProfitFactor"]
            )

            if (
                ys["ProfitFactor"]
                >
                1
            ):
                profitable_years += 1

        min_year_trades = min(
            min_year_trades,
            ys["N"]
        )


    summary_rows.append({

        "Strategy":
            name,

        **stats,

        "WorstYearPF":
            (
                min(year_pfs)
                if year_pfs
                else np.nan
            ),

        "ProfitableYears":
            profitable_years,

        "MinYearTrades":
            min_year_trades
    })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# YEAR BY YEAR
# ============================================================

year_rows = []


for name, subset in groups.items():

    for year in MAIN_YEARS:

        yd = subset[
            subset["Year"]
            ==
            year
        ]

        ys = calculate_stats(
            yd
        )

        year_rows.append({

            "Strategy":
                name,

            "Year":
                year,

            **ys
        })


year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# PRICE-MOVE BUCKETS
#
# Useful to see if shorts prefer different magnitude.
# ============================================================

long_bins = [
    -np.inf,
    0.00,
    0.20,
    0.35,
    0.50,
    0.70,
    0.90,
    1.20,
    1.50,
    np.inf
]

long_labels = [
    "<0",
    "0-.20",
    ".20-.35",
    ".35-.50",
    ".50-.70",
    ".70-.90",
    ".90-1.20",
    "1.20-1.50",
    ">=1.50"
]


short_bins = [
    -np.inf,
    -1.50,
    -1.20,
    -0.90,
    -0.70,
    -0.50,
    -0.35,
    -0.20,
    0.00,
    np.inf
]

short_labels = [
    "<=-1.50",
    "-1.50--1.20",
    "-1.20--.90",
    "-.90--.70",
    "-.70--.50",
    "-.50--.35",
    "-.35--.20",
    "-.20-0",
    ">=0"
]


long_bucket_data = long_raw.copy()

long_bucket_data["Bucket"] = pd.cut(
    long_bucket_data["PriceMove15ATR"],
    bins=long_bins,
    labels=long_labels,
    include_lowest=True
)


short_bucket_data = short_raw.copy()

short_bucket_data["Bucket"] = pd.cut(
    short_bucket_data["PriceMove15ATR"],
    bins=short_bins,
    labels=short_labels,
    include_lowest=True
)


bucket_rows = []


for bucket in long_labels:

    subset = long_bucket_data[
        long_bucket_data["Bucket"]
        ==
        bucket
    ]

    stats = calculate_stats(
        subset
    )

    bucket_rows.append({

        "Direction":
            "LONG",

        "Bucket":
            bucket,

        **stats
    })


for bucket in short_labels:

    subset = short_bucket_data[
        short_bucket_data["Bucket"]
        ==
        bucket
    ]

    stats = calculate_stats(
        subset
    )

    bucket_rows.append({

        "Direction":
            "SHORT",

        "Bucket":
            bucket,

        **stats
    })


bucket_df = pd.DataFrame(
    bucket_rows
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_DIR
    /
    "all_long_short_macd_trades.csv",
    index=False
)

long_raw.to_csv(
    OUTPUT_DIR
    /
    "long_raw.csv",
    index=False
)

short_raw.to_csv(
    OUTPUT_DIR
    /
    "short_raw.csv",
    index=False
)

long_filtered.to_csv(
    OUTPUT_DIR
    /
    "long_pricemove_035_090.csv",
    index=False
)

short_filtered.to_csv(
    OUTPUT_DIR
    /
    "short_pricemove_m090_m035.csv",
    index=False
)

summary_df.to_csv(
    OUTPUT_DIR
    /
    "summary.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "year_by_year.csv",
    index=False
)

bucket_df.to_csv(
    OUTPUT_DIR
    /
    "pricemove_buckets.csv",
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
    400
)

pd.set_option(
    "display.max_rows",
    300
)


print()
print("=" * 200)
print("OVERALL SUMMARY")
print("=" * 200)

print(
    summary_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 200)
print("YEAR BY YEAR")
print("=" * 200)

print(
    year_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 200)
print("PRICE MOVE BUCKETS")
print("=" * 200)

print(
    bucket_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 200)
print("DONE")
print("=" * 200)

print()
print("Output folder:")
print(OUTPUT_DIR)