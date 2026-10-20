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
    r"\MACD_Short_PriceMoveATR_Plateau_Test"
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

MAIN_YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026
]


# ============================================================
# SHORT PRICE-MOVE BANDS
#
# PriceMove15ATR =
# (Close_t - Close_t-1) / ATR14_t
#
# Bearish values are negative.
# ============================================================

SHORT_BANDS = [

    ("M1_80_M1_10", -1.80, -1.10),

    ("M1_70_M1_10", -1.70, -1.10),

    ("M1_60_M1_10", -1.60, -1.10),

    ("M1_50_M1_10", -1.50, -1.10),

    ("M1_50_M1_20", -1.50, -1.20),

    ("M1_50_M1_25", -1.50, -1.25),

    ("M1_45_M1_15", -1.45, -1.15),

    ("M1_40_M1_10", -1.40, -1.10),

    ("M1_40_M1_20", -1.40, -1.20),

    ("M1_35_M1_10", -1.35, -1.10),

    ("M1_30_M1_00", -1.30, -1.00),

    ("M1_25_M1_00", -1.25, -1.00),

    ("M1_60_M1_20", -1.60, -1.20),

    ("M1_70_M1_20", -1.70, -1.20),

    ("M1_80_M1_20", -1.80, -1.20),

    ("M2_00_M1_00", -2.00, -1.00),

    ("M2_00_M1_20", -2.00, -1.20),

    ("M2_00_M1_50", -2.00, -1.50)
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

    drawdown = (
        equity
        -
        running_max
    )

    return abs(
        drawdown.min()
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
# BUILD SHORT TRADE
# ============================================================

def build_short_trade(signal_time):

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
# BUILD SHORT DATASET
# ============================================================

signals = df15[
    df15["BearCross"]
    &
    df15["TimeValid"]
].copy()


print()
print("=" * 160)
print("BUILDING SHORT DATASET")
print("=" * 160)

print(
    "Raw bearish MACD crosses:",
    len(signals)
)


rows = []


for signal_time in signals.index:

    pos = df15.index.get_loc(
        signal_time
    )

    if pos < 1:
        continue


    trade = build_short_trade(
        signal_time
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


print(
    "Completed short trades:",
    len(df)
)


# ============================================================
# RAW BASELINE
# ============================================================

raw_stats = calculate_stats(
    df
)


# ============================================================
# TEST ALL BANDS
# ============================================================

overall_rows = []
year_rows = []


for band_name, lower, upper in SHORT_BANDS:

    subset = df[
        (
            df["PriceMove15ATR"]
            >=
            lower
        )
        &
        (
            df["PriceMove15ATR"]
            <=
            upper
        )
    ].copy()


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


        year_rows.append({

            "Band":
                band_name,

            "LowerATR":
                lower,

            "UpperATR":
                upper,

            "Year":
                year,

            **ys
        })


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


    overall_rows.append({

        "Band":
            band_name,

        "LowerATR":
            lower,

        "UpperATR":
            upper,

        "N":
            stats["N"],

        "RetentionVsRawPct":
            (
                len(subset)
                /
                len(df)
                *
                100
                if len(df) > 0
                else np.nan
            ),

        "MeanPriceMoveATR":
            subset[
                "PriceMove15ATR"
            ].mean(),

        "MedianPriceMoveATR":
            subset[
                "PriceMove15ATR"
            ].median(),

        "AveragePnL":
            stats["AveragePnL"],

        "NetPoints":
            stats["NetPoints"],

        "ProfitFactor":
            stats["ProfitFactor"],

        "SQN":
            stats["SQN"],

        "MaxDrawdown":
            stats["MaxDrawdown"],

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


overall_df = pd.DataFrame(
    overall_rows
)

year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# RANKING
# ============================================================

ranking_df = overall_df.copy()


ranking_df["All5Years"] = (
    ranking_df[
        "ProfitableYears"
    ]
    ==
    5
)


ranking_df = ranking_df.sort_values(

    [
        "All5Years",
        "WorstYearPF",
        "RetentionVsRawPct",
        "ProfitFactor"
    ],

    ascending=[
        False,
        False,
        False,
        False
    ]
)


# ============================================================
# FINE BUCKETS
#
# Gives the shape of short PriceMoveATR
# ============================================================

bucket_edges = [

    -np.inf,
    -2.50,
    -2.00,
    -1.75,
    -1.50,
    -1.30,
    -1.20,
    -1.10,
    -1.00,
    -0.90,
    -0.70,
    -0.50,
    -0.35,
    -0.20,
    0.00,
    np.inf
]


bucket_labels = [

    "<=-2.50",

    "-2.50 to -2.00",

    "-2.00 to -1.75",

    "-1.75 to -1.50",

    "-1.50 to -1.30",

    "-1.30 to -1.20",

    "-1.20 to -1.10",

    "-1.10 to -1.00",

    "-1.00 to -.90",

    "-.90 to -.70",

    "-.70 to -.50",

    "-.50 to -.35",

    "-.35 to -.20",

    "-.20 to 0",

    ">=0"
]


bucket_data = df.copy()


bucket_data["Bucket"] = pd.cut(
    bucket_data["PriceMove15ATR"],
    bins=bucket_edges,
    labels=bucket_labels,
    include_lowest=True
)


bucket_rows = []


for bucket in bucket_labels:

    subset = bucket_data[
        bucket_data["Bucket"]
        ==
        bucket
    ]

    stats = calculate_stats(
        subset
    )


    bucket_rows.append({

        "Bucket":
            bucket,

        **stats
    })


bucket_df = pd.DataFrame(
    bucket_rows
)


# ============================================================
# RAW YEAR-BY-YEAR
# ============================================================

raw_year_rows = []


for year in MAIN_YEARS:

    yd = df[
        df["Year"]
        ==
        year
    ]

    ys = calculate_stats(
        yd
    )


    raw_year_rows.append({

        "Year":
            year,

        **ys
    })


raw_year_df = pd.DataFrame(
    raw_year_rows
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_DIR
    /
    "all_short_macd_trades.csv",
    index=False
)

overall_df.to_csv(
    OUTPUT_DIR
    /
    "short_pricemove_band_results.csv",
    index=False
)

ranking_df.to_csv(
    OUTPUT_DIR
    /
    "short_pricemove_band_ranking.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "short_pricemove_year_by_year.csv",
    index=False
)

bucket_df.to_csv(
    OUTPUT_DIR
    /
    "short_pricemove_fine_buckets.csv",
    index=False
)

raw_year_df.to_csv(
    OUTPUT_DIR
    /
    "raw_short_year_by_year.csv",
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
print("=" * 210)
print("RAW SHORT BASELINE")
print("=" * 210)

print(
    pd.DataFrame(
        [raw_stats]
    )
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 210)
print("SHORT PRICEMOVE ATR BAND RANKING")
print("=" * 210)

print(
    ranking_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 210)
print("FINE PRICEMOVE BUCKETS")
print("=" * 210)

print(
    bucket_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 210)
print("YEAR BY YEAR")
print("=" * 210)

print(
    year_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 210)
print("DONE")
print("=" * 210)

print()
print("Output folder:")
print(OUTPUT_DIR)