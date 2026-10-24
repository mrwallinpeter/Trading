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
    r"\MACD_FirstCandle_Context_Study"
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

ATR14_LENGTH = 14

ATR5_LENGTH = 5
ATR5_SMA_LENGTH = 20

START_HOUR = 10
END_HOUR = 18

MOVE_MIN = 0.35
MOVE_MAX = 0.90

HOLD_MINUTES = 240


# Sideways definition
SIDEWAYS_LOOKBACK = 20
SIDEWAYS_MAX_RANGE = 25.0
SIDEWAYS_MAX_DRIFT = 10.0


# ============================================================
# LOAD 1-MIN DATA
# ============================================================

print()
print("=" * 140)
print("LOADING DATA")
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
# ATR HELPER
# ============================================================

def wilder_atr(data, length):

    prev_close = data["Close"].shift(1)

    tr = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - prev_close).abs(),
            (data["Low"] - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1.0 / length,
        adjust=False
    ).mean()


# ============================================================
# ATR FEATURES
# ============================================================

df15["ATR14"] = wilder_atr(
    df15,
    ATR14_LENGTH
)

df15["ATR5"] = wilder_atr(
    df15,
    ATR5_LENGTH
)

df15["ATR5_SMA20"] = (
    df15["ATR5"]
    .rolling(
        ATR5_SMA_LENGTH
    )
    .mean()
)

df15["ATR5Expansion"] = (
    df15["ATR5"]
    /
    df15["ATR5_SMA20"]
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
# STATS FUNCTIONS
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
            "SQN": np.nan
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
            )
    }


# ============================================================
# BUILD 240-MIN TRADE
# ============================================================

def build_trade(signal_time):

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
        pd.Timedelta(minutes=2)
    ):
        return None


    exit_price = float(
        df1.iloc[
            exit_pos
        ]["Close"]
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
            exit_price
            -
            entry_price
    }


# ============================================================
# SIDEWAYS RANGE CHECK
#
# Previous 20 bars only.
# Range ends one bar before cross.
#
# Also check whether cross close remains inside range.
# ============================================================

def sideways_info(pos):

    if pos < SIDEWAYS_LOOKBACK:

        return {
            "SidewaysRangeValid": False,
            "CrossInsideSideways": False,
            "SidewaysHeight": np.nan,
            "SidewaysDrift": np.nan
        }


    start = (
        pos
        -
        SIDEWAYS_LOOKBACK
    )

    end = (
        pos
        -
        1
    )


    window = df15.iloc[
        start:end + 1
    ]


    range_high = float(
        window["High"].max()
    )

    range_low = float(
        window["Low"].min()
    )

    range_height = (
        range_high
        -
        range_low
    )


    first_close = float(
        df15.iloc[start]["Close"]
    )

    last_close = float(
        df15.iloc[end]["Close"]
    )

    drift = abs(
        last_close
        -
        first_close
    )


    valid = (
        range_height
        <=
        SIDEWAYS_MAX_RANGE
        and
        drift
        <=
        SIDEWAYS_MAX_DRIFT
    )


    cross_close = float(
        df15.iloc[pos]["Close"]
    )


    inside = (
        valid
        and
        range_low
        <=
        cross_close
        <=
        range_high
    )


    return {

        "SidewaysRangeValid":
            valid,

        "CrossInsideSideways":
            inside,

        "SidewaysHeight":
            range_height,

        "SidewaysDrift":
            drift
    }


# ============================================================
# BUILD MACD CROSS DATASET
# ============================================================

signals = df15[
    df15["BullCross"]
    &
    df15["TimeValid"]
].copy()


print()
print("=" * 140)
print("BUILDING CONTEXT DATASET")
print("=" * 140)

print(
    "Raw MACD crosses:",
    len(signals)
)


rows = []


for signal_time in signals.index:

    pos = df15.index.get_loc(
        signal_time
    )

    if pos < 4:
        continue


    trade = build_trade(
        signal_time
    )

    if trade is None:
        continue


    atr14 = float(
        df15.iloc[pos]["ATR14"]
    )

    if (
        not np.isfinite(atr14)
        or
        atr14 <= 0
    ):
        continue


    # --------------------------------------------------------
    # FIRST-CANDLE PRICE MOVE
    # --------------------------------------------------------

    close0 = float(
        df15.iloc[pos]["Close"]
    )

    close1 = float(
        df15.iloc[pos - 1]["Close"]
    )


    price_move_15_atr = (
        close0
        -
        close1
    ) / atr14


    qualified = (
        MOVE_MIN
        <=
        price_move_15_atr
        <=
        MOVE_MAX
    )


    group = (
        "QUALIFIED"
        if qualified
        else "REJECTED"
    )


    # --------------------------------------------------------
    # CLOSE LOCATION
    # --------------------------------------------------------

    high0 = float(
        df15.iloc[pos]["High"]
    )

    low0 = float(
        df15.iloc[pos]["Low"]
    )

    candle_range = (
        high0
        -
        low0
    )


    close_location = (
        (
            close0
            -
            low0
        )
        /
        candle_range
        if candle_range > 0
        else 0.5
    )


    # --------------------------------------------------------
    # HISTOGRAM SLOPE 3
    # --------------------------------------------------------

    hist0 = float(
        df15.iloc[pos]["Hist"]
    )

    hist3 = float(
        df15.iloc[pos - 3]["Hist"]
    )


    hist_slope3 = (
        hist0
        -
        hist3
    ) / 3.0


    hist_slope3_norm = (
        hist_slope3
        /
        atr14
    )


    # --------------------------------------------------------
    # ATR EXPANSION
    # --------------------------------------------------------

    atr5_expansion = float(
        df15.iloc[pos]["ATR5Expansion"]
    )


    # --------------------------------------------------------
    # SIDEWAYS
    # --------------------------------------------------------

    sw = sideways_info(
        pos
    )


    # --------------------------------------------------------
    # TIME OF DAY
    # --------------------------------------------------------

    hour_decimal = (
        signal_time.hour
        +
        signal_time.minute
        /
        60.0
    )


    rows.append({

        "SignalTime":
            signal_time,

        "Year":
            signal_time.year,

        "Group":
            group,

        "PriceMove15ATR":
            price_move_15_atr,

        "ATR14":
            atr14,

        "ATR5Expansion":
            atr5_expansion,

        "CloseLocation":
            close_location,

        "HistSlope3Norm":
            hist_slope3_norm,

        "HourDecimal":
            hour_decimal,

        "SidewaysRangeValid":
            sw["SidewaysRangeValid"],

        "CrossInsideSideways":
            sw["CrossInsideSideways"],

        "SidewaysHeight":
            sw["SidewaysHeight"],

        "SidewaysDrift":
            sw["SidewaysDrift"],

        **trade
    })


df = pd.DataFrame(
    rows
)


# ============================================================
# BASIC GROUP STATS
# ============================================================

group_rows = []


for group in [
    "QUALIFIED",
    "REJECTED"
]:

    d = df[
        df["Group"]
        ==
        group
    ]

    group_rows.append({

        "Group":
            group,

        **calculate_stats(
            d
        )
    })


group_stats = pd.DataFrame(
    group_rows
)


# ============================================================
# CONTEXT SUMMARY
# ============================================================

context_features = [

    "ATR5Expansion",
    "CloseLocation",
    "HistSlope3Norm",
    "HourDecimal",
    "SidewaysHeight",
    "SidewaysDrift"
]


summary_rows = []


for feature in context_features:

    for group in [
        "QUALIFIED",
        "REJECTED"
    ]:

        d = (
            df[
                df["Group"]
                ==
                group
            ][feature]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )


        summary_rows.append({

            "Feature":
                feature,

            "Group":
                group,

            "N":
                len(d),

            "Mean":
                d.mean(),

            "Median":
                d.median(),

            "Std":
                d.std(),

            "P10":
                d.quantile(0.10),

            "P25":
                d.quantile(0.25),

            "P75":
                d.quantile(0.75),

            "P90":
                d.quantile(0.90)
        })


context_summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# BOOLEAN CONTEXT SUMMARY
# ============================================================

boolean_rows = []


for field in [

    "SidewaysRangeValid",
    "CrossInsideSideways"

]:

    for group in [
        "QUALIFIED",
        "REJECTED"
    ]:

        d = df[
            df["Group"]
            ==
            group
        ]


        boolean_rows.append({

            "Feature":
                field,

            "Group":
                group,

            "N":
                len(d),

            "TrueN":
                int(
                    d[field].sum()
                ),

            "TruePct":
                d[field].mean()
                *
                100
        })


boolean_summary = pd.DataFrame(
    boolean_rows
)


# ============================================================
# BUCKET FUNCTION
# ============================================================

def run_bucket_test(
    feature,
    bins,
    labels
):

    temp = df[
        [
            feature,
            "PnL",
            "Year",
            "Group"
        ]
    ].copy()


    temp["Bucket"] = pd.cut(
        temp[feature],
        bins=bins,
        labels=labels,
        include_lowest=True
    )


    rows = []


    for group in [
        "QUALIFIED",
        "REJECTED"
    ]:

        for bucket in labels:

            d = temp[
                (
                    temp["Group"]
                    ==
                    group
                )
                &
                (
                    temp["Bucket"]
                    ==
                    bucket
                )
            ]


            stats = calculate_stats(
                d
            )


            rows.append({

                "Feature":
                    feature,

                "Group":
                    group,

                "Bucket":
                    bucket,

                **stats
            })


    return pd.DataFrame(
        rows
    )


# ============================================================
# ATR EXPANSION BUCKETS
# ============================================================

atr_bucket_df = run_bucket_test(

    "ATR5Expansion",

    bins=[
        -np.inf,
        0.90,
        1.00,
        1.10,
        1.20,
        1.30,
        np.inf
    ],

    labels=[
        "<0.90",
        "0.90-1.00",
        "1.00-1.10",
        "1.10-1.20",
        "1.20-1.30",
        ">=1.30"
    ]
)


# ============================================================
# CLOSE LOCATION BUCKETS
# ============================================================

close_bucket_df = run_bucket_test(

    "CloseLocation",

    bins=[
        -np.inf,
        0.70,
        0.80,
        0.90,
        0.95,
        np.inf
    ],

    labels=[
        "<0.70",
        "0.70-0.80",
        "0.80-0.90",
        "0.90-0.95",
        ">=0.95"
    ]
)


# ============================================================
# HIST SLOPE BUCKETS
# ============================================================

hist_bucket_df = run_bucket_test(

    "HistSlope3Norm",

    bins=[
        -np.inf,
        0.00,
        0.02,
        0.04,
        0.08,
        np.inf
    ],

    labels=[
        "<=0.00",
        "0.00-0.02",
        "0.02-0.04",
        "0.04-0.08",
        ">0.08"
    ]
)


# ============================================================
# TIME OF DAY BUCKETS
# ============================================================

time_bucket_df = run_bucket_test(

    "HourDecimal",

    bins=[
        10.0,
        12.0,
        14.0,
        16.0,
        18.0
    ],

    labels=[
        "10-12",
        "12-14",
        "14-16",
        "16-18"
    ]
)


# ============================================================
# YEAR-BY-YEAR GROUP STATS
# ============================================================

year_rows = []


for group in [
    "QUALIFIED",
    "REJECTED"
]:

    for year in sorted(
        df["Year"].unique()
    ):

        d = df[
            (
                df["Group"]
                ==
                group
            )
            &
            (
                df["Year"]
                ==
                year
            )
        ]


        year_rows.append({

            "Group":
                group,

            "Year":
                year,

            **calculate_stats(
                d
            )
        })


year_stats = pd.DataFrame(
    year_rows
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_DIR
    /
    "macd_first_candle_context_dataset.csv",
    index=False
)

group_stats.to_csv(
    OUTPUT_DIR
    /
    "qualified_vs_rejected_stats.csv",
    index=False
)

context_summary.to_csv(
    OUTPUT_DIR
    /
    "context_distribution_summary.csv",
    index=False
)

boolean_summary.to_csv(
    OUTPUT_DIR
    /
    "sideways_context_summary.csv",
    index=False
)

atr_bucket_df.to_csv(
    OUTPUT_DIR
    /
    "atr_expansion_buckets.csv",
    index=False
)

close_bucket_df.to_csv(
    OUTPUT_DIR
    /
    "close_location_buckets.csv",
    index=False
)

hist_bucket_df.to_csv(
    OUTPUT_DIR
    /
    "hist_slope_buckets.csv",
    index=False
)

time_bucket_df.to_csv(
    OUTPUT_DIR
    /
    "time_of_day_buckets.csv",
    index=False
)

year_stats.to_csv(
    OUTPUT_DIR
    /
    "qualified_vs_rejected_year_by_year.csv",
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
    350
)

pd.set_option(
    "display.max_rows",
    300
)


print()
print("=" * 180)
print("QUALIFIED VS REJECTED")
print("=" * 180)

print(
    group_stats
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 180)
print("YEAR-BY-YEAR")
print("=" * 180)

print(
    year_stats
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 180)
print("CONTEXT DISTRIBUTIONS")
print("=" * 180)

print(
    context_summary
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 180)
print("SIDEWAYS FREQUENCY")
print("=" * 180)

print(
    boolean_summary
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 180)
print("ATR EXPANSION BUCKETS")
print("=" * 180)

print(
    atr_bucket_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 180)
print("CLOSE LOCATION BUCKETS")
print("=" * 180)

print(
    close_bucket_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 180)
print("HISTOGRAM SLOPE BUCKETS")
print("=" * 180)

print(
    hist_bucket_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 180)
print("TIME OF DAY BUCKETS")
print("=" * 180)

print(
    time_bucket_df
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