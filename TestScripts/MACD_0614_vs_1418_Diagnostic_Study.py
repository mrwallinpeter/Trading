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
    r"\MACD_0614_vs_1418_Diagnostic_Study"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14

PRICE_MOVE_MIN = 0.35
PRICE_MOVE_MAX = 0.90

PRIOR_MOVE_MIN = -2.0
PRIOR_MOVE_MAX = 2.0

RANGE_LOOKBACK = 20
MAX_RANGE_HEIGHT = 25.0
MAX_RANGE_DRIFT = 10.0

DIRECTION_LOOKBACK = 20

HOLD_MINUTES = 240

MAIN_YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026
]


TIME_BLOCKS = {
    "06_10": (6, 10),
    "10_14": (10, 14),
    "06_14": (6, 14),
    "14_18": (14, 18)
}


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 180)
print("LOADING DATA")
print("=" * 180)

df1 = pd.read_csv(ONE_MIN_FILE)

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
# ATR5 EXPANSION
# ============================================================

atr5 = tr.ewm(
    alpha=1.0 / 5,
    adjust=False
).mean()

df15["ATR5"] = atr5

df15["ATR5SMA20"] = (
    df15["ATR5"]
    .rolling(20)
    .mean()
)

df15["ATR5Expansion"] = (
    df15["ATR5"]
    /
    df15["ATR5SMA20"]
)


# ============================================================
# VOLUME EXPANSION
# ============================================================

df15["VolumeSMA20"] = (
    df15["Volume"]
    .rolling(20)
    .mean()
)

df15["VolumeExpansion"] = (
    df15["Volume"]
    /
    df15["VolumeSMA20"]
)


# ============================================================
# EXTRA FEATURES
# ============================================================

df15["HistDelta1"] = (
    df15["Hist"]
    -
    df15["Hist"].shift(1)
)

df15["HistSlope3"] = (
    df15["Hist"]
    -
    df15["Hist"].shift(3)
) / 3.0

df15["HistHeightNorm"] = (
    df15["Hist"]
    /
    df15["ATR14"]
)

df15["HistDeltaNorm"] = (
    df15["HistDelta1"]
    /
    df15["ATR14"]
)

df15["HistSlope3Norm"] = (
    df15["HistSlope3"]
    /
    df15["ATR14"]
)

df15["BodyATR"] = (
    (df15["Close"] - df15["Open"]).abs()
    /
    df15["ATR14"]
)

df15["RangeATR"] = (
    (df15["High"] - df15["Low"])
    /
    df15["ATR14"]
)

candle_range = (
    df15["High"]
    -
    df15["Low"]
)

df15["CloseLocation"] = np.where(
    candle_range > 0,
    (
        df15["Close"]
        -
        df15["Low"]
    )
    /
    candle_range,
    np.nan
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

    peak = np.maximum.accumulate(
        equity
    )

    dd = equity - peak

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

    target_time = (
        entry_time
        +
        pd.Timedelta(
            minutes=HOLD_MINUTES
        )
    )

    exit_pos = df1.index.searchsorted(
        target_time,
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
        target_time
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
# RANGE INFO
# ============================================================

def get_range_info(pos):

    start_pos = (
        pos
        -
        RANGE_LOOKBACK
    )

    end_pos = (
        pos
        -
        1
    )

    window = df15.iloc[
        start_pos:end_pos + 1
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
        df15.iloc[
            start_pos
        ]["Close"]
    )

    last_close = float(
        df15.iloc[
            end_pos
        ]["Close"]
    )

    range_drift = abs(
        last_close
        -
        first_close
    )

    valid_range = (
        range_height
        <=
        MAX_RANGE_HEIGHT
        and
        range_drift
        <=
        MAX_RANGE_DRIFT
    )

    cross_close = float(
        df15.iloc[
            pos
        ]["Close"]
    )

    cross_inside = (
        valid_range
        and
        range_low
        <=
        cross_close
        <=
        range_high
    )

    return {
        "RangeHigh":
            range_high,

        "RangeLow":
            range_low,

        "RangeHeight":
            range_height,

        "RangeDrift":
            range_drift,

        "ValidRange":
            valid_range,

        "CrossInsideRange":
            cross_inside
    }


# ============================================================
# BUILD FINAL SETUP DATASET
# ============================================================

signals = df15[
    df15["BullCross"]
].copy()


print()
print("=" * 180)
print("BUILDING DATASET")
print("=" * 180)

rows = []


for signal_time in signals.index:

    pos = df15.index.get_loc(
        signal_time
    )

    if pos < max(
        RANGE_LOOKBACK,
        DIRECTION_LOOKBACK,
        20
    ):
        continue


    trade = build_trade(
        signal_time
    )

    if trade is None:
        continue


    row = df15.iloc[
        pos
    ]

    atr14 = float(
        row["ATR14"]
    )

    if (
        not np.isfinite(atr14)
        or
        atr14 <= 0
    ):
        continue


    close_now = float(
        row["Close"]
    )

    close_prev = float(
        df15.iloc[
            pos - 1
        ]["Close"]
    )


    price_move = (
        close_now
        -
        close_prev
    ) / atr14


    direction_start_close = float(
        df15.iloc[
            pos - DIRECTION_LOOKBACK
        ]["Close"]
    )


    prior_move_points = (
        close_prev
        -
        direction_start_close
    )

    prior_move_atr = (
        prior_move_points
        /
        atr14
    )


    range_info = get_range_info(
        pos
    )


    rows.append({

        "SignalTime":
            signal_time,

        "Year":
            signal_time.year,

        "Hour":
            signal_time.hour,

        "PriceMove15ATR":
            price_move,

        "PriorMoveATR":
            prior_move_atr,

        "ATR14":
            atr14,

        "ATR5Expansion":
            row["ATR5Expansion"],

        "VolumeExpansion":
            row["VolumeExpansion"],

        "HistHeightNorm":
            row["HistHeightNorm"],

        "HistDeltaNorm":
            row["HistDeltaNorm"],

        "HistSlope3Norm":
            row["HistSlope3Norm"],

        "BodyATR":
            row["BodyATR"],

        "RangeATR":
            row["RangeATR"],

        "CloseLocation":
            row["CloseLocation"],

        **range_info,

        **trade
    })


df = pd.DataFrame(
    rows
)


final_setup = df[
    (
        df["PriceMove15ATR"]
        >=
        PRICE_MOVE_MIN
    )
    &
    (
        df["PriceMove15ATR"]
        <=
        PRICE_MOVE_MAX
    )
    &
    (
        ~df["CrossInsideRange"]
    )
    &
    (
        df["PriorMoveATR"]
        >=
        PRIOR_MOVE_MIN
    )
    &
    (
        df["PriorMoveATR"]
        <=
        PRIOR_MOVE_MAX
    )
].copy()


print(
    "Final setup trades:",
    len(final_setup)
)


# ============================================================
# ASSIGN TIME BLOCK
# ============================================================

def get_time_block(hour):

    if 6 <= hour < 10:
        return "06_10"

    if 10 <= hour < 14:
        return "10_14"

    if 14 <= hour < 18:
        return "14_18"

    return "OTHER"


final_setup["TimeBlock"] = (
    final_setup["Hour"]
    .apply(
        get_time_block
    )
)


# ============================================================
# FEATURE LIST
# ============================================================

FEATURES = [

    "PriceMove15ATR",
    "PriorMoveATR",

    "ATR14",
    "ATR5Expansion",

    "VolumeExpansion",

    "HistHeightNorm",
    "HistDeltaNorm",
    "HistSlope3Norm",

    "BodyATR",
    "RangeATR",
    "CloseLocation",

    "RangeHeight",
    "RangeDrift"
]


# ============================================================
# BLOCK PERFORMANCE
# ============================================================

block_perf_rows = []


for block_name, (
    start_hour,
    end_hour
) in TIME_BLOCKS.items():

    subset = final_setup[
        (
            final_setup["Hour"]
            >=
            start_hour
        )
        &
        (
            final_setup["Hour"]
            <
            end_hour
        )
    ]

    stats = calculate_stats(
        subset
    )

    block_perf_rows.append({

        "TimeBlock":
            block_name,

        **stats
    })


block_perf_df = pd.DataFrame(
    block_perf_rows
)


# ============================================================
# YEAR-BY-YEAR PERFORMANCE
# ============================================================

year_rows = []


for block_name, (
    start_hour,
    end_hour
) in TIME_BLOCKS.items():

    for year in MAIN_YEARS:

        subset = final_setup[
            (
                final_setup["Hour"]
                >=
                start_hour
            )
            &
            (
                final_setup["Hour"]
                <
                end_hour
            )
            &
            (
                final_setup["Year"]
                ==
                year
            )
        ]

        stats = calculate_stats(
            subset
        )

        year_rows.append({

            "TimeBlock":
                block_name,

            "Year":
                year,

            **stats
        })


year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# FEATURE MEAN / MEDIAN BY BLOCK
# ============================================================

feature_rows = []


for block in [
    "06_10",
    "10_14",
    "14_18"
]:

    subset = final_setup[
        final_setup["TimeBlock"]
        ==
        block
    ]


    for feature in FEATURES:

        x = (
            subset[feature]
            .dropna()
        )

        feature_rows.append({

            "TimeBlock":
                block,

            "Feature":
                feature,

            "N":
                len(x),

            "Mean":
                x.mean(),

            "Median":
                x.median(),

            "Std":
                x.std(),

            "P25":
                x.quantile(0.25),

            "P75":
                x.quantile(0.75)
        })


feature_summary_df = pd.DataFrame(
    feature_rows
)


# ============================================================
# WINNERS VS LOSERS
# ============================================================

winner_loser_rows = []


for block in [
    "06_10",
    "10_14",
    "14_18"
]:

    block_df = final_setup[
        final_setup["TimeBlock"]
        ==
        block
    ]


    for result_name, condition in [

        (
            "WINNER",
            block_df["PnL"] > 0
        ),

        (
            "LOSER",
            block_df["PnL"] <= 0
        )
    ]:

        subset = block_df[
            condition
        ]


        for feature in FEATURES:

            x = (
                subset[feature]
                .dropna()
            )

            winner_loser_rows.append({

                "TimeBlock":
                    block,

                "Result":
                    result_name,

                "Feature":
                    feature,

                "N":
                    len(x),

                "Mean":
                    x.mean(),

                "Median":
                    x.median()
            })


winner_loser_df = pd.DataFrame(
    winner_loser_rows
)


# ============================================================
# CORRELATION WITH PNL
# ============================================================

correlation_rows = []


for block in [
    "06_10",
    "10_14",
    "06_14",
    "14_18"
]:

    start_hour, end_hour = (
        TIME_BLOCKS[
            block
        ]
    )

    subset = final_setup[
        (
            final_setup["Hour"]
            >=
            start_hour
        )
        &
        (
            final_setup["Hour"]
            <
            end_hour
        )
    ]


    for feature in FEATURES:

        temp = subset[
            [
                feature,
                "PnL"
            ]
        ].dropna()

        if len(temp) >= 3:

            corr = (
                temp[feature]
                .corr(
                    temp["PnL"]
                )
            )

        else:

            corr = np.nan


        correlation_rows.append({

            "TimeBlock":
                block,

            "Feature":
                feature,

            "N":
                len(temp),

            "CorrelationWithPnL":
                corr
        })


correlation_df = pd.DataFrame(
    correlation_rows
)


# ============================================================
# QUINTILE STUDY INSIDE 06-14
#
# Diagnostic only.
# ============================================================

morning = final_setup[
    (
        final_setup["Hour"]
        >=
        6
    )
    &
    (
        final_setup["Hour"]
        <
        14
    )
].copy()


quintile_rows = []


for feature in FEATURES:

    valid = morning[
        [
            feature,
            "PnL",
            "Year"
        ]
    ].dropna().copy()


    if (
        len(valid) < 25
        or
        valid[feature].nunique() < 5
    ):
        continue


    try:

        valid["Bucket"] = pd.qcut(
            valid[feature],
            5,
            duplicates="drop"
        )

    except Exception:
        continue


    for bucket, subset in valid.groupby(
        "Bucket",
        observed=True
    ):

        stats = calculate_stats(
            subset
        )

        year_pfs = []
        profitable_years = 0


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


        quintile_rows.append({

            "Feature":
                feature,

            "Bucket":
                str(bucket),

            "N":
                stats["N"],

            "AveragePnL":
                stats["AveragePnL"],

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
                profitable_years
        })


quintile_df = pd.DataFrame(
    quintile_rows
)


# ============================================================
# COMPARE 06-14 DIRECTLY AGAINST 14-18
# ============================================================

morning = final_setup[
    (
        final_setup["Hour"]
        >=
        6
    )
    &
    (
        final_setup["Hour"]
        <
        14
    )
]

strong = final_setup[
    (
        final_setup["Hour"]
        >=
        14
    )
    &
    (
        final_setup["Hour"]
        <
        18
    )
]


comparison_rows = []


for feature in FEATURES:

    a = (
        morning[feature]
        .dropna()
    )

    b = (
        strong[feature]
        .dropna()
    )


    comparison_rows.append({

        "Feature":
            feature,

        "Morning06_14_N":
            len(a),

        "Morning06_14_Mean":
            a.mean(),

        "Morning06_14_Median":
            a.median(),

        "Strong14_18_N":
            len(b),

        "Strong14_18_Mean":
            b.mean(),

        "Strong14_18_Median":
            b.median(),

        "MeanDifference_14_18_minus_06_14":
            (
                b.mean()
                -
                a.mean()
            ),

        "MedianDifference_14_18_minus_06_14":
            (
                b.median()
                -
                a.median()
            )
    })


comparison_df = pd.DataFrame(
    comparison_rows
)


# ============================================================
# SAVE
# ============================================================

final_setup.to_csv(
    OUTPUT_DIR
    /
    "final_setup_dataset.csv",
    index=False
)

block_perf_df.to_csv(
    OUTPUT_DIR
    /
    "time_block_performance.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "time_block_year_by_year.csv",
    index=False
)

feature_summary_df.to_csv(
    OUTPUT_DIR
    /
    "feature_summary_by_time_block.csv",
    index=False
)

winner_loser_df.to_csv(
    OUTPUT_DIR
    /
    "winner_loser_feature_comparison.csv",
    index=False
)

correlation_df.to_csv(
    OUTPUT_DIR
    /
    "feature_pnl_correlations.csv",
    index=False
)

quintile_df.to_csv(
    OUTPUT_DIR
    /
    "0614_feature_quintiles.csv",
    index=False
)

comparison_df.to_csv(
    OUTPUT_DIR
    /
    "0614_vs_1418_feature_comparison.csv",
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
    500
)


print()
print("=" * 220)
print("TIME BLOCK PERFORMANCE")
print("=" * 220)

print(
    block_perf_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("YEAR BY YEAR")
print("=" * 220)

print(
    year_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("FEATURE SUMMARY BY TIME BLOCK")
print("=" * 220)

print(
    feature_summary_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("06-14 VS 14-18 FEATURE COMPARISON")
print("=" * 220)

print(
    comparison_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("WINNERS VS LOSERS")
print("=" * 220)

print(
    winner_loser_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("FEATURE CORRELATIONS WITH PNL")
print("=" * 220)

print(
    correlation_df
    .sort_values(
        [
            "TimeBlock",
            "CorrelationWithPnL"
        ],
        ascending=[
            True,
            False
        ]
    )
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("06-14 FEATURE QUINTILES")
print("=" * 220)

print(
    quintile_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("DONE")
print("=" * 220)

print()
print("Output folder:")
print(OUTPUT_DIR)