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
    r"\MACD_General_Indicator_Comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR14_LENGTH = 14
ATR5_LENGTH = 5

ATR14_SMA_LENGTH = 50
ATR5_SMA_LENGTH = 20

VOLUME_SMA_LENGTH = 20

START_HOUR = 10
END_HOUR = 18

HOLD_MINUTES = 240

MAIN_YEARS = [2022, 2023, 2024, 2025, 2026]


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 150)
print("LOADING DATA")
print("=" * 150)

df1 = pd.read_csv(ONE_MIN_FILE)

df1["DateTime"] = pd.to_datetime(df1["DateTime"])

df1 = (
    df1
    .sort_values("DateTime")
    .drop_duplicates("DateTime")
    .set_index("DateTime")
)

for col in ["Open", "High", "Low", "Close", "Volume"]:
    df1[col] = pd.to_numeric(
        df1[col],
        errors="coerce"
    )

df1 = df1.dropna(
    subset=["Open", "High", "Low", "Close"]
)

print("Rows:", len(df1))
print("Start:", df1.index.min())
print("End:  ", df1.index.max())


# ============================================================
# BUILD 15-MIN BARS
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
    subset=["Open", "High", "Low", "Close"]
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
# ATR FUNCTION
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

df15["ATR14_SMA50"] = (
    df15["ATR14"]
    .rolling(ATR14_SMA_LENGTH)
    .mean()
)

df15["ATR5_SMA20"] = (
    df15["ATR5"]
    .rolling(ATR5_SMA_LENGTH)
    .mean()
)


df15["RelATR14"] = (
    df15["ATR14"]
    /
    df15["ATR14_SMA50"]
)

df15["ATR5Expansion"] = (
    df15["ATR5"]
    /
    df15["ATR5_SMA20"]
)

df15["ATR14Expansion5"] = (
    df15["ATR14"]
    /
    df15["ATR14"].shift(5)
)


# ============================================================
# VOLUME
# ============================================================

df15["VolumeSMA20"] = (
    df15["Volume"]
    .rolling(VOLUME_SMA_LENGTH)
    .mean()
)

df15["VolumeExpansion"] = (
    df15["Volume"]
    /
    df15["VolumeSMA20"]
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

    equity = np.concatenate([
        [0.0],
        np.cumsum(pnl)
    ])

    running_max = np.maximum.accumulate(
        equity
    )

    dd = equity - running_max

    return abs(
        dd.min()
    )


def calculate_stats(d):

    pnl = (
        d["PnL"]
        .dropna()
        .to_numpy(dtype=float)
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
            np.mean(pnl > 0) * 100,

        "NetPoints":
            pnl.sum(),

        "AveragePnL":
            pnl.mean(),

        "MedianPnL":
            np.median(pnl),

        "ProfitFactor":
            profit_factor(pnl),

        "SQN":
            sqn(pnl),

        "MaxDrawdown":
            max_drawdown(pnl)
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
        df1.iloc[entry_pos]["Open"]
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
        df1.iloc[exit_pos]["Close"]
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
            exit_price - entry_price
    }


# ============================================================
# BUILD FEATURE DATASET
# ============================================================

signals = df15[
    df15["BullCross"]
    &
    df15["TimeValid"]
].copy()


print()
print("=" * 150)
print("BUILDING FEATURE DATASET")
print("=" * 150)

print("Raw MACD crosses:", len(signals))


rows = []


for signal_time in signals.index:

    pos = df15.index.get_loc(
        signal_time
    )

    if pos < 5:
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


    open0 = float(
        df15.iloc[pos]["Open"]
    )

    high0 = float(
        df15.iloc[pos]["High"]
    )

    low0 = float(
        df15.iloc[pos]["Low"]
    )

    close0 = float(
        df15.iloc[pos]["Close"]
    )

    close_prev = float(
        df15.iloc[pos - 1]["Close"]
    )


    candle_range = (
        high0
        -
        low0
    )

    body = abs(
        close0
        -
        open0
    )


    # --------------------------------------------------------
    # PRICE MOVE
    # --------------------------------------------------------

    price_move_15_atr = (
        close0
        -
        close_prev
    ) / atr14


    # --------------------------------------------------------
    # RANGE / BODY
    # --------------------------------------------------------

    range_atr = (
        candle_range
        /
        atr14
    )

    body_atr = (
        body
        /
        atr14
    )


    # --------------------------------------------------------
    # CLOSE LOCATION
    # --------------------------------------------------------

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
    # HISTOGRAM FEATURES
    # --------------------------------------------------------

    hist0 = float(
        df15.iloc[pos]["Hist"]
    )

    hist1 = float(
        df15.iloc[pos - 1]["Hist"]
    )

    hist3 = float(
        df15.iloc[pos - 3]["Hist"]
    )


    hist_delta_norm = (
        hist0
        -
        hist1
    ) / atr14


    hist_slope3_norm = (
        (
            hist0
            -
            hist3
        )
        /
        3.0
    ) / atr14


    rows.append({

        "SignalTime":
            signal_time,

        "Year":
            signal_time.year,

        "PriceMove15ATR":
            price_move_15_atr,

        "ATR14":
            atr14,

        "RelATR14":
            float(
                df15.iloc[pos]["RelATR14"]
            ),

        "ATR5Expansion":
            float(
                df15.iloc[pos]["ATR5Expansion"]
            ),

        "ATR14Expansion5":
            float(
                df15.iloc[pos]["ATR14Expansion5"]
            ),

        "RangeATR":
            range_atr,

        "BodyATR":
            body_atr,

        "CloseLocation":
            close_location,

        "HistDeltaNorm":
            hist_delta_norm,

        "HistSlope3Norm":
            hist_slope3_norm,

        "VolumeExpansion":
            float(
                df15.iloc[pos]["VolumeExpansion"]
            ),

        **trade
    })


df = pd.DataFrame(
    rows
)


print("Completed trades:", len(df))


# ============================================================
# RAW BASELINE
# ============================================================

raw_stats = calculate_stats(
    df
)

print()
print("RAW BASELINE")

print(
    pd.DataFrame(
        [raw_stats]
    )
    .round(4)
    .to_string(index=False)
)


# ============================================================
# BENCHMARK
# ============================================================

benchmark = df[
    (
        df["PriceMove15ATR"]
        >=
        0.35
    )
    &
    (
        df["PriceMove15ATR"]
        <=
        0.90
    )
].copy()


benchmark_stats = calculate_stats(
    benchmark
)


# ============================================================
# GENERAL TEST ENGINE
#
# We test percentile ranges rather than exact optimized values.
#
# Each feature tests broad central / lower / upper sections.
# ============================================================

FEATURES = [

    "ATR14",

    "RelATR14",

    "ATR5Expansion",

    "ATR14Expansion5",

    "RangeATR",

    "BodyATR",

    "CloseLocation",

    "HistDeltaNorm",

    "HistSlope3Norm",

    "VolumeExpansion"
]


# ============================================================
# PERCENTILE RANGE DEFINITIONS
#
# These are intentionally broad.
# ============================================================

PERCENTILE_TESTS = [

    (0.00, 0.40),

    (0.00, 0.50),

    (0.00, 0.60),

    (0.10, 0.50),

    (0.10, 0.60),

    (0.20, 0.60),

    (0.20, 0.70),

    (0.20, 0.80),

    (0.30, 0.70),

    (0.30, 0.80),

    (0.40, 0.80),

    (0.40, 0.90),

    (0.50, 1.00),

    (0.60, 1.00)
]


overall_rows = []
year_rows = []


# ============================================================
# ADD BENCHMARK FIRST
# ============================================================

benchmark_year_pf = []


for year in MAIN_YEARS:

    yd = benchmark[
        benchmark["Year"]
        ==
        year
    ]

    ys = calculate_stats(
        yd
    )

    benchmark_year_pf.append(
        ys["ProfitFactor"]
    )


overall_rows.append({

    "Feature":
        "PriceMove15ATR",

    "Test":
        "BENCHMARK_0.35_0.90",

    "LowerValue":
        0.35,

    "UpperValue":
        0.90,

    "RetentionPct":
        len(benchmark)
        /
        len(df)
        *
        100,

    **benchmark_stats,

    "WorstYearPF":
        np.nanmin(
            benchmark_year_pf
        ),

    "ProfitableYears":
        sum(
            pf > 1
            for pf in benchmark_year_pf
            if np.isfinite(pf)
        )
})


for year in MAIN_YEARS:

    yd = benchmark[
        benchmark["Year"]
        ==
        year
    ]

    ys = calculate_stats(
        yd
    )

    year_rows.append({

        "Feature":
            "PriceMove15ATR",

        "Test":
            "BENCHMARK_0.35_0.90",

        "Year":
            year,

        **ys
    })


# ============================================================
# TEST FEATURES
# ============================================================

for feature in FEATURES:

    values = (
        df[feature]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )


    if len(values) == 0:
        continue


    for qlow, qhigh in PERCENTILE_TESTS:

        lower = values.quantile(
            qlow
        )

        upper = values.quantile(
            qhigh
        )


        subset = df[
            (
                df[feature]
                >=
                lower
            )
            &
            (
                df[feature]
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


        test_name = (
            f"P{int(qlow*100):02d}_"
            f"P{int(qhigh*100):02d}"
        )


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

                "Feature":
                    feature,

                "Test":
                    test_name,

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


        worst_pf = (
            min(year_pfs)
            if len(year_pfs) > 0
            else np.nan
        )


        overall_rows.append({

            "Feature":
                feature,

            "Test":
                test_name,

            "LowerValue":
                lower,

            "UpperValue":
                upper,

            "RetentionPct":
                len(subset)
                /
                len(df)
                *
                100,

            **stats,

            "WorstYearPF":
                worst_pf,

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
# ADD MIN YEAR TRADES FOR BENCHMARK
# ============================================================

benchmark_min_year_n = min(

    len(
        benchmark[
            benchmark["Year"]
            ==
            year
        ]
    )

    for year in MAIN_YEARS
)

overall_df.loc[
    overall_df["Test"]
    ==
    "BENCHMARK_0.35_0.90",
    "MinYearTrades"
] = benchmark_min_year_n


# ============================================================
# RANKING
#
# Main priority:
#
# 1. all 5 years PF > 1
# 2. worst-year PF
# 3. retention
# 4. PF
# ============================================================

ranking = overall_df.copy()


ranking["All5Years"] = (
    ranking["ProfitableYears"]
    ==
    5
)


ranking = ranking.sort_values(

    [
        "All5Years",
        "WorstYearPF",
        "RetentionPct",
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
# BEST PER FEATURE
# ============================================================

best_rows = []


for feature in overall_df[
    "Feature"
].unique():

    f = overall_df[
        overall_df["Feature"]
        ==
        feature
    ].copy()


    f["All5Years"] = (
        f["ProfitableYears"]
        ==
        5
    )


    f = f.sort_values(

        [
            "All5Years",
            "WorstYearPF",
            "RetentionPct",
            "ProfitFactor"
        ],

        ascending=[
            False,
            False,
            False,
            False
        ]
    )


    best_rows.append(
        f.iloc[0]
    )


best_df = pd.DataFrame(
    best_rows
)


best_df["All5Years"] = (
    best_df["ProfitableYears"]
    ==
    5
)


best_df = best_df.sort_values(

    [
        "All5Years",
        "WorstYearPF",
        "RetentionPct",
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
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_DIR
    /
    "macd_general_indicator_dataset.csv",
    index=False
)

overall_df.to_csv(
    OUTPUT_DIR
    /
    "all_indicator_tests.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "indicator_tests_year_by_year.csv",
    index=False
)

ranking.to_csv(
    OUTPUT_DIR
    /
    "indicator_ranking.csv",
    index=False
)

best_df.to_csv(
    OUTPUT_DIR
    /
    "best_candidate_per_indicator.csv",
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


display_cols = [

    "Feature",
    "Test",
    "LowerValue",
    "UpperValue",
    "N",
    "RetentionPct",
    "AveragePnL",
    "NetPoints",
    "ProfitFactor",
    "SQN",
    "MaxDrawdown",
    "WorstYearPF",
    "ProfitableYears",
    "MinYearTrades"
]


print()
print("=" * 210)
print("BENCHMARK")
print("=" * 210)

print(
    overall_df[
        overall_df["Test"]
        ==
        "BENCHMARK_0.35_0.90"
    ][display_cols]
    .round(4)
    .to_string(index=False)
)


print()
print("=" * 210)
print("BEST CANDIDATE PER INDICATOR")
print("=" * 210)

print(
    best_df[
        display_cols
    ]
    .round(4)
    .to_string(index=False)
)


print()
print("=" * 210)
print("TOP 30 GENERAL INDICATOR TESTS")
print("=" * 210)

print(
    ranking[
        display_cols
    ]
    .head(30)
    .round(4)
    .to_string(index=False)
)


print()
print("=" * 210)
print("DONE")
print("=" * 210)

print()
print("Output folder:")
print(OUTPUT_DIR)