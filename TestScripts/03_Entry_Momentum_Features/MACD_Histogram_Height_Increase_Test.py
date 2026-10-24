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
    r"\MACD_Histogram_Height_Increase_Test"
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
# BROAD PERCENTILE TESTS
# ============================================================

PERCENTILE_TESTS = [

    (0.00, 0.40),
    (0.00, 0.50),
    (0.00, 0.60),

    (0.10, 0.50),
    (0.10, 0.60),
    (0.10, 0.70),

    (0.20, 0.60),
    (0.20, 0.70),
    (0.20, 0.80),

    (0.30, 0.70),
    (0.30, 0.80),
    (0.30, 0.90),

    (0.40, 0.80),
    (0.40, 0.90),

    (0.50, 1.00),
    (0.60, 1.00)
]


# ============================================================
# LOAD 1-MIN DATA
# ============================================================

print()
print("=" * 150)
print("LOADING DATA")
print("=" * 150)

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

    gross_profit = pnl[
        pnl > 0
    ].sum()

    gross_loss = -pnl[
        pnl < 0
    ].sum()


    if gross_loss == 0:

        if gross_profit > 0:
            return np.inf

        return np.nan


    return (
        gross_profit
        /
        gross_loss
    )


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
        np.sqrt(
            len(pnl)
        )
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

            "N":
                0,

            "WinRatePct":
                np.nan,

            "NetPoints":
                np.nan,

            "AveragePnL":
                np.nan,

            "MedianPnL":
                np.nan,

            "ProfitFactor":
                np.nan,

            "SQN":
                np.nan,

            "MaxDrawdown":
                np.nan
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
# BUILD MACD DATASET
# ============================================================

signals = df15[
    df15["BullCross"]
    &
    df15["TimeValid"]
].copy()


print()
print("=" * 150)
print("BUILDING MACD DATASET")
print("=" * 150)

print(
    "Raw MACD crosses:",
    len(signals)
)


rows = []


for signal_time in signals.index:

    pos = df15.index.get_loc(
        signal_time
    )


    if pos < 1:
        continue


    trade = build_trade(
        signal_time
    )


    if trade is None:
        continue


    atr = float(
        df15.iloc[
            pos
        ]["ATR14"]
    )


    if (
        not np.isfinite(atr)
        or
        atr <= 0
    ):
        continue


    hist_now = float(
        df15.iloc[
            pos
        ]["Hist"]
    )

    hist_prev = float(
        df15.iloc[
            pos - 1
        ]["Hist"]
    )


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


    # ========================================================
    # FEATURE 1:
    # HEIGHT OF FIRST POSITIVE HISTOGRAM BAR
    # ========================================================

    hist_height_norm = (
        hist_now
        /
        atr
    )


    # ========================================================
    # FEATURE 2:
    # INCREASE FROM PREVIOUS HISTOGRAM BAR
    #
    # Previous bar is normally negative or zero
    # because this is a bullish zero-cross.
    # ========================================================

    hist_increase_norm = (
        hist_now
        -
        hist_prev
    ) / atr


    # ========================================================
    # PRICE MOVE BENCHMARK
    # ========================================================

    price_move_15_atr = (
        close_now
        -
        close_prev
    ) / atr


    rows.append({

        "SignalTime":
            signal_time,

        "Year":
            signal_time.year,

        "Hist":
            hist_now,

        "HistPrev":
            hist_prev,

        "ATR14":
            atr,

        "HistHeightNorm":
            hist_height_norm,

        "HistIncreaseNorm":
            hist_increase_norm,

        "PriceMove15ATR":
            price_move_15_atr,

        **trade
    })


df = pd.DataFrame(
    rows
)


print(
    "Completed MACD trades:",
    len(df)
)


# ============================================================
# RAW BASELINE
# ============================================================

raw_stats = calculate_stats(
    df
)


# ============================================================
# PRICE-MOVE BENCHMARK
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
# TEST FEATURES SEPARATELY
# ============================================================

FEATURES = [

    "HistHeightNorm",
    "HistIncreaseNorm"
]


overall_rows = []
year_rows = []


# ============================================================
# BENCHMARK ROW
# ============================================================

benchmark_pfs = []
benchmark_min_n = np.inf


for year in MAIN_YEARS:

    yd = benchmark[
        benchmark["Year"]
        ==
        year
    ]


    ys = calculate_stats(
        yd
    )


    benchmark_pfs.append(
        ys["ProfitFactor"]
    )


    benchmark_min_n = min(
        benchmark_min_n,
        ys["N"]
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


overall_rows.append({

    "Feature":
        "PriceMove15ATR",

    "Test":
        "BENCHMARK_0.35_0.90",

    "LowerValue":
        0.35,

    "UpperValue":
        0.90,

    "N":
        benchmark_stats["N"],

    "RetentionPct":
        len(benchmark)
        /
        len(df)
        *
        100,

    "AveragePnL":
        benchmark_stats[
            "AveragePnL"
        ],

    "NetPoints":
        benchmark_stats[
            "NetPoints"
        ],

    "ProfitFactor":
        benchmark_stats[
            "ProfitFactor"
        ],

    "SQN":
        benchmark_stats[
            "SQN"
        ],

    "MaxDrawdown":
        benchmark_stats[
            "MaxDrawdown"
        ],

    "WorstYearPF":
        min(
            benchmark_pfs
        ),

    "ProfitableYears":
        sum(
            x > 1
            for x in benchmark_pfs
        ),

    "MinYearTrades":
        benchmark_min_n
})


# ============================================================
# HISTOGRAM TESTS
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


        test_name = (
            f"P{int(qlow * 100):02d}_"
            f"P{int(qhigh * 100):02d}"
        )


        year_pfs = []
        min_year_n = np.inf


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


            min_year_n = min(
                min_year_n,
                ys["N"]
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

            "N":
                stats["N"],

            "RetentionPct":
                len(subset)
                /
                len(df)
                *
                100,

            "AveragePnL":
                stats[
                    "AveragePnL"
                ],

            "NetPoints":
                stats[
                    "NetPoints"
                ],

            "ProfitFactor":
                stats[
                    "ProfitFactor"
                ],

            "SQN":
                stats[
                    "SQN"
                ],

            "MaxDrawdown":
                stats[
                    "MaxDrawdown"
                ],

            "WorstYearPF":
                min(
                    year_pfs
                )
                if year_pfs
                else np.nan,

            "ProfitableYears":
                sum(
                    x > 1
                    for x in year_pfs
                ),

            "MinYearTrades":
                min_year_n
        })


overall_df = pd.DataFrame(
    overall_rows
)

year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# CORRELATIONS
#
# Useful because you specifically want to know how similar
# these are to PriceMove15ATR.
# ============================================================

correlation_df = df[
    [
        "PriceMove15ATR",
        "HistHeightNorm",
        "HistIncreaseNorm"
    ]
].corr()


# ============================================================
# RANK RESULTS
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
        "ProfitFactor",
        "RetentionPct"
    ],

    ascending=[
        False,
        False,
        False,
        False
    ]
)


# ============================================================
# BEST TEST PER FEATURE
# ============================================================

best_rows = []


for feature in [
    "PriceMove15ATR",
    "HistHeightNorm",
    "HistIncreaseNorm"
]:

    f = ranking[
        ranking["Feature"]
        ==
        feature
    ]


    if len(f) > 0:

        best_rows.append(
            f.iloc[0]
        )


best_df = pd.DataFrame(
    best_rows
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_DIR
    /
    "macd_histogram_feature_dataset.csv",
    index=False
)

overall_df.to_csv(
    OUTPUT_DIR
    /
    "all_histogram_tests.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "histogram_tests_year_by_year.csv",
    index=False
)

ranking.to_csv(
    OUTPUT_DIR
    /
    "histogram_test_ranking.csv",
    index=False
)

best_df.to_csv(
    OUTPUT_DIR
    /
    "best_histogram_candidates.csv",
    index=False
)

correlation_df.to_csv(
    OUTPUT_DIR
    /
    "feature_correlations.csv"
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
    200
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
print("=" * 190)
print("RAW MACD")
print("=" * 190)

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
print("=" * 190)
print("FEATURE CORRELATIONS")
print("=" * 190)

print(
    correlation_df
    .round(4)
    .to_string()
)


print()
print("=" * 190)
print("BEST RESULT PER FEATURE")
print("=" * 190)

print(
    best_df[
        display_cols
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 190)
print("TOP 20 TESTS")
print("=" * 190)

print(
    ranking[
        display_cols
    ]
    .head(20)
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 190)
print("YEAR-BY-YEAR FOR BEST HISTOGRAM TESTS")
print("=" * 190)


for _, best in best_df.iterrows():

    feature = best[
        "Feature"
    ]

    test = best[
        "Test"
    ]


    print()
    print(
        feature,
        test
    )


    temp = year_df[
        (
            year_df["Feature"]
            ==
            feature
        )
        &
        (
            year_df["Test"]
            ==
            test
        )
    ]


    print(
        temp
        .round(4)
        .to_string(
            index=False
        )
    )


print()
print("=" * 190)
print("DONE")
print("=" * 190)

print()
print("Output folder:")
print(OUTPUT_DIR)