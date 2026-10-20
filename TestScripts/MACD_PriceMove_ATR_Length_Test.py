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
    r"\MACD_PriceMove_ATR_Length_Test"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTHS = [
    5,
    7,
    10,
    14,
    20,
    30
]

HOLD_MINUTES = 240

MAIN_YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026
]


# ============================================================
# PERCENTILE BANDS
#
# These are deliberately broad.
#
# Example:
# P20-P60 means keep observations between
# the 20th and 60th percentile of PriceMoveATR
# for that ATR length.
# ============================================================

PERCENTILE_BANDS = [

    (10, 50),
    (10, 60),
    (20, 60),
    (20, 70),
    (20, 80),
    (30, 70),
    (30, 80),
    (30, 90),
    (40, 80),
    (40, 90)
]


# ============================================================
# LOAD 1-MINUTE DATA
# ============================================================

print()
print("=" * 180)
print("LOADING DATA")
print("=" * 180)

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
# TRUE RANGE
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


# ============================================================
# ATRs
#
# Wilder ATR = RMA
# pandas equivalent:
# ewm(alpha = 1 / length, adjust = False)
# ============================================================

for length in ATR_LENGTHS:

    df15[f"ATR{length}"] = tr.ewm(
        alpha=1.0 / length,
        adjust=False
    ).mean()


# ============================================================
# PRICE MOVE
# ============================================================

df15["PriceMovePoints"] = (
    df15["Close"]
    -
    df15["Close"].shift(1)
)

for length in ATR_LENGTHS:

    df15[f"PriceMoveATR{length}"] = (
        df15["PriceMovePoints"]
        /
        df15[f"ATR{length}"]
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
# BUILD RAW BULLISH CROSS DATASET
# ============================================================

print()
print("=" * 180)
print("BUILDING BULLISH CROSS DATASET")
print("=" * 180)

signals = df15[
    df15["BullCross"]
].copy()

rows = []


for signal_time in signals.index:

    trade = build_trade(
        signal_time
    )

    if trade is None:
        continue


    row = df15.loc[
        signal_time
    ]


    result = {

        "SignalTime":
            signal_time,

        "Year":
            signal_time.year,

        "Hour":
            signal_time.hour,

        "PriceMovePoints":
            row[
                "PriceMovePoints"
            ],

        **trade
    }


    for length in ATR_LENGTHS:

        result[
            f"ATR{length}"
        ] = row[
            f"ATR{length}"
        ]

        result[
            f"PriceMoveATR{length}"
        ] = row[
            f"PriceMoveATR{length}"
        ]


    rows.append(
        result
    )


df = pd.DataFrame(
    rows
)


print(
    "Completed bullish crosses:",
    len(df)
)


# ============================================================
# ORIGINAL ATR14 BENCHMARK
# ============================================================

benchmark = df[
    (
        df["PriceMoveATR14"]
        >= 0.35
    )
    &
    (
        df["PriceMoveATR14"]
        <= 0.90
    )
].copy()


benchmark_stats = calculate_stats(
    benchmark
)


benchmark_year_rows = []


for year in MAIN_YEARS:

    yd = benchmark[
        benchmark["Year"]
        ==
        year
    ]

    ys = calculate_stats(
        yd
    )

    benchmark_year_rows.append({

        "Year":
            year,

        **ys
    })


benchmark_year_df = pd.DataFrame(
    benchmark_year_rows
)


# ============================================================
# PERCENTILE BAND TEST
# ============================================================

result_rows = []
year_rows = []


for length in ATR_LENGTHS:

    feature = f"PriceMoveATR{length}"

    valid = df[
        df[feature]
        .notna()
    ].copy()


    print()
    print(
        f"Testing ATR{length}..."
    )


    for low_pct, high_pct in PERCENTILE_BANDS:

        low_value = valid[
            feature
        ].quantile(
            low_pct / 100.0
        )

        high_value = valid[
            feature
        ].quantile(
            high_pct / 100.0
        )


        subset = valid[
            (
                valid[feature]
                >=
                low_value
            )
            &
            (
                valid[feature]
                <=
                high_value
            )
        ].copy()


        stats = calculate_stats(
            subset
        )


        yearly_pfs = []
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

                "ATRLength":
                    length,

                "PercentileBand":
                    f"P{low_pct}-P{high_pct}",

                "LowValue":
                    low_value,

                "HighValue":
                    high_value,

                "Year":
                    year,

                **ys
            })


            if np.isfinite(
                ys["ProfitFactor"]
            ):

                yearly_pfs.append(
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


        result_rows.append({

            "ATRLength":
                length,

            "PercentileBand":
                f"P{low_pct}-P{high_pct}",

            "LowPercentile":
                low_pct,

            "HighPercentile":
                high_pct,

            "LowValue":
                low_value,

            "HighValue":
                high_value,

            "N":
                stats["N"],

            "RetentionPct":
                (
                    stats["N"]
                    /
                    len(valid)
                    *
                    100
                ),

            "WinRatePct":
                stats["WinRatePct"],

            "NetPoints":
                stats["NetPoints"],

            "AveragePnL":
                stats["AveragePnL"],

            "MedianPnL":
                stats["MedianPnL"],

            "ProfitFactor":
                stats["ProfitFactor"],

            "SQN":
                stats["SQN"],

            "MaxDrawdown":
                stats["MaxDrawdown"],

            "WorstYearPF":
                (
                    min(yearly_pfs)
                    if yearly_pfs
                    else np.nan
                ),

            "ProfitableYears":
                profitable_years,

            "MinYearTrades":
                min_year_trades
        })


results_df = pd.DataFrame(
    result_rows
)

years_df = pd.DataFrame(
    year_rows
)


# ============================================================
# BEST CANDIDATE PER ATR LENGTH
#
# Stability first.
# ============================================================

best_rows = []


for length in ATR_LENGTHS:

    temp = results_df[
        results_df["ATRLength"]
        ==
        length
    ].copy()


    temp = temp.sort_values(
        [
            "ProfitableYears",
            "WorstYearPF",
            "ProfitFactor",
            "N"
        ],
        ascending=[
            False,
            False,
            False,
            False
        ]
    )


    best_rows.append(
        temp.iloc[0]
    )


best_df = pd.DataFrame(
    best_rows
)


# ============================================================
# GLOBAL RANKING
# ============================================================

ranking_df = results_df.copy()

ranking_df["All5Years"] = (
    ranking_df["ProfitableYears"]
    ==
    5
)

ranking_df = ranking_df.sort_values(
    [
        "All5Years",
        "WorstYearPF",
        "ProfitFactor",
        "N"
    ],
    ascending=[
        False,
        False,
        False,
        False
    ]
)


# ============================================================
# SAME NUMERIC RANGE TEST
#
# Secondary diagnostic:
# apply 0.35-0.90 to all ATR lengths
# just to see how sensitive the original range is.
# ============================================================

same_range_rows = []


for length in ATR_LENGTHS:

    feature = f"PriceMoveATR{length}"

    subset = df[
        (
            df[feature]
            >= 0.35
        )
        &
        (
            df[feature]
            <= 0.90
        )
    ].copy()


    stats = calculate_stats(
        subset
    )


    yearly_pfs = []
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

            yearly_pfs.append(
                ys["ProfitFactor"]
            )

            if (
                ys["ProfitFactor"]
                >
                1
            ):
                profitable_years += 1


    same_range_rows.append({

        "ATRLength":
            length,

        "Range":
            "0.35-0.90",

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
                min(yearly_pfs)
                if yearly_pfs
                else np.nan
            ),

        "ProfitableYears":
            profitable_years
    })


same_range_df = pd.DataFrame(
    same_range_rows
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_DIR
    /
    "all_bullish_crosses_with_atrs.csv",
    index=False
)

results_df.to_csv(
    OUTPUT_DIR
    /
    "atr_length_percentile_band_results.csv",
    index=False
)

years_df.to_csv(
    OUTPUT_DIR
    /
    "atr_length_year_by_year.csv",
    index=False
)

best_df.to_csv(
    OUTPUT_DIR
    /
    "best_candidate_per_atr_length.csv",
    index=False
)

ranking_df.to_csv(
    OUTPUT_DIR
    /
    "global_ranking.csv",
    index=False
)

same_range_df.to_csv(
    OUTPUT_DIR
    /
    "same_035_090_range_all_atrs.csv",
    index=False
)

benchmark_year_df.to_csv(
    OUTPUT_DIR
    /
    "atr14_035_090_year_by_year.csv",
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
    500
)

pd.set_option(
    "display.max_rows",
    1000
)


print()
print("=" * 220)
print("ORIGINAL ATR14 0.35-0.90 BENCHMARK")
print("=" * 220)

print(
    pd.DataFrame(
        [benchmark_stats]
    )
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("ATR14 0.35-0.90 YEAR BY YEAR")
print("=" * 220)

print(
    benchmark_year_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("BEST PERCENTILE BAND PER ATR LENGTH")
print("=" * 220)

print(
    best_df[
        [
            "ATRLength",
            "PercentileBand",
            "LowValue",
            "HighValue",
            "N",
            "AveragePnL",
            "ProfitFactor",
            "SQN",
            "MaxDrawdown",
            "WorstYearPF",
            "ProfitableYears",
            "MinYearTrades"
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("SAME NUMERIC RANGE 0.35-0.90 ACROSS ATR LENGTHS")
print("=" * 220)

print(
    same_range_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("TOP 30 PERCENTILE-BAND RESULTS")
print("=" * 220)

print(
    ranking_df[
        [
            "ATRLength",
            "PercentileBand",
            "LowValue",
            "HighValue",
            "N",
            "RetentionPct",
            "AveragePnL",
            "ProfitFactor",
            "SQN",
            "MaxDrawdown",
            "WorstYearPF",
            "ProfitableYears",
            "MinYearTrades"
        ]
    ]
    .head(30)
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