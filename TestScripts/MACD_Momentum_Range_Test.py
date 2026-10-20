import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\TestScripts"
    r"\MACD_Momentum_Strength"
    r"\macd_momentum_feature_dataset.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\TestScripts"
    r"\MACD_Momentum_Range_Test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# YEARS USED FOR STABILITY TEST
# ============================================================

STABILITY_YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026
]


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 140)
print("LOADING MACD MOMENTUM DATASET")
print("=" * 140)

df = pd.read_csv(
    INPUT_FILE
)

df["SignalTime"] = pd.to_datetime(
    df["SignalTime"]
)

print("Trades:", len(df))
print("Start:", df["SignalTime"].min())
print("End:  ", df["SignalTime"].max())


# ============================================================
# PERFORMANCE FUNCTIONS
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

    return gross_profit / gross_loss


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
# RAW BASELINE
# ============================================================

raw_stats = calculate_stats(
    df
)

RAW_N = raw_stats[
    "N"
]

print()
print("=" * 140)
print("RAW MACD BASELINE")
print("=" * 140)

print(
    f"N:          {raw_stats['N']}"
)

print(
    f"Win rate:   {raw_stats['WinRatePct']:.4f}%"
)

print(
    f"Avg PnL:    {raw_stats['AveragePnL']:.4f}"
)

print(
    f"Net:        {raw_stats['NetPoints']:.2f}"
)

print(
    f"PF:         {raw_stats['ProfitFactor']:.4f}"
)

print(
    f"SQN:        {raw_stats['SQN']:.4f}"
)

print(
    f"Max DD:     {raw_stats['MaxDrawdown']:.2f}"
)


# ============================================================
# FILTER DEFINITIONS
#
# IMPORTANT:
#
# Each filter is tested INDEPENDENTLY.
#
# We are NOT combining momentum variables yet.
# ============================================================

tests = []


# ============================================================
# 1. CLOSE LOCATION
#
# 0 = close at candle low
# 1 = close at candle high
#
# Previous Q4:
# approximately 0.868 - 0.933
#
# Test broad neighboring ranges.
# ============================================================

close_location_tests = [

    (
        "CloseLocation >= 0.70",
        df["CloseLocation"] >= 0.70
    ),

    (
        "CloseLocation >= 0.75",
        df["CloseLocation"] >= 0.75
    ),

    (
        "CloseLocation >= 0.80",
        df["CloseLocation"] >= 0.80
    ),

    (
        "CloseLocation >= 0.85",
        df["CloseLocation"] >= 0.85
    ),

    (
        "CloseLocation >= 0.90",
        df["CloseLocation"] >= 0.90
    ),

    (
        "CloseLocation 0.70-0.95",
        (
            (df["CloseLocation"] >= 0.70)
            &
            (df["CloseLocation"] <= 0.95)
        )
    ),

    (
        "CloseLocation 0.75-0.95",
        (
            (df["CloseLocation"] >= 0.75)
            &
            (df["CloseLocation"] <= 0.95)
        )
    ),

    (
        "CloseLocation 0.80-0.95",
        (
            (df["CloseLocation"] >= 0.80)
            &
            (df["CloseLocation"] <= 0.95)
        )
    ),

    (
        "CloseLocation 0.85-0.95",
        (
            (df["CloseLocation"] >= 0.85)
            &
            (df["CloseLocation"] <= 0.95)
        )
    )
]


for name, mask in close_location_tests:

    tests.append(
        (
            "CloseLocation",
            name,
            mask
        )
    )


# ============================================================
# 2. PRICE MOVE 15 / ATR
#
# Previous result:
#
# Q2:
# roughly 0.35 - 0.63 ATR
#
# Q3:
# roughly 0.63 - 0.90 ATR
#
# Both were good.
#
# So now test BROAD ranges around them.
# ============================================================

price15_tests = [

    (
        "PriceMove15ATR 0.30-0.90",
        (
            (df["PriceMove15ATR"] >= 0.30)
            &
            (df["PriceMove15ATR"] <= 0.90)
        )
    ),

    (
        "PriceMove15ATR 0.35-0.90",
        (
            (df["PriceMove15ATR"] >= 0.35)
            &
            (df["PriceMove15ATR"] <= 0.90)
        )
    ),

    (
        "PriceMove15ATR 0.40-0.90",
        (
            (df["PriceMove15ATR"] >= 0.40)
            &
            (df["PriceMove15ATR"] <= 0.90)
        )
    ),

    (
        "PriceMove15ATR 0.40-1.00",
        (
            (df["PriceMove15ATR"] >= 0.40)
            &
            (df["PriceMove15ATR"] <= 1.00)
        )
    ),

    (
        "PriceMove15ATR 0.40-1.10",
        (
            (df["PriceMove15ATR"] >= 0.40)
            &
            (df["PriceMove15ATR"] <= 1.10)
        )
    ),

    (
        "PriceMove15ATR 0.50-1.00",
        (
            (df["PriceMove15ATR"] >= 0.50)
            &
            (df["PriceMove15ATR"] <= 1.00)
        )
    ),

    (
        "PriceMove15ATR 0.50-1.10",
        (
            (df["PriceMove15ATR"] >= 0.50)
            &
            (df["PriceMove15ATR"] <= 1.10)
        )
    ),

    (
        "PriceMove15ATR 0.60-1.00",
        (
            (df["PriceMove15ATR"] >= 0.60)
            &
            (df["PriceMove15ATR"] <= 1.00)
        )
    ),

    (
        "PriceMove15ATR 0.60-1.10",
        (
            (df["PriceMove15ATR"] >= 0.60)
            &
            (df["PriceMove15ATR"] <= 1.10)
        )
    ),

    (
        "PriceMove15ATR 0.35-1.20",
        (
            (df["PriceMove15ATR"] >= 0.35)
            &
            (df["PriceMove15ATR"] <= 1.20)
        )
    )
]


for name, mask in price15_tests:

    tests.append(
        (
            "PriceMove15ATR",
            name,
            mask
        )
    )


# ============================================================
# 3. 3-BAR HISTOGRAM SLOPE / ATR
#
# Previous Q1:
#
# approximately -0.2055 to +0.0192
#
# It was surprisingly strong.
#
# Test broad upper limits instead of picking exact quintile.
# ============================================================

hist_slope_tests = [

    (
        "HistSlope3Norm <= 0.00",
        df["HistSlope3Norm"] <= 0.00
    ),

    (
        "HistSlope3Norm <= 0.02",
        df["HistSlope3Norm"] <= 0.02
    ),

    (
        "HistSlope3Norm <= 0.04",
        df["HistSlope3Norm"] <= 0.04
    ),

    (
        "HistSlope3Norm <= 0.06",
        df["HistSlope3Norm"] <= 0.06
    ),

    (
        "HistSlope3Norm <= 0.08",
        df["HistSlope3Norm"] <= 0.08
    ),

    (
        "HistSlope3Norm <= 0.10",
        df["HistSlope3Norm"] <= 0.10
    ),

    (
        "HistSlope3Norm -0.10 to 0.02",
        (
            (df["HistSlope3Norm"] >= -0.10)
            &
            (df["HistSlope3Norm"] <= 0.02)
        )
    ),

    (
        "HistSlope3Norm -0.05 to 0.02",
        (
            (df["HistSlope3Norm"] >= -0.05)
            &
            (df["HistSlope3Norm"] <= 0.02)
        )
    ),

    (
        "HistSlope3Norm -0.05 to 0.04",
        (
            (df["HistSlope3Norm"] >= -0.05)
            &
            (df["HistSlope3Norm"] <= 0.04)
        )
    )
]


for name, mask in hist_slope_tests:

    tests.append(
        (
            "HistSlope3Norm",
            name,
            mask
        )
    )


# ============================================================
# RUN TESTS
# ============================================================

overall_rows = []
year_rows = []


for feature, name, mask in tests:

    filtered = df[
        mask
    ].copy()


    stats = calculate_stats(
        filtered
    )


    retention = (
        len(filtered)
        /
        RAW_N
        *
        100
    )


    overall_rows.append({

        "Feature":
            feature,

        "Test":
            name,

        "N":
            stats["N"],

        "RetentionPct":
            retention,

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
            stats["MaxDrawdown"]
    })


    # --------------------------------------------------------
    # YEAR-BY-YEAR
    # --------------------------------------------------------

    for year in STABILITY_YEARS:

        yd = filtered[
            filtered["Year"]
            ==
            year
        ]


        year_stats = calculate_stats(
            yd
        )


        year_rows.append({

            "Feature":
                feature,

            "Test":
                name,

            "Year":
                year,

            **year_stats
        })


overall_df = pd.DataFrame(
    overall_rows
)

year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# STABILITY SUMMARY
# ============================================================

stability_rows = []


for _, result in overall_df.iterrows():

    feature = result[
        "Feature"
    ]

    test = result[
        "Test"
    ]


    y = year_df[
        (year_df["Feature"] == feature)
        &
        (year_df["Test"] == test)
    ]


    pf = (
        y["ProfitFactor"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )


    stability_rows.append({

        "Feature":
            feature,

        "Test":
            test,

        "N":
            result["N"],

        "RetentionPct":
            result["RetentionPct"],

        "WinRatePct":
            result["WinRatePct"],

        "AveragePnL":
            result["AveragePnL"],

        "NetPoints":
            result["NetPoints"],

        "ProfitFactor":
            result["ProfitFactor"],

        "SQN":
            result["SQN"],

        "MaxDrawdown":
            result["MaxDrawdown"],

        "WorstYearPF":
            pf.min()
            if len(pf) > 0
            else np.nan,

        "MeanYearPF":
            pf.mean()
            if len(pf) > 0
            else np.nan,

        "ProfitableYears":
            int(
                np.sum(
                    y["NetPoints"] > 0
                )
            ),

        "PFAbove1Years":
            int(
                np.sum(
                    y["ProfitFactor"] > 1
                )
            ),

        "MinYearTrades":
            int(
                y["N"].min()
            ),

        "WorstYearAvgPnL":
            y["AveragePnL"].min()
    })


stability_df = pd.DataFrame(
    stability_rows
)


# ============================================================
# ADD RAW BASELINE BY YEAR
# ============================================================

baseline_rows = []


for year in STABILITY_YEARS:

    yd = df[
        df["Year"]
        ==
        year
    ]


    stats = calculate_stats(
        yd
    )


    baseline_rows.append({

        "Year":
            year,

        **stats
    })


baseline_year_df = pd.DataFrame(
    baseline_rows
)


# ============================================================
# SAVE FILES
# ============================================================

overall_df.to_csv(
    OUTPUT_DIR
    /
    "momentum_range_overall.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "momentum_range_year_by_year.csv",
    index=False
)

stability_df.to_csv(
    OUTPUT_DIR
    /
    "momentum_range_stability.csv",
    index=False
)

baseline_year_df.to_csv(
    OUTPUT_DIR
    /
    "raw_baseline_year_by_year.csv",
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
print("ALL RANGE TESTS — SORTED BY WORST YEAR PF")
print("=" * 180)

print(
    stability_df
    .sort_values(
        [
            "WorstYearPF",
            "ProfitFactor"
        ],
        ascending=[
            False,
            False
        ]
    )
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 180)
print("CLOSE LOCATION TESTS")
print("=" * 180)

print(
    stability_df[
        stability_df["Feature"]
        ==
        "CloseLocation"
    ]
    .sort_values(
        "WorstYearPF",
        ascending=False
    )
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 180)
print("PRICE MOVE 15 / ATR TESTS")
print("=" * 180)

print(
    stability_df[
        stability_df["Feature"]
        ==
        "PriceMove15ATR"
    ]
    .sort_values(
        "WorstYearPF",
        ascending=False
    )
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 180)
print("HISTOGRAM 3-BAR SLOPE TESTS")
print("=" * 180)

print(
    stability_df[
        stability_df["Feature"]
        ==
        "HistSlope3Norm"
    ]
    .sort_values(
        "WorstYearPF",
        ascending=False
    )
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 180)
print("YEAR-BY-YEAR RESULTS")
print("=" * 180)

print(
    year_df
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