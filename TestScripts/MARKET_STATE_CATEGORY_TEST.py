import numpy as np
import pandas as pd

# ==================================================================================================
# SETTINGS
# ==================================================================================================

CSV_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"

OUTPUT_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\MARKET_STATE_CATEGORY_TEST.csv"

YEAR_OUTPUT_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\MARKET_STATE_CATEGORY_YEAR_TEST.csv"

SIGNAL_OUTPUT_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\MARKET_STATE_SIGNALS.csv"

# MACD
MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

# State calculations
BLOCK_LENGTH = 20

ATR_LENGTH = 14
RANK_LENGTH = 200

BB_LENGTH = 20
BB_MULT = 2.0

VOL_VARIATION_LENGTH = 20

VOLUME_LENGTH = 20

# Forward return
HOLD_MINUTES = 210

# Analysis years
START_YEAR = 2022
END_YEAR = 2026


# ==================================================================================================
# HELPERS
# ==================================================================================================

def section(title):
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def profit_factor(x):
    x = pd.Series(x).dropna()

    gross_profit = x[x > 0].sum()
    gross_loss = -x[x < 0].sum()

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else np.nan

    return gross_profit / gross_loss


def performance_stats(df, return_col="Return210"):
    if len(df) == 0:
        return {
            "N": 0,
            "Avg": np.nan,
            "PF": np.nan,
            "WinPct": np.nan,
            "TotalPts": np.nan
        }

    r = df[return_col].dropna()

    return {
        "N": len(r),
        "Avg": r.mean(),
        "PF": profit_factor(r),
        "WinPct": (r > 0).mean() * 100.0,
        "TotalPts": r.sum()
    }


def percentile_rank_rolling(series, length):
    """
    Pine-like rolling percentile rank.

    Returns approximately 0-100:
    0   = near bottom of recent distribution
    100 = near top
    """

    def rank_last(window):
        if len(window) < length:
            return np.nan

        current = window[-1]

        valid = window[~np.isnan(window)]

        if len(valid) < length:
            return np.nan

        less = np.sum(valid < current)
        equal = np.sum(valid == current)

        # Approximate percentile rank including ties
        rank = (
            less +
            0.5 * max(equal - 1, 0)
        ) / max(len(valid) - 1, 1)

        return rank * 100.0

    return series.rolling(
        length,
        min_periods=length
    ).apply(
        rank_last,
        raw=True
    )


def level_1_to_5(percentile):
    """
    Same five-level coding used in Pine:

    0-20   -> 1
    20-40  -> 2
    40-60  -> 3
    60-80  -> 4
    80-100 -> 5
    """

    result = pd.Series(
        np.nan,
        index=percentile.index
    )

    result.loc[percentile < 20] = 1

    result.loc[
        (percentile >= 20) &
        (percentile < 40)
    ] = 2

    result.loc[
        (percentile >= 40) &
        (percentile < 60)
    ] = 3

    result.loc[
        (percentile >= 60) &
        (percentile < 80)
    ] = 4

    result.loc[
        percentile >= 80
    ] = 5

    return result.astype("Int64")


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

df1 = (
    df1
    .set_index(time_col)
    .sort_index()
)

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

required = [
    "Open",
    "High",
    "Low",
    "Close"
]

for c in required:
    if c not in df1.columns:
        raise ValueError(f"Missing required column: {c}")

print(f"1-minute rows: {len(df1):,}")
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
    subset=[
        "Open",
        "High",
        "Low",
        "Close"
    ]
)

print(f"15-minute bars: {len(df15):,}")


# ==================================================================================================
# MACD
# ==================================================================================================

section("MACD")

ema_fast = df15["Close"].ewm(
    span=MACD_FAST,
    adjust=False
).mean()

ema_slow = df15["Close"].ewm(
    span=MACD_SLOW,
    adjust=False
).mean()

macd_line = ema_fast - ema_slow

macd_signal_line = macd_line.ewm(
    span=MACD_SIGNAL,
    adjust=False
).mean()

df15["MACDHist"] = (
    macd_line -
    macd_signal_line
)

df15["BullCross"] = (
    (df15["MACDHist"].shift(1) <= 0) &
    (df15["MACDHist"] > 0)
)


# ==================================================================================================
# ATR
# ==================================================================================================

prev_close = df15["Close"].shift(1)

true_range = pd.concat(
    [
        df15["High"] - df15["Low"],
        (df15["High"] - prev_close).abs(),
        (df15["Low"] - prev_close).abs()
    ],
    axis=1
).max(axis=1)

# Wilder ATR
df15["ATR"] = true_range.ewm(
    alpha=1 / ATR_LENGTH,
    adjust=False
).mean()


# ==================================================================================================
# 1. VOLATILITY LEVEL
# ==================================================================================================

section("CALCULATING MARKET STATE FEATURES")

df15["ATRPercentile"] = percentile_rank_rolling(
    df15["ATR"],
    RANK_LENGTH
)

df15["VolCode"] = level_1_to_5(
    df15["ATRPercentile"]
)


# ==================================================================================================
# 2. VOLATILITY VARIATION
# ==================================================================================================

df15["VolVariationRaw"] = df15[
    "ATRPercentile"
].rolling(
    VOL_VARIATION_LENGTH
).std(
    ddof=0
)

df15["VolVariationPercentile"] = percentile_rank_rolling(
    df15["VolVariationRaw"],
    RANK_LENGTH
)

df15["VolVariationCode"] = level_1_to_5(
    df15["VolVariationPercentile"]
)


# ==================================================================================================
# 3. BOLLINGER WIDTH
# ==================================================================================================

df15["BBMid"] = df15[
    "Close"
].rolling(
    BB_LENGTH
).mean()

bb_std = df15[
    "Close"
].rolling(
    BB_LENGTH
).std(
    ddof=0
)

df15["BBUpper"] = (
    df15["BBMid"] +
    BB_MULT * bb_std
)

df15["BBLower"] = (
    df15["BBMid"] -
    BB_MULT * bb_std
)

df15["BBWidth"] = (
    df15["BBUpper"] -
    df15["BBLower"]
)

df15["BBWidthPercentile"] = percentile_rank_rolling(
    df15["BBWidth"],
    RANK_LENGTH
)

df15["BBWidthCode"] = level_1_to_5(
    df15["BBWidthPercentile"]
)


# ==================================================================================================
# 4. TREND
# ==================================================================================================

df15["DirectionScore"] = (
    (
        df15["Close"] -
        df15["Close"].shift(BLOCK_LENGTH)
    ) /
    df15["ATR"]
)

df15["AbsTrendScore"] = df15[
    "DirectionScore"
].abs()

df15["TrendPercentile"] = percentile_rank_rolling(
    df15["AbsTrendScore"],
    RANK_LENGTH
)

df15["TrendCode"] = level_1_to_5(
    df15["TrendPercentile"]
)

df15["TrendDirection"] = np.where(
    df15["DirectionScore"] > 0,
    "UP",
    np.where(
        df15["DirectionScore"] < 0,
        "DOWN",
        "SIDEWAYS"
    )
)

df15["TrendState"] = (
    df15["TrendDirection"] +
    "_" +
    df15["TrendCode"].astype(str)
)


# ==================================================================================================
# 5. VOLUME
# ==================================================================================================

if "Volume" in df15.columns:

    df15["AvgVolume"] = df15[
        "Volume"
    ].rolling(
        VOLUME_LENGTH
    ).mean()

    df15["VolumePercentile"] = percentile_rank_rolling(
        df15["AvgVolume"],
        RANK_LENGTH
    )

    df15["VolumeCode"] = level_1_to_5(
        df15["VolumePercentile"]
    )

else:

    print("WARNING: No Volume column found.")

    df15["VolumePercentile"] = np.nan
    df15["VolumeCode"] = pd.Series(
        pd.NA,
        index=df15.index,
        dtype="Int64"
    )


# ==================================================================================================
# BUILD SIGNAL TABLE
# ==================================================================================================

signals = df15[
    df15["BullCross"]
].copy()

signals = signals[
    (signals.index.year >= START_YEAR) &
    (signals.index.year <= END_YEAR)
]

signals = signals.dropna(
    subset=[
        "TrendCode",
        "VolCode",
        "VolVariationCode",
        "BBWidthCode"
    ]
)

print()
print(f"Eligible bullish MACD crosses: {len(signals):,}")


# ==================================================================================================
# 210-MINUTE FORWARD RETURN
# ==================================================================================================

section("CALCULATING 210-MINUTE RETURNS")

times_1m = df1.index.values

opens_1m = df1[
    "Open"
].to_numpy()

entry_times = []
entry_prices = []

exit_times = []
exit_prices = []

returns = []

valid_signal_times = []

for signal_time in signals.index:

    # First 1-minute Open strictly AFTER the completed 15-minute signal bar
    entry_pos = np.searchsorted(
        times_1m,
        np.datetime64(signal_time),
        side="right"
    )

    if entry_pos >= len(df1):
        continue

    entry_time = df1.index[
        entry_pos
    ]

    entry_price = opens_1m[
        entry_pos
    ]

    target_exit_time = (
        entry_time +
        pd.Timedelta(
            minutes=HOLD_MINUTES
        )
    )

    exit_pos = np.searchsorted(
        times_1m,
        np.datetime64(target_exit_time),
        side="left"
    )

    if exit_pos >= len(df1):
        continue

    exit_time = df1.index[
        exit_pos
    ]

    # Reject large gaps
    if exit_time > target_exit_time + pd.Timedelta(minutes=2):
        continue

    exit_price = opens_1m[
        exit_pos
    ]

    ret = (
        exit_price -
        entry_price
    )

    valid_signal_times.append(
        signal_time
    )

    entry_times.append(
        entry_time
    )

    entry_prices.append(
        entry_price
    )

    exit_times.append(
        exit_time
    )

    exit_prices.append(
        exit_price
    )

    returns.append(
        ret
    )


signals = signals.loc[
    valid_signal_times
].copy()

signals["EntryTime"] = entry_times
signals["EntryPrice"] = entry_prices

signals["ExitTime"] = exit_times
signals["ExitPrice"] = exit_prices

signals["Return210"] = returns

signals["Year"] = signals.index.year

signals["Winner"] = (
    signals["Return210"] > 0
).astype(int)


# ==================================================================================================
# BASELINE
# ==================================================================================================

section("BASELINE")

baseline = performance_stats(
    signals
)

print(f"N:       {baseline['N']:,}")
print(f"Avg:     {baseline['Avg']:.3f}")
print(f"PF:      {baseline['PF']:.3f}")
print(f"Win %:   {baseline['WinPct']:.2f}")
print(f"Total:   {baseline['TotalPts']:.2f}")


# ==================================================================================================
# CATEGORY TEST
# ==================================================================================================

features = {
    "TREND_LEVEL": "TrendCode",
    "VOL": "VolCode",
    "VOL_VARIATION": "VolVariationCode",
    "BB_WIDTH": "BBWidthCode",
    "VOLUME": "VolumeCode"
}

summary_rows = []
year_rows = []


for feature_name, column in features.items():

    section(feature_name)

    for level in range(1, 6):

        subset = signals[
            signals[column] == level
        ]

        stats = performance_stats(
            subset
        )

        positive_years = 0
        tested_years = 0

        yearly_avgs = []

        for year in range(
            START_YEAR,
            END_YEAR + 1
        ):

            year_df = subset[
                subset["Year"] == year
            ]

            if len(year_df) == 0:
                continue

            year_stats = performance_stats(
                year_df
            )

            tested_years += 1

            if year_stats["Avg"] > 0:
                positive_years += 1

            yearly_avgs.append(
                year_stats["Avg"]
            )

            year_rows.append(
                {
                    "Feature": feature_name,
                    "Level": level,
                    "Year": year,
                    **year_stats
                }
            )

        delta_avg = (
            stats["Avg"] -
            baseline["Avg"]
        )

        pf_delta = (
            stats["PF"] -
            baseline["PF"]
        )

        summary_rows.append(
            {
                "Feature": feature_name,
                "Level": level,

                "N": stats["N"],
                "Avg": stats["Avg"],
                "PF": stats["PF"],
                "WinPct": stats["WinPct"],
                "TotalPts": stats["TotalPts"],

                "BaselineAvg": baseline["Avg"],
                "BaselinePF": baseline["PF"],

                "AvgVsBaseline": delta_avg,
                "PFVsBaseline": pf_delta,

                "PositiveYears": positive_years,
                "TestedYears": tested_years,

                "WorstYearAvg":
                    min(yearly_avgs)
                    if yearly_avgs
                    else np.nan,

                "BestYearAvg":
                    max(yearly_avgs)
                    if yearly_avgs
                    else np.nan
            }
        )

        print(
            f"{level} | "
            f"N {stats['N']:4d} | "
            f"Avg {stats['Avg']:7.3f} | "
            f"PF {stats['PF']:5.3f} | "
            f"Win {stats['WinPct']:5.2f}% | "
            f"vsBase {delta_avg:+7.3f} | "
            f"Years {positive_years}/{tested_years}"
        )


# ==================================================================================================
# TREND DIRECTION + STRENGTH
# ==================================================================================================

section("TREND DIRECTION + STRENGTH")

for direction in [
    "UP",
    "DOWN"
]:

    for level in range(1, 6):

        subset = signals[
            (signals["TrendDirection"] == direction) &
            (signals["TrendCode"] == level)
        ]

        stats = performance_stats(
            subset
        )

        if stats["N"] == 0:
            continue

        positive_years = 0
        tested_years = 0

        yearly_avgs = []

        for year in range(
            START_YEAR,
            END_YEAR + 1
        ):

            year_df = subset[
                subset["Year"] == year
            ]

            if len(year_df) == 0:
                continue

            ys = performance_stats(
                year_df
            )

            tested_years += 1

            if ys["Avg"] > 0:
                positive_years += 1

            yearly_avgs.append(
                ys["Avg"]
            )

            year_rows.append(
                {
                    "Feature":
                        "TREND_" + direction,

                    "Level":
                        level,

                    "Year":
                        year,

                    **ys
                }
            )

        summary_rows.append(
            {
                "Feature":
                    "TREND_" + direction,

                "Level":
                    level,

                "N":
                    stats["N"],

                "Avg":
                    stats["Avg"],

                "PF":
                    stats["PF"],

                "WinPct":
                    stats["WinPct"],

                "TotalPts":
                    stats["TotalPts"],

                "BaselineAvg":
                    baseline["Avg"],

                "BaselinePF":
                    baseline["PF"],

                "AvgVsBaseline":
                    stats["Avg"] -
                    baseline["Avg"],

                "PFVsBaseline":
                    stats["PF"] -
                    baseline["PF"],

                "PositiveYears":
                    positive_years,

                "TestedYears":
                    tested_years,

                "WorstYearAvg":
                    min(yearly_avgs)
                    if yearly_avgs
                    else np.nan,

                "BestYearAvg":
                    max(yearly_avgs)
                    if yearly_avgs
                    else np.nan
            }
        )

        print(
            f"{direction:<4} {level} | "
            f"N {stats['N']:4d} | "
            f"Avg {stats['Avg']:7.3f} | "
            f"PF {stats['PF']:5.3f} | "
            f"Win {stats['WinPct']:5.2f}% | "
            f"Years {positive_years}/{tested_years}"
        )


# ==================================================================================================
# RESULTS
# ==================================================================================================

summary = pd.DataFrame(
    summary_rows
)

year_summary = pd.DataFrame(
    year_rows
)


# ==================================================================================================
# AUTOMATIC RANKING
# ==================================================================================================

section("MOST INTERESTING CATEGORIES")

# Don't let tiny subsets dominate ranking.
# Require at least 100 observations.

ranked = summary[
    summary["N"] >= 100
].copy()

ranked["StabilityRatio"] = (
    ranked["PositiveYears"] /
    ranked["TestedYears"]
)

# Rank primarily by:
# 1. year stability
# 2. improvement over baseline
# 3. sample size is visible for judgment

ranked = ranked.sort_values(
    [
        "StabilityRatio",
        "AvgVsBaseline",
        "PFVsBaseline"
    ],
    ascending=[
        False,
        False,
        False
    ]
)

print(
    ranked[
        [
            "Feature",
            "Level",
            "N",
            "Avg",
            "PF",
            "WinPct",
            "AvgVsBaseline",
            "PositiveYears",
            "TestedYears",
            "WorstYearAvg"
        ]
    ].head(20).to_string(
        index=False
    )
)


# ==================================================================================================
# FEATURE-LEVEL SEPARATION
# ==================================================================================================

section("FEATURE SEPARATION")

for feature in [
    "TREND_LEVEL",
    "VOL",
    "VOL_VARIATION",
    "BB_WIDTH",
    "VOLUME"
]:

    f = summary[
        summary["Feature"] == feature
    ]

    if len(f) == 0:
        continue

    best = f.loc[
        f["Avg"].idxmax()
    ]

    worst = f.loc[
        f["Avg"].idxmin()
    ]

    spread = (
        best["Avg"] -
        worst["Avg"]
    )

    print()
    print(feature)

    print(
        f"Best level:  {int(best['Level'])} "
        f"| N {int(best['N'])} "
        f"| Avg {best['Avg']:.3f} "
        f"| PF {best['PF']:.3f}"
    )

    print(
        f"Worst level: {int(worst['Level'])} "
        f"| N {int(worst['N'])} "
        f"| Avg {worst['Avg']:.3f} "
        f"| PF {worst['PF']:.3f}"
    )

    print(
        f"Best-worst Avg spread: "
        f"{spread:.3f} points"
    )


# ==================================================================================================
# SAVE
# ==================================================================================================

summary.to_csv(
    OUTPUT_PATH,
    index=False
)

year_summary.to_csv(
    YEAR_OUTPUT_PATH,
    index=False
)

signals.to_csv(
    SIGNAL_OUTPUT_PATH
)


section("DONE")

print(
    f"Category results:\n{OUTPUT_PATH}"
)

print()
print(
    f"Year results:\n{YEAR_OUTPUT_PATH}"
)

print()
print(
    f"Signal-level data:\n{SIGNAL_OUTPUT_PATH}"
)