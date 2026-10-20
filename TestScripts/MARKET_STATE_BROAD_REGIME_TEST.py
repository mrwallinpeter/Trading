import numpy as np
import pandas as pd

# ======================================================================================
# SETTINGS
# ======================================================================================

CSV_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"

OUTPUT_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\MARKET_STATE_BROAD_REGIME_TEST.csv"

YEAR_OUTPUT_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\MARKET_STATE_BROAD_REGIME_YEAR_TEST.csv"

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

BLOCK_LENGTH = 20
ATR_LENGTH = 14
RANK_LENGTH = 200

BB_LENGTH = 20
BB_MULT = 2.0

VOL_VARIATION_LENGTH = 20
VOLUME_LENGTH = 20

HOLD_MINUTES = 210

START_YEAR = 2022
END_YEAR = 2026


# ======================================================================================
# HELPERS
# ======================================================================================

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


def stats(df):
    if len(df) == 0:
        return {
            "N": 0,
            "Avg": np.nan,
            "PF": np.nan,
            "WinPct": np.nan,
            "TotalPts": np.nan
        }

    r = df["Return210"].dropna()

    return {
        "N": len(r),
        "Avg": r.mean(),
        "PF": profit_factor(r),
        "WinPct": (r > 0).mean() * 100.0,
        "TotalPts": r.sum()
    }


def percentile_rank_rolling(series, length):

    def rank_last(window):
        if len(window) < length:
            return np.nan

        current = window[-1]
        valid = window[~np.isnan(window)]

        if len(valid) < length:
            return np.nan

        less = np.sum(valid < current)
        equal = np.sum(valid == current)

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


def evaluate_group(signals, feature, group_name, mask, baseline):

    subset = signals[mask].copy()

    s = stats(subset)

    positive_years = 0
    tested_years = 0
    year_rows = []

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        y = subset[
            subset["Year"] == year
        ]

        if len(y) == 0:
            continue

        ys = stats(y)

        tested_years += 1

        if ys["Avg"] > 0:
            positive_years += 1

        year_rows.append(
            {
                "Feature": feature,
                "Group": group_name,
                "Year": year,
                **ys
            }
        )

    row = {
        "Feature": feature,
        "Group": group_name,
        "N": s["N"],
        "Avg": s["Avg"],
        "PF": s["PF"],
        "WinPct": s["WinPct"],
        "TotalPts": s["TotalPts"],
        "BaselineAvg": baseline["Avg"],
        "BaselinePF": baseline["PF"],
        "AvgVsBaseline": s["Avg"] - baseline["Avg"],
        "PFVsBaseline": s["PF"] - baseline["PF"],
        "PositiveYears": positive_years,
        "TestedYears": tested_years
    }

    return row, year_rows


# ======================================================================================
# LOAD DATA
# ======================================================================================

section("LOADING DATA")

df1 = pd.read_csv(CSV_PATH)

time_col = None

for c in [
    "DateTime",
    "Datetime",
    "datetime",
    "Timestamp",
    "timestamp",
    "Time",
    "time",
    "Date",
    "date"
]:
    if c in df1.columns:
        time_col = c
        break

if time_col is None:
    raise ValueError("Could not find timestamp column.")

df1[time_col] = pd.to_datetime(
    df1[time_col]
)

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

print(f"1m rows: {len(df1):,}")
print(f"Start:   {df1.index.min()}")
print(f"End:     {df1.index.max()}")


# ======================================================================================
# RESAMPLE 15M
# ======================================================================================

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

print(f"15m bars: {len(df15):,}")


# ======================================================================================
# MACD
# ======================================================================================

section("MACD")

ema_fast = df15["Close"].ewm(
    span=MACD_FAST,
    adjust=False
).mean()

ema_slow = df15["Close"].ewm(
    span=MACD_SLOW,
    adjust=False
).mean()

macd = ema_fast - ema_slow

macd_signal = macd.ewm(
    span=MACD_SIGNAL,
    adjust=False
).mean()

df15["MACDHist"] = (
    macd -
    macd_signal
)

df15["BullCross"] = (
    (df15["MACDHist"].shift(1) <= 0) &
    (df15["MACDHist"] > 0)
)


# ======================================================================================
# ATR
# ======================================================================================

prev_close = df15["Close"].shift(1)

true_range = pd.concat(
    [
        df15["High"] - df15["Low"],
        (df15["High"] - prev_close).abs(),
        (df15["Low"] - prev_close).abs()
    ],
    axis=1
).max(axis=1)

df15["ATR"] = true_range.ewm(
    alpha=1 / ATR_LENGTH,
    adjust=False
).mean()


# ======================================================================================
# VOL
# ======================================================================================

section("CALCULATING STATE FEATURES")

df15["ATRPercentile"] = percentile_rank_rolling(
    df15["ATR"],
    RANK_LENGTH
)

df15["VolCode"] = level_1_to_5(
    df15["ATRPercentile"]
)


# ======================================================================================
# VOL VARIATION
# ======================================================================================

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


# ======================================================================================
# BOLLINGER WIDTH
# ======================================================================================

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


# ======================================================================================
# TREND
# ======================================================================================

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


# ======================================================================================
# VOLUME
# ======================================================================================

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


# ======================================================================================
# SIGNALS
# ======================================================================================

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

print(
    f"Eligible bullish MACD crosses: {len(signals):,}"
)


# ======================================================================================
# 210M RETURNS
# ======================================================================================

section("CALCULATING 210-MINUTE RETURNS")

times_1m = df1.index.values

opens_1m = df1[
    "Open"
].to_numpy()

valid_times = []
entry_times = []
entry_prices = []
exit_times = []
exit_prices = []
returns = []

for signal_time in signals.index:

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

    target_exit = (
        entry_time +
        pd.Timedelta(
            minutes=HOLD_MINUTES
        )
    )

    exit_pos = np.searchsorted(
        times_1m,
        np.datetime64(target_exit),
        side="left"
    )

    if exit_pos >= len(df1):
        continue

    exit_time = df1.index[
        exit_pos
    ]

    if (
        exit_time >
        target_exit +
        pd.Timedelta(
            minutes=2
        )
    ):
        continue

    exit_price = opens_1m[
        exit_pos
    ]

    ret = (
        exit_price -
        entry_price
    )

    valid_times.append(
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
    valid_times
].copy()

signals["EntryTime"] = entry_times
signals["EntryPrice"] = entry_prices

signals["ExitTime"] = exit_times
signals["ExitPrice"] = exit_prices

signals["Return210"] = returns
signals["Year"] = signals.index.year


# ======================================================================================
# BASELINE
# ======================================================================================

section("BASELINE")

baseline = stats(
    signals
)

print(
    f"N:     {baseline['N']:,}"
)

print(
    f"Avg:   {baseline['Avg']:.3f}"
)

print(
    f"PF:    {baseline['PF']:.3f}"
)

print(
    f"Win %: {baseline['WinPct']:.2f}"
)


# ======================================================================================
# BROAD REGIMES
# ======================================================================================

summary_rows = []
year_rows = []


def run_test(feature, group_name, mask):

    row, yrs = evaluate_group(
        signals,
        feature,
        group_name,
        mask,
        baseline
    )

    summary_rows.append(
        row
    )

    year_rows.extend(
        yrs
    )

    print(
        f"{group_name:<24} "
        f"N {row['N']:4d} | "
        f"Avg {row['Avg']:7.3f} | "
        f"PF {row['PF']:5.3f} | "
        f"Win {row['WinPct']:5.2f}% | "
        f"vsBase {row['AvgVsBaseline']:+7.3f} | "
        f"Years {row['PositiveYears']}/{row['TestedYears']}"
    )


# ======================================================================================
# VOL VARIATION
# ======================================================================================

section("VOL VARIATION - BROAD REGIMES")

run_test(
    "VOL_VARIATION",
    "LOW 1-2",
    signals["VolVariationCode"].isin(
        [1, 2]
    )
)

run_test(
    "VOL_VARIATION",
    "MID 3",
    signals["VolVariationCode"] == 3
)

run_test(
    "VOL_VARIATION",
    "HIGH 4-5",
    signals["VolVariationCode"].isin(
        [4, 5]
    )
)


# ======================================================================================
# BOLLINGER WIDTH
# ======================================================================================

section("BB WIDTH - BROAD REGIMES")

run_test(
    "BB_WIDTH",
    "COMPRESSED 1",
    signals["BBWidthCode"] == 1
)

run_test(
    "BB_WIDTH",
    "MID 2-4",
    signals["BBWidthCode"].isin(
        [2, 3, 4]
    )
)

run_test(
    "BB_WIDTH",
    "EXTREME 5",
    signals["BBWidthCode"] == 5
)


# ======================================================================================
# TREND DOWN
# ======================================================================================

section("TREND DOWN - BROAD REGIMES")

run_test(
    "TREND_DOWN",
    "DOWN 1",
    (
        (signals["TrendDirection"] == "DOWN") &
        (signals["TrendCode"] == 1)
    )
)

run_test(
    "TREND_DOWN",
    "DOWN 2-3",
    (
        (signals["TrendDirection"] == "DOWN") &
        (signals["TrendCode"].isin([2, 3]))
    )
)

run_test(
    "TREND_DOWN",
    "DOWN 4-5",
    (
        (signals["TrendDirection"] == "DOWN") &
        (signals["TrendCode"].isin([4, 5]))
    )
)


# ======================================================================================
# VOL LEVEL
# ======================================================================================

section("VOL LEVEL - BROAD REGIMES")

run_test(
    "VOL",
    "LOW 1",
    signals["VolCode"] == 1
)

run_test(
    "VOL",
    "MID 2-4",
    signals["VolCode"].isin(
        [2, 3, 4]
    )
)

run_test(
    "VOL",
    "HIGH 5",
    signals["VolCode"] == 5
)


# ======================================================================================
# VOLUME
# ======================================================================================

if "VolumeCode" in signals.columns:

    section("VOLUME - BROAD REGIMES")

    run_test(
        "VOLUME",
        "LOW 1",
        signals["VolumeCode"] == 1
    )

    run_test(
        "VOLUME",
        "MID 2-4",
        signals["VolumeCode"].isin(
            [2, 3, 4]
        )
    )

    run_test(
        "VOLUME",
        "HIGH 5",
        signals["VolumeCode"] == 5
    )


# ======================================================================================
# COMBINATIONS
# ======================================================================================

section("SIMPLE COMBINATIONS")

run_test(
    "COMBO",
    "VOLVAR 1-2 + BB 2-4",
    (
        signals["VolVariationCode"].isin([1, 2]) &
        signals["BBWidthCode"].isin([2, 3, 4])
    )
)

run_test(
    "COMBO",
    "VOLVAR 1-2 + DOWN 2-3",
    (
        signals["VolVariationCode"].isin([1, 2]) &
        (signals["TrendDirection"] == "DOWN") &
        signals["TrendCode"].isin([2, 3])
    )
)

run_test(
    "COMBO",
    "BB 2-4 + DOWN 2-3",
    (
        signals["BBWidthCode"].isin([2, 3, 4]) &
        (signals["TrendDirection"] == "DOWN") &
        signals["TrendCode"].isin([2, 3])
    )
)

run_test(
    "COMBO",
    "ALL THREE",
    (
        signals["VolVariationCode"].isin([1, 2]) &
        signals["BBWidthCode"].isin([2, 3, 4]) &
        (signals["TrendDirection"] == "DOWN") &
        signals["TrendCode"].isin([2, 3])
    )
)


# ======================================================================================
# SAVE RESULTS
# ======================================================================================

summary = pd.DataFrame(
    summary_rows
)

year_summary = pd.DataFrame(
    year_rows
)

summary.to_csv(
    OUTPUT_PATH,
    index=False
)

year_summary.to_csv(
    YEAR_OUTPUT_PATH,
    index=False
)


# ======================================================================================
# RANKING
# ======================================================================================

section("RANKED RESULTS")

ranked = summary.copy()

ranked["StabilityRatio"] = (
    ranked["PositiveYears"] /
    ranked["TestedYears"]
)

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
            "Group",
            "N",
            "Avg",
            "PF",
            "WinPct",
            "AvgVsBaseline",
            "PositiveYears",
            "TestedYears"
        ]
    ].to_string(
        index=False
    )
)


section("DONE")

print(
    f"Summary saved:\n{OUTPUT_PATH}"
)

print()

print(
    f"Year detail saved:\n{YEAR_OUTPUT_PATH}"
)