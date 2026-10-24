import numpy as np
import pandas as pd


# ==================================================================================================
# SETTINGS
# ==================================================================================================

CSV_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"

OUTPUT_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\MARKET_STATE_LONG_SHORT_TEST.csv"

YEAR_OUTPUT_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\MARKET_STATE_LONG_SHORT_YEAR_TEST.csv"

SIGNAL_OUTPUT_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\MARKET_STATE_LONG_SHORT_SIGNALS.csv"


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


# ==================================================================================================
# HELPERS
# ==================================================================================================

def section(title):
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def profit_factor(series):
    r = pd.Series(series).dropna()

    gp = r[r > 0].sum()
    gl = -r[r < 0].sum()

    if gl == 0:
        return np.inf if gp > 0 else np.nan

    return gp / gl


def stats(df):
    if len(df) == 0:
        return {
            "N": 0,
            "Avg": np.nan,
            "PF": np.nan,
            "WinPct": np.nan,
            "TotalPts": np.nan
        }

    r = df["DirectionalReturn"].dropna()

    return {
        "N": len(r),
        "Avg": r.mean(),
        "PF": profit_factor(r),
        "WinPct": (r > 0).mean() * 100.0,
        "TotalPts": r.sum()
    }


def percentile_rank_rolling(series, length):

    def rank_last(window):
        current = window[-1]

        if np.isnan(current):
            return np.nan

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


def level_1_to_5(p):
    out = pd.Series(
        np.nan,
        index=p.index
    )

    out.loc[p < 20] = 1
    out.loc[(p >= 20) & (p < 40)] = 2
    out.loc[(p >= 40) & (p < 60)] = 3
    out.loc[(p >= 60) & (p < 80)] = 4
    out.loc[p >= 80] = 5

    return out.astype("Int64")


def print_stats(name, subset):
    s = stats(subset)

    print(
        f"{name:<30} "
        f"N {s['N']:4d} | "
        f"Avg {s['Avg']:7.3f} | "
        f"PF {s['PF']:6.3f} | "
        f"Win {s['WinPct']:6.2f}% | "
        f"Total {s['TotalPts']:9.2f}"
    )

    return s


# ==================================================================================================
# LOAD 1-MINUTE DATA
# ==================================================================================================

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

print(f"1m rows: {len(df1):,}")
print(f"Start:   {df1.index.min()}")
print(f"End:     {df1.index.max()}")


# ==================================================================================================
# BUILD 15-MINUTE BARS
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

print(f"15m bars: {len(df15):,}")


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

macd = ema_fast - ema_slow

macd_signal_line = macd.ewm(
    span=MACD_SIGNAL,
    adjust=False
).mean()

df15["MACDHist"] = (
    macd -
    macd_signal_line
)

df15["BullCross"] = (
    (df15["MACDHist"].shift(1) <= 0) &
    (df15["MACDHist"] > 0)
)

df15["BearCross"] = (
    (df15["MACDHist"].shift(1) >= 0) &
    (df15["MACDHist"] < 0)
)


# ==================================================================================================
# ATR
# ==================================================================================================

section("CALCULATING MARKET STATE")

prev_close = df15["Close"].shift(1)

tr = pd.concat(
    [
        df15["High"] - df15["Low"],
        (df15["High"] - prev_close).abs(),
        (df15["Low"] - prev_close).abs()
    ],
    axis=1
).max(axis=1)

df15["ATR"] = tr.ewm(
    alpha=1 / ATR_LENGTH,
    adjust=False
).mean()


# ==================================================================================================
# VOLATILITY LEVEL
# ==================================================================================================

df15["ATRPercentile"] = percentile_rank_rolling(
    df15["ATR"],
    RANK_LENGTH
)

df15["VolCode"] = level_1_to_5(
    df15["ATRPercentile"]
)


# ==================================================================================================
# VOLATILITY VARIATION
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
# BOLLINGER BAND WIDTH
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
# TREND
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


# ==================================================================================================
# VOLUME
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


# ==================================================================================================
# BUILD LONG + SHORT SIGNAL TABLE
# ==================================================================================================

long_signals = df15[
    df15["BullCross"]
].copy()

long_signals["Side"] = "LONG"


short_signals = df15[
    df15["BearCross"]
].copy()

short_signals["Side"] = "SHORT"


signals = pd.concat(
    [
        long_signals,
        short_signals
    ]
).sort_index()


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
    f"LONG signals:  {(signals['Side'] == 'LONG').sum():,}"
)

print(
    f"SHORT signals: {(signals['Side'] == 'SHORT').sum():,}"
)


# ==================================================================================================
# CALCULATE 210-MINUTE RETURNS
# ==================================================================================================

section("CALCULATING 210-MINUTE RETURNS")

times_1m = df1.index.values
opens_1m = df1["Open"].to_numpy()


valid_rows = []


for signal_time, row in signals.iterrows():

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
        pd.Timedelta(minutes=2)
    ):
        continue


    exit_price = opens_1m[
        exit_pos
    ]


    raw_return = (
        exit_price -
        entry_price
    )


    if row["Side"] == "LONG":
        directional_return = raw_return

    else:
        directional_return = -raw_return


    result = row.to_dict()

    result["SignalTime"] = signal_time
    result["EntryTime"] = entry_time
    result["EntryPrice"] = entry_price
    result["ExitTime"] = exit_time
    result["ExitPrice"] = exit_price
    result["RawReturn"] = raw_return
    result["DirectionalReturn"] = directional_return
    result["Year"] = signal_time.year

    valid_rows.append(
        result
    )


signals = pd.DataFrame(
    valid_rows
)

signals = signals.set_index(
    "SignalTime"
)


print(
    f"Valid LONG:  {(signals['Side'] == 'LONG').sum():,}"
)

print(
    f"Valid SHORT: {(signals['Side'] == 'SHORT').sum():,}"
)


# ==================================================================================================
# RESULT STORAGE
# ==================================================================================================

summary_rows = []
year_rows = []


def evaluate(side, feature, group, mask):

    side_df = signals[
        signals["Side"] == side
    ]

    subset = side_df[
        mask.loc[side_df.index]
    ]

    s = stats(subset)

    positive_years = 0
    tested_years = 0

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
                "Side": side,
                "Feature": feature,
                "Group": group,
                "Year": year,
                **ys
            }
        )

    row = {
        "Side": side,
        "Feature": feature,
        "Group": group,
        **s,
        "PositiveYears": positive_years,
        "TestedYears": tested_years
    }

    summary_rows.append(
        row
    )

    print(
        f"{group:<28} "
        f"N {s['N']:4d} | "
        f"Avg {s['Avg']:7.3f} | "
        f"PF {s['PF']:6.3f} | "
        f"Win {s['WinPct']:6.2f}% | "
        f"Years {positive_years}/{tested_years}"
    )


# ==================================================================================================
# TEST EACH SIDE
# ==================================================================================================

for side in [
    "LONG",
    "SHORT"
]:

    side_df = signals[
        signals["Side"] == side
    ]


    # ==============================================================================================
    # BASELINE
    # ==============================================================================================

    section(
        f"{side} - BASELINE"
    )

    base_stats = print_stats(
        side,
        side_df
    )

    summary_rows.append(
        {
            "Side": side,
            "Feature": "BASELINE",
            "Group": "ALL",
            **base_stats,
            "PositiveYears": np.nan,
            "TestedYears": np.nan
        }
    )


    # ==============================================================================================
    # TREND DIRECTION + LEVEL
    # ==============================================================================================

    section(
        f"{side} - TREND DIRECTION + LEVEL"
    )

    for direction in [
        "UP",
        "DOWN"
    ]:

        for level in range(
            1,
            6
        ):

            mask = (
                (signals["TrendDirection"] == direction) &
                (signals["TrendCode"] == level)
            )

            evaluate(
                side,
                "TREND",
                f"{direction} {level}",
                mask
            )


    # ==============================================================================================
    # BB LEVEL
    # ==============================================================================================

    section(
        f"{side} - BB WIDTH"
    )

    for level in range(
        1,
        6
    ):

        evaluate(
            side,
            "BB_WIDTH",
            f"BB {level}",
            signals["BBWidthCode"] == level
        )


    # ==============================================================================================
    # VOL LEVEL
    # ==============================================================================================

    section(
        f"{side} - VOL"
    )

    for level in range(
        1,
        6
    ):

        evaluate(
            side,
            "VOL",
            f"VOL {level}",
            signals["VolCode"] == level
        )


    # ==============================================================================================
    # VOL VARIATION
    # ==============================================================================================

    section(
        f"{side} - VOL VARIATION"
    )

    for level in range(
        1,
        6
    ):

        evaluate(
            side,
            "VOL_VARIATION",
            f"VOLVAR {level}",
            signals["VolVariationCode"] == level
        )


    # ==============================================================================================
    # VOLUME
    # ==============================================================================================

    if "VolumeCode" in signals.columns:

        section(
            f"{side} - VOLUME"
        )

        for level in range(
            1,
            6
        ):

            evaluate(
                side,
                "VOLUME",
                f"VOLUME {level}",
                signals["VolumeCode"] == level
            )


    # ==============================================================================================
    # TREND + BB MATRIX
    # ==============================================================================================

    section(
        f"{side} - TREND + BB COMBINATIONS"
    )

    for direction in [
        "UP",
        "DOWN"
    ]:

        for trend_level in range(
            1,
            6
        ):

            for bb_level in range(
                1,
                6
            ):

                mask = (
                    (signals["TrendDirection"] == direction) &
                    (signals["TrendCode"] == trend_level) &
                    (signals["BBWidthCode"] == bb_level)
                )

                evaluate(
                    side,
                    "TREND_BB",
                    f"{direction} {trend_level} + BB {bb_level}",
                    mask
                )


    # ==============================================================================================
    # BROAD TREND + BB REGIMES
    # ==============================================================================================

    section(
        f"{side} - BROAD REGIMES"
    )


    broad_tests = [

        (
            "UP 1-2 + BB 1-4",
            (
                (signals["TrendDirection"] == "UP") &
                (signals["TrendCode"].isin([1, 2])) &
                (signals["BBWidthCode"].isin([1, 2, 3, 4]))
            )
        ),

        (
            "UP 3 + BB 1-4",
            (
                (signals["TrendDirection"] == "UP") &
                (signals["TrendCode"] == 3) &
                (signals["BBWidthCode"].isin([1, 2, 3, 4]))
            )
        ),

        (
            "UP 4-5 + BB 1-4",
            (
                (signals["TrendDirection"] == "UP") &
                (signals["TrendCode"].isin([4, 5])) &
                (signals["BBWidthCode"].isin([1, 2, 3, 4]))
            )
        ),

        (
            "DOWN 1-2 + BB 1-4",
            (
                (signals["TrendDirection"] == "DOWN") &
                (signals["TrendCode"].isin([1, 2])) &
                (signals["BBWidthCode"].isin([1, 2, 3, 4]))
            )
        ),

        (
            "DOWN 3 + BB 1-4",
            (
                (signals["TrendDirection"] == "DOWN") &
                (signals["TrendCode"] == 3) &
                (signals["BBWidthCode"].isin([1, 2, 3, 4]))
            )
        ),

        (
            "DOWN 4-5 + BB 1-4",
            (
                (signals["TrendDirection"] == "DOWN") &
                (signals["TrendCode"].isin([4, 5])) &
                (signals["BBWidthCode"].isin([1, 2, 3, 4]))
            )
        ),

        (
            "UP 1-5 + BB 5",
            (
                (signals["TrendDirection"] == "UP") &
                (signals["BBWidthCode"] == 5)
            )
        ),

        (
            "DOWN 1-5 + BB 5",
            (
                (signals["TrendDirection"] == "DOWN") &
                (signals["BBWidthCode"] == 5)
            )
        )
    ]


    for name, mask in broad_tests:

        evaluate(
            side,
            "BROAD_REGIME",
            name,
            mask
        )


# ==================================================================================================
# SUMMARY TABLES
# ==================================================================================================

summary = pd.DataFrame(
    summary_rows
)

year_summary = pd.DataFrame(
    year_rows
)


# ==================================================================================================
# BEST / WORST LONG
# ==================================================================================================

section("LONG - BEST COMBINATIONS")

long_combo = summary[
    (summary["Side"] == "LONG") &
    (summary["Feature"] == "TREND_BB") &
    (summary["N"] >= 100)
].copy()

long_combo = long_combo.sort_values(
    [
        "PositiveYears",
        "Avg"
    ],
    ascending=[
        False,
        False
    ]
)

print(
    long_combo[
        [
            "Group",
            "N",
            "Avg",
            "PF",
            "WinPct",
            "PositiveYears"
        ]
    ].head(
        15
    ).to_string(
        index=False
    )
)


section("LONG - WORST COMBINATIONS")

print(
    long_combo.sort_values(
        "Avg",
        ascending=True
    )[
        [
            "Group",
            "N",
            "Avg",
            "PF",
            "WinPct",
            "PositiveYears"
        ]
    ].head(
        15
    ).to_string(
        index=False
    )
)


# ==================================================================================================
# BEST / WORST SHORT
# ==================================================================================================

section("SHORT - BEST COMBINATIONS")

short_combo = summary[
    (summary["Side"] == "SHORT") &
    (summary["Feature"] == "TREND_BB") &
    (summary["N"] >= 100)
].copy()

short_combo = short_combo.sort_values(
    [
        "PositiveYears",
        "Avg"
    ],
    ascending=[
        False,
        False
    ]
)

print(
    short_combo[
        [
            "Group",
            "N",
            "Avg",
            "PF",
            "WinPct",
            "PositiveYears"
        ]
    ].head(
        15
    ).to_string(
        index=False
    )
)


section("SHORT - WORST COMBINATIONS")

print(
    short_combo.sort_values(
        "Avg",
        ascending=True
    )[
        [
            "Group",
            "N",
            "Avg",
            "PF",
            "WinPct",
            "PositiveYears"
        ]
    ].head(
        15
    ).to_string(
        index=False
    )
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
    f"Summary:\n{OUTPUT_PATH}"
)

print()

print(
    f"Year results:\n{YEAR_OUTPUT_PATH}"
)

print()

print(
    f"Signal-level data:\n{SIGNAL_OUTPUT_PATH}"
)