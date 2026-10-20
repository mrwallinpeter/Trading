import numpy as np
import pandas as pd


# ==================================================================================================
# SETTINGS
# ==================================================================================================

CSV_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"

OUTPUT_RAW = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\TREND_RAW_BUCKETS.csv"
OUTPUT_STATE = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\TREND_PERCENTILE_STATES.csv"
OUTPUT_YEAR = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\TREND_RAW_VS_PERCENTILE_YEAR.csv"

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14

TREND_LOOKBACK = 20
PERCENTILE_LOOKBACK = 200

HOLD_MINUTES = 210

START_YEAR = 2022
END_YEAR = 2026


# Raw displacement buckets in ATR units
RAW_EDGES = [
    -np.inf,
    -5,
    -4,
    -3,
    -2,
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    np.inf
]


# ==================================================================================================
# HELPERS
# ==================================================================================================

def section(title):
    print()
    print("=" * 125)
    print(title)
    print("=" * 125)


def profit_factor(series):
    r = pd.Series(series).dropna()

    gross_profit = r[r > 0].sum()
    gross_loss = -r[r < 0].sum()

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else np.nan

    return gross_profit / gross_loss


def calc_stats(df):
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


def year_stability(subset):

    positive_years = 0
    tested_years = 0
    yearly_avgs = []
    details = []

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        y = subset[
            subset["Year"] == year
        ]

        if len(y) == 0:
            continue

        s = calc_stats(y)

        tested_years += 1

        if s["Avg"] > 0:
            positive_years += 1

        yearly_avgs.append(
            s["Avg"]
        )

        details.append(
            (year, s)
        )

    worst = (
        min(yearly_avgs)
        if yearly_avgs
        else np.nan
    )

    best = (
        max(yearly_avgs)
        if yearly_avgs
        else np.nan
    )

    return (
        positive_years,
        tested_years,
        worst,
        best,
        details
    )


def rolling_percentile_rank(series, lookback):

    arr = series.to_numpy(dtype=float)
    result = np.full(len(arr), np.nan)

    for i in range(
        lookback - 1,
        len(arr)
    ):

        window = arr[
            i - lookback + 1:
            i + 1
        ]

        current = arr[i]

        if not np.isfinite(current):
            continue

        valid = window[
            np.isfinite(window)
        ]

        if len(valid) == 0:
            continue

        result[i] = (
            np.sum(valid <= current) /
            len(valid) *
            100.0
        )

    return pd.Series(
        result,
        index=series.index
    )


# ==================================================================================================
# LOAD 1-MIN DATA
# ==================================================================================================

section("LOADING DATA")

df1 = pd.read_csv(
    CSV_PATH
)

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
    raise ValueError(
        "Could not find timestamp column."
    )


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


df1 = df1.rename(
    columns=rename_map
)


print(
    f"1m rows: {len(df1):,}"
)

print(
    f"Start: {df1.index.min()}"
)

print(
    f"End:   {df1.index.max()}"
)


# ==================================================================================================
# BUILD 15-MIN BARS
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
).agg(
    agg
)


df15 = df15.dropna(
    subset=[
        "Open",
        "High",
        "Low",
        "Close"
    ]
)


print(
    f"15m bars: {len(df15):,}"
)


# ==================================================================================================
# MACD
# ==================================================================================================

section("MACD")


ema_fast = df15[
    "Close"
].ewm(
    span=MACD_FAST,
    adjust=False
).mean()


ema_slow = df15[
    "Close"
].ewm(
    span=MACD_SLOW,
    adjust=False
).mean()


macd = (
    ema_fast -
    ema_slow
)


signal_line = macd.ewm(
    span=MACD_SIGNAL,
    adjust=False
).mean()


df15["MACDHist"] = (
    macd -
    signal_line
)


df15["BullCross"] = (
    (df15["MACDHist"].shift(1) <= 0) &
    (df15["MACDHist"] > 0)
)


# ==================================================================================================
# ATR
# ==================================================================================================

section("ATR + TREND FEATURES")


prev_close = df15[
    "Close"
].shift(1)


tr = pd.concat(
    [
        df15["High"] - df15["Low"],

        (
            df15["High"] -
            prev_close
        ).abs(),

        (
            df15["Low"] -
            prev_close
        ).abs()
    ],
    axis=1
).max(
    axis=1
)


df15["ATR14"] = tr.ewm(
    alpha=1 / ATR_LENGTH,
    adjust=False
).mean()


# ==================================================================================================
# RAW 20-BAR DISPLACEMENT
# ==================================================================================================

df15["TrendRaw"] = (
    df15["Close"] -
    df15["Close"].shift(
        TREND_LOOKBACK
    )
) / df15["ATR14"]


df15["TrendStrengthRaw"] = (
    df15["TrendRaw"].abs()
)


# ==================================================================================================
# PERCENTILE STRENGTH
# ==================================================================================================

section("CALCULATING ROLLING TREND PERCENTILE")


df15["TrendStrengthPct"] = rolling_percentile_rank(
    df15["TrendStrengthRaw"],
    PERCENTILE_LOOKBACK
)


# Five levels:
# 1 = 0-20
# 2 = 20-40
# 3 = 40-60
# 4 = 60-80
# 5 = 80-100

df15["TrendLevel"] = pd.cut(
    df15["TrendStrengthPct"],
    bins=[
        -np.inf,
        20,
        40,
        60,
        80,
        np.inf
    ],
    labels=[
        1,
        2,
        3,
        4,
        5
    ],
    right=False
).astype(
    "float"
)


df15["TrendDirection"] = np.where(
    df15["TrendRaw"] > 0,
    "UP",
    np.where(
        df15["TrendRaw"] < 0,
        "DOWN",
        "SIDEWAYS"
    )
)


df15["TrendState"] = np.where(
    df15["TrendDirection"] == "SIDEWAYS",
    "SIDEWAYS",
    df15["TrendDirection"] +
    df15["TrendLevel"]
    .fillna(0)
    .astype(int)
    .astype(str)
)


# ==================================================================================================
# SIGNALS
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
        "TrendRaw",
        "TrendStrengthPct",
        "TrendLevel"
    ]
)


print(
    f"Bullish MACD crosses: {len(signals):,}"
)


# ==================================================================================================
# 210-MIN RETURNS
# ==================================================================================================

section("CALCULATING 210-MINUTE RETURNS")


times = df1.index.values
opens = df1["Open"].to_numpy()


valid_times = []
entry_times = []
entry_prices = []
exit_times = []
exit_prices = []
returns = []


for signal_time in signals.index:

    entry_pos = np.searchsorted(
        times,
        np.datetime64(signal_time),
        side="right"
    )


    if entry_pos >= len(df1):
        continue


    entry_time = df1.index[
        entry_pos
    ]

    entry_price = opens[
        entry_pos
    ]


    target_time = (
        entry_time +
        pd.Timedelta(
            minutes=HOLD_MINUTES
        )
    )


    exit_pos = np.searchsorted(
        times,
        np.datetime64(target_time),
        side="left"
    )


    if exit_pos >= len(df1):
        continue


    exit_time = df1.index[
        exit_pos
    ]


    if (
        exit_time >
        target_time +
        pd.Timedelta(
            minutes=2
        )
    ):
        continue


    exit_price = opens[
        exit_pos
    ]


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
        exit_price -
        entry_price
    )


signals = signals.loc[
    valid_times
].copy()


signals["EntryTime"] = entry_times
signals["EntryPrice"] = entry_prices

signals["ExitTime"] = exit_times
signals["ExitPrice"] = exit_prices

signals["Return210"] = returns

signals["Year"] = (
    signals.index.year
)


# ==================================================================================================
# BASELINE
# ==================================================================================================

section("BASELINE")


base = calc_stats(
    signals
)


print(
    f"N:     {base['N']:,}"
)

print(
    f"Avg:   {base['Avg']:.3f}"
)

print(
    f"PF:    {base['PF']:.3f}"
)

print(
    f"Win %: {base['WinPct']:.2f}"
)


# ==================================================================================================
# RAW BUCKETS
# ==================================================================================================

section("RAW TREND BUCKETS")


signals["RawBucket"] = pd.cut(
    signals["TrendRaw"],
    bins=RAW_EDGES,
    right=False
)


raw_rows = []
year_rows = []


for bucket, subset in signals.groupby(
    "RawBucket",
    observed=True
):

    s = calc_stats(
        subset
    )


    (
        positive_years,
        tested_years,
        worst,
        best,
        details
    ) = year_stability(
        subset
    )


    raw_rows.append(
        {
            "Bucket": str(bucket),
            **s,
            "PositiveYears": positive_years,
            "TestedYears": tested_years,
            "WorstYearAvg": worst,
            "BestYearAvg": best
        }
    )


    print(
        f"{str(bucket):>14} | "
        f"N {s['N']:4d} | "
        f"Avg {s['Avg']:7.3f} | "
        f"PF {s['PF']:6.3f} | "
        f"Win {s['WinPct']:6.2f}% | "
        f"Years {positive_years}/{tested_years}"
    )


    for year, ys in details:

        year_rows.append(
            {
                "Representation":
                    "RAW",

                "State":
                    str(bucket),

                "Year":
                    year,

                **ys
            }
        )


# ==================================================================================================
# PERCENTILE TREND STATES
# ==================================================================================================

section("PERCENTILE TREND STATES")


state_rows = []


state_order = [
    "DOWN1",
    "DOWN2",
    "DOWN3",
    "DOWN4",
    "DOWN5",
    "UP1",
    "UP2",
    "UP3",
    "UP4",
    "UP5"
]


for state in state_order:

    subset = signals[
        signals["TrendState"] == state
    ]


    s = calc_stats(
        subset
    )


    (
        positive_years,
        tested_years,
        worst,
        best,
        details
    ) = year_stability(
        subset
    )


    state_rows.append(
        {
            "State": state,
            **s,
            "PositiveYears": positive_years,
            "TestedYears": tested_years,
            "WorstYearAvg": worst,
            "BestYearAvg": best
        }
    )


    print(
        f"{state:<8} | "
        f"N {s['N']:4d} | "
        f"Avg {s['Avg']:7.3f} | "
        f"PF {s['PF']:6.3f} | "
        f"Win {s['WinPct']:6.2f}% | "
        f"Years {positive_years}/{tested_years}"
    )


    for year, ys in details:

        year_rows.append(
            {
                "Representation":
                    "PERCENTILE",

                "State":
                    state,

                "Year":
                    year,

                **ys
            }
        )


# ==================================================================================================
# BROAD RAW GROUPS
# ==================================================================================================

section("BROAD RAW TREND GROUPS")


raw_groups = {
    "RAW <= -3":
        signals["TrendRaw"] <= -3,

    "RAW -3 to -2":
        (
            (signals["TrendRaw"] > -3) &
            (signals["TrendRaw"] <= -2)
        ),

    "RAW -2 to -1":
        (
            (signals["TrendRaw"] > -2) &
            (signals["TrendRaw"] <= -1)
        ),

    "RAW -1 to 0":
        (
            (signals["TrendRaw"] > -1) &
            (signals["TrendRaw"] < 0)
        ),

    "RAW 0 to 1":
        (
            (signals["TrendRaw"] >= 0) &
            (signals["TrendRaw"] < 1)
        ),

    "RAW 1 to 2":
        (
            (signals["TrendRaw"] >= 1) &
            (signals["TrendRaw"] < 2)
        ),

    "RAW 2 to 3":
        (
            (signals["TrendRaw"] >= 2) &
            (signals["TrendRaw"] < 3)
        ),

    "RAW >= 3":
        signals["TrendRaw"] >= 3
}


for name, mask in raw_groups.items():

    subset = signals[
        mask
    ]

    s = calc_stats(
        subset
    )

    (
        positive_years,
        tested_years,
        worst,
        best,
        _
    ) = year_stability(
        subset
    )


    print(
        f"{name:<15} | "
        f"N {s['N']:4d} | "
        f"Avg {s['Avg']:7.3f} | "
        f"PF {s['PF']:6.3f} | "
        f"Years {positive_years}/{tested_years}"
    )


# ==================================================================================================
# BROAD PERCENTILE GROUPS
# ==================================================================================================

section("BROAD PERCENTILE TREND GROUPS")


pct_groups = {
    "DOWN1":
        signals["TrendState"] == "DOWN1",

    "DOWN2-3":
        signals["TrendState"].isin(
            [
                "DOWN2",
                "DOWN3"
            ]
        ),

    "DOWN4-5":
        signals["TrendState"].isin(
            [
                "DOWN4",
                "DOWN5"
            ]
        ),

    "UP1":
        signals["TrendState"] == "UP1",

    "UP2-3":
        signals["TrendState"].isin(
            [
                "UP2",
                "UP3"
            ]
        ),

    "UP4-5":
        signals["TrendState"].isin(
            [
                "UP4",
                "UP5"
            ]
        )
}


for name, mask in pct_groups.items():

    subset = signals[
        mask
    ]

    s = calc_stats(
        subset
    )

    (
        positive_years,
        tested_years,
        worst,
        best,
        _
    ) = year_stability(
        subset
    )


    print(
        f"{name:<10} | "
        f"N {s['N']:4d} | "
        f"Avg {s['Avg']:7.3f} | "
        f"PF {s['PF']:6.3f} | "
        f"Years {positive_years}/{tested_years}"
    )


# ==================================================================================================
# CROSS-TAB:
# RAW DISPLACEMENT VS PERCENTILE STATE
# ==================================================================================================

section("RAW VALUE INSIDE EACH PERCENTILE STATE")


for state in state_order:

    subset = signals[
        signals["TrendState"] == state
    ]

    if len(subset) == 0:
        continue


    print(
        f"{state:<8} | "
        f"N {len(subset):4d} | "
        f"Raw mean {subset['TrendRaw'].mean():7.3f} | "
        f"Raw median {subset['TrendRaw'].median():7.3f} | "
        f"Raw P25 {subset['TrendRaw'].quantile(0.25):7.3f} | "
        f"Raw P75 {subset['TrendRaw'].quantile(0.75):7.3f}"
    )


# ==================================================================================================
# SAVE
# ==================================================================================================

raw_df = pd.DataFrame(
    raw_rows
)


state_df = pd.DataFrame(
    state_rows
)


year_df = pd.DataFrame(
    year_rows
)


raw_df.to_csv(
    OUTPUT_RAW,
    index=False
)


state_df.to_csv(
    OUTPUT_STATE,
    index=False
)


year_df.to_csv(
    OUTPUT_YEAR,
    index=False
)


section("DONE")

print(
    f"Raw buckets:\n{OUTPUT_RAW}"
)

print()

print(
    f"Percentile states:\n{OUTPUT_STATE}"
)

print()

print(
    f"Year results:\n{OUTPUT_YEAR}"
)