import numpy as np
import pandas as pd


# ==================================================================================================
# SETTINGS
# ==================================================================================================

CSV_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14
HOLD_MINUTES = 90

START_YEAR = 2022
END_YEAR = 2026

A_MIN = 0.35
A_MAX = 0.90

B_MIN = -2.0
B_MAX = 2.0

TIME_WINDOWS = [
    ("06:00-14:00", "06:00", "14:00"),
    ("15:30-17:00", "15:30", "17:00"),
    ("18:30-20:30", "18:30", "20:30"),
]


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


def calc_stats(df):
    if len(df) == 0:
        return {
            "N": 0,
            "Avg": np.nan,
            "PF": np.nan,
            "WinPct": np.nan,
            "TotalPts": np.nan
        }

    r = df["Return90"].dropna()

    return {
        "N": len(r),
        "Avg": r.mean(),
        "PF": profit_factor(r),
        "WinPct": (r > 0).mean() * 100,
        "TotalPts": r.sum()
    }


def time_to_minutes(s):
    h, m = map(int, s.split(":"))
    return h * 60 + m


def in_time_window(index, start_str, end_str):
    start_min = time_to_minutes(start_str)
    end_min = time_to_minutes(end_str)

    mins = index.hour * 60 + index.minute

    return (
        (mins >= start_min) &
        (mins < end_min)
    )


# ==================================================================================================
# LOAD DATA
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
print(f"Start: {df1.index.min()}")
print(f"End:   {df1.index.max()}")


# ==================================================================================================
# BUILD 15M BARS
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
    subset=["Open", "High", "Low", "Close"]
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

signal_line = macd.ewm(
    span=MACD_SIGNAL,
    adjust=False
).mean()

df15["MACDHist"] = macd - signal_line

df15["BullCross"] = (
    (df15["MACDHist"].shift(1) <= 0) &
    (df15["MACDHist"] > 0)
)


# ==================================================================================================
# ATR + ORIGINAL TWO FILTERS
# ==================================================================================================

section("ATR + ORIGINAL TWO FILTERS")

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
    alpha=1 / ATR_LENGTH,
    adjust=False
).mean()


# A = close - close[1]
df15["A_CloseClose1"] = (
    df15["Close"] -
    df15["Close"].shift(1)
) / df15["ATR14"]


# B = close - close[20]
df15["B_CloseClose20"] = (
    df15["Close"] -
    df15["Close"].shift(20)
) / df15["ATR14"]


df15["A_PASS"] = (
    (df15["A_CloseClose1"] >= A_MIN) &
    (df15["A_CloseClose1"] <= A_MAX)
)

df15["B_PASS"] = (
    (df15["B_CloseClose20"] >= B_MIN) &
    (df15["B_CloseClose20"] <= B_MAX)
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
        "A_CloseClose1",
        "B_CloseClose20",
        "ATR14"
    ]
)

print(f"Bullish MACD crosses: {len(signals):,}")


# ==================================================================================================
# 90-MINUTE RETURNS
# ==================================================================================================

section("CALCULATING 90-MINUTE RETURNS")

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

    entry_time = df1.index[entry_pos]
    entry_price = opens[entry_pos]

    target_time = (
        entry_time +
        pd.Timedelta(minutes=HOLD_MINUTES)
    )

    exit_pos = np.searchsorted(
        times,
        np.datetime64(target_time),
        side="left"
    )

    if exit_pos >= len(df1):
        continue

    exit_time = df1.index[exit_pos]

    if (
        exit_time >
        target_time +
        pd.Timedelta(minutes=2)
    ):
        continue

    exit_price = opens[exit_pos]

    valid_times.append(signal_time)
    entry_times.append(entry_time)
    entry_prices.append(entry_price)
    exit_times.append(exit_time)
    exit_prices.append(exit_price)
    returns.append(
        exit_price - entry_price
    )


signals = signals.loc[valid_times].copy()

signals["EntryTime"] = entry_times
signals["EntryPrice"] = entry_prices
signals["ExitTime"] = exit_times
signals["ExitPrice"] = exit_prices
signals["Return90"] = returns
signals["Year"] = signals.index.year


# ==================================================================================================
# COMBINATIONS
# ==================================================================================================

combinations = {
    "BASELINE": pd.Series(
        True,
        index=signals.index
    ),

    "A_ONLY": signals["A_PASS"],

    "B_ONLY": signals["B_PASS"],

    "A+B": (
        signals["A_PASS"] &
        signals["B_PASS"]
    )
}


# ==================================================================================================
# ALL THREE TIME WINDOWS COMBINED
# ==================================================================================================

combined_time_mask = pd.Series(
    False,
    index=signals.index
)

for _, start, end in TIME_WINDOWS:

    combined_time_mask |= in_time_window(
        signals.index,
        start,
        end
    )


section("ALL THREE TIME WINDOWS COMBINED")

for name, filter_mask in combinations.items():

    subset = signals[
        filter_mask &
        combined_time_mask
    ]

    s = calc_stats(subset)

    positive_years = 0

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        y = subset[
            subset["Year"] == year
        ]

        if len(y) > 0:
            ys = calc_stats(y)

            if ys["Avg"] > 0:
                positive_years += 1

    print(
        f"{name:<10} | "
        f"N {s['N']:4d} | "
        f"Avg {s['Avg']:7.3f} | "
        f"PF {s['PF']:6.3f} | "
        f"Win {s['WinPct']:6.2f}% | "
        f"Years {positive_years}/5"
    )


# ==================================================================================================
# EACH TIME WINDOW SEPARATELY
# ==================================================================================================

for window_name, start, end in TIME_WINDOWS:

    section(
        f"TIME WINDOW {window_name}"
    )

    time_mask = in_time_window(
        signals.index,
        start,
        end
    )

    for name, filter_mask in combinations.items():

        subset = signals[
            filter_mask &
            time_mask
        ]

        s = calc_stats(subset)

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

            tested_years += 1

            ys = calc_stats(y)

            if ys["Avg"] > 0:
                positive_years += 1

        print(
            f"{name:<10} | "
            f"N {s['N']:4d} | "
            f"Avg {s['Avg']:7.3f} | "
            f"PF {s['PF']:6.3f} | "
            f"Win {s['WinPct']:6.2f}% | "
            f"Years {positive_years}/{tested_years}"
        )


# ==================================================================================================
# A+B YEAR BREAKDOWN
# ==================================================================================================

section("A+B - YEAR BY YEAR FOR EACH TIME WINDOW")

for window_name, start, end in TIME_WINDOWS:

    print()
    print(window_name)
    print("-" * 80)

    time_mask = in_time_window(
        signals.index,
        start,
        end
    )

    subset = signals[
        signals["A_PASS"] &
        signals["B_PASS"] &
        time_mask
    ]

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

        print(
            f"{year} | "
            f"N {s['N']:4d} | "
            f"Avg {s['Avg']:7.3f} | "
            f"PF {s['PF']:6.3f} | "
            f"Win {s['WinPct']:6.2f}% | "
            f"Total {s['TotalPts']:8.2f}"
        )


section("DONE")