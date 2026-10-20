import numpy as np
import pandas as pd


# ==================================================================================================
# SETTINGS
# ==================================================================================================

CSV_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MYM_DEC26_1MIN.csv"

OUTPUT_SUMMARY = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\MYM_AB_PORTABILITY_SUMMARY.csv"

OUTPUT_YEAR = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\MYM_AB_PORTABILITY_YEAR.csv"

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14
HOLD_MINUTES = 210

START_YEAR = 2022
END_YEAR = 2026

FILTER_MIN = 0.35
FILTER_MAX = 0.90


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

    r = df["Return210"].dropna()

    return {
        "N": len(r),
        "Avg": r.mean(),
        "PF": profit_factor(r),
        "WinPct": (r > 0).mean() * 100.0,
        "TotalPts": r.sum()
    }


# ==================================================================================================
# LOAD
# ==================================================================================================

section("LOADING MNQ DATA")

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

required = ["Open", "High", "Low", "Close"]

for c in required:
    if c not in df1.columns:
        raise ValueError(f"Missing column: {c}")

print(f"1m rows: {len(df1):,}")
print(f"Start: {df1.index.min()}")
print(f"End:   {df1.index.max()}")


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
    subset=["Open", "High", "Low", "Close"]
)

print(f"15m bars: {len(df15):,}")


# ==================================================================================================
# MACD
# ==================================================================================================

section("CALCULATING MACD")

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

df15["MACDHist"] = (
    macd -
    signal_line
)

df15["BullCross"] = (
    (df15["MACDHist"].shift(1) <= 0) &
    (df15["MACDHist"] > 0)
)


# ==================================================================================================
# ATR + FILTERS
# ==================================================================================================

section("CALCULATING ATR + A/B FILTERS")

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


# A = original filter
df15["A"] = (
    df15["Close"] -
    df15["Close"].shift(1)
) / df15["ATR14"]


# B = proposed second filter
df15["B"] = (
    df15["Close"] -
    df15["Open"].shift(1)
) / df15["ATR14"]


df15["A_PASS"] = (
    (df15["A"] >= FILTER_MIN) &
    (df15["A"] <= FILTER_MAX)
)

df15["B_PASS"] = (
    (df15["B"] >= FILTER_MIN) &
    (df15["B"] <= FILTER_MAX)
)


# ==================================================================================================
# BULLISH MACD CROSS SIGNALS
# ==================================================================================================

signals = df15[
    df15["BullCross"]
].copy()

signals = signals[
    (signals.index.year >= START_YEAR) &
    (signals.index.year <= END_YEAR)
]

signals = signals.dropna(
    subset=["A", "B", "ATR14"]
)

print(f"Bullish MACD crosses: {len(signals):,}")


# ==================================================================================================
# 210-MINUTE FORWARD RETURN
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

    entry_time = df1.index[entry_pos]
    entry_price = opens[entry_pos]

    target_exit_time = (
        entry_time +
        pd.Timedelta(minutes=HOLD_MINUTES)
    )

    exit_pos = np.searchsorted(
        times,
        np.datetime64(target_exit_time),
        side="left"
    )

    if exit_pos >= len(df1):
        continue

    exit_time = df1.index[exit_pos]

    if (
        exit_time >
        target_exit_time +
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
signals["Year"] = signals.index.year


# ==================================================================================================
# COMBINATIONS
# ==================================================================================================

combinations = {
    "BASELINE": np.ones(
        len(signals),
        dtype=bool
    ),

    "A": signals["A_PASS"].to_numpy(),

    "B": signals["B_PASS"].to_numpy(),

    "A+B": (
        signals["A_PASS"] &
        signals["B_PASS"]
    ).to_numpy()
}


# ==================================================================================================
# OVERALL RESULTS
# ==================================================================================================

section("OVERALL PORTABILITY RESULTS")

summary_rows = []
year_rows = []

for name, mask in combinations.items():

    subset = signals.iloc[
        np.where(mask)[0]
    ]

    s = calc_stats(subset)

    positive_years = 0
    tested_years = 0
    yearly_avgs = []

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        y = subset[
            subset["Year"] == year
        ]

        if len(y) == 0:
            continue

        ys = calc_stats(y)

        tested_years += 1

        if ys["Avg"] > 0:
            positive_years += 1

        yearly_avgs.append(
            ys["Avg"]
        )

        year_rows.append(
            {
                "Combination": name,
                "Year": year,
                **ys
            }
        )

    worst_year_avg = (
        min(yearly_avgs)
        if yearly_avgs
        else np.nan
    )

    best_year_avg = (
        max(yearly_avgs)
        if yearly_avgs
        else np.nan
    )

    summary_rows.append(
        {
            "Combination": name,
            **s,
            "PositiveYears": positive_years,
            "TestedYears": tested_years,
            "WorstYearAvg": worst_year_avg,
            "BestYearAvg": best_year_avg
        }
    )

    print(
        f"{name:<10} | "
        f"N {s['N']:5d} | "
        f"Avg {s['Avg']:8.3f} | "
        f"PF {s['PF']:7.3f} | "
        f"Win {s['WinPct']:6.2f}% | "
        f"Years {positive_years}/{tested_years} | "
        f"Worst {worst_year_avg:+8.3f}"
    )


# ==================================================================================================
# YEAR BY YEAR
# ==================================================================================================

section("YEAR-BY-YEAR RESULTS")

year_df = pd.DataFrame(year_rows)

for year in range(
    START_YEAR,
    END_YEAR + 1
):

    print()
    print(f"YEAR {year}")
    print("-" * 90)

    temp = year_df[
        year_df["Year"] == year
    ]

    if len(temp) == 0:
        print("No trades.")
        continue

    print(
        temp[
            [
                "Combination",
                "N",
                "Avg",
                "PF",
                "WinPct",
                "TotalPts"
            ]
        ].to_string(
            index=False
        )
    )


# ==================================================================================================
# RETENTION FUNNEL
# ==================================================================================================

section("TRADE FUNNEL")

base_n = len(signals)

for name, mask in combinations.items():

    n = int(mask.sum())

    pct_all = (
        n / base_n * 100
        if base_n > 0
        else np.nan
    )

    print(
        f"{name:<10} | "
        f"N {n:5d} | "
        f"{pct_all:6.2f}% of baseline"
    )


# ==================================================================================================
# INCREMENTAL VALUE
# ==================================================================================================

section("INCREMENTAL VALUE")

summary_df = pd.DataFrame(
    summary_rows
)

baseline_row = summary_df[
    summary_df["Combination"] == "BASELINE"
].iloc[0]

for _, row in summary_df.iterrows():

    if row["Combination"] == "BASELINE":
        continue

    print()
    print(row["Combination"])

    print(
        f"PF change:  "
        f"{baseline_row['PF']:.3f} -> {row['PF']:.3f} "
        f"({row['PF'] - baseline_row['PF']:+.3f})"
    )

    print(
        f"Avg change: "
        f"{baseline_row['Avg']:.3f} -> {row['Avg']:.3f} "
        f"({row['Avg'] - baseline_row['Avg']:+.3f})"
    )

    print(
        f"Trades:     "
        f"{int(baseline_row['N']):,} -> {int(row['N']):,}"
    )

    print(
        f"Years:      "
        f"{int(row['PositiveYears'])}/{int(row['TestedYears'])}"
    )


# ==================================================================================================
# SAVE
# ==================================================================================================

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False
)

year_df.to_csv(
    OUTPUT_YEAR,
    index=False
)


section("DONE")

print(f"Summary:\n{OUTPUT_SUMMARY}")
print()
print(f"Year results:\n{OUTPUT_YEAR}")