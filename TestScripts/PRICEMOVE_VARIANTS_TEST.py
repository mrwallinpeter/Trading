import numpy as np
import pandas as pd


# ==================================================================================================
# SETTINGS
# ==================================================================================================

CSV_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"

OUTPUT_BUCKETS = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\PRICEMOVE_VARIANTS_BUCKETS.csv"
OUTPUT_YEARS = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\PRICEMOVE_VARIANTS_YEAR.csv"

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14
PRIOR_BARS = 20
HOLD_MINUTES = 210

START_YEAR = 2022
END_YEAR = 2026

# 10 broad buckets
BUCKET_EDGES = [
    -np.inf,
    -1.0,
    -0.75,
    -0.50,
    -0.25,
    0.0,
    0.25,
    0.50,
    0.75,
    1.0,
    np.inf
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

section("ATR + FEATURE VARIANTS")

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


# ==================================================================================================
# THREE PRICE-MOVE DEFINITIONS
# ==================================================================================================

# A: current definition
df15["A_Close_ClosePrev"] = (
    df15["Close"] -
    df15["Close"].shift(1)
) / df15["ATR14"]


# B: proposed definition
df15["B_Close_OpenPrev"] = (
    df15["Close"] -
    df15["Open"].shift(1)
) / df15["ATR14"]


# C: candle body only
df15["C_Close_OpenCurrent"] = (
    df15["Close"] -
    df15["Open"]
) / df15["ATR14"]


# Keep PriorMoveATR only as a descriptive column for now.
# Do NOT filter on it in this first comparison.
df15["PriorMoveATR"] = (
    df15["Close"].shift(1) -
    df15["Close"].shift(PRIOR_BARS)
) / df15["ATR14"]


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
        "A_Close_ClosePrev",
        "B_Close_OpenPrev",
        "C_Close_OpenCurrent"
    ]
)

print(f"Bullish MACD crosses: {len(signals):,}")


# ==================================================================================================
# 210-MINUTE RETURNS
# ==================================================================================================

section("CALCULATING 210-MINUTE RETURNS")

times_1m = df1.index.values
opens_1m = df1["Open"].to_numpy()

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

    entry_time = df1.index[entry_pos]
    entry_price = opens_1m[entry_pos]

    target_exit_time = (
        entry_time +
        pd.Timedelta(minutes=HOLD_MINUTES)
    )

    exit_pos = np.searchsorted(
        times_1m,
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

    exit_price = opens_1m[exit_pos]

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
# BASELINE
# ==================================================================================================

section("BASELINE")

baseline = stats(signals)

print(f"N:     {baseline['N']:,}")
print(f"Avg:   {baseline['Avg']:.3f}")
print(f"PF:    {baseline['PF']:.3f}")
print(f"Win %: {baseline['WinPct']:.2f}")


# ==================================================================================================
# BUCKET TEST
# ==================================================================================================

features = {
    "A_CLOSE_CLOSEPREV": "A_Close_ClosePrev",
    "B_CLOSE_OPENPREV": "B_Close_OpenPrev",
    "C_CLOSE_OPENCURRENT": "C_Close_OpenCurrent"
}

bucket_rows = []
year_rows = []


for feature_name, col in features.items():

    section(feature_name)

    signals["Bucket"] = pd.cut(
        signals[col],
        bins=BUCKET_EDGES,
        right=False,
        include_lowest=True
    )

    grouped = signals.groupby(
        "Bucket",
        observed=True
    )

    for bucket, subset in grouped:

        s = stats(subset)

        positive_years = 0
        tested_years = 0
        worst_year_avg = np.nan
        best_year_avg = np.nan
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

            ys = stats(y)

            tested_years += 1

            if ys["Avg"] > 0:
                positive_years += 1

            yearly_avgs.append(
                ys["Avg"]
            )

            year_rows.append(
                {
                    "Feature": feature_name,
                    "Bucket": str(bucket),
                    "Year": year,
                    **ys
                }
            )

        if yearly_avgs:
            worst_year_avg = min(yearly_avgs)
            best_year_avg = max(yearly_avgs)

        row = {
            "Feature": feature_name,
            "Bucket": str(bucket),
            "N": s["N"],
            "Avg": s["Avg"],
            "PF": s["PF"],
            "WinPct": s["WinPct"],
            "TotalPts": s["TotalPts"],
            "AvgVsBaseline": s["Avg"] - baseline["Avg"],
            "PositiveYears": positive_years,
            "TestedYears": tested_years,
            "WorstYearAvg": worst_year_avg,
            "BestYearAvg": best_year_avg
        }

        bucket_rows.append(
            row
        )

        print(
            f"{str(bucket):>18} | "
            f"N {s['N']:4d} | "
            f"Avg {s['Avg']:7.3f} | "
            f"PF {s['PF']:6.3f} | "
            f"Win {s['WinPct']:6.2f}% | "
            f"Years {positive_years}/{tested_years}"
        )


# ==================================================================================================
# SAVE
# ==================================================================================================

bucket_df = pd.DataFrame(
    bucket_rows
)

year_df = pd.DataFrame(
    year_rows
)

bucket_df.to_csv(
    OUTPUT_BUCKETS,
    index=False
)

year_df.to_csv(
    OUTPUT_YEARS,
    index=False
)


# ==================================================================================================
# FEATURE SEPARATION
# ==================================================================================================

section("FEATURE SEPARATION")

for feature_name in features.keys():

    temp = bucket_df[
        bucket_df["Feature"] == feature_name
    ].copy()

    temp = temp[
        temp["N"] >= 100
    ]

    if len(temp) == 0:
        continue

    best = temp.sort_values(
        "Avg",
        ascending=False
    ).iloc[0]

    worst = temp.sort_values(
        "Avg",
        ascending=True
    ).iloc[0]

    spread = (
        best["Avg"] -
        worst["Avg"]
    )

    print()
    print(feature_name)

    print(
        f"Best bucket:  {best['Bucket']} | "
        f"N {int(best['N'])} | "
        f"Avg {best['Avg']:.3f} | "
        f"PF {best['PF']:.3f} | "
        f"Years {int(best['PositiveYears'])}/{int(best['TestedYears'])}"
    )

    print(
        f"Worst bucket: {worst['Bucket']} | "
        f"N {int(worst['N'])} | "
        f"Avg {worst['Avg']:.3f} | "
        f"PF {worst['PF']:.3f} | "
        f"Years {int(worst['PositiveYears'])}/{int(worst['TestedYears'])}"
    )

    print(
        f"Best-worst Avg spread: {spread:.3f} points"
    )


# ==================================================================================================
# CORRELATION / REDUNDANCY
# ==================================================================================================

section("FEATURE CORRELATIONS")

corr = signals[
    [
        "A_Close_ClosePrev",
        "B_Close_OpenPrev",
        "C_Close_OpenCurrent",
        "PriorMoveATR"
    ]
].corr()

print(
    corr.to_string()
)


# ==================================================================================================
# OPTIONAL: OLD FILTER RANGE APPLIED TO EACH DEFINITION
# ==================================================================================================

section("SAME 0.35 TO 0.90 RANGE APPLIED TO EACH FEATURE")

for feature_name, col in features.items():

    subset = signals[
        (signals[col] >= 0.35) &
        (signals[col] <= 0.90)
    ]

    s = stats(subset)

    positive_years = 0

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        y = subset[
            subset["Year"] == year
        ]

        if len(y) > 0 and stats(y)["Avg"] > 0:
            positive_years += 1

    print(
        f"{feature_name:<24} "
        f"N {s['N']:4d} | "
        f"Avg {s['Avg']:7.3f} | "
        f"PF {s['PF']:6.3f} | "
        f"Win {s['WinPct']:6.2f}% | "
        f"Years {positive_years}/5"
    )


section("DONE")

print(
    f"Bucket results:\n{OUTPUT_BUCKETS}"
)

print()

print(
    f"Year results:\n{OUTPUT_YEARS}"
)