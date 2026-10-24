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
    r"\ATR14_Intraday_Cycle"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

ATR_LENGTH = 14

START_YEAR = 2022
END_YEAR = 2026


# ============================================================
# LOAD 1-MIN DATA
# ============================================================

print()
print("=" * 120)
print("LOADING DATA")
print("=" * 120)

df1 = pd.read_csv(ONE_MIN_FILE)

df1["DateTime"] = pd.to_datetime(df1["DateTime"])

df1 = (
    df1
    .sort_values("DateTime")
    .drop_duplicates("DateTime")
    .set_index("DateTime")
)

for c in ["Open", "High", "Low", "Close", "Volume"]:
    df1[c] = pd.to_numeric(
        df1[c],
        errors="coerce"
    )

df1 = df1.dropna(
    subset=["Open", "High", "Low", "Close"]
)

print("1m rows:", len(df1))
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
    subset=["Open", "High", "Low", "Close"]
)

print()
print("15m bars:", len(df15))


# ============================================================
# ATR14 — WILDER
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
# DATE FILTER
# ============================================================

df15 = df15[
    (df15.index.year >= START_YEAR)
    &
    (df15.index.year <= END_YEAR)
].copy()

df15 = df15[
    df15["ATR14"].notna()
].copy()


# ============================================================
# TIME VARIABLES
# ============================================================

df15["Date"] = df15.index.date
df15["Year"] = df15.index.year

df15["Time"] = df15.index.strftime("%H:%M")

df15["Hour"] = df15.index.hour


# ============================================================
# FIND DAILY ATR MINIMUM AND MAXIMUM
# ============================================================

daily_rows = []

for date, day in df15.groupby("Date"):

    if len(day) < 20:
        continue

    min_idx = day["ATR14"].idxmin()
    max_idx = day["ATR14"].idxmax()

    daily_rows.append({
        "Date": date,
        "Year": min_idx.year,

        "MinTime": min_idx.strftime("%H:%M"),
        "MinHour": min_idx.hour,
        "MinATR": day.loc[min_idx, "ATR14"],

        "MaxTime": max_idx.strftime("%H:%M"),
        "MaxHour": max_idx.hour,
        "MaxATR": day.loc[max_idx, "ATR14"],

        "ATRRange": (
            day.loc[max_idx, "ATR14"]
            -
            day.loc[min_idx, "ATR14"]
        )
    })


daily = pd.DataFrame(daily_rows)

print()
print("Days analyzed:", len(daily))


# ============================================================
# 15-MIN TIME DISTRIBUTION
# ============================================================

min_time_counts = (
    daily["MinTime"]
    .value_counts()
    .sort_index()
)

max_time_counts = (
    daily["MaxTime"]
    .value_counts()
    .sort_index()
)

time_distribution = pd.DataFrame({
    "MinCount": min_time_counts,
    "MaxCount": max_time_counts
}).fillna(0)

time_distribution["MinPct"] = (
    time_distribution["MinCount"]
    /
    len(daily)
    *
    100
)

time_distribution["MaxPct"] = (
    time_distribution["MaxCount"]
    /
    len(daily)
    *
    100
)


# ============================================================
# HOURLY DISTRIBUTION
# ============================================================

hourly_min = (
    daily["MinHour"]
    .value_counts()
    .sort_index()
)

hourly_max = (
    daily["MaxHour"]
    .value_counts()
    .sort_index()
)

hourly_distribution = pd.DataFrame({
    "MinCount": hourly_min,
    "MaxCount": hourly_max
}).fillna(0)

hourly_distribution["MinPct"] = (
    hourly_distribution["MinCount"]
    /
    len(daily)
    *
    100
)

hourly_distribution["MaxPct"] = (
    hourly_distribution["MaxCount"]
    /
    len(daily)
    *
    100
)


# ============================================================
# AVERAGE ATR BY CLOCK TIME
# ============================================================

atr_by_time = (
    df15
    .groupby("Time")["ATR14"]
    .agg(
        N="count",
        MeanATR="mean",
        MedianATR="median",
        StdATR="std"
    )
)

atr_by_time["MeanRelativeToDailyMean"] = np.nan


# ============================================================
# NORMALIZE ATR WITHIN EACH DAY
#
# This helps distinguish:
# "ATR is high because this year is volatile"
# from
# "ATR is high because it is 15:30."
# ============================================================

daily_mean_atr = (
    df15
    .groupby("Date")["ATR14"]
    .transform("mean")
)

df15["ATR_RelativeDailyMean"] = (
    df15["ATR14"]
    /
    daily_mean_atr
)

relative_by_time = (
    df15
    .groupby("Time")["ATR_RelativeDailyMean"]
    .agg(
        MeanRelativeATR="mean",
        MedianRelativeATR="median"
    )
)

atr_by_time = atr_by_time.join(
    relative_by_time
)


# ============================================================
# YEAR-BY-YEAR MAX/MIN HOURS
# ============================================================

year_hour_rows = []

for year in sorted(daily["Year"].unique()):

    d = daily[
        daily["Year"] == year
    ]

    for hour in range(24):

        min_n = int(
            (d["MinHour"] == hour).sum()
        )

        max_n = int(
            (d["MaxHour"] == hour).sum()
        )

        year_hour_rows.append({
            "Year": year,
            "Hour": hour,
            "Days": len(d),
            "MinCount": min_n,
            "MaxCount": max_n,
            "MinPct": (
                min_n / len(d) * 100
                if len(d) > 0
                else np.nan
            ),
            "MaxPct": (
                max_n / len(d) * 100
                if len(d) > 0
                else np.nan
            )
        })

year_hour = pd.DataFrame(
    year_hour_rows
)


# ============================================================
# PRINT TOP TIMES
# ============================================================

print()
print("=" * 120)
print("MOST COMMON DAILY ATR14 MINIMUM TIMES — 15-MINUTE")
print("=" * 120)

print(
    time_distribution
    .sort_values(
        "MinCount",
        ascending=False
    )
    .head(20)
    .round(2)
    .to_string()
)


print()
print("=" * 120)
print("MOST COMMON DAILY ATR14 MAXIMUM TIMES — 15-MINUTE")
print("=" * 120)

print(
    time_distribution
    .sort_values(
        "MaxCount",
        ascending=False
    )
    .head(20)
    .round(2)
    .to_string()
)


# ============================================================
# PRINT HOURS
# ============================================================

print()
print("=" * 120)
print("DAILY ATR14 MINIMUM — BY HOUR")
print("=" * 120)

print(
    hourly_distribution[
        ["MinCount", "MinPct"]
    ]
    .round(2)
    .to_string()
)


print()
print("=" * 120)
print("DAILY ATR14 MAXIMUM — BY HOUR")
print("=" * 120)

print(
    hourly_distribution[
        ["MaxCount", "MaxPct"]
    ]
    .round(2)
    .to_string()
)


# ============================================================
# AVERAGE INTRADAY ATR CYCLE
# ============================================================

print()
print("=" * 140)
print("AVERAGE ATR14 BY 15-MINUTE CLOCK TIME")
print("=" * 140)

print(
    atr_by_time[
        [
            "N",
            "MeanATR",
            "MedianATR",
            "MeanRelativeATR",
            "MedianRelativeATR"
        ]
    ]
    .round(4)
    .to_string()
)


# ============================================================
# TOP/BOTTOM TIMES BY NORMALIZED ATR
# ============================================================

print()
print("=" * 120)
print("HIGHEST NORMALIZED ATR TIMES")
print("=" * 120)

print(
    atr_by_time
    .sort_values(
        "MeanRelativeATR",
        ascending=False
    )
    .head(20)[
        [
            "N",
            "MeanATR",
            "MeanRelativeATR"
        ]
    ]
    .round(4)
    .to_string()
)


print()
print("=" * 120)
print("LOWEST NORMALIZED ATR TIMES")
print("=" * 120)

print(
    atr_by_time
    .sort_values(
        "MeanRelativeATR",
        ascending=True
    )
    .head(20)[
        [
            "N",
            "MeanATR",
            "MeanRelativeATR"
        ]
    ]
    .round(4)
    .to_string()
)


# ============================================================
# SAVE RESULTS
# ============================================================

daily.to_csv(
    OUTPUT_DIR / "daily_atr_min_max.csv",
    index=False
)

time_distribution.to_csv(
    OUTPUT_DIR / "atr_min_max_by_15m_time.csv"
)

hourly_distribution.to_csv(
    OUTPUT_DIR / "atr_min_max_by_hour.csv"
)

atr_by_time.to_csv(
    OUTPUT_DIR / "average_atr_by_time.csv"
)

year_hour.to_csv(
    OUTPUT_DIR / "year_by_year_hour_distribution.csv",
    index=False
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 120)
print("DONE")
print("=" * 120)

print()
print("ATR = Wilder ATR14 on reconstructed 15m bars.")
print()
print(
    "For each calendar day the script finds the exact "
    "15m bar containing the day's ATR minimum and maximum."
)

print()
print(
    "It also calculates ATR by clock time, normalized "
    "relative to each day's average ATR."
)

print()
print("Output folder:")
print(OUTPUT_DIR)