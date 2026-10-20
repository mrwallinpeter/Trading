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
    r"\MACD_15M_Cross_PreCrossUpBars"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

YEARS = [2022, 2023, 2024, 2025, 2026]

HOLDS = [
    120,
    150,
    180,
    210,
    240
]

MAX_COUNT_SEARCH = 50


# ============================================================
# LOAD 1-MIN DATA
# ============================================================

print()
print("=" * 180)
print("LOADING 1-MIN DATA")
print("=" * 180)

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

print("Rows:", len(df1))
print("Start:", df1.index.min())
print("End:  ", df1.index.max())


# ============================================================
# BUILD 15M BARS
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
# MACD
# ============================================================

ema_fast = df15["Close"].ewm(
    span=MACD_FAST,
    adjust=False
).mean()

ema_slow = df15["Close"].ewm(
    span=MACD_SLOW,
    adjust=False
).mean()

df15["MACD"] = (
    ema_fast
    -
    ema_slow
)

df15["Signal"] = df15["MACD"].ewm(
    span=MACD_SIGNAL,
    adjust=False
).mean()

df15["Hist"] = (
    df15["MACD"]
    -
    df15["Signal"]
)


# ============================================================
# BULLISH HISTOGRAM ZERO-CROSS
# ============================================================

df15["BullCross"] = (
    (df15["Hist"].shift(1) <= 0)
    &
    (df15["Hist"] > 0)
)


# ============================================================
# COUNT CONSECUTIVE HISTOGRAM INCREASES BEFORE CROSS
#
# Cross bar itself is NOT counted.
#
# For signal at t:
#
# Hist[t-1] > Hist[t-2]  -> count 1
# Hist[t-2] > Hist[t-3]  -> count 2
# Hist[t-3] > Hist[t-4]  -> count 3
# ...
#
# Stop at first failure.
# ============================================================

hist = df15["Hist"].to_numpy(
    dtype=float
)

pre_cross_up_bars = np.zeros(
    len(df15),
    dtype=int
)

for i in range(2, len(df15)):

    count = 0
    j = i - 1

    while (
        j >= 1
        and
        count < MAX_COUNT_SEARCH
        and
        np.isfinite(hist[j])
        and
        np.isfinite(hist[j - 1])
        and
        hist[j] > hist[j - 1]
    ):

        count += 1
        j -= 1

    pre_cross_up_bars[i] = count


df15["PreCrossUpBars"] = (
    pre_cross_up_bars
)


# ============================================================
# BUCKET
# ============================================================

def up_bar_group(x):

    if x == 0:
        return "0"

    if x == 1:
        return "1"

    if x == 2:
        return "2"

    if x == 3:
        return "3"

    if x == 4:
        return "4"

    if x == 5:
        return "5"

    return "6_PLUS"


df15["PreCrossUpGroup"] = (
    df15["PreCrossUpBars"]
    .apply(up_bar_group)
)


# ============================================================
# MASTER
# ============================================================

master = df15[
    df15.index.year.isin(YEARS)
].copy()

master["Year"] = (
    master.index.year
)


# ============================================================
# SIGNAL COUNTS
# ============================================================

signals = master[
    master["BullCross"]
].copy()

print()
print("=" * 180)
print("BULLISH MACD CROSS COUNTS")
print("=" * 180)

print(
    "Total bullish crosses:",
    len(signals)
)

print()
print(
    "Pre-cross histogram up-bar groups:"
)

print(
    signals["PreCrossUpGroup"]
    .value_counts()
    .reindex(
        [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6_PLUS"
        ],
        fill_value=0
    )
    .to_string()
)

print()
print(
    "Raw PreCrossUpBars distribution:"
)

print(
    signals["PreCrossUpBars"]
    .describe(
        percentiles=[
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
    .round(2)
    .to_string()
)


# ============================================================
# 1-MIN EXECUTION
# ============================================================

minute_times = df1.index.values.astype(
    "datetime64[ns]"
)

minute_open = df1[
    "Open"
].to_numpy(
    dtype=float
)

minute_close = df1[
    "Close"
].to_numpy(
    dtype=float
)

bar_times_np = master.index.values.astype(
    "datetime64[ns]"
)

entry_positions = np.searchsorted(
    minute_times,
    bar_times_np,
    side="right"
)

valid_entry = (
    entry_positions
    <
    len(minute_times)
)

master = master.iloc[
    np.where(valid_entry)[0]
].copy()

entry_positions = (
    entry_positions[
        valid_entry
    ]
)

master["EntryTime"] = pd.to_datetime(
    minute_times[
        entry_positions
    ]
)

master["EntryPrice"] = (
    minute_open[
        entry_positions
    ]
)


# ============================================================
# PRECOMPUTE RETURNS
# ============================================================

print()
print("=" * 180)
print("PRECOMPUTING RETURNS")
print("=" * 180)

entry_times_np = master[
    "EntryTime"
].values.astype(
    "datetime64[ns]"
)

entry_prices_np = master[
    "EntryPrice"
].to_numpy(
    dtype=float
)


for hold in HOLDS:

    target_times = (
        entry_times_np
        +
        np.timedelta64(
            hold,
            "m"
        )
    )

    exit_positions = np.searchsorted(
        minute_times,
        target_times,
        side="left"
    )

    pnl = np.full(
        len(master),
        np.nan
    )

    valid_exit = (
        exit_positions
        <
        len(minute_times)
    )

    valid_idx = np.where(
        valid_exit
    )[0]

    actual_exit_times = minute_times[
        exit_positions[
            valid_idx
        ]
    ]

    tolerance_ok = (
        actual_exit_times
        <=
        target_times[
            valid_idx
        ]
        +
        np.timedelta64(
            2,
            "m"
        )
    )

    final_idx = (
        valid_idx[
            tolerance_ok
        ]
    )

    pnl[
        final_idx
    ] = (
        minute_close[
            exit_positions[
                final_idx
            ]
        ]
        -
        entry_prices_np[
            final_idx
        ]
    )

    master[
        f"PnL_{hold}"
    ] = pnl

    print(
        f"{hold:>3} min -> "
        f"{np.isfinite(pnl).sum()} valid bars"
    )


# ============================================================
# STATS
# ============================================================

def profit_factor(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    pnl = pnl[
        np.isfinite(pnl)
    ]

    gp = pnl[
        pnl > 0
    ].sum()

    gl = -pnl[
        pnl < 0
    ].sum()

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

    pnl = pnl[
        np.isfinite(pnl)
    ]

    if len(pnl) < 2:
        return np.nan

    sd = pnl.std(
        ddof=1
    )

    if sd == 0:
        return np.nan

    return (
        np.sqrt(len(pnl))
        *
        pnl.mean()
        /
        sd
    )


def max_drawdown(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    pnl = pnl[
        np.isfinite(pnl)
    ]

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

    return abs(
        (
            equity
            -
            peak
        ).min()
    )


def stats(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    pnl = pnl[
        np.isfinite(pnl)
    ]

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
        "N": len(pnl),
        "WinRatePct": np.mean(pnl > 0) * 100,
        "NetPoints": pnl.sum(),
        "AveragePnL": pnl.mean(),
        "MedianPnL": np.median(pnl),
        "ProfitFactor": profit_factor(pnl),
        "SQN": sqn(pnl),
        "MaxDrawdown": max_drawdown(pnl)
    }


# ============================================================
# GROUPS
# ============================================================

GROUPS = [
    "ALL",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6_PLUS"
]


# ============================================================
# TEST
# ============================================================

summary_rows = []
year_rows = []


for hold in HOLDS:

    pnl_col = (
        f"PnL_{hold}"
    )

    eligible = master[
        master[pnl_col].notna()
    ].copy()

    signal_mask = eligible[
        "BullCross"
    ].to_numpy(
        dtype=bool
    )

    groups = eligible[
        "PreCrossUpGroup"
    ].to_numpy()

    pnl_all = eligible[
        pnl_col
    ].to_numpy(
        dtype=float
    )

    years = eligible[
        "Year"
    ].to_numpy()


    for group in GROUPS:

        if group == "ALL":

            mask = (
                signal_mask
            )

        else:

            mask = (
                signal_mask
                &
                (groups == group)
            )

        idx = np.where(
            mask
        )[0]

        pnl = (
            pnl_all[
                idx
            ]
        )

        signal_years = (
            years[
                idx
            ]
        )

        s = stats(
            pnl
        )

        year_pfs = []

        for year in YEARS:

            ypnl = pnl[
                signal_years
                ==
                year
            ]

            ys = stats(
                ypnl
            )

            year_rows.append({
                "Group": group,
                "HoldingMinutes": hold,
                "Year": year,
                **ys
            })

            if np.isfinite(
                ys["ProfitFactor"]
            ):
                year_pfs.append(
                    ys["ProfitFactor"]
                )

        worst_pf = (
            min(year_pfs)
            if year_pfs
            else np.nan
        )

        best_pf = (
            max(year_pfs)
            if year_pfs
            else np.nan
        )

        profitable_years = sum(
            pf > 1
            for pf in year_pfs
        )

        summary_rows.append({
            "Group": group,
            "HoldingMinutes": hold,
            **s,
            "WorstYearPF": worst_pf,
            "BestYearPF": best_pf,
            "ProfitableYears": profitable_years
        })


# ============================================================
# OUTPUT
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

year_df = pd.DataFrame(
    year_rows
)

summary.to_csv(
    OUTPUT_DIR / "summary.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR / "year_by_year.csv",
    index=False
)

master[
    master["BullCross"]
].to_csv(
    OUTPUT_DIR / "signal_detail.csv"
)


# ============================================================
# DISPLAY
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.max_rows",
    5000
)

pd.set_option(
    "display.width",
    1000
)


print()
print("=" * 260)
print("PROFIT FACTOR — PRE-CROSS HISTOGRAM UP BARS")
print("=" * 260)

print(
    summary.pivot(
        index="HoldingMinutes",
        columns="Group",
        values="ProfitFactor"
    )
    .round(4)
    .to_string()
)


print()
print("=" * 260)
print("AVERAGE PNL")
print("=" * 260)

print(
    summary.pivot(
        index="HoldingMinutes",
        columns="Group",
        values="AveragePnL"
    )
    .round(4)
    .to_string()
)


print()
print("=" * 260)
print("SAMPLE SIZE")
print("=" * 260)

print(
    summary.pivot(
        index="HoldingMinutes",
        columns="Group",
        values="N"
    )
    .to_string()
)


print()
print("=" * 260)
print("WORST-YEAR PF")
print("=" * 260)

print(
    summary.pivot(
        index="HoldingMinutes",
        columns="Group",
        values="WorstYearPF"
    )
    .round(4)
    .to_string()
)


print()
print("=" * 260)
print("PROFITABLE YEARS")
print("=" * 260)

print(
    summary.pivot(
        index="HoldingMinutes",
        columns="Group",
        values="ProfitableYears"
    )
    .to_string()
)


# ============================================================
# YEAR-BY-YEAR
# ============================================================

for group in GROUPS:

    print()
    print("=" * 240)
    print(
        f"YEAR-BY-YEAR PF — PRE-CROSS UP BARS = {group}"
    )
    print("=" * 240)

    d = year_df[
        year_df["Group"] == group
    ].pivot(
        index="HoldingMinutes",
        columns="Year",
        values="ProfitFactor"
    )

    print(
        d
        .round(4)
        .to_string()
    )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 240)
print("DONE")
print("=" * 240)

print()
print(
    "Signal = bullish 15m MACD(5,36,5) histogram zero-cross"
)

print()
print(
    "PreCrossUpBars = consecutive histogram increases "
    "immediately before the zero-cross"
)

print(
    "Cross bar itself is NOT counted"
)

print()
print(
    "Groups = 0, 1, 2, 3, 4, 5, 6+"
)

print()
print(
    "NO other filters"
)

print(
    "NO PriceMove filter"
)

print(
    "NO PriorMove filter"
)

print(
    "NO range filter"
)

print(
    "NO simulated 4h filter"
)

print(
    "NO time-of-day filter"
)

print()
print(
    "Holds = 120, 150, 180, 210, 240 minutes"
)

print()
print("Output folder:")
print(OUTPUT_DIR)