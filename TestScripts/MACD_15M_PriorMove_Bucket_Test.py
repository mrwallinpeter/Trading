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
    r"\MACD_15M_PriorMove_Bucket_Test"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14
PRIOR_LOOKBACK = 20

YEARS = [2022, 2023, 2024, 2025, 2026]

HOLDS = [
    120,
    150,
    180,
    210,
    240
]


# ============================================================
# LOAD 1-MIN DATA
# ============================================================

print()
print("=" * 180)
print("LOADING 1-MIN DATA")
print("=" * 180)

df1 = pd.read_csv(ONE_MIN_FILE)

df1["DateTime"] = pd.to_datetime(
    df1["DateTime"]
)

df1 = (
    df1
    .sort_values("DateTime")
    .drop_duplicates("DateTime")
    .set_index("DateTime")
)

for c in [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]:
    df1[c] = pd.to_numeric(
        df1[c],
        errors="coerce"
    )

df1 = df1.dropna(
    subset=[
        "Open",
        "High",
        "Low",
        "Close"
    ]
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
    subset=[
        "Open",
        "High",
        "Low",
        "Close"
    ]
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

df15["BullCross"] = (
    (df15["Hist"].shift(1) <= 0)
    &
    (df15["Hist"] > 0)
)


# ============================================================
# ATR14
#
# Wilder-style ATR using ewm alpha = 1/14
# Current cross bar ATR is used.
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
# PRIOR MOVE ATR
#
# PriorMovePoints =
# Close[t-1] - Close[t-20]
#
# PriorMoveATR14 =
# PriorMovePoints / ATR14[t]
# ============================================================

df15["PriorMovePoints"] = (
    df15["Close"].shift(1)
    -
    df15["Close"].shift(PRIOR_LOOKBACK)
)

df15["PriorMoveATR14"] = (
    df15["PriorMovePoints"]
    /
    df15["ATR14"]
)


# ============================================================
# 0.5 BUCKETS BETWEEN -2 AND +2
# ============================================================

def prior_bucket(x):

    if not np.isfinite(x):
        return "NA"

    if x < -2.0:
        return "LT_-2"

    if x < -1.5:
        return "-2_-1.5"

    if x < -1.0:
        return "-1.5_-1"

    if x < -0.5:
        return "-1_-0.5"

    if x < 0.0:
        return "-0.5_0"

    if x < 0.5:
        return "0_0.5"

    if x < 1.0:
        return "0.5_1"

    if x < 1.5:
        return "1_1.5"

    if x <= 2.0:
        return "1.5_2"

    return "GT_2"


df15["PriorBucket"] = (
    df15["PriorMoveATR14"]
    .apply(prior_bucket)
)


# ============================================================
# MASTER
# ============================================================

master = df15[
    df15.index.year.isin(YEARS)
    &
    df15["PriorMoveATR14"].notna()
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
print("PRIOR MOVE 0.5-BUCKET COUNTS")

bucket_order = [
    "LT_-2",
    "-2_-1.5",
    "-1.5_-1",
    "-1_-0.5",
    "-0.5_0",
    "0_0.5",
    "0.5_1",
    "1_1.5",
    "1.5_2",
    "GT_2"
]

print(
    signals["PriorBucket"]
    .value_counts()
    .reindex(
        bucket_order,
        fill_value=0
    )
    .to_string()
)


# ============================================================
# 1-MIN EXECUTION
# ============================================================

minute_times = df1.index.values.astype(
    "datetime64[ns]"
)

minute_open = df1["Open"].to_numpy(
    dtype=float
)

minute_close = df1["Close"].to_numpy(
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
# RETURNS
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

    final_idx = valid_idx[
        tolerance_ok
    ]

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

    gp = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()

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
# TEST DEFINITIONS
# ============================================================

BUCKET_GROUPS = [
    "-2_-1.5",
    "-1.5_-1",
    "-1_-0.5",
    "-0.5_0",
    "0_0.5",
    "0.5_1",
    "1_1.5",
    "1.5_2"
]


SPECIAL_GROUPS = [
    "ALL",
    "M2_P2",
    "CENTER_025",
    "CENTER_050",
    "EXCL_CENTER_025",
    "EXCL_CENTER_050"
]


# ============================================================
# BUILD MASK
# ============================================================

def make_mask(df, group):

    bull = df[
        "BullCross"
    ].to_numpy(
        dtype=bool
    )

    pm = df[
        "PriorMoveATR14"
    ].to_numpy(
        dtype=float
    )

    buckets = df[
        "PriorBucket"
    ].to_numpy()

    if group == "ALL":

        return bull


    if group in BUCKET_GROUPS:

        return (
            bull
            &
            (buckets == group)
        )


    if group == "M2_P2":

        return (
            bull
            &
            (pm >= -2.0)
            &
            (pm <= 2.0)
        )


    if group == "CENTER_025":

        return (
            bull
            &
            (pm >= -0.25)
            &
            (pm <= 0.25)
        )


    if group == "CENTER_050":

        return (
            bull
            &
            (pm >= -0.50)
            &
            (pm <= 0.50)
        )


    if group == "EXCL_CENTER_025":

        return (
            bull
            &
            (pm >= -2.0)
            &
            (pm <= 2.0)
            &
            (np.abs(pm) > 0.25)
        )


    if group == "EXCL_CENTER_050":

        return (
            bull
            &
            (pm >= -2.0)
            &
            (pm <= 2.0)
            &
            (np.abs(pm) > 0.50)
        )


    raise ValueError(
        f"Unknown group: {group}"
    )


# ============================================================
# RUN TESTS
# ============================================================

ALL_GROUPS = (
    ["ALL"]
    +
    BUCKET_GROUPS
    +
    [
        "M2_P2",
        "CENTER_025",
        "CENTER_050",
        "EXCL_CENTER_025",
        "EXCL_CENTER_050"
    ]
)

summary_rows = []
year_rows = []


for hold in HOLDS:

    pnl_col = f"PnL_{hold}"

    eligible = master[
        master[pnl_col].notna()
    ].copy()

    pnl_all = eligible[
        pnl_col
    ].to_numpy(
        dtype=float
    )

    years = eligible[
        "Year"
    ].to_numpy()


    for group in ALL_GROUPS:

        mask = make_mask(
            eligible,
            group
        )

        idx = np.where(
            mask
        )[0]

        pnl = pnl_all[
            idx
        ]

        signal_years = years[
            idx
        ]

        s = stats(
            pnl
        )

        year_pfs = []

        for year in YEARS:

            ypnl = pnl[
                signal_years == year
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

        profitable_years = sum(
            pf > 1
            for pf in year_pfs
        )

        summary_rows.append({
            "Group": group,
            "HoldingMinutes": hold,
            **s,
            "WorstYearPF": worst_pf,
            "ProfitableYears": profitable_years
        })


# ============================================================
# DATAFRAMES
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# SAVE
# ============================================================

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
# DISPLAY SETTINGS
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


# ============================================================
# 0.5 BUCKET RESULTS
# ============================================================

bucket_summary = summary[
    summary["Group"].isin(
        BUCKET_GROUPS
    )
].copy()


print()
print("=" * 260)
print("PRIOR MOVE ATR14 — 0.5 BUCKETS — PROFIT FACTOR")
print("=" * 260)

print(
    bucket_summary.pivot(
        index="HoldingMinutes",
        columns="Group",
        values="ProfitFactor"
    )
    .reindex(
        columns=BUCKET_GROUPS
    )
    .round(4)
    .to_string()
)


print()
print("=" * 260)
print("PRIOR MOVE ATR14 — 0.5 BUCKETS — AVERAGE PNL")
print("=" * 260)

print(
    bucket_summary.pivot(
        index="HoldingMinutes",
        columns="Group",
        values="AveragePnL"
    )
    .reindex(
        columns=BUCKET_GROUPS
    )
    .round(4)
    .to_string()
)


print()
print("=" * 260)
print("PRIOR MOVE ATR14 — 0.5 BUCKETS — SAMPLE SIZE")
print("=" * 260)

print(
    bucket_summary.pivot(
        index="HoldingMinutes",
        columns="Group",
        values="N"
    )
    .reindex(
        columns=BUCKET_GROUPS
    )
    .to_string()
)


print()
print("=" * 260)
print("PRIOR MOVE ATR14 — 0.5 BUCKETS — WORST-YEAR PF")
print("=" * 260)

print(
    bucket_summary.pivot(
        index="HoldingMinutes",
        columns="Group",
        values="WorstYearPF"
    )
    .reindex(
        columns=BUCKET_GROUPS
    )
    .round(4)
    .to_string()
)


# ============================================================
# CENTRAL / EXCLUSION TESTS
# ============================================================

special_order = [
    "ALL",
    "M2_P2",
    "CENTER_025",
    "CENTER_050",
    "EXCL_CENTER_025",
    "EXCL_CENTER_050"
]

special = summary[
    summary["Group"].isin(
        special_order
    )
].copy()


print()
print("=" * 260)
print("CENTRAL REGION / EXCLUSION TEST — PROFIT FACTOR")
print("=" * 260)

print(
    special.pivot(
        index="HoldingMinutes",
        columns="Group",
        values="ProfitFactor"
    )
    .reindex(
        columns=special_order
    )
    .round(4)
    .to_string()
)


print()
print("=" * 260)
print("CENTRAL REGION / EXCLUSION TEST — AVERAGE PNL")
print("=" * 260)

print(
    special.pivot(
        index="HoldingMinutes",
        columns="Group",
        values="AveragePnL"
    )
    .reindex(
        columns=special_order
    )
    .round(4)
    .to_string()
)


print()
print("=" * 260)
print("CENTRAL REGION / EXCLUSION TEST — SAMPLE SIZE")
print("=" * 260)

print(
    special.pivot(
        index="HoldingMinutes",
        columns="Group",
        values="N"
    )
    .reindex(
        columns=special_order
    )
    .to_string()
)


print()
print("=" * 260)
print("CENTRAL REGION / EXCLUSION TEST — WORST-YEAR PF")
print("=" * 260)

print(
    special.pivot(
        index="HoldingMinutes",
        columns="Group",
        values="WorstYearPF"
    )
    .reindex(
        columns=special_order
    )
    .round(4)
    .to_string()
)


print()
print("=" * 260)
print("CENTRAL REGION / EXCLUSION TEST — PROFITABLE YEARS")
print("=" * 260)

print(
    special.pivot(
        index="HoldingMinutes",
        columns="Group",
        values="ProfitableYears"
    )
    .reindex(
        columns=special_order
    )
    .to_string()
)


# ============================================================
# YEAR-BY-YEAR FOR SPECIAL GROUPS
# ============================================================

for group in special_order:

    print()
    print("=" * 240)
    print(
        f"YEAR-BY-YEAR PF — {group}"
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
    "PriorMoveATR14 = "
    "(Close[t-1] - Close[t-20]) / ATR14[t]"
)

print()
print("0.5 buckets tested from -2 to +2")

print()
print("Special groups:")

print(
    "M2_P2            = -2 <= PriorMove <= +2"
)

print(
    "CENTER_025       = -0.25 <= PriorMove <= +0.25"
)

print(
    "CENTER_050       = -0.50 <= PriorMove <= +0.50"
)

print(
    "EXCL_CENTER_025  = -2..+2 excluding -0.25..+0.25"
)

print(
    "EXCL_CENTER_050  = -2..+2 excluding -0.50..+0.50"
)

print()
print("NO PriceMove filter")
print("NO histogram up-bar filter")
print("NO range filter")
print("NO 4h filter")
print("NO time filter")

print()
print(
    "Holds = 120, 150, 180, 210, 240 minutes"
)

print()
print("Output folder:")
print(OUTPUT_DIR)