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
    r"\MACD_30M_NoRange_Test"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

SESSION_START = 6
SESSION_END = 22

YEARS = [2022, 2023, 2024, 2025, 2026]

HOLDS = list(range(30, 481, 30))

RANGE_LOOKBACK = 10
RANGE_MAX_HEIGHT = 25.0
RANGE_MAX_DRIFT = 10.0

N_RANDOM_RUNS = 1000
RANDOM_SEED = 42


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
    df1[c] = pd.to_numeric(df1[c], errors="coerce")

df1 = df1.dropna(
    subset=["Open", "High", "Low", "Close"]
)

print("Rows:", len(df1))
print("Start:", df1.index.min())
print("End:  ", df1.index.max())


# ============================================================
# BUILD 30M BARS
# ============================================================

df30 = df1.resample(
    "30min",
    label="right",
    closed="right"
).agg({
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum"
})

df30 = df30.dropna(
    subset=["Open", "High", "Low", "Close"]
)

print()
print("30m bars:", len(df30))


# ============================================================
# MACD
# ============================================================

fast = df30["Close"].ewm(
    span=MACD_FAST,
    adjust=False
).mean()

slow = df30["Close"].ewm(
    span=MACD_SLOW,
    adjust=False
).mean()

df30["MACD"] = fast - slow

df30["SignalLine"] = df30["MACD"].ewm(
    span=MACD_SIGNAL,
    adjust=False
).mean()

df30["Hist"] = (
    df30["MACD"]
    -
    df30["SignalLine"]
)

df30["BullCross"] = (
    (df30["Hist"].shift(1) <= 0)
    &
    (df30["Hist"] > 0)
)


# ============================================================
# 30M RANGE FILTER
#
# Previous 10 completed bars = 5 hours
# ============================================================

df30["RangeHigh"] = (
    df30["High"]
    .shift(1)
    .rolling(RANGE_LOOKBACK)
    .max()
)

df30["RangeLow"] = (
    df30["Low"]
    .shift(1)
    .rolling(RANGE_LOOKBACK)
    .min()
)

df30["RangeHeight"] = (
    df30["RangeHigh"]
    -
    df30["RangeLow"]
)

df30["RangeDrift"] = (
    df30["Close"].shift(1)
    -
    df30["Close"].shift(RANGE_LOOKBACK)
).abs()

df30["ValidRange"] = (
    (df30["RangeHeight"] <= RANGE_MAX_HEIGHT)
    &
    (df30["RangeDrift"] <= RANGE_MAX_DRIFT)
)

df30["InsideValidRange"] = (
    df30["ValidRange"]
    &
    (df30["Close"] >= df30["RangeLow"])
    &
    (df30["Close"] <= df30["RangeHigh"])
)

df30["NoRangePass"] = (
    ~df30["InsideValidRange"]
)


# ============================================================
# SESSION UNIVERSE
# ============================================================

master = df30[
    df30.index.year.isin(YEARS)
    &
    (df30.index.hour >= SESSION_START)
    &
    (df30.index.hour < SESSION_END)
].copy()

master["Year"] = master.index.year
master["Hour"] = master.index.hour


print()
print("=" * 180)
print("SIGNAL COUNTS")
print("=" * 180)

all_crosses = master[
    master["BullCross"]
]

not_range_crosses = master[
    master["BullCross"]
    &
    master["NoRangePass"]
]

removed_crosses = master[
    master["BullCross"]
    &
    master["InsideValidRange"]
]

print("All bullish crosses:      ", len(all_crosses))
print("NOT_RANGE crosses:        ", len(not_range_crosses))
print("REMOVED_RANGE crosses:    ", len(removed_crosses))
print(
    "Removed %:",
    round(
        100 * len(removed_crosses) / len(all_crosses),
        2
    )
)


# ============================================================
# 1M EXECUTION ARRAYS
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


# ============================================================
# ENTRY
# ============================================================

bar_times_np = master.index.values.astype(
    "datetime64[ns]"
)

entry_positions = np.searchsorted(
    minute_times,
    bar_times_np,
    side="right"
)

valid_entry = (
    entry_positions < len(minute_times)
)

master = master.iloc[
    np.where(valid_entry)[0]
].copy()

entry_positions = entry_positions[
    valid_entry
]

master["EntryTime"] = pd.to_datetime(
    minute_times[entry_positions]
)

master["EntryPrice"] = minute_open[
    entry_positions
]


# ============================================================
# PRECOMPUTE HOLD RETURNS
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
        np.timedelta64(hold, "m")
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
        exit_positions < len(minute_times)
    )

    valid_idx = np.where(
        valid_exit
    )[0]

    actual_exit_times = minute_times[
        exit_positions[valid_idx]
    ]

    tolerance_ok = (
        actual_exit_times
        <=
        target_times[valid_idx]
        +
        np.timedelta64(2, "m")
    )

    final_idx = valid_idx[
        tolerance_ok
    ]

    pnl[final_idx] = (
        minute_close[
            exit_positions[final_idx]
        ]
        -
        entry_prices_np[final_idx]
    )

    master[f"PnL_{hold}"] = pnl

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

    sd = pnl.std(ddof=1)

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

    equity = np.concatenate([
        [0.0],
        np.cumsum(pnl)
    ])

    peak = np.maximum.accumulate(
        equity
    )

    return abs(
        (equity - peak).min()
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

        "N":
            len(pnl),

        "WinRatePct":
            np.mean(pnl > 0) * 100,

        "NetPoints":
            pnl.sum(),

        "AveragePnL":
            pnl.mean(),

        "MedianPnL":
            np.median(pnl),

        "ProfitFactor":
            profit_factor(pnl),

        "SQN":
            sqn(pnl),

        "MaxDrawdown":
            max_drawdown(pnl)
    }


def percentile_vs_random(
    random_values,
    actual
):

    r = np.asarray(
        random_values,
        dtype=float
    )

    r = r[
        np.isfinite(r)
    ]

    if len(r) == 0:
        return np.nan

    return (
        np.mean(
            r <= actual
        )
        *
        100
    )


# ============================================================
# GROUPS
# ============================================================

GROUPS = {
    "ALL":
        lambda d: d["BullCross"].to_numpy(
            dtype=bool
        ),

    "NOT_RANGE":
        lambda d: (
            d["BullCross"]
            &
            d["NoRangePass"]
        ).to_numpy(
            dtype=bool
        ),

    "REMOVED_RANGE":
        lambda d: (
            d["BullCross"]
            &
            d["InsideValidRange"]
        ).to_numpy(
            dtype=bool
        )
}


# ============================================================
# TEST
# ============================================================

summary_rows = []
year_rows = []

rng = np.random.default_rng(
    RANDOM_SEED
)


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

    years_all = eligible[
        "Year"
    ].to_numpy()

    hours_all = eligible[
        "Hour"
    ].to_numpy()

    bull_all = eligible[
        "BullCross"
    ].to_numpy(
        dtype=bool
    )


    for group_name, group_func in GROUPS.items():

        signal_mask = group_func(
            eligible
        )

        signal_idx = np.where(
            signal_mask
        )[0]

        signal_pnl = pnl_all[
            signal_idx
        ]

        signal_years = years_all[
            signal_idx
        ]

        signal_hours = hours_all[
            signal_idx
        ]

        actual = stats(
            signal_pnl
        )


        # ====================================================
        # YEAR STABILITY
        # ====================================================

        year_pfs = []

        for year in YEARS:

            ypnl = signal_pnl[
                signal_years == year
            ]

            ys = stats(
                ypnl
            )

            year_rows.append({

                "Group":
                    group_name,

                "HoldingMinutes":
                    hold,

                "Year":
                    year,

                **ys
            })

            pf = ys[
                "ProfitFactor"
            ]

            if np.isfinite(pf):
                year_pfs.append(pf)


        worst_year_pf = (
            min(year_pfs)
            if year_pfs
            else np.nan
        )

        best_year_pf = (
            max(year_pfs)
            if year_pfs
            else np.nan
        )

        profitable_years = sum(
            pf > 1
            for pf in year_pfs
        )


        # ====================================================
        # RANDOM CONTROL
        #
        # Match Year + Hour
        # Exclude ALL MACD cross bars
        # ====================================================

        strata = []

        unique_pairs = sorted(
            set(
                zip(
                    signal_years,
                    signal_hours
                )
            )
        )


        for year, hour in unique_pairs:

            required_n = int(
                np.sum(
                    (signal_years == year)
                    &
                    (signal_hours == hour)
                )
            )

            pool = np.where(
                (years_all == year)
                &
                (hours_all == hour)
                &
                (~bull_all)
            )[0]

            if len(pool) < required_n:

                raise RuntimeError(
                    f"Insufficient random pool: "
                    f"group={group_name}, "
                    f"hold={hold}, "
                    f"year={year}, "
                    f"hour={hour}, "
                    f"need={required_n}, "
                    f"have={len(pool)}"
                )

            strata.append(
                (
                    pool,
                    required_n
                )
            )


        random_avg = []
        random_pf = []
        random_sqn = []


        for _ in range(
            N_RANDOM_RUNS
        ):

            chosen = []

            for pool, required_n in strata:

                picked = rng.choice(
                    pool,
                    size=required_n,
                    replace=False
                )

                chosen.extend(
                    picked.tolist()
                )

            rpnl = pnl_all[
                chosen
            ]

            random_avg.append(
                rpnl.mean()
            )

            random_pf.append(
                profit_factor(rpnl)
            )

            random_sqn.append(
                sqn(rpnl)
            )


        summary_rows.append({

            "Group":
                group_name,

            "HoldingMinutes":
                hold,

            "HoldingHours":
                hold / 60.0,

            **actual,

            "WorstYearPF":
                worst_year_pf,

            "BestYearPF":
                best_year_pf,

            "ProfitableYears":
                profitable_years,

            "RandomMedianAvg":
                np.nanmedian(
                    random_avg
                ),

            "RandomMedianPF":
                np.nanmedian(
                    random_pf
                ),

            "MACDPercentile_Avg":
                percentile_vs_random(
                    random_avg,
                    actual["AveragePnL"]
                ),

            "MACDPercentile_PF":
                percentile_vs_random(
                    random_pf,
                    actual["ProfitFactor"]
                ),

            "MACDPercentile_SQN":
                percentile_vs_random(
                    random_sqn,
                    actual["SQN"]
                )
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
    3000
)

pd.set_option(
    "display.width",
    900
)


print()
print("=" * 260)
print("PROFIT FACTOR SURFACE")
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
print("AVERAGE PNL SURFACE")
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


print()
print("=" * 260)
print("PF PERCENTILE VS MATCHED RANDOM")
print("=" * 260)

print(
    summary.pivot(
        index="HoldingMinutes",
        columns="Group",
        values="MACDPercentile_PF"
    )
    .round(1)
    .to_string()
)


# ============================================================
# YEAR-BY-YEAR
# ============================================================

for group in [
    "ALL",
    "NOT_RANGE",
    "REMOVED_RANGE"
]:

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
# TOP RESULTS
# ============================================================

print()
print("=" * 280)
print("TOP RESULTS BY GROUP — SORTED BY WORST-YEAR PF")
print("=" * 280)

for group in [
    "ALL",
    "NOT_RANGE",
    "REMOVED_RANGE"
]:

    print()
    print(
        f"--- {group} ---"
    )

    d = summary[
        summary["Group"] == group
    ].sort_values(
        [
            "WorstYearPF",
            "MACDPercentile_PF"
        ],
        ascending=[
            False,
            False
        ]
    )

    print(
        d.head(10)
        .round(4)
        .to_string(
            index=False
        )
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
    "Signal = bullish 30m MACD(5,36,5) histogram zero-cross"
)

print(
    "Session = 06:00 <= signal hour < 22:00"
)

print(
    "Range = previous 10 completed 30m bars"
)

print(
    "Valid range = height <= 25 points AND drift <= 10 points"
)

print(
    "NOT_RANGE rejects only crosses whose close remains inside a valid range"
)

print(
    "No 4h filter is used"
)

print(
    "Holds = 30 to 480 minutes in 30-minute increments"
)

print()
print("Output folder:")
print(OUTPUT_DIR)