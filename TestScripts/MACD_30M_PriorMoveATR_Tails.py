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
    r"\MACD_30M_PriorMoveATR_Tails"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14

PRIOR_LOOKBACK = 10      # 10 x 30m = 5 hours
LOWER_CUT = -2.0
UPPER_CUT = 2.0

SESSION_START = 6
SESSION_END = 22

HOLDS = list(range(30, 481, 30))

YEARS = [2022, 2023, 2024, 2025, 2026]

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

df1["DateTime"] = pd.to_datetime(
    df1["DateTime"]
)

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
# WILDER ATR14
# ============================================================

prev_close = df30["Close"].shift(1)

tr1 = (
    df30["High"]
    -
    df30["Low"]
)

tr2 = (
    df30["High"]
    -
    prev_close
).abs()

tr3 = (
    df30["Low"]
    -
    prev_close
).abs()

df30["TR"] = pd.concat(
    [tr1, tr2, tr3],
    axis=1
).max(axis=1)

df30["ATR14"] = df30["TR"].ewm(
    alpha=1.0 / ATR_LENGTH,
    adjust=False
).mean()


# ============================================================
# PRIOR MOVE ATR
#
# PriorMoveATR14 =
# (Close[t-1] - Close[t-10]) / ATR14[t]
# ============================================================

df30["PriorMovePoints"] = (
    df30["Close"].shift(1)
    -
    df30["Close"].shift(PRIOR_LOOKBACK)
)

df30["PriorMoveATR14"] = (
    df30["PriorMovePoints"]
    /
    df30["ATR14"]
)


# ============================================================
# THREE PRIOR-MOVE STATES
# ============================================================

conditions = [
    df30["PriorMoveATR14"] < LOWER_CUT,

    (
        (df30["PriorMoveATR14"] >= LOWER_CUT)
        &
        (df30["PriorMoveATR14"] <= UPPER_CUT)
    ),

    df30["PriorMoveATR14"] > UPPER_CUT
]

choices = [
    "NEG_TAIL",
    "MID",
    "POS_TAIL"
]

df30["PriorGroup"] = np.select(
    conditions,
    choices,
    default="UNKNOWN"
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
    &
    df30["ATR14"].notna()
    &
    df30["PriorMoveATR14"].notna()
].copy()

master["Year"] = master.index.year
master["Hour"] = master.index.hour


# ============================================================
# SIGNAL COUNTS
# ============================================================

signals = master[
    master["BullCross"]
].copy()


print()
print("=" * 180)
print("SIGNAL COUNTS")
print("=" * 180)

print(
    "All bullish crosses:",
    len(signals)
)

print()
print("Crosses by PriorMove group:")

print(
    signals["PriorGroup"]
    .value_counts()
    .sort_index()
    .to_string()
)

print()
print("Percent of signals:")

print(
    (
        signals["PriorGroup"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )
    .round(2)
    .to_string()
)


# ============================================================
# DISTRIBUTION
# ============================================================

print()
print("=" * 180)
print("PRIOR MOVE ATR DISTRIBUTION — BULLISH CROSSES")
print("=" * 180)

print(
    signals["PriorMoveATR14"]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
    .round(4)
    .to_string()
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
# ENTRY = FIRST 1M OPEN AFTER COMPLETED 30M BAR
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
    entry_positions
    <
    len(minute_times)
)

master = master.iloc[
    np.where(valid_entry)[0]
].copy()

entry_positions = entry_positions[
    valid_entry
]

master["EntryTime"] = pd.to_datetime(
    minute_times[
        entry_positions
    ]
)

master["EntryPrice"] = minute_open[
    entry_positions
]


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

        "N":
            len(pnl),

        "WinRatePct":
            np.mean(
                pnl > 0
            ) * 100,

        "NetPoints":
            pnl.sum(),

        "AveragePnL":
            pnl.mean(),

        "MedianPnL":
            np.median(
                pnl
            ),

        "ProfitFactor":
            profit_factor(
                pnl
            ),

        "SQN":
            sqn(
                pnl
            ),

        "MaxDrawdown":
            max_drawdown(
                pnl
            )
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

GROUPS = [
    "ALL",
    "NEG_TAIL",
    "MID",
    "POS_TAIL"
]


# ============================================================
# MAIN TEST
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

    prior_groups_all = eligible[
        "PriorGroup"
    ].to_numpy()


    for group in GROUPS:

        if group == "ALL":

            signal_mask = bull_all

        else:

            signal_mask = (
                bull_all
                &
                (prior_groups_all == group)
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
        # YEAR-BY-YEAR
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
                    group,

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
                year_pfs.append(
                    pf
                )


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
        # MATCHED RANDOM
        #
        # Match Year + Hour.
        # Exclude all actual bullish MACD cross bars.
        #
        # For tail groups, random entries are NOT constrained
        # to the same PriorMove state. This matches the previous
        # control design and measures whether the selected MACD
        # subset is unusual vs comparable times.
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
                    f"group={group}, "
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
                profit_factor(
                    rpnl
                )
            )

            random_sqn.append(
                sqn(
                    rpnl
                )
            )


        summary_rows.append({

            "Group":
                group,

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
                    actual[
                        "AveragePnL"
                    ]
                ),

            "MACDPercentile_PF":
                percentile_vs_random(
                    random_pf,
                    actual[
                        "ProfitFactor"
                    ]
                ),

            "MACDPercentile_SQN":
                percentile_vs_random(
                    random_sqn,
                    actual[
                        "SQN"
                    ]
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
# YEAR-BY-YEAR PF
# ============================================================

for group in GROUPS:

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

for group in GROUPS:

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
    "PriorMoveATR14 = "
    "(Close[t-1] - Close[t-10]) / ATR14[t]"
)

print(
    "10 x 30m = 5-hour prior-move lookback"
)

print()
print(
    "NEG_TAIL = PriorMoveATR14 < -2"
)

print(
    "MID      = -2 <= PriorMoveATR14 <= +2"
)

print(
    "POS_TAIL = PriorMoveATR14 > +2"
)

print()
print(
    "No 4h filter"
)

print(
    "No range filter"
)

print(
    "Holds = 30 to 480 minutes in 30-minute increments"
)

print()
print("Output folder:")
print(OUTPUT_DIR)