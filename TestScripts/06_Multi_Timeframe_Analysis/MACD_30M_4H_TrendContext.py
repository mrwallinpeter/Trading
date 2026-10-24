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
    r"\MACD_30M_4H_TrendContext"
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

HOLDS = list(
    range(
        30,
        481,
        30
    )
)

YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026
]

N_RANDOM_RUNS = 1000
RANDOM_SEED = 42


# ============================================================
# LOAD 1-MIN DATA
# ============================================================

print()
print("=" * 180)
print("LOADING 1-MIN DATA")
print("=" * 180)

df1 = pd.read_csv(
    ONE_MIN_FILE
)

df1["DateTime"] = pd.to_datetime(
    df1["DateTime"]
)

df1 = (
    df1
    .sort_values("DateTime")
    .drop_duplicates("DateTime")
    .set_index("DateTime")
)

for col in [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]:
    df1[col] = pd.to_numeric(
        df1[col],
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
# RESAMPLE FUNCTION
# ============================================================

def build_bars(freq):

    d = df1.resample(
        freq,
        label="right",
        closed="right"
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })

    return d.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close"
        ]
    )


# ============================================================
# BUILD 30M AND 4H
# ============================================================

print()
print("=" * 180)
print("BUILDING BARS")
print("=" * 180)

df30 = build_bars("30min")
df4h = build_bars("4h")

print("30m bars:", len(df30))
print("4h bars: ", len(df4h))


# ============================================================
# MACD FUNCTION
# ============================================================

def add_macd(d):

    d = d.copy()

    fast = d["Close"].ewm(
        span=MACD_FAST,
        adjust=False
    ).mean()

    slow = d["Close"].ewm(
        span=MACD_SLOW,
        adjust=False
    ).mean()

    d["MACD"] = (
        fast
        -
        slow
    )

    d["SignalLine"] = d["MACD"].ewm(
        span=MACD_SIGNAL,
        adjust=False
    ).mean()

    d["Hist"] = (
        d["MACD"]
        -
        d["SignalLine"]
    )

    d["BullCross"] = (
        (d["Hist"].shift(1) <= 0)
        &
        (d["Hist"] > 0)
    )

    return d


df30 = add_macd(df30)
df4h = add_macd(df4h)


# ============================================================
# MAP LAST COMPLETED 4H STATE ONTO EVERY 30M BAR
#
# This is causal:
# at a 30m signal timestamp we only use a 4h bar that has
# already completed at or before that timestamp.
# ============================================================

df30["Hist4H"] = (
    df4h["Hist"]
    .reindex(
        df30.index,
        method="ffill"
    )
)

df30["MACD4H"] = (
    df4h["MACD"]
    .reindex(
        df30.index,
        method="ffill"
    )
)

df30["FourHState"] = np.where(
    df30["Hist4H"] > 0,
    "BULL",
    "BEAR"
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
    df30["Hist4H"].notna()
].copy()

master["Year"] = master.index.year
master["Hour"] = master.index.hour


print()
print("=" * 180)
print("SESSION / SIGNAL COUNTS")
print("=" * 180)

print(
    f"Session: {SESSION_START:02d}:00 <= signal hour < {SESSION_END:02d}:00"
)

print(
    "30m bars:",
    len(master)
)

print(
    "Bullish 30m MACD crosses:",
    int(master["BullCross"].sum())
)

print()
print("Bullish crosses by 4h state:")

print(
    master[
        master["BullCross"]
    ]["FourHState"]
    .value_counts()
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
# ENTRY
#
# First 1m Open AFTER completed 30m signal bar
# ============================================================

signal_times_np = master.index.values.astype(
    "datetime64[ns]"
)

entry_positions = np.searchsorted(
    minute_times,
    signal_times_np,
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
# PRECOMPUTE PNL FOR EVERY HOLD
#
# Each hold has its own valid population.
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

    targets = (
        entry_times_np
        +
        np.timedelta64(
            hold,
            "m"
        )
    )

    exit_positions = np.searchsorted(
        minute_times,
        targets,
        side="left"
    )

    pnl = np.full(
        len(master),
        np.nan
    )

    valid = (
        exit_positions
        <
        len(minute_times)
    )

    valid_idx = np.where(
        valid
    )[0]

    actual_times = minute_times[
        exit_positions[
            valid_idx
        ]
    ]

    tolerance_ok = (
        actual_times
        <=
        targets[
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
        np.sqrt(
            len(pnl)
        )
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

    dd = (
        equity
        -
        peak
    )

    return abs(
        dd.min()
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
# GROUP DEFINITIONS
# ============================================================

GROUPS = {
    "ALL":
        lambda d: np.ones(
            len(d),
            dtype=bool
        ),

    "4H_BULL":
        lambda d: (
            d["FourHState"]
            ==
            "BULL"
        ).to_numpy(),

    "4H_BEAR":
        lambda d: (
            d["FourHState"]
            ==
            "BEAR"
        ).to_numpy()
}


# ============================================================
# MAIN TEST
# ============================================================

summary_rows = []
year_rows = []

rng = np.random.default_rng(
    RANDOM_SEED
)


print()
print("=" * 180)
print("RUNNING 30M × 4H CONTEXT TEST")
print("=" * 180)


for hold in HOLDS:

    pnl_col = (
        f"PnL_{hold}"
    )

    eligible = master[
        master[pnl_col].notna()
    ].copy()

    pnl_all = eligible[
        pnl_col
    ].to_numpy(
        dtype=float
    )

    bull_all = eligible[
        "BullCross"
    ].to_numpy(
        dtype=bool
    )

    years_all = eligible[
        "Year"
    ].to_numpy()

    hours_all = eligible[
        "Hour"
    ].to_numpy()

    state_all = eligible[
        "FourHState"
    ].to_numpy()


    for group_name, group_func in GROUPS.items():

        context_mask = group_func(
            eligible
        )

        signal_mask = (
            bull_all
            &
            context_mask
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

        signal_states = state_all[
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
                signal_years
                ==
                year
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
        # Match:
        # Year
        # Hour
        # 4h state
        #
        # For ALL, the state is therefore automatically
        # preserved in the random sample as well.
        # ====================================================

        strata = []


        signal_strata = list(
            zip(
                signal_years,
                signal_hours,
                signal_states
            )
        )


        unique_strata = sorted(
            set(
                signal_strata
            )
        )


        for (
            year,
            hour,
            state
        ) in unique_strata:

            required_n = int(
                np.sum(
                    (
                        signal_years
                        ==
                        year
                    )
                    &
                    (
                        signal_hours
                        ==
                        hour
                    )
                    &
                    (
                        signal_states
                        ==
                        state
                    )
                )
            )


            pool = np.where(
                (
                    years_all
                    ==
                    year
                )
                &
                (
                    hours_all
                    ==
                    hour
                )
                &
                (
                    state_all
                    ==
                    state
                )
                &
                (
                    ~bull_all
                )
            )[0]


            if len(pool) < required_n:

                raise RuntimeError(
                    f"Insufficient random pool: "
                    f"group={group_name}, "
                    f"hold={hold}, "
                    f"year={year}, "
                    f"hour={hour}, "
                    f"state={state}, "
                    f"need={required_n}, "
                    f"have={len(pool)}"
                )


            strata.append(
                (
                    pool,
                    required_n
                )
            )


        # ====================================================
        # RANDOM RUNS
        # ====================================================

        random_avg = []
        random_pf = []
        random_sqn = []


        for _ in range(
            N_RANDOM_RUNS
        ):

            chosen = []


            for (
                pool,
                required_n
            ) in strata:

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
                np.mean(
                    rpnl
                )
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


        # ====================================================
        # SUMMARY
        # ====================================================

        summary_rows.append({

            "Group":
                group_name,

            "HoldingMinutes":
                hold,

            "HoldingHours":
                hold / 60,

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
# OUTPUT DATAFRAMES
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
    OUTPUT_DIR
    /
    "summary.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "year_by_year.csv",
    index=False
)

master[
    master["BullCross"]
].to_csv(
    OUTPUT_DIR
    /
    "signal_detail.csv"
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
    2000
)

pd.set_option(
    "display.width",
    800
)


print()
print("=" * 300)
print("SUMMARY")
print("=" * 300)

print(
    summary
    .sort_values(
        [
            "Group",
            "HoldingMinutes"
        ]
    )
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# COMPACT PF TABLE
# ============================================================

print()
print("=" * 220)
print("PROFIT FACTOR SURFACE")
print("=" * 220)

pf_surface = summary.pivot(
    index="HoldingMinutes",
    columns="Group",
    values="ProfitFactor"
)

print(
    pf_surface
    .round(4)
    .to_string()
)


# ============================================================
# AVG PNL SURFACE
# ============================================================

print()
print("=" * 220)
print("AVERAGE PNL SURFACE")
print("=" * 220)

avg_surface = summary.pivot(
    index="HoldingMinutes",
    columns="Group",
    values="AveragePnL"
)

print(
    avg_surface
    .round(4)
    .to_string()
)


# ============================================================
# RANDOM PERCENTILE SURFACE
# ============================================================

print()
print("=" * 220)
print("PF PERCENTILE VS MATCHED RANDOM")
print("=" * 220)

pct_surface = summary.pivot(
    index="HoldingMinutes",
    columns="Group",
    values="MACDPercentile_PF"
)

print(
    pct_surface
    .round(1)
    .to_string()
)


# ============================================================
# WORST-YEAR PF SURFACE
# ============================================================

print()
print("=" * 220)
print("WORST-YEAR PF")
print("=" * 220)

worst_surface = summary.pivot(
    index="HoldingMinutes",
    columns="Group",
    values="WorstYearPF"
)

print(
    worst_surface
    .round(4)
    .to_string()
)


# ============================================================
# YEAR PF — 4H BULL
# ============================================================

print()
print("=" * 220)
print("YEAR-BY-YEAR PF — 4H BULL")
print("=" * 220)

bull_year = year_df[
    year_df["Group"]
    ==
    "4H_BULL"
].pivot(
    index="HoldingMinutes",
    columns="Year",
    values="ProfitFactor"
)

print(
    bull_year
    .round(4)
    .to_string()
)


# ============================================================
# YEAR PF — 4H BEAR
# ============================================================

print()
print("=" * 220)
print("YEAR-BY-YEAR PF — 4H BEAR")
print("=" * 220)

bear_year = year_df[
    year_df["Group"]
    ==
    "4H_BEAR"
].pivot(
    index="HoldingMinutes",
    columns="Year",
    values="ProfitFactor"
)

print(
    bear_year
    .round(4)
    .to_string()
)


# ============================================================
# BEST RESULTS BY GROUP
# ============================================================

print()
print("=" * 260)
print("TOP RESULTS BY GROUP — SORTED BY WORST-YEAR PF")
print("=" * 260)

for group in [
    "ALL",
    "4H_BULL",
    "4H_BEAR"
]:

    print()
    print(
        f"--- {group} ---"
    )

    d = summary[
        summary["Group"]
        ==
        group
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
        d.head(8)
        .round(4)
        .to_string(
            index=False
        )
    )


print()
print("=" * 220)
print("DONE")
print("=" * 220)

print()
print(
    "30m signal: bullish MACD(5,36,5) histogram zero-cross"
)

print(
    "4h context: last completed 4h MACD(5,36,5) histogram state"
)

print(
    "4H_BULL = Hist4H > 0"
)

print(
    "4H_BEAR = Hist4H <= 0"
)

print(
    "Session: 06:00 <= signal hour < 22:00"
)

print(
    "Holds: 30 to 480 minutes in 30-minute increments"
)

print(
    "Random control matched by Year + Hour + 4h state"
)

print()
print(
    "Output folder:"
)

print(
    OUTPUT_DIR
)