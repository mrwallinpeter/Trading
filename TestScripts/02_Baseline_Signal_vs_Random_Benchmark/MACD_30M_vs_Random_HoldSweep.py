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
    r"\MACD_30M_vs_Random_HoldSweep_FAST"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TIMEFRAME = "30min"

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

HOLD_MINUTES_LIST = [
    60,
    120,
    180,
    240,
    300,
    360,
    480
]

N_RANDOM_RUNS = 1000
RANDOM_SEED = 42

YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026
]


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
# BUILD 30-MIN BARS
# ============================================================

print()
print("=" * 180)
print("BUILDING 30-MIN BARS")
print("=" * 180)

df30 = df1.resample(
    TIMEFRAME,
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
    subset=[
        "Open",
        "High",
        "Low",
        "Close"
    ]
)

print("30m bars:", len(df30))


# ============================================================
# MACD
# ============================================================

df30["FastEMA"] = df30["Close"].ewm(
    span=MACD_FAST,
    adjust=False
).mean()

df30["SlowEMA"] = df30["Close"].ewm(
    span=MACD_SLOW,
    adjust=False
).mean()

df30["MACD"] = (
    df30["FastEMA"]
    -
    df30["SlowEMA"]
)

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
# PREP NUMPY ARRAYS FOR FAST EXIT LOOKUPS
# ============================================================

print()
print("=" * 180)
print("PREPARING FAST 1-MIN ARRAYS")
print("=" * 180)

minute_times = df1.index.values
minute_open = df1["Open"].to_numpy(
    dtype=float
)
minute_close = df1["Close"].to_numpy(
    dtype=float
)


# ============================================================
# BUILD ELIGIBLE 30M BAR TABLE
# ============================================================

print()
print("=" * 180)
print("BUILDING ELIGIBLE 30-MIN TABLE")
print("=" * 180)

signal_times = df30.index[
    df30.index.year.isin(YEARS)
]

signal_times_np = signal_times.values


# First 1m timestamp strictly after completed 30m bar
entry_positions = np.searchsorted(
    minute_times,
    signal_times_np,
    side="right"
)

valid_entry_mask = (
    entry_positions
    <
    len(minute_times)
)

signal_times = signal_times[
    valid_entry_mask
]

signal_times_np = signal_times.values

entry_positions = entry_positions[
    valid_entry_mask
]

entry_times_np = minute_times[
    entry_positions
]

entry_prices = minute_open[
    entry_positions
]


eligible = pd.DataFrame({
    "SignalTime": signal_times,
    "Year": signal_times.year,
    "Hour": signal_times.hour,
    "EntryPos": entry_positions,
    "EntryTime": pd.to_datetime(
        entry_times_np
    ),
    "EntryPrice": entry_prices,
    "BullCross": df30.loc[
        signal_times,
        "BullCross"
    ].to_numpy(dtype=bool)
})


print(
    "Eligible 30m bars before hold validation:",
    len(eligible)
)

print(
    "MACD bullish crosses:",
    int(
        eligible["BullCross"].sum()
    )
)


# ============================================================
# PRECOMPUTE ALL FORWARD PNL COLUMNS
# ============================================================

print()
print("=" * 180)
print("PRECOMPUTING ALL FORWARD RETURNS")
print("=" * 180)

entry_times_np = eligible[
    "EntryTime"
].values.astype(
    "datetime64[ns]"
)

entry_prices_np = eligible[
    "EntryPrice"
].to_numpy(
    dtype=float
)


for hold_minutes in HOLD_MINUTES_LIST:

    print(
        f"Precomputing {hold_minutes} min..."
    )

    target_times = (
        entry_times_np
        +
        np.timedelta64(
            hold_minutes,
            "m"
        )
    )


    exit_positions = np.searchsorted(
        minute_times,
        target_times,
        side="left"
    )


    valid_exit = (
        exit_positions
        <
        len(minute_times)
    )


    exit_times = np.full(
        len(eligible),
        np.datetime64("NaT"),
        dtype="datetime64[ns]"
    )


    valid_pos = exit_positions[
        valid_exit
    ]

    exit_times[
        valid_exit
    ] = minute_times[
        valid_pos
    ]


    # Require exit <= target + 2 minutes
    tolerance_ok = np.zeros(
        len(eligible),
        dtype=bool
    )

    tolerance_ok[
        valid_exit
    ] = (
        exit_times[
            valid_exit
        ]
        <=
        target_times[
            valid_exit
        ]
        +
        np.timedelta64(
            2,
            "m"
        )
    )


    final_valid = (
        valid_exit
        &
        tolerance_ok
    )


    exit_prices = np.full(
        len(eligible),
        np.nan,
        dtype=float
    )

    exit_prices[
        final_valid
    ] = minute_close[
        exit_positions[
            final_valid
        ]
    ]


    pnl = (
        exit_prices
        -
        entry_prices_np
    )


    eligible[
        f"PnL_{hold_minutes}"
    ] = pnl


# ============================================================
# KEEP ONLY ROWS VALID FOR ALL HOLD PERIODS
#
# This keeps MACD/random sample membership identical across holds.
# ============================================================

pnl_cols = [
    f"PnL_{h}"
    for h in HOLD_MINUTES_LIST
]

eligible = eligible.dropna(
    subset=pnl_cols
).copy()

eligible = eligible.reset_index(
    drop=True
)


print()
print(
    "Eligible bars valid for every hold:",
    len(eligible)
)

print(
    "Completed MACD crosses valid for every hold:",
    int(
        eligible["BullCross"].sum()
    )
)


# ============================================================
# STAT FUNCTIONS
# ============================================================

def profit_factor_array(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

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


def sqn_array(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    n = len(pnl)

    if n < 2:
        return np.nan

    std = pnl.std(
        ddof=1
    )

    if std == 0:
        return np.nan

    return (
        np.sqrt(n)
        *
        pnl.mean()
        /
        std
    )


def max_drawdown_array(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

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


def calculate_stats_array(pnl):

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
            profit_factor_array(
                pnl
            ),

        "SQN":
            sqn_array(
                pnl
            ),

        "MaxDrawdown":
            max_drawdown_array(
                pnl
            )
    }


# ============================================================
# SPLIT MACD / RANDOM POOL
# ============================================================

macd_mask = eligible[
    "BullCross"
].to_numpy(
    dtype=bool
)

random_mask = ~macd_mask

macd_indices = np.where(
    macd_mask
)[0]

random_indices = np.where(
    random_mask
)[0]


# ============================================================
# STRATA
# ============================================================

macd_strata = (
    eligible.loc[
        macd_indices,
        [
            "Year",
            "Hour"
        ]
    ]
    .groupby(
        [
            "Year",
            "Hour"
        ]
    )
    .size()
    .reset_index(
        name="RequiredN"
    )
)


# ============================================================
# PREBUILD RANDOM POOL INDICES BY STRATUM
# ============================================================

pool_lookup = {}


for _, r in macd_strata.iterrows():

    year = int(
        r["Year"]
    )

    hour = int(
        r["Hour"]
    )

    required_n = int(
        r["RequiredN"]
    )


    pool_idx = eligible.index[
        (
            eligible["Year"]
            ==
            year
        )
        &
        (
            eligible["Hour"]
            ==
            hour
        )
        &
        (
            ~eligible["BullCross"]
        )
    ].to_numpy()


    if len(pool_idx) < required_n:

        raise RuntimeError(
            f"Insufficient pool "
            f"Year={year}, Hour={hour}: "
            f"need {required_n}, "
            f"have {len(pool_idx)}"
        )


    pool_lookup[
        (
            year,
            hour
        )
    ] = (
        pool_idx,
        required_n
    )


# ============================================================
# DRAW RANDOM SETS ONCE
# ============================================================

print()
print("=" * 180)
print(
    f"DRAWING {N_RANDOM_RUNS} MATCHED RANDOM SETS"
)
print("=" * 180)

rng = np.random.default_rng(
    RANDOM_SEED
)

random_sets = np.empty(
    (
        N_RANDOM_RUNS,
        len(macd_indices)
    ),
    dtype=np.int32
)


for run in range(
    N_RANDOM_RUNS
):

    cursor = 0


    for (
        year,
        hour
    ), (
        pool_idx,
        required_n
    ) in pool_lookup.items():

        chosen = rng.choice(
            pool_idx,
            size=required_n,
            replace=False
        )


        random_sets[
            run,
            cursor:
            cursor + required_n
        ] = chosen


        cursor += required_n


print(
    "Random matrix shape:",
    random_sets.shape
)


# ============================================================
# HOLD SWEEP
# ============================================================

print()
print("=" * 180)
print("RUNNING FAST HOLD SWEEP")
print("=" * 180)

summary_rows = []
year_rows = []
random_rows = []


years_array = eligible[
    "Year"
].to_numpy()


for hold_minutes in HOLD_MINUTES_LIST:

    print(
        f"Testing {hold_minutes} min..."
    )

    pnl_all = eligible[
        f"PnL_{hold_minutes}"
    ].to_numpy(
        dtype=float
    )


    # ========================================================
    # ACTUAL MACD
    # ========================================================

    macd_pnl = pnl_all[
        macd_indices
    ]

    macd_stats = calculate_stats_array(
        macd_pnl
    )


    macd_years = years_array[
        macd_indices
    ]

    year_pfs = []


    for year in YEARS:

        ypnl = macd_pnl[
            macd_years
            ==
            year
        ]

        ys = calculate_stats_array(
            ypnl
        )

        year_rows.append({

            "HoldingMinutes":
                hold_minutes,

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


    # ========================================================
    # RANDOM MATRIX
    # ========================================================

    random_pnl_matrix = pnl_all[
        random_sets
    ]


    random_net = (
        random_pnl_matrix.sum(
            axis=1
        )
    )

    random_avg = (
        random_pnl_matrix.mean(
            axis=1
        )
    )

    random_win = (
        (
            random_pnl_matrix
            >
            0
        )
        .mean(
            axis=1
        )
        *
        100
    )


    gross_profit = np.where(
        random_pnl_matrix > 0,
        random_pnl_matrix,
        0
    ).sum(
        axis=1
    )

    gross_loss = -np.where(
        random_pnl_matrix < 0,
        random_pnl_matrix,
        0
    ).sum(
        axis=1
    )


    random_pf = np.divide(
        gross_profit,
        gross_loss,
        out=np.full(
            N_RANDOM_RUNS,
            np.nan
        ),
        where=(
            gross_loss
            !=
            0
        )
    )


    random_std = random_pnl_matrix.std(
        axis=1,
        ddof=1
    )

    random_sqn = (
        np.sqrt(
            random_pnl_matrix.shape[1]
        )
        *
        random_avg
        /
        random_std
    )


    # ========================================================
    # RANDOM PERCENTILES
    # ========================================================

    def pct_le(values, actual):

        values = values[
            np.isfinite(
                values
            )
        ]

        if len(values) == 0:
            return np.nan

        return (
            np.mean(
                values
                <=
                actual
            )
            *
            100
        )


    summary_rows.append({

        "HoldingMinutes":
            hold_minutes,

        **macd_stats,

        "WorstYearPF":
            worst_year_pf,

        "BestYearPF":
            best_year_pf,

        "ProfitableYears":
            profitable_years,

        "RandomMedianAvg":
            np.median(
                random_avg
            ),

        "RandomMedianPF":
            np.nanmedian(
                random_pf
            ),

        "RandomMedianNet":
            np.median(
                random_net
            ),

        "RandomMedianSQN":
            np.nanmedian(
                random_sqn
            ),

        "MACDPercentile_Avg":
            pct_le(
                random_avg,
                macd_stats[
                    "AveragePnL"
                ]
            ),

        "MACDPercentile_PF":
            pct_le(
                random_pf,
                macd_stats[
                    "ProfitFactor"
                ]
            ),

        "MACDPercentile_Net":
            pct_le(
                random_net,
                macd_stats[
                    "NetPoints"
                ]
            ),

        "MACDPercentile_SQN":
            pct_le(
                random_sqn,
                macd_stats[
                    "SQN"
                ]
            ),

        "MACDPercentile_WinRate":
            pct_le(
                random_win,
                macd_stats[
                    "WinRatePct"
                ]
            )
    })


    # Save individual random-run values
    for run in range(
        N_RANDOM_RUNS
    ):

        random_rows.append({

            "Run":
                run + 1,

            "HoldingMinutes":
                hold_minutes,

            "WinRatePct":
                random_win[
                    run
                ],

            "NetPoints":
                random_net[
                    run
                ],

            "AveragePnL":
                random_avg[
                    run
                ],

            "ProfitFactor":
                random_pf[
                    run
                ],

            "SQN":
                random_sqn[
                    run
                ]
        })


summary_df = pd.DataFrame(
    summary_rows
)

year_df = pd.DataFrame(
    year_rows
)

random_df = pd.DataFrame(
    random_rows
)


# ============================================================
# YEAR PF MATRIX
# ============================================================

year_pf_matrix = year_df.pivot_table(
    index="HoldingMinutes",
    columns="Year",
    values="ProfitFactor"
)


# ============================================================
# SAVE
# ============================================================

eligible.to_csv(
    OUTPUT_DIR
    /
    "eligible_30m_forward_returns.csv",
    index=False
)

summary_df.to_csv(
    OUTPUT_DIR
    /
    "hold_sweep_summary.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "hold_sweep_year_by_year.csv",
    index=False
)

year_pf_matrix.to_csv(
    OUTPUT_DIR
    /
    "hold_sweep_year_pf_matrix.csv"
)

random_df.to_csv(
    OUTPUT_DIR
    /
    "random_runs_all_holds.csv",
    index=False
)

macd_strata.to_csv(
    OUTPUT_DIR
    /
    "macd_year_hour_distribution.csv",
    index=False
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
    1000
)

pd.set_option(
    "display.width",
    700
)


print()
print("=" * 260)
print("30M RAW MACD VS MATCHED RANDOM — FAST HOLD SWEEP")
print("=" * 260)

print(
    summary_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 260)
print("MACD YEAR-BY-YEAR PROFIT FACTOR")
print("=" * 260)

print(
    year_pf_matrix
    .round(4)
    .to_string()
)


print()
print("=" * 260)
print("KEY COMPARISON")
print("=" * 260)

key_cols = [
    "HoldingMinutes",
    "N",
    "AveragePnL",
    "ProfitFactor",
    "WorstYearPF",
    "ProfitableYears",
    "RandomMedianAvg",
    "RandomMedianPF",
    "MACDPercentile_Avg",
    "MACDPercentile_PF"
]

print(
    summary_df[
        key_cols
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 260)
print("DONE")
print("=" * 260)

print()
print(
    "MACD trades:",
    len(macd_indices)
)

print(
    "Random simulations:",
    N_RANDOM_RUNS
)

print(
    "Holding periods:",
    HOLD_MINUTES_LIST
)

print()
print("Output folder:")
print(OUTPUT_DIR)