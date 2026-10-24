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
    r"\MACD_30M_ParamSweep_vs_Random_FAST"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TIMEFRAME = "30min"

MACD_CONFIGS = [
    (3, 18, 3),
    (4, 24, 4),
    (5, 36, 5),
    (6, 24, 4),
    (8, 36, 5),
]

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
# PREP FAST NUMPY ARRAYS
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


# ============================================================
# BUILD MASTER ELIGIBLE 30M TABLE
# ============================================================

print()
print("=" * 180)
print("BUILDING MASTER 30-MIN ENTRY TABLE")
print("=" * 180)

signal_times = df30.index[
    df30.index.year.isin(
        YEARS
    )
]

signal_times_np = signal_times.values.astype(
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

signal_times = signal_times[
    valid_entry
]

signal_times_np = signal_times.values.astype(
    "datetime64[ns]"
)

entry_positions = entry_positions[
    valid_entry
]

entry_times = minute_times[
    entry_positions
]

entry_prices = minute_open[
    entry_positions
]


eligible = pd.DataFrame({
    "SignalTime": signal_times,
    "Year": signal_times.year,
    "Hour": signal_times.hour,
    "EntryTime": pd.to_datetime(
        entry_times
    ),
    "EntryPrice": entry_prices
})


# ============================================================
# PRECOMPUTE FORWARD PNL FOR ALL HOLD PERIODS
# ============================================================

print()
print("=" * 180)
print("PRECOMPUTING FORWARD RETURNS")
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

    pnl = np.full(
        len(eligible),
        np.nan,
        dtype=float
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

    final_exit_positions = exit_positions[
        final_idx
    ]

    pnl[
        final_idx
    ] = (
        minute_close[
            final_exit_positions
        ]
        -
        entry_prices_np[
            final_idx
        ]
    )

    eligible[
        f"PnL_{hold_minutes}"
    ] = pnl


# Keep bars valid for all hold periods
pnl_cols = [
    f"PnL_{h}"
    for h in HOLD_MINUTES_LIST
]

eligible = eligible.dropna(
    subset=pnl_cols
).reset_index(
    drop=True
)

print(
    "Eligible bars valid for all holds:",
    len(eligible)
)


# ============================================================
# STAT FUNCTIONS
# ============================================================

def profit_factor(pnl):

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


def sqn(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    if len(pnl) < 2:
        return np.nan

    std = pnl.std(
        ddof=1
    )

    if std == 0:
        return np.nan

    return (
        np.sqrt(
            len(pnl)
        )
        *
        pnl.mean()
        /
        std
    )


def max_drawdown(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    if len(pnl) == 0:
        return np.nan

    equity = np.concatenate(
        [
            [0.0],
            np.cumsum(
                pnl
            )
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
        np.isfinite(
            pnl
        )
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

    random_values = np.asarray(
        random_values,
        dtype=float
    )

    random_values = random_values[
        np.isfinite(
            random_values
        )
    ]

    if len(
        random_values
    ) == 0:
        return np.nan

    return (
        np.mean(
            random_values
            <=
            actual
        )
        *
        100
    )


# ============================================================
# MAIN PARAMETER LOOP
# ============================================================

summary_rows = []
year_rows = []
random_rows = []
signal_count_rows = []

rng_master = np.random.default_rng(
    RANDOM_SEED
)

years_all = eligible[
    "Year"
].to_numpy()

hours_all = eligible[
    "Hour"
].to_numpy()


for config_number, (
    fast,
    slow,
    signal_len
) in enumerate(
    MACD_CONFIGS,
    start=1
):

    label = (
        f"MACD_{fast}_{slow}_{signal_len}"
    )

    print()
    print("=" * 180)
    print(
        f"TESTING {label}"
    )
    print("=" * 180)


    # ========================================================
    # CALCULATE THIS MACD
    # ========================================================

    fast_ema = df30[
        "Close"
    ].ewm(
        span=fast,
        adjust=False
    ).mean()

    slow_ema = df30[
        "Close"
    ].ewm(
        span=slow,
        adjust=False
    ).mean()

    macd_line = (
        fast_ema
        -
        slow_ema
    )

    signal_line = macd_line.ewm(
        span=signal_len,
        adjust=False
    ).mean()

    hist = (
        macd_line
        -
        signal_line
    )

    bull_cross = (
        (hist.shift(1) <= 0)
        &
        (hist > 0)
    )


    # Align with eligible table
    bull_map = bull_cross.reindex(
        eligible[
            "SignalTime"
        ]
    ).fillna(
        False
    ).to_numpy(
        dtype=bool
    )

    macd_indices = np.where(
        bull_map
    )[0]

    random_candidate_indices = np.where(
        ~bull_map
    )[0]

    print(
        "Bullish crosses:",
        len(macd_indices)
    )


    signal_count_rows.append({

        "MACD":
            label,

        "Fast":
            fast,

        "Slow":
            slow,

        "Signal":
            signal_len,

        "N_Crosses":
            len(
                macd_indices
            )
    })


    # ========================================================
    # STRATIFY BY YEAR + HOUR
    # ========================================================

    strata = {}

    macd_year = years_all[
        macd_indices
    ]

    macd_hour = hours_all[
        macd_indices
    ]


    unique_pairs = sorted(
        set(
            zip(
                macd_year,
                macd_hour
            )
        )
    )


    for year, hour in unique_pairs:

        required_n = np.sum(
            (macd_year == year)
            &
            (macd_hour == hour)
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
                ~bull_map
            )
        )[0]


        if len(pool) < required_n:

            raise RuntimeError(
                f"{label}: insufficient random pool "
                f"for Year={year}, Hour={hour}. "
                f"Need {required_n}, have {len(pool)}."
            )


        strata[
            (
                year,
                hour
            )
        ] = (
            pool,
            required_n
        )


    # ========================================================
    # DRAW MATCHED RANDOM SETS
    #
    # Separate reproducible RNG stream per MACD config.
    # ========================================================

    rng = np.random.default_rng(
        RANDOM_SEED
        +
        config_number
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
            pool,
            required_n
        ) in strata.items():

            chosen = rng.choice(
                pool,
                size=required_n,
                replace=False
            )

            random_sets[
                run,
                cursor:
                cursor + required_n
            ] = chosen

            cursor += required_n


    # ========================================================
    # HOLD LOOP
    # ========================================================

    for hold_minutes in HOLD_MINUTES_LIST:

        pnl_all = eligible[
            f"PnL_{hold_minutes}"
        ].to_numpy(
            dtype=float
        )


        # ----------------------------------------------------
        # ACTUAL MACD
        # ----------------------------------------------------

        macd_pnl = pnl_all[
            macd_indices
        ]

        s = stats(
            macd_pnl
        )


        # ----------------------------------------------------
        # YEAR STATS
        # ----------------------------------------------------

        year_pfs = []


        for year in YEARS:

            ypnl = macd_pnl[
                macd_year
                ==
                year
            ]

            ys = stats(
                ypnl
            )


            year_rows.append({

                "MACD":
                    label,

                "Fast":
                    fast,

                "Slow":
                    slow,

                "Signal":
                    signal_len,

                "HoldingMinutes":
                    hold_minutes,

                "Year":
                    year,

                **ys
            })


            if np.isfinite(
                ys[
                    "ProfitFactor"
                ]
            ):

                year_pfs.append(
                    ys[
                        "ProfitFactor"
                    ]
                )


        worst_year_pf = (
            min(
                year_pfs
            )
            if year_pfs
            else np.nan
        )

        best_year_pf = (
            max(
                year_pfs
            )
            if year_pfs
            else np.nan
        )

        profitable_years = sum(
            pf > 1
            for pf in year_pfs
        )


        # ----------------------------------------------------
        # RANDOM
        # ----------------------------------------------------

        random_pnl_matrix = pnl_all[
            random_sets
        ]


        random_net = random_pnl_matrix.sum(
            axis=1
        )

        random_avg = random_pnl_matrix.mean(
            axis=1
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


        gp = np.where(
            random_pnl_matrix > 0,
            random_pnl_matrix,
            0
        ).sum(
            axis=1
        )

        gl = -np.where(
            random_pnl_matrix < 0,
            random_pnl_matrix,
            0
        ).sum(
            axis=1
        )


        random_pf = np.divide(
            gp,
            gl,
            out=np.full(
                N_RANDOM_RUNS,
                np.nan
            ),
            where=(
                gl != 0
            )
        )


        random_std = random_pnl_matrix.std(
            axis=1,
            ddof=1
        )

        random_sqn = (
            np.sqrt(
                random_pnl_matrix.shape[
                    1
                ]
            )
            *
            random_avg
            /
            random_std
        )


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        summary_rows.append({

            "MACD":
                label,

            "Fast":
                fast,

            "Slow":
                slow,

            "Signal":
                signal_len,

            "HoldingMinutes":
                hold_minutes,

            **s,

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

            "MACDPercentile_Avg":
                percentile_vs_random(
                    random_avg,
                    s[
                        "AveragePnL"
                    ]
                ),

            "MACDPercentile_PF":
                percentile_vs_random(
                    random_pf,
                    s[
                        "ProfitFactor"
                    ]
                ),

            "MACDPercentile_WinRate":
                percentile_vs_random(
                    random_win,
                    s[
                        "WinRatePct"
                    ]
                ),

            "MACDPercentile_SQN":
                percentile_vs_random(
                    random_sqn,
                    s[
                        "SQN"
                    ]
                )
        })


        # ----------------------------------------------------
        # OPTIONAL RANDOM DETAIL
        # ----------------------------------------------------

        for run in range(
            N_RANDOM_RUNS
        ):

            random_rows.append({

                "MACD":
                    label,

                "HoldingMinutes":
                    hold_minutes,

                "Run":
                    run + 1,

                "AveragePnL":
                    random_avg[
                        run
                    ],

                "ProfitFactor":
                    random_pf[
                        run
                    ],

                "WinRatePct":
                    random_win[
                        run
                    ],

                "SQN":
                    random_sqn[
                        run
                    ]
            })


# ============================================================
# DATAFRAMES
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

year_df = pd.DataFrame(
    year_rows
)

random_df = pd.DataFrame(
    random_rows
)

signal_count_df = pd.DataFrame(
    signal_count_rows
)


# ============================================================
# YEAR PF MATRIX
# ============================================================

year_pf_matrix = year_df.pivot_table(
    index=[
        "MACD",
        "HoldingMinutes"
    ],
    columns="Year",
    values="ProfitFactor"
)


# ============================================================
# RANKING TABLE
#
# Focus on evidence vs random first.
# ============================================================

ranking_df = summary_df[
    [
        "MACD",
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
].copy()


ranking_df = ranking_df.sort_values(
    by=[
        "MACDPercentile_PF",
        "MACDPercentile_Avg",
        "WorstYearPF"
    ],
    ascending=[
        False,
        False,
        False
    ]
)


# ============================================================
# SAVE
# ============================================================

summary_df.to_csv(
    OUTPUT_DIR
    /
    "macd_param_hold_summary.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "macd_param_year_by_year.csv",
    index=False
)

random_df.to_csv(
    OUTPUT_DIR
    /
    "random_runs_detail.csv",
    index=False
)

signal_count_df.to_csv(
    OUTPUT_DIR
    /
    "signal_counts.csv",
    index=False
)

year_pf_matrix.to_csv(
    OUTPUT_DIR
    /
    "year_pf_matrix.csv"
)

ranking_df.to_csv(
    OUTPUT_DIR
    /
    "ranking.csv",
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
    2000
)

pd.set_option(
    "display.width",
    700
)


print()
print("=" * 260)
print("SIGNAL COUNTS")
print("=" * 260)

print(
    signal_count_df
    .to_string(
        index=False
    )
)


print()
print("=" * 260)
print("30M MACD PARAMETER SWEEP VS RANDOM")
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
print("TOP RESULTS — SORTED BY RANDOM PERCENTILE")
print("=" * 260)

print(
    ranking_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 260)
print("YEAR-BY-YEAR PROFIT FACTOR")
print("=" * 260)

print(
    year_pf_matrix
    .round(4)
    .to_string()
)


print()
print("=" * 260)
print("HOW TO READ THIS")
print("=" * 260)

print()
print(
    "Do NOT choose the highest PF mechanically."
)

print(
    "We want a parameter family where:"
)

print(
    "1. MACD beats matched random timing."
)

print(
    "2. Preferably Avg/PF percentile > 90%."
)

print(
    "3. Several neighboring holding periods behave similarly."
)

print(
    "4. Year-to-year PF is reasonably stable."
)

print(
    "5. Similar MACD parameterizations show related behavior."
)

print()
print(
    "If only one exact MACD/hold combination works, "
    "that is a data-mining warning."
)


print()
print("=" * 260)
print("DONE")
print("=" * 260)

print()
print("Output folder:")
print(OUTPUT_DIR)