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
    r"\MACD_30M_LocalRobustness_vs_Random_FAST"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TIMEFRAME = "30min"

FAST_LENGTHS = [
    6,
    7,
    8,
    9,
    10
]

SLOW_LENGTHS = [
    30,
    36,
    42
]

SIGNAL_LENGTH = 5

HOLD_MINUTES_LIST = [
    60,
    90,
    120,
    150,
    180
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
# FAST NUMPY ARRAYS
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
# BUILD MASTER ENTRY TABLE
# ============================================================

print()
print("=" * 180)
print("BUILDING MASTER ENTRY TABLE")
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

entry_positions = entry_positions[
    valid_entry
]

entry_times = minute_times[
    entry_positions
]

entry_prices = minute_open[
    entry_positions
]


master = pd.DataFrame({
    "SignalTime": signal_times,
    "Year": signal_times.year,
    "Hour": signal_times.hour,
    "EntryTime": pd.to_datetime(
        entry_times
    ),
    "EntryPrice": entry_prices
})


print(
    "Master eligible 30m bars:",
    len(master)
)


# ============================================================
# PRECOMPUTE FORWARD PNL
#
# IMPORTANT:
# Do NOT drop rows based on all holding periods.
# Each holding period gets its own valid universe later.
# ============================================================

print()
print("=" * 180)
print("PRECOMPUTING FORWARD RETURNS")
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
        len(master),
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

    master[
        f"PnL_{hold_minutes}"
    ] = pnl

    print(
        "  Valid bars:",
        int(
            np.isfinite(
                pnl
            ).sum()
        )
    )


# ============================================================
# STATS
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

    if len(random_values) == 0:
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
# PRECOMPUTE MACD CROSS SERIES FOR ALL CONFIGURATIONS
# ============================================================

print()
print("=" * 180)
print("PRECOMPUTING MACD CROSS SERIES")
print("=" * 180)

macd_crosses = {}

close30 = df30[
    "Close"
]


for fast in FAST_LENGTHS:

    for slow in SLOW_LENGTHS:

        if fast >= slow:
            continue

        label = (
            f"MACD_{fast}_{slow}_{SIGNAL_LENGTH}"
        )

        fast_ema = close30.ewm(
            span=fast,
            adjust=False
        ).mean()

        slow_ema = close30.ewm(
            span=slow,
            adjust=False
        ).mean()

        macd_line = (
            fast_ema
            -
            slow_ema
        )

        signal_line = macd_line.ewm(
            span=SIGNAL_LENGTH,
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

        macd_crosses[
            label
        ] = bull_cross


# ============================================================
# RESULTS
# ============================================================

summary_rows = []
year_rows = []
signal_count_rows = []
fallback_rows = []

config_number = 0


# ============================================================
# PARAMETER LOOP
# ============================================================

for fast in FAST_LENGTHS:

    for slow in SLOW_LENGTHS:

        if fast >= slow:
            continue

        config_number += 1

        label = (
            f"MACD_{fast}_{slow}_{SIGNAL_LENGTH}"
        )

        print()
        print("=" * 180)
        print(
            f"TESTING {label}"
        )
        print("=" * 180)

        bull_cross_series = macd_crosses[
            label
        ]


        # ====================================================
        # HOLD LOOP
        #
        # Each hold gets its own valid universe.
        # ====================================================

        for hold_minutes in HOLD_MINUTES_LIST:

            pnl_col = (
                f"PnL_{hold_minutes}"
            )

            eligible = master[
                master[
                    pnl_col
                ].notna()
            ].copy()

            eligible = eligible.reset_index(
                drop=True
            )


            bull_map = bull_cross_series.reindex(
                eligible[
                    "SignalTime"
                ]
            ).fillna(
                False
            ).to_numpy(
                dtype=bool
            )

            years_all = eligible[
                "Year"
            ].to_numpy()

            hours_all = eligible[
                "Hour"
            ].to_numpy()

            pnl_all = eligible[
                pnl_col
            ].to_numpy(
                dtype=float
            )


            macd_indices = np.where(
                bull_map
            )[0]

            macd_year = years_all[
                macd_indices
            ]

            macd_hour = hours_all[
                macd_indices
            ]

            macd_pnl = pnl_all[
                macd_indices
            ]


            print(
                f"  Hold {hold_minutes}: "
                f"N={len(macd_indices)}"
            )


            signal_count_rows.append({

                "MACD":
                    label,

                "Fast":
                    fast,

                "Slow":
                    slow,

                "Signal":
                    SIGNAL_LENGTH,

                "HoldingMinutes":
                    hold_minutes,

                "ValidUniverseN":
                    len(
                        eligible
                    ),

                "N_Crosses":
                    len(
                        macd_indices
                    )
            })


            # =================================================
            # ACTUAL MACD STATS
            # =================================================

            s = stats(
                macd_pnl
            )


            # =================================================
            # YEAR-BY-YEAR
            # =================================================

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
                        SIGNAL_LENGTH,

                    "HoldingMinutes":
                        hold_minutes,

                    "Year":
                        year,

                    **ys
                })


                pf = ys[
                    "ProfitFactor"
                ]

                if np.isfinite(
                    pf
                ):

                    year_pfs.append(
                        pf
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


            # =================================================
            # BUILD YEAR/HOUR RANDOM STRATA
            # =================================================

            strata = {}

            unique_pairs = sorted(
                set(
                    zip(
                        macd_year,
                        macd_hour
                    )
                )
            )


            for year, hour in unique_pairs:

                required_n = int(
                    np.sum(
                        (macd_year == year)
                        &
                        (macd_hour == hour)
                    )
                )


                exact_pool = np.where(
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


                # ---------------------------------------------
                # Normally exact year/hour matching is used.
                #
                # Rare fallback:
                # If no control exists in the exact hour,
                # expand to +/-1 hour within the SAME year.
                #
                # This prevents a single sparse bucket from
                # terminating the whole robustness test.
                # ---------------------------------------------

                if len(
                    exact_pool
                ) >= required_n:

                    pool = exact_pool
                    match_type = "EXACT"

                else:

                    hour_distance = np.minimum(
                        np.abs(
                            hours_all
                            -
                            hour
                        ),
                        24
                        -
                        np.abs(
                            hours_all
                            -
                            hour
                        )
                    )

                    expanded_pool = np.where(
                        (
                            years_all
                            ==
                            year
                        )
                        &
                        (
                            hour_distance
                            <=
                            1
                        )
                        &
                        (
                            ~bull_map
                        )
                    )[0]


                    if len(
                        expanded_pool
                    ) < required_n:

                        raise RuntimeError(
                            f"{label}, hold={hold_minutes}: "
                            f"insufficient pool even after +/-1h fallback "
                            f"Year={year}, Hour={hour}. "
                            f"Need={required_n}, "
                            f"Available={len(expanded_pool)}"
                        )


                    pool = expanded_pool
                    match_type = "FALLBACK_1H"


                    fallback_rows.append({

                        "MACD":
                            label,

                        "HoldingMinutes":
                            hold_minutes,

                        "Year":
                            year,

                        "Hour":
                            hour,

                        "RequiredN":
                            required_n,

                        "ExactAvailable":
                            len(
                                exact_pool
                            ),

                        "ExpandedAvailable":
                            len(
                                expanded_pool
                            ),

                        "MatchType":
                            match_type
                    })


                strata[
                    (
                        year,
                        hour
                    )
                ] = (
                    pool,
                    required_n
                )


            # =================================================
            # RANDOM SETS
            # =================================================

            rng = np.random.default_rng(
                RANDOM_SEED
                +
                config_number * 100
                +
                hold_minutes
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


            # =================================================
            # RANDOM STATISTICS
            # =================================================

            random_pnl = pnl_all[
                random_sets
            ]


            random_avg = random_pnl.mean(
                axis=1
            )

            random_net = random_pnl.sum(
                axis=1
            )

            random_win = (
                (
                    random_pnl
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
                random_pnl > 0,
                random_pnl,
                0
            ).sum(
                axis=1
            )

            gl = -np.where(
                random_pnl < 0,
                random_pnl,
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


            random_std = random_pnl.std(
                axis=1,
                ddof=1
            )

            random_sqn = (
                np.sqrt(
                    random_pnl.shape[
                        1
                    ]
                )
                *
                random_avg
                /
                random_std
            )


            # =================================================
            # SUMMARY
            # =================================================

            summary_rows.append({

                "MACD":
                    label,

                "Fast":
                    fast,

                "Slow":
                    slow,

                "Signal":
                    SIGNAL_LENGTH,

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


# ============================================================
# DATAFRAMES
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

year_df = pd.DataFrame(
    year_rows
)

signal_count_df = pd.DataFrame(
    signal_count_rows
)

fallback_df = pd.DataFrame(
    fallback_rows
)


# ============================================================
# RANKING
# ============================================================

ranking_df = summary_df[
    [
        "MACD",
        "Fast",
        "Slow",
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
    [
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
# SAVE
# ============================================================

summary_df.to_csv(
    OUTPUT_DIR
    /
    "local_robustness_summary.csv",
    index=False
)

ranking_df.to_csv(
    OUTPUT_DIR
    /
    "ranking.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "year_by_year.csv",
    index=False
)

year_pf_matrix.to_csv(
    OUTPUT_DIR
    /
    "year_pf_matrix.csv"
)

signal_count_df.to_csv(
    OUTPUT_DIR
    /
    "signal_counts.csv",
    index=False
)

fallback_df.to_csv(
    OUTPUT_DIR
    /
    "random_matching_fallbacks.csv",
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
    4000
)

pd.set_option(
    "display.width",
    700
)


print()
print("=" * 260)
print("TOP RESULTS")
print("=" * 260)

print(
    ranking_df
    .head(
        50
    )
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# SURFACES
# ============================================================

for hold_minutes in HOLD_MINUTES_LIST:

    d = summary_df[
        summary_df[
            "HoldingMinutes"
        ]
        ==
        hold_minutes
    ]


    print()
    print("=" * 220)
    print(
        f"PF SURFACE — HOLD {hold_minutes} MIN"
    )
    print("=" * 220)

    print(
        d.pivot(
            index="Fast",
            columns="Slow",
            values="ProfitFactor"
        )
        .round(4)
        .to_string()
    )


    print()
    print(
        f"RANDOM PF PERCENTILE — HOLD {hold_minutes} MIN"
    )

    print(
        d.pivot(
            index="Fast",
            columns="Slow",
            values="MACDPercentile_PF"
        )
        .round(1)
        .to_string()
    )


    print()
    print(
        f"WORST-YEAR PF — HOLD {hold_minutes} MIN"
    )

    print(
        d.pivot(
            index="Fast",
            columns="Slow",
            values="WorstYearPF"
        )
        .round(4)
        .to_string()
    )


# ============================================================
# FALLBACK REPORT
# ============================================================

print()
print("=" * 220)
print("RANDOM MATCHING FALLBACKS")
print("=" * 220)

if len(
    fallback_df
) == 0:

    print(
        "None. All random samples used exact Year + Hour matching."
    )

else:

    print(
        fallback_df
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Fallbacks use only +/-1 hour and remain inside the same year."
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
print("DONE")
print("=" * 260)

print()
print("Output folder:")
print(OUTPUT_DIR)