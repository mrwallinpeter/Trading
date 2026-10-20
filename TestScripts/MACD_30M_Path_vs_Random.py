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
    r"\MACD_30M_Path_vs_Random"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TIMEFRAME = "30min"

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

HORIZONS = [
    60,
    120,
    180,
    240,
    300
]

BARRIERS = [
    5.0,
    10.0,
    15.0,
    20.0
]

MAX_HORIZON = max(HORIZONS)

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
# MACD(5,36,5)
# ============================================================

fast_ema = df30["Close"].ewm(
    span=MACD_FAST,
    adjust=False
).mean()

slow_ema = df30["Close"].ewm(
    span=MACD_SLOW,
    adjust=False
).mean()

df30["MACD"] = (
    fast_ema
    -
    slow_ema
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

df30["BearCross"] = (
    (df30["Hist"].shift(1) >= 0)
    &
    (df30["Hist"] < 0)
)


# ============================================================
# NUMPY ARRAYS
# ============================================================

minute_times = df1.index.values.astype(
    "datetime64[ns]"
)

minute_open = df1["Open"].to_numpy(
    dtype=float
)

minute_high = df1["High"].to_numpy(
    dtype=float
)

minute_low = df1["Low"].to_numpy(
    dtype=float
)

minute_close = df1["Close"].to_numpy(
    dtype=float
)


# ============================================================
# BUILD MASTER ENTRY TABLE
# ============================================================

print()
print("=" * 180)
print("BUILDING MASTER 30-MIN ENTRY TABLE")
print("=" * 180)

signal_times = df30.index[
    df30.index.year.isin(YEARS)
]

signal_times_np = signal_times.values.astype(
    "datetime64[ns]"
)

entry_positions = np.searchsorted(
    minute_times,
    signal_times_np,
    side="right"
)

valid = (
    entry_positions
    <
    len(minute_times)
)

signal_times = signal_times[
    valid
]

entry_positions = entry_positions[
    valid
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
    "EntryPos": entry_positions,
    "EntryTime": pd.to_datetime(entry_times),
    "EntryPrice": entry_prices,
    "BullCross": df30.loc[
        signal_times,
        "BullCross"
    ].to_numpy(dtype=bool),
    "BearCross": df30.loc[
        signal_times,
        "BearCross"
    ].to_numpy(dtype=bool)
})


print("Master bars:", len(master))

print(
    "Bullish crosses:",
    int(master["BullCross"].sum())
)

print(
    "Bearish crosses:",
    int(master["BearCross"].sum())
)


# ============================================================
# PATH CALCULATION
#
# Direction:
# LONG
#   MFE = max(high - entry)
#   MAE = max(entry - low)
#
# SHORT
#   MFE = max(entry - low)
#   MAE = max(high - entry)
# ============================================================

print()
print("=" * 180)
print("CALCULATING FORWARD PATH STATISTICS")
print("=" * 180)


path_rows = []


for i, row in master.iterrows():

    if (
        i % 5000
        ==
        0
    ):
        print(
            f"Processed {i} / {len(master)}"
        )


    entry_pos = int(
        row["EntryPos"]
    )

    entry_time = np.datetime64(
        row["EntryTime"]
    )

    entry_price = float(
        row["EntryPrice"]
    )


    max_target = (
        entry_time
        +
        np.timedelta64(
            MAX_HORIZON,
            "m"
        )
    )


    max_exit_pos = np.searchsorted(
        minute_times,
        max_target,
        side="right"
    )


    if max_exit_pos <= entry_pos:
        continue


    path_times = minute_times[
        entry_pos:
        max_exit_pos
    ]

    path_high = minute_high[
        entry_pos:
        max_exit_pos
    ]

    path_low = minute_low[
        entry_pos:
        max_exit_pos
    ]

    path_close = minute_close[
        entry_pos:
        max_exit_pos
    ]


    base_record = {
        "MasterIndex": i,
        "SignalTime": row["SignalTime"],
        "Year": row["Year"],
        "Hour": row["Hour"],
        "EntryTime": row["EntryTime"],
        "EntryPrice": entry_price,
        "BullCross": row["BullCross"],
        "BearCross": row["BearCross"]
    }


    # ========================================================
    # LONG + SHORT calculations
    # ========================================================

    for direction in [
        "LONG",
        "SHORT"
    ]:

        record = dict(
            base_record
        )

        record[
            "Direction"
        ] = direction


        for horizon in HORIZONS:

            target_time = (
                entry_time
                +
                np.timedelta64(
                    horizon,
                    "m"
                )
            )


            h_end = np.searchsorted(
                path_times,
                target_time,
                side="right"
            )


            if h_end <= 0:

                record[
                    f"Valid_{horizon}"
                ] = False

                continue


            actual_last_time = path_times[
                h_end - 1
            ]


            # Require data to reach horizon within 2 minutes
            if (
                actual_last_time
                <
                target_time
                -
                np.timedelta64(
                    2,
                    "m"
                )
            ):

                record[
                    f"Valid_{horizon}"
                ] = False

                continue


            record[
                f"Valid_{horizon}"
            ] = True


            h_high = path_high[
                :h_end
            ]

            h_low = path_low[
                :h_end
            ]

            h_close = path_close[
                :h_end
            ]

            h_times = path_times[
                :h_end
            ]


            if direction == "LONG":

                favorable = (
                    h_high
                    -
                    entry_price
                )

                adverse = (
                    entry_price
                    -
                    h_low
                )

                mfe_idx = int(
                    np.argmax(
                        favorable
                    )
                )

                mae_idx = int(
                    np.argmax(
                        adverse
                    )
                )

                mfe = float(
                    favorable[
                        mfe_idx
                    ]
                )

                mae = float(
                    adverse[
                        mae_idx
                    ]
                )

                end_pnl = float(
                    h_close[-1]
                    -
                    entry_price
                )


            else:

                favorable = (
                    entry_price
                    -
                    h_low
                )

                adverse = (
                    h_high
                    -
                    entry_price
                )

                mfe_idx = int(
                    np.argmax(
                        favorable
                    )
                )

                mae_idx = int(
                    np.argmax(
                        adverse
                    )
                )

                mfe = float(
                    favorable[
                        mfe_idx
                    ]
                )

                mae = float(
                    adverse[
                        mae_idx
                    ]
                )

                end_pnl = float(
                    entry_price
                    -
                    h_close[-1]
                )


            mfe_minutes = (
                h_times[
                    mfe_idx
                ]
                -
                entry_time
            ) / np.timedelta64(
                1,
                "m"
            )

            mae_minutes = (
                h_times[
                    mae_idx
                ]
                -
                entry_time
            ) / np.timedelta64(
                1,
                "m"
            )


            record[
                f"MFE_{horizon}"
            ] = mfe

            record[
                f"MAE_{horizon}"
            ] = mae

            record[
                f"EndPnL_{horizon}"
            ] = end_pnl

            record[
                f"MinutesToMFE_{horizon}"
            ] = float(
                mfe_minutes
            )

            record[
                f"MinutesToMAE_{horizon}"
            ] = float(
                mae_minutes
            )


            # =================================================
            # FIRST-TOUCH BARRIERS
            # =================================================

            for barrier in BARRIERS:

                barrier_name = str(
                    int(barrier)
                )


                fav_hit = np.where(
                    favorable
                    >=
                    barrier
                )[0]

                adv_hit = np.where(
                    adverse
                    >=
                    barrier
                )[0]


                fav_first_idx = (
                    fav_hit[0]
                    if len(fav_hit)
                    else None
                )

                adv_first_idx = (
                    adv_hit[0]
                    if len(adv_hit)
                    else None
                )


                if (
                    fav_first_idx is None
                    and
                    adv_first_idx is None
                ):

                    outcome = "NEITHER"


                elif (
                    fav_first_idx is not None
                    and
                    adv_first_idx is None
                ):

                    outcome = "FAVORABLE_FIRST"


                elif (
                    fav_first_idx is None
                    and
                    adv_first_idx is not None
                ):

                    outcome = "ADVERSE_FIRST"


                elif (
                    fav_first_idx
                    <
                    adv_first_idx
                ):

                    outcome = "FAVORABLE_FIRST"


                elif (
                    adv_first_idx
                    <
                    fav_first_idx
                ):

                    outcome = "ADVERSE_FIRST"


                else:

                    # Both touched during same 1m bar.
                    # We cannot know intrabar order.
                    outcome = "SAME_BAR"


                record[
                    f"Barrier{barrier_name}_{horizon}"
                ] = outcome


        path_rows.append(
            record
        )


paths = pd.DataFrame(
    path_rows
)


# ============================================================
# SIGNAL DATASETS
# ============================================================

bull_paths = paths[
    (
        paths["Direction"]
        ==
        "LONG"
    )
    &
    (
        paths["BullCross"]
    )
].copy()


bear_paths = paths[
    (
        paths["Direction"]
        ==
        "SHORT"
    )
    &
    (
        paths["BearCross"]
    )
].copy()


# Random controls:
# exclude ALL actual MACD-cross bars

no_cross_master = master[
    ~(
        master["BullCross"]
        |
        master["BearCross"]
    )
].copy()


long_random_paths = paths[
    (
        paths["Direction"]
        ==
        "LONG"
    )
    &
    (
        paths["MasterIndex"]
        .isin(
            no_cross_master.index
        )
    )
].copy()


short_random_paths = paths[
    (
        paths["Direction"]
        ==
        "SHORT"
    )
    &
    (
        paths["MasterIndex"]
        .isin(
            no_cross_master.index
        )
    )
].copy()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def pf(pnl):

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


def percentile(
    values,
    actual
):

    values = np.asarray(
        values,
        dtype=float
    )

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


# ============================================================
# MATCHED RANDOM ANALYSIS
# ============================================================

print()
print("=" * 180)
print("MATCHED RANDOM ANALYSIS")
print("=" * 180)


summary_rows = []
random_rows = []
year_rows = []


def analyse_direction(
    name,
    signal_df,
    pool_df
):

    rng = np.random.default_rng(
        RANDOM_SEED
        +
        (
            1
            if name == "BULL"
            else 10001
        )
    )


    for horizon in HORIZONS:

        valid_col = (
            f"Valid_{horizon}"
        )


        sig = signal_df[
            signal_df[
                valid_col
            ]
            ==
            True
        ].copy()


        pool = pool_df[
            pool_df[
                valid_col
            ]
            ==
            True
        ].copy()


        # ====================================================
        # BUILD MATCHING STRATA
        # ====================================================

        strata = []


        counts = (
            sig.groupby(
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


        for _, r in counts.iterrows():

            year = int(
                r["Year"]
            )

            hour = int(
                r["Hour"]
            )

            required_n = int(
                r["RequiredN"]
            )


            p = pool[
                (
                    pool["Year"]
                    ==
                    year
                )
                &
                (
                    pool["Hour"]
                    ==
                    hour
                )
            ]


            if len(p) < required_n:

                raise RuntimeError(
                    f"{name} horizon={horizon}: "
                    f"insufficient random pool "
                    f"Year={year}, Hour={hour}, "
                    f"need={required_n}, have={len(p)}"
                )


            strata.append(
                (
                    p.index.to_numpy(),
                    required_n
                )
            )


        # ====================================================
        # ACTUAL SIGNAL SUMMARY
        # ====================================================

        mfe_col = (
            f"MFE_{horizon}"
        )

        mae_col = (
            f"MAE_{horizon}"
        )

        end_col = (
            f"EndPnL_{horizon}"
        )

        mfe_time_col = (
            f"MinutesToMFE_{horizon}"
        )


        actual_mean_mfe = sig[
            mfe_col
        ].mean()

        actual_median_mfe = sig[
            mfe_col
        ].median()

        actual_mean_mae = sig[
            mae_col
        ].mean()

        actual_median_mae = sig[
            mae_col
        ].median()

        actual_mean_end = sig[
            end_col
        ].mean()

        actual_pf = pf(
            sig[
                end_col
            ]
        )

        actual_median_mfe_time = sig[
            mfe_time_col
        ].median()


        # ====================================================
        # YEAR-BY-YEAR SIGNAL PATH STATS
        # ====================================================

        for year in YEARS:

            yd = sig[
                sig["Year"]
                ==
                year
            ]


            year_rows.append({

                "Direction":
                    name,

                "Horizon":
                    horizon,

                "Year":
                    year,

                "N":
                    len(yd),

                "MeanMFE":
                    yd[
                        mfe_col
                    ].mean(),

                "MedianMFE":
                    yd[
                        mfe_col
                    ].median(),

                "MeanMAE":
                    yd[
                        mae_col
                    ].mean(),

                "MedianMAE":
                    yd[
                        mae_col
                    ].median(),

                "AverageEndPnL":
                    yd[
                        end_col
                    ].mean(),

                "ProfitFactor":
                    pf(
                        yd[
                            end_col
                        ]
                    )
                    if len(yd)
                    else np.nan
            })


        # ====================================================
        # RANDOM SIMULATIONS
        # ====================================================

        random_mean_mfe = []
        random_mean_mae = []
        random_mean_end = []
        random_pf = []
        random_median_mfe_time = []


        random_barrier_results = {
            b: []
            for b in BARRIERS
        }


        for run in range(
            N_RANDOM_RUNS
        ):

            chosen = []


            for (
                pool_indices,
                required_n
            ) in strata:

                picked = rng.choice(
                    pool_indices,
                    size=required_n,
                    replace=False
                )

                chosen.extend(
                    picked.tolist()
                )


            rd = pool.loc[
                chosen
            ]


            random_mean_mfe.append(
                rd[
                    mfe_col
                ].mean()
            )

            random_mean_mae.append(
                rd[
                    mae_col
                ].mean()
            )

            random_mean_end.append(
                rd[
                    end_col
                ].mean()
            )

            random_pf.append(
                pf(
                    rd[
                        end_col
                    ]
                )
            )

            random_median_mfe_time.append(
                rd[
                    mfe_time_col
                ].median()
            )


            for barrier in BARRIERS:

                bname = str(
                    int(barrier)
                )

                bcol = (
                    f"Barrier{bname}_{horizon}"
                )


                favorable_pct = (
                    (
                        rd[
                            bcol
                        ]
                        ==
                        "FAVORABLE_FIRST"
                    )
                    .mean()
                    *
                    100
                )


                random_barrier_results[
                    barrier
                ].append(
                    favorable_pct
                )


        # ====================================================
        # MAIN SUMMARY
        # ====================================================

        summary_rows.append({

            "Direction":
                name,

            "Horizon":
                horizon,

            "N":
                len(sig),

            "MeanMFE":
                actual_mean_mfe,

            "RandomMedianMeanMFE":
                np.median(
                    random_mean_mfe
                ),

            "MFEPercentileVsRandom":
                percentile(
                    random_mean_mfe,
                    actual_mean_mfe
                ),

            "MedianMFE":
                actual_median_mfe,

            "MeanMAE":
                actual_mean_mae,

            "RandomMedianMeanMAE":
                np.median(
                    random_mean_mae
                ),

            "MedianMAE":
                actual_median_mae,

            "AverageEndPnL":
                actual_mean_end,

            "RandomMedianAverageEndPnL":
                np.median(
                    random_mean_end
                ),

            "EndPnLPercentileVsRandom":
                percentile(
                    random_mean_end,
                    actual_mean_end
                ),

            "ProfitFactor":
                actual_pf,

            "RandomMedianPF":
                np.nanmedian(
                    random_pf
                ),

            "PFPercentileVsRandom":
                percentile(
                    random_pf,
                    actual_pf
                ),

            "MedianMinutesToMFE":
                actual_median_mfe_time,

            "RandomMedianMinutesToMFE":
                np.median(
                    random_median_mfe_time
                )
        })


        # ====================================================
        # BARRIER SUMMARY
        # ====================================================

        for barrier in BARRIERS:

            bname = str(
                int(barrier)
            )

            bcol = (
                f"Barrier{bname}_{horizon}"
            )


            favorable_first = (
                sig[
                    bcol
                ]
                ==
                "FAVORABLE_FIRST"
            )

            adverse_first = (
                sig[
                    bcol
                ]
                ==
                "ADVERSE_FIRST"
            )

            same_bar = (
                sig[
                    bcol
                ]
                ==
                "SAME_BAR"
            )

            neither = (
                sig[
                    bcol
                ]
                ==
                "NEITHER"
            )


            actual_fav_pct = (
                favorable_first.mean()
                *
                100
            )


            random_vals = (
                random_barrier_results[
                    barrier
                ]
            )


            random_rows.append({

                "Direction":
                    name,

                "Horizon":
                    horizon,

                "BarrierPoints":
                    barrier,

                "N":
                    len(sig),

                "FavorableFirstPct":
                    actual_fav_pct,

                "AdverseFirstPct":
                    adverse_first.mean()
                    *
                    100,

                "SameBarPct":
                    same_bar.mean()
                    *
                    100,

                "NeitherPct":
                    neither.mean()
                    *
                    100,

                "RandomMedianFavorableFirstPct":
                    np.median(
                        random_vals
                    ),

                "FavorableFirstPercentileVsRandom":
                    percentile(
                        random_vals,
                        actual_fav_pct
                    )
            })


# ============================================================
# RUN BOTH DIRECTIONS
# ============================================================

analyse_direction(
    "BULL_LONG",
    bull_paths,
    long_random_paths
)

analyse_direction(
    "BEAR_SHORT",
    bear_paths,
    short_random_paths
)


summary_df = pd.DataFrame(
    summary_rows
)

barrier_df = pd.DataFrame(
    random_rows
)

year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# SAVE
# ============================================================

paths.to_csv(
    OUTPUT_DIR
    /
    "all_path_detail.csv",
    index=False
)

bull_paths.to_csv(
    OUTPUT_DIR
    /
    "bull_cross_path_detail.csv",
    index=False
)

bear_paths.to_csv(
    OUTPUT_DIR
    /
    "bear_cross_path_detail.csv",
    index=False
)

summary_df.to_csv(
    OUTPUT_DIR
    /
    "path_summary_vs_random.csv",
    index=False
)

barrier_df.to_csv(
    OUTPUT_DIR
    /
    "barrier_first_touch_vs_random.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "path_year_by_year.csv",
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
    3000
)

pd.set_option(
    "display.width",
    700
)


print()
print("=" * 260)
print("PATH SUMMARY VS MATCHED RANDOM")
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
print("FIRST-TOUCH BARRIER TEST")
print("=" * 260)

print(
    barrier_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 260)
print("YEAR-BY-YEAR PATH SUMMARY")
print("=" * 260)

print(
    year_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 260)
print("WHAT TO LOOK FOR")
print("=" * 260)

print()
print(
    "1. MFEPercentileVsRandom:"
)

print(
    "   Does MACD produce larger favorable excursions than random?"
)

print()
print(
    "2. MedianMinutesToMFE:"
)

print(
    "   How quickly does the available move typically develop?"
)

print()
print(
    "3. FavorableFirstPct:"
)

print(
    "   Does +5/+10/+15/+20 tend to occur before the equal adverse move?"
)

print()
print(
    "4. FavorableFirstPercentileVsRandom:"
)

print(
    "   Is that first-touch advantage actually better than random timing?"
)

print()
print(
    "5. Bull and bear separately:"
)

print(
    "   30m MACD may be useful in one direction even if not the other."
)

print()
print(
    "6. Fixed-time PnL vs MFE:"
)

print(
    "   If MFE is strong but EndPnL is weak, the issue may be exit/management "
    "rather than signal timing."
)


print()
print("=" * 260)
print("DONE")
print("=" * 260)

print()
print("Output folder:")
print(OUTPUT_DIR)