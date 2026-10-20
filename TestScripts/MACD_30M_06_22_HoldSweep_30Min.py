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
    r"\MACD_30M_06_22_HoldSweep_30Min"
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

HOLD_MINUTES_LIST = list(
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
# BUILD 30M BARS
# ============================================================

print()
print("=" * 180)
print("BUILDING 30-MIN BARS")
print("=" * 180)

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
    subset=[
        "Open",
        "High",
        "Low",
        "Close"
    ]
)

print(
    "30m bars:",
    len(df30)
)


# ============================================================
# MACD
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


# ============================================================
# MASTER 30M UNIVERSE
# ============================================================

master = df30[
    df30.index.year.isin(YEARS)
    &
    (df30.index.hour >= SESSION_START)
    &
    (df30.index.hour < SESSION_END)
].copy()

print()
print("=" * 180)
print("SESSION")
print("=" * 180)

print(
    f"Using signal times {SESSION_START:02d}:00 <= hour < {SESSION_END:02d}:00"
)

print(
    "30m bars in session:",
    len(master)
)

print(
    "Raw bullish MACD crosses in session:",
    int(master["BullCross"].sum())
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

minute_close = df1["Close"].to_numpy(
    dtype=float
)


# ============================================================
# ENTRY MAPPING
#
# Entry = first 1m Open AFTER completed 30m signal bar
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

master["EntryPos"] = entry_positions

master["EntryTime"] = pd.to_datetime(
    minute_times[
        entry_positions
    ]
)

master["EntryPrice"] = minute_open[
    entry_positions
]

master["Year"] = master.index.year
master["Hour"] = master.index.hour


# ============================================================
# PRECOMPUTE PNL FOR EACH HOLD
#
# IMPORTANT:
# Each holding period gets its OWN valid population.
# We do NOT require validity for all holds simultaneously.
# ============================================================

print()
print("=" * 180)
print("PRECOMPUTING HOLDING PERIOD RETURNS")
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


for hold in HOLD_MINUTES_LIST:

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
        np.nan,
        dtype=float
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
        f"{hold:>3} min -> valid bars: "
        f"{np.isfinite(pnl).sum()}"
    )


# ============================================================
# STAT FUNCTIONS
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
# RUN HOLD SWEEP
# ============================================================

print()
print("=" * 180)
print("RUNNING 30-MIN HOLD SWEEP")
print("=" * 180)

summary_rows = []
year_rows = []

rng = np.random.default_rng(
    RANDOM_SEED
)


for hold in HOLD_MINUTES_LIST:

    pnl_col = (
        f"PnL_{hold}"
    )


    # ========================================================
    # VALID UNIVERSE FOR THIS HOLD
    # ========================================================

    eligible = master[
        master[
            pnl_col
        ].notna()
    ].copy()


    bull_mask = eligible[
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

    pnl_all = eligible[
        pnl_col
    ].to_numpy(
        dtype=float
    )


    macd_indices = np.where(
        bull_mask
    )[0]

    macd_pnl = pnl_all[
        macd_indices
    ]

    macd_years = years_all[
        macd_indices
    ]

    macd_hours = hours_all[
        macd_indices
    ]


    actual = stats(
        macd_pnl
    )


    # ========================================================
    # YEAR BY YEAR
    # ========================================================

    year_pfs = []


    for year in YEARS:

        ypnl = macd_pnl[
            macd_years
            ==
            year
        ]

        ys = stats(
            ypnl
        )


        year_rows.append({

            "HoldingMinutes":
                hold,

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
    # MATCHED RANDOM STRATA — YEAR + HOUR
    # ========================================================

    strata = []


    pairs = sorted(
        set(
            zip(
                macd_years,
                macd_hours
            )
        )
    )


    for year, hour in pairs:

        required_n = int(
            np.sum(
                (
                    macd_years
                    ==
                    year
                )
                &
                (
                    macd_hours
                    ==
                    hour
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
                ~bull_mask
            )
        )[0]


        if len(pool) < required_n:

            raise RuntimeError(
                f"Hold={hold}: insufficient random pool "
                f"Year={year}, Hour={hour}, "
                f"need={required_n}, have={len(pool)}"
            )


        strata.append(
            (
                pool,
                required_n
            )
        )


    # ========================================================
    # RANDOM SIMULATIONS
    # ========================================================

    random_avg = []
    random_pf = []
    random_net = []
    random_sqn = []


    for run in range(
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
            rpnl.mean()
        )

        random_net.append(
            rpnl.sum()
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


    # ========================================================
    # SUMMARY
    # ========================================================

    summary_rows.append({

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
# DATAFRAMES
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

year_df = pd.DataFrame(
    year_rows
)


year_pf_matrix = year_df.pivot_table(
    index="HoldingMinutes",
    columns="Year",
    values="ProfitFactor"
)


# ============================================================
# SAVE
# ============================================================

summary_df.to_csv(
    OUTPUT_DIR
    /
    "hold_sweep_summary.csv",
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
print("=" * 280)
print("30M MACD(5,36,5) — SESSION 06-22 — 30-MIN HOLD SWEEP")
print("=" * 280)

print(
    summary_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 240)
print("YEAR-BY-YEAR PROFIT FACTOR")
print("=" * 240)

print(
    year_pf_matrix
    .round(4)
    .to_string()
)


print()
print("=" * 240)
print("TOP HOLDS BY RANDOM PF PERCENTILE")
print("=" * 240)

print(
    summary_df
    .sort_values(
        [
            "MACDPercentile_PF",
            "WorstYearPF"
        ],
        ascending=[
            False,
            False
        ]
    )
    .head(16)
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 240)
print("DONE")
print("=" * 240)

print()
print(
    "Session: 06:00 <= signal hour < 22:00"
)

print(
    "Holds: 30 to 480 minutes in 30-minute increments"
)

print(
    "Signal: bullish 30m MACD(5,36,5) histogram zero-cross"
)

print(
    "Random control: matched by Year + Hour"
)

print()
print("Output folder:")
print(OUTPUT_DIR)