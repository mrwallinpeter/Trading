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
    r"\MACD_30M_vs_Random_5H"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TIMEFRAME = "30min"

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

HOLD_MINUTES = 300

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
# BUILD FORWARD RETURN TABLE FOR EVERY ELIGIBLE 30-MIN BAR
# ============================================================

print()
print("=" * 180)
print("BUILDING 5-HOUR FORWARD RETURNS")
print("=" * 180)

rows = []


for signal_time in df30.index:

    # Restrict main statistical years
    if signal_time.year not in YEARS:
        continue


    # Entry = first 1-minute open after completed 30m bar
    entry_pos = df1.index.searchsorted(
        signal_time,
        side="right"
    )

    if entry_pos >= len(df1):
        continue


    entry_time = df1.index[
        entry_pos
    ]

    entry_price = float(
        df1.iloc[
            entry_pos
        ]["Open"]
    )


    # Fixed five-hour exit
    target_exit = (
        entry_time
        +
        pd.Timedelta(
            minutes=HOLD_MINUTES
        )
    )

    exit_pos = df1.index.searchsorted(
        target_exit,
        side="left"
    )

    if exit_pos >= len(df1):
        continue


    exit_time = df1.index[
        exit_pos
    ]


    # Same tolerance used in previous work
    if (
        exit_time
        >
        target_exit
        +
        pd.Timedelta(minutes=2)
    ):
        continue


    exit_price = float(
        df1.iloc[
            exit_pos
        ]["Close"]
    )

    pnl = (
        exit_price
        -
        entry_price
    )


    rows.append({

        "SignalTime":
            signal_time,

        "Year":
            signal_time.year,

        "Month":
            signal_time.month,

        "Hour":
            signal_time.hour,

        "EntryTime":
            entry_time,

        "EntryPrice":
            entry_price,

        "ExitTime":
            exit_time,

        "ExitPrice":
            exit_price,

        "PnL":
            pnl,

        "BullCross":
            bool(
                df30.loc[
                    signal_time,
                    "BullCross"
                ]
            )
    })


forward = pd.DataFrame(
    rows
)


print(
    "Eligible 30m bars:",
    len(forward)
)

print(
    "Completed MACD bullish crosses:",
    int(
        forward["BullCross"].sum()
    )
)


# ============================================================
# STAT FUNCTIONS
# ============================================================

def profit_factor(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    gross_profit = pnl[
        pnl > 0
    ].sum()

    gross_loss = -pnl[
        pnl < 0
    ].sum()

    if gross_loss == 0:

        if gross_profit > 0:
            return np.inf

        return np.nan

    return (
        gross_profit
        /
        gross_loss
    )


def sqn(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    if len(pnl) < 2:
        return np.nan

    std = np.std(
        pnl,
        ddof=1
    )

    if std == 0:
        return np.nan

    return (
        np.sqrt(len(pnl))
        *
        np.mean(pnl)
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
            np.cumsum(pnl)
        ]
    )

    peak = np.maximum.accumulate(
        equity
    )

    drawdown = equity - peak

    return abs(
        drawdown.min()
    )


def calculate_stats(d):

    pnl = d[
        "PnL"
    ].to_numpy(
        dtype=float
    )

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
            np.median(pnl),

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


# ============================================================
# ACTUAL MACD TRADES
# ============================================================

macd = forward[
    forward["BullCross"]
].copy()

macd = macd.sort_values(
    "SignalTime"
)

macd_stats = calculate_stats(
    macd
)


# ============================================================
# YEAR-BY-YEAR MACD
# ============================================================

macd_year_rows = []


for year in YEARS:

    yd = macd[
        macd["Year"]
        ==
        year
    ]

    s = calculate_stats(
        yd
    )

    macd_year_rows.append({

        "Year":
            year,

        **s
    })


macd_year_df = pd.DataFrame(
    macd_year_rows
)


# ============================================================
# BUILD RANDOM POOLS
#
# Match random signals by:
#   Year
#   Hour
#
# This controls for temporal distribution while testing whether
# MACD timing itself adds information.
#
# Actual MACD cross bars are excluded from random pool.
# ============================================================

random_pool = forward[
    ~forward["BullCross"]
].copy()


# Count MACD signals per Year/Hour
macd_strata = (
    macd
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


print()
print("=" * 180)
print("MACD SIGNAL DISTRIBUTION BY YEAR/HOUR")
print("=" * 180)

print(
    macd_strata
    .to_string(
        index=False
    )
)


# ============================================================
# CHECK RANDOM POOL SIZES
# ============================================================

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


    available = len(
        random_pool[
            (random_pool["Year"] == year)
            &
            (random_pool["Hour"] == hour)
        ]
    )


    if available < required_n:

        raise RuntimeError(
            f"Insufficient random pool for Year={year}, "
            f"Hour={hour}: need {required_n}, "
            f"have {available}"
        )


# ============================================================
# RANDOM SIMULATIONS
# ============================================================

print()
print("=" * 180)
print(
    f"RUNNING {N_RANDOM_RUNS} MATCHED RANDOM SIMULATIONS"
)
print("=" * 180)

rng = np.random.default_rng(
    RANDOM_SEED
)

random_run_rows = []


for run in range(
    1,
    N_RANDOM_RUNS + 1
):

    sampled_parts = []


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


        pool = random_pool[
            (random_pool["Year"] == year)
            &
            (random_pool["Hour"] == hour)
        ]


        chosen_idx = rng.choice(
            pool.index.to_numpy(),
            size=required_n,
            replace=False
        )


        sampled_parts.append(
            pool.loc[
                chosen_idx
            ]
        )


    random_trades = pd.concat(
        sampled_parts,
        ignore_index=True
    )

    random_trades = random_trades.sort_values(
        "SignalTime"
    )


    s = calculate_stats(
        random_trades
    )


    random_run_rows.append({

        "Run":
            run,

        **s
    })


random_runs = pd.DataFrame(
    random_run_rows
)


# ============================================================
# RANDOM DISTRIBUTION SUMMARY
# ============================================================

metrics = [
    "WinRatePct",
    "NetPoints",
    "AveragePnL",
    "MedianPnL",
    "ProfitFactor",
    "SQN",
    "MaxDrawdown"
]


distribution_rows = []


for metric in metrics:

    values = (
        random_runs[
            metric
        ]
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        .dropna()
        .to_numpy()
    )


    distribution_rows.append({

        "Metric":
            metric,

        "RandomMean":
            np.mean(
                values
            ),

        "RandomMedian":
            np.median(
                values
            ),

        "P05":
            np.percentile(
                values,
                5
            ),

        "P25":
            np.percentile(
                values,
                25
            ),

        "P75":
            np.percentile(
                values,
                75
            ),

        "P95":
            np.percentile(
                values,
                95
            )
    })


distribution_df = pd.DataFrame(
    distribution_rows
)


# ============================================================
# MACD PERCENTILE VS RANDOM
# ============================================================

percentile_rows = []


for metric in [
    "WinRatePct",
    "NetPoints",
    "AveragePnL",
    "ProfitFactor",
    "SQN"
]:

    actual_value = macd_stats[
        metric
    ]

    random_values = (
        random_runs[
            metric
        ]
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        .dropna()
        .to_numpy()
    )


    percentile = (
        np.mean(
            random_values
            <=
            actual_value
        )
        *
        100
    )


    percentile_rows.append({

        "Metric":
            metric,

        "MACDValue":
            actual_value,

        "RandomMedian":
            np.median(
                random_values
            ),

        "RandomMean":
            np.mean(
                random_values
            ),

        "MACDPercentile":
            percentile,

        "PctRandomBetter":
            100 - percentile
    })


percentile_df = pd.DataFrame(
    percentile_rows
)


# ============================================================
# YEARLY MACD PF
# ============================================================

year_pfs = (
    macd_year_df[
        "ProfitFactor"
    ]
    .replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )
    .dropna()
)


worst_year_pf = (
    year_pfs.min()
    if len(year_pfs)
    else np.nan
)

best_year_pf = (
    year_pfs.max()
    if len(year_pfs)
    else np.nan
)

profitable_years = int(
    (
        year_pfs > 1
    ).sum()
)

losing_years = int(
    (
        year_pfs < 1
    ).sum()
)


# ============================================================
# SAVE
# ============================================================

forward.to_csv(
    OUTPUT_DIR
    /
    "all_30m_forward_returns.csv",
    index=False
)

macd.to_csv(
    OUTPUT_DIR
    /
    "macd_cross_trades.csv",
    index=False
)

macd_year_df.to_csv(
    OUTPUT_DIR
    /
    "macd_year_by_year.csv",
    index=False
)

random_runs.to_csv(
    OUTPUT_DIR
    /
    "random_1000_runs.csv",
    index=False
)

distribution_df.to_csv(
    OUTPUT_DIR
    /
    "random_distribution_summary.csv",
    index=False
)

percentile_df.to_csv(
    OUTPUT_DIR
    /
    "macd_vs_random_percentiles.csv",
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
    600
)


print()
print("=" * 220)
print("RAW 30M MACD — 5-HOUR HOLD")
print("=" * 220)

print(
    pd.DataFrame(
        [
            macd_stats
        ]
    )
    .round(4)
    .to_string(
        index=False
    )
)


print()
print(
    "Worst year PF:",
    round(
        worst_year_pf,
        4
    )
)

print(
    "Best year PF:",
    round(
        best_year_pf,
        4
    )
)

print(
    "Profitable years:",
    profitable_years,
    "/",
    len(YEARS)
)

print(
    "Losing years:",
    losing_years,
    "/",
    len(YEARS)
)


print()
print("=" * 220)
print("RAW MACD — YEAR BY YEAR")
print("=" * 220)

print(
    macd_year_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("MATCHED RANDOM DISTRIBUTION")
print("=" * 220)

print(
    distribution_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("MACD VS RANDOM")
print("=" * 220)

print(
    percentile_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("INTERPRETATION GUIDE")
print("=" * 220)

print()
print(
    "MACD percentile ~50%  = no apparent advantage over random."
)

print(
    "MACD percentile >90%  = MACD beats about 90% of matched random runs."
)

print(
    "MACD percentile >95%  = stronger evidence that MACD timing matters."
)

print(
    "MACD percentile >99%  = very strong separation from this random baseline."
)

print()
print(
    "Do not judge from aggregate PF alone."
)

print(
    "Also inspect year-by-year PF and stability."
)


print()
print("=" * 220)
print("DONE")
print("=" * 220)

print()
print(
    "MACD trades:",
    len(macd)
)

print(
    "Random simulations:",
    N_RANDOM_RUNS
)

print(
    "Holding period:",
    HOLD_MINUTES,
    "minutes =",
    HOLD_MINUTES / 60,
    "hours"
)

print()
print("Output folder:")
print(OUTPUT_DIR)