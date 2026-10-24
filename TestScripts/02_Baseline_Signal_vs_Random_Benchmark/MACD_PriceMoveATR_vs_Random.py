import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\TestScripts"
    r"\MACD_Momentum_Strength"
    r"\macd_momentum_feature_dataset.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\TestScripts"
    r"\MACD_PriceMoveATR_vs_Random"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

PRICE_MOVE_MIN = 0.35
PRICE_MOVE_MAX = 0.90

RANDOM_SIMULATIONS = 1000
RANDOM_SEED = 42

STABILITY_YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026
]


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 140)
print("LOADING DATASET")
print("=" * 140)

df = pd.read_csv(
    INPUT_FILE
)

df["SignalTime"] = pd.to_datetime(
    df["SignalTime"]
)

print("Rows:", len(df))
print("Start:", df["SignalTime"].min())
print("End:  ", df["SignalTime"].max())


# ============================================================
# FILTER MACD TRADES
# ============================================================

macd = df[
    (df["PriceMove15ATR"] >= PRICE_MOVE_MIN)
    &
    (df["PriceMove15ATR"] <= PRICE_MOVE_MAX)
].copy()

print()
print("=" * 140)
print("MACD FILTER")
print("=" * 140)

print("PriceMove15ATR range:")
print(
    PRICE_MOVE_MIN,
    "to",
    PRICE_MOVE_MAX
)

print("Filtered MACD trades:", len(macd))


# ============================================================
# PERFORMANCE FUNCTIONS
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


def t_stat(pnl):

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
        np.mean(pnl)
        /
        (
            std
            /
            np.sqrt(len(pnl))
        )
    )


def max_drawdown(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    if len(pnl) == 0:
        return np.nan

    equity = np.cumsum(
        pnl
    )

    equity = np.concatenate(
        [
            [0.0],
            equity
        ]
    )

    running_max = np.maximum.accumulate(
        equity
    )

    dd = (
        equity
        -
        running_max
    )

    return abs(
        dd.min()
    )


def calculate_stats(d):

    pnl = (
        d["PnL"]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if len(pnl) == 0:

        return {
            "N": 0,
            "Winners": 0,
            "Losers": 0,
            "Flats": 0,
            "WinRatePct": np.nan,
            "LossRatePct": np.nan,
            "NetPoints": np.nan,
            "AveragePnL": np.nan,
            "MedianPnL": np.nan,
            "StdPnL": np.nan,
            "AverageWin": np.nan,
            "MedianWin": np.nan,
            "AverageLoss": np.nan,
            "MedianLoss": np.nan,
            "PayoffRatio": np.nan,
            "ProfitFactor": np.nan,
            "SQN": np.nan,
            "TStat": np.nan,
            "MaxDrawdownPoints": np.nan
        }

    winners = pnl[
        pnl > 0
    ]

    losers = pnl[
        pnl < 0
    ]

    flats = pnl[
        pnl == 0
    ]

    avg_win = (
        winners.mean()
        if len(winners) > 0
        else np.nan
    )

    avg_loss = (
        losers.mean()
        if len(losers) > 0
        else np.nan
    )

    payoff = (
        avg_win
        /
        abs(avg_loss)
        if (
            len(winners) > 0
            and
            len(losers) > 0
            and
            avg_loss != 0
        )
        else np.nan
    )

    return {

        "N":
            len(pnl),

        "Winners":
            len(winners),

        "Losers":
            len(losers),

        "Flats":
            len(flats),

        "WinRatePct":
            len(winners)
            /
            len(pnl)
            *
            100,

        "LossRatePct":
            len(losers)
            /
            len(pnl)
            *
            100,

        "NetPoints":
            pnl.sum(),

        "AveragePnL":
            pnl.mean(),

        "MedianPnL":
            np.median(
                pnl
            ),

        "StdPnL":
            np.std(
                pnl,
                ddof=1
            ),

        "AverageWin":
            avg_win,

        "MedianWin":
            (
                np.median(
                    winners
                )
                if len(winners) > 0
                else np.nan
            ),

        "AverageLoss":
            avg_loss,

        "MedianLoss":
            (
                np.median(
                    losers
                )
                if len(losers) > 0
                else np.nan
            ),

        "PayoffRatio":
            payoff,

        "ProfitFactor":
            profit_factor(
                pnl
            ),

        "SQN":
            sqn(
                pnl
            ),

        "TStat":
            t_stat(
                pnl
            ),

        "MaxDrawdownPoints":
            max_drawdown(
                pnl
            )
    }


# ============================================================
# MACD STATS
# ============================================================

def stats_by_year(d):

    rows = []

    rows.append({
        "Year":
            "ALL",

        **calculate_stats(
            d
        )
    })

    for year in sorted(
        d["Year"].unique()
    ):

        yd = d[
            d["Year"]
            ==
            year
        ]

        rows.append({
            "Year":
                str(year),

            **calculate_stats(
                yd
            )
        })

    return pd.DataFrame(
        rows
    )


macd_stats = stats_by_year(
    macd
)


# ============================================================
# RANDOM CONTROL
#
# IMPORTANT:
#
# The input dataset contains only MACD crosses.
#
# Therefore we cannot construct a true random-time control
# directly from this file.
#
# We must rebuild all eligible 15-minute timestamps from
# the original 1-minute data.
# ============================================================

ONE_MIN_FILE = Path(
    r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"
)

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14

START_HOUR = 10
END_HOUR = 18

HOLD_MINUTES = 240


print()
print("=" * 140)
print("BUILDING TRUE RANDOM-TIME CONTROL POOL")
print("=" * 140)


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


# ============================================================
# 15-MIN BARS
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


# ============================================================
# ATR
# ============================================================

def wilder_atr(data, length):

    prev_close = data[
        "Close"
    ].shift(1)

    tr = pd.concat(
        [
            data["High"]
            -
            data["Low"],

            (
                data["High"]
                -
                prev_close
            ).abs(),

            (
                data["Low"]
                -
                prev_close
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1.0 / length,
        adjust=False
    ).mean()


df15["ATR"] = wilder_atr(
    df15,
    ATR_LENGTH
)


# ============================================================
# PRICE MOVE 15 / ATR
# ============================================================

df15["PriceMove15ATR"] = (
    (
        df15["Close"]
        -
        df15["Close"].shift(1)
    )
    /
    df15["ATR"]
)


# ============================================================
# ELIGIBLE TIME WINDOW
# ============================================================

df15["TimeValid"] = (
    (df15.index.hour >= START_HOUR)
    &
    (df15.index.hour < END_HOUR)
)


# ============================================================
# BUILD ALL ELIGIBLE OUTCOMES
# ============================================================

rows = []


for signal_time, bar in df15[
    df15["TimeValid"]
].iterrows():

    value = bar[
        "PriceMove15ATR"
    ]

    if not np.isfinite(
        value
    ):
        continue


    # Same momentum regime
    if not (
        PRICE_MOVE_MIN
        <=
        value
        <=
        PRICE_MOVE_MAX
    ):
        continue


    # --------------------------------------------------------
    # ENTRY
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    target_time = (
        entry_time
        +
        pd.Timedelta(
            minutes=HOLD_MINUTES
        )
    )

    exit_pos = df1.index.searchsorted(
        target_time,
        side="left"
    )

    if exit_pos >= len(df1):
        continue

    exit_time = df1.index[
        exit_pos
    ]

    if (
        exit_time
        >
        target_time
        +
        pd.Timedelta(
            minutes=2
        )
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

        "PriceMove15ATR":
            float(value),

        "EntryPrice":
            entry_price,

        "ExitPrice":
            exit_price,

        "PnL":
            pnl
    })


random_pool = pd.DataFrame(
    rows
)

print(
    "Eligible timestamps in same momentum regime:",
    len(random_pool)
)


# ============================================================
# YEAR MATCHING
# ============================================================

macd_counts = (
    macd
    .groupby("Year")
    .size()
    .to_dict()
)


print()
print("MACD sample size by year:")

for year, count in macd_counts.items():

    print(
        year,
        count
    )


# ============================================================
# RANDOM SIMULATIONS
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

random_rows = []


print()
print("=" * 140)
print(
    f"RUNNING {RANDOM_SIMULATIONS:,} MATCHED RANDOM SIMULATIONS"
)
print("=" * 140)


for simulation in range(
    1,
    RANDOM_SIMULATIONS + 1
):

    sampled_parts = []


    for year, n_trades in macd_counts.items():

        pool = random_pool[
            random_pool["Year"]
            ==
            year
        ]

        if len(pool) < n_trades:

            raise ValueError(
                f"Not enough eligible timestamps in {year}. "
                f"Pool={len(pool)}, needed={n_trades}"
            )

        positions = rng.choice(
            len(pool),
            size=n_trades,
            replace=False
        )

        sample = pool.iloc[
            positions
        ].copy()

        sampled_parts.append(
            sample
        )


        stats = calculate_stats(
            sample
        )

        random_rows.append({

            "Simulation":
                simulation,

            "Year":
                str(year),

            **stats
        })


    sampled_all = pd.concat(
        sampled_parts,
        ignore_index=True
    )

    stats = calculate_stats(
        sampled_all
    )

    random_rows.append({

        "Simulation":
            simulation,

        "Year":
            "ALL",

        **stats
    })


random_stats = pd.DataFrame(
    random_rows
)


# ============================================================
# COMPARISON
# ============================================================

comparison_rows = []


for _, macd_row in macd_stats.iterrows():

    year = macd_row[
        "Year"
    ]

    r = random_stats[
        random_stats["Year"]
        ==
        year
    ]

    comparison_rows.append({

        "Year":
            year,

        "N":
            int(
                macd_row["N"]
            ),

        "MACD_WinRate":
            macd_row["WinRatePct"],

        "Random_WinRate_Mean":
            r["WinRatePct"].mean(),

        "MACD_AvgPnL":
            macd_row["AveragePnL"],

        "Random_AvgPnL_Mean":
            r["AveragePnL"].mean(),

        "MACD_PF":
            macd_row["ProfitFactor"],

        "Random_PF_Median":
            r["ProfitFactor"]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .median(),

        "MACD_Net":
            macd_row["NetPoints"],

        "Random_Net_Mean":
            r["NetPoints"].mean(),

        "MACD_SQN":
            macd_row["SQN"],

        "Random_SQN_Mean":
            r["SQN"].mean(),

        "MACD_MaxDD":
            macd_row[
                "MaxDrawdownPoints"
            ],

        "Random_MaxDD_Mean":
            r[
                "MaxDrawdownPoints"
            ].mean()
    })


comparison_df = pd.DataFrame(
    comparison_rows
)


# ============================================================
# PERCENTILES
# ============================================================

metrics = [
    "WinRatePct",
    "AveragePnL",
    "ProfitFactor",
    "NetPoints",
    "SQN",
    "MaxDrawdownPoints"
]


percentile_rows = []


for _, macd_row in macd_stats.iterrows():

    year = macd_row[
        "Year"
    ]

    r = random_stats[
        random_stats["Year"]
        ==
        year
    ]

    for metric in metrics:

        values = (
            r[metric]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        if len(values) == 0:
            continue

        macd_value = macd_row[
            metric
        ]

        if metric == "MaxDrawdownPoints":

            percentile = (
                np.mean(
                    values
                    >=
                    macd_value
                )
                *
                100
            )

        else:

            percentile = (
                np.mean(
                    values
                    <=
                    macd_value
                )
                *
                100
            )

        percentile_rows.append({

            "Year":
                year,

            "Metric":
                metric,

            "MACD":
                macd_value,

            "RandomMean":
                values.mean(),

            "RandomMedian":
                values.median(),

            "RandomP05":
                values.quantile(
                    0.05
                ),

            "RandomP95":
                values.quantile(
                    0.95
                ),

            "MACDPercentileVsRandom":
                percentile
        })


percentile_df = pd.DataFrame(
    percentile_rows
)


# ============================================================
# SAVE
# ============================================================

macd.to_csv(
    OUTPUT_DIR
    /
    "macd_price_move_filtered_trades.csv",
    index=False
)

random_pool.to_csv(
    OUTPUT_DIR
    /
    "eligible_random_pool_same_momentum.csv",
    index=False
)

macd_stats.to_csv(
    OUTPUT_DIR
    /
    "macd_statistics.csv",
    index=False
)

random_stats.to_csv(
    OUTPUT_DIR
    /
    "random_simulation_statistics.csv",
    index=False
)

comparison_df.to_csv(
    OUTPUT_DIR
    /
    "macd_vs_random_comparison.csv",
    index=False
)

percentile_df.to_csv(
    OUTPUT_DIR
    /
    "macd_vs_random_percentiles.csv",
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
    "display.width",
    320
)


print()
print("=" * 180)
print(
    "MACD 0.35-0.90 ATR VS RANDOM 0.35-0.90 ATR"
)
print("=" * 180)

print(
    comparison_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 180)
print(
    "MACD POSITION VS RANDOM DISTRIBUTION"
)
print("=" * 180)

print(
    percentile_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 180)
print("DONE")
print("=" * 180)

print()
print("Output folder:")
print(OUTPUT_DIR)