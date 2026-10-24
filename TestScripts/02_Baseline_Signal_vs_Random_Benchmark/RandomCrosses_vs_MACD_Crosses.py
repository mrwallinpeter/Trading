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
    r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MACDvsRandom"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

START_HOUR = 10
END_HOUR = 18

HOLD_MINUTES = 240

RANDOM_SIMULATIONS = 1000

RANDOM_SEED = 42


# ============================================================
# LOAD 1-MIN DATA
# ============================================================

print()
print("=" * 120)
print("LOADING 1-MIN DATA")
print("=" * 120)

df1 = pd.read_csv(ONE_MIN_FILE)

df1["DateTime"] = pd.to_datetime(df1["DateTime"])

df1 = (
    df1
    .sort_values("DateTime")
    .drop_duplicates("DateTime")
    .set_index("DateTime")
)

for col in ["Open", "High", "Low", "Close", "Volume"]:
    df1[col] = pd.to_numeric(df1[col], errors="coerce")

df1 = df1.dropna(
    subset=["Open", "High", "Low", "Close"]
)

print("Rows:", len(df1))
print("Start:", df1.index.min())
print("End:  ", df1.index.max())


# ============================================================
# BUILD 15-MIN DATA
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
    subset=["Open", "High", "Low", "Close"]
)


# ============================================================
# MACD
# ============================================================

df15["FastEMA"] = (
    df15["Close"]
    .ewm(span=MACD_FAST, adjust=False)
    .mean()
)

df15["SlowEMA"] = (
    df15["Close"]
    .ewm(span=MACD_SLOW, adjust=False)
    .mean()
)

df15["MACD"] = (
    df15["FastEMA"]
    -
    df15["SlowEMA"]
)

df15["Signal"] = (
    df15["MACD"]
    .ewm(span=MACD_SIGNAL, adjust=False)
    .mean()
)

df15["Hist"] = (
    df15["MACD"]
    -
    df15["Signal"]
)

df15["BullCross"] = (
    (df15["Hist"].shift(1) <= 0)
    &
    (df15["Hist"] > 0)
)


# ============================================================
# ELIGIBLE TIME WINDOW
# ============================================================

df15["TimeValid"] = (
    (df15.index.hour >= START_HOUR)
    &
    (df15.index.hour < END_HOUR)
)

df15["Year"] = df15.index.year


# ============================================================
# BUILD ALL POSSIBLE LONG TRADES
#
# This is important:
#
# We calculate the 240m return for EVERY eligible 15m
# timestamp first.
#
# Then:
#
# MACD = subset where bullish cross occurred
# RANDOM = random timestamps from same population
#
# Thus execution treatment is identical.
# ============================================================

print()
print("=" * 120)
print("BUILDING ALL ELIGIBLE 240-MINUTE TRADES")
print("=" * 120)


eligible = df15[
    df15["TimeValid"]
].copy()


rows = []


for signal_time, bar in eligible.iterrows():

    # --------------------------------------------------------
    # ENTRY = first 1m bar AFTER completed 15m bar
    # --------------------------------------------------------

    entry_pos = df1.index.searchsorted(
        signal_time,
        side="right"
    )

    if entry_pos >= len(df1):
        continue

    entry_time = df1.index[entry_pos]

    entry_price = float(
        df1.iloc[entry_pos]["Open"]
    )


    # --------------------------------------------------------
    # EXIT = 240 minutes after entry
    # --------------------------------------------------------

    exit_target = (
        entry_time
        +
        pd.Timedelta(minutes=HOLD_MINUTES)
    )

    exit_pos = df1.index.searchsorted(
        exit_target,
        side="left"
    )

    if exit_pos >= len(df1):
        continue

    exit_time = df1.index[exit_pos]


    # Prevent large gaps from pretending to be 240m exits.
    if (
        exit_time
        >
        exit_target
        +
        pd.Timedelta(minutes=2)
    ):
        continue


    exit_price = float(
        df1.iloc[exit_pos]["Close"]
    )

    pnl = (
        exit_price
        -
        entry_price
    )


    rows.append({

        "SignalTime":
            signal_time,

        "EntryTime":
            entry_time,

        "ExitTime":
            exit_time,

        "Year":
            signal_time.year,

        "BullCross":
            bool(bar["BullCross"]),

        "Hist":
            float(bar["Hist"]),

        "EntryPrice":
            entry_price,

        "ExitPrice":
            exit_price,

        "PnL":
            pnl
    })


all_trades = pd.DataFrame(rows)


macd_trades = all_trades[
    all_trades["BullCross"]
].copy()


print("Eligible completed timestamps:", len(all_trades))
print("Completed MACD trades:", len(macd_trades))

print()
print("MACD trades per year:")

print(
    macd_trades
    .groupby("Year")
    .size()
    .to_string()
)


# ============================================================
# STATISTICS
# ============================================================

def profit_factor(pnl):

    pnl = np.asarray(pnl, dtype=float)

    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()

    if gross_loss == 0:

        if gross_profit > 0:
            return np.inf

        return np.nan

    return gross_profit / gross_loss


def max_drawdown(pnl):

    pnl = np.asarray(pnl, dtype=float)

    if len(pnl) == 0:
        return np.nan

    equity = np.cumsum(pnl)

    equity_with_zero = np.concatenate(
        [
            np.array([0.0]),
            equity
        ]
    )

    running_max = np.maximum.accumulate(
        equity_with_zero
    )

    drawdown = (
        equity_with_zero
        -
        running_max
    )

    return abs(drawdown.min())


def sqn(pnl):

    pnl = np.asarray(pnl, dtype=float)

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

    pnl = np.asarray(pnl, dtype=float)

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


def calculate_stats(d):

    pnl = d["PnL"].to_numpy(dtype=float)

    n = len(pnl)

    if n == 0:
        return {}


    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    flats = pnl[pnl == 0]


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


    payoff_ratio = (
        avg_win / abs(avg_loss)
        if (
            len(winners) > 0
            and
            len(losers) > 0
            and
            avg_loss != 0
        )
        else np.nan
    )


    win_rate = (
        len(winners)
        /
        n
    )


    loss_rate = (
        len(losers)
        /
        n
    )


    expectancy_from_components = (
        win_rate * avg_win
        +
        loss_rate * avg_loss
    )


    return {

        "N":
            n,

        "Winners":
            len(winners),

        "Losers":
            len(losers),

        "Flats":
            len(flats),

        "WinRatePct":
            win_rate * 100,

        "LossRatePct":
            loss_rate * 100,

        "NetPoints":
            pnl.sum(),

        "AveragePnL":
            pnl.mean(),

        "MedianPnL":
            np.median(pnl),

        "StdPnL":
            np.std(
                pnl,
                ddof=1
            ),

        "MinTrade":
            pnl.min(),

        "MaxTrade":
            pnl.max(),

        "AverageWin":
            avg_win,

        "MedianWin":
            (
                np.median(winners)
                if len(winners) > 0
                else np.nan
            ),

        "AverageLoss":
            avg_loss,

        "MedianLoss":
            (
                np.median(losers)
                if len(losers) > 0
                else np.nan
            ),

        "PayoffRatio":
            payoff_ratio,

        "ProfitFactor":
            profit_factor(pnl),

        "Expectancy":
            expectancy_from_components,

        "SQN":
            sqn(pnl),

        "TStat":
            t_stat(pnl),

        "MaxDrawdownPoints":
            max_drawdown(pnl)
    }


# ============================================================
# MACD STATISTICS
# ============================================================

macd_rows = []


# Overall
stats = calculate_stats(macd_trades)

macd_rows.append({

    "Year":
        "ALL",

    **stats
})


# By year
for year in sorted(
    macd_trades["Year"].unique()
):

    d = macd_trades[
        macd_trades["Year"]
        ==
        year
    ]

    stats = calculate_stats(d)

    macd_rows.append({

        "Year":
            str(year),

        **stats
    })


macd_stats = pd.DataFrame(macd_rows)


# ============================================================
# RANDOM CONTROL
#
# Each simulation:
#
# For each year:
#   sample exactly same N as MACD trades that year
#
# This avoids differences in yearly sample composition.
# ============================================================

print()
print("=" * 120)
print(
    f"RUNNING {RANDOM_SIMULATIONS:,} MATCHED RANDOM SIMULATIONS"
)
print("=" * 120)


rng = np.random.default_rng(
    RANDOM_SEED
)


macd_counts = (
    macd_trades
    .groupby("Year")
    .size()
    .to_dict()
)


random_rows = []


for simulation in range(
    1,
    RANDOM_SIMULATIONS + 1
):

    random_parts = []


    for year, n_macd in macd_counts.items():

        pool = all_trades[
            all_trades["Year"]
            ==
            year
        ]


        if len(pool) < n_macd:
            raise ValueError(
                f"Not enough eligible random timestamps "
                f"in {year}"
            )


        chosen_positions = rng.choice(
            len(pool),
            size=n_macd,
            replace=False
        )


        sampled = pool.iloc[
            chosen_positions
        ].copy()


        random_parts.append(
            sampled
        )


        # -----------------------------------------------
        # Year statistics
        # -----------------------------------------------

        s = calculate_stats(
            sampled
        )

        random_rows.append({

            "Simulation":
                simulation,

            "Year":
                str(year),

            **s
        })


    # ---------------------------------------------------
    # Overall random portfolio for this simulation
    # ---------------------------------------------------

    random_all = pd.concat(
        random_parts,
        ignore_index=True
    )


    s = calculate_stats(
        random_all
    )

    random_rows.append({

        "Simulation":
            simulation,

        "Year":
            "ALL",

        **s
    })


random_stats = pd.DataFrame(
    random_rows
)


# ============================================================
# RANDOM DISTRIBUTION SUMMARY
# ============================================================

metrics = [

    "WinRatePct",

    "NetPoints",

    "AveragePnL",

    "MedianPnL",

    "StdPnL",

    "AverageWin",

    "AverageLoss",

    "PayoffRatio",

    "ProfitFactor",

    "Expectancy",

    "SQN",

    "TStat",

    "MaxDrawdownPoints"
]


random_summary_rows = []


for year in macd_stats["Year"]:

    r = random_stats[
        random_stats["Year"]
        ==
        year
    ]

    m = macd_stats[
        macd_stats["Year"]
        ==
        year
    ].iloc[0]


    for metric in metrics:

        values = (
            r[metric]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )


        macd_value = m[metric]


        if len(values) == 0:
            continue


        random_mean = values.mean()
        random_median = values.median()

        p05 = values.quantile(0.05)
        p25 = values.quantile(0.25)
        p75 = values.quantile(0.75)
        p95 = values.quantile(0.95)


        # For most metrics, higher = better.
        # Drawdown is the exception.
        if metric == "MaxDrawdownPoints":

            percentile_better = (
                np.mean(
                    values >= macd_value
                )
                *
                100
            )

        else:

            percentile_better = (
                np.mean(
                    values <= macd_value
                )
                *
                100
            )


        random_summary_rows.append({

            "Year":
                year,

            "Metric":
                metric,

            "MACD":
                macd_value,

            "RandomMean":
                random_mean,

            "RandomMedian":
                random_median,

            "RandomP05":
                p05,

            "RandomP25":
                p25,

            "RandomP75":
                p75,

            "RandomP95":
                p95,

            "MACDPercentileVsRandom":
                percentile_better
        })


comparison = pd.DataFrame(
    random_summary_rows
)


# ============================================================
# COMPACT YEAR COMPARISON
# ============================================================

compact_rows = []


for _, macd_row in macd_stats.iterrows():

    year = macd_row["Year"]

    r = random_stats[
        random_stats["Year"]
        ==
        year
    ]


    compact_rows.append({

        "Year":
            year,

        "N":
            int(macd_row["N"]),

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
            macd_row["MaxDrawdownPoints"],

        "Random_MaxDD_Mean":
            r["MaxDrawdownPoints"].mean()
    })


compact_df = pd.DataFrame(
    compact_rows
)


# ============================================================
# SAVE FILES
# ============================================================

all_trades_file = (
    OUTPUT_DIR
    /
    "all_eligible_240m_trades.csv"
)

macd_trades_file = (
    OUTPUT_DIR
    /
    "macd_240m_trades.csv"
)

macd_stats_file = (
    OUTPUT_DIR
    /
    "macd_statistics_by_year.csv"
)

random_stats_file = (
    OUTPUT_DIR
    /
    "random_simulation_statistics.csv"
)

comparison_file = (
    OUTPUT_DIR
    /
    "macd_vs_random_metric_comparison.csv"
)

compact_file = (
    OUTPUT_DIR
    /
    "macd_vs_random_year_summary.csv"
)


all_trades.to_csv(
    all_trades_file,
    index=False
)

macd_trades.to_csv(
    macd_trades_file,
    index=False
)

macd_stats.to_csv(
    macd_stats_file,
    index=False
)

random_stats.to_csv(
    random_stats_file,
    index=False
)

comparison.to_csv(
    comparison_file,
    index=False
)

compact_df.to_csv(
    compact_file,
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
    300
)

pd.set_option(
    "display.max_rows",
    200
)


print()
print("=" * 160)
print("MACD — FULL STATISTICS BY YEAR")
print("=" * 160)

print(
    macd_stats
    .round(4)
    .to_string(index=False)
)


print()
print("=" * 160)
print("MACD VS RANDOM — COMPACT YEAR COMPARISON")
print("=" * 160)

print(
    compact_df
    .round(4)
    .to_string(index=False)
)


# ============================================================
# KEY METRICS WITH RANDOM PERCENTILE
# ============================================================

key_metrics = [
    "WinRatePct",
    "AveragePnL",
    "ProfitFactor",
    "NetPoints",
    "SQN",
    "MaxDrawdownPoints"
]


print()
print("=" * 160)
print("MACD POSITION INSIDE RANDOM DISTRIBUTION")
print("=" * 160)

print(
    comparison[
        comparison["Metric"]
        .isin(key_metrics)
    ]
    .round(4)
    .to_string(index=False)
)


print()
print("=" * 160)
print("DONE")
print("=" * 160)

print()
print("MACD trades:")
print(macd_trades_file)

print()
print("MACD statistics:")
print(macd_stats_file)

print()
print("Random simulations:")
print(random_stats_file)

print()
print("MACD vs random:")
print(comparison_file)

print()
print("Compact year summary:")
print(compact_file)