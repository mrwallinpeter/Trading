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
    r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\TestScripts\MACDvsRandom_NoSideways"
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
# SIDEWAYS SETTINGS
# ============================================================

LOOKBACK = 20

MAX_RANGE_POINTS = 25.0
MAX_DRIFT_POINTS = 10.0


# ============================================================
# LOAD 1-MIN DATA
# ============================================================

print()
print("=" * 120)
print("LOADING 1-MIN DATA")
print("=" * 120)

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
    subset=[
        "Open",
        "High",
        "Low",
        "Close"
    ]
)


# ============================================================
# MACD
# ============================================================

df15["FastEMA"] = (
    df15["Close"]
    .ewm(
        span=MACD_FAST,
        adjust=False
    )
    .mean()
)

df15["SlowEMA"] = (
    df15["Close"]
    .ewm(
        span=MACD_SLOW,
        adjust=False
    )
    .mean()
)

df15["MACD"] = (
    df15["FastEMA"]
    -
    df15["SlowEMA"]
)

df15["Signal"] = (
    df15["MACD"]
    .ewm(
        span=MACD_SIGNAL,
        adjust=False
    )
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
# TIME FILTER
# ============================================================

df15["TimeValid"] = (
    (df15.index.hour >= START_HOUR)
    &
    (df15.index.hour < END_HOUR)
)

df15["Year"] = df15.index.year


# ============================================================
# SIDEWAYS CHECK
#
# One single 20-bar window:
#
# bars 1...20 immediately before the cross
#
# sideways if:
#
# range height <= 25
# first-last drift <= 10
#
# trade is rejected only if cross close is still inside range
# ============================================================

def sideways_info(df, pos):

    if pos < LOOKBACK:
        return {
            "RangeValid": False,
            "CrossInsideRange": False,
            "RangeHigh": np.nan,
            "RangeLow": np.nan,
            "RangeHeight": np.nan,
            "RangeDrift": np.nan
        }


    start_pos = pos - LOOKBACK
    end_pos = pos - 1

    window = df.iloc[
        start_pos:end_pos + 1
    ]


    range_high = float(
        window["High"].max()
    )

    range_low = float(
        window["Low"].min()
    )

    range_height = (
        range_high
        -
        range_low
    )


    first_close = float(
        df.iloc[start_pos]["Close"]
    )

    last_close = float(
        df.iloc[end_pos]["Close"]
    )

    range_drift = abs(
        last_close
        -
        first_close
    )


    range_valid = (
        range_height <= MAX_RANGE_POINTS
        and
        range_drift <= MAX_DRIFT_POINTS
    )


    cross_close = float(
        df.iloc[pos]["Close"]
    )


    cross_inside = (
        range_valid
        and
        cross_close >= range_low
        and
        cross_close <= range_high
    )


    return {
        "RangeValid":
            range_valid,

        "CrossInsideRange":
            cross_inside,

        "RangeHigh":
            range_high,

        "RangeLow":
            range_low,

        "RangeHeight":
            range_height,

        "RangeDrift":
            range_drift
    }


# ============================================================
# BUILD ELIGIBLE 240-MIN TRADES
# ============================================================

print()
print("=" * 120)
print("BUILDING ELIGIBLE 240-MINUTE TRADES")
print("=" * 120)


eligible = df15[
    df15["TimeValid"]
].copy()


rows = []


for signal_time, bar in eligible.iterrows():

    pos15 = df15.index.get_loc(
        signal_time
    )


    range_info = sideways_info(
        df15,
        pos15
    )


    # --------------------------------------------------------
    # ENTRY = first 1m bar AFTER completed 15m bar
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
    # EXIT = 240 MINUTES AFTER ENTRY
    # --------------------------------------------------------

    exit_target = (
        entry_time
        +
        pd.Timedelta(
            minutes=HOLD_MINUTES
        )
    )


    exit_pos = df1.index.searchsorted(
        exit_target,
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
        exit_target
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
            pnl,

        "RangeValid":
            range_info["RangeValid"],

        "CrossInsideRange":
            range_info["CrossInsideRange"],

        "RangeHigh":
            range_info["RangeHigh"],

        "RangeLow":
            range_info["RangeLow"],

        "RangeHeight":
            range_info["RangeHeight"],

        "RangeDrift":
            range_info["RangeDrift"]
    })


all_trades = pd.DataFrame(
    rows
)


# ============================================================
# MACD GROUPS
# ============================================================

raw_macd = all_trades[
    all_trades["BullCross"]
].copy()


filtered_macd = raw_macd[
    ~raw_macd["CrossInsideRange"]
].copy()


removed_macd = raw_macd[
    raw_macd["CrossInsideRange"]
].copy()


print()
print("RAW MACD trades:", len(raw_macd))
print("Filtered MACD trades:", len(filtered_macd))
print("Removed sideways MACD trades:", len(removed_macd))

print()
print("Removed percentage:")

if len(raw_macd) > 0:
    print(
        round(
            len(removed_macd)
            /
            len(raw_macd)
            *
            100,
            2
        ),
        "%"
    )


print()
print("RAW vs FILTERED trades per year:")

count_table = pd.DataFrame({
    "Raw":
        raw_macd
        .groupby("Year")
        .size(),

    "Filtered":
        filtered_macd
        .groupby("Year")
        .size(),

    "Removed":
        removed_macd
        .groupby("Year")
        .size()
}).fillna(0).astype(int)

print(
    count_table.to_string()
)


# ============================================================
# STATISTICS
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

    pnl = d[
        "PnL"
    ].to_numpy(
        dtype=float
    )


    n = len(pnl)


    if n == 0:
        return {}


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


    payoff_ratio = (
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
            n,

        "Winners":
            len(winners),

        "Losers":
            len(losers),

        "Flats":
            len(flats),

        "WinRatePct":
            len(winners)
            /
            n
            *
            100,

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

        "AverageWin":
            avg_win,

        "AverageLoss":
            avg_loss,

        "PayoffRatio":
            payoff_ratio,

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
            ),

        "MinTrade":
            pnl.min(),

        "MaxTrade":
            pnl.max()
    }


# ============================================================
# BUILD YEAR STAT TABLE
# ============================================================

def stats_by_year(d, name):

    rows = []


    all_stats = calculate_stats(
        d
    )

    rows.append({

        "Strategy":
            name,

        "Year":
            "ALL",

        **all_stats
    })


    for year in sorted(
        d["Year"].unique()
    ):

        yd = d[
            d["Year"]
            ==
            year
        ]


        stats = calculate_stats(
            yd
        )


        rows.append({

            "Strategy":
                name,

            "Year":
                str(year),

            **stats
        })


    return pd.DataFrame(
        rows
    )


raw_stats = stats_by_year(
    raw_macd,
    "RAW_MACD"
)

filtered_stats = stats_by_year(
    filtered_macd,
    "MACD_NO_SIDEWAYS"
)

removed_stats = stats_by_year(
    removed_macd,
    "REMOVED_SIDEWAYS"
)


strategy_stats = pd.concat(
    [
        raw_stats,
        filtered_stats,
        removed_stats
    ],
    ignore_index=True
)


# ============================================================
# RANDOM CONTROL
#
# IMPORTANT:
#
# Random samples are taken from timestamps that ALSO are not
# inside a valid sideways range.
#
# This means:
#
# MACD | NOT SIDEWAYS
#
# is compared with:
#
# RANDOM | NOT SIDEWAYS
#
# ============================================================

random_pool = all_trades[
    ~all_trades[
        "CrossInsideRange"
    ]
].copy()


filtered_counts = (
    filtered_macd
    .groupby("Year")
    .size()
    .to_dict()
)


rng = np.random.default_rng(
    RANDOM_SEED
)


random_rows = []


print()
print("=" * 120)
print(
    f"RUNNING {RANDOM_SIMULATIONS:,} MATCHED RANDOM SIMULATIONS"
)
print("=" * 120)


for simulation in range(
    1,
    RANDOM_SIMULATIONS + 1
):

    sampled_parts = []


    for year, n_trades in filtered_counts.items():

        pool = random_pool[
            random_pool["Year"]
            ==
            year
        ]


        if len(pool) < n_trades:
            raise ValueError(
                f"Not enough random candidates for {year}"
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
# MACD VS RANDOM SUMMARY
# ============================================================

comparison_rows = []


for _, macd_row in filtered_stats.iterrows():

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
            macd_row["MaxDrawdownPoints"],

        "Random_MaxDD_Mean":
            r["MaxDrawdownPoints"].mean()
    })


comparison_df = pd.DataFrame(
    comparison_rows
)


# ============================================================
# RANDOM PERCENTILES
# ============================================================

percentile_rows = []


metrics = [
    "WinRatePct",
    "AveragePnL",
    "ProfitFactor",
    "NetPoints",
    "SQN",
    "MaxDrawdownPoints"
]


for _, macd_row in filtered_stats.iterrows():

    year = macd_row["Year"]

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


        macd_value = macd_row[
            metric
        ]


        if len(values) == 0:
            continue


        if metric == "MaxDrawdownPoints":

            percentile = (
                np.mean(
                    values >= macd_value
                )
                *
                100
            )

        else:

            percentile = (
                np.mean(
                    values <= macd_value
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

all_trades.to_csv(
    OUTPUT_DIR
    /
    "all_eligible_trades.csv",
    index=False
)

raw_macd.to_csv(
    OUTPUT_DIR
    /
    "raw_macd_trades.csv",
    index=False
)

filtered_macd.to_csv(
    OUTPUT_DIR
    /
    "macd_no_sideways_trades.csv",
    index=False
)

removed_macd.to_csv(
    OUTPUT_DIR
    /
    "removed_sideways_trades.csv",
    index=False
)

strategy_stats.to_csv(
    OUTPUT_DIR
    /
    "strategy_statistics.csv",
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
    "macd_no_sideways_vs_random.csv",
    index=False
)

percentile_df.to_csv(
    OUTPUT_DIR
    /
    "macd_no_sideways_random_percentiles.csv",
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


print()
print("=" * 160)
print("RAW MACD VS MACD EXCLUDING SIDEWAYS")
print("=" * 160)

print(
    strategy_stats
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 160)
print("FILTERED MACD VS MATCHED RANDOM")
print("=" * 160)

print(
    comparison_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 160)
print("FILTERED MACD POSITION VS RANDOM")
print("=" * 160)

print(
    percentile_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 160)
print("DONE")
print("=" * 160)

print()
print("Output folder:")
print(OUTPUT_DIR)