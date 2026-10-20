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
    r"\MACD_FinalSetup_TimeBlock_Test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14

HOLD_MINUTES = 240

PRICE_MOVE_MIN = 0.35
PRICE_MOVE_MAX = 0.90

PRIOR_MOVE_MIN = -2.00
PRIOR_MOVE_MAX = 2.00

RANGE_LOOKBACK = 20
MAX_RANGE_HEIGHT = 25.0
MAX_RANGE_DRIFT = 10.0

DIRECTION_LOOKBACK = 20

MAIN_YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026
]


# ============================================================
# TIME BLOCKS
# ============================================================

TIME_BLOCKS = [

    ("00_06", 0, 6),
    ("06_10", 6, 10),
    ("10_14", 10, 14),
    ("14_18", 14, 18),
    ("18_22", 18, 22),
    ("22_24", 22, 24),

    ("06_22", 6, 22),
    ("10_22", 10, 22),
    ("14_22", 14, 22),
    ("10_24", 10, 24),
    ("14_24", 14, 24)
]


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 170)
print("LOADING DATA")
print("=" * 170)

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
# ATR14
# ============================================================

prev_close = df15["Close"].shift(1)

tr = pd.concat(
    [
        df15["High"] - df15["Low"],
        (df15["High"] - prev_close).abs(),
        (df15["Low"] - prev_close).abs()
    ],
    axis=1
).max(axis=1)

df15["ATR14"] = tr.ewm(
    alpha=1.0 / ATR_LENGTH,
    adjust=False
).mean()


# ============================================================
# STATS
# ============================================================

def profit_factor(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    gp = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()

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


# ============================================================
# BUILD TRADE
# ============================================================

def build_trade(signal_time):

    entry_pos = df1.index.searchsorted(
        signal_time,
        side="right"
    )

    if entry_pos >= len(df1):
        return None

    entry_time = df1.index[
        entry_pos
    ]

    entry_price = float(
        df1.iloc[
            entry_pos
        ]["Open"]
    )

    target = (
        entry_time
        +
        pd.Timedelta(
            minutes=HOLD_MINUTES
        )
    )

    exit_pos = df1.index.searchsorted(
        target,
        side="left"
    )

    if exit_pos >= len(df1):
        return None

    exit_time = df1.index[
        exit_pos
    ]

    if (
        exit_time
        >
        target
        +
        pd.Timedelta(
            minutes=2
        )
    ):
        return None

    exit_price = float(
        df1.iloc[
            exit_pos
        ]["Close"]
    )

    return {

        "EntryTime":
            entry_time,

        "ExitTime":
            exit_time,

        "EntryPrice":
            entry_price,

        "ExitPrice":
            exit_price,

        "PnL":
            exit_price
            -
            entry_price
    }


# ============================================================
# RANGE CHECK
# ============================================================

def get_range_info(pos):

    if pos < RANGE_LOOKBACK:

        return {
            "RangeHigh": np.nan,
            "RangeLow": np.nan,
            "RangeHeight": np.nan,
            "RangeDrift": np.nan,
            "ValidRange": False,
            "CrossInsideRange": False
        }

    start_pos = (
        pos
        -
        RANGE_LOOKBACK
    )

    end_pos = (
        pos
        -
        1
    )

    window = df15.iloc[
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

    close_first = float(
        df15.iloc[
            start_pos
        ]["Close"]
    )

    close_last = float(
        df15.iloc[
            end_pos
        ]["Close"]
    )

    range_drift = abs(
        close_last
        -
        close_first
    )

    valid_range = (
        range_height
        <=
        MAX_RANGE_HEIGHT
        and
        range_drift
        <=
        MAX_RANGE_DRIFT
    )

    cross_close = float(
        df15.iloc[
            pos
        ]["Close"]
    )

    cross_inside = (
        valid_range
        and
        range_low
        <=
        cross_close
        <=
        range_high
    )

    return {

        "RangeHigh":
            range_high,

        "RangeLow":
            range_low,

        "RangeHeight":
            range_height,

        "RangeDrift":
            range_drift,

        "ValidRange":
            valid_range,

        "CrossInsideRange":
            cross_inside
    }


# ============================================================
# BUILD DATASET
# ============================================================

signals = df15[
    df15["BullCross"]
].copy()


print()
print("=" * 170)
print("BUILDING FINAL SETUP DATASET")
print("=" * 170)

print(
    "Raw bullish crosses:",
    len(signals)
)


rows = []


for signal_time in signals.index:

    pos = df15.index.get_loc(
        signal_time
    )

    if pos < max(
        RANGE_LOOKBACK,
        DIRECTION_LOOKBACK
    ):
        continue


    trade = build_trade(
        signal_time
    )

    if trade is None:
        continue


    atr14 = float(
        df15.iloc[
            pos
        ]["ATR14"]
    )

    if (
        not np.isfinite(atr14)
        or
        atr14 <= 0
    ):
        continue


    close_now = float(
        df15.iloc[
            pos
        ]["Close"]
    )

    close_prev = float(
        df15.iloc[
            pos - 1
        ]["Close"]
    )


    price_move_15_atr = (
        close_now
        -
        close_prev
    ) / atr14


    direction_start_pos = (
        pos
        -
        DIRECTION_LOOKBACK
    )

    direction_start_close = float(
        df15.iloc[
            direction_start_pos
        ]["Close"]
    )

    prior_move_points = (
        close_prev
        -
        direction_start_close
    )

    prior_move_atr = (
        prior_move_points
        /
        atr14
    )


    range_info = get_range_info(
        pos
    )


    rows.append({

        "SignalTime":
            signal_time,

        "Hour":
            signal_time.hour,

        "Year":
            signal_time.year,

        "PriceMove15ATR":
            price_move_15_atr,

        "PriorMoveATR":
            prior_move_atr,

        "ATR14":
            atr14,

        **range_info,

        **trade
    })


df = pd.DataFrame(
    rows
)


# ============================================================
# FINAL FIXED SETUP
# ============================================================

final_setup = df[
    (
        df["PriceMove15ATR"]
        >=
        PRICE_MOVE_MIN
    )
    &
    (
        df["PriceMove15ATR"]
        <=
        PRICE_MOVE_MAX
    )
    &
    (
        ~df["CrossInsideRange"]
    )
    &
    (
        df["PriorMoveATR"]
        >=
        PRIOR_MOVE_MIN
    )
    &
    (
        df["PriorMoveATR"]
        <=
        PRIOR_MOVE_MAX
    )
].copy()


print(
    "Final setup trades:",
    len(final_setup)
)


# ============================================================
# ALL-TIME BASELINE
# ============================================================

baseline_stats = calculate_stats(
    final_setup
)


# ============================================================
# TIME BLOCK TEST
# ============================================================

block_rows = []
year_rows = []


for name, start_hour, end_hour in TIME_BLOCKS:

    subset = final_setup[
        (
            final_setup["Hour"]
            >=
            start_hour
        )
        &
        (
            final_setup["Hour"]
            <
            end_hour
        )
    ].copy()


    stats = calculate_stats(
        subset
    )


    year_pfs = []
    profitable_years = 0
    min_year_trades = np.inf


    for year in MAIN_YEARS:

        yd = subset[
            subset["Year"]
            ==
            year
        ]

        ys = calculate_stats(
            yd
        )


        year_rows.append({

            "TimeBlock":
                name,

            "StartHour":
                start_hour,

            "EndHour":
                end_hour,

            "Year":
                year,

            **ys
        })


        if np.isfinite(
            ys["ProfitFactor"]
        ):

            year_pfs.append(
                ys["ProfitFactor"]
            )

            if (
                ys["ProfitFactor"]
                >
                1
            ):

                profitable_years += 1


        min_year_trades = min(
            min_year_trades,
            ys["N"]
        )


    block_rows.append({

        "TimeBlock":
            name,

        "StartHour":
            start_hour,

        "EndHour":
            end_hour,

        "N":
            stats["N"],

        "RetentionVsAllTimesPct":
            (
                len(subset)
                /
                len(final_setup)
                *
                100
                if len(final_setup) > 0
                else np.nan
            ),

        "WinRatePct":
            stats["WinRatePct"],

        "NetPoints":
            stats["NetPoints"],

        "AveragePnL":
            stats["AveragePnL"],

        "MedianPnL":
            stats["MedianPnL"],

        "ProfitFactor":
            stats["ProfitFactor"],

        "SQN":
            stats["SQN"],

        "MaxDrawdown":
            stats["MaxDrawdown"],

        "WorstYearPF":
            (
                min(year_pfs)
                if year_pfs
                else np.nan
            ),

        "ProfitableYears":
            profitable_years,

        "MinYearTrades":
            min_year_trades
    })


block_df = pd.DataFrame(
    block_rows
)

year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# RANK TIME BLOCKS
# ============================================================

ranking_df = block_df.copy()


ranking_df["All5Years"] = (
    ranking_df["ProfitableYears"]
    ==
    5
)


ranking_df = ranking_df.sort_values(

    [
        "All5Years",
        "WorstYearPF",
        "RetentionVsAllTimesPct",
        "ProfitFactor"
    ],

    ascending=[
        False,
        False,
        False,
        False
    ]
)


# ============================================================
# YEAR-BY-YEAR ALL-TIME BASELINE
# ============================================================

baseline_year_rows = []


for year in MAIN_YEARS:

    yd = final_setup[
        final_setup["Year"]
        ==
        year
    ]

    ys = calculate_stats(
        yd
    )

    baseline_year_rows.append({

        "Year":
            year,

        **ys
    })


baseline_year_df = pd.DataFrame(
    baseline_year_rows
)


# ============================================================
# SAVE
# ============================================================

final_setup.to_csv(
    OUTPUT_DIR
    /
    "final_setup_all_times.csv",
    index=False
)

block_df.to_csv(
    OUTPUT_DIR
    /
    "time_block_results.csv",
    index=False
)

ranking_df.to_csv(
    OUTPUT_DIR
    /
    "time_block_ranking.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "time_block_year_by_year.csv",
    index=False
)

baseline_year_df.to_csv(
    OUTPUT_DIR
    /
    "baseline_all_times_year_by_year.csv",
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
    400
)

pd.set_option(
    "display.max_rows",
    300
)


print()
print("=" * 210)
print("ALL-TIMES FINAL SETUP BASELINE")
print("=" * 210)

print(
    pd.DataFrame(
        [baseline_stats]
    )
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 210)
print("TIME BLOCK RANKING")
print("=" * 210)

print(
    ranking_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 210)
print("YEAR BY YEAR — TIME BLOCKS")
print("=" * 210)

print(
    year_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 210)
print("DONE")
print("=" * 210)

print()
print("Output folder:")
print(OUTPUT_DIR)