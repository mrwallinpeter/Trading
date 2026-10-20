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
    r"\MACD_TimeBlock_HoldingPeriod_Test"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14

PRICE_MOVE_MIN = 0.35
PRICE_MOVE_MAX = 0.90

PRIOR_MOVE_MIN = -2.0
PRIOR_MOVE_MAX = 2.0

RANGE_LOOKBACK = 20
MAX_RANGE_HEIGHT = 25.0
MAX_RANGE_DRIFT = 10.0

DIRECTION_LOOKBACK = 20

HOLDING_PERIODS = [
    60,
    90,
    120,
    150,
    180,
    210,
    240
]

TIME_BLOCKS = {
    "06_10": (6, 10),
    "10_14": (10, 14),
    "14_18": (14, 18),
    "18_22": (18, 22),
    "06_14": (6, 14),
    "14_22": (14, 22)
}

MAIN_YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026
]


# ============================================================
# LOAD 1-MINUTE DATA
# ============================================================

print()
print("=" * 180)
print("LOADING DATA")
print("=" * 180)

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

    peak = np.maximum.accumulate(
        equity
    )

    dd = equity - peak

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
# RANGE INFO
# ============================================================

def get_range_info(pos):

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

    first_close = float(
        df15.iloc[
            start_pos
        ]["Close"]
    )

    last_close = float(
        df15.iloc[
            end_pos
        ]["Close"]
    )

    range_drift = abs(
        last_close
        -
        first_close
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
# BUILD FIXED ENTRY DATASET
# ============================================================

signals = df15[
    df15["BullCross"]
].copy()


print()
print("=" * 180)
print("BUILDING FIXED ENTRY DATASET")
print("=" * 180)

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


    price_move = (
        close_now
        -
        close_prev
    ) / atr14


    prior_start_close = float(
        df15.iloc[
            pos - DIRECTION_LOOKBACK
        ]["Close"]
    )


    prior_move_points = (
        close_prev
        -
        prior_start_close
    )

    prior_move_atr = (
        prior_move_points
        /
        atr14
    )


    range_info = get_range_info(
        pos
    )


    if not (
        PRICE_MOVE_MIN
        <=
        price_move
        <=
        PRICE_MOVE_MAX
    ):
        continue


    if range_info[
        "CrossInsideRange"
    ]:
        continue


    if not (
        PRIOR_MOVE_MIN
        <=
        prior_move_atr
        <=
        PRIOR_MOVE_MAX
    ):
        continue


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


    rows.append({
        "SignalTime":
            signal_time,

        "Year":
            signal_time.year,

        "Hour":
            signal_time.hour,

        "EntryTime":
            entry_time,

        "EntryPrice":
            entry_price,

        "PriceMove15ATR":
            price_move,

        "PriorMoveATR":
            prior_move_atr,

        "ATR14":
            atr14,

        **range_info
    })


entries = pd.DataFrame(
    rows
)


print(
    "Fixed qualifying entries:",
    len(entries)
)


# ============================================================
# BUILD TRADES FOR EACH HOLDING PERIOD
# ============================================================

all_trade_rows = []


for hold_minutes in HOLDING_PERIODS:

    print(
        f"Processing holding period: "
        f"{hold_minutes} minutes"
    )


    for _, row in entries.iterrows():

        entry_time = row[
            "EntryTime"
        ]

        target_time = (
            entry_time
            +
            pd.Timedelta(
                minutes=hold_minutes
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
            row[
                "EntryPrice"
            ]
        )


        all_trade_rows.append({

            **row.to_dict(),

            "HoldingMinutes":
                hold_minutes,

            "ExitTime":
                exit_time,

            "ExitPrice":
                exit_price,

            "PnL":
                pnl
        })


trades = pd.DataFrame(
    all_trade_rows
)


# ============================================================
# OVERALL TIME BLOCK × HOLDING PERIOD
# ============================================================

overall_rows = []


for block_name, (
    start_hour,
    end_hour
) in TIME_BLOCKS.items():

    for hold_minutes in HOLDING_PERIODS:

        subset = trades[
            (
                trades["Hour"]
                >=
                start_hour
            )
            &
            (
                trades["Hour"]
                <
                end_hour
            )
            &
            (
                trades["HoldingMinutes"]
                ==
                hold_minutes
            )
        ]


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


        overall_rows.append({

            "TimeBlock":
                block_name,

            "StartHour":
                start_hour,

            "EndHour":
                end_hour,

            "HoldingMinutes":
                hold_minutes,

            **stats,

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


overall_df = pd.DataFrame(
    overall_rows
)


# ============================================================
# YEAR-BY-YEAR MATRIX
# ============================================================

year_rows = []


for block_name, (
    start_hour,
    end_hour
) in TIME_BLOCKS.items():

    for hold_minutes in HOLDING_PERIODS:

        for year in MAIN_YEARS:

            subset = trades[
                (
                    trades["Hour"]
                    >=
                    start_hour
                )
                &
                (
                    trades["Hour"]
                    <
                    end_hour
                )
                &
                (
                    trades["HoldingMinutes"]
                    ==
                    hold_minutes
                )
                &
                (
                    trades["Year"]
                    ==
                    year
                )
            ]


            stats = calculate_stats(
                subset
            )


            year_rows.append({

                "TimeBlock":
                    block_name,

                "HoldingMinutes":
                    hold_minutes,

                "Year":
                    year,

                **stats
            })


year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# PF MATRIX
# ============================================================

pf_matrix = overall_df.pivot(
    index="HoldingMinutes",
    columns="TimeBlock",
    values="ProfitFactor"
)


# ============================================================
# AVERAGE PNL MATRIX
# ============================================================

avg_matrix = overall_df.pivot(
    index="HoldingMinutes",
    columns="TimeBlock",
    values="AveragePnL"
)


# ============================================================
# WORST-YEAR PF MATRIX
# ============================================================

worst_pf_matrix = overall_df.pivot(
    index="HoldingMinutes",
    columns="TimeBlock",
    values="WorstYearPF"
)


# ============================================================
# PROFITABLE YEARS MATRIX
# ============================================================

prof_year_matrix = overall_df.pivot(
    index="HoldingMinutes",
    columns="TimeBlock",
    values="ProfitableYears"
)


# ============================================================
# RANKING
# ============================================================

ranking_df = overall_df.copy()

ranking_df["All5Years"] = (
    ranking_df["ProfitableYears"]
    ==
    5
)

ranking_df = ranking_df.sort_values(
    [
        "All5Years",
        "WorstYearPF",
        "ProfitFactor",
        "AveragePnL"
    ],
    ascending=[
        False,
        False,
        False,
        False
    ]
)


# ============================================================
# SAVE
# ============================================================

entries.to_csv(
    OUTPUT_DIR
    /
    "fixed_entries.csv",
    index=False
)

trades.to_csv(
    OUTPUT_DIR
    /
    "all_holding_period_trades.csv",
    index=False
)

overall_df.to_csv(
    OUTPUT_DIR
    /
    "timeblock_holding_summary.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "timeblock_holding_year_by_year.csv",
    index=False
)

ranking_df.to_csv(
    OUTPUT_DIR
    /
    "timeblock_holding_ranking.csv",
    index=False
)

pf_matrix.to_csv(
    OUTPUT_DIR
    /
    "profit_factor_matrix.csv"
)

avg_matrix.to_csv(
    OUTPUT_DIR
    /
    "average_pnl_matrix.csv"
)

worst_pf_matrix.to_csv(
    OUTPUT_DIR
    /
    "worst_year_pf_matrix.csv"
)

prof_year_matrix.to_csv(
    OUTPUT_DIR
    /
    "profitable_years_matrix.csv"
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
    500
)

pd.set_option(
    "display.max_rows",
    1000
)


print()
print("=" * 220)
print("PROFIT FACTOR MATRIX")
print("=" * 220)

print(
    pf_matrix
    .round(4)
    .to_string()
)


print()
print("=" * 220)
print("AVERAGE PNL MATRIX")
print("=" * 220)

print(
    avg_matrix
    .round(4)
    .to_string()
)


print()
print("=" * 220)
print("WORST-YEAR PF MATRIX")
print("=" * 220)

print(
    worst_pf_matrix
    .round(4)
    .to_string()
)


print()
print("=" * 220)
print("PROFITABLE YEARS MATRIX")
print("=" * 220)

print(
    prof_year_matrix
    .to_string()
)


print()
print("=" * 220)
print("RANKING")
print("=" * 220)

print(
    ranking_df[
        [
            "TimeBlock",
            "HoldingMinutes",
            "N",
            "AveragePnL",
            "ProfitFactor",
            "SQN",
            "MaxDrawdown",
            "WorstYearPF",
            "ProfitableYears",
            "MinYearTrades"
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("YEAR BY YEAR")
print("=" * 220)

print(
    year_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("DONE")
print("=" * 220)

print()
print("Output folder:")
print(OUTPUT_DIR)