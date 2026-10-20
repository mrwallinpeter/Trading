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
    r"\MACD_HoldingPeriod_WalkForward"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIXED ENTRY SETTINGS
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


# ============================================================
# HOLDING PERIODS
# ============================================================

HOLDING_PERIODS = [
    60,
    90,
    120,
    150,
    180,
    210,
    240
]


# ============================================================
# TIME BLOCKS
# ============================================================

TIME_BLOCKS = {

    "06_10": (6, 10),

    "10_14": (10, 14),

    "06_14": (6, 14),

    "14_18": (14, 18),

    "18_22": (18, 22),

    "14_22": (14, 22)
}


# ============================================================
# WALK-FORWARD YEARS
#
# 2021 is partial.
#
# 2023:
#   train = 2022
#
# 2024:
#   train = 2022-2023
#
# 2025:
#   train = 2022-2024
#
# 2026:
#   train = 2022-2025
#
# ============================================================

TRAIN_START_YEAR = 2022

TEST_YEARS = [
    2023,
    2024,
    2025,
    2026
]


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 190)
print("LOADING DATA")
print("=" * 190)

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
# STAT FUNCTIONS
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

    drawdown = equity - peak

    return abs(
        drawdown.min()
    )


def calculate_stats(df):

    pnl = (
        df["PnL"]
        .dropna()
        .to_numpy(dtype=float)
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
            np.mean(pnl > 0) * 100,

        "NetPoints":
            pnl.sum(),

        "AveragePnL":
            pnl.mean(),

        "MedianPnL":
            np.median(pnl),

        "ProfitFactor":
            profit_factor(pnl),

        "SQN":
            sqn(pnl),

        "MaxDrawdown":
            max_drawdown(pnl)
    }


# ============================================================
# RANGE
# ============================================================

def get_range_info(pos):

    start_pos = pos - RANGE_LOOKBACK
    end_pos = pos - 1

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
        df15.iloc[start_pos]["Close"]
    )

    last_close = float(
        df15.iloc[end_pos]["Close"]
    )

    range_drift = abs(
        last_close
        -
        first_close
    )

    valid_range = (
        range_height <= MAX_RANGE_HEIGHT
        and
        range_drift <= MAX_RANGE_DRIFT
    )

    cross_close = float(
        df15.iloc[pos]["Close"]
    )

    cross_inside = (
        valid_range
        and
        range_low <= cross_close <= range_high
    )

    return {
        "RangeHigh": range_high,
        "RangeLow": range_low,
        "RangeHeight": range_height,
        "RangeDrift": range_drift,
        "ValidRange": valid_range,
        "CrossInsideRange": cross_inside
    }


# ============================================================
# BUILD FIXED ENTRY SET
# ============================================================

print()
print("=" * 190)
print("BUILDING FIXED QUALIFYING ENTRIES")
print("=" * 190)

signals = df15[
    df15["BullCross"]
]

entry_rows = []


for signal_time in signals.index:

    pos = df15.index.get_loc(
        signal_time
    )

    if pos < max(
        RANGE_LOOKBACK,
        DIRECTION_LOOKBACK
    ):
        continue


    atr = float(
        df15.iloc[pos]["ATR14"]
    )

    if (
        not np.isfinite(atr)
        or
        atr <= 0
    ):
        continue


    close_now = float(
        df15.iloc[pos]["Close"]
    )

    close_prev = float(
        df15.iloc[pos - 1]["Close"]
    )


    price_move = (
        close_now
        -
        close_prev
    ) / atr


    prior_close = float(
        df15.iloc[
            pos - DIRECTION_LOOKBACK
        ]["Close"]
    )

    prior_move = (
        close_prev
        -
        prior_close
    ) / atr


    range_info = get_range_info(
        pos
    )


    # ----------------------------
    # FINAL FIXED ENTRY FILTER
    # ----------------------------

    if not (
        PRICE_MOVE_MIN
        <= price_move
        <= PRICE_MOVE_MAX
    ):
        continue


    if range_info["CrossInsideRange"]:
        continue


    if not (
        PRIOR_MOVE_MIN
        <= prior_move
        <= PRIOR_MOVE_MAX
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


    entry_rows.append({

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
            prior_move,

        **range_info
    })


entries = pd.DataFrame(
    entry_rows
)

print(
    "Fixed qualifying entries:",
    len(entries)
)


# ============================================================
# CALCULATE EVERY HOLDING PERIOD
# ============================================================

print()
print("=" * 190)
print("CALCULATING ALL HOLDING PERIOD OUTCOMES")
print("=" * 190)

trade_rows = []


for hold in HOLDING_PERIODS:

    print(
        "Holding period:",
        hold
    )


    for _, row in entries.iterrows():

        target_time = (
            row["EntryTime"]
            +
            pd.Timedelta(
                minutes=hold
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


        # Same strict execution rule
        if (
            exit_time
            >
            target_time
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
            row["EntryPrice"]
        )


        trade_rows.append({

            **row.to_dict(),

            "HoldingMinutes":
                hold,

            "ExitTime":
                exit_time,

            "ExitPrice":
                exit_price,

            "PnL":
                pnl
        })


trades = pd.DataFrame(
    trade_rows
)


# ============================================================
# TRAINING PERFORMANCE FOR ONE HOLD
# ============================================================

def training_metrics(
    block_trades,
    hold,
    train_years
):

    d = block_trades[
        (
            block_trades["HoldingMinutes"]
            ==
            hold
        )
        &
        (
            block_trades["Year"]
            .isin(train_years)
        )
    ]


    overall = calculate_stats(
        d
    )


    year_pfs = []
    profitable_years = 0
    year_trade_counts = []


    for year in train_years:

        yd = d[
            d["Year"] == year
        ]

        ys = calculate_stats(
            yd
        )

        year_trade_counts.append(
            ys["N"]
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


    worst_pf = (
        min(year_pfs)
        if year_pfs
        else np.nan
    )


    return {

        "HoldingMinutes":
            hold,

        "TrainN":
            overall["N"],

        "TrainAveragePnL":
            overall["AveragePnL"],

        "TrainPF":
            overall["ProfitFactor"],

        "TrainSQN":
            overall["SQN"],

        "TrainMaxDD":
            overall["MaxDrawdown"],

        "TrainWorstYearPF":
            worst_pf,

        "TrainProfitableYears":
            profitable_years,

        "TrainTotalYears":
            len(train_years),

        "TrainMinYearTrades":
            (
                min(year_trade_counts)
                if year_trade_counts
                else 0
            )
    }


# ============================================================
# SELECTION RULE
#
# Priority:
#
# 1. Number of profitable training years
# 2. Worst training-year PF
# 3. Overall training PF
# 4. Average training PnL
#
# This deliberately favors stability over maximum PF.
# ============================================================

def choose_holding_period(
    block_trades,
    train_years
):

    rows = []

    for hold in HOLDING_PERIODS:

        rows.append(
            training_metrics(
                block_trades,
                hold,
                train_years
            )
        )


    candidates = pd.DataFrame(
        rows
    )


    # NaNs should rank last
    candidates[
        "TrainWorstYearPF_rank"
    ] = candidates[
        "TrainWorstYearPF"
    ].fillna(-999)

    candidates[
        "TrainPF_rank"
    ] = candidates[
        "TrainPF"
    ].fillna(-999)

    candidates[
        "TrainAveragePnL_rank"
    ] = candidates[
        "TrainAveragePnL"
    ].fillna(-999)


    candidates = candidates.sort_values(

        [
            "TrainProfitableYears",
            "TrainWorstYearPF_rank",
            "TrainPF_rank",
            "TrainAveragePnL_rank"
        ],

        ascending=[
            False,
            False,
            False,
            False
        ]
    )


    winner = candidates.iloc[0]


    return (
        int(
            winner[
                "HoldingMinutes"
            ]
        ),
        candidates
    )


# ============================================================
# WALK FORWARD
# ============================================================

print()
print("=" * 190)
print("EXPANDING WALK-FORWARD")
print("=" * 190)


walk_rows = []

selection_detail_rows = []

oos_trade_frames = []


for block_name, (
    start_hour,
    end_hour
) in TIME_BLOCKS.items():


    block_trades = trades[
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
    ].copy()


    for test_year in TEST_YEARS:


        train_years = list(
            range(
                TRAIN_START_YEAR,
                test_year
            )
        )


        chosen_hold, candidates = (
            choose_holding_period(
                block_trades,
                train_years
            )
        )


        # Save every training candidate
        for _, candidate in candidates.iterrows():

            selection_detail_rows.append({

                "TimeBlock":
                    block_name,

                "TestYear":
                    test_year,

                "TrainYears":
                    f"{train_years[0]}-{train_years[-1]}",

                "CandidateHold":
                    int(
                        candidate[
                            "HoldingMinutes"
                        ]
                    ),

                "Selected":
                    int(
                        candidate[
                            "HoldingMinutes"
                        ]
                    )
                    ==
                    chosen_hold,

                "TrainN":
                    candidate[
                        "TrainN"
                    ],

                "TrainAveragePnL":
                    candidate[
                        "TrainAveragePnL"
                    ],

                "TrainPF":
                    candidate[
                        "TrainPF"
                    ],

                "TrainWorstYearPF":
                    candidate[
                        "TrainWorstYearPF"
                    ],

                "TrainProfitableYears":
                    candidate[
                        "TrainProfitableYears"
                    ],

                "TrainTotalYears":
                    candidate[
                        "TrainTotalYears"
                    ],

                "TrainMinYearTrades":
                    candidate[
                        "TrainMinYearTrades"
                    ],

                "TrainSQN":
                    candidate[
                        "TrainSQN"
                    ],

                "TrainMaxDD":
                    candidate[
                        "TrainMaxDD"
                    ]
            })


        selected_training = candidates[
            candidates["HoldingMinutes"]
            ==
            chosen_hold
        ].iloc[0]


        # --------------------------------------------
        # PURE OOS YEAR
        # --------------------------------------------

        test = block_trades[
            (
                block_trades["Year"]
                ==
                test_year
            )
            &
            (
                block_trades["HoldingMinutes"]
                ==
                chosen_hold
            )
        ].copy()


        test_stats = calculate_stats(
            test
        )


        walk_rows.append({

            "TimeBlock":
                block_name,

            "TestYear":
                test_year,

            "TrainYears":
                f"{train_years[0]}-{train_years[-1]}",

            "SelectedHold":
                chosen_hold,

            "TrainN":
                selected_training[
                    "TrainN"
                ],

            "TrainPF":
                selected_training[
                    "TrainPF"
                ],

            "TrainWorstYearPF":
                selected_training[
                    "TrainWorstYearPF"
                ],

            "TrainProfitableYears":
                selected_training[
                    "TrainProfitableYears"
                ],

            "OOS_N":
                test_stats[
                    "N"
                ],

            "OOS_WinRatePct":
                test_stats[
                    "WinRatePct"
                ],

            "OOS_NetPoints":
                test_stats[
                    "NetPoints"
                ],

            "OOS_AveragePnL":
                test_stats[
                    "AveragePnL"
                ],

            "OOS_ProfitFactor":
                test_stats[
                    "ProfitFactor"
                ],

            "OOS_SQN":
                test_stats[
                    "SQN"
                ],

            "OOS_MaxDrawdown":
                test_stats[
                    "MaxDrawdown"
                ]
        })


        if len(test) > 0:

            temp = test.copy()

            temp[
                "WalkForwardSelectedHold"
            ] = chosen_hold

            temp[
                "WalkForwardTestYear"
            ] = test_year

            temp[
                "WalkForwardTimeBlock"
            ] = block_name

            oos_trade_frames.append(
                temp
            )


walk_df = pd.DataFrame(
    walk_rows
)

selection_detail_df = pd.DataFrame(
    selection_detail_rows
)


if oos_trade_frames:

    oos_trades = pd.concat(
        oos_trade_frames,
        ignore_index=True
    )

else:

    oos_trades = pd.DataFrame()


# ============================================================
# AGGREGATE OOS RESULTS BY TIME BLOCK
# ============================================================

summary_rows = []


for block_name in TIME_BLOCKS:


    d = oos_trades[
        oos_trades[
            "WalkForwardTimeBlock"
        ]
        ==
        block_name
    ].copy()


    stats = calculate_stats(
        d
    )


    yearly_pfs = []
    profitable_years = 0
    min_year_trades = np.inf


    for year in TEST_YEARS:

        yd = d[
            d[
                "WalkForwardTestYear"
            ]
            ==
            year
        ]

        ys = calculate_stats(
            yd
        )


        if np.isfinite(
            ys["ProfitFactor"]
        ):

            yearly_pfs.append(
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


    summary_rows.append({

        "TimeBlock":
            block_name,

        "OOS_N":
            stats["N"],

        "OOS_WinRatePct":
            stats["WinRatePct"],

        "OOS_NetPoints":
            stats["NetPoints"],

        "OOS_AveragePnL":
            stats["AveragePnL"],

        "OOS_ProfitFactor":
            stats["ProfitFactor"],

        "OOS_SQN":
            stats["SQN"],

        "OOS_MaxDrawdown":
            stats["MaxDrawdown"],

        "OOS_WorstYearPF":
            (
                min(yearly_pfs)
                if yearly_pfs
                else np.nan
            ),

        "OOS_ProfitableYears":
            profitable_years,

        "OOS_TotalYears":
            len(TEST_YEARS),

        "OOS_MinYearTrades":
            min_year_trades
    })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# HOLD SELECTION HISTORY
# ============================================================

hold_history = walk_df.pivot(
    index="TestYear",
    columns="TimeBlock",
    values="SelectedHold"
)


# ============================================================
# OOS PF BY YEAR
# ============================================================

oos_pf_matrix = walk_df.pivot(
    index="TestYear",
    columns="TimeBlock",
    values="OOS_ProfitFactor"
)


# ============================================================
# OOS AVERAGE PNL BY YEAR
# ============================================================

oos_avg_matrix = walk_df.pivot(
    index="TestYear",
    columns="TimeBlock",
    values="OOS_AveragePnL"
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

walk_df.to_csv(
    OUTPUT_DIR
    /
    "walkforward_year_by_year.csv",
    index=False
)

selection_detail_df.to_csv(
    OUTPUT_DIR
    /
    "walkforward_training_candidates.csv",
    index=False
)

oos_trades.to_csv(
    OUTPUT_DIR
    /
    "walkforward_oos_trades.csv",
    index=False
)

summary_df.to_csv(
    OUTPUT_DIR
    /
    "walkforward_oos_summary.csv",
    index=False
)

hold_history.to_csv(
    OUTPUT_DIR
    /
    "selected_holding_periods.csv"
)

oos_pf_matrix.to_csv(
    OUTPUT_DIR
    /
    "oos_profit_factor_matrix.csv"
)

oos_avg_matrix.to_csv(
    OUTPUT_DIR
    /
    "oos_average_pnl_matrix.csv"
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
print("SELECTED HOLDING PERIOD — EACH OOS YEAR")
print("=" * 220)

print(
    hold_history
    .to_string()
)


print()
print("=" * 220)
print("PURE OOS PROFIT FACTOR — EACH YEAR")
print("=" * 220)

print(
    oos_pf_matrix
    .round(4)
    .to_string()
)


print()
print("=" * 220)
print("PURE OOS AVERAGE PNL — EACH YEAR")
print("=" * 220)

print(
    oos_avg_matrix
    .round(4)
    .to_string()
)


print()
print("=" * 220)
print("WALK-FORWARD YEAR-BY-YEAR DETAILS")
print("=" * 220)

print(
    walk_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("AGGREGATE PURE OOS RESULTS")
print("=" * 220)

print(
    summary_df
    .sort_values(
        [
            "OOS_ProfitableYears",
            "OOS_WorstYearPF",
            "OOS_ProfitFactor"
        ],
        ascending=[
            False,
            False,
            False
        ]
    )
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