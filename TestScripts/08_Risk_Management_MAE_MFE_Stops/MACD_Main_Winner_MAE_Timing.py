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
    r"\MACD_Main_Winner_MAE_Timing"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STRATEGY SETTINGS
# ============================================================

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14

PRICE_MOVE_MIN = 0.35
PRICE_MOVE_MAX = 0.90

PRIOR_LOOKBACK = 20
PRIOR_MOVE_MIN = -2.0
PRIOR_MOVE_MAX = 2.0

SESSION_START = 14
SESSION_END = 18

HOLD_MINUTES = 210

RANGE_LOOKBACK = 20
MAX_RANGE_HEIGHT = 25.0
MAX_RANGE_DRIFT = 10.0

YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026
]


# ============================================================
# TIME BUCKETS FOR WHEN MAXIMUM MAE OCCURRED
# ============================================================

TIME_BUCKETS = [
    (0, 15),
    (15, 30),
    (30, 45),
    (45, 60),
    (60, 90),
    (90, 120),
    (120, 150),
    (150, 180),
    (180, 210.0001)
]


# ============================================================
# MAE SIZE BUCKETS
# ============================================================

MAE_ATR_BUCKETS = [
    (-np.inf, 1.0, "<1 ATR"),
    (1.0, 2.0, "1-2 ATR"),
    (2.0, 3.0, "2-3 ATR"),
    (3.0, 4.0, "3-4 ATR"),
    (4.0, np.inf, ">=4 ATR")
]


# ============================================================
# CUMULATIVE TIME CHECKPOINTS
# ============================================================

CUMULATIVE_MINUTES = [
    15,
    30,
    45,
    60,
    90,
    120,
    150,
    180,
    210
]


# ============================================================
# LOAD DATA
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

df15["FastEMA"] = df15["Close"].ewm(
    span=MACD_FAST,
    adjust=False
).mean()

df15["SlowEMA"] = df15["Close"].ewm(
    span=MACD_SLOW,
    adjust=False
).mean()

df15["MACD"] = (
    df15["FastEMA"]
    -
    df15["SlowEMA"]
)

df15["Signal"] = df15["MACD"].ewm(
    span=MACD_SIGNAL,
    adjust=False
).mean()

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
# ORIGINAL 20-BAR RANGE FILTER
# ============================================================

def get_range_info(pos):

    start_pos = pos - RANGE_LOOKBACK
    end_pos = pos - 1

    if start_pos < 0:
        return None

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
        range_height <= MAX_RANGE_HEIGHT
        and
        range_drift <= MAX_RANGE_DRIFT
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
        <= cross_close
        <= range_high
    )

    return {
        "RangeHigh": range_high,
        "RangeLow": range_low,
        "RangeHeight": range_height,
        "RangeDrift": range_drift,
        "ValidRange": valid_range,
        "CrossInside": cross_inside
    }


# ============================================================
# BUILD MAIN NON-RANGE TRADES
# ============================================================

print()
print("=" * 180)
print("BUILDING MAIN NON-RANGE TRADES")
print("=" * 180)

trade_rows = []


for signal_time in df15.index[
    df15["BullCross"]
]:

    pos = df15.index.get_loc(
        signal_time
    )

    if pos < max(
        PRIOR_LOOKBACK,
        RANGE_LOOKBACK
    ):
        continue


    # Session
    if not (
        SESSION_START
        <= signal_time.hour
        <
        SESSION_END
    ):
        continue


    # ATR
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


    # PriceMove
    price_move_atr = (
        close_now
        -
        close_prev
    ) / atr14

    if not (
        PRICE_MOVE_MIN
        <= price_move_atr
        <= PRICE_MOVE_MAX
    ):
        continue


    # PriorMove
    prior_start_close = float(
        df15.iloc[
            pos - PRIOR_LOOKBACK
        ]["Close"]
    )

    prior_move_atr = (
        close_prev
        -
        prior_start_close
    ) / atr14

    if not (
        PRIOR_MOVE_MIN
        <= prior_move_atr
        <= PRIOR_MOVE_MAX
    ):
        continue


    # No-range condition
    range_info = get_range_info(
        pos
    )

    if range_info is None:
        continue

    if range_info["CrossInside"]:
        continue


    # Entry
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


    # Timed exit
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


    trade_rows.append({
        "SignalTime": signal_time,
        "EntryTime": entry_time,
        "ExitTime": exit_time,
        "EntryPrice": entry_price,
        "ExitPrice": exit_price,
        "PnL": pnl,
        "Year": signal_time.year,
        "ATR14": atr14,
        "PriceMoveATR14": price_move_atr,
        "PriorMoveATR14": prior_move_atr
    })


trades = pd.DataFrame(
    trade_rows
)

print(
    "Main non-range trades:",
    len(trades)
)

print(
    "Winners:",
    int(
        (trades["PnL"] > 0).sum()
    )
)

print(
    "Losers:",
    int(
        (trades["PnL"] <= 0).sum()
    )
)


# ============================================================
# KEEP WINNERS ONLY
# ============================================================

winners = trades[
    trades["PnL"] > 0
].copy()


# ============================================================
# CALCULATE EXACT MAE TIMING
# ============================================================

print()
print("=" * 180)
print("CALCULATING EXACT WINNER MAE TIMING")
print("=" * 180)

winner_rows = []


for _, trade in winners.iterrows():

    entry_time = trade[
        "EntryTime"
    ]

    exit_time = trade[
        "ExitTime"
    ]

    entry_price = float(
        trade["EntryPrice"]
    )

    atr14 = float(
        trade["ATR14"]
    )

    path = df1.loc[
        entry_time:exit_time
    ].copy()

    if len(path) == 0:
        continue


    # Lowest low = maximum adverse excursion
    lowest_low = float(
        path["Low"].min()
    )

    mae_time = (
        path["Low"]
        .idxmin()
    )

    mae_points = max(
        0.0,
        entry_price
        -
        lowest_low
    )

    mae_atr = (
        mae_points
        /
        atr14
    )


    # Time from entry to maximum MAE
    minutes_to_mae = (
        mae_time
        -
        entry_time
    ).total_seconds() / 60.0


    # Also calculate MFE for context
    highest_high = float(
        path["High"].max()
    )

    mfe_time = (
        path["High"]
        .idxmax()
    )

    mfe_points = max(
        0.0,
        highest_high
        -
        entry_price
    )

    mfe_atr = (
        mfe_points
        /
        atr14
    )

    minutes_to_mfe = (
        mfe_time
        -
        entry_time
    ).total_seconds() / 60.0


    winner_rows.append({

        **trade.to_dict(),

        "LowestLow":
            lowest_low,

        "MAETime":
            mae_time,

        "MAE_Points":
            mae_points,

        "MAE_ATR":
            mae_atr,

        "MinutesToMAE":
            minutes_to_mae,

        "HighestHigh":
            highest_high,

        "MFETime":
            mfe_time,

        "MFE_Points":
            mfe_points,

        "MFE_ATR":
            mfe_atr,

        "MinutesToMFE":
            minutes_to_mfe
    })


winner_detail = pd.DataFrame(
    winner_rows
)


# ============================================================
# TIME BUCKET FUNCTION
# ============================================================

def get_time_bucket(minutes):

    for lo, hi in TIME_BUCKETS:

        if (
            minutes >= lo
            and
            minutes < hi
        ):
            return f"{int(lo)}-{int(hi)}"

    return "OTHER"


winner_detail[
    "MAETimeBucket"
] = winner_detail[
    "MinutesToMAE"
].apply(
    get_time_bucket
)


# ============================================================
# MAE SIZE BUCKET FUNCTION
# ============================================================

def get_mae_size_bucket(mae_atr):

    for lo, hi, label in MAE_ATR_BUCKETS:

        if (
            mae_atr >= lo
            and
            mae_atr < hi
        ):
            return label

    return "OTHER"


winner_detail[
    "MAESizeBucket"
] = winner_detail[
    "MAE_ATR"
].apply(
    get_mae_size_bucket
)


# ============================================================
# OVERALL TIME DISTRIBUTION
# ============================================================

time_distribution = (
    winner_detail
    .groupby(
        "MAETimeBucket"
    )
    .agg(
        N=(
            "MAE_ATR",
            "size"
        ),

        MeanMAE_ATR=(
            "MAE_ATR",
            "mean"
        ),

        MedianMAE_ATR=(
            "MAE_ATR",
            "median"
        ),

        MeanMAE_Points=(
            "MAE_Points",
            "mean"
        ),

        MedianMAE_Points=(
            "MAE_Points",
            "median"
        ),

        MeanPnL=(
            "PnL",
            "mean"
        )
    )
    .reset_index()
)


# Correct chronological order
bucket_order = [
    "0-15",
    "15-30",
    "30-45",
    "45-60",
    "60-90",
    "90-120",
    "120-150",
    "150-180",
    "180-210"
]

time_distribution[
    "MAETimeBucket"
] = pd.Categorical(
    time_distribution[
        "MAETimeBucket"
    ],
    categories=bucket_order,
    ordered=True
)

time_distribution = (
    time_distribution
    .sort_values(
        "MAETimeBucket"
    )
)


time_distribution[
    "PctOfWinners"
] = (
    100.0
    *
    time_distribution["N"]
    /
    len(winner_detail)
)


# ============================================================
# CUMULATIVE TIME DISTRIBUTION
#
# % of winners whose FINAL / eventual MAE
# had already occurred by X minutes
# ============================================================

cumulative_rows = []

for minute in CUMULATIVE_MINUTES:

    count = int(
        (
            winner_detail[
                "MinutesToMAE"
            ]
            <= minute
        ).sum()
    )

    cumulative_rows.append({

        "Minutes":
            minute,

        "WinnerCount":
            count,

        "TotalWinners":
            len(
                winner_detail
            ),

        "PctEventualMAEReached":
            (
                100.0
                *
                count
                /
                len(winner_detail)
            )
    })


cumulative_df = pd.DataFrame(
    cumulative_rows
)


# ============================================================
# MAE SIZE GROUP TIMING
#
# Especially interested in:
# 3-4 ATR
# >=4 ATR
# ============================================================

mae_size_timing = (
    winner_detail
    .groupby(
        "MAESizeBucket"
    )
    .agg(
        N=(
            "MinutesToMAE",
            "size"
        ),

        MeanMinutesToMAE=(
            "MinutesToMAE",
            "mean"
        ),

        MedianMinutesToMAE=(
            "MinutesToMAE",
            "median"
        ),

        P25MinutesToMAE=(
            "MinutesToMAE",
            lambda x:
                x.quantile(0.25)
        ),

        P75MinutesToMAE=(
            "MinutesToMAE",
            lambda x:
                x.quantile(0.75)
        ),

        MeanMAE_ATR=(
            "MAE_ATR",
            "mean"
        ),

        MedianMAE_ATR=(
            "MAE_ATR",
            "median"
        ),

        MeanPnL=(
            "PnL",
            "mean"
        )
    )
    .reset_index()
)


mae_size_order = [
    "<1 ATR",
    "1-2 ATR",
    "2-3 ATR",
    "3-4 ATR",
    ">=4 ATR"
]

mae_size_timing[
    "MAESizeBucket"
] = pd.Categorical(
    mae_size_timing[
        "MAESizeBucket"
    ],
    categories=mae_size_order,
    ordered=True
)

mae_size_timing = (
    mae_size_timing
    .sort_values(
        "MAESizeBucket"
    )
)


# ============================================================
# CROSS-TABLE:
# MAE SIZE × TIME OF MAE
# ============================================================

size_time_matrix = pd.crosstab(
    winner_detail[
        "MAESizeBucket"
    ],
    winner_detail[
        "MAETimeBucket"
    ]
)

size_time_matrix = (
    size_time_matrix
    .reindex(
        index=mae_size_order,
        columns=bucket_order,
        fill_value=0
    )
)


# ============================================================
# PERCENTAGE MATRIX
# Each row = distribution of timing within MAE size group
# ============================================================

size_time_pct = (
    size_time_matrix
    .div(
        size_time_matrix.sum(
            axis=1
        ),
        axis=0
    )
    *
    100.0
)


# ============================================================
# YEAR-BY-YEAR TIMING
# ============================================================

year_rows = []


for year in YEARS:

    yd = winner_detail[
        winner_detail["Year"]
        ==
        year
    ]

    if len(yd) == 0:
        continue

    year_rows.append({

        "Year":
            year,

        "N_Winners":
            len(yd),

        "MedianMinutesToMAE":
            yd[
                "MinutesToMAE"
            ].median(),

        "MeanMinutesToMAE":
            yd[
                "MinutesToMAE"
            ].mean(),

        "P25MinutesToMAE":
            yd[
                "MinutesToMAE"
            ].quantile(0.25),

        "P75MinutesToMAE":
            yd[
                "MinutesToMAE"
            ].quantile(0.75),

        "PctMAEBy30Min":
            100.0
            *
            (
                yd["MinutesToMAE"]
                <= 30
            ).mean(),

        "PctMAEBy60Min":
            100.0
            *
            (
                yd["MinutesToMAE"]
                <= 60
            ).mean(),

        "PctMAEBy90Min":
            100.0
            *
            (
                yd["MinutesToMAE"]
                <= 90
            ).mean(),

        "PctMAEBy120Min":
            100.0
            *
            (
                yd["MinutesToMAE"]
                <= 120
            ).mean(),

        "PctMAEBy180Min":
            100.0
            *
            (
                yd["MinutesToMAE"]
                <= 180
            ).mean(),

        "MedianMAE_ATR":
            yd[
                "MAE_ATR"
            ].median(),

        "P90MAE_ATR":
            yd[
                "MAE_ATR"
            ].quantile(0.90),

        "P95MAE_ATR":
            yd[
                "MAE_ATR"
            ].quantile(0.95)
    })


year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# DEEP-MAE WINNERS ONLY
#
# This is probably the most interesting section.
# ============================================================

deep_mae = winner_detail[
    winner_detail["MAE_ATR"]
    >= 3.0
].copy()


deep_summary_rows = []


for threshold in [
    3.0,
    3.5,
    4.0
]:

    d = winner_detail[
        winner_detail["MAE_ATR"]
        >= threshold
    ]

    if len(d) == 0:
        continue

    deep_summary_rows.append({

        "MAEThresholdATR":
            threshold,

        "N":
            len(d),

        "PctOfAllWinners":
            100.0
            *
            len(d)
            /
            len(winner_detail),

        "MedianMinutesToMAE":
            d[
                "MinutesToMAE"
            ].median(),

        "MeanMinutesToMAE":
            d[
                "MinutesToMAE"
            ].mean(),

        "PctMAEBy30Min":
            100.0
            *
            (
                d["MinutesToMAE"]
                <= 30
            ).mean(),

        "PctMAEBy60Min":
            100.0
            *
            (
                d["MinutesToMAE"]
                <= 60
            ).mean(),

        "PctMAEBy90Min":
            100.0
            *
            (
                d["MinutesToMAE"]
                <= 90
            ).mean(),

        "PctMAEBy120Min":
            100.0
            *
            (
                d["MinutesToMAE"]
                <= 120
            ).mean(),

        "PctMAEAfter150Min":
            100.0
            *
            (
                d["MinutesToMAE"]
                > 150
            ).mean(),

        "MedianFinalPnL":
            d[
                "PnL"
            ].median(),

        "MeanFinalPnL":
            d[
                "PnL"
            ].mean()
    })


deep_summary_df = pd.DataFrame(
    deep_summary_rows
)


# ============================================================
# SAVE
# ============================================================

winner_detail.to_csv(
    OUTPUT_DIR
    /
    "winner_mae_timing_detail.csv",
    index=False
)

time_distribution.to_csv(
    OUTPUT_DIR
    /
    "winner_mae_time_distribution.csv",
    index=False
)

cumulative_df.to_csv(
    OUTPUT_DIR
    /
    "winner_mae_cumulative_timing.csv",
    index=False
)

mae_size_timing.to_csv(
    OUTPUT_DIR
    /
    "winner_mae_size_timing.csv",
    index=False
)

size_time_matrix.to_csv(
    OUTPUT_DIR
    /
    "winner_mae_size_time_matrix_counts.csv"
)

size_time_pct.to_csv(
    OUTPUT_DIR
    /
    "winner_mae_size_time_matrix_pct.csv"
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "winner_mae_timing_by_year.csv",
    index=False
)

deep_mae.to_csv(
    OUTPUT_DIR
    /
    "deep_mae_winners_ge_3atr.csv",
    index=False
)

deep_summary_df.to_csv(
    OUTPUT_DIR
    /
    "deep_mae_timing_summary.csv",
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
    500
)


print()
print("=" * 220)
print("WINNER MAE TIMING DISTRIBUTION")
print("=" * 220)

print(
    time_distribution
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("CUMULATIVE: WHEN HAS EVENTUAL MAE ALREADY OCCURRED?")
print("=" * 220)

print(
    cumulative_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("MAE SIZE GROUP VS TIME-TO-MAE")
print("=" * 220)

print(
    mae_size_timing
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("MAE SIZE × TIME BUCKET — COUNTS")
print("=" * 220)

print(
    size_time_matrix
    .to_string()
)


print()
print("=" * 220)
print("MAE SIZE × TIME BUCKET — ROW %")
print("=" * 220)

print(
    size_time_pct
    .round(2)
    .to_string()
)


print()
print("=" * 220)
print("YEAR-BY-YEAR WINNER MAE TIMING")
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
print("DEEP MAE WINNERS — >=3 ATR / >=3.5 ATR / >=4 ATR")
print("=" * 220)

print(
    deep_summary_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("DEEPEST WINNING TRADES")
print("=" * 220)

deepest = (
    winner_detail
    .sort_values(
        "MAE_ATR",
        ascending=False
    )
    [
        [
            "SignalTime",
            "EntryTime",
            "Year",
            "ATR14",
            "MAE_Points",
            "MAE_ATR",
            "MinutesToMAE",
            "MFE_Points",
            "MFE_ATR",
            "MinutesToMFE",
            "PnL"
        ]
    ]
    .head(25)
)

print(
    deepest
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