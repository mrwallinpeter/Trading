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
    r"\MACD_Range_WaitThenBreakout_Test"
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

PRIOR_LOOKBACK = 20
PRIOR_MOVE_MIN = -2.0
PRIOR_MOVE_MAX = 2.0


# ============================================================
# RANGE SETTINGS
# ============================================================

RANGE_LOOKBACK = 10
RECENT_RANGE_SEARCH_BARS = 4

MAX_RANGE_HEIGHT = 25.0
MAX_RANGE_DRIFT = 10.0


# ============================================================
# WAIT PERIODS
# ============================================================

WAIT_BARS = [
    2,
    3,
    4,
    5,
    6
]


# ============================================================
# REQUIRED BREAKOUT DISTANCE
# ============================================================

BREAKOUT_ATR_THRESHOLDS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30
]


MAIN_YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026
]


SESSIONS = {

    "ALL_TIMES": {
        "start_hour": None,
        "end_hour": None,
        "hold_minutes": 240
    },

    "06_22": {
        "start_hour": 6,
        "end_hour": 22,
        "hold_minutes": 240
    },

    "14_18": {
        "start_hour": 14,
        "end_hour": 18,
        "hold_minutes": 210
    }
}


# ============================================================
# LOAD DATA
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
        [[0.0], np.cumsum(pnl)]
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
# 10-BAR RANGE
# ============================================================

def calculate_range_before(anchor_pos):

    start_pos = (
        anchor_pos
        -
        RANGE_LOOKBACK
    )

    end_pos = (
        anchor_pos
        -
        1
    )

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
        range_height
        <=
        MAX_RANGE_HEIGHT
        and
        range_drift
        <=
        MAX_RANGE_DRIFT
    )


    return {

        "RangeStartTime":
            df15.index[start_pos],

        "RangeEndTime":
            df15.index[end_pos],

        "RangeHigh":
            range_high,

        "RangeLow":
            range_low,

        "RangeHeight":
            range_height,

        "RangeDrift":
            range_drift,

        "ValidRange":
            valid_range
    }


# ============================================================
# FIND MOST RECENT RANGE DURING PREVIOUS 4 BARS
# ============================================================

def find_recent_range(cross_pos):

    for bars_back in range(
        1,
        RECENT_RANGE_SEARCH_BARS + 1
    ):

        anchor_pos = (
            cross_pos
            -
            bars_back
            +
            1
        )

        info = calculate_range_before(
            anchor_pos
        )

        if info is None:
            continue

        if info["ValidRange"]:

            return {
                **info,
                "RangeBarsBack":
                    bars_back
            }

    return None


# ============================================================
# BUILD ORIGINAL MACD SETUPS
#
# IMPORTANT:
# We do NOT require breakout on MACD cross bar.
# ============================================================

print()
print("=" * 180)
print("BUILDING RECENT RANGE MACD SETUPS")
print("=" * 180)

signals = df15[
    df15["BullCross"]
]

setup_rows = []


for signal_time in signals.index:

    pos = df15.index.get_loc(
        signal_time
    )

    if pos < max(
        PRIOR_LOOKBACK,
        RANGE_LOOKBACK
        +
        RECENT_RANGE_SEARCH_BARS
    ):
        continue


    atr14 = float(
        df15.iloc[pos]["ATR14"]
    )

    if (
        not np.isfinite(atr14)
        or
        atr14 <= 0
    ):
        continue


    close_now = float(
        df15.iloc[pos]["Close"]
    )

    close_prev = float(
        df15.iloc[pos - 1]["Close"]
    )


    # ========================================================
    # PRICE MOVE
    # ========================================================

    price_move_atr = (
        close_now
        -
        close_prev
    ) / atr14


    if not (
        PRICE_MOVE_MIN
        <=
        price_move_atr
        <=
        PRICE_MOVE_MAX
    ):
        continue


    # ========================================================
    # PRIOR MOVE
    # ========================================================

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
        <=
        prior_move_atr
        <=
        PRIOR_MOVE_MAX
    ):
        continue


    # ========================================================
    # FIND RECENT RANGE
    # ========================================================

    range_info = find_recent_range(
        pos
    )

    if range_info is None:
        continue


    setup_rows.append({

        "SignalTime":
            signal_time,

        "SignalPos":
            pos,

        "Year":
            signal_time.year,

        "Month":
            signal_time.month,

        "Hour":
            signal_time.hour,

        "ATR14":
            atr14,

        "CrossClose":
            close_now,

        "PriceMoveATR14":
            price_move_atr,

        "PriorMoveATR14":
            prior_move_atr,

        **range_info
    })


setups = pd.DataFrame(
    setup_rows
)


print(
    "Qualified MACD setups with recent range:",
    len(setups)
)


# ============================================================
# BUILD WAIT-PERIOD CANDIDATES
#
# Freeze:
#   RangeHigh
#   RangeLow
#   ATR14
#
# at original MACD cross.
#
# After N completed 15m bars:
#
# BreakoutATR =
# (Close[t+N] - frozen RangeHigh) / cross ATR14
#
# ============================================================

candidate_rows = []


for _, row in setups.iterrows():

    signal_pos = int(
        row["SignalPos"]
    )

    range_high = float(
        row["RangeHigh"]
    )

    atr14 = float(
        row["ATR14"]
    )


    for wait_bars in WAIT_BARS:

        check_pos = (
            signal_pos
            +
            wait_bars
        )

        if check_pos >= len(df15):
            continue


        check_time = df15.index[
            check_pos
        ]

        check_close = float(
            df15.iloc[
                check_pos
            ]["Close"]
        )


        breakout_points = (
            check_close
            -
            range_high
        )

        breakout_atr = (
            breakout_points
            /
            atr14
        )


        candidate_rows.append({

            **row.to_dict(),

            "WaitBars":
                wait_bars,

            "WaitMinutes":
                wait_bars * 15,

            "CheckTime":
                check_time,

            "CheckClose":
                check_close,

            "BreakoutPoints":
                breakout_points,

            "BreakoutATR":
                breakout_atr
        })


candidates = pd.DataFrame(
    candidate_rows
)


# ============================================================
# BUILD ACTUAL TRADES FOR EACH SESSION / WAIT
# ============================================================

trade_rows = []


for session_name, settings in SESSIONS.items():

    start_hour = settings[
        "start_hour"
    ]

    end_hour = settings[
        "end_hour"
    ]

    hold_minutes = settings[
        "hold_minutes"
    ]


    if start_hour is None:

        session_candidates = (
            candidates.copy()
        )

    else:

        # Session based on ORIGINAL MACD signal time
        session_candidates = candidates[
            (
                candidates["Hour"]
                >=
                start_hour
            )
            &
            (
                candidates["Hour"]
                <
                end_hour
            )
        ].copy()


    for _, row in session_candidates.iterrows():

        check_time = row[
            "CheckTime"
        ]


        entry_pos = df1.index.searchsorted(
            check_time,
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


        target_time = (
            entry_time
            +
            pd.Timedelta(
                minutes=
                hold_minutes
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


        trade_rows.append({

            **row.to_dict(),

            "Session":
                session_name,

            "HoldingMinutes":
                hold_minutes,

            "EntryTime":
                entry_time,

            "EntryPrice":
                entry_price,

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
# WAIT × BREAKOUT THRESHOLD ANALYSIS
# ============================================================

summary_rows = []
year_rows = []


for session_name in SESSIONS:

    sd = trades[
        trades["Session"]
        ==
        session_name
    ]


    for wait_bars in WAIT_BARS:

        wd = sd[
            sd["WaitBars"]
            ==
            wait_bars
        ]


        for threshold in BREAKOUT_ATR_THRESHOLDS:

            d = wd[
                wd["BreakoutATR"]
                >=
                threshold
            ]


            stats = calculate_stats(
                d
            )


            year_pfs = []
            profitable_years = 0
            losing_years = 0
            min_year_trades = np.inf


            for year in MAIN_YEARS:

                yd = d[
                    d["Year"]
                    ==
                    year
                ]


                ys = calculate_stats(
                    yd
                )


                min_year_trades = min(
                    min_year_trades,
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

                    elif (
                        ys["ProfitFactor"]
                        <
                        1
                    ):
                        losing_years += 1


                year_rows.append({

                    "Session":
                        session_name,

                    "WaitBars":
                        wait_bars,

                    "WaitMinutes":
                        wait_bars * 15,

                    "BreakoutATRThreshold":
                        threshold,

                    "Year":
                        year,

                    **ys
                })


            summary_rows.append({

                "Session":
                    session_name,

                "WaitBars":
                    wait_bars,

                "WaitMinutes":
                    wait_bars * 15,

                "BreakoutATRThreshold":
                    threshold,

                **stats,

                "WorstYearPF":
                    (
                        min(year_pfs)
                        if year_pfs
                        else np.nan
                    ),

                "BestYearPF":
                    (
                        max(year_pfs)
                        if year_pfs
                        else np.nan
                    ),

                "ProfitableYears":
                    profitable_years,

                "LosingYears":
                    losing_years,

                "MinYearTrades":
                    min_year_trades
            })


summary_df = pd.DataFrame(
    summary_rows
)

year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# PF MATRIX FOR 06-22
#
# Rows = wait bars
# Columns = breakout ATR requirement
# ============================================================

matrix_0622 = (
    summary_df[
        summary_df["Session"]
        ==
        "06_22"
    ]
    .pivot(
        index="WaitBars",
        columns="BreakoutATRThreshold",
        values="ProfitFactor"
    )
)


# ============================================================
# WORST-YEAR PF MATRIX FOR 06-22
# ============================================================

worst_matrix_0622 = (
    summary_df[
        summary_df["Session"]
        ==
        "06_22"
    ]
    .pivot(
        index="WaitBars",
        columns="BreakoutATRThreshold",
        values="WorstYearPF"
    )
)


# ============================================================
# 4-BAR WAIT YEAR DETAIL
# ============================================================

four_bar_years = year_df[
    year_df["WaitBars"]
    ==
    4
].copy()


# ============================================================
# SAVE
# ============================================================

setups.to_csv(
    OUTPUT_DIR
    /
    "recent_range_macd_setups.csv",
    index=False
)

candidates.to_csv(
    OUTPUT_DIR
    /
    "wait_candidates.csv",
    index=False
)

trades.to_csv(
    OUTPUT_DIR
    /
    "wait_breakout_trades.csv",
    index=False
)

summary_df.to_csv(
    OUTPUT_DIR
    /
    "wait_breakout_summary.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "wait_breakout_year_by_year.csv",
    index=False
)

matrix_0622.to_csv(
    OUTPUT_DIR
    /
    "06_22_pf_matrix.csv"
)

worst_matrix_0622.to_csv(
    OUTPUT_DIR
    /
    "06_22_worst_year_pf_matrix.csv"
)

four_bar_years.to_csv(
    OUTPUT_DIR
    /
    "four_bar_wait_year_by_year.csv",
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
    2000
)

pd.set_option(
    "display.width",
    500
)


print()
print("=" * 220)
print("WAIT × BREAKOUT SUMMARY")
print("=" * 220)

print(
    summary_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("06-22 PROFIT FACTOR MATRIX")
print("=" * 220)

print(
    matrix_0622
    .round(4)
    .to_string()
)


print()
print("=" * 220)
print("06-22 WORST-YEAR PF MATRIX")
print("=" * 220)

print(
    worst_matrix_0622
    .round(4)
    .to_string()
)


print()
print("=" * 220)
print("4-BAR WAIT — YEAR BY YEAR")
print("=" * 220)

print(
    four_bar_years
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