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
    r"\MACD_ATR14_Absolute_Level_Study"
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

RANGE_LOOKBACK = 20
MAX_RANGE_HEIGHT = 25.0
MAX_RANGE_DRIFT = 10.0

MAIN_YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026
]


# ============================================================
# ANALYSIS WINDOWS
# ============================================================

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
# FIXED ABSOLUTE ATR14 BUCKETS
# ============================================================
#
# These are ATR14 POINT values, not normalized values.
#
# Example:
# ATR14 = 8 means average 15m true range is about 8 MES points.
#
# ============================================================

ATR_BUCKETS = [
    (-np.inf, 5.0, "<5"),
    (5.0, 7.0, "5-7"),
    (7.0, 9.0, "7-9"),
    (9.0, 11.0, "9-11"),
    (11.0, 13.0, "11-13"),
    (13.0, 16.0, "13-16"),
    (16.0, np.inf, ">=16")
]


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 180)
print("LOADING DATA")
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
# WILDER ATR14
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

    drawdown = (
        equity
        -
        peak
    )

    return abs(
        drawdown.min()
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
# RANGE FUNCTION
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

    cross_inside_range = (
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
            cross_inside_range
    }


# ============================================================
# BUILD FINAL QUALIFIED SIGNALS
# ============================================================

print()
print("=" * 180)
print("BUILDING FINAL NON-RANGE SIGNALS")
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
        RANGE_LOOKBACK,
        PRIOR_LOOKBACK
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


    # ========================================================
    # PRICE MOVE FILTER
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
    # PRIOR MOVE FILTER
    # ========================================================

    prior_start_close = float(
        df15.iloc[
            pos - PRIOR_LOOKBACK
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


    if not (
        PRIOR_MOVE_MIN
        <=
        prior_move_atr
        <=
        PRIOR_MOVE_MAX
    ):
        continue


    # ========================================================
    # RANGE FILTER
    # ========================================================

    range_info = get_range_info(
        pos
    )

    if range_info[
        "CrossInsideRange"
    ]:
        continue


    setup_rows.append({

        "SignalTime":
            signal_time,

        "Year":
            signal_time.year,

        "Month":
            signal_time.month,

        "Hour":
            signal_time.hour,

        "ATR14":
            atr14,

        "PriceMoveATR14":
            price_move_atr,

        "PriorMovePoints":
            prior_move_points,

        "PriorMoveATR14":
            prior_move_atr,

        **range_info
    })


setups = pd.DataFrame(
    setup_rows
)


print(
    "Final non-range qualified signals:",
    len(setups)
)

print()

print(
    "ATR14 distribution:"
)

print(
    setups["ATR14"]
    .describe(
        percentiles=[
            0.10,
            0.20,
            0.25,
            0.40,
            0.50,
            0.60,
            0.75,
            0.80,
            0.90
        ]
    )
    .round(4)
)


# ============================================================
# BUILD SESSION-SPECIFIC TRADES
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

        session_setups = (
            setups.copy()
        )

    else:

        session_setups = setups[
            (
                setups["Hour"]
                >=
                start_hour
            )
            &
            (
                setups["Hour"]
                <
                end_hour
            )
        ].copy()


    print()

    print(
        session_name,
        "signals:",
        len(session_setups)
    )


    for _, row in session_setups.iterrows():

        signal_time = row[
            "SignalTime"
        ]


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
# OVERALL SESSION RESULTS
# ============================================================

overall_rows = []


for session_name in SESSIONS:

    d = trades[
        trades["Session"]
        ==
        session_name
    ]

    stats = calculate_stats(
        d
    )


    yearly_pfs = []

    for year in MAIN_YEARS:

        yd = d[
            d["Year"]
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


    overall_rows.append({

        "Session":
            session_name,

        **stats,

        "WorstYearPF":
            (
                min(yearly_pfs)
                if yearly_pfs
                else np.nan
            )
    })


overall_df = pd.DataFrame(
    overall_rows
)


# ============================================================
# FIXED ATR14 BUCKET ANALYSIS
# ============================================================

bucket_rows = []
bucket_year_rows = []


for session_name in SESSIONS:

    session_trades = trades[
        trades["Session"]
        ==
        session_name
    ]


    for low, high, label in ATR_BUCKETS:


        if np.isneginf(low):

            d = session_trades[
                session_trades["ATR14"]
                <
                high
            ]


        elif np.isposinf(high):

            d = session_trades[
                session_trades["ATR14"]
                >=
                low
            ]


        else:

            d = session_trades[
                (
                    session_trades["ATR14"]
                    >=
                    low
                )
                &
                (
                    session_trades["ATR14"]
                    <
                    high
                )
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


            bucket_year_rows.append({

                "Session":
                    session_name,

                "ATRBucket":
                    label,

                "Year":
                    year,

                **ys
            })


        bucket_rows.append({

            "Session":
                session_name,

            "ATRBucket":
                label,

            "ATRLow":
                low,

            "ATRHigh":
                high,

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


bucket_df = pd.DataFrame(
    bucket_rows
)

bucket_year_df = pd.DataFrame(
    bucket_year_rows
)


# ============================================================
# ATR QUINTILES
#
# Quintiles are calculated separately inside each session.
# This gives approximately equal trade counts.
# ============================================================

quintile_rows = []
quintile_year_rows = []


for session_name in SESSIONS:

    d = trades[
        trades["Session"]
        ==
        session_name
    ].copy()


    if len(d) < 5:
        continue


    d["ATRQuintile"] = pd.qcut(
        d["ATR14"],
        q=5,
        labels=[
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "Q5"
        ],
        duplicates="drop"
    )


    for q in [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5"
    ]:

        qd = d[
            d["ATRQuintile"]
            ==
            q
        ]


        if len(qd) == 0:
            continue


        stats = calculate_stats(
            qd
        )


        atr_min = qd[
            "ATR14"
        ].min()

        atr_max = qd[
            "ATR14"
        ].max()


        year_pfs = []
        profitable_years = 0
        losing_years = 0
        min_year_trades = np.inf


        for year in MAIN_YEARS:

            yd = qd[
                qd["Year"]
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


            quintile_year_rows.append({

                "Session":
                    session_name,

                "ATRQuintile":
                    q,

                "Year":
                    year,

                **ys
            })


        quintile_rows.append({

            "Session":
                session_name,

            "ATRQuintile":
                q,

            "ATRMin":
                atr_min,

            "ATRMax":
                atr_max,

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


quintile_df = pd.DataFrame(
    quintile_rows
)

quintile_year_df = pd.DataFrame(
    quintile_year_rows
)


# ============================================================
# PF MATRICES
# ============================================================

bucket_pf_matrix = (
    bucket_year_df
    .pivot_table(
        index=[
            "Session",
            "ATRBucket"
        ],
        columns="Year",
        values="ProfitFactor"
    )
)


quintile_pf_matrix = (
    quintile_year_df
    .pivot_table(
        index=[
            "Session",
            "ATRQuintile"
        ],
        columns="Year",
        values="ProfitFactor"
    )
)


# ============================================================
# SAVE
# ============================================================

setups.to_csv(
    OUTPUT_DIR
    /
    "qualified_nonrange_signals.csv",
    index=False
)

trades.to_csv(
    OUTPUT_DIR
    /
    "all_trades.csv",
    index=False
)

overall_df.to_csv(
    OUTPUT_DIR
    /
    "overall_session_summary.csv",
    index=False
)

bucket_df.to_csv(
    OUTPUT_DIR
    /
    "atr14_fixed_bucket_summary.csv",
    index=False
)

bucket_year_df.to_csv(
    OUTPUT_DIR
    /
    "atr14_fixed_bucket_year_by_year.csv",
    index=False
)

bucket_pf_matrix.to_csv(
    OUTPUT_DIR
    /
    "atr14_fixed_bucket_pf_matrix.csv"
)

quintile_df.to_csv(
    OUTPUT_DIR
    /
    "atr14_quintile_summary.csv",
    index=False
)

quintile_year_df.to_csv(
    OUTPUT_DIR
    /
    "atr14_quintile_year_by_year.csv",
    index=False
)

quintile_pf_matrix.to_csv(
    OUTPUT_DIR
    /
    "atr14_quintile_pf_matrix.csv"
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
print("OVERALL SESSION RESULTS")
print("=" * 220)

print(
    overall_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("ABSOLUTE ATR14 BUCKETS")
print("=" * 220)

print(
    bucket_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("ABSOLUTE ATR14 BUCKET × YEAR PF")
print("=" * 220)

print(
    bucket_pf_matrix
    .round(4)
    .to_string()
)


print()
print("=" * 220)
print("ATR14 QUINTILES")
print("=" * 220)

print(
    quintile_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("ATR14 QUINTILE × YEAR PF")
print("=" * 220)

print(
    quintile_pf_matrix
    .round(4)
    .to_string()
)


print()
print("=" * 220)
print("DONE")
print("=" * 220)

print()
print("Output folder:")
print(OUTPUT_DIR)