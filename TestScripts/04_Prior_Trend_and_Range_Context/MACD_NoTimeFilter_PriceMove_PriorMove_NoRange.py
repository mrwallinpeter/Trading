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
    r"\MACD_NoTimeFilter_PriceMove_PriorMove_NoRange"
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
# LOAD 1-MIN DATA
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
# BUILD 240-MIN LONG TRADE
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
#
# IMPORTANT:
# NO TIME FILTER
# ============================================================

signals = df15[
    df15["BullCross"]
].copy()


print()
print("=" * 170)
print("BUILDING DATASET — NO TIME FILTER")
print("=" * 170)

print(
    "Raw bullish MACD crosses:",
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


    # ========================================================
    # CROSS-BAR PRICE MOVE
    # ========================================================

    price_move_15_atr = (
        close_now
        -
        close_prev
    ) / atr14


    # ========================================================
    # PRIOR 20-BAR DIRECTION
    #
    # Close[t-1] - Close[t-20]
    # normalized by current ATR14.
    # ========================================================

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

        "SignalHour":
            signal_time.hour,

        "SignalMinute":
            signal_time.minute,

        "Year":
            signal_time.year,

        "ATR14":
            atr14,

        "PriceMove15ATR":
            price_move_15_atr,

        "PriorMovePoints":
            prior_move_points,

        "PriorMoveATR":
            prior_move_atr,

        **range_info,

        **trade
    })


df = pd.DataFrame(
    rows
)


print(
    "Completed trades:",
    len(df)
)


# ============================================================
# FILTER PROGRESSION
# ============================================================

raw = df.copy()


price_move = raw[
    (
        raw["PriceMove15ATR"]
        >=
        PRICE_MOVE_MIN
    )
    &
    (
        raw["PriceMove15ATR"]
        <=
        PRICE_MOVE_MAX
    )
].copy()


price_move_no_range = price_move[
    ~price_move[
        "CrossInsideRange"
    ]
].copy()


final_setup = price_move_no_range[
    (
        price_move_no_range["PriorMoveATR"]
        >=
        PRIOR_MOVE_MIN
    )
    &
    (
        price_move_no_range["PriorMoveATR"]
        <=
        PRIOR_MOVE_MAX
    )
].copy()


print()
print("=" * 170)
print("COUNTS")
print("=" * 170)

print(
    "RAW:",
    len(raw)
)

print(
    "PriceMove 0.35-0.90:",
    len(price_move)
)

print(
    "PriceMove + NO RANGE:",
    len(price_move_no_range)
)

print(
    "PriceMove + NO RANGE + PriorMove -2 to +2:",
    len(final_setup)
)


# ============================================================
# SUMMARY
# ============================================================

groups = {

    "RAW_ALL_TIMES":
        raw,

    "PRICEMOVE_ALL_TIMES":
        price_move,

    "PRICEMOVE_NO_RANGE_ALL_TIMES":
        price_move_no_range,

    "FINAL_PM_RANGE_PRIOR_ALL_TIMES":
        final_setup
}


summary_rows = []


for name, subset in groups.items():

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

            if ys["ProfitFactor"] > 1:
                profitable_years += 1

        min_year_trades = min(
            min_year_trades,
            ys["N"]
        )


    summary_rows.append({

        "Strategy":
            name,

        **stats,

        "RetentionVsRawPct":
            (
                len(subset)
                /
                len(raw)
                *
                100
                if len(raw) > 0
                else np.nan
            ),

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


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# YEAR BY YEAR
# ============================================================

year_rows = []


for name, subset in groups.items():

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

            "Strategy":
                name,

            "Year":
                year,

            **ys
        })


year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# HOURLY BREAKDOWN
#
# Final setup only.
# SignalHour = hour of completed 15m MACD-cross bar.
# ============================================================

hour_rows = []


for hour in range(24):

    subset = final_setup[
        final_setup["SignalHour"]
        ==
        hour
    ]

    stats = calculate_stats(
        subset
    )

    year_pfs = []
    profitable_years = 0


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

            if ys["ProfitFactor"] > 1:
                profitable_years += 1


    hour_rows.append({

        "Hour":
            hour,

        **stats,

        "WorstYearPF":
            (
                min(year_pfs)
                if year_pfs
                else np.nan
            ),

        "ProfitableYears":
            profitable_years
    })


hour_df = pd.DataFrame(
    hour_rows
)


# ============================================================
# 2-HOUR BLOCK BREAKDOWN
#
# Gives larger samples than individual hours.
# ============================================================

two_hour_rows = []


for start_hour in range(
    0,
    24,
    2
):

    end_hour = (
        start_hour
        +
        2
    )

    subset = final_setup[
        (
            final_setup["SignalHour"]
            >=
            start_hour
        )
        &
        (
            final_setup["SignalHour"]
            <
            end_hour
        )
    ]

    stats = calculate_stats(
        subset
    )

    year_pfs = []
    profitable_years = 0


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

            if ys["ProfitFactor"] > 1:
                profitable_years += 1


    two_hour_rows.append({

        "TimeBlock":
            f"{start_hour:02d}:00-{end_hour:02d}:00",

        **stats,

        "WorstYearPF":
            (
                min(year_pfs)
                if year_pfs
                else np.nan
            ),

        "ProfitableYears":
            profitable_years
    })


two_hour_df = pd.DataFrame(
    two_hour_rows
)


# ============================================================
# QUARTER-HOUR BREAKDOWN
#
# Useful because MACD signals occur on 00,15,30,45.
# ============================================================

minute_rows = []


for minute in [
    0,
    15,
    30,
    45
]:

    subset = final_setup[
        final_setup["SignalMinute"]
        ==
        minute
    ]

    stats = calculate_stats(
        subset
    )

    minute_rows.append({

        "SignalMinute":
            minute,

        **stats
    })


minute_df = pd.DataFrame(
    minute_rows
)


# ============================================================
# SAVE
# ============================================================

raw.to_csv(
    OUTPUT_DIR
    /
    "raw_all_times.csv",
    index=False
)

price_move.to_csv(
    OUTPUT_DIR
    /
    "pricemove_all_times.csv",
    index=False
)

price_move_no_range.to_csv(
    OUTPUT_DIR
    /
    "pricemove_no_range_all_times.csv",
    index=False
)

final_setup.to_csv(
    OUTPUT_DIR
    /
    "final_pricemove_prior_no_range_all_times.csv",
    index=False
)

summary_df.to_csv(
    OUTPUT_DIR
    /
    "summary.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "year_by_year.csv",
    index=False
)

hour_df.to_csv(
    OUTPUT_DIR
    /
    "hourly_breakdown.csv",
    index=False
)

two_hour_df.to_csv(
    OUTPUT_DIR
    /
    "two_hour_breakdown.csv",
    index=False
)

minute_df.to_csv(
    OUTPUT_DIR
    /
    "quarter_hour_breakdown.csv",
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
print("SUMMARY — NO TIME FILTER")
print("=" * 210)

print(
    summary_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 210)
print("YEAR BY YEAR")
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
print("FINAL SETUP — HOURLY BREAKDOWN")
print("=" * 210)

print(
    hour_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 210)
print("FINAL SETUP — 2-HOUR BLOCKS")
print("=" * 210)

print(
    two_hour_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 210)
print("FINAL SETUP — SIGNAL MINUTE")
print("=" * 210)

print(
    minute_df
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