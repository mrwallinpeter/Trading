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
    r"\MACD_Range_Breakout_Strategy_Test"
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

PRIOR_LOOKBACK = 20

RANGE_LOOKBACK = 20
MAX_RANGE_HEIGHT = 25.0
MAX_RANGE_DRIFT = 10.0

HOLD_MINUTES = 210

BREAKOUT_MAX_BARS = [
    1,
    2,
    3,
    4
]

MAIN_YEARS = [
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
# BUILD 15-MINUTE DATA
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
# STATISTICS
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
        "N": len(pnl),
        "WinRatePct": np.mean(pnl > 0) * 100,
        "NetPoints": pnl.sum(),
        "AveragePnL": pnl.mean(),
        "MedianPnL": np.median(pnl),
        "ProfitFactor": profit_factor(pnl),
        "SQN": sqn(pnl),
        "MaxDrawdown": max_drawdown(pnl)
    }


# ============================================================
# RANGE
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
# TRADE FROM ACTUAL BREAKOUT ENTRY
# ============================================================

def build_trade(entry_signal_time):

    entry_pos = df1.index.searchsorted(
        entry_signal_time,
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
        return None

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
        return None

    exit_price = float(
        df1.iloc[
            exit_pos
        ]["Close"]
    )

    return {
        "EntryTime": entry_time,
        "EntryPrice": entry_price,
        "ExitTime": exit_time,
        "ExitPrice": exit_price,
        "PnL": exit_price - entry_price
    }


# ============================================================
# FIND RANGE SETUPS
# ============================================================

print()
print("=" * 180)
print("BUILDING RANGE SETUPS")
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


    price_move = (
        close_now
        -
        close_prev
    ) / atr14


    prior_start_close = float(
        df15.iloc[
            pos - PRIOR_LOOKBACK
        ]["Close"]
    )

    prior_move = (
        close_prev
        -
        prior_start_close
    ) / atr14


    # --------------------------------------------
    # TWO MAIN FILTERS
    # --------------------------------------------

    if not (
        PRICE_MOVE_MIN
        <= price_move
        <= PRICE_MOVE_MAX
    ):
        continue


    if not (
        PRIOR_MOVE_MIN
        <= prior_move
        <= PRIOR_MOVE_MAX
    ):
        continue


    # --------------------------------------------
    # REQUIRE RANGE
    # --------------------------------------------

    range_info = get_range_info(
        pos
    )

    if not range_info[
        "CrossInsideRange"
    ]:
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

        "PriceMoveATR14":
            price_move,

        "PriorMoveATR14":
            prior_move,

        **range_info
    })


setups = pd.DataFrame(
    setup_rows
)


print(
    "Qualified MACD crosses inside range:",
    len(setups)
)


# ============================================================
# BASELINE:
# IMMEDIATE ENTRY ON ORIGINAL CROSS
# ============================================================

baseline_rows = []


for _, row in setups.iterrows():

    trade = build_trade(
        row["SignalTime"]
    )

    if trade is None:
        continue

    baseline_rows.append({

        **row.to_dict(),

        "Strategy":
            "IMMEDIATE",

        "BreakoutBarsAllowed":
            0,

        "BreakoutBar":
            0,

        **trade
    })


# ============================================================
# BREAKOUT STRATEGIES
#
# RangeHigh is frozen at original range.
#
# Breakout confirmation:
# subsequent 15m CLOSE > RangeHigh.
# ============================================================

breakout_rows = []


for max_bars in BREAKOUT_MAX_BARS:

    for _, setup in setups.iterrows():

        signal_pos = int(
            setup["SignalPos"]
        )

        range_high = float(
            setup["RangeHigh"]
        )

        breakout_found = False


        for bars_after in range(
            1,
            max_bars + 1
        ):

            check_pos = (
                signal_pos
                +
                bars_after
            )

            if check_pos >= len(df15):
                break


            check_bar = df15.iloc[
                check_pos
            ]

            check_time = df15.index[
                check_pos
            ]


            # ====================================
            # CONFIRMED CLOSE ABOVE RANGE HIGH
            # ====================================

            if (
                check_bar["Close"]
                >
                range_high
            ):

                trade = build_trade(
                    check_time
                )

                if trade is not None:

                    breakout_rows.append({

                        **setup.to_dict(),

                        "Strategy":
                            f"BREAKOUT_{max_bars}BAR",

                        "BreakoutBarsAllowed":
                            max_bars,

                        "BreakoutBar":
                            bars_after,

                        "BreakoutSignalTime":
                            check_time,

                        "BreakoutClose":
                            float(
                                check_bar["Close"]
                            ),

                        "BreakoutDistance":
                            float(
                                check_bar["Close"]
                                -
                                range_high
                            ),

                        **trade
                    })

                breakout_found = True
                break


# ============================================================
# COMBINE
# ============================================================

baseline_df = pd.DataFrame(
    baseline_rows
)

breakout_df = pd.DataFrame(
    breakout_rows
)

all_results = pd.concat(
    [
        baseline_df,
        breakout_df
    ],
    ignore_index=True
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


strategy_names = [
    "IMMEDIATE",
    "BREAKOUT_1BAR",
    "BREAKOUT_2BAR",
    "BREAKOUT_3BAR",
    "BREAKOUT_4BAR"
]


for strategy in strategy_names:

    d = all_results[
        all_results["Strategy"]
        ==
        strategy
    ]

    stats = calculate_stats(
        d
    )


    yearly_pfs = []
    profitable_years = 0
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

            yearly_pfs.append(
                ys["ProfitFactor"]
            )

            if (
                ys["ProfitFactor"]
                >
                1
            ):
                profitable_years += 1


    summary_rows.append({

        "Strategy":
            strategy,

        **stats,

        "SetupCount":
            len(setups),

        "EntryRatePct":
            (
                len(d)
                /
                len(setups)
                *
                100
                if len(setups) > 0
                else np.nan
            ),

        "WorstYearPF":
            (
                min(yearly_pfs)
                if yearly_pfs
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
# YEAR-BY-YEAR
# ============================================================

year_rows = []


for strategy in strategy_names:

    d = all_results[
        all_results["Strategy"]
        ==
        strategy
    ]


    for year in MAIN_YEARS:

        yd = d[
            d["Year"]
            ==
            year
        ]

        stats = calculate_stats(
            yd
        )

        year_rows.append({

            "Strategy":
                strategy,

            "Year":
                year,

            **stats
        })


year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# BREAKOUT BAR DISTRIBUTION
# ============================================================

distribution_rows = []


for max_bars in BREAKOUT_MAX_BARS:

    strategy = (
        f"BREAKOUT_{max_bars}BAR"
    )

    d = all_results[
        all_results["Strategy"]
        ==
        strategy
    ]


    for breakout_bar in range(
        1,
        max_bars + 1
    ):

        bd = d[
            d["BreakoutBar"]
            ==
            breakout_bar
        ]

        stats = calculate_stats(
            bd
        )

        distribution_rows.append({

            "Strategy":
                strategy,

            "BreakoutBar":
                breakout_bar,

            **stats
        })


distribution_df = pd.DataFrame(
    distribution_rows
)


# ============================================================
# SAVE
# ============================================================

setups.to_csv(
    OUTPUT_DIR
    /
    "range_setups.csv",
    index=False
)

all_results.to_csv(
    OUTPUT_DIR
    /
    "all_range_strategy_trades.csv",
    index=False
)

summary_df.to_csv(
    OUTPUT_DIR
    /
    "range_strategy_summary.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "range_strategy_year_by_year.csv",
    index=False
)

distribution_df.to_csv(
    OUTPUT_DIR
    /
    "breakout_bar_distribution.csv",
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
    500
)

pd.set_option(
    "display.max_rows",
    1000
)


print()
print("=" * 220)
print("RANGE STRATEGY SUMMARY")
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
print("BREAKOUT BAR DISTRIBUTION")
print("=" * 220)

print(
    distribution_df
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