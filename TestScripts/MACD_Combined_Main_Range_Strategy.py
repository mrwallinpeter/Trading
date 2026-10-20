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
    r"\MACD_Combined_Main_Range_Strategy"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# COMMON SETTINGS
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

MAIN_HOLD_MINUTES = 210
RANGE_HOLD_MINUTES = 210

MAIN_YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026
]


# ============================================================
# MAIN STRATEGY RANGE FILTER
# ============================================================

MAIN_RANGE_LOOKBACK = 20
MAIN_MAX_RANGE_HEIGHT = 25.0
MAIN_MAX_RANGE_DRIFT = 10.0


# ============================================================
# RANGE CONTINUATION STRATEGY
# ============================================================

RECENT_RANGE_LOOKBACK = 10
RECENT_RANGE_SEARCH_BARS = 4

RECENT_MAX_RANGE_HEIGHT = 25.0
RECENT_MAX_RANGE_DRIFT = 10.0

RANGE_WAIT_BARS = 6

# least-fitted version:
# after 6 bars, close simply needs to be above RangeHigh
RANGE_BREAKOUT_ATR_MIN = 0.00


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
# 15-MINUTE DATA
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

    dd = equity - peak

    return abs(
        dd.min()
    )


def stats(d):

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
# EXECUTION FUNCTION
# ============================================================

def build_trade(
    signal_time,
    entry_after_time,
    hold_minutes,
    strategy_name
):

    entry_pos = df1.index.searchsorted(
        entry_after_time,
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
            minutes=hold_minutes
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

        "Strategy":
            strategy_name,

        "SignalTime":
            signal_time,

        "EntryTime":
            entry_time,

        "EntryPrice":
            entry_price,

        "ExitTime":
            exit_time,

        "ExitPrice":
            exit_price,

        "HoldingMinutes":
            hold_minutes,

        "PnL":
            exit_price - entry_price
    }


# ============================================================
# MAIN 20-BAR RANGE
# ============================================================

def main_range_info(pos):

    start_pos = (
        pos
        -
        MAIN_RANGE_LOOKBACK
    )

    end_pos = (
        pos
        -
        1
    )

    if start_pos < 0:
        return None


    w = df15.iloc[
        start_pos:end_pos + 1
    ]


    range_high = float(
        w["High"].max()
    )

    range_low = float(
        w["Low"].min()
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
        MAIN_MAX_RANGE_HEIGHT
        and
        range_drift
        <=
        MAIN_MAX_RANGE_DRIFT
    )


    cross_close = float(
        df15.iloc[pos]["Close"]
    )


    cross_inside = (
        valid_range
        and
        range_low
        <= cross_close
        <= range_high
    )


    return {

        "MainRangeHigh":
            range_high,

        "MainRangeLow":
            range_low,

        "MainRangeHeight":
            range_height,

        "MainRangeDrift":
            range_drift,

        "MainValidRange":
            valid_range,

        "MainCrossInside":
            cross_inside
    }


# ============================================================
# RECENT 10-BAR RANGE
# ============================================================

def recent_range_before(anchor_pos):

    start_pos = (
        anchor_pos
        -
        RECENT_RANGE_LOOKBACK
    )

    end_pos = (
        anchor_pos
        -
        1
    )

    if start_pos < 0:
        return None


    w = df15.iloc[
        start_pos:end_pos + 1
    ]


    range_high = float(
        w["High"].max()
    )

    range_low = float(
        w["Low"].min()
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


    valid = (
        range_height
        <=
        RECENT_MAX_RANGE_HEIGHT
        and
        range_drift
        <=
        RECENT_MAX_RANGE_DRIFT
    )


    return {

        "RecentRangeStart":
            df15.index[start_pos],

        "RecentRangeEnd":
            df15.index[end_pos],

        "RecentRangeHigh":
            range_high,

        "RecentRangeLow":
            range_low,

        "RecentRangeHeight":
            range_height,

        "RecentRangeDrift":
            range_drift,

        "RecentRangeValid":
            valid
    }


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

        info = recent_range_before(
            anchor_pos
        )

        if (
            info is not None
            and
            info["RecentRangeValid"]
        ):

            return {

                **info,

                "RecentRangeBarsBack":
                    bars_back
            }

    return None


# ============================================================
# BUILD BASE MACD SIGNAL POPULATION
# ============================================================

print()
print("=" * 180)
print("BUILDING BASE MACD SIGNALS")
print("=" * 180)


signal_rows = []


for signal_time in df15.index[
    df15["BullCross"]
]:

    pos = df15.index.get_loc(
        signal_time
    )


    if pos < max(
        PRIOR_LOOKBACK,
        MAIN_RANGE_LOOKBACK,
        RECENT_RANGE_LOOKBACK
        +
        RECENT_RANGE_SEARCH_BARS
    ):
        continue


    # 14-18 session
    if not (
        SESSION_START
        <= signal_time.hour
        <
        SESSION_END
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
        df15.iloc[
            pos - 1
        ]["Close"]
    )


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


    main_info = main_range_info(
        pos
    )

    recent_info = find_recent_range(
        pos
    )


    signal_rows.append({

        "SignalTime":
            signal_time,

        "SignalPos":
            pos,

        "Year":
            signal_time.year,

        "ATR14":
            atr14,

        "CrossClose":
            close_now,

        "PriceMoveATR14":
            price_move_atr,

        "PriorMoveATR14":
            prior_move_atr,

        **main_info,

        "HasRecentRange":
            recent_info is not None,

        **(
            recent_info
            if recent_info is not None
            else {}
        )
    })


signals = pd.DataFrame(
    signal_rows
)

print(
    "Base qualified 14-18 MACD signals:",
    len(signals)
)


# ============================================================
# STRATEGY A:
# MAIN NON-RANGE
# ============================================================

main_trades = []


for _, row in signals.iterrows():

    # Reject only if cross is inside valid
    # original 20-bar range.

    if row["MainCrossInside"]:
        continue


    trade = build_trade(
        signal_time=
            row["SignalTime"],

        entry_after_time=
            row["SignalTime"],

        hold_minutes=
            MAIN_HOLD_MINUTES,

        strategy_name=
            "MAIN_NON_RANGE"
    )


    if trade is None:
        continue


    trade.update({

        "Year":
            row["Year"],

        "ATR14":
            row["ATR14"],

        "PriceMoveATR14":
            row["PriceMoveATR14"],

        "PriorMoveATR14":
            row["PriorMoveATR14"]
    })


    main_trades.append(
        trade
    )


main_trades = pd.DataFrame(
    main_trades
)


# ============================================================
# STRATEGY B:
# RECENT-RANGE CONTINUATION
#
# Wait 6 completed bars after cross.
# Require check close >= frozen recent RangeHigh.
# ============================================================

range_trades = []


for _, row in signals.iterrows():

    if not row["HasRecentRange"]:
        continue


    pos = int(
        row["SignalPos"]
    )

    check_pos = (
        pos
        +
        RANGE_WAIT_BARS
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


    range_high = float(
        row["RecentRangeHigh"]
    )


    breakout_points = (
        check_close
        -
        range_high
    )


    breakout_atr = (
        breakout_points
        /
        float(row["ATR14"])
    )


    if (
        breakout_atr
        <
        RANGE_BREAKOUT_ATR_MIN
    ):
        continue


    trade = build_trade(
        signal_time=
            row["SignalTime"],

        entry_after_time=
            check_time,

        hold_minutes=
            RANGE_HOLD_MINUTES,

        strategy_name=
            "RECENT_RANGE_CONTINUATION"
    )


    if trade is None:
        continue


    trade.update({

        "Year":
            row["Year"],

        "ATR14":
            row["ATR14"],

        "PriceMoveATR14":
            row["PriceMoveATR14"],

        "PriorMoveATR14":
            row["PriorMoveATR14"],

        "RecentRangeHigh":
            range_high,

        "RecentRangeLow":
            row["RecentRangeLow"],

        "RecentRangeBarsBack":
            row["RecentRangeBarsBack"],

        "CheckTime":
            check_time,

        "CheckClose":
            check_close,

        "BreakoutPoints":
            breakout_points,

        "BreakoutATR":
            breakout_atr,

        "MainCrossInside":
            row["MainCrossInside"]
    })


    range_trades.append(
        trade
    )


range_trades = pd.DataFrame(
    range_trades
)


# ============================================================
# COMBINE BOTH STRATEGIES
# ============================================================

combined = pd.concat(
    [
        main_trades,
        range_trades
    ],
    ignore_index=True
)


combined = combined.sort_values(
    [
        "EntryTime",
        "Strategy"
    ]
).reset_index(
    drop=True
)


# ============================================================
# OVERLAP ANALYSIS
#
# Same MACD SignalTime used by both branches.
# ============================================================

main_signal_times = set(
    main_trades["SignalTime"]
)

range_signal_times = set(
    range_trades["SignalTime"]
)

overlap_signal_times = (
    main_signal_times
    &
    range_signal_times
)

range_only_signal_times = (
    range_signal_times
    -
    main_signal_times
)

main_only_signal_times = (
    main_signal_times
    -
    range_signal_times
)


# ============================================================
# OPTIONAL COMPLEMENTARY VERSION
#
# This represents:
#
# MAIN strategy +
# RANGE strategy only when MAIN did NOT already take signal.
#
# This avoids two trades arising from same MACD signal.
# ============================================================

range_only_trades = range_trades[
    range_trades["SignalTime"]
    .isin(
        range_only_signal_times
    )
].copy()


complementary_combined = pd.concat(
    [
        main_trades,
        range_only_trades
    ],
    ignore_index=True
)


complementary_combined = (
    complementary_combined
    .sort_values(
        [
            "EntryTime",
            "Strategy"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


datasets = {

    "MAIN_NON_RANGE":
        main_trades,

    "RANGE_ALL":
        range_trades,

    "RANGE_ONLY_NEW_SIGNALS":
        range_only_trades,

    "COMBINED_STACKED":
        combined,

    "COMBINED_COMPLEMENTARY":
        complementary_combined
}


for name, d in datasets.items():

    s = stats(
        d
    )

    year_pfs = []
    profitable_years = 0
    losing_years = 0


    for year in MAIN_YEARS:

        yd = d[
            d["Year"]
            ==
            year
        ]

        pf = stats(
            yd
        )["ProfitFactor"]


        if np.isfinite(pf):

            year_pfs.append(
                pf
            )

            if pf > 1:
                profitable_years += 1

            elif pf < 1:
                losing_years += 1


    summary_rows.append({

        "System":
            name,

        **s,

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
            losing_years
    })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# YEAR-BY-YEAR
# ============================================================

year_rows = []


for name, d in datasets.items():

    for year in MAIN_YEARS:

        yd = d[
            d["Year"]
            ==
            year
        ]

        year_rows.append({

            "System":
                name,

            "Year":
                year,

            **stats(yd)
        })


year_df = pd.DataFrame(
    year_rows
)


pf_matrix = (
    year_df
    .pivot(
        index="System",
        columns="Year",
        values="ProfitFactor"
    )
)


# ============================================================
# OVERLAP REPORT
# ============================================================

overlap_df = pd.DataFrame({

    "Metric": [
        "Main signals",
        "Range signals",
        "Signals qualifying for BOTH",
        "Main-only signals",
        "Range-only new signals"
    ],

    "Count": [
        len(main_signal_times),
        len(range_signal_times),
        len(overlap_signal_times),
        len(main_only_signal_times),
        len(range_only_signal_times)
    ]
})


# ============================================================
# SAVE
# ============================================================

signals.to_csv(
    OUTPUT_DIR
    /
    "base_signals.csv",
    index=False
)

main_trades.to_csv(
    OUTPUT_DIR
    /
    "main_non_range_trades.csv",
    index=False
)

range_trades.to_csv(
    OUTPUT_DIR
    /
    "recent_range_continuation_trades.csv",
    index=False
)

range_only_trades.to_csv(
    OUTPUT_DIR
    /
    "range_only_new_signals.csv",
    index=False
)

combined.to_csv(
    OUTPUT_DIR
    /
    "combined_stacked_trades.csv",
    index=False
)

complementary_combined.to_csv(
    OUTPUT_DIR
    /
    "combined_complementary_trades.csv",
    index=False
)

summary_df.to_csv(
    OUTPUT_DIR
    /
    "combined_summary.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "combined_year_by_year.csv",
    index=False
)

pf_matrix.to_csv(
    OUTPUT_DIR
    /
    "combined_pf_matrix.csv"
)

overlap_df.to_csv(
    OUTPUT_DIR
    /
    "overlap_report.csv",
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
print("OVERLAP BETWEEN STRATEGIES")
print("=" * 220)

print(
    overlap_df.to_string(
        index=False
    )
)


print()
print("=" * 220)
print("OVERALL STATISTICS")
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
print("YEAR-BY-YEAR PROFIT FACTOR")
print("=" * 220)

print(
    pf_matrix
    .round(4)
    .to_string()
)


print()
print("=" * 220)
print("YEAR-BY-YEAR FULL STATISTICS")
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