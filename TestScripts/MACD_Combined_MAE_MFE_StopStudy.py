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
    r"\MACD_Combined_MAE_MFE_StopStudy"
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
# MAIN NON-RANGE SETTINGS
# ============================================================

MAIN_RANGE_LOOKBACK = 20
MAIN_MAX_RANGE_HEIGHT = 25.0
MAIN_MAX_RANGE_DRIFT = 10.0


# ============================================================
# RECENT RANGE CONTINUATION SETTINGS
# ============================================================

RECENT_RANGE_LOOKBACK = 10
RECENT_RANGE_SEARCH_BARS = 4

RECENT_MAX_RANGE_HEIGHT = 25.0
RECENT_MAX_RANGE_DRIFT = 10.0

RANGE_WAIT_BARS = 6
RANGE_BREAKOUT_ATR_MIN = 0.00


# ============================================================
# STOP TESTS
# ============================================================

STOP_ATR_LEVELS = [
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
    2.50,
    3.00
]

PERCENTILES = [
    0.10,
    0.20,
    0.25,
    0.50,
    0.75,
    0.80,
    0.90,
    0.95,
    0.975,
    0.99
]


# ============================================================
# LOAD 1-MIN DATA
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

    drawdown = equity - peak

    return abs(
        drawdown.min()
    )


def calculate_stats(d, pnl_col="PnL"):

    pnl = (
        d[pnl_col]
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
# MAIN 20-BAR RANGE
# ============================================================

def main_range_info(pos):

    start_pos = pos - MAIN_RANGE_LOOKBACK
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
        <=
        cross_close
        <=
        range_high
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
# EXECUTION
# ============================================================

def build_trade(
    signal_time,
    entry_after_time,
    hold_minutes,
    strategy_name,
    atr14
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

        "ATR14":
            atr14,

        "PnL":
            exit_price
            -
            entry_price
    }


# ============================================================
# BUILD BASE QUALIFIED SIGNALS
# ============================================================

print()
print("=" * 180)
print("BUILDING BASE 14-18 SIGNALS")
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

    if not (
        SESSION_START
        <=
        signal_time.hour
        <
        SESSION_END
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

        "PriceMoveATR14":
            price_move_atr,

        "PriorMoveATR14":
            prior_move_atr,

        **main_info,

        "HasRecentRange":
            recent_info
            is not None,

        **(
            recent_info
            if recent_info
            is not None
            else {}
        )
    })


signals = pd.DataFrame(
    signal_rows
)

print(
    "Base qualified signals:",
    len(signals)
)


# ============================================================
# MAIN TRADES
# ============================================================

main_rows = []


for _, row in signals.iterrows():

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
            "MAIN_NON_RANGE",

        atr14=
            row["ATR14"]
    )

    if trade is None:
        continue

    trade.update({

        "Year":
            row["Year"],

        "PriceMoveATR14":
            row["PriceMoveATR14"],

        "PriorMoveATR14":
            row["PriorMoveATR14"]
    })

    main_rows.append(
        trade
    )


main_trades = pd.DataFrame(
    main_rows
)


# ============================================================
# RANGE CONTINUATION TRADES
# ============================================================

range_rows = []


for _, row in signals.iterrows():

    if not row["HasRecentRange"]:
        continue

    signal_pos = int(
        row["SignalPos"]
    )

    check_pos = (
        signal_pos
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
        row["ATR14"]
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
            "RECENT_RANGE_CONTINUATION",

        atr14=
            row["ATR14"]
    )

    if trade is None:
        continue

    trade.update({

        "Year":
            row["Year"],

        "PriceMoveATR14":
            row["PriceMoveATR14"],

        "PriorMoveATR14":
            row["PriorMoveATR14"],

        "BreakoutATR":
            breakout_atr,

        "MainCrossInside":
            row["MainCrossInside"]
    })

    range_rows.append(
        trade
    )


range_trades = pd.DataFrame(
    range_rows
)


# ============================================================
# COMPLEMENTARY VERSION
# ============================================================

main_signal_times = set(
    main_trades["SignalTime"]
)

range_signal_times = set(
    range_trades["SignalTime"]
)

range_only_signal_times = (
    range_signal_times
    -
    main_signal_times
)

range_only = range_trades[
    range_trades["SignalTime"]
    .isin(
        range_only_signal_times
    )
].copy()


combined_stacked = pd.concat(
    [
        main_trades,
        range_trades
    ],
    ignore_index=True
).sort_values(
    "EntryTime"
).reset_index(
    drop=True
)


combined_complementary = pd.concat(
    [
        main_trades,
        range_only
    ],
    ignore_index=True
).sort_values(
    "EntryTime"
).reset_index(
    drop=True
)


# ============================================================
# MAE / MFE
#
# Long trades:
#
# MAE = Entry - lowest low
# MFE = highest high - Entry
#
# Uses all 1-minute bars from entry through timed exit.
# ============================================================

def add_mae_mfe(d):

    rows = []

    for _, trade in d.iterrows():

        entry_time = trade[
            "EntryTime"
        ]

        exit_time = trade[
            "ExitTime"
        ]

        entry_price = float(
            trade["EntryPrice"]
        )

        path = df1.loc[
            entry_time:exit_time
        ]

        if len(path) == 0:
            continue

        lowest_low = float(
            path["Low"].min()
        )

        highest_high = float(
            path["High"].max()
        )

        mae_points = max(
            0.0,
            entry_price
            -
            lowest_low
        )

        mfe_points = max(
            0.0,
            highest_high
            -
            entry_price
        )

        atr14 = float(
            trade["ATR14"]
        )

        mae_atr = (
            mae_points
            /
            atr14
        )

        mfe_atr = (
            mfe_points
            /
            atr14
        )

        rows.append({

            **trade.to_dict(),

            "LowestLow":
                lowest_low,

            "HighestHigh":
                highest_high,

            "MAE_Points":
                mae_points,

            "MFE_Points":
                mfe_points,

            "MAE_ATR":
                mae_atr,

            "MFE_ATR":
                mfe_atr,

            "OriginalWinner":
                trade["PnL"] > 0
        })

    return pd.DataFrame(
        rows
    )


print()
print("=" * 180)
print("CALCULATING 1-MIN MAE / MFE")
print("=" * 180)


datasets = {

    "MAIN_NON_RANGE":
        add_mae_mfe(
            main_trades
        ),

    "RANGE_ALL":
        add_mae_mfe(
            range_trades
        ),

    "RANGE_ONLY_NEW":
        add_mae_mfe(
            range_only
        ),

    "COMBINED_STACKED":
        add_mae_mfe(
            combined_stacked
        ),

    "COMBINED_COMPLEMENTARY":
        add_mae_mfe(
            combined_complementary
        )
}


# ============================================================
# DISTRIBUTION PROFILE
# ============================================================

distribution_rows = []


for system_name, d in datasets.items():

    for outcome_name, mask in [

        (
            "ALL",
            np.ones(
                len(d),
                dtype=bool
            )
        ),

        (
            "WINNERS",
            d["PnL"] > 0
        ),

        (
            "LOSERS",
            d["PnL"] <= 0
        )
    ]:

        x = d[
            mask
        ]

        for metric in [
            "MAE_Points",
            "MAE_ATR",
            "MFE_Points",
            "MFE_ATR"
        ]:

            values = (
                x[metric]
                .dropna()
            )

            if len(values) == 0:
                continue

            row = {

                "System":
                    system_name,

                "Outcome":
                    outcome_name,

                "Metric":
                    metric,

                "N":
                    len(values),

                "Mean":
                    values.mean(),

                "Median":
                    values.median()
            }

            for p in PERCENTILES:

                row[
                    f"P{p*100:g}"
                ] = values.quantile(
                    p
                )

            distribution_rows.append(
                row
            )


distribution_df = pd.DataFrame(
    distribution_rows
)


# ============================================================
# YEAR-BY-YEAR WINNER MAE
# ============================================================

winner_year_rows = []


for system_name, d in datasets.items():

    winners = d[
        d["PnL"] > 0
    ]

    for year in MAIN_YEARS:

        yd = winners[
            winners["Year"]
            ==
            year
        ]

        if len(yd) == 0:
            continue

        winner_year_rows.append({

            "System":
                system_name,

            "Year":
                year,

            "N_Winners":
                len(yd),

            "MAE_ATR_Median":
                yd["MAE_ATR"]
                .median(),

            "MAE_ATR_P75":
                yd["MAE_ATR"]
                .quantile(0.75),

            "MAE_ATR_P80":
                yd["MAE_ATR"]
                .quantile(0.80),

            "MAE_ATR_P90":
                yd["MAE_ATR"]
                .quantile(0.90),

            "MAE_ATR_P95":
                yd["MAE_ATR"]
                .quantile(0.95),

            "MAE_Points_P90":
                yd["MAE_Points"]
                .quantile(0.90),

            "MAE_Points_P95":
                yd["MAE_Points"]
                .quantile(0.95)
        })


winner_year_df = pd.DataFrame(
    winner_year_rows
)


# ============================================================
# STOP SIMULATION
#
# Conservative assumption:
# if 1-min Low <= stop, stop is filled exactly at stop price.
#
# No slippage/costs.
# ============================================================

def simulate_stop_on_trade(
    trade,
    stop_atr
):

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

    stop_distance = (
        stop_atr
        *
        atr14
    )

    stop_price = (
        entry_price
        -
        stop_distance
    )

    path = df1.loc[
        entry_time:exit_time
    ]

    hit = path[
        path["Low"]
        <=
        stop_price
    ]

    if len(hit) > 0:

        stop_time = hit.index[0]

        return {

            "StopHit":
                True,

            "StopExitTime":
                stop_time,

            "StopPnL":
                -stop_distance
        }

    return {

        "StopHit":
            False,

        "StopExitTime":
            trade["ExitTime"],

        "StopPnL":
            trade["PnL"]
    }


stop_summary_rows = []
stop_year_rows = []


for system_name, d in datasets.items():

    original_winners = int(
        (d["PnL"] > 0).sum()
    )

    for stop_atr in STOP_ATR_LEVELS:

        sim_rows = []

        for _, trade in d.iterrows():

            result = simulate_stop_on_trade(
                trade,
                stop_atr
            )

            sim_rows.append({

                **trade.to_dict(),
                **result
            })


        sim = pd.DataFrame(
            sim_rows
        )


        stopped_original_winners = int(
            (
                (sim["PnL"] > 0)
                &
                sim["StopHit"]
            ).sum()
        )


        winner_retention_pct = (

            100.0
            *
            (
                original_winners
                -
                stopped_original_winners
            )
            /
            original_winners

            if original_winners > 0

            else np.nan
        )


        original_losers = int(
            (sim["PnL"] <= 0)
            .sum()
        )

        stopped_losers = int(
            (
                (sim["PnL"] <= 0)
                &
                sim["StopHit"]
            ).sum()
        )


        loser_stop_pct = (

            100.0
            *
            stopped_losers
            /
            original_losers

            if original_losers > 0

            else np.nan
        )


        s = calculate_stats(
            sim,
            pnl_col="StopPnL"
        )


        year_pfs = []

        profitable_years = 0
        losing_years = 0


        for year in MAIN_YEARS:

            yd = sim[
                sim["Year"]
                ==
                year
            ]

            ys = calculate_stats(
                yd,
                pnl_col="StopPnL"
            )


            pf = ys[
                "ProfitFactor"
            ]

            if np.isfinite(pf):

                year_pfs.append(
                    pf
                )

                if pf > 1:
                    profitable_years += 1

                elif pf < 1:
                    losing_years += 1


            stop_year_rows.append({

                "System":
                    system_name,

                "StopATR":
                    stop_atr,

                "Year":
                    year,

                **ys
            })


        stop_summary_rows.append({

            "System":
                system_name,

            "StopATR":
                stop_atr,

            "StopDistanceMeanPoints":
                (
                    stop_atr
                    *
                    d["ATR14"].mean()
                ),

            **s,

            "OriginalWinners":
                original_winners,

            "StoppedOriginalWinners":
                stopped_original_winners,

            "WinnerRetentionPct":
                winner_retention_pct,

            "OriginalLosers":
                original_losers,

            "StoppedLosers":
                stopped_losers,

            "LosersStoppedPct":
                loser_stop_pct,

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


stop_summary_df = pd.DataFrame(
    stop_summary_rows
)

stop_year_df = pd.DataFrame(
    stop_year_rows
)


# ============================================================
# BASELINE WITHOUT STOP
# ============================================================

baseline_rows = []


for system_name, d in datasets.items():

    s = calculate_stats(
        d
    )

    year_pfs = []


    for year in MAIN_YEARS:

        yd = d[
            d["Year"]
            ==
            year
        ]

        pf = calculate_stats(
            yd
        )["ProfitFactor"]

        if np.isfinite(pf):
            year_pfs.append(
                pf
            )


    baseline_rows.append({

        "System":
            system_name,

        **s,

        "WorstYearPF":
            (
                min(year_pfs)
                if year_pfs
                else np.nan
            )
    })


baseline_df = pd.DataFrame(
    baseline_rows
)


# ============================================================
# WINNER MAE SUMMARY
# ============================================================

winner_mae_rows = []


for system_name, d in datasets.items():

    winners = d[
        d["PnL"] > 0
    ]

    losers = d[
        d["PnL"] <= 0
    ]


    winner_mae_rows.append({

        "System":
            system_name,

        "N":
            len(d),

        "Winners":
            len(winners),

        "Losers":
            len(losers),

        "Winner_MAE_ATR_Median":
            winners["MAE_ATR"]
            .median(),

        "Winner_MAE_ATR_P75":
            winners["MAE_ATR"]
            .quantile(0.75),

        "Winner_MAE_ATR_P80":
            winners["MAE_ATR"]
            .quantile(0.80),

        "Winner_MAE_ATR_P90":
            winners["MAE_ATR"]
            .quantile(0.90),

        "Winner_MAE_ATR_P95":
            winners["MAE_ATR"]
            .quantile(0.95),

        "Winner_MAE_Points_P90":
            winners["MAE_Points"]
            .quantile(0.90),

        "Winner_MAE_Points_P95":
            winners["MAE_Points"]
            .quantile(0.95),

        "Loser_MAE_ATR_Median":
            losers["MAE_ATR"]
            .median(),

        "Loser_MAE_ATR_P75":
            losers["MAE_ATR"]
            .quantile(0.75),

        "Loser_MAE_ATR_P90":
            losers["MAE_ATR"]
            .quantile(0.90)
    })


winner_mae_df = pd.DataFrame(
    winner_mae_rows
)


# ============================================================
# SAVE
# ============================================================

for system_name, d in datasets.items():

    safe_name = (
        system_name
        .lower()
    )

    d.to_csv(
        OUTPUT_DIR
        /
        f"{safe_name}_mae_mfe_trades.csv",
        index=False
    )


distribution_df.to_csv(
    OUTPUT_DIR
    /
    "mae_mfe_distribution.csv",
    index=False
)

winner_year_df.to_csv(
    OUTPUT_DIR
    /
    "winner_mae_year_by_year.csv",
    index=False
)

winner_mae_df.to_csv(
    OUTPUT_DIR
    /
    "winner_mae_summary.csv",
    index=False
)

baseline_df.to_csv(
    OUTPUT_DIR
    /
    "baseline_no_stop.csv",
    index=False
)

stop_summary_df.to_csv(
    OUTPUT_DIR
    /
    "stop_atr_summary.csv",
    index=False
)

stop_year_df.to_csv(
    OUTPUT_DIR
    /
    "stop_atr_year_by_year.csv",
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
    3000
)

pd.set_option(
    "display.width",
    600
)


print()
print("=" * 240)
print("BASELINE — NO STOP")
print("=" * 240)

print(
    baseline_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 240)
print("WINNER / LOSER MAE SUMMARY")
print("=" * 240)

print(
    winner_mae_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 240)
print("FULL MAE / MFE DISTRIBUTION")
print("=" * 240)

print(
    distribution_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 240)
print("WINNER MAE — YEAR BY YEAR")
print("=" * 240)

print(
    winner_year_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 240)
print("ATR STOP SIMULATION")
print("=" * 240)

print(
    stop_summary_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 240)
print("STOP PF × YEAR")
print("=" * 240)

pf_matrix = (
    stop_year_df
    .pivot_table(
        index=[
            "System",
            "StopATR"
        ],
        columns="Year",
        values="ProfitFactor"
    )
)

print(
    pf_matrix
    .round(4)
    .to_string()
)


print()
print("=" * 240)
print("DONE")
print("=" * 240)

print()
print("Output folder:")
print(OUTPUT_DIR)