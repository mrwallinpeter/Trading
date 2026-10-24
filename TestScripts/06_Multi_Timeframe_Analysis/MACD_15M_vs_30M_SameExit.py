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
    r"\MACD_15M_vs_30M_SameExit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

YEARS = [2022, 2023, 2024, 2025, 2026]

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
RANGE_MAX_HEIGHT = 25.0
RANGE_MAX_DRIFT = 10.0

SESSION_START = 14
SESSION_END = 18

ARM_WINDOWS = [
    30,
    60,
    90,
    120,
    180
]

HOLDS = [
    120,
    150,
    180,
    210,
    240,
    300
]


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 180)
print("LOADING 1-MIN DATA")
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

for c in [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]:
    df1[c] = pd.to_numeric(
        df1[c],
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
# RESAMPLE
# ============================================================

def build_bars(freq):

    d = df1.resample(
        freq,
        label="right",
        closed="right"
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })

    return d.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close"
        ]
    )


df15 = build_bars("15min")
df30 = build_bars("30min")

print()
print("15m bars:", len(df15))
print("30m bars:", len(df30))


# ============================================================
# MACD
# ============================================================

def add_macd(d):

    fast = d["Close"].ewm(
        span=MACD_FAST,
        adjust=False
    ).mean()

    slow = d["Close"].ewm(
        span=MACD_SLOW,
        adjust=False
    ).mean()

    d["MACD"] = fast - slow

    d["MACDSignal"] = d["MACD"].ewm(
        span=MACD_SIGNAL,
        adjust=False
    ).mean()

    d["Hist"] = (
        d["MACD"]
        -
        d["MACDSignal"]
    )

    d["BullCross"] = (
        (d["Hist"].shift(1) <= 0)
        &
        (d["Hist"] > 0)
    )

    return d


df15 = add_macd(df15)
df30 = add_macd(df30)


# ============================================================
# 15M ATR
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
    alpha=1 / ATR_LENGTH,
    adjust=False
).mean()


# ============================================================
# PRICE MOVE
# ============================================================

df15["PriceMoveATR14"] = (
    (
        df15["Close"]
        -
        df15["Close"].shift(1)
    )
    /
    df15["ATR14"]
)


# ============================================================
# PRIOR MOVE
# ============================================================

df15["PriorMoveATR14"] = (
    (
        df15["Close"].shift(1)
        -
        df15["Close"].shift(PRIOR_LOOKBACK)
    )
    /
    df15["ATR14"]
)


# ============================================================
# ORIGINAL RANGE FILTER
# ============================================================

df15["RangeHigh"] = (
    df15["High"]
    .shift(1)
    .rolling(RANGE_LOOKBACK)
    .max()
)

df15["RangeLow"] = (
    df15["Low"]
    .shift(1)
    .rolling(RANGE_LOOKBACK)
    .min()
)

df15["RangeHeight"] = (
    df15["RangeHigh"]
    -
    df15["RangeLow"]
)

df15["RangeDrift"] = (
    df15["Close"].shift(1)
    -
    df15["Close"].shift(RANGE_LOOKBACK)
).abs()

df15["ValidRange"] = (
    (df15["RangeHeight"] <= RANGE_MAX_HEIGHT)
    &
    (df15["RangeDrift"] <= RANGE_MAX_DRIFT)
)

df15["InsideValidRange"] = (
    df15["ValidRange"]
    &
    (df15["Close"] >= df15["RangeLow"])
    &
    (df15["Close"] <= df15["RangeHigh"])
)

df15["NoRangePass"] = (
    ~df15["InsideValidRange"]
)


# ============================================================
# COMPLETE 15M SETUP
# ============================================================

df15["Qualified"] = (
    df15["BullCross"]
    &
    (df15["PriceMoveATR14"] >= PRICE_MOVE_MIN)
    &
    (df15["PriceMoveATR14"] <= PRICE_MOVE_MAX)
    &
    df15["NoRangePass"]
    &
    (df15["PriorMoveATR14"] >= PRIOR_MOVE_MIN)
    &
    (df15["PriorMoveATR14"] <= PRIOR_MOVE_MAX)
    &
    (df15.index.hour >= SESSION_START)
    &
    (df15.index.hour < SESSION_END)
    &
    df15.index.year.isin(YEARS)
)

qualified = df15[
    df15["Qualified"]
].copy()

print()
print("=" * 180)
print("QUALIFIED 15M SETUPS")
print("=" * 180)

print("N =", len(qualified))


# ============================================================
# NUMPY EXECUTION ARRAYS
# ============================================================

minute_times = df1.index.values.astype(
    "datetime64[ns]"
)

minute_open = df1["Open"].to_numpy(
    dtype=float
)

minute_close = df1["Close"].to_numpy(
    dtype=float
)

cross30_times = df30.index[
    df30["BullCross"]
].values.astype(
    "datetime64[ns]"
)


# ============================================================
# EXECUTION HELPERS
# ============================================================

def first_open_after(signal_time):

    t = np.datetime64(signal_time)

    p = np.searchsorted(
        minute_times,
        t,
        side="right"
    )

    if p >= len(minute_times):
        return None

    return (
        pd.Timestamp(minute_times[p]),
        float(minute_open[p])
    )


def close_at_target(target_time):

    t = np.datetime64(target_time)

    p = np.searchsorted(
        minute_times,
        t,
        side="left"
    )

    if p >= len(minute_times):
        return None

    actual = minute_times[p]

    if (
        actual
        >
        t + np.timedelta64(2, "m")
    ):
        return None

    return (
        pd.Timestamp(actual),
        float(minute_close[p])
    )


# ============================================================
# BUILD SIGNAL / DELAY DIAGNOSTICS
# ============================================================

print()
print("=" * 180)
print("FINDING NEXT 30M CROSS AFTER EACH 15M SETUP")
print("=" * 180)

signal_rows = []


for signal_time, r in qualified.iterrows():

    # --------------------------------------------------------
    # Original 15m entry
    # --------------------------------------------------------

    base_entry = first_open_after(
        signal_time
    )

    if base_entry is None:
        continue

    base_entry_time, base_entry_price = base_entry


    # --------------------------------------------------------
    # Find next 30m bullish cross at or after 15m signal
    # --------------------------------------------------------

    signal_np = np.datetime64(
        signal_time
    )

    idx = np.searchsorted(
        cross30_times,
        signal_np,
        side="left"
    )

    if idx >= len(cross30_times):

        trigger_time = pd.NaT
        trigger_entry_time = pd.NaT
        trigger_entry_price = np.nan

        signal_delay = np.nan
        entry_delay = np.nan

    else:

        trigger_np = cross30_times[idx]

        trigger_time = pd.Timestamp(
            trigger_np
        )

        delayed_entry = first_open_after(
            trigger_time
        )

        if delayed_entry is None:

            trigger_entry_time = pd.NaT
            trigger_entry_price = np.nan

            signal_delay = np.nan
            entry_delay = np.nan

        else:

            (
                trigger_entry_time,
                trigger_entry_price
            ) = delayed_entry

            signal_delay = (
                trigger_time
                -
                signal_time
            ).total_seconds() / 60.0

            entry_delay = (
                trigger_entry_time
                -
                base_entry_time
            ).total_seconds() / 60.0


    # --------------------------------------------------------
    # 30m state at / immediately before signal
    # --------------------------------------------------------

    prior30_idx = df30.index.searchsorted(
        signal_time,
        side="right"
    ) - 1

    if prior30_idx >= 0:

        hist30_at_signal = float(
            df30.iloc[
                prior30_idx
            ]["Hist"]
        )

        last30_time = df30.index[
            prior30_idx
        ]

    else:

        hist30_at_signal = np.nan
        last30_time = pd.NaT


    signal_rows.append({

        "Signal15mTime":
            signal_time,

        "Year":
            signal_time.year,

        "PriceMoveATR14":
            r["PriceMoveATR14"],

        "PriorMoveATR14":
            r["PriorMoveATR14"],

        "BaseEntryTime":
            base_entry_time,

        "BaseEntryPrice":
            base_entry_price,

        "LastCompleted30mTimeAt15mSignal":
            last30_time,

        "Hist30At15mSignal":
            hist30_at_signal,

        "Hist30PositiveAt15mSignal":
            (
                hist30_at_signal > 0
                if np.isfinite(hist30_at_signal)
                else False
            ),

        "Next30mCrossTime":
            trigger_time,

        "SignalTo30mCrossMinutes":
            signal_delay,

        "DelayedEntryTime":
            trigger_entry_time,

        "DelayedEntryPrice":
            trigger_entry_price,

        "EntryDelayMinutes":
            entry_delay
    })


signals = pd.DataFrame(
    signal_rows
)


# ============================================================
# DELAY DISTRIBUTION
# ============================================================

delay_distribution = (
    signals[
        "SignalTo30mCrossMinutes"
    ]
    .value_counts(
        dropna=False
    )
    .sort_index()
    .rename_axis(
        "DelayMinutes"
    )
    .reset_index(
        name="N"
    )
)

delay_distribution[
    "PctOfQualifiedSignals"
] = (
    delay_distribution["N"]
    /
    len(signals)
    *
    100
)


# ============================================================
# PAIRED SAME-EXIT TEST
#
# IMPORTANT:
#
# For each signal:
#
# BASE:
# entry immediately after 15m signal
#
# DELAYED:
# entry after 30m cross
#
# BOTH EXIT AT:
# BaseEntryTime + original holding period
#
# Therefore delayed trade has:
#
# EffectiveHold =
# original hold - actual entry delay
# ============================================================

trade_rows = []


for _, r in signals.iterrows():

    if pd.isna(
        r["DelayedEntryTime"]
    ):
        continue


    for arm_window in ARM_WINDOWS:

        delay = float(
            r["SignalTo30mCrossMinutes"]
        )

        if delay > arm_window:
            continue


        for original_hold in HOLDS:

            absolute_exit_target = (
                r["BaseEntryTime"]
                +
                pd.Timedelta(
                    minutes=original_hold
                )
            )


            # Delayed entry must occur before original exit
            if (
                r["DelayedEntryTime"]
                >=
                absolute_exit_target
            ):
                continue


            exit_data = close_at_target(
                absolute_exit_target
            )

            if exit_data is None:
                continue


            (
                exit_time,
                exit_price
            ) = exit_data


            base_pnl = (
                exit_price
                -
                r["BaseEntryPrice"]
            )


            delayed_pnl = (
                exit_price
                -
                r["DelayedEntryPrice"]
            )


            effective_delayed_hold = (
                exit_time
                -
                r["DelayedEntryTime"]
            ).total_seconds() / 60.0


            price_change_before_30m_entry = (
                r["DelayedEntryPrice"]
                -
                r["BaseEntryPrice"]
            )


            trade_rows.append({

                "ArmWindowMinutes":
                    arm_window,

                "OriginalHoldMinutes":
                    original_hold,

                "Signal15mTime":
                    r["Signal15mTime"],

                "Year":
                    r["Year"],

                "SignalTo30mCrossMinutes":
                    r[
                        "SignalTo30mCrossMinutes"
                    ],

                "EntryDelayMinutes":
                    r[
                        "EntryDelayMinutes"
                    ],

                "BaseEntryTime":
                    r["BaseEntryTime"],

                "BaseEntryPrice":
                    r["BaseEntryPrice"],

                "DelayedEntryTime":
                    r["DelayedEntryTime"],

                "DelayedEntryPrice":
                    r["DelayedEntryPrice"],

                "PriceMoveBeforeDelayedEntry":
                    price_change_before_30m_entry,

                "AbsoluteExitTime":
                    exit_time,

                "ExitPrice":
                    exit_price,

                "DelayedEffectiveHoldMinutes":
                    effective_delayed_hold,

                "BasePnL":
                    base_pnl,

                "DelayedPnL":
                    delayed_pnl,

                "PnLDifference_DelayedMinusBase":
                    delayed_pnl
                    -
                    base_pnl
            })


paired = pd.DataFrame(
    trade_rows
)


# ============================================================
# STATS
# ============================================================

def profit_factor(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    gp = pnl[
        pnl > 0
    ].sum()

    gl = -pnl[
        pnl < 0
    ].sum()

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

    sd = pnl.std(
        ddof=1
    )

    if sd == 0:
        return np.nan

    return (
        np.sqrt(len(pnl))
        *
        pnl.mean()
        /
        sd
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

    return abs(
        (
            equity
            -
            peak
        ).min()
    )


def stats_from_pnl(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    pnl = pnl[
        np.isfinite(pnl)
    ]

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
            np.median(pnl),

        "ProfitFactor":
            profit_factor(pnl),

        "SQN":
            sqn(pnl),

        "MaxDrawdown":
            max_drawdown(pnl)
    }


# ============================================================
# SUMMARY — PAIRED SIGNALS ONLY
# ============================================================

summary_rows = []
year_rows = []


for arm_window in ARM_WINDOWS:

    for hold in HOLDS:

        d = paired[
            (
                paired["ArmWindowMinutes"]
                ==
                arm_window
            )
            &
            (
                paired["OriginalHoldMinutes"]
                ==
                hold
            )
        ].copy()


        base_stats = stats_from_pnl(
            d["BasePnL"]
        )

        delayed_stats = stats_from_pnl(
            d["DelayedPnL"]
        )


        base_year_pfs = []
        delayed_year_pfs = []


        for year in YEARS:

            yd = d[
                d["Year"]
                ==
                year
            ]

            bpf = profit_factor(
                yd["BasePnL"]
            ) if len(yd) else np.nan

            dpf = profit_factor(
                yd["DelayedPnL"]
            ) if len(yd) else np.nan


            year_rows.append({

                "ArmWindowMinutes":
                    arm_window,

                "OriginalHoldMinutes":
                    hold,

                "Year":
                    year,

                "N":
                    len(yd),

                "BasePF":
                    bpf,

                "DelayedPF":
                    dpf,

                "BaseAveragePnL":
                    yd[
                        "BasePnL"
                    ].mean()
                    if len(yd)
                    else np.nan,

                "DelayedAveragePnL":
                    yd[
                        "DelayedPnL"
                    ].mean()
                    if len(yd)
                    else np.nan
            })


            if np.isfinite(bpf):
                base_year_pfs.append(bpf)

            if np.isfinite(dpf):
                delayed_year_pfs.append(dpf)


        summary_rows.append({

            "ArmWindowMinutes":
                arm_window,

            "OriginalHoldMinutes":
                hold,

            "N":
                len(d),

            "MedianCrossDelay":
                d[
                    "SignalTo30mCrossMinutes"
                ].median()
                if len(d)
                else np.nan,

            "MeanCrossDelay":
                d[
                    "SignalTo30mCrossMinutes"
                ].mean()
                if len(d)
                else np.nan,

            "MedianEntryDelay":
                d[
                    "EntryDelayMinutes"
                ].median()
                if len(d)
                else np.nan,

            "MedianDelayedEffectiveHold":
                d[
                    "DelayedEffectiveHoldMinutes"
                ].median()
                if len(d)
                else np.nan,

            "MeanPriceMoveBefore30mEntry":
                d[
                    "PriceMoveBeforeDelayedEntry"
                ].mean()
                if len(d)
                else np.nan,

            "MedianPriceMoveBefore30mEntry":
                d[
                    "PriceMoveBeforeDelayedEntry"
                ].median()
                if len(d)
                else np.nan,

            "BaseAvgPnL":
                base_stats[
                    "AveragePnL"
                ],

            "BasePF":
                base_stats[
                    "ProfitFactor"
                ],

            "BaseWorstYearPF":
                min(base_year_pfs)
                if base_year_pfs
                else np.nan,

            "DelayedAvgPnL":
                delayed_stats[
                    "AveragePnL"
                ],

            "DelayedPF":
                delayed_stats[
                    "ProfitFactor"
                ],

            "DelayedWorstYearPF":
                min(delayed_year_pfs)
                if delayed_year_pfs
                else np.nan,

            "AvgPnLDifference":
                d[
                    "PnLDifference_DelayedMinusBase"
                ].mean()
                if len(d)
                else np.nan
        })


summary = pd.DataFrame(
    summary_rows
)

year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# 210-MIN FOCUS
# ============================================================

focus210 = summary[
    summary[
        "OriginalHoldMinutes"
    ]
    ==
    210
].copy()


# ============================================================
# SAVE
# ============================================================

signals.to_csv(
    OUTPUT_DIR
    /
    "signal_delay_detail.csv",
    index=False
)

delay_distribution.to_csv(
    OUTPUT_DIR
    /
    "delay_distribution.csv",
    index=False
)

paired.to_csv(
    OUTPUT_DIR
    /
    "paired_same_exit_trade_detail.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR
    /
    "paired_summary.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "paired_year_by_year.csv",
    index=False
)

focus210.to_csv(
    OUTPUT_DIR
    /
    "focus_210min.csv",
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
    800
)


print()
print("=" * 240)
print("30M CROSS DELAY DISTRIBUTION")
print("=" * 240)

print(
    delay_distribution
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 240)
print("30M STATE WHEN 15M SIGNAL OCCURS")
print("=" * 240)

print(
    "15m signals:",
    len(signals)
)

print(
    "30m histogram already positive:",
    int(
        signals[
            "Hist30PositiveAt15mSignal"
        ].sum()
    )
)

print(
    "Percent already positive:",
    round(
        signals[
            "Hist30PositiveAt15mSignal"
        ].mean()
        *
        100,
        2
    )
)


print()
print("=" * 300)
print("PAIRED SAME-EXIT SUMMARY")
print("=" * 300)

print(
    summary
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 300)
print("210-MIN ORIGINAL HOLD — SAME ABSOLUTE EXIT")
print("=" * 300)

print(
    focus210[
        [
            "ArmWindowMinutes",
            "N",
            "MedianCrossDelay",
            "MedianEntryDelay",
            "MedianDelayedEffectiveHold",
            "MeanPriceMoveBefore30mEntry",
            "MedianPriceMoveBefore30mEntry",
            "BaseAvgPnL",
            "BasePF",
            "BaseWorstYearPF",
            "DelayedAvgPnL",
            "DelayedPF",
            "DelayedWorstYearPF",
            "AvgPnLDifference"
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 300)
print("YEAR-BY-YEAR — 210 MIN")
print("=" * 300)

print(
    year_df[
        year_df[
            "OriginalHoldMinutes"
        ]
        ==
        210
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 240)
print("INTERPRETATION")
print("=" * 240)

print()
print(
    "Base and delayed trades use EXACTLY the same signals "
    "within each arm window."
)

print(
    "Both trades use EXACTLY the same final exit timestamp."
)

print(
    "Therefore the only difference is the entry price/timing."
)

print()
print(
    "DelayedEffectiveHold = original hold minus actual entry delay."
)

print()
print(
    "PriceMoveBefore30mEntry tells us how much price has already "
    "moved between the 15m entry and the later 30m entry."
)

print()
print("=" * 240)
print("DONE")
print("=" * 240)

print()
print("Output folder:")
print(OUTPUT_DIR)