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
    r"\MACD_15M_Setup_WaitFor30M_Cross"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026
]

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

ARM_WINDOWS_MINUTES = [
    30,
    60,
    90,
    120,
    180
]

HOLD_MINUTES_LIST = [
    60,
    90,
    120,
    150,
    180,
    210,
    240,
    300
]


# ============================================================
# LOAD 1-MIN DATA
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
# BUILD 15M BARS
# ============================================================

print()
print("=" * 180)
print("BUILDING 15-MIN BARS")
print("=" * 180)

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

print(
    "15m bars:",
    len(df15)
)


# ============================================================
# BUILD 30M BARS
# ============================================================

print()
print("=" * 180)
print("BUILDING 30-MIN BARS")
print("=" * 180)

df30 = df1.resample(
    "30min",
    label="right",
    closed="right"
).agg({
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum"
})

df30 = df30.dropna(
    subset=[
        "Open",
        "High",
        "Low",
        "Close"
    ]
)

print(
    "30m bars:",
    len(df30)
)


# ============================================================
# MACD FUNCTION
# ============================================================

def add_macd(df):

    fast = df["Close"].ewm(
        span=MACD_FAST,
        adjust=False
    ).mean()

    slow = df["Close"].ewm(
        span=MACD_SLOW,
        adjust=False
    ).mean()

    df["MACD"] = (
        fast
        -
        slow
    )

    df["MACDSignal"] = df["MACD"].ewm(
        span=MACD_SIGNAL,
        adjust=False
    ).mean()

    df["Hist"] = (
        df["MACD"]
        -
        df["MACDSignal"]
    )

    df["BullCross"] = (
        (df["Hist"].shift(1) <= 0)
        &
        (df["Hist"] > 0)
    )

    return df


df15 = add_macd(
    df15
)

df30 = add_macd(
    df30
)


# ============================================================
# 15M WILDER ATR14
# ============================================================

prev_close = df15[
    "Close"
].shift(1)

tr = pd.concat(
    [
        df15["High"] - df15["Low"],
        (
            df15["High"]
            -
            prev_close
        ).abs(),
        (
            df15["Low"]
            -
            prev_close
        ).abs()
    ],
    axis=1
).max(
    axis=1
)

df15["ATR14"] = tr.ewm(
    alpha=1.0 / ATR_LENGTH,
    adjust=False
).mean()


# ============================================================
# 15M PRICE MOVE
#
# (Close_t - Close_t-1) / ATR14_t
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
# 15M PRIOR MOVE
#
# (Close_t-1 - Close_t-20) / ATR14_t
# ============================================================

df15["PriorMoveATR14"] = (
    (
        df15["Close"].shift(1)
        -
        df15["Close"].shift(
            PRIOR_LOOKBACK
        )
    )
    /
    df15["ATR14"]
)


# ============================================================
# ORIGINAL 20-BAR RANGE
#
# Previous EXACTLY 20 bars:
# t-20 ... t-1
#
# RangeHigh = max High
# RangeLow  = min Low
#
# Height <= 25
# Drift <= 10
#
# Reject only if cross-bar close lies INSIDE valid range.
# ============================================================

df15["RangeHigh"] = (
    df15["High"]
    .shift(1)
    .rolling(
        RANGE_LOOKBACK
    )
    .max()
)

df15["RangeLow"] = (
    df15["Low"]
    .shift(1)
    .rolling(
        RANGE_LOOKBACK
    )
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
    df15["Close"].shift(
        RANGE_LOOKBACK
    )
).abs()

df15["ValidRange"] = (
    (
        df15["RangeHeight"]
        <=
        RANGE_MAX_HEIGHT
    )
    &
    (
        df15["RangeDrift"]
        <=
        RANGE_MAX_DRIFT
    )
)

df15["InsideValidRange"] = (
    df15["ValidRange"]
    &
    (
        df15["Close"]
        >=
        df15["RangeLow"]
    )
    &
    (
        df15["Close"]
        <=
        df15["RangeHigh"]
    )
)

df15["NoRangePass"] = (
    ~df15[
        "InsideValidRange"
    ]
)


# ============================================================
# 15M FILTER FLAGS
# ============================================================

df15["PriceMovePass"] = (
    (
        df15["PriceMoveATR14"]
        >=
        PRICE_MOVE_MIN
    )
    &
    (
        df15["PriceMoveATR14"]
        <=
        PRICE_MOVE_MAX
    )
)

df15["PriorMovePass"] = (
    (
        df15["PriorMoveATR14"]
        >=
        PRIOR_MOVE_MIN
    )
    &
    (
        df15["PriorMoveATR14"]
        <=
        PRIOR_MOVE_MAX
    )
)

df15["SessionPass"] = (
    (
        df15.index.hour
        >=
        SESSION_START
    )
    &
    (
        df15.index.hour
        <
        SESSION_END
    )
)


# ============================================================
# COMPLETE 15M SETUP
# ============================================================

df15["Qualified15m"] = (
    df15["BullCross"]
    &
    df15["PriceMovePass"]
    &
    df15["NoRangePass"]
    &
    df15["PriorMovePass"]
    &
    df15["SessionPass"]
)


qualified15 = df15[
    df15["Qualified15m"]
    &
    df15.index.year.isin(
        YEARS
    )
].copy()


print()
print("=" * 180)
print("15M QUALIFIED SETUPS")
print("=" * 180)

print(
    "Qualified 15m signals:",
    len(
        qualified15
    )
)


# ============================================================
# 30M BULLISH CROSS TIMES
# ============================================================

cross30_times = df30.index[
    df30["BullCross"]
    &
    df30.index.year.isin(
        YEARS
    )
]

cross30_np = (
    cross30_times
    .values
    .astype(
        "datetime64[ns]"
    )
)

print(
    "30m bullish crosses:",
    len(
        cross30_times
    )
)


# ============================================================
# 1M NUMPY ARRAYS
# ============================================================

minute_times = (
    df1.index
    .values
    .astype(
        "datetime64[ns]"
    )
)

minute_open = df1[
    "Open"
].to_numpy(
    dtype=float
)

minute_close = df1[
    "Close"
].to_numpy(
    dtype=float
)


# ============================================================
# STATISTICS
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
        np.sqrt(
            len(pnl)
        )
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
            np.cumsum(
                pnl
            )
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
            (
                np.mean(
                    pnl > 0
                )
                *
                100
            ),

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
# HELPER:
# EXECUTE FIRST 1M OPEN AFTER SIGNAL
# ============================================================

def get_entry(
    signal_time
):

    signal_np = np.datetime64(
        signal_time
    )

    pos = np.searchsorted(
        minute_times,
        signal_np,
        side="right"
    )

    if pos >= len(
        minute_times
    ):
        return None

    return (
        pos,
        pd.Timestamp(
            minute_times[
                pos
            ]
        ),
        float(
            minute_open[
                pos
            ]
        )
    )


# ============================================================
# HELPER:
# EXIT AFTER HOLD PERIOD
# ============================================================

def get_exit(
    entry_time,
    hold_minutes
):

    target = (
        np.datetime64(
            entry_time
        )
        +
        np.timedelta64(
            hold_minutes,
            "m"
        )
    )

    pos = np.searchsorted(
        minute_times,
        target,
        side="left"
    )

    if pos >= len(
        minute_times
    ):
        return None

    actual_time = minute_times[
        pos
    ]

    if (
        actual_time
        >
        target
        +
        np.timedelta64(
            2,
            "m"
        )
    ):
        return None

    return (
        pd.Timestamp(
            actual_time
        ),
        float(
            minute_close[
                pos
            ]
        )
    )


# ============================================================
# BUILD IMMEDIATE 15M BASELINE
# ============================================================

print()
print("=" * 180)
print("BUILDING IMMEDIATE 15M BASELINE")
print("=" * 180)

trade_rows = []


for signal_time, r in qualified15.iterrows():

    entry = get_entry(
        signal_time
    )

    if entry is None:
        continue

    (
        entry_pos,
        entry_time,
        entry_price
    ) = entry


    for hold_minutes in HOLD_MINUTES_LIST:

        exit_data = get_exit(
            entry_time,
            hold_minutes
        )

        if exit_data is None:
            continue

        (
            exit_time,
            exit_price
        ) = exit_data

        trade_rows.append({

            "Strategy":
                "15M_IMMEDIATE",

            "ArmWindowMinutes":
                0,

            "Signal15mTime":
                signal_time,

            "Trigger30mTime":
                pd.NaT,

            "WaitMinutes":
                0.0,

            "Year":
                signal_time.year,

            "Hour":
                signal_time.hour,

            "PriceMoveATR14":
                r[
                    "PriceMoveATR14"
                ],

            "PriorMoveATR14":
                r[
                    "PriorMoveATR14"
                ],

            "EntryTime":
                entry_time,

            "EntryPrice":
                entry_price,

            "HoldingMinutes":
                hold_minutes,

            "ExitTime":
                exit_time,

            "ExitPrice":
                exit_price,

            "PnL":
                (
                    exit_price
                    -
                    entry_price
                )
        })


# ============================================================
# WAIT FOR 30M CROSS
# ============================================================

print()
print("=" * 180)
print("BUILDING 15M-ARMED / 30M-TRIGGERED TRADES")
print("=" * 180)


trigger_rows = []


for signal_time, r in qualified15.iterrows():

    signal_np = np.datetime64(
        signal_time
    )


    # First 30m cross AT OR AFTER the 15m signal.
    #
    # Same-timestamp 15m + 30m cross is allowed because both
    # completed bars are known at that instant.
    cross_idx = np.searchsorted(
        cross30_np,
        signal_np,
        side="left"
    )


    if cross_idx >= len(
        cross30_np
    ):
        continue


    trigger_np = cross30_np[
        cross_idx
    ]

    trigger_time = pd.Timestamp(
        trigger_np
    )

    wait_minutes = (
        trigger_np
        -
        signal_np
    ) / np.timedelta64(
        1,
        "m"
    )


    for arm_window in ARM_WINDOWS_MINUTES:

        if (
            wait_minutes
            >
            arm_window
        ):
            continue


        entry = get_entry(
            trigger_time
        )

        if entry is None:
            continue


        (
            entry_pos,
            entry_time,
            entry_price
        ) = entry


        trigger_rows.append({

            "ArmWindowMinutes":
                arm_window,

            "Signal15mTime":
                signal_time,

            "Trigger30mTime":
                trigger_time,

            "WaitMinutes":
                float(
                    wait_minutes
                ),

            "Year":
                signal_time.year,

            "Hour":
                signal_time.hour,

            "PriceMoveATR14":
                r[
                    "PriceMoveATR14"
                ],

            "PriorMoveATR14":
                r[
                    "PriorMoveATR14"
                ],

            "EntryTime":
                entry_time,

            "EntryPrice":
                entry_price
        })


trigger_df = pd.DataFrame(
    trigger_rows
)


print(
    "Triggered rows across all windows:",
    len(
        trigger_df
    )
)


# ============================================================
# ADD FIXED HOLD EXITS
# ============================================================

if len(
    trigger_df
):

    for _, r in trigger_df.iterrows():

        for hold_minutes in HOLD_MINUTES_LIST:

            exit_data = get_exit(
                r[
                    "EntryTime"
                ],
                hold_minutes
            )

            if exit_data is None:
                continue

            (
                exit_time,
                exit_price
            ) = exit_data


            trade_rows.append({

                "Strategy":
                    "WAIT_30M_CROSS",

                "ArmWindowMinutes":
                    int(
                        r[
                            "ArmWindowMinutes"
                        ]
                    ),

                "Signal15mTime":
                    r[
                        "Signal15mTime"
                    ],

                "Trigger30mTime":
                    r[
                        "Trigger30mTime"
                    ],

                "WaitMinutes":
                    r[
                        "WaitMinutes"
                    ],

                "Year":
                    r[
                        "Year"
                    ],

                "Hour":
                    r[
                        "Hour"
                    ],

                "PriceMoveATR14":
                    r[
                        "PriceMoveATR14"
                    ],

                "PriorMoveATR14":
                    r[
                        "PriorMoveATR14"
                    ],

                "EntryTime":
                    r[
                        "EntryTime"
                    ],

                "EntryPrice":
                    r[
                        "EntryPrice"
                    ],

                "HoldingMinutes":
                    hold_minutes,

                "ExitTime":
                    exit_time,

                "ExitPrice":
                    exit_price,

                "PnL":
                    (
                        exit_price
                        -
                        r[
                            "EntryPrice"
                        ]
                    )
            })


trades = pd.DataFrame(
    trade_rows
)


# ============================================================
# RETENTION / WAIT STATS
# ============================================================

retention_rows = []

base_n = len(
    qualified15
)


for arm_window in ARM_WINDOWS_MINUTES:

    d = trigger_df[
        trigger_df[
            "ArmWindowMinutes"
        ]
        ==
        arm_window
    ]


    retention_rows.append({

        "ArmWindowMinutes":
            arm_window,

        "Qualified15mSignals":
            base_n,

        "Triggered30mEntries":
            len(
                d
            ),

        "RetentionPct":
            (
                len(d)
                /
                base_n
                *
                100
            )
            if base_n
            else np.nan,

        "MedianWaitMinutes":
            d[
                "WaitMinutes"
            ].median()
            if len(d)
            else np.nan,

        "MeanWaitMinutes":
            d[
                "WaitMinutes"
            ].mean()
            if len(d)
            else np.nan
    })


retention_df = pd.DataFrame(
    retention_rows
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []
year_rows = []


strategy_settings = [
    (
        "15M_IMMEDIATE",
        0
    )
]

strategy_settings += [
    (
        "WAIT_30M_CROSS",
        x
    )
    for x in ARM_WINDOWS_MINUTES
]


for (
    strategy,
    arm_window
) in strategy_settings:

    for hold_minutes in HOLD_MINUTES_LIST:

        d = trades[
            (
                trades[
                    "Strategy"
                ]
                ==
                strategy
            )
            &
            (
                trades[
                    "ArmWindowMinutes"
                ]
                ==
                arm_window
            )
            &
            (
                trades[
                    "HoldingMinutes"
                ]
                ==
                hold_minutes
            )
        ].copy()


        s = calculate_stats(
            d
        )


        year_pfs = []


        for year in YEARS:

            yd = d[
                d[
                    "Year"
                ]
                ==
                year
            ]

            ys = calculate_stats(
                yd
            )


            year_rows.append({

                "Strategy":
                    strategy,

                "ArmWindowMinutes":
                    arm_window,

                "HoldingMinutes":
                    hold_minutes,

                "Year":
                    year,

                **ys
            })


            pf_value = ys[
                "ProfitFactor"
            ]

            if np.isfinite(
                pf_value
            ):

                year_pfs.append(
                    pf_value
                )


        summary_rows.append({

            "Strategy":
                strategy,

            "ArmWindowMinutes":
                arm_window,

            "HoldingMinutes":
                hold_minutes,

            **s,

            "WorstYearPF":
                (
                    min(
                        year_pfs
                    )
                    if year_pfs
                    else np.nan
                ),

            "BestYearPF":
                (
                    max(
                        year_pfs
                    )
                    if year_pfs
                    else np.nan
                ),

            "ProfitableYears":
                sum(
                    x > 1
                    for x in year_pfs
                ),

            "MedianWaitMinutes":
                (
                    d[
                        "WaitMinutes"
                    ].median()
                    if len(d)
                    else np.nan
                )
        })


summary_df = pd.DataFrame(
    summary_rows
)

year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# YEAR PF MATRIX
# ============================================================

year_pf_matrix = year_df.pivot_table(
    index=[
        "Strategy",
        "ArmWindowMinutes",
        "HoldingMinutes"
    ],
    columns="Year",
    values="ProfitFactor"
)


# ============================================================
# RANKING
# ============================================================

ranking_df = summary_df.sort_values(
    [
        "WorstYearPF",
        "ProfitFactor",
        "AveragePnL"
    ],
    ascending=[
        False,
        False,
        False
    ]
)


# ============================================================
# SAVE
# ============================================================

qualified15.to_csv(
    OUTPUT_DIR
    /
    "qualified_15m_signals.csv"
)

trigger_df.to_csv(
    OUTPUT_DIR
    /
    "30m_trigger_detail.csv",
    index=False
)

trades.to_csv(
    OUTPUT_DIR
    /
    "trade_detail.csv",
    index=False
)

retention_df.to_csv(
    OUTPUT_DIR
    /
    "trigger_retention.csv",
    index=False
)

summary_df.to_csv(
    OUTPUT_DIR
    /
    "summary.csv",
    index=False
)

ranking_df.to_csv(
    OUTPUT_DIR
    /
    "ranking.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "year_by_year.csv",
    index=False
)

year_pf_matrix.to_csv(
    OUTPUT_DIR
    /
    "year_pf_matrix.csv"
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
print("30M TRIGGER RETENTION")
print("=" * 240)

print(
    retention_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 280)
print("SUMMARY")
print("=" * 280)

print(
    summary_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 280)
print("TOP RESULTS — SORTED BY WORST-YEAR PF")
print("=" * 280)

print(
    ranking_df
    .head(
        30
    )
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 280)
print("YEAR-BY-YEAR PF")
print("=" * 280)

print(
    year_pf_matrix
    .round(4)
    .to_string()
)


# ============================================================
# DIRECT COMPARISON AT 210 MIN
# ============================================================

print()
print("=" * 280)
print("DIRECT COMPARISON — 210 MIN HOLD")
print("=" * 280)

compare_210 = summary_df[
    summary_df[
        "HoldingMinutes"
    ]
    ==
    210
][
    [
        "Strategy",
        "ArmWindowMinutes",
        "N",
        "WinRatePct",
        "NetPoints",
        "AveragePnL",
        "MedianPnL",
        "ProfitFactor",
        "SQN",
        "MaxDrawdown",
        "WorstYearPF",
        "ProfitableYears",
        "MedianWaitMinutes"
    ]
]

print(
    compare_210
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 280)
print("DONE")
print("=" * 280)

print()
print(
    "15M setup:"
)

print(
    "Bullish MACD(5,36,5) cross"
)

print(
    "PriceMoveATR14 0.35 to 0.90"
)

print(
    "Original previous-20-bar no-range rule"
)

print(
    "PriorMoveATR14 -2 to +2"
)

print(
    "Session 14:00 <= hour < 18:00"
)

print()
print(
    "30M trigger:"
)

print(
    "First bullish 30m MACD(5,36,5) cross at or after qualifying 15m setup."
)

print()
print(
    "If no 30m cross occurs before the arm window expires, no trade."
)

print()
print("Output folder:")
print(OUTPUT_DIR)