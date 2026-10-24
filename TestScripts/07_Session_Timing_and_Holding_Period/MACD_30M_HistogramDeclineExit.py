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
    r"\MACD_30M_HistogramDeclineExit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TIMEFRAME = "30min"

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

YEARS = [
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
# BUILD 30-MIN BARS
# ============================================================

print()
print("=" * 180)
print("BUILDING 30-MIN BARS")
print("=" * 180)

df30 = df1.resample(
    TIMEFRAME,
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

print("30m bars:", len(df30))


# ============================================================
# MACD
# ============================================================

df30["FastEMA"] = df30["Close"].ewm(
    span=MACD_FAST,
    adjust=False
).mean()

df30["SlowEMA"] = df30["Close"].ewm(
    span=MACD_SLOW,
    adjust=False
).mean()

df30["MACD"] = (
    df30["FastEMA"]
    -
    df30["SlowEMA"]
)

df30["SignalLine"] = df30["MACD"].ewm(
    span=MACD_SIGNAL,
    adjust=False
).mean()

df30["Hist"] = (
    df30["MACD"]
    -
    df30["SignalLine"]
)


# Bullish histogram zero-cross
df30["BullCross"] = (
    (df30["Hist"].shift(1) <= 0)
    &
    (df30["Hist"] > 0)
)


# Histogram has declined versus previous completed 30m bar
df30["HistDecline"] = (
    df30["Hist"]
    <
    df30["Hist"].shift(1)
)


# ============================================================
# STAT FUNCTIONS
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

    std = pnl.std(
        ddof=1
    )

    if std == 0:
        return np.nan

    return (
        np.sqrt(
            len(pnl)
        )
        *
        pnl.mean()
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
            np.cumsum(
                pnl
            )
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
            "MaxDrawdown": np.nan,
            "MedianHoldMinutes": np.nan,
            "MeanHoldMinutes": np.nan
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
            ),

        "MedianHoldMinutes":
            d[
                "HoldingMinutes"
            ].median(),

        "MeanHoldMinutes":
            d[
                "HoldingMinutes"
            ].mean()
    }


# ============================================================
# BUILD TRADES
# ============================================================

print()
print("=" * 180)
print("BUILDING HISTOGRAM-DECLINE EXIT TRADES")
print("=" * 180)

trade_rows = []


cross_positions = np.where(
    df30[
        "BullCross"
    ].to_numpy(
        dtype=bool
    )
)[0]


for cross_pos in cross_positions:

    signal_time = df30.index[
        cross_pos
    ]


    if signal_time.year not in YEARS:
        continue


    # ========================================================
    # ENTRY
    #
    # First 1m open after completed bullish-cross 30m bar
    # ========================================================

    entry_pos_1m = df1.index.searchsorted(
        signal_time,
        side="right"
    )

    if entry_pos_1m >= len(df1):
        continue


    entry_time = df1.index[
        entry_pos_1m
    ]

    entry_price = float(
        df1.iloc[
            entry_pos_1m
        ]["Open"]
    )


    # ========================================================
    # FIND FIRST HISTOGRAM DECLINE AFTER CROSS
    #
    # Start at cross_pos + 1.
    #
    # Example:
    #
    # cross hist = +0.5
    # next       = +1.2  -> continue
    # next       = +1.8  -> continue
    # next       = +1.4  -> EXIT
    #
    # ========================================================

    exit_bar_pos = None


    for pos in range(
        cross_pos + 1,
        len(df30)
    ):

        current_hist = float(
            df30.iloc[
                pos
            ]["Hist"]
        )

        previous_hist = float(
            df30.iloc[
                pos - 1
            ]["Hist"]
        )


        if (
            np.isfinite(
                current_hist
            )
            and
            np.isfinite(
                previous_hist
            )
            and
            current_hist
            <
            previous_hist
        ):

            exit_bar_pos = pos
            break


    if exit_bar_pos is None:
        continue


    exit_signal_time = df30.index[
        exit_bar_pos
    ]


    # ========================================================
    # EXECUTION
    #
    # We only know histogram declined after that 30m bar closes.
    # Therefore exit at first available 1m OPEN AFTER that bar.
    # ========================================================

    exit_pos_1m = df1.index.searchsorted(
        exit_signal_time,
        side="right"
    )

    if exit_pos_1m >= len(df1):
        continue


    exit_time = df1.index[
        exit_pos_1m
    ]

    exit_price = float(
        df1.iloc[
            exit_pos_1m
        ]["Open"]
    )


    pnl = (
        exit_price
        -
        entry_price
    )


    hold_minutes = (
        exit_time
        -
        entry_time
    ).total_seconds() / 60.0


    # ========================================================
    # PATH MAE / MFE FOR DIAGNOSTICS
    # ========================================================

    path = df1.loc[
        entry_time:
        exit_time
    ]


    if len(path):

        mfe = (
            path["High"].max()
            -
            entry_price
        )

        mae = (
            entry_price
            -
            path["Low"].min()
        )

    else:

        mfe = np.nan
        mae = np.nan


    trade_rows.append({

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

        "CrossHist":
            float(
                df30.iloc[
                    cross_pos
                ]["Hist"]
            ),

        "ExitSignalTime":
            exit_signal_time,

        "ExitTime":
            exit_time,

        "ExitPrice":
            exit_price,

        "ExitHist":
            float(
                df30.iloc[
                    exit_bar_pos
                ]["Hist"]
            ),

        "PreviousHist":
            float(
                df30.iloc[
                    exit_bar_pos - 1
                ]["Hist"]
            ),

        "BarsHeld30m":
            (
                exit_bar_pos
                -
                cross_pos
            ),

        "HoldingMinutes":
            hold_minutes,

        "PnL":
            pnl,

        "MFE":
            mfe,

        "MAE":
            mae
    })


trades = pd.DataFrame(
    trade_rows
)


print(
    "Completed trades:",
    len(trades)
)


# ============================================================
# OVERALL STATISTICS
# ============================================================

overall_stats = calculate_stats(
    trades
)


# ============================================================
# YEAR BY YEAR
# ============================================================

year_rows = []


for year in YEARS:

    d = trades[
        trades["Year"]
        ==
        year
    ]

    s = calculate_stats(
        d
    )


    year_rows.append({

        "Year":
            year,

        **s
    })


year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# HOLDING-TIME DISTRIBUTION
# ============================================================

hold_rows = []


hold_buckets = [
    (0, 30),
    (30, 60),
    (60, 90),
    (90, 120),
    (120, 180),
    (180, 240),
    (240, 360),
    (360, np.inf)
]


for lo, hi in hold_buckets:

    if np.isinf(
        hi
    ):

        d = trades[
            trades[
                "HoldingMinutes"
            ]
            >=
            lo
        ]

        label = (
            f">={lo}"
        )

    else:

        d = trades[
            (
                trades[
                    "HoldingMinutes"
                ]
                >=
                lo
            )
            &
            (
                trades[
                    "HoldingMinutes"
                ]
                <
                hi
            )
        ]

        label = (
            f"{lo}-{hi}"
        )


    if len(d):

        s = calculate_stats(
            d
        )

        hold_rows.append({

            "HoldBucket":
                label,

            **s
        })


hold_df = pd.DataFrame(
    hold_rows
)


# ============================================================
# BARS HELD DISTRIBUTION
# ============================================================

bars_rows = []


for bars_held in sorted(
    trades[
        "BarsHeld30m"
    ].unique()
):

    d = trades[
        trades[
            "BarsHeld30m"
        ]
        ==
        bars_held
    ]

    s = calculate_stats(
        d
    )


    bars_rows.append({

        "BarsHeld30m":
            bars_held,

        **s
    })


bars_df = pd.DataFrame(
    bars_rows
)


# ============================================================
# SAVE
# ============================================================

trades.to_csv(
    OUTPUT_DIR
    /
    "trade_detail.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "year_by_year.csv",
    index=False
)

hold_df.to_csv(
    OUTPUT_DIR
    /
    "holding_time_distribution.csv",
    index=False
)

bars_df.to_csv(
    OUTPUT_DIR
    /
    "bars_held_distribution.csv",
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
    700
)


print()
print("=" * 220)
print("30M MACD(5,36,5) — EXIT ON FIRST HISTOGRAM DECLINE")
print("=" * 220)

print(
    pd.DataFrame(
        [
            overall_stats
        ]
    )
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
print("HOLDING-TIME DISTRIBUTION")
print("=" * 220)

print(
    hold_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("30M BARS HELD DISTRIBUTION")
print("=" * 220)

print(
    bars_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("PATH DIAGNOSTICS")
print("=" * 220)

print(
    "Mean MFE:",
    round(
        trades[
            "MFE"
        ].mean(),
        4
    )
)

print(
    "Median MFE:",
    round(
        trades[
            "MFE"
        ].median(),
        4
    )
)

print(
    "Mean MAE:",
    round(
        trades[
            "MAE"
        ].mean(),
        4
    )
)

print(
    "Median MAE:",
    round(
        trades[
            "MAE"
        ].median(),
        4
    )
)


print()
print("=" * 220)
print("DONE")
print("=" * 220)

print()
print(
    "Exit rule:"
)

print(
    "First completed 30m bar after bullish cross where Hist[t] < Hist[t-1]."
)

print(
    "Execution: first 1m Open after that completed 30m bar."
)

print()
print("Output folder:")
print(OUTPUT_DIR)