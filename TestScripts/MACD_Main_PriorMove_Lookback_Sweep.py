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
    r"\MACD_Main_PriorMove_Lookback_Sweep"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIXED MAIN STRATEGY SETTINGS
# ============================================================

TIMEFRAME = "15min"

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14

PRICE_MOVE_MIN = 0.35
PRICE_MOVE_MAX = 0.90

PRIOR_MOVE_MIN = -2.0
PRIOR_MOVE_MAX = 2.0

# Sweep only this parameter
PRIOR_LOOKBACKS = [
    10,   # 2.5h
    15,   # 3.75h
    20,   # 5h
    25,   # 6.25h
    30,   # 7.5h
    40    # 10h
]

# Original range filter
RANGE_LOOKBACK = 20
MAX_RANGE_HEIGHT = 25.0
MAX_RANGE_DRIFT = 10.0

# Main session
SESSION_START_HOUR = 14
SESSION_END_HOUR = 18

# Original main hold
HOLD_MINUTES = 210

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
# BUILD 15-MIN BARS
# ============================================================

print()
print("=" * 180)
print("BUILDING 15-MIN BARS")
print("=" * 180)

df15 = df1.resample(
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

df15 = df15.dropna(
    subset=[
        "Open",
        "High",
        "Low",
        "Close"
    ]
)

print("15m bars:", len(df15))


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
# PRICE MOVE
# ============================================================

df15["PriceMoveATR14"] = (
    df15["Close"]
    -
    df15["Close"].shift(1)
) / df15["ATR14"]


# ============================================================
# ORIGINAL RANGE FILTER
# ============================================================

range_high = (
    df15["High"]
    .shift(1)
    .rolling(RANGE_LOOKBACK)
    .max()
)

range_low = (
    df15["Low"]
    .shift(1)
    .rolling(RANGE_LOOKBACK)
    .min()
)

range_height = (
    range_high
    -
    range_low
)

range_drift = (
    df15["Close"].shift(1)
    -
    df15["Close"].shift(RANGE_LOOKBACK)
).abs()

valid_range = (
    (range_height <= MAX_RANGE_HEIGHT)
    &
    (range_drift <= MAX_RANGE_DRIFT)
)

cross_inside_valid_range = (
    valid_range
    &
    (df15["Close"] >= range_low)
    &
    (df15["Close"] <= range_high)
)

df15["PassNoRange"] = (
    ~cross_inside_valid_range
)


# ============================================================
# STATS FUNCTIONS
# ============================================================

def profit_factor(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()

    if gross_loss == 0:

        if gross_profit > 0:
            return np.inf

        return np.nan

    return gross_profit / gross_loss


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


def stats_from_df(d):

    pnl = d["PnL"].dropna().to_numpy(
        dtype=float
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
# BASE SIGNALS
# ============================================================

print()
print("=" * 180)
print("BUILDING BASE MAIN SIGNALS")
print("=" * 180)

base_rows = []

max_needed = max(
    max(PRIOR_LOOKBACKS),
    RANGE_LOOKBACK
)


for signal_time in df15.index[
    df15["BullCross"]
]:

    pos = df15.index.get_loc(
        signal_time
    )

    if pos < max_needed:
        continue


    row = df15.iloc[pos]

    atr14 = float(
        row["ATR14"]
    )

    price_move = float(
        row["PriceMoveATR14"]
    )

    if (
        not np.isfinite(atr14)
        or
        not np.isfinite(price_move)
        or
        atr14 <= 0
    ):
        continue


    # PriceMove fixed filter
    if not (
        PRICE_MOVE_MIN
        <= price_move
        <= PRICE_MOVE_MAX
    ):
        continue


    # Original no-range fixed filter
    if not bool(
        row["PassNoRange"]
    ):
        continue


    # Fixed session
    if not (
        SESSION_START_HOUR
        <= signal_time.hour
        < SESSION_END_HOUR
    ):
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


    # Exit
    target_exit = (
        entry_time
        +
        pd.Timedelta(
            minutes=HOLD_MINUTES
        )
    )

    exit_pos = df1.index.searchsorted(
        target_exit,
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
        target_exit
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


    base_rows.append({

        "SignalTime": signal_time,
        "SignalPos": pos,
        "Year": signal_time.year,
        "EntryTime": entry_time,
        "EntryPrice": entry_price,
        "ExitTime": exit_time,
        "ExitPrice": exit_price,
        "ATR14": atr14,
        "PriceMoveATR14": price_move,
        "PnL": pnl
    })


base = pd.DataFrame(
    base_rows
)

print(
    "Base PriceMove + NoRange + 14-18 trades:",
    len(base)
)


# ============================================================
# PRIOR LOOKBACK SWEEP
# ============================================================

print()
print("=" * 180)
print("PRIOR MOVE LOOKBACK SWEEP")
print("=" * 180)

summary_rows = []
year_rows = []
detail_rows = []


for lookback in PRIOR_LOOKBACKS:

    qualified_rows = []


    for _, trade in base.iterrows():

        pos = int(
            trade["SignalPos"]
        )

        atr14 = float(
            trade["ATR14"]
        )

        prior_move = (
            float(
                df15.iloc[
                    pos - 1
                ]["Close"]
            )
            -
            float(
                df15.iloc[
                    pos - lookback
                ]["Close"]
            )
        ) / atr14


        passes_prior = (
            PRIOR_MOVE_MIN
            <= prior_move
            <= PRIOR_MOVE_MAX
        )


        detail_rows.append({

            "LookbackBars": lookback,
            "LookbackHours": lookback * 0.25,
            "SignalTime": trade["SignalTime"],
            "Year": trade["Year"],
            "PriorMoveATR14": prior_move,
            "PassPriorMove": passes_prior,
            "PnL": trade["PnL"]
        })


        if passes_prior:

            qualified_rows.append(
                trade.to_dict()
            )


    d = pd.DataFrame(
        qualified_rows
    )


    s = stats_from_df(
        d
    )


    year_pfs = []
    profitable_years = 0
    losing_years = 0
    min_year_trades = np.inf


    for year in YEARS:

        yd = d[
            d["Year"]
            ==
            year
        ] if len(d) else pd.DataFrame(
            columns=[
                "PnL"
            ]
        )

        ys = stats_from_df(
            yd
        )

        min_year_trades = min(
            min_year_trades,
            ys["N"]
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


        year_rows.append({

            "LookbackBars": lookback,
            "LookbackHours": lookback * 0.25,
            "Year": year,
            **ys
        })


    summary_rows.append({

        "LookbackBars":
            lookback,

        "LookbackHours":
            lookback * 0.25,

        **s,

        "WorstYearPF":
            min(year_pfs)
            if year_pfs
            else np.nan,

        "BestYearPF":
            max(year_pfs)
            if year_pfs
            else np.nan,

        "ProfitableYears":
            profitable_years,

        "LosingYears":
            losing_years,

        "MinYearTrades":
            (
                min_year_trades
                if min_year_trades != np.inf
                else 0
            ),

        "RetentionPct":
            (
                s["N"]
                /
                len(base)
                *
                100
            )
            if len(base)
            else np.nan
    })


summary_df = pd.DataFrame(
    summary_rows
)

year_df = pd.DataFrame(
    year_rows
)

detail_df = pd.DataFrame(
    detail_rows
)


# ============================================================
# BASELINE WITHOUT PRIOR FILTER
# ============================================================

base_stats = stats_from_df(
    base
)

base_year_rows = []

for year in YEARS:

    yd = base[
        base["Year"]
        ==
        year
    ]

    ys = stats_from_df(
        yd
    )

    base_year_rows.append({

        "Year": year,
        **ys
    })

base_year_df = pd.DataFrame(
    base_year_rows
)


# ============================================================
# YEAR PF MATRIX
# ============================================================

year_pf_matrix = year_df.pivot_table(
    index=[
        "LookbackBars",
        "LookbackHours"
    ],
    columns="Year",
    values="ProfitFactor"
)


# ============================================================
# SAVE
# ============================================================

base.to_csv(
    OUTPUT_DIR
    /
    "base_before_prior_filter.csv",
    index=False
)

summary_df.to_csv(
    OUTPUT_DIR
    /
    "prior_lookback_summary.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "prior_lookback_year_by_year.csv",
    index=False
)

year_pf_matrix.to_csv(
    OUTPUT_DIR
    /
    "prior_lookback_year_pf_matrix.csv"
)

detail_df.to_csv(
    OUTPUT_DIR
    /
    "prior_lookback_trade_detail.csv",
    index=False
)

base_year_df.to_csv(
    OUTPUT_DIR
    /
    "base_without_prior_year_by_year.csv",
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
    600
)


print()
print("=" * 220)
print("BASE WITHOUT PRIOR MOVE FILTER")
print("=" * 220)

print(
    pd.DataFrame(
        [
            base_stats
        ]
    )
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("PRIOR MOVE LOOKBACK SWEEP")
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
    year_pf_matrix
    .round(4)
    .to_string()
)


print()
print("=" * 220)
print("BASE WITHOUT PRIOR FILTER — YEAR BY YEAR")
print("=" * 220)

print(
    base_year_df
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