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
    r"\MACD_Range_ExactBreakoutTiming_Test"
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

MAX_BREAKOUT_BAR = 8

MAIN_YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026
]

GROUPS = {
    "BAR_1_2": (1, 2),
    "BAR_3_4": (3, 4),
    "BAR_4_6": (4, 6),
    "BAR_5_8": (5, 8)
}


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 180)
print("LOADING DATA")
print("=" * 180)

# pyarrow parser is several times faster; fall back if not installed
try:
    df1 = pd.read_csv(ONE_MIN_FILE, engine="pyarrow")
except ImportError:
    df1 = pd.read_csv(ONE_MIN_FILE)

df1["DateTime"] = pd.to_datetime(df1["DateTime"])

df1 = (
    df1
    .sort_values("DateTime")
    .drop_duplicates("DateTime")
    .set_index("DateTime")
)

for col in ["Open", "High", "Low", "Close", "Volume"]:
    df1[col] = pd.to_numeric(
        df1[col],
        errors="coerce"
    )

df1 = df1.dropna(
    subset=["Open", "High", "Low", "Close"]
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
    subset=["Open", "High", "Low", "Close"]
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

    pnl = np.asarray(pnl, dtype=float)

    gp = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()

    if gl == 0:
        if gp > 0:
            return np.inf
        return np.nan

    return gp / gl


def sqn(pnl):

    pnl = np.asarray(pnl, dtype=float)

    if len(pnl) < 2:
        return np.nan

    std = np.std(pnl, ddof=1)

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

    pnl = np.asarray(pnl, dtype=float)

    if len(pnl) == 0:
        return np.nan

    equity = np.concatenate(
        [[0.0], np.cumsum(pnl)]
    )

    peak = np.maximum.accumulate(equity)

    dd = equity - peak

    return abs(dd.min())


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
# BUILD RANGE SETUPS
#
# Vectorized: every candidate cross is evaluated at once with
# numpy arrays instead of row-by-row .iloc lookups.
# ============================================================

print()
print("=" * 180)
print("BUILDING QUALIFIED RANGE SETUPS")
print("=" * 180)

close15 = df15["Close"].to_numpy(dtype=float)
atr15 = df15["ATR14"].to_numpy(dtype=float)

# Range = the RANGE_LOOKBACK bars before the signal bar
range_high_all = (
    df15["High"]
    .rolling(RANGE_LOOKBACK)
    .max()
    .shift(1)
    .to_numpy(dtype=float)
)

range_low_all = (
    df15["Low"]
    .rolling(RANGE_LOOKBACK)
    .min()
    .shift(1)
    .to_numpy(dtype=float)
)

pos = np.flatnonzero(
    df15["BullCross"].to_numpy()
)

pos = pos[
    pos >= max(
        RANGE_LOOKBACK,
        PRIOR_LOOKBACK
    )
]

pos = pos[
    np.isfinite(atr15[pos])
    &
    (atr15[pos] > 0)
]

atr14 = atr15[pos]

close_now = close15[pos]
close_prev = close15[pos - 1]

price_move = (
    close_now
    -
    close_prev
) / atr14

prior_move = (
    close_prev
    -
    close15[pos - PRIOR_LOOKBACK]
) / atr14

range_high = range_high_all[pos]
range_low = range_low_all[pos]

range_height = (
    range_high
    -
    range_low
)

range_drift = np.abs(
    close_prev
    -
    close15[pos - RANGE_LOOKBACK]
)

valid_range = (
    (range_height <= MAX_RANGE_HEIGHT)
    &
    (range_drift <= MAX_RANGE_DRIFT)
)

cross_inside = (
    valid_range
    &
    (range_low <= close_now)
    &
    (close_now <= range_high)
)

keep = (
    (PRICE_MOVE_MIN <= price_move)
    & (price_move <= PRICE_MOVE_MAX)
    & (PRIOR_MOVE_MIN <= prior_move)
    & (prior_move <= PRIOR_MOVE_MAX)
    & cross_inside
)

pos = pos[keep]
signal_times = df15.index[pos]

setups = pd.DataFrame({
    "SignalTime": signal_times,
    "SignalPos": pos,
    "Year": signal_times.year,
    "Month": signal_times.month,
    "Hour": signal_times.hour,
    "ATR14": atr14[keep],
    "PriceMoveATR14": price_move[keep],
    "PriorMoveATR14": prior_move[keep],
    "RangeHigh": range_high[keep],
    "RangeLow": range_low[keep],
    "RangeHeight": range_height[keep],
    "RangeDrift": range_drift[keep],
    "ValidRange": valid_range[keep],
    "CrossInsideRange": cross_inside[keep]
})

print(
    "Qualified crosses inside range:",
    len(setups)
)


# ============================================================
# FIND FIRST BREAKOUT BAR
#
# Important:
# This records the FIRST subsequent 15m close > frozen RangeHigh.
#
# So "ExactBreakoutBar = 4" means:
# bars 1,2,3 did NOT close above RangeHigh,
# but bar 4 DID.
# ============================================================

signal_pos = setups["SignalPos"].to_numpy()
setup_range_high = setups["RangeHigh"].to_numpy(dtype=float)

# Closes of the next MAX_BREAKOUT_BAR bars for each setup.
# Padding with -inf means bars past the end never count as a breakout.
padded_close = np.concatenate([
    close15,
    np.full(MAX_BREAKOUT_BAR, -np.inf)
])

next_closes = np.lib.stride_tricks.sliding_window_view(
    padded_close[1:],
    MAX_BREAKOUT_BAR
)[signal_pos]

broke_out = (
    next_closes
    >
    setup_range_high[:, None]
)

first_breakout_bar = broke_out.argmax(axis=1) + 1

idx = np.flatnonzero(
    broke_out.any(axis=1)
)


# ============================================================
# BUILD TRADES
# ============================================================

n1 = len(df1)

opens1 = df1["Open"].to_numpy(dtype=float)
closes1 = df1["Close"].to_numpy(dtype=float)

breakout_pos = (
    signal_pos[idx]
    +
    first_breakout_bar[idx]
)

entry_pos = df1.index.searchsorted(
    df15.index[breakout_pos],
    side="right"
)

ok = entry_pos < n1
idx, breakout_pos, entry_pos = idx[ok], breakout_pos[ok], entry_pos[ok]

entry_time = df1.index[entry_pos]

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

ok = exit_pos < n1
idx, breakout_pos, entry_pos, exit_pos = idx[ok], breakout_pos[ok], entry_pos[ok], exit_pos[ok]
entry_time, target_time = entry_time[ok], target_time[ok]

exit_time = df1.index[exit_pos]

ok = (
    exit_time
    <=
    target_time
    +
    pd.Timedelta(minutes=2)
)
idx, breakout_pos, entry_pos, exit_pos = idx[ok], breakout_pos[ok], entry_pos[ok], exit_pos[ok]
entry_time, exit_time = entry_time[ok], exit_time[ok]

breakout_close = close15[breakout_pos]
entry_price = opens1[entry_pos]
exit_price = closes1[exit_pos]

results = setups.iloc[idx].reset_index(drop=True)

results["ExactBreakoutBar"] = first_breakout_bar[idx]
results["BreakoutSignalTime"] = df15.index[breakout_pos]
results["BreakoutClose"] = breakout_close
results["BreakoutDistance"] = breakout_close - setup_range_high[idx]
results["EntryTime"] = entry_time
results["EntryPrice"] = entry_price
results["ExitTime"] = exit_time
results["ExitPrice"] = exit_price
results["PnL"] = exit_price - entry_price

print(
    "Setups with breakout within",
    MAX_BREAKOUT_BAR,
    "bars:",
    len(results)
)


# ============================================================
# EXACT BAR SUMMARY
# ============================================================

exact_rows = []


for breakout_bar in range(
    1,
    MAX_BREAKOUT_BAR + 1
):

    d = results[
        results["ExactBreakoutBar"]
        ==
        breakout_bar
    ]

    stats = calculate_stats(
        d
    )

    year_pfs = []
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

            year_pfs.append(
                ys["ProfitFactor"]
            )

            if (
                ys["ProfitFactor"]
                >
                1
            ):
                profitable_years += 1

    exact_rows.append({

        "ExactBreakoutBar":
            breakout_bar,

        **stats,

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


exact_df = pd.DataFrame(
    exact_rows
)


# ============================================================
# EXACT BAR × YEAR
# ============================================================

year_rows = []


for breakout_bar in range(
    1,
    MAX_BREAKOUT_BAR + 1
):

    d = results[
        results["ExactBreakoutBar"]
        ==
        breakout_bar
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

            "ExactBreakoutBar":
                breakout_bar,

            "Year":
                year,

            **stats
        })


year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# GROUPED TIMING WINDOWS
# ============================================================

group_rows = []


for group_name, (
    start_bar,
    end_bar
) in GROUPS.items():

    d = results[
        (
            results["ExactBreakoutBar"]
            >=
            start_bar
        )
        &
        (
            results["ExactBreakoutBar"]
            <=
            end_bar
        )
    ]

    stats = calculate_stats(
        d
    )

    year_pfs = []
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

            year_pfs.append(
                ys["ProfitFactor"]
            )

            if (
                ys["ProfitFactor"]
                >
                1
            ):
                profitable_years += 1

    group_rows.append({

        "Group":
            group_name,

        "StartBar":
            start_bar,

        "EndBar":
            end_bar,

        **stats,

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


group_df = pd.DataFrame(
    group_rows
)


# ============================================================
# GROUP × YEAR
# ============================================================

group_year_rows = []


for group_name, (
    start_bar,
    end_bar
) in GROUPS.items():

    d = results[
        (
            results["ExactBreakoutBar"]
            >=
            start_bar
        )
        &
        (
            results["ExactBreakoutBar"]
            <=
            end_bar
        )
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

        group_year_rows.append({

            "Group":
                group_name,

            "Year":
                year,

            **stats
        })


group_year_df = pd.DataFrame(
    group_year_rows
)


# ============================================================
# PF MATRIX
# ============================================================

pf_matrix = year_df.pivot(
    index="ExactBreakoutBar",
    columns="Year",
    values="ProfitFactor"
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

results.to_csv(
    OUTPUT_DIR
    /
    "exact_breakout_trades.csv",
    index=False
)

exact_df.to_csv(
    OUTPUT_DIR
    /
    "exact_breakout_summary.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "exact_breakout_year_by_year.csv",
    index=False
)

group_df.to_csv(
    OUTPUT_DIR
    /
    "breakout_group_summary.csv",
    index=False
)

group_year_df.to_csv(
    OUTPUT_DIR
    /
    "breakout_group_year_by_year.csv",
    index=False
)

pf_matrix.to_csv(
    OUTPUT_DIR
    /
    "exact_breakout_pf_matrix.csv"
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
print("EXACT BREAKOUT BAR SUMMARY")
print("=" * 220)

print(
    exact_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("EXACT BREAKOUT BAR × YEAR PF")
print("=" * 220)

print(
    pf_matrix
    .round(4)
    .to_string()
)


print()
print("=" * 220)
print("GROUPED BREAKOUT WINDOWS")
print("=" * 220)

print(
    group_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("GROUPED WINDOWS × YEAR")
print("=" * 220)

print(
    group_year_df
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