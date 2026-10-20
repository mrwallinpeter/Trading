# ============================================================
# MACD_15M_ATR_PERCENTILE_EXCLUSION_TEST.py
#
# Tests ATR percentile as an additional filter on:
#
#   Bullish MACD histogram zero cross
#   PriceMoveATR14 = 0.35 ... 0.90
#   PriorMoveATR14 = -2.0 ... +2.0
#
# Main tests:
#
#   1. Exclude lowest ATR percentile:
#      >0, >5, >10, >15 ... >60
#
#   2. 5-percentile buckets:
#      0-5, 5-10, ... 95-100
#
#   3. Broad percentile buckets
#
#   4. ATR percentile lookback robustness:
#      50, 100, 200, 400
#
# Entry:
#   First following 1-minute Open
#
# Exit:
#   210 minutes later using first 1m timestamp >= target
#
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\TestScripts"
    r"\MACD_ATR_Percentile_Test"
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

PRIOR_MOVE_LOOKBACK = 20
PRIOR_MOVE_MIN = -2.0
PRIOR_MOVE_MAX = 2.0

HOLD_MINUTES = 210

START_YEAR = 2022
END_YEAR = 2026

DEFAULT_PERCENTILE_LOOKBACK = 100

PERCENTILE_LOOKBACKS = [
    50,
    100,
    200,
    400,
]


# ============================================================
# LOAD 1-MINUTE DATA
# ============================================================

print("=" * 120)
print("LOADING DATA")
print("=" * 120)

df = pd.read_csv(INPUT_FILE)

# Try to identify datetime column
datetime_col = None

for col in df.columns:
    c = col.lower()

    if (
        "date" in c
        or "time" in c
        or "timestamp" in c
    ):
        datetime_col = col
        break

if datetime_col is None:
    raise ValueError(
        "Could not identify datetime column.\n"
        f"Columns found: {list(df.columns)}"
    )

df[datetime_col] = pd.to_datetime(df[datetime_col])

df = df.set_index(datetime_col)

df = df.sort_index()


# Normalize column names
column_map = {}

for col in df.columns:

    c = col.lower().strip()

    if c == "open":
        column_map[col] = "Open"

    elif c == "high":
        column_map[col] = "High"

    elif c == "low":
        column_map[col] = "Low"

    elif c == "close":
        column_map[col] = "Close"

    elif c == "volume":
        column_map[col] = "Volume"

df = df.rename(columns=column_map)


required_cols = [
    "Open",
    "High",
    "Low",
    "Close",
]

missing = [
    c
    for c in required_cols
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )


print(f"1m rows: {len(df):,}")
print(f"Start: {df.index.min()}")
print(f"End:   {df.index.max()}")


# ============================================================
# BUILD 15-MINUTE BARS
# ============================================================

print()
print("=" * 120)
print("BUILDING 15-MINUTE BARS")
print("=" * 120)

agg = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
}

if "Volume" in df.columns:
    agg["Volume"] = "sum"


bars = (
    df.resample(
        "15min",
        label="right",
        closed="right",
    )
    .agg(agg)
    .dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )
)

print(f"15m bars: {len(bars):,}")


# ============================================================
# MACD
# ============================================================

ema_fast = bars["Close"].ewm(
    span=MACD_FAST,
    adjust=False,
).mean()

ema_slow = bars["Close"].ewm(
    span=MACD_SLOW,
    adjust=False,
).mean()

macd = ema_fast - ema_slow

signal = macd.ewm(
    span=MACD_SIGNAL,
    adjust=False,
).mean()

hist = macd - signal

bars["MACD"] = macd
bars["MACDSignal"] = signal
bars["Hist"] = hist

bars["BullCross"] = (
    (bars["Hist"].shift(1) <= 0)
    &
    (bars["Hist"] > 0)
)


# ============================================================
# WILDER ATR14
# ============================================================

prev_close = bars["Close"].shift(1)

tr = pd.concat(
    [
        bars["High"] - bars["Low"],
        (bars["High"] - prev_close).abs(),
        (bars["Low"] - prev_close).abs(),
    ],
    axis=1,
).max(axis=1)

bars["ATR"] = tr.ewm(
    alpha=1.0 / ATR_LENGTH,
    adjust=False,
).mean()


# ============================================================
# ORIGINAL TWO FILTERS
# ============================================================

bars["PriceMoveATR"] = (
    bars["Close"]
    -
    bars["Close"].shift(1)
) / bars["ATR"]


bars["PriorMoveATR"] = (
    bars["Close"].shift(1)
    -
    bars["Close"].shift(PRIOR_MOVE_LOOKBACK)
) / bars["ATR"]


bars["PriceMovePass"] = (
    (bars["PriceMoveATR"] >= PRICE_MOVE_MIN)
    &
    (bars["PriceMoveATR"] <= PRICE_MOVE_MAX)
)


bars["PriorMovePass"] = (
    (bars["PriorMoveATR"] >= PRIOR_MOVE_MIN)
    &
    (bars["PriorMoveATR"] <= PRIOR_MOVE_MAX)
)


bars["BaseSignal"] = (
    bars["BullCross"]
    &
    bars["PriceMovePass"]
    &
    bars["PriorMovePass"]
)


# ============================================================
# RESEARCH PERIOD
# ============================================================

bars = bars[
    (bars.index.year >= START_YEAR)
    &
    (bars.index.year <= END_YEAR)
].copy()


# ============================================================
# ATR PERCENTILE
#
# Current ATR is compared against ONLY previous observations.
#
# pct = fraction of previous ATR values lower than current ATR.
# ============================================================

def rolling_percentile_rank(series, lookback):

    arr = series.to_numpy(dtype=float)

    result = np.full(
        len(arr),
        np.nan,
        dtype=float,
    )

    for i in range(
        lookback,
        len(arr),
    ):

        current = arr[i]

        previous = arr[
            i - lookback:i
        ]

        valid = previous[
            ~np.isnan(previous)
        ]

        if (
            np.isnan(current)
            or
            len(valid) == 0
        ):
            continue

        result[i] = (
            np.sum(valid < current)
            /
            len(valid)
            *
            100.0
        )

    return pd.Series(
        result,
        index=series.index,
    )


# ============================================================
# TRADE CONSTRUCTION
# ============================================================

def make_trades(
    signal_mask,
    percentile_series,
):

    signal_times = bars.index[
        signal_mask.fillna(False)
    ]

    trades = []

    one_min_index = df.index

    for signal_time in signal_times:

        # --------------------------------------------
        # Entry = first 1m bar AFTER completed 15m bar
        # --------------------------------------------

        entry_pos = one_min_index.searchsorted(
            signal_time,
            side="right",
        )

        if entry_pos >= len(df):
            continue

        entry_time = one_min_index[entry_pos]

        entry_price = df.iloc[
            entry_pos
        ]["Open"]


        # --------------------------------------------
        # Exit target
        # --------------------------------------------

        target_time = (
            entry_time
            +
            pd.Timedelta(
                minutes=HOLD_MINUTES
            )
        )

        exit_pos = one_min_index.searchsorted(
            target_time,
            side="left",
        )

        if exit_pos >= len(df):
            continue

        exit_time = one_min_index[exit_pos]

        # Same tolerance convention used previously
        if (
            exit_time
            >
            target_time
            +
            pd.Timedelta(minutes=2)
        ):
            continue

        exit_price = df.iloc[
            exit_pos
        ]["Close"]


        pnl_points = (
            exit_price -
            entry_price
        )

        trades.append(
            {
                "SignalTime": signal_time,
                "EntryTime": entry_time,
                "ExitTime": exit_time,
                "EntryPrice": entry_price,
                "ExitPrice": exit_price,
                "PnL": pnl_points,
                "ATRPercentile": percentile_series.loc[
                    signal_time
                ],
                "Year": signal_time.year,
            }
        )

    return pd.DataFrame(trades)


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def metrics(trades):

    if len(trades) == 0:
        return {
            "N": 0,
            "WinRate": np.nan,
            "Net": np.nan,
            "Avg": np.nan,
            "PF": np.nan,
            "MaxDD": np.nan,
        }

    pnl = trades["PnL"].to_numpy()

    wins = pnl[
        pnl > 0
    ]

    losses = pnl[
        pnl < 0
    ]

    gross_profit = wins.sum()

    gross_loss = -losses.sum()

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    equity = np.cumsum(pnl)

    running_max = np.maximum.accumulate(
        np.insert(
            equity,
            0,
            0.0,
        )
    )[1:]

    dd = running_max - equity

    max_dd = (
        dd.max()
        if len(dd)
        else 0.0
    )

    return {
        "N": len(trades),
        "WinRate": (
            (pnl > 0).mean()
            *
            100.0
        ),
        "Net": pnl.sum(),
        "Avg": pnl.mean(),
        "PF": pf,
        "MaxDD": max_dd,
    }


# ============================================================
# YEAR METRICS
# ============================================================

def year_metrics(trades):

    rows = []

    for year in range(
        START_YEAR,
        END_YEAR + 1,
    ):

        t = trades[
            trades["Year"] == year
        ]

        m = metrics(t)

        rows.append(
            {
                "Year": year,
                **m,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# RUN ONE PERCENTILE LOOKBACK
# ============================================================

def run_percentile_test(
    lookback,
):

    print()
    print("=" * 120)
    print(
        f"ATR PERCENTILE LOOKBACK = {lookback}"
    )
    print("=" * 120)

    atr_pct = rolling_percentile_rank(
        bars["ATR"],
        lookback,
    )

    bars[f"ATRPercentile_{lookback}"] = atr_pct


    # ========================================================
    # BASE TRADES
    # ========================================================

    base_trades = make_trades(
        bars["BaseSignal"],
        atr_pct,
    )

    base_m = metrics(
        base_trades
    )

    print()
    print("BASE — NO ATR PERCENTILE FILTER")
    print(
        f"N={base_m['N']}  "
        f"PF={base_m['PF']:.4f}  "
        f"WR={base_m['WinRate']:.2f}%  "
        f"Avg={base_m['Avg']:.4f}  "
        f"Net={base_m['Net']:.2f}  "
        f"MaxDD={base_m['MaxDD']:.2f}"
    )


    # ========================================================
    # TEST 1:
    # EXCLUDE LOWEST PERCENTILES
    # ========================================================

    exclusion_thresholds = [
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
    ]

    exclusion_rows = []

    print()
    print("-" * 120)
    print("CUMULATIVE LOW-ATR EXCLUSION")
    print("-" * 120)

    for threshold in exclusion_thresholds:

        mask = (
            bars["BaseSignal"]
            &
            (atr_pct >= threshold)
        )

        trades = make_trades(
            mask,
            atr_pct,
        )

        m = metrics(
            trades
        )

        ym = year_metrics(
            trades
        )

        valid_year_pf = ym[
            "PF"
        ].replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        worst_year_pf = (
            valid_year_pf.min()
            if len(valid_year_pf)
            else np.nan
        )

        positive_years = (
            (ym["PF"] > 1.0)
            .sum()
        )

        row = {
            "ATR_Min_Percentile": threshold,
            **m,
            "WorstYearPF": worst_year_pf,
            "PositiveYears": positive_years,
        }

        exclusion_rows.append(
            row
        )

        print(
            f">={threshold:2d}  "
            f"N={m['N']:4d}  "
            f"PF={m['PF']:7.4f}  "
            f"WR={m['WinRate']:6.2f}%  "
            f"Avg={m['Avg']:8.4f}  "
            f"Net={m['Net']:9.2f}  "
            f"DD={m['MaxDD']:8.2f}  "
            f"WorstPF={worst_year_pf:7.4f}  "
            f"Years>1={positive_years}/5"
        )

    exclusion_df = pd.DataFrame(
        exclusion_rows
    )


    # ========================================================
    # TEST 2:
    # 5-PERCENTILE BUCKETS
    # ========================================================

    bucket_rows = []

    print()
    print("-" * 120)
    print("5-PERCENTILE BUCKETS")
    print("-" * 120)

    for low in range(
        0,
        100,
        5,
    ):

        high = low + 5

        if high < 100:

            mask = (
                bars["BaseSignal"]
                &
                (atr_pct >= low)
                &
                (atr_pct < high)
            )

        else:

            mask = (
                bars["BaseSignal"]
                &
                (atr_pct >= low)
                &
                (atr_pct <= high)
            )

        trades = make_trades(
            mask,
            atr_pct,
        )

        m = metrics(
            trades
        )

        row = {
            "Bucket": f"{low}-{high}",
            "Low": low,
            "High": high,
            **m,
        }

        bucket_rows.append(
            row
        )

        print(
            f"{low:02d}-{high:03d}  "
            f"N={m['N']:4d}  "
            f"PF={m['PF']:7.4f}  "
            f"WR={m['WinRate']:6.2f}%  "
            f"Avg={m['Avg']:8.4f}  "
            f"Net={m['Net']:9.2f}"
        )

    bucket_df = pd.DataFrame(
        bucket_rows
    )


    # ========================================================
    # TEST 3:
    # BROAD REGIMES
    # ========================================================

    broad_ranges = [
        (0, 20),
        (20, 40),
        (40, 60),
        (60, 80),
        (80, 100),

        (10, 100),
        (20, 100),
        (30, 100),
        (40, 100),
        (50, 100),

        (10, 80),
        (20, 80),
        (20, 90),
        (30, 90),
        (30, 80),
        (40, 80),
    ]

    broad_rows = []

    print()
    print("-" * 120)
    print("BROAD ATR PERCENTILE REGIMES")
    print("-" * 120)

    for low, high in broad_ranges:

        mask = (
            bars["BaseSignal"]
            &
            (atr_pct >= low)
            &
            (atr_pct <= high)
        )

        trades = make_trades(
            mask,
            atr_pct,
        )

        m = metrics(
            trades
        )

        ym = year_metrics(
            trades
        )

        valid_pf = ym[
            "PF"
        ].replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        worst_pf = (
            valid_pf.min()
            if len(valid_pf)
            else np.nan
        )

        positive_years = (
            (ym["PF"] > 1.0)
            .sum()
        )

        broad_rows.append(
            {
                "Range": f"{low}-{high}",
                "Low": low,
                "High": high,
                **m,
                "WorstYearPF": worst_pf,
                "PositiveYears": positive_years,
            }
        )

        print(
            f"{low:02d}-{high:03d}  "
            f"N={m['N']:4d}  "
            f"PF={m['PF']:7.4f}  "
            f"Avg={m['Avg']:8.4f}  "
            f"WorstPF={worst_pf:7.4f}  "
            f"Years>1={positive_years}/5"
        )

    broad_df = pd.DataFrame(
        broad_rows
    )


    # ========================================================
    # YEAR-BY-YEAR FOR EACH EXCLUSION THRESHOLD
    # ========================================================

    year_rows = []

    for threshold in exclusion_thresholds:

        mask = (
            bars["BaseSignal"]
            &
            (atr_pct >= threshold)
        )

        trades = make_trades(
            mask,
            atr_pct,
        )

        ym = year_metrics(
            trades
        )

        for _, row in ym.iterrows():

            year_rows.append(
                {
                    "ATR_Min_Percentile": threshold,
                    "Year": int(
                        row["Year"]
                    ),
                    "N": row["N"],
                    "PF": row["PF"],
                    "WinRate": row["WinRate"],
                    "Avg": row["Avg"],
                    "Net": row["Net"],
                    "MaxDD": row["MaxDD"],
                }
            )

    year_df = pd.DataFrame(
        year_rows
    )


    # ========================================================
    # SAVE
    # ========================================================

    exclusion_df.to_csv(
        OUTPUT_DIR /
        f"atr_exclusion_LB{lookback}.csv",
        index=False,
    )

    bucket_df.to_csv(
        OUTPUT_DIR /
        f"atr_5pct_buckets_LB{lookback}.csv",
        index=False,
    )

    broad_df.to_csv(
        OUTPUT_DIR /
        f"atr_broad_ranges_LB{lookback}.csv",
        index=False,
    )

    year_df.to_csv(
        OUTPUT_DIR /
        f"atr_year_by_year_LB{lookback}.csv",
        index=False,
    )

    base_trades.to_csv(
        OUTPUT_DIR /
        f"base_trades_LB{lookback}.csv",
        index=False,
    )

    return {
        "lookback": lookback,
        "base": base_m,
        "exclusion": exclusion_df,
        "buckets": bucket_df,
        "broad": broad_df,
        "year": year_df,
    }


# ============================================================
# RUN ALL LOOKBACKS
# ============================================================

all_results = []

for lookback in PERCENTILE_LOOKBACKS:

    result = run_percentile_test(
        lookback
    )

    all_results.append(
        result
    )


# ============================================================
# LOOKBACK ROBUSTNESS SUMMARY
#
# Rather than choosing best threshold independently for each
# lookback, show all threshold/lookback combinations.
# ============================================================

robustness_rows = []

for result in all_results:

    lookback = result[
        "lookback"
    ]

    exclusion = result[
        "exclusion"
    ]

    for _, row in exclusion.iterrows():

        robustness_rows.append(
            {
                "PercentileLookback": lookback,
                "MinPercentile": row[
                    "ATR_Min_Percentile"
                ],
                "N": row["N"],
                "PF": row["PF"],
                "WinRate": row["WinRate"],
                "Avg": row["Avg"],
                "Net": row["Net"],
                "MaxDD": row["MaxDD"],
                "WorstYearPF": row[
                    "WorstYearPF"
                ],
                "PositiveYears": row[
                    "PositiveYears"
                ],
            }
        )

robustness_df = pd.DataFrame(
    robustness_rows
)

robustness_df.to_csv(
    OUTPUT_DIR /
    "atr_percentile_lookback_robustness.csv",
    index=False,
)


# ============================================================
# PRINT ROBUSTNESS MATRIX
# ============================================================

print()
print("=" * 120)
print("PF ROBUSTNESS MATRIX")
print("=" * 120)

pf_matrix = robustness_df.pivot(
    index="MinPercentile",
    columns="PercentileLookback",
    values="PF",
)

print(
    pf_matrix.round(4).to_string()
)


print()
print("=" * 120)
print("WORST-YEAR PF MATRIX")
print("=" * 120)

worst_matrix = robustness_df.pivot(
    index="MinPercentile",
    columns="PercentileLookback",
    values="WorstYearPF",
)

print(
    worst_matrix.round(4).to_string()
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 120)
print("DONE")
print("=" * 120)

print()
print(f"Output directory:")
print(OUTPUT_DIR)

print()
print(
    "Main thing to inspect first:"
)

print(
    "Does PF / Avg / WorstYearPF improve smoothly as "
    "the lowest ATR percentiles are excluded?"
)

print()
print(
    "A broad plateau is more interesting than one "
    "single optimal cutoff."
)