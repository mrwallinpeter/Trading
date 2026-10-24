# ============================================================
# MACD_ATR_BUCKET_MAE_MFE_60MIN.py
#
# PURPOSE
#
# Study pure MACD zero-crosses with NO other entry filters.
#
# Bullish MACD histogram cross:
#     hist[t-1] <= 0 and hist[t] > 0
#     -> hypothetical LONG
#
# Bearish MACD histogram cross:
#     hist[t-1] >= 0 and hist[t] < 0
#     -> hypothetical SHORT
#
# Entry:
#     First following 1-minute Open after completed 15m signal bar
#
# Measure:
#     MFE / MAE / return at:
#
#       15m
#       30m
#       45m
#       60m
#
# ATR regime:
#     ATR14 percentile relative to previous 100 completed 15m bars
#
# Buckets:
#       0-20
#      20-40
#      40-60
#      60-80
#      80-100
#
# Every signal is studied independently.
# Positions are allowed to overlap because this is a
# signal-behavior study, not yet a trading simulation.
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
    r"\MACD_ATR_MAE_MFE_60MIN"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14

ATR_PERCENTILE_LOOKBACK = 100

HORIZONS = [
    15,
    30,
    45,
    60,
]

ATR_BUCKETS = [
    (0, 20),
    (20, 40),
    (40, 60),
    (60, 80),
    (80, 100),
]

START_YEAR = 2022
END_YEAR = 2026

# Entry/exit lookup tolerance for missing 1m timestamps.
MAX_TIME_SLIPPAGE_MINUTES = 2


# ============================================================
# LOAD 1-MINUTE DATA
# ============================================================

print("=" * 120)
print("LOADING DATA")
print("=" * 120)

df = pd.read_csv(INPUT_FILE)

datetime_col = None

for col in df.columns:
    name = col.lower().strip()

    if (
        "datetime" in name
        or "timestamp" in name
        or "date" in name
        or "time" in name
    ):
        datetime_col = col
        break

if datetime_col is None:
    raise ValueError(
        "Could not identify datetime column.\n"
        f"Columns found: {list(df.columns)}"
    )

df[datetime_col] = pd.to_datetime(df[datetime_col])

df = df.set_index(datetime_col).sort_index()


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

rename_map = {}

for col in df.columns:
    name = col.lower().strip()

    if name == "open":
        rename_map[col] = "Open"

    elif name == "high":
        rename_map[col] = "High"

    elif name == "low":
        rename_map[col] = "Low"

    elif name == "close":
        rename_map[col] = "Close"

    elif name == "volume":
        rename_map[col] = "Volume"

df = df.rename(columns=rename_map)

required = [
    "Open",
    "High",
    "Low",
    "Close",
]

missing = [
    col
    for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available: {list(df.columns)}"
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

macd_line = ema_fast - ema_slow

signal_line = macd_line.ewm(
    span=MACD_SIGNAL,
    adjust=False,
).mean()

hist = macd_line - signal_line

bars["MACD"] = macd_line
bars["Signal"] = signal_line
bars["Hist"] = hist

bars["BullCross"] = (
    (bars["Hist"].shift(1) <= 0)
    &
    (bars["Hist"] > 0)
)

bars["BearCross"] = (
    (bars["Hist"].shift(1) >= 0)
    &
    (bars["Hist"] < 0)
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

bars["ATR14"] = tr.ewm(
    alpha=1.0 / ATR_LENGTH,
    adjust=False,
).mean()


# ============================================================
# ATR PERCENTILE
#
# Current ATR is compared ONLY with the PREVIOUS N ATR values.
#
# 80 means:
# Current ATR is greater than about 80% of the previous
# 100 ATR observations.
# ============================================================

def rolling_percentile_rank(series, lookback):

    values = series.to_numpy(dtype=float)

    result = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    for i in range(lookback, len(values)):

        current = values[i]

        previous = values[
            i - lookback:i
        ]

        previous = previous[
            ~np.isnan(previous)
        ]

        if np.isnan(current):
            continue

        if len(previous) == 0:
            continue

        result[i] = (
            np.sum(previous < current)
            /
            len(previous)
            *
            100.0
        )

    return pd.Series(
        result,
        index=series.index,
    )


bars["ATRPercentile"] = rolling_percentile_rank(
    bars["ATR14"],
    ATR_PERCENTILE_LOOKBACK,
)


# ============================================================
# ATR BUCKET
# ============================================================

def get_atr_bucket(percentile):

    if pd.isna(percentile):
        return None

    if percentile < 20:
        return "00-20"

    if percentile < 40:
        return "20-40"

    if percentile < 60:
        return "40-60"

    if percentile < 80:
        return "60-80"

    return "80-100"


bars["ATRBucket"] = bars["ATRPercentile"].apply(
    get_atr_bucket
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
# BUILD SIGNAL TABLE
# ============================================================

bull_signals = bars[
    bars["BullCross"]
].copy()

bull_signals["Direction"] = "LONG"


bear_signals = bars[
    bars["BearCross"]
].copy()

bear_signals["Direction"] = "SHORT"


signals = pd.concat(
    [
        bull_signals,
        bear_signals,
    ]
).sort_index()


signals = signals[
    signals["ATRPercentile"].notna()
].copy()


print()
print("=" * 120)
print("SIGNALS")
print("=" * 120)

print(
    f"Bullish crosses: "
    f"{(signals['Direction'] == 'LONG').sum():,}"
)

print(
    f"Bearish crosses: "
    f"{(signals['Direction'] == 'SHORT').sum():,}"
)

print(
    f"Total crosses:   "
    f"{len(signals):,}"
)


# ============================================================
# 1-MINUTE ARRAYS FOR FAST ACCESS
# ============================================================

one_min_index = df.index

open_array = df["Open"].to_numpy(dtype=float)
high_array = df["High"].to_numpy(dtype=float)
low_array = df["Low"].to_numpy(dtype=float)
close_array = df["Close"].to_numpy(dtype=float)


# ============================================================
# PROCESS EACH SIGNAL
# ============================================================

records = []


for signal_time, row in signals.iterrows():

    direction = row["Direction"]


    # --------------------------------------------------------
    # ENTRY
    #
    # First 1-minute bar strictly AFTER completed signal bar.
    # --------------------------------------------------------

    entry_pos = one_min_index.searchsorted(
        signal_time,
        side="right",
    )

    if entry_pos >= len(df):
        continue

    entry_time = one_min_index[entry_pos]

    entry_price = open_array[entry_pos]


    # --------------------------------------------------------
    # Require complete 60-minute path
    # --------------------------------------------------------

    final_target = (
        entry_time
        +
        pd.Timedelta(minutes=60)
    )

    final_pos = one_min_index.searchsorted(
        final_target,
        side="left",
    )

    if final_pos >= len(df):
        continue

    final_time = one_min_index[final_pos]

    if (
        final_time
        >
        final_target
        +
        pd.Timedelta(
            minutes=MAX_TIME_SLIPPAGE_MINUTES
        )
    ):
        continue


    result = {
        "SignalTime": signal_time,
        "Direction": direction,
        "EntryTime": entry_time,
        "EntryPrice": entry_price,
        "ATR14": row["ATR14"],
        "ATRPercentile": row["ATRPercentile"],
        "ATRBucket": row["ATRBucket"],
        "Year": signal_time.year,
    }


    # ========================================================
    # CUMULATIVE MFE / MAE AT EACH HORIZON
    # ========================================================

    for horizon in HORIZONS:

        target_time = (
            entry_time
            +
            pd.Timedelta(
                minutes=horizon
            )
        )

        end_pos = one_min_index.searchsorted(
            target_time,
            side="left",
        )

        if end_pos >= len(df):
            continue

        actual_end_time = one_min_index[end_pos]

        if (
            actual_end_time
            >
            target_time
            +
            pd.Timedelta(
                minutes=MAX_TIME_SLIPPAGE_MINUTES
            )
        ):
            continue


        # Include entry bar through horizon bar.
        path_high = np.max(
            high_array[
                entry_pos:end_pos + 1
            ]
        )

        path_low = np.min(
            low_array[
                entry_pos:end_pos + 1
            ]
        )

        exit_price = close_array[end_pos]


        if direction == "LONG":

            mfe = (
                path_high -
                entry_price
            )

            mae = (
                entry_price -
                path_low
            )

            ret = (
                exit_price -
                entry_price
            )

        else:

            mfe = (
                entry_price -
                path_low
            )

            mae = (
                path_high -
                entry_price
            )

            ret = (
                entry_price -
                exit_price
            )


        result[f"MFE_{horizon}m"] = mfe
        result[f"MAE_{horizon}m"] = mae
        result[f"Return_{horizon}m"] = ret


    # ========================================================
    # INDIVIDUAL 15-MINUTE BLOCK EXCURSIONS
    #
    # Block 1 = 0-15m
    # Block 2 = 15-30m
    # Block 3 = 30-45m
    # Block 4 = 45-60m
    #
    # Measured relative to ORIGINAL ENTRY PRICE.
    # ========================================================

    block_start_minute = 0

    for block_end_minute in HORIZONS:

        block_start_time = (
            entry_time
            +
            pd.Timedelta(
                minutes=block_start_minute
            )
        )

        block_end_time = (
            entry_time
            +
            pd.Timedelta(
                minutes=block_end_minute
            )
        )

        start_pos = one_min_index.searchsorted(
            block_start_time,
            side="left",
        )

        end_pos = one_min_index.searchsorted(
            block_end_time,
            side="left",
        )

        if (
            start_pos >= len(df)
            or
            end_pos >= len(df)
        ):
            continue

        block_high = np.max(
            high_array[
                start_pos:end_pos + 1
            ]
        )

        block_low = np.min(
            low_array[
                start_pos:end_pos + 1
            ]
        )


        if direction == "LONG":

            block_mfe = max(
                0.0,
                block_high -
                entry_price,
            )

            block_mae = max(
                0.0,
                entry_price -
                block_low,
            )

        else:

            block_mfe = max(
                0.0,
                entry_price -
                block_low,
            )

            block_mae = max(
                0.0,
                block_high -
                entry_price,
            )


        block_number = (
            block_end_minute // 15
        )

        result[
            f"Block{block_number}_MFE"
        ] = block_mfe

        result[
            f"Block{block_number}_MAE"
        ] = block_mae

        block_start_minute = block_end_minute


    records.append(result)


trades = pd.DataFrame(records)


# ============================================================
# SAVE RAW SIGNAL STUDY
# ============================================================

trades.to_csv(
    OUTPUT_DIR /
    "macd_cross_mae_mfe_raw.csv",
    index=False,
)


# ============================================================
# SUMMARY FUNCTION
# ============================================================

def summarize_group(group):

    result = {
        "N": len(group),
    }

    for horizon in HORIZONS:

        mfe_col = f"MFE_{horizon}m"
        mae_col = f"MAE_{horizon}m"
        ret_col = f"Return_{horizon}m"

        result[
            f"AvgMFE_{horizon}m"
        ] = group[mfe_col].mean()

        result[
            f"MedianMFE_{horizon}m"
        ] = group[mfe_col].median()

        result[
            f"AvgMAE_{horizon}m"
        ] = group[mae_col].mean()

        result[
            f"MedianMAE_{horizon}m"
        ] = group[mae_col].median()

        result[
            f"AvgReturn_{horizon}m"
        ] = group[ret_col].mean()

        result[
            f"WinPct_{horizon}m"
        ] = (
            group[ret_col] > 0
        ).mean() * 100.0


        # MFE / MAE ratio
        avg_mae = group[
            mae_col
        ].mean()

        avg_mfe = group[
            mfe_col
        ].mean()

        result[
            f"MFE_MAE_Ratio_{horizon}m"
        ] = (
            avg_mfe / avg_mae
            if avg_mae > 0
            else np.nan
        )

    return pd.Series(result)


# ============================================================
# ATR BUCKET SUMMARY — COMBINED
#
# Long and short returns are direction-adjusted,
# therefore positive means movement in direction of cross.
# ============================================================

combined_summary = (
    trades.groupby(
        "ATRBucket",
        sort=True,
    )
    .apply(
        summarize_group,
        include_groups=False,
    )
    .reset_index()
)

combined_summary.to_csv(
    OUTPUT_DIR /
    "atr_bucket_summary_combined.csv",
    index=False,
)


# ============================================================
# ATR BUCKET SUMMARY — LONG / SHORT SEPARATELY
# ============================================================

direction_summary = (
    trades.groupby(
        [
            "Direction",
            "ATRBucket",
        ],
        sort=True,
    )
    .apply(
        summarize_group,
        include_groups=False,
    )
    .reset_index()
)

direction_summary.to_csv(
    OUTPUT_DIR /
    "atr_bucket_summary_by_direction.csv",
    index=False,
)


# ============================================================
# YEAR-BY-YEAR ATR BUCKET SUMMARY
# ============================================================

year_summary = (
    trades.groupby(
        [
            "Year",
            "Direction",
            "ATRBucket",
        ],
        sort=True,
    )
    .apply(
        summarize_group,
        include_groups=False,
    )
    .reset_index()
)

year_summary.to_csv(
    OUTPUT_DIR /
    "atr_bucket_summary_by_year.csv",
    index=False,
)


# ============================================================
# 15-MINUTE BLOCK SUMMARY
# ============================================================

block_rows = []

for bucket in sorted(
    trades["ATRBucket"].dropna().unique()
):

    bucket_df = trades[
        trades["ATRBucket"] == bucket
    ]

    for direction in [
        "LONG",
        "SHORT",
        "COMBINED",
    ]:

        if direction == "COMBINED":
            g = bucket_df
        else:
            g = bucket_df[
                bucket_df["Direction"] == direction
            ]

        if len(g) == 0:
            continue

        for block in range(1, 5):

            block_rows.append(
                {
                    "ATRBucket": bucket,
                    "Direction": direction,
                    "Block": block,
                    "Minutes": (
                        f"{(block - 1) * 15}-"
                        f"{block * 15}"
                    ),
                    "N": len(g),
                    "AvgMFE": g[
                        f"Block{block}_MFE"
                    ].mean(),
                    "MedianMFE": g[
                        f"Block{block}_MFE"
                    ].median(),
                    "AvgMAE": g[
                        f"Block{block}_MAE"
                    ].mean(),
                    "MedianMAE": g[
                        f"Block{block}_MAE"
                    ].median(),
                }
            )


block_summary = pd.DataFrame(
    block_rows
)

block_summary.to_csv(
    OUTPUT_DIR /
    "atr_bucket_15min_blocks.csv",
    index=False,
)


# ============================================================
# PRINT MAIN RESULT
# ============================================================

print()
print("=" * 120)
print("ATR BUCKET SUMMARY — ALL MACD CROSSES")
print("=" * 120)

display_cols = [
    "ATRBucket",
    "N",
    "AvgMFE_15m",
    "AvgMAE_15m",
    "AvgReturn_15m",
    "AvgMFE_30m",
    "AvgMAE_30m",
    "AvgReturn_30m",
    "AvgMFE_45m",
    "AvgMAE_45m",
    "AvgReturn_45m",
    "AvgMFE_60m",
    "AvgMAE_60m",
    "AvgReturn_60m",
    "MFE_MAE_Ratio_60m",
]

print(
    combined_summary[
        display_cols
    ]
    .round(3)
    .to_string(index=False)
)


# ============================================================
# PRINT LONG / SHORT AT 60 MIN
# ============================================================

print()
print("=" * 120)
print("60-MINUTE RESULT — LONG VS SHORT")
print("=" * 120)

direction_cols = [
    "Direction",
    "ATRBucket",
    "N",
    "AvgMFE_60m",
    "AvgMAE_60m",
    "AvgReturn_60m",
    "WinPct_60m",
    "MFE_MAE_Ratio_60m",
]

print(
    direction_summary[
        direction_cols
    ]
    .round(3)
    .to_string(index=False)
)


# ============================================================
# SIMPLE BUCKET RANKING
#
# High AvgReturn + MFE/MAE > 1 is desirable.
# This is descriptive only — not an optimized rule.
# ============================================================

ranking = combined_summary[
    [
        "ATRBucket",
        "N",
        "AvgMFE_60m",
        "AvgMAE_60m",
        "AvgReturn_60m",
        "WinPct_60m",
        "MFE_MAE_Ratio_60m",
    ]
].copy()

ranking = ranking.sort_values(
    "AvgReturn_60m",
    ascending=False,
)

print()
print("=" * 120)
print("ATR BUCKET RANKING BY 60-MINUTE AVERAGE RETURN")
print("=" * 120)

print(
    ranking
    .round(3)
    .to_string(index=False)
)


ranking.to_csv(
    OUTPUT_DIR /
    "atr_bucket_60min_ranking.csv",
    index=False,
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 120)
print("DONE")
print("=" * 120)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Interpretation:")
print(
    "Good ATR buckets should ideally show:"
)
print(
    "  - positive direction-adjusted AvgReturn"
)
print(
    "  - AvgMFE clearly larger than AvgMAE"
)
print(
    "  - similar behavior at 15/30/45/60 minutes"
)
print(
    "  - reasonable sample size"
)
print(
    "  - similar results across years and LONG/SHORT"
)