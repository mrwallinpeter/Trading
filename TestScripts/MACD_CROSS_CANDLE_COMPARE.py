import os
import numpy as np
import pandas as pd

# ==================================================================================================
# SETTINGS
# ==================================================================================================

CSV_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "MACD_CROSS_CANDLE_COMPARE"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14

START_YEAR = 2022
END_YEAR = 2026

HORIZONS = [15, 30, 45, 60]

# Old filter
OLD_PRICE_MOVE_MIN = 0.35
OLD_PRICE_MOVE_MAX = 0.90


# ==================================================================================================
# HELPERS
# ==================================================================================================

def section(title):
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def wilder_atr(df, length=14):
    prev_close = df["Close"].shift(1)

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(
        alpha=1.0 / length,
        adjust=False
    ).mean()

    return atr


def summarize(df):
    if len(df) == 0:
        return {
            "N": 0
        }

    out = {
        "N": len(df)
    }

    for h in HORIZONS:
        mfe = df[f"MFE_{h}m"]
        mae = df[f"MAE_{h}m"]
        ret = df[f"Return_{h}m"]

        out[f"AvgMFE_{h}m"] = mfe.mean()
        out[f"AvgMAE_{h}m"] = mae.mean()
        out[f"AvgReturn_{h}m"] = ret.mean()
        out[f"WinPct_{h}m"] = (ret > 0).mean() * 100

        avg_mae = mae.mean()

        if avg_mae != 0:
            out[f"MFE_MAE_Ratio_{h}m"] = mfe.mean() / avg_mae
        else:
            out[f"MFE_MAE_Ratio_{h}m"] = np.nan

    return out


def make_summary(df, group_col):
    rows = []

    for name, g in df.groupby(group_col, observed=True):
        row = {
            group_col: name
        }

        row.update(summarize(g))
        rows.append(row)

    return pd.DataFrame(rows)


# ==================================================================================================
# LOAD 1-MINUTE DATA
# ==================================================================================================

section("LOADING DATA")

df1 = pd.read_csv(CSV_PATH)

# Find timestamp column
possible_time_cols = [
    "DateTime",
    "Datetime",
    "datetime",
    "Timestamp",
    "timestamp",
    "Time",
    "time",
    "Date",
    "date"
]

time_col = None

for c in possible_time_cols:
    if c in df1.columns:
        time_col = c
        break

if time_col is None:
    raise ValueError(
        "Could not find timestamp column. Columns found:\n"
        + str(df1.columns.tolist())
    )

df1[time_col] = pd.to_datetime(df1[time_col])
df1 = df1.set_index(time_col)
df1 = df1.sort_index()

# Normalize OHLC names
rename_map = {}

for c in df1.columns:
    lc = c.lower()

    if lc == "open":
        rename_map[c] = "Open"
    elif lc == "high":
        rename_map[c] = "High"
    elif lc == "low":
        rename_map[c] = "Low"
    elif lc == "close":
        rename_map[c] = "Close"
    elif lc == "volume":
        rename_map[c] = "Volume"

df1 = df1.rename(columns=rename_map)

required_cols = ["Open", "High", "Low", "Close"]

for c in required_cols:
    if c not in df1.columns:
        raise ValueError(f"Missing required column: {c}")

print(f"1m rows: {len(df1):,}")
print(f"Start: {df1.index.min()}")
print(f"End:   {df1.index.max()}")


# ==================================================================================================
# BUILD 15-MINUTE BARS
# ==================================================================================================

section("BUILDING 15-MINUTE BARS")

agg = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last"
}

if "Volume" in df1.columns:
    agg["Volume"] = "sum"

df15 = df1.resample(
    "15min",
    label="right",
    closed="right"
).agg(agg)

df15 = df15.dropna(
    subset=["Open", "High", "Low", "Close"]
)

print(f"15m bars: {len(df15):,}")


# ==================================================================================================
# MACD
# ==================================================================================================

ema_fast = df15["Close"].ewm(
    span=MACD_FAST,
    adjust=False
).mean()

ema_slow = df15["Close"].ewm(
    span=MACD_SLOW,
    adjust=False
).mean()

macd = ema_fast - ema_slow

signal_line = macd.ewm(
    span=MACD_SIGNAL,
    adjust=False
).mean()

hist = macd - signal_line

df15["MACDHist"] = hist

bull_cross = (
    (df15["MACDHist"].shift(1) <= 0) &
    (df15["MACDHist"] > 0)
)

bear_cross = (
    (df15["MACDHist"].shift(1) >= 0) &
    (df15["MACDHist"] < 0)
)

df15["BullCross"] = bull_cross
df15["BearCross"] = bear_cross


# ==================================================================================================
# ATR
# ==================================================================================================

df15["ATR14"] = wilder_atr(
    df15,
    ATR_LENGTH
)


# ==================================================================================================
# CROSS-CANDLE FEATURES
# ==================================================================================================

# Direction:
# +1 bullish MACD cross
# -1 bearish MACD cross

df15["DirectionValue"] = 0

df15.loc[df15["BullCross"], "DirectionValue"] = 1
df15.loc[df15["BearCross"], "DirectionValue"] = -1


# --------------------------------------------------------------------------------------------------
# OLD PRICE MOVE FILTER
#
# Original bullish concept:
#
#     (Close_t - Close_t-1) / ATR
#
# For shorts we mirror it so positive always means movement in signal direction.
# --------------------------------------------------------------------------------------------------

df15["PriceMoveATR"] = (
    df15["DirectionValue"] *
    (df15["Close"] - df15["Close"].shift(1)) /
    df15["ATR14"]
)


# --------------------------------------------------------------------------------------------------
# CROSS-CANDLE BODY
#
# Positive = body points in MACD-cross direction
# Negative = body points against MACD-cross direction
# --------------------------------------------------------------------------------------------------

df15["DirectionalBodyATR"] = (
    df15["DirectionValue"] *
    (df15["Close"] - df15["Open"]) /
    df15["ATR14"]
)


# Absolute body size
df15["BodyATR"] = (
    (df15["Close"] - df15["Open"]).abs() /
    df15["ATR14"]
)


# --------------------------------------------------------------------------------------------------
# CLOSE LOCATION WITHIN CROSS CANDLE
#
# Always direction-adjusted:
#
# Bull:
#       1.0 = close at high
#       0.0 = close at low
#
# Bear:
#       1.0 = close at low
#       0.0 = close at high
# --------------------------------------------------------------------------------------------------

candle_range = df15["High"] - df15["Low"]

bull_close_location = (
    (df15["Close"] - df15["Low"]) /
    candle_range
)

bear_close_location = (
    (df15["High"] - df15["Close"]) /
    candle_range
)

df15["DirectionalCloseLocation"] = np.where(
    df15["DirectionValue"] == 1,
    bull_close_location,
    bear_close_location
)

df15.loc[candle_range == 0, "DirectionalCloseLocation"] = 0.5


# --------------------------------------------------------------------------------------------------
# WICKS
#
# Direction-adjusted:
#
# For longs:
#   favourable wick = upper wick
#   adverse wick    = lower wick
#
# For shorts:
#   favourable wick = lower wick
#   adverse wick    = upper wick
# --------------------------------------------------------------------------------------------------

body_high = df15[["Open", "Close"]].max(axis=1)
body_low = df15[["Open", "Close"]].min(axis=1)

upper_wick = df15["High"] - body_high
lower_wick = body_low - df15["Low"]

df15["FavourableWickATR"] = np.where(
    df15["DirectionValue"] == 1,
    upper_wick / df15["ATR14"],
    lower_wick / df15["ATR14"]
)

df15["AdverseWickATR"] = np.where(
    df15["DirectionValue"] == 1,
    lower_wick / df15["ATR14"],
    upper_wick / df15["ATR14"]
)


# ==================================================================================================
# CREATE SIGNAL TABLE
# ==================================================================================================

signals = df15[
    df15["BullCross"] |
    df15["BearCross"]
].copy()

signals = signals[
    (signals.index.year >= START_YEAR) &
    (signals.index.year <= END_YEAR)
].copy()

signals["Direction"] = np.where(
    signals["BullCross"],
    "LONG",
    "SHORT"
)

signals["Year"] = signals.index.year

signals["OldPriceMovePass"] = (
    (signals["PriceMoveATR"] >= OLD_PRICE_MOVE_MIN) &
    (signals["PriceMoveATR"] <= OLD_PRICE_MOVE_MAX)
)


section("SIGNALS")

print(f"Bullish crosses: {(signals['Direction'] == 'LONG').sum():,}")
print(f"Bearish crosses: {(signals['Direction'] == 'SHORT').sum():,}")
print(f"Total crosses:   {len(signals):,}")

print()
print(
    f"Old PriceMoveATR {OLD_PRICE_MOVE_MIN:.2f}-{OLD_PRICE_MOVE_MAX:.2f}: "
    f"{signals['OldPriceMovePass'].sum():,}"
)


# ==================================================================================================
# 1-MINUTE PATH ANALYSIS
# ==================================================================================================

section("CALCULATING POST-CROSS PATH")

times_1m = df1.index.values

opens_1m = df1["Open"].to_numpy()
highs_1m = df1["High"].to_numpy()
lows_1m = df1["Low"].to_numpy()
closes_1m = df1["Close"].to_numpy()

rows = []

for signal_time, s in signals.iterrows():

    # First 1-minute bar strictly AFTER completed 15m signal bar
    entry_pos = np.searchsorted(
        times_1m,
        np.datetime64(signal_time),
        side="right"
    )

    if entry_pos >= len(df1):
        continue

    entry_time = df1.index[entry_pos]
    entry_price = opens_1m[entry_pos]

    direction = int(s["DirectionValue"])

    row = {
        "SignalTime": signal_time,
        "EntryTime": entry_time,
        "Direction": s["Direction"],
        "DirectionValue": direction,
        "Year": s["Year"],

        "EntryPrice": entry_price,

        "ATR14": s["ATR14"],

        "PriceMoveATR": s["PriceMoveATR"],
        "OldPriceMovePass": s["OldPriceMovePass"],

        "DirectionalBodyATR": s["DirectionalBodyATR"],
        "BodyATR": s["BodyATR"],

        "DirectionalCloseLocation": s["DirectionalCloseLocation"],

        "FavourableWickATR": s["FavourableWickATR"],
        "AdverseWickATR": s["AdverseWickATR"]
    }

    valid_trade = True

    for h in HORIZONS:

        target_time = entry_time + pd.Timedelta(minutes=h)

        end_pos = np.searchsorted(
            times_1m,
            np.datetime64(target_time),
            side="left"
        )

        if end_pos >= len(df1):
            valid_trade = False
            break

        actual_end_time = df1.index[end_pos]

        # Require target bar reasonably close
        if actual_end_time > target_time + pd.Timedelta(minutes=2):
            valid_trade = False
            break

        path_high = highs_1m[entry_pos:end_pos + 1].max()
        path_low = lows_1m[entry_pos:end_pos + 1].min()

        exit_price = closes_1m[end_pos]

        if direction == 1:
            mfe = path_high - entry_price
            mae = entry_price - path_low
            ret = exit_price - entry_price

        else:
            mfe = entry_price - path_low
            mae = path_high - entry_price
            ret = entry_price - exit_price

        row[f"MFE_{h}m"] = mfe
        row[f"MAE_{h}m"] = mae
        row[f"Return_{h}m"] = ret

    if valid_trade:
        rows.append(row)


results = pd.DataFrame(rows)

print(f"Signals with valid 60-minute path: {len(results):,}")


# ==================================================================================================
# CATEGORICAL BUCKETS
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# Directional body
#
# Negative = candle body against the MACD direction.
# --------------------------------------------------------------------------------------------------

body_bins = [
    -np.inf,
    0.0,
    0.10,
    0.25,
    0.50,
    np.inf
]

body_labels = [
    "Against direction",
    "00-0.10",
    "0.10-0.25",
    "0.25-0.50",
    "0.50+"
]

results["BodyBucket"] = pd.cut(
    results["DirectionalBodyATR"],
    bins=body_bins,
    labels=body_labels,
    right=False
)


# --------------------------------------------------------------------------------------------------
# Close location
# --------------------------------------------------------------------------------------------------

close_bins = [
    0.0,
    0.20,
    0.40,
    0.60,
    0.80,
    1.000001
]

close_labels = [
    "00-20%",
    "20-40%",
    "40-60%",
    "60-80%",
    "80-100%"
]

results["CloseLocationBucket"] = pd.cut(
    results["DirectionalCloseLocation"],
    bins=close_bins,
    labels=close_labels,
    include_lowest=True,
    right=False
)


# --------------------------------------------------------------------------------------------------
# PriceMoveATR buckets
#
# This lets us see whether the old 0.35-0.90 range sits inside a broad plateau.
# --------------------------------------------------------------------------------------------------

price_move_bins = [
    -np.inf,
    0.0,
    0.20,
    0.35,
    0.50,
    0.70,
    0.90,
    1.20,
    np.inf
]

price_move_labels = [
    "<0",
    "0-0.20",
    "0.20-0.35",
    "0.35-0.50",
    "0.50-0.70",
    "0.70-0.90",
    "0.90-1.20",
    "1.20+"
]

results["PriceMoveBucket"] = pd.cut(
    results["PriceMoveATR"],
    bins=price_move_bins,
    labels=price_move_labels,
    right=False
)


# ==================================================================================================
# MAIN COMPARISON
# ==================================================================================================

comparison_rows = []


def add_comparison(name, mask):

    g = results[mask]

    row = {
        "Test": name
    }

    row.update(summarize(g))

    comparison_rows.append(row)


add_comparison(
    "ALL MACD CROSSES",
    pd.Series(True, index=results.index)
)

add_comparison(
    "OLD PriceMoveATR 0.35-0.90",
    results["OldPriceMovePass"]
)

add_comparison(
    "Cross candle body SAME direction",
    results["DirectionalBodyATR"] > 0
)

add_comparison(
    "Body >= 0.10 ATR",
    results["DirectionalBodyATR"] >= 0.10
)

add_comparison(
    "Body >= 0.25 ATR",
    results["DirectionalBodyATR"] >= 0.25
)

add_comparison(
    "Close in directional top 40%",
    results["DirectionalCloseLocation"] >= 0.60
)

add_comparison(
    "Close in directional top 20%",
    results["DirectionalCloseLocation"] >= 0.80
)

add_comparison(
    "Body >=0.10 + Close >=60%",
    (
        (results["DirectionalBodyATR"] >= 0.10) &
        (results["DirectionalCloseLocation"] >= 0.60)
    )
)

add_comparison(
    "Body >=0.25 + Close >=60%",
    (
        (results["DirectionalBodyATR"] >= 0.25) &
        (results["DirectionalCloseLocation"] >= 0.60)
    )
)

add_comparison(
    "OLD filter + Close >=60%",
    (
        results["OldPriceMovePass"] &
        (results["DirectionalCloseLocation"] >= 0.60)
    )
)

comparison = pd.DataFrame(comparison_rows)


# ==================================================================================================
# BUCKET SUMMARIES
# ==================================================================================================

body_summary = make_summary(
    results,
    "BodyBucket"
)

close_summary = make_summary(
    results,
    "CloseLocationBucket"
)

price_move_summary = make_summary(
    results,
    "PriceMoveBucket"
)


# ==================================================================================================
# LONG / SHORT COMPARISON
# ==================================================================================================

direction_rows = []

tests = {
    "ALL": pd.Series(True, index=results.index),

    "OLD PriceMove 0.35-0.90":
        results["OldPriceMovePass"],

    "Body >=0.10":
        results["DirectionalBodyATR"] >= 0.10,

    "Body >=0.25":
        results["DirectionalBodyATR"] >= 0.25,

    "Close >=60%":
        results["DirectionalCloseLocation"] >= 0.60,

    "Body >=0.10 + Close >=60%":
        (
            (results["DirectionalBodyATR"] >= 0.10) &
            (results["DirectionalCloseLocation"] >= 0.60)
        )
}

for test_name, mask in tests.items():

    for direction in ["LONG", "SHORT"]:

        g = results[
            mask &
            (results["Direction"] == direction)
        ]

        row = {
            "Test": test_name,
            "Direction": direction
        }

        row.update(summarize(g))

        direction_rows.append(row)

direction_summary = pd.DataFrame(direction_rows)


# ==================================================================================================
# YEAR STABILITY
# ==================================================================================================

year_rows = []

for test_name, mask in tests.items():

    for year in sorted(results["Year"].unique()):

        g = results[
            mask &
            (results["Year"] == year)
        ]

        row = {
            "Test": test_name,
            "Year": year
        }

        row.update(summarize(g))

        year_rows.append(row)

year_summary = pd.DataFrame(year_rows)


# ==================================================================================================
# BODY x CLOSE LOCATION CROSS-TAB
# ==================================================================================================

cross_rows = []

for body_bucket in body_labels:

    for close_bucket in close_labels:

        g = results[
            (results["BodyBucket"].astype(str) == body_bucket) &
            (results["CloseLocationBucket"].astype(str) == close_bucket)
        ]

        if len(g) == 0:
            continue

        row = {
            "BodyBucket": body_bucket,
            "CloseLocationBucket": close_bucket
        }

        row.update(summarize(g))

        cross_rows.append(row)

body_close_cross = pd.DataFrame(cross_rows)


# ==================================================================================================
# PRINT RESULTS
# ==================================================================================================

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 250)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")


section("MAIN COMPARISON — CROSS CANDLE VS OLD FILTER")

cols = [
    "Test",
    "N",
    "AvgMFE_15m",
    "AvgMAE_15m",
    "AvgReturn_15m",
    "AvgMFE_30m",
    "AvgMAE_30m",
    "AvgReturn_30m",
    "AvgMFE_60m",
    "AvgMAE_60m",
    "AvgReturn_60m",
    "WinPct_60m",
    "MFE_MAE_Ratio_60m"
]

print(
    comparison[cols].to_string(index=False)
)


section("DIRECTIONAL BODY SIZE BUCKETS")

print(
    body_summary[
        [
            "BodyBucket",
            "N",
            "AvgMFE_15m",
            "AvgMAE_15m",
            "AvgReturn_15m",
            "AvgMFE_30m",
            "AvgMAE_30m",
            "AvgReturn_30m",
            "AvgMFE_60m",
            "AvgMAE_60m",
            "AvgReturn_60m",
            "WinPct_60m",
            "MFE_MAE_Ratio_60m"
        ]
    ].to_string(index=False)
)


section("DIRECTIONAL CLOSE LOCATION BUCKETS")

print(
    close_summary[
        [
            "CloseLocationBucket",
            "N",
            "AvgMFE_15m",
            "AvgMAE_15m",
            "AvgReturn_15m",
            "AvgMFE_30m",
            "AvgMAE_30m",
            "AvgReturn_30m",
            "AvgMFE_60m",
            "AvgMAE_60m",
            "AvgReturn_60m",
            "WinPct_60m",
            "MFE_MAE_Ratio_60m"
        ]
    ].to_string(index=False)
)


section("OLD PRICE MOVE — BROAD BUCKETS")

print(
    price_move_summary[
        [
            "PriceMoveBucket",
            "N",
            "AvgMFE_15m",
            "AvgMAE_15m",
            "AvgReturn_15m",
            "AvgMFE_30m",
            "AvgMAE_30m",
            "AvgReturn_30m",
            "AvgMFE_60m",
            "AvgMAE_60m",
            "AvgReturn_60m",
            "WinPct_60m",
            "MFE_MAE_Ratio_60m"
        ]
    ].to_string(index=False)
)


section("LONG VS SHORT — 60 MINUTES")

print(
    direction_summary[
        [
            "Test",
            "Direction",
            "N",
            "AvgMFE_60m",
            "AvgMAE_60m",
            "AvgReturn_60m",
            "WinPct_60m",
            "MFE_MAE_Ratio_60m"
        ]
    ].to_string(index=False)
)


section("YEAR STABILITY — 60 MINUTES")

print(
    year_summary[
        [
            "Test",
            "Year",
            "N",
            "AvgMFE_60m",
            "AvgMAE_60m",
            "AvgReturn_60m",
            "WinPct_60m",
            "MFE_MAE_Ratio_60m"
        ]
    ].to_string(index=False)
)


# ==================================================================================================
# SAVE CSV FILES
# ==================================================================================================

results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "raw_signals.csv"
    ),
    index=False
)

comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "main_comparison.csv"
    ),
    index=False
)

body_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "body_bucket_summary.csv"
    ),
    index=False
)

close_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "close_location_summary.csv"
    ),
    index=False
)

price_move_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "price_move_bucket_summary.csv"
    ),
    index=False
)

direction_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "direction_summary.csv"
    ),
    index=False
)

year_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "year_summary.csv"
    ),
    index=False
)

body_close_cross.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "body_x_close_location.csv"
    ),
    index=False
)


# ==================================================================================================
# DONE
# ==================================================================================================

section("DONE")

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("What to look for:")
print("  1. Does body size improve MFE/MAE and return monotonically?")
print("  2. Does closing near the directional extreme improve the result?")
print("  3. Is Body + Close Location better than the old PriceMoveATR filter?")
print("  4. Does the old 0.35-0.90 filter sit inside a broad good PriceMove region?")
print("  5. Are results similar for LONG and SHORT?")
print("  6. Are results positive and reasonably stable in 2022, 2023, 2024, 2025 and 2026?")