import os
import numpy as np
import pandas as pd

# ==================================================================================================
# SETTINGS
# ==================================================================================================

CSV_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "MACD_PRICEMOVE_ACTIVITY_TEST"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_SIGNAL_LENGTH = 14

ACTIVITY_ATR_LENGTH = 5
ACTIVITY_BASE_LENGTH = 200

START_YEAR = 2022
END_YEAR = 2026

PRICE_MIN = 0.50
PRICE_MAX = 0.75

HORIZONS = [15, 30, 45, 60]

ACTIVITY_BINS = [
    -np.inf,
    0.60,
    0.70,
    0.80,
    1.00,
    1.20,
    np.inf
]

ACTIVITY_LABELS = [
    "<0.60",
    "0.60-0.70",
    "0.70-0.80",
    "0.80-1.00",
    "1.00-1.20",
    "1.20+"
]


# ==================================================================================================
# HELPERS
# ==================================================================================================

def section(title):
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def wilder_atr(df, length):
    prev_close = df["Close"].shift(1)

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1.0 / length,
        adjust=False
    ).mean()


def summarize(df):
    out = {"N": len(df)}

    if len(df) == 0:
        return out

    for h in HORIZONS:
        mfe = df[f"MFE_{h}m"]
        mae = df[f"MAE_{h}m"]
        ret = df[f"Return_{h}m"]

        avg_mfe = mfe.mean()
        avg_mae = mae.mean()

        out[f"AvgMFE_{h}m"] = avg_mfe
        out[f"AvgMAE_{h}m"] = avg_mae
        out[f"AvgReturn_{h}m"] = ret.mean()
        out[f"WinPct_{h}m"] = (ret > 0).mean() * 100
        out[f"MFE_MAE_Ratio_{h}m"] = avg_mfe / avg_mae if avg_mae != 0 else np.nan

    return out


# ==================================================================================================
# LOAD DATA
# ==================================================================================================

section("LOADING DATA")

df1 = pd.read_csv(CSV_PATH)

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
    raise ValueError("Could not find timestamp column.")

df1[time_col] = pd.to_datetime(df1[time_col])
df1 = df1.set_index(time_col).sort_index()

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

section("CALCULATING MACD + ATR + ACTIVITY")

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

df15["MACDHist"] = macd - signal_line

df15["BullCross"] = (
    (df15["MACDHist"].shift(1) <= 0) &
    (df15["MACDHist"] > 0)
)

df15["BearCross"] = (
    (df15["MACDHist"].shift(1) >= 0) &
    (df15["MACDHist"] < 0)
)


# ==================================================================================================
# ATR14 FOR PRICEMOVE
# ==================================================================================================

df15["ATR14"] = wilder_atr(
    df15,
    ATR_SIGNAL_LENGTH
)


# ==================================================================================================
# ATR5 ACTIVITY RATIO
# ==================================================================================================

df15["ATR5"] = wilder_atr(
    df15,
    ACTIVITY_ATR_LENGTH
)

df15["ATR5_SMA200"] = df15["ATR5"].rolling(
    ACTIVITY_BASE_LENGTH
).mean()

df15["ActivityRatio"] = (
    df15["ATR5"] /
    df15["ATR5_SMA200"]
)


# ==================================================================================================
# DIRECTION + PRICE MOVE
# ==================================================================================================

df15["DirectionValue"] = 0

df15.loc[df15["BullCross"], "DirectionValue"] = 1
df15.loc[df15["BearCross"], "DirectionValue"] = -1

df15["PriceMoveATR"] = (
    df15["DirectionValue"] *
    (df15["Close"] - df15["Close"].shift(1)) /
    df15["ATR14"]
)


# ==================================================================================================
# SIGNALS
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

signals = signals[
    (signals["PriceMoveATR"] >= PRICE_MIN) &
    (signals["PriceMoveATR"] <= PRICE_MAX)
].copy()

signals = signals.dropna(
    subset=["ActivityRatio"]
)

section("QUALIFIED SIGNALS")

print(f"PriceMove 0.50-0.75 signals: {len(signals):,}")
print(f"Long:  {(signals['Direction'] == 'LONG').sum():,}")
print(f"Short: {(signals['Direction'] == 'SHORT').sum():,}")


# ==================================================================================================
# 1-MINUTE PATH
# ==================================================================================================

section("CALCULATING POST-CROSS PATH")

times_1m = df1.index.values

opens_1m = df1["Open"].to_numpy()
highs_1m = df1["High"].to_numpy()
lows_1m = df1["Low"].to_numpy()
closes_1m = df1["Close"].to_numpy()

rows = []

for signal_time, s in signals.iterrows():

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
        "Year": s["Year"],
        "EntryPrice": entry_price,
        "ATR14": s["ATR14"],
        "ATR5": s["ATR5"],
        "ATR5_SMA200": s["ATR5_SMA200"],
        "ActivityRatio": s["ActivityRatio"],
        "PriceMoveATR": s["PriceMoveATR"]
    }

    valid_trade = True

    for h in HORIZONS:

        target_time = (
            entry_time +
            pd.Timedelta(minutes=h)
        )

        end_pos = np.searchsorted(
            times_1m,
            np.datetime64(target_time),
            side="left"
        )

        if end_pos >= len(df1):
            valid_trade = False
            break

        actual_end_time = df1.index[end_pos]

        if actual_end_time > target_time + pd.Timedelta(minutes=2):
            valid_trade = False
            break

        path_high = highs_1m[
            entry_pos:end_pos + 1
        ].max()

        path_low = lows_1m[
            entry_pos:end_pos + 1
        ].min()

        exit_price = closes_1m[
            end_pos
        ]

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

print(f"Signals with valid path: {len(results):,}")


# ==================================================================================================
# ACTIVITY BUCKETS
# ==================================================================================================

results["ActivityBucket"] = pd.cut(
    results["ActivityRatio"],
    bins=ACTIVITY_BINS,
    labels=ACTIVITY_LABELS,
    right=False
)

bucket_rows = []

for bucket in ACTIVITY_LABELS:

    g = results[
        results["ActivityBucket"].astype(str) == bucket
    ]

    if len(g) == 0:
        continue

    row = {
        "ActivityBucket": bucket
    }

    row.update(
        summarize(g)
    )

    bucket_rows.append(row)

bucket_summary = pd.DataFrame(bucket_rows)


# ==================================================================================================
# CUMULATIVE FILTER TESTS
# ==================================================================================================

thresholds = [
    0.50,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.90,
    1.00
]

threshold_rows = []

for threshold in thresholds:

    g = results[
        results["ActivityRatio"] >= threshold
    ]

    row = {
        "MinActivityRatio": threshold
    }

    row.update(
        summarize(g)
    )

    threshold_rows.append(row)

threshold_summary = pd.DataFrame(threshold_rows)


# ==================================================================================================
# LONG / SHORT BY ACTIVITY BUCKET
# ==================================================================================================

direction_rows = []

for bucket in ACTIVITY_LABELS:

    for direction in ["LONG", "SHORT"]:

        g = results[
            (results["ActivityBucket"].astype(str) == bucket) &
            (results["Direction"] == direction)
        ]

        if len(g) == 0:
            continue

        row = {
            "ActivityBucket": bucket,
            "Direction": direction
        }

        row.update(
            summarize(g)
        )

        direction_rows.append(row)

direction_summary = pd.DataFrame(direction_rows)


# ==================================================================================================
# YEAR STABILITY BY ACTIVITY BUCKET
# ==================================================================================================

year_rows = []

for bucket in ACTIVITY_LABELS:

    for year in sorted(
        results["Year"].unique()
    ):

        g = results[
            (results["ActivityBucket"].astype(str) == bucket) &
            (results["Year"] == year)
        ]

        if len(g) == 0:
            continue

        row = {
            "ActivityBucket": bucket,
            "Year": year
        }

        row.update(
            summarize(g)
        )

        year_rows.append(row)

year_summary = pd.DataFrame(year_rows)


# ==================================================================================================
# PRINT
# ==================================================================================================

pd.set_option(
    "display.max_columns",
    100
)

pd.set_option(
    "display.width",
    260
)

pd.set_option(
    "display.float_format",
    lambda x: f"{x:.3f}"
)


section("ACTIVITY BUCKET SUMMARY")

print(
    bucket_summary[
        [
            "ActivityBucket",
            "N",
            "AvgReturn_15m",
            "AvgReturn_30m",
            "AvgReturn_45m",
            "AvgReturn_60m",
            "AvgMFE_60m",
            "AvgMAE_60m",
            "WinPct_60m",
            "MFE_MAE_Ratio_60m"
        ]
    ].to_string(index=False)
)


section("CUMULATIVE MINIMUM ACTIVITY FILTER")

print(
    threshold_summary[
        [
            "MinActivityRatio",
            "N",
            "AvgReturn_15m",
            "AvgReturn_30m",
            "AvgReturn_45m",
            "AvgReturn_60m",
            "AvgMFE_60m",
            "AvgMAE_60m",
            "WinPct_60m",
            "MFE_MAE_Ratio_60m"
        ]
    ].to_string(index=False)
)


section("LONG VS SHORT BY ACTIVITY BUCKET")

print(
    direction_summary[
        [
            "ActivityBucket",
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


section("YEAR STABILITY BY ACTIVITY BUCKET")

print(
    year_summary[
        [
            "ActivityBucket",
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
# SAVE
# ==================================================================================================

results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "raw_signals.csv"
    ),
    index=False
)

bucket_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "activity_bucket_summary.csv"
    ),
    index=False
)

threshold_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "activity_threshold_summary.csv"
    ),
    index=False
)

direction_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "activity_long_short.csv"
    ),
    index=False
)

year_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "activity_year_summary.csv"
    ),
    index=False
)


section("DONE")

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Main question:")
print("Does low ATR5 relative to its 200-bar average identify weak low-activity trades?")