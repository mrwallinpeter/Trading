import pandas as pd
from pathlib import Path


# ============================================================
# PATH
# ============================================================

ONE_MIN_FILE = Path(
    r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"
)


# ============================================================
# SETTINGS
# ============================================================

START = pd.Timestamp("2026-05-01 00:00:00")
END = pd.Timestamp("2026-09-18 23:59:59")

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14

PRICE_MOVE_MIN = 0.35
PRICE_MOVE_MAX = 0.90

PRIOR_LOOKBACK = 20
PRIOR_MOVE_MIN = -2.0
PRIOR_MOVE_MAX = 2.0


# ============================================================
# LOAD 1-MIN DATA
# ============================================================

print()
print("=" * 100)
print("LOADING 1-MIN DATA")
print("=" * 100)

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

for c in ["Open", "High", "Low", "Close", "Volume"]:
    df1[c] = pd.to_numeric(
        df1[c],
        errors="coerce"
    )

df1 = df1.dropna(
    subset=["Open", "High", "Low", "Close"]
)

print("1m rows:", len(df1))
print("Start:", df1.index.min())
print("End:  ", df1.index.max())


# ============================================================
# BUILD 15M BARS
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

print()
print("15m bars:", len(df15))


# ============================================================
# MACD
# ============================================================

ema_fast = df15["Close"].ewm(
    span=MACD_FAST,
    adjust=False
).mean()

ema_slow = df15["Close"].ewm(
    span=MACD_SLOW,
    adjust=False
).mean()

df15["MACD"] = (
    ema_fast
    -
    ema_slow
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

tr1 = (
    df15["High"]
    -
    df15["Low"]
)

tr2 = (
    df15["High"]
    -
    prev_close
).abs()

tr3 = (
    df15["Low"]
    -
    prev_close
).abs()

df15["TR"] = pd.concat(
    [tr1, tr2, tr3],
    axis=1
).max(axis=1)

df15["ATR14"] = df15["TR"].ewm(
    alpha=1.0 / ATR_LENGTH,
    adjust=False
).mean()


# ============================================================
# PRICE MOVE
#
# (Close[t] - Close[t-1]) / ATR14[t]
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

df15["PriceMovePass"] = (
    (df15["PriceMoveATR14"] >= PRICE_MOVE_MIN)
    &
    (df15["PriceMoveATR14"] <= PRICE_MOVE_MAX)
)


# ============================================================
# PRIOR MOVE
#
# (Close[t-1] - Close[t-20]) / ATR14[t]
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

df15["PriorMovePass"] = (
    (df15["PriorMoveATR14"] >= PRIOR_MOVE_MIN)
    &
    (df15["PriorMoveATR14"] <= PRIOR_MOVE_MAX)
)


# ============================================================
# VALIDATION WINDOW
# ============================================================

test = df15[
    (df15.index >= START)
    &
    (df15.index <= END)
].copy()


# ============================================================
# COUNTS
# ============================================================

bars_in_window = len(test)

bull_crosses = int(
    test["BullCross"].sum()
)

cross_plus_price = int(
    (
        test["BullCross"]
        &
        test["PriceMovePass"]
    ).sum()
)

qualified = int(
    (
        test["BullCross"]
        &
        test["PriceMovePass"]
        &
        test["PriorMovePass"]
    ).sum()
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 100)
print("PINE VS PYTHON COUNT VALIDATION")
print("=" * 100)

print()
print("Window:")
print("Start:", START)
print("End:  ", END)

print()
print("Python counts:")
print(
    f"15m bars           = {bars_in_window}"
)
print(
    f"Bull MACD crosses  = {bull_crosses}"
)
print(
    f"Cross + PriceMove  = {cross_plus_price}"
)
print(
    f"Qualified          = {qualified}"
)

print()
print("Pine reference:")
print(
    "15m bars           = 9220"
)
print(
    "Bull MACD crosses  = 620"
)
print(
    "Cross + PriceMove  = 273"
)
print(
    "Qualified          = 159"
)


# ============================================================
# EXACT COMPARISON
# ============================================================

print()
print("=" * 100)
print("COMPARISON")
print("=" * 100)

checks = {
    "15m bars": (
        bars_in_window,
        9220
    ),
    "Bull MACD crosses": (
        bull_crosses,
        620
    ),
    "Cross + PriceMove": (
        cross_plus_price,
        273
    ),
    "Qualified": (
        qualified,
        159
    )
}

all_match = True

for name, (python_value, pine_value) in checks.items():

    match = (
        python_value
        ==
        pine_value
    )

    all_match = (
        all_match
        and
        match
    )

    print(
        f"{name:<22} "
        f"Python={python_value:<6} "
        f"Pine={pine_value:<6} "
        f"{'MATCH' if match else 'DIFFER'}"
    )


print()
print("=" * 100)

if all_match:

    print(
        "RESULT: EXACT MATCH AT ALL STAGES"
    )

else:

    print(
        "RESULT: COUNTS DIFFER — investigate bar construction, "
        "timezone, contract data, or ATR/MACD initialization."
    )

print("=" * 100)


# ============================================================
# OPTIONAL SIGNAL EXPORT
# ============================================================

signal_detail = test[
    test["BullCross"]
].copy()

signal_detail[
    [
        "Open",
        "High",
        "Low",
        "Close",
        "Hist",
        "ATR14",
        "PriceMoveATR14",
        "PriceMovePass",
        "PriorMoveATR14",
        "PriorMovePass"
    ]
].to_csv(
    Path(
        r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\TestScripts"
    )
    /
    "MACD_15M_Pine_Count_Validation_signals.csv"
)

print()
print(
    "Signal detail CSV saved."
)