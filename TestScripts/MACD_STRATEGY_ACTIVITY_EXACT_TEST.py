import os
import numpy as np
import pandas as pd

# ==================================================================================================
# SETTINGS
# ==================================================================================================

CSV_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "MACD_STRATEGY_ACTIVITY_EXACT_TEST"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# MACD
MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

# Existing strategy filters
ATR_LENGTH = 14

PRICE_MOVE_MIN = 0.35
PRICE_MOVE_MAX = 0.90

PRIOR_LOOKBACK = 20
PRIOR_MOVE_MIN = -2.0
PRIOR_MOVE_MAX = 2.0

# Activity diagnostic
ACTIVITY_ATR_LENGTH = 5
ACTIVITY_BASE_LENGTH = 200

# Pine-equivalent hold
HOLD_MINUTES = 210

# Full research period
START_YEAR = 2022
END_YEAR = 2026

# Same Pine comparison window
PINE_START = pd.Timestamp("2026-05-01 00:00:00")
PINE_END = pd.Timestamp("2026-09-18 23:59:00")

# Activity variants
ACTIVITY_TESTS = {
    "NO ACTIVITY FILTER": None,
    "Activity >= 0.50": ("min", 0.50),
    "Activity >= 0.60": ("min", 0.60),
    "Activity >= 0.70": ("min", 0.70),
    "Activity >= 0.75": ("min", 0.75),
    "Activity 0.60-0.70": ("range", 0.60, 0.70),
    "Activity 0.80-1.00": ("range", 0.80, 1.00),
    "Activity >= 1.20": ("min", 1.20),
}


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


def activity_pass(value, rule):
    if rule is None:
        return True

    if pd.isna(value):
        return False

    if rule[0] == "min":
        return value >= rule[1]

    if rule[0] == "range":
        return value >= rule[1] and value < rule[2]

    return False


def profit_factor(pnls):
    gross_profit = pnls[pnls > 0].sum()
    gross_loss = -pnls[pnls < 0].sum()

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else np.nan

    return gross_profit / gross_loss


def max_drawdown(pnls):
    equity = pnls.cumsum()
    peak = equity.cummax()
    dd = peak - equity

    if len(dd) == 0:
        return np.nan

    return dd.max()


def summarize_trades(trades):
    if len(trades) == 0:
        return {
            "N": 0,
            "AvgPnL": np.nan,
            "NetPnL": np.nan,
            "PF": np.nan,
            "WinPct": np.nan,
            "MaxDD": np.nan
        }

    pnl = trades["PnL"]

    return {
        "N": len(trades),
        "AvgPnL": pnl.mean(),
        "NetPnL": pnl.sum(),
        "PF": profit_factor(pnl),
        "WinPct": (pnl > 0).mean() * 100,
        "MaxDD": max_drawdown(pnl)
    }


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


# ==================================================================================================
# EXISTING FILTERS
# ==================================================================================================

df15["ATR14"] = wilder_atr(
    df15,
    ATR_LENGTH
)

df15["PriceMoveATR"] = (
    (df15["Close"] - df15["Close"].shift(1)) /
    df15["ATR14"]
)

df15["PriceMovePass"] = (
    (df15["PriceMoveATR"] >= PRICE_MOVE_MIN) &
    (df15["PriceMoveATR"] <= PRICE_MOVE_MAX)
)

df15["PriorMoveATR"] = (
    (df15["Close"].shift(1) - df15["Close"].shift(PRIOR_LOOKBACK)) /
    df15["ATR14"]
)

df15["PriorMovePass"] = (
    (df15["PriorMoveATR"] >= PRIOR_MOVE_MIN) &
    (df15["PriorMoveATR"] <= PRIOR_MOVE_MAX)
)

df15["BaseSignal"] = (
    df15["BullCross"] &
    df15["PriceMovePass"] &
    df15["PriorMovePass"]
)


# ==================================================================================================
# ACTIVITY RATIO
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
# SIGNAL TABLE
# ==================================================================================================

signals = df15[
    df15["BaseSignal"]
].copy()

signals = signals[
    (signals.index.year >= START_YEAR) &
    (signals.index.year <= END_YEAR)
].copy()

signals["Year"] = signals.index.year

section("BASE SIGNALS")

print(f"Base MACD + PriceMove + PriorMove signals: {len(signals):,}")


# ==================================================================================================
# 1-MINUTE ARRAYS
# ==================================================================================================

times = df1.index.values
opens = df1["Open"].to_numpy()


# ==================================================================================================
# ONE-POSITION-AT-A-TIME BACKTEST
# ==================================================================================================

def run_strategy(signal_df, activity_rule=None):

    trades = []

    position_exit_time = None

    blocked_signals = 0
    rejected_activity = 0

    for signal_time, s in signal_df.iterrows():

        # ------------------------------------------------------------------
        # Activity filter
        # ------------------------------------------------------------------

        if not activity_pass(
            s["ActivityRatio"],
            activity_rule
        ):
            rejected_activity += 1
            continue

        # ------------------------------------------------------------------
        # Signal occurs while existing trade is still open
        # ------------------------------------------------------------------

        if position_exit_time is not None and signal_time < position_exit_time:
            blocked_signals += 1
            continue

        # ------------------------------------------------------------------
        # ENTRY
        # First 1-minute open strictly after completed 15m signal bar
        # ------------------------------------------------------------------

        entry_pos = np.searchsorted(
            times,
            np.datetime64(signal_time),
            side="right"
        )

        if entry_pos >= len(df1):
            continue

        entry_time = df1.index[entry_pos]
        entry_price = opens[entry_pos]

        # ------------------------------------------------------------------
        # EXIT
        # 210 minutes after actual entry
        #
        # This mirrors:
        # signal bar t
        # entry fill t+1
        # exit fill t+15
        #
        # = 14 x 15m = 210 minutes actual holding time
        # ------------------------------------------------------------------

        target_exit_time = (
            entry_time +
            pd.Timedelta(minutes=HOLD_MINUTES)
        )

        exit_pos = np.searchsorted(
            times,
            np.datetime64(target_exit_time),
            side="left"
        )

        if exit_pos >= len(df1):
            continue

        actual_exit_time = df1.index[exit_pos]

        if actual_exit_time > target_exit_time + pd.Timedelta(minutes=2):
            continue

        exit_price = opens[exit_pos]

        pnl = exit_price - entry_price

        trades.append({
            "SignalTime": signal_time,
            "EntryTime": entry_time,
            "ExitTime": actual_exit_time,
            "EntryPrice": entry_price,
            "ExitPrice": exit_price,
            "PnL": pnl,
            "Year": signal_time.year,
            "ActivityRatio": s["ActivityRatio"],
            "PriceMoveATR": s["PriceMoveATR"],
            "PriorMoveATR": s["PriorMoveATR"]
        })

        position_exit_time = actual_exit_time

    return (
        pd.DataFrame(trades),
        blocked_signals,
        rejected_activity
    )


# ==================================================================================================
# RUN ALL ACTIVITY TESTS
# ==================================================================================================

section("FULL HISTORY — EXACT STRATEGY COMPARISON")

summary_rows = []
all_trade_sets = {}

for test_name, rule in ACTIVITY_TESTS.items():

    trades, blocked, rejected = run_strategy(
        signals,
        rule
    )

    all_trade_sets[test_name] = trades

    row = {
        "Test": test_name,
        "RejectedByActivity": rejected,
        "BlockedWhileInPosition": blocked
    }

    row.update(
        summarize_trades(trades)
    )

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 250)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")

print(
    summary[
        [
            "Test",
            "N",
            "AvgPnL",
            "NetPnL",
            "PF",
            "WinPct",
            "MaxDD",
            "RejectedByActivity",
            "BlockedWhileInPosition"
        ]
    ].to_string(index=False)
)


# ==================================================================================================
# YEAR STABILITY
# ==================================================================================================

section("YEAR STABILITY")

year_rows = []

for test_name, trades in all_trade_sets.items():

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        gy = trades[
            trades["Year"] == year
        ]

        row = {
            "Test": test_name,
            "Year": year
        }

        row.update(
            summarize_trades(gy)
        )

        year_rows.append(row)

year_summary = pd.DataFrame(year_rows)

print(
    year_summary[
        [
            "Test",
            "Year",
            "N",
            "AvgPnL",
            "NetPnL",
            "PF",
            "WinPct",
            "MaxDD"
        ]
    ].to_string(index=False)
)


# ==================================================================================================
# STABILITY RANKING
# ==================================================================================================

section("STABILITY RANKING")

stability_rows = []

for test_name, trades in all_trade_sets.items():

    yearly_pf = []
    yearly_avg = []

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        gy = trades[
            trades["Year"] == year
        ]

        if len(gy) == 0:
            continue

        yearly_avg.append(
            gy["PnL"].mean()
        )

        yearly_pf.append(
            profit_factor(
                gy["PnL"]
            )
        )

    base = summarize_trades(
        trades
    )

    stability_rows.append({
        "Test": test_name,
        "N": base["N"],
        "PF": base["PF"],
        "AvgPnL": base["AvgPnL"],
        "NetPnL": base["NetPnL"],
        "MaxDD": base["MaxDD"],
        "PositiveYears": sum(x > 0 for x in yearly_avg),
        "NegativeYears": sum(x < 0 for x in yearly_avg),
        "WorstYearAvg": min(yearly_avg) if yearly_avg else np.nan,
        "WorstYearPF": min(yearly_pf) if yearly_pf else np.nan
    })

stability = pd.DataFrame(
    stability_rows
)

print(
    stability.sort_values(
        [
            "PositiveYears",
            "WorstYearPF",
            "PF"
        ],
        ascending=[
            False,
            False,
            False
        ]
    ).to_string(index=False)
)


# ==================================================================================================
# SAME MAY-SEP 2026 WINDOW AS PINE
# ==================================================================================================

section("PINE COMPARISON WINDOW — MAY 1 TO SEP 18 2026")

pine_rows = []

for test_name, trades in all_trade_sets.items():

    gp = trades[
        (trades["SignalTime"] >= PINE_START) &
        (trades["SignalTime"] <= PINE_END)
    ]

    row = {
        "Test": test_name
    }

    row.update(
        summarize_trades(gp)
    )

    pine_rows.append(row)

pine_summary = pd.DataFrame(
    pine_rows
)

print(
    pine_summary[
        [
            "Test",
            "N",
            "AvgPnL",
            "NetPnL",
            "PF",
            "WinPct",
            "MaxDD"
        ]
    ].to_string(index=False)
)


# ==================================================================================================
# SAVE FILES
# ==================================================================================================

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "full_history_summary.csv"
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

stability.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "stability_ranking.csv"
    ),
    index=False
)

pine_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "pine_window_summary.csv"
    ),
    index=False
)

for test_name, trades in all_trade_sets.items():

    safe_name = (
        test_name
        .replace(" ", "_")
        .replace(">=", "GE")
        .replace(".", "p")
    )

    trades.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"trades_{safe_name}.csv"
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
print("This test uses:")
print("  MACD bullish zero-cross")
print(f"  PriceMoveATR {PRICE_MOVE_MIN:.2f}-{PRICE_MOVE_MAX:.2f}")
print(f"  PriorMoveATR {PRIOR_MOVE_MIN:.2f} to {PRIOR_MOVE_MAX:.2f}")
print("  one position at a time")
print("  210-minute actual hold")
print()
print("Main question:")
print("Does ActivityRatio improve the ACTUAL strategy rather than independent signals?")