import numpy as np
import pandas as pd


# ======================================================================================
# SETTINGS
# ======================================================================================

CSV_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"

OUTPUT_TRADES = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\STOPLOSS_MAE_TRADES.csv"
OUTPUT_POINTS = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\STOPLOSS_POINTS_TEST.csv"
OUTPUT_ATR = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\STOPLOSS_ATR_TEST.csv"

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14

PRICE_MOVE_MIN = 0.35
PRICE_MOVE_MAX = 0.90

PRIOR_MOVE_MIN = -2.0
PRIOR_MOVE_MAX = 2.0

PRIOR_BARS = 20

HOLD_MINUTES = 210

START_YEAR = 2022
END_YEAR = 2026


# Stop grids
POINT_STOPS = [
    2.5, 5, 7.5, 10, 12.5, 15,
    17.5, 20, 22.5, 25, 30, 35,
    40, 45, 50, 60
]

ATR_STOPS = [
    0.25, 0.50, 0.75, 1.00,
    1.25, 1.50, 1.75, 2.00,
    2.50, 3.00, 4.00
]


# ======================================================================================
# HELPERS
# ======================================================================================

def section(title):
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def profit_factor(series):
    r = pd.Series(series).dropna()

    gross_profit = r[r > 0].sum()
    gross_loss = -r[r < 0].sum()

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else np.nan

    return gross_profit / gross_loss


def summary_stats(returns):
    r = pd.Series(returns).dropna()

    if len(r) == 0:
        return {
            "N": 0,
            "Avg": np.nan,
            "PF": np.nan,
            "WinPct": np.nan,
            "TotalPts": np.nan
        }

    return {
        "N": len(r),
        "Avg": r.mean(),
        "PF": profit_factor(r),
        "WinPct": (r > 0).mean() * 100,
        "TotalPts": r.sum()
    }


# ======================================================================================
# LOAD 1-MINUTE DATA
# ======================================================================================

section("LOADING DATA")

df1 = pd.read_csv(CSV_PATH)

time_col = None

for c in [
    "DateTime", "Datetime", "datetime",
    "Timestamp", "timestamp",
    "Time", "time",
    "Date", "date"
]:
    if c in df1.columns:
        time_col = c
        break

if time_col is None:
    raise ValueError("Could not find timestamp column.")

df1[time_col] = pd.to_datetime(df1[time_col])

df1 = (
    df1
    .set_index(time_col)
    .sort_index()
)

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


# ======================================================================================
# BUILD 15-MINUTE BARS
# ======================================================================================

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


# ======================================================================================
# MACD
# ======================================================================================

section("CALCULATING MACD")

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


# ======================================================================================
# ATR
# ======================================================================================

section("CALCULATING ATR + FILTERS")

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
    alpha=1 / ATR_LENGTH,
    adjust=False
).mean()


# ======================================================================================
# ORIGINAL TWO FILTERS
# ======================================================================================

df15["PriceMoveATR"] = (
    df15["Close"] -
    df15["Close"].shift(1)
) / df15["ATR14"]


df15["PriorMoveATR"] = (
    df15["Close"].shift(1) -
    df15["Close"].shift(PRIOR_BARS)
) / df15["ATR14"]


df15["ValidSignal"] = (
    df15["BullCross"] &
    (df15["PriceMoveATR"] >= PRICE_MOVE_MIN) &
    (df15["PriceMoveATR"] <= PRICE_MOVE_MAX) &
    (df15["PriorMoveATR"] >= PRIOR_MOVE_MIN) &
    (df15["PriorMoveATR"] <= PRIOR_MOVE_MAX)
)


candidate_signals = df15[
    df15["ValidSignal"]
].copy()

candidate_signals = candidate_signals[
    (candidate_signals.index.year >= START_YEAR) &
    (candidate_signals.index.year <= END_YEAR)
]

print(
    f"Signals satisfying two filters: {len(candidate_signals):,}"
)


# ======================================================================================
# BUILD FIXED ONE-POSITION TRADE COHORT
# ======================================================================================

section("BUILDING ONE-POSITION BASELINE TRADES")

times = df1.index.values
opens = df1["Open"].to_numpy()
highs = df1["High"].to_numpy()
lows = df1["Low"].to_numpy()

trades = []

next_allowed_entry_time = pd.Timestamp.min

for signal_time, row in candidate_signals.iterrows():

    entry_pos = np.searchsorted(
        times,
        np.datetime64(signal_time),
        side="right"
    )

    if entry_pos >= len(df1):
        continue

    entry_time = df1.index[entry_pos]

    # One-position rule based on original fixed 210m strategy
    if entry_time < next_allowed_entry_time:
        continue

    entry_price = opens[entry_pos]

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

    exit_time = df1.index[exit_pos]

    if exit_time > target_exit_time + pd.Timedelta(minutes=2):
        continue

    exit_price = opens[exit_pos]

    # 1-minute bars traversed while trade is active.
    # Include entry bar through final bar before/at scheduled exit.
    path_high = highs[entry_pos:exit_pos + 1]
    path_low = lows[entry_pos:exit_pos + 1]

    if len(path_low) == 0:
        continue

    baseline_return = exit_price - entry_price

    min_low = np.nanmin(path_low)
    max_high = np.nanmax(path_high)

    mae_points = max(
        0.0,
        entry_price - min_low
    )

    mfe_points = max(
        0.0,
        max_high - entry_price
    )

    atr = row["ATR14"]

    mae_atr = (
        mae_points / atr
        if atr > 0
        else np.nan
    )

    mfe_atr = (
        mfe_points / atr
        if atr > 0
        else np.nan
    )

    trades.append({
        "SignalTime": signal_time,
        "EntryTime": entry_time,
        "ExitTime": exit_time,
        "EntryPrice": entry_price,
        "ExitPrice": exit_price,
        "ATR14": atr,
        "PriceMoveATR": row["PriceMoveATR"],
        "PriorMoveATR": row["PriorMoveATR"],
        "BaselineReturn": baseline_return,
        "MAE_Points": mae_points,
        "MFE_Points": mfe_points,
        "MAE_ATR": mae_atr,
        "MFE_ATR": mfe_atr,
        "Year": signal_time.year,
        "EntryPos": entry_pos,
        "ExitPos": exit_pos
    })

    next_allowed_entry_time = exit_time


trades = pd.DataFrame(trades)

print(f"One-position trades: {len(trades):,}")


# ======================================================================================
# BASELINE
# ======================================================================================

section("BASELINE - NO STOP")

baseline = summary_stats(
    trades["BaselineReturn"]
)

print(f"N:       {baseline['N']:,}")
print(f"Avg:     {baseline['Avg']:.3f}")
print(f"PF:      {baseline['PF']:.3f}")
print(f"Win %:   {baseline['WinPct']:.2f}")
print(f"Total:   {baseline['TotalPts']:.2f}")


# ======================================================================================
# MAE OF WINNERS
# ======================================================================================

section("MAXIMUM ADVERSE EXCURSION - WINNERS ONLY")

winners = trades[
    trades["BaselineReturn"] > 0
].copy()

losers = trades[
    trades["BaselineReturn"] <= 0
].copy()

print(f"Winners: {len(winners):,}")
print(f"Losers:  {len(losers):,}")

percentiles = [
    50,
    60,
    70,
    75,
    80,
    85,
    90,
    95,
    97.5,
    99
]

print()
print("Winner MAE in POINTS")
print("-" * 60)

for p in percentiles:
    value = np.percentile(
        winners["MAE_Points"],
        p
    )

    print(
        f"{p:5.1f}% of winners have MAE <= {value:7.3f} points"
    )


print()
print("Winner MAE in ATR")
print("-" * 60)

for p in percentiles:
    value = np.percentile(
        winners["MAE_ATR"].dropna(),
        p
    )

    print(
        f"{p:5.1f}% of winners have MAE <= {value:7.3f} ATR"
    )


# ======================================================================================
# LOSER MAE DISTRIBUTION
# ======================================================================================

section("MAE - LOSERS")

for p in [
    25,
    50,
    75,
    90
]:
    value = np.percentile(
        losers["MAE_Points"],
        p
    )

    print(
        f"{p:3d}% loser MAE percentile: {value:7.3f} points"
    )


# ======================================================================================
# STOP SIMULATION FUNCTION
# ======================================================================================

def simulate_stop(trade, stop_points):

    entry_pos = int(
        trade["EntryPos"]
    )

    exit_pos = int(
        trade["ExitPos"]
    )

    entry_price = trade[
        "EntryPrice"
    ]

    stop_price = (
        entry_price -
        stop_points
    )

    # Start from entry bar.
    # Conservative rule:
    # if 1m low touches stop, assume stop is executed.
    for pos in range(
        entry_pos,
        exit_pos + 1
    ):

        bar_open = opens[pos]
        bar_low = lows[pos]

        if bar_low <= stop_price:

            # Gap below stop:
            # assume fill at bar open if open is worse than stop.
            fill_price = min(
                stop_price,
                bar_open
            )

            return (
                fill_price -
                entry_price
            ), True

    return (
        trade["BaselineReturn"],
        False
    )


# ======================================================================================
# FIXED POINT STOPS
# ======================================================================================

section("FIXED POINT STOP TEST")

point_rows = []

baseline_winners = (
    trades["BaselineReturn"] > 0
)

n_baseline_winners = baseline_winners.sum()


for stop_points in POINT_STOPS:

    returns = []
    stopped = []

    for _, trade in trades.iterrows():

        ret, hit = simulate_stop(
            trade,
            stop_points
        )

        returns.append(ret)
        stopped.append(hit)


    returns = pd.Series(
        returns,
        index=trades.index
    )

    stopped = pd.Series(
        stopped,
        index=trades.index
    )


    s = summary_stats(
        returns
    )


    winners_stopped = (
        stopped &
        baseline_winners
    ).sum()


    winners_kept = (
        n_baseline_winners -
        winners_stopped
    )


    winner_retention = (
        winners_kept /
        n_baseline_winners *
        100
    )


    losers_stopped = (
        stopped &
        (~baseline_winners)
    ).sum()


    loser_stop_pct = (
        losers_stopped /
        (~baseline_winners).sum() *
        100
    )


    row = {
        "StopPoints": stop_points,
        "N": s["N"],
        "Avg": s["Avg"],
        "PF": s["PF"],
        "WinPct": s["WinPct"],
        "TotalPts": s["TotalPts"],
        "StoppedPct": stopped.mean() * 100,
        "WinnerRetentionPct": winner_retention,
        "BaselineWinnersStopped": winners_stopped,
        "LosersStoppedPct": loser_stop_pct
    }

    point_rows.append(row)

    print(
        f"Stop {stop_points:5.1f} pts | "
        f"PF {s['PF']:6.3f} | "
        f"Avg {s['Avg']:7.3f} | "
        f"Win {s['WinPct']:6.2f}% | "
        f"Keep winners {winner_retention:6.2f}% | "
        f"Stop losers {loser_stop_pct:6.2f}%"
    )


point_results = pd.DataFrame(
    point_rows
)


# ======================================================================================
# ATR STOP TEST
# ======================================================================================

section("ATR STOP TEST")

atr_rows = []


for atr_multiple in ATR_STOPS:

    returns = []
    stopped = []

    for _, trade in trades.iterrows():

        stop_points = (
            trade["ATR14"] *
            atr_multiple
        )

        ret, hit = simulate_stop(
            trade,
            stop_points
        )

        returns.append(ret)
        stopped.append(hit)


    returns = pd.Series(
        returns,
        index=trades.index
    )

    stopped = pd.Series(
        stopped,
        index=trades.index
    )


    s = summary_stats(
        returns
    )


    winners_stopped = (
        stopped &
        baseline_winners
    ).sum()


    winners_kept = (
        n_baseline_winners -
        winners_stopped
    )


    winner_retention = (
        winners_kept /
        n_baseline_winners *
        100
    )


    losers_stopped = (
        stopped &
        (~baseline_winners)
    ).sum()


    loser_stop_pct = (
        losers_stopped /
        (~baseline_winners).sum() *
        100
    )


    row = {
        "StopATR": atr_multiple,
        "N": s["N"],
        "Avg": s["Avg"],
        "PF": s["PF"],
        "WinPct": s["WinPct"],
        "TotalPts": s["TotalPts"],
        "StoppedPct": stopped.mean() * 100,
        "WinnerRetentionPct": winner_retention,
        "BaselineWinnersStopped": winners_stopped,
        "LosersStoppedPct": loser_stop_pct
    }

    atr_rows.append(row)


    print(
        f"Stop {atr_multiple:4.2f} ATR | "
        f"PF {s['PF']:6.3f} | "
        f"Avg {s['Avg']:7.3f} | "
        f"Win {s['WinPct']:6.2f}% | "
        f"Keep winners {winner_retention:6.2f}% | "
        f"Stop losers {loser_stop_pct:6.2f}%"
    )


atr_results = pd.DataFrame(
    atr_rows
)


# ======================================================================================
# BEST STOPS WHILE KEEPING MOST WINNERS
# ======================================================================================

section("POINT STOPS - KEEPING >= 90% OF ORIGINAL WINNERS")

keep90_points = point_results[
    point_results["WinnerRetentionPct"] >= 90
].copy()

if len(keep90_points) > 0:

    print(
        keep90_points[
            [
                "StopPoints",
                "PF",
                "Avg",
                "WinPct",
                "WinnerRetentionPct",
                "LosersStoppedPct",
                "TotalPts"
            ]
        ].sort_values(
            "PF",
            ascending=False
        ).to_string(
            index=False
        )
    )

else:
    print("No point stop retained >=90% of winners.")


section("ATR STOPS - KEEPING >= 90% OF ORIGINAL WINNERS")

keep90_atr = atr_results[
    atr_results["WinnerRetentionPct"] >= 90
].copy()

if len(keep90_atr) > 0:

    print(
        keep90_atr[
            [
                "StopATR",
                "PF",
                "Avg",
                "WinPct",
                "WinnerRetentionPct",
                "LosersStoppedPct",
                "TotalPts"
            ]
        ].sort_values(
            "PF",
            ascending=False
        ).to_string(
            index=False
        )
    )

else:
    print("No ATR stop retained >=90% of winners.")


# ======================================================================================
# YEAR-BY-YEAR MAE OF WINNERS
# ======================================================================================

section("WINNER MAE BY YEAR")

for year in range(
    START_YEAR,
    END_YEAR + 1
):

    y = winners[
        winners["Year"] == year
    ]

    if len(y) == 0:
        continue

    p90_points = np.percentile(
        y["MAE_Points"],
        90
    )

    p95_points = np.percentile(
        y["MAE_Points"],
        95
    )

    p90_atr = np.percentile(
        y["MAE_ATR"].dropna(),
        90
    )

    p95_atr = np.percentile(
        y["MAE_ATR"].dropna(),
        95
    )

    print(
        f"{year} | "
        f"N winners {len(y):4d} | "
        f"P90 {p90_points:6.2f} pts / {p90_atr:5.2f} ATR | "
        f"P95 {p95_points:6.2f} pts / {p95_atr:5.2f} ATR"
    )


# ======================================================================================
# SAVE
# ======================================================================================

trades.to_csv(
    OUTPUT_TRADES,
    index=False
)

point_results.to_csv(
    OUTPUT_POINTS,
    index=False
)

atr_results.to_csv(
    OUTPUT_ATR,
    index=False
)


section("DONE")

print(f"Trades:\n{OUTPUT_TRADES}")
print()
print(f"Point stop results:\n{OUTPUT_POINTS}")
print()
print(f"ATR stop results:\n{OUTPUT_ATR}")