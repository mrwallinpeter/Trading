import numpy as np
import pandas as pd


# ==================================================================================================
# SETTINGS
# ==================================================================================================

CSV_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"

OUTPUT_MATRIX = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\TARGET_STOP_MATRIX_90MIN.csv"
OUTPUT_YEAR = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\TARGET_STOP_MATRIX_90MIN_YEAR.csv"
OUTPUT_TRADES = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\TARGET_STOP_MATRIX_90MIN_TRADES.csv"

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14

PRICE_MOVE_MIN = 0.35
PRICE_MOVE_MAX = 0.90

PRIOR_MOVE_MIN = -2.0
PRIOR_MOVE_MAX = 2.0

PRIOR_BARS = 20

HOLD_MINUTES = 90

START_YEAR = 2022
END_YEAR = 2026

TARGET_STEP = 2.5
TARGET_MIN = 5.0

STOP_MIN_ATR = 0.50
STOP_MAX_ATR = 6.00
STOP_STEP_ATR = 0.25

# Only used to prevent an absurd target grid if one extreme data spike exists.
# Set to None if you want absolutely no cap.
MAX_TARGET_CAP = 100.0


# ==================================================================================================
# HELPERS
# ==================================================================================================

def section(title):
    print()
    print("=" * 130)
    print(title)
    print("=" * 130)


def profit_factor(values):
    r = np.asarray(values, dtype=float)
    r = r[np.isfinite(r)]

    gross_profit = r[r > 0].sum()
    gross_loss = -r[r < 0].sum()

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else np.nan

    return gross_profit / gross_loss


def max_drawdown(returns):
    r = np.asarray(returns, dtype=float)

    if len(r) == 0:
        return np.nan

    equity = np.cumsum(r)
    equity_with_zero = np.concatenate(([0.0], equity))

    running_max = np.maximum.accumulate(equity_with_zero)
    drawdown = running_max - equity_with_zero

    return drawdown.max()


def calc_stats(returns):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]

    if len(r) == 0:
        return {
            "N": 0,
            "Avg": np.nan,
            "PF": np.nan,
            "WinPct": np.nan,
            "TotalPts": np.nan,
            "MaxDD": np.nan
        }

    return {
        "N": len(r),
        "Avg": r.mean(),
        "PF": profit_factor(r),
        "WinPct": (r > 0).mean() * 100.0,
        "TotalPts": r.sum(),
        "MaxDD": max_drawdown(r)
    }


# ==================================================================================================
# LOAD DATA
# ==================================================================================================

section("LOADING DATA")

df1 = pd.read_csv(CSV_PATH)

time_col = None

for c in [
    "DateTime",
    "Datetime",
    "datetime",
    "Timestamp",
    "timestamp",
    "Time",
    "time",
    "Date",
    "date"
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

required = ["Open", "High", "Low", "Close"]

for c in required:
    if c not in df1.columns:
        raise ValueError(f"Missing column: {c}")

print(f"1-minute rows: {len(df1):,}")
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

print(f"15-minute bars: {len(df15):,}")


# ==================================================================================================
# MACD
# ==================================================================================================

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

macd_signal_line = macd.ewm(
    span=MACD_SIGNAL,
    adjust=False
).mean()

df15["MACDHist"] = (
    macd -
    macd_signal_line
)

df15["BullCross"] = (
    (df15["MACDHist"].shift(1) <= 0) &
    (df15["MACDHist"] > 0)
)


# ==================================================================================================
# ATR
# ==================================================================================================

section("CALCULATING ATR + ORIGINAL TWO FILTERS")

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


# ==================================================================================================
# TWO ORIGINAL FILTERS
# ==================================================================================================

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

signals = df15[
    df15["ValidSignal"]
].copy()

signals = signals[
    (signals.index.year >= START_YEAR) &
    (signals.index.year <= END_YEAR)
]

print(f"Signals satisfying filters: {len(signals):,}")


# ==================================================================================================
# BUILD 90-MINUTE TRADE COHORT
# ==================================================================================================

section("BUILDING 90-MINUTE TRADE COHORT")

times = df1.index.values
opens = df1["Open"].to_numpy()
highs = df1["High"].to_numpy()
lows = df1["Low"].to_numpy()

trade_rows = []

# Fixed one-position cohort based on the 90-minute no-stop/no-target strategy.
next_allowed_entry_time = pd.Timestamp.min

for signal_time, row in signals.iterrows():

    entry_pos = np.searchsorted(
        times,
        np.datetime64(signal_time),
        side="right"
    )

    if entry_pos >= len(df1):
        continue

    entry_time = df1.index[entry_pos]

    if entry_time < next_allowed_entry_time:
        continue

    entry_price = opens[entry_pos]

    scheduled_exit_time = (
        entry_time +
        pd.Timedelta(minutes=HOLD_MINUTES)
    )

    exit_pos = np.searchsorted(
        times,
        np.datetime64(scheduled_exit_time),
        side="left"
    )

    if exit_pos >= len(df1):
        continue

    exit_time = df1.index[exit_pos]

    if (
        exit_time >
        scheduled_exit_time +
        pd.Timedelta(minutes=2)
    ):
        continue

    exit_price = opens[exit_pos]

    path_high = highs[
        entry_pos:exit_pos + 1
    ]

    path_low = lows[
        entry_pos:exit_pos + 1
    ]

    if len(path_high) == 0:
        continue

    mfe = max(
        0.0,
        np.nanmax(path_high) - entry_price
    )

    mae = max(
        0.0,
        entry_price - np.nanmin(path_low)
    )

    atr = row["ATR14"]

    baseline_return = (
        exit_price -
        entry_price
    )

    trade_rows.append(
        {
            "SignalTime": signal_time,
            "EntryTime": entry_time,
            "ScheduledExitTime": exit_time,
            "EntryPrice": entry_price,
            "ScheduledExitPrice": exit_price,
            "ATR14": atr,
            "PriceMoveATR": row["PriceMoveATR"],
            "PriorMoveATR": row["PriorMoveATR"],
            "Baseline90Return": baseline_return,
            "MFE90Points": mfe,
            "MAE90Points": mae,
            "MFE90ATR": mfe / atr if atr > 0 else np.nan,
            "MAE90ATR": mae / atr if atr > 0 else np.nan,
            "Year": signal_time.year,
            "EntryPos": entry_pos,
            "ExitPos": exit_pos
        }
    )

    next_allowed_entry_time = exit_time


trades = pd.DataFrame(trade_rows)

print(f"90-minute one-position trades: {len(trades):,}")

baseline_stats = calc_stats(
    trades["Baseline90Return"]
)

print()
print("90-minute baseline, no target / no stop")
print(f"Avg:      {baseline_stats['Avg']:.3f}")
print(f"PF:       {baseline_stats['PF']:.3f}")
print(f"Win %:    {baseline_stats['WinPct']:.2f}")
print(f"Total:    {baseline_stats['TotalPts']:.2f}")
print(f"Max DD:   {baseline_stats['MaxDD']:.2f}")


# ==================================================================================================
# MFE DISTRIBUTION
# ==================================================================================================

section("90-MINUTE MFE DISTRIBUTION")

mfe_values = trades["MFE90Points"].dropna()

for p in [
    50,
    60,
    70,
    75,
    80,
    85,
    90,
    95,
    97.5,
    99,
    100
]:
    value = np.percentile(
        mfe_values,
        p
    )

    print(
        f"MFE percentile {p:5.1f}% = {value:7.2f} points"
    )


observed_max_mfe = mfe_values.max()

target_max = (
    np.floor(
        observed_max_mfe /
        TARGET_STEP
    ) *
    TARGET_STEP
)

if MAX_TARGET_CAP is not None:
    target_max = min(
        target_max,
        MAX_TARGET_CAP
    )

target_grid = np.arange(
    TARGET_MIN,
    target_max + TARGET_STEP / 2,
    TARGET_STEP
)

stop_grid = np.arange(
    STOP_MIN_ATR,
    STOP_MAX_ATR + STOP_STEP_ATR / 2,
    STOP_STEP_ATR
)


print()
print(f"Observed maximum 90m MFE: {observed_max_mfe:.2f} pts")
print(f"Profit targets tested: {TARGET_MIN:.1f} to {target_max:.1f} pts")
print(f"Number of targets: {len(target_grid)}")
print(
    f"ATR stops tested: {STOP_MIN_ATR:.2f} to "
    f"{STOP_MAX_ATR:.2f} ATR"
)
print(f"Number of stops: {len(stop_grid)}")
print(
    f"Total combinations: "
    f"{len(target_grid) * len(stop_grid):,}"
)


# ==================================================================================================
# SIMULATION
# ==================================================================================================

def simulate_trade(trade, target_points, stop_atr):

    entry_pos = int(
        trade["EntryPos"]
    )

    exit_pos = int(
        trade["ExitPos"]
    )

    entry_price = float(
        trade["EntryPrice"]
    )

    atr = float(
        trade["ATR14"]
    )

    target_price = (
        entry_price +
        target_points
    )

    stop_points = (
        atr *
        stop_atr
    )

    stop_price = (
        entry_price -
        stop_points
    )


    # Search each 1-minute bar chronologically.
    for pos in range(
        entry_pos,
        exit_pos + 1
    ):

        bar_open = opens[pos]
        bar_high = highs[pos]
        bar_low = lows[pos]


        # Gap below stop.
        if bar_open <= stop_price:
            return (
                bar_open - entry_price,
                "STOP"
            )


        # Gap above target.
        if bar_open >= target_price:
            return (
                bar_open - entry_price,
                "TARGET"
            )


        hit_stop = (
            bar_low <= stop_price
        )

        hit_target = (
            bar_high >= target_price
        )


        # With OHLC data we cannot know intrabar order.
        # Conservative assumption:
        # if both stop and target are hit in the same 1m bar,
        # count the stop first.
        if hit_stop and hit_target:
            return (
                -stop_points,
                "STOP_BOTH"
            )


        if hit_stop:
            return (
                -stop_points,
                "STOP"
            )


        if hit_target:
            return (
                target_points,
                "TARGET"
            )


    # Neither was hit.
    return (
        float(trade["Baseline90Return"]),
        "TIME"
    )


# ==================================================================================================
# FULL TARGET x STOP MATRIX
# ==================================================================================================

section("TESTING TARGET x ATR STOP MATRIX")

matrix_rows = []
year_rows = []

total_combos = (
    len(target_grid) *
    len(stop_grid)
)

combo_number = 0


for target_points in target_grid:

    for stop_atr in stop_grid:

        combo_number += 1

        returns = []
        exit_reasons = []
        years = []

        for _, trade in trades.iterrows():

            ret, reason = simulate_trade(
                trade,
                target_points,
                stop_atr
            )

            returns.append(ret)
            exit_reasons.append(reason)
            years.append(
                int(trade["Year"])
            )


        returns = np.asarray(
            returns,
            dtype=float
        )

        exit_reasons = np.asarray(
            exit_reasons,
            dtype=object
        )

        years = np.asarray(
            years,
            dtype=int
        )


        s = calc_stats(
            returns
        )


        target_hits = np.isin(
            exit_reasons,
            ["TARGET"]
        ).sum()

        stop_hits = np.isin(
            exit_reasons,
            ["STOP", "STOP_BOTH"]
        ).sum()

        time_exits = (
            exit_reasons == "TIME"
        ).sum()


        positive_years = 0
        tested_years = 0

        yearly_avgs = []


        for year in range(
            START_YEAR,
            END_YEAR + 1
        ):

            mask = (
                years == year
            )

            if mask.sum() == 0:
                continue


            ys = calc_stats(
                returns[mask]
            )

            tested_years += 1

            if ys["Avg"] > 0:
                positive_years += 1

            yearly_avgs.append(
                ys["Avg"]
            )


            year_rows.append(
                {
                    "TargetPts": target_points,
                    "StopATR": stop_atr,
                    "Year": year,
                    **ys
                }
            )


        row = {
            "TargetPts": target_points,
            "StopATR": stop_atr,

            **s,

            "TargetHitPct":
                target_hits /
                len(returns) *
                100.0,

            "StopHitPct":
                stop_hits /
                len(returns) *
                100.0,

            "TimeExitPct":
                time_exits /
                len(returns) *
                100.0,

            "PositiveYears":
                positive_years,

            "TestedYears":
                tested_years,

            "WorstYearAvg":
                min(yearly_avgs)
                if yearly_avgs
                else np.nan,

            "BestYearAvg":
                max(yearly_avgs)
                if yearly_avgs
                else np.nan
        }


        matrix_rows.append(
            row
        )


        if (
            combo_number % 25 == 0 or
            combo_number == total_combos
        ):
            print(
                f"Completed "
                f"{combo_number:,} / "
                f"{total_combos:,} combinations"
            )


results = pd.DataFrame(
    matrix_rows
)

year_results = pd.DataFrame(
    year_rows
)


# ==================================================================================================
# SAVE FULL RESULTS FIRST
# ==================================================================================================

results.to_csv(
    OUTPUT_MATRIX,
    index=False
)

year_results.to_csv(
    OUTPUT_YEAR,
    index=False
)

trades.to_csv(
    OUTPUT_TRADES,
    index=False
)


# ==================================================================================================
# BEST OVERALL
# ==================================================================================================

section("TOP 30 BY PROFIT FACTOR")

eligible = results[
    results["N"] >= 100
].copy()

top_pf = eligible.sort_values(
    [
        "PF",
        "PositiveYears",
        "Avg"
    ],
    ascending=[
        False,
        False,
        False
    ]
)

print(
    top_pf[
        [
            "TargetPts",
            "StopATR",
            "N",
            "Avg",
            "PF",
            "WinPct",
            "TotalPts",
            "MaxDD",
            "TargetHitPct",
            "StopHitPct",
            "TimeExitPct",
            "PositiveYears",
            "WorstYearAvg"
        ]
    ].head(
        30
    ).to_string(
        index=False
    )
)


# ==================================================================================================
# BEST STABLE
# ==================================================================================================

section("TOP 30 - 5/5 POSITIVE YEARS")

stable5 = eligible[
    eligible["PositiveYears"] == 5
].copy()

stable5 = stable5.sort_values(
    [
        "PF",
        "Avg"
    ],
    ascending=[
        False,
        False
    ]
)

if len(stable5) > 0:

    print(
        stable5[
            [
                "TargetPts",
                "StopATR",
                "N",
                "Avg",
                "PF",
                "WinPct",
                "TotalPts",
                "MaxDD",
                "TargetHitPct",
                "StopHitPct",
                "TimeExitPct",
                "WorstYearAvg"
            ]
        ].head(
            30
        ).to_string(
            index=False
        )
    )

else:
    print("No combinations were positive in all five years.")


# ==================================================================================================
# BEST 4/5 OR BETTER
# ==================================================================================================

section("TOP 30 - >=4/5 POSITIVE YEARS")

stable4 = eligible[
    eligible["PositiveYears"] >= 4
].copy()

stable4 = stable4.sort_values(
    [
        "PF",
        "Avg",
        "WorstYearAvg"
    ],
    ascending=[
        False,
        False,
        False
    ]
)

print(
    stable4[
        [
            "TargetPts",
            "StopATR",
            "N",
            "Avg",
            "PF",
            "WinPct",
            "TotalPts",
            "MaxDD",
            "TargetHitPct",
            "StopHitPct",
            "TimeExitPct",
            "PositiveYears",
            "WorstYearAvg"
        ]
    ].head(
        30
    ).to_string(
        index=False
    )
)


# ==================================================================================================
# BEST STOP FOR EACH TARGET
# ==================================================================================================

section("BEST STOP FOR EACH PROFIT TARGET")

best_target_rows = []

for target_points in target_grid:

    temp = results[
        results["TargetPts"] == target_points
    ].copy()

    temp = temp[
        temp["PositiveYears"] >= 4
    ]

    if len(temp) == 0:
        continue

    temp = temp.sort_values(
        [
            "PF",
            "Avg"
        ],
        ascending=[
            False,
            False
        ]
    )

    best_target_rows.append(
        temp.iloc[0]
    )


if best_target_rows:

    best_by_target = pd.DataFrame(
        best_target_rows
    )

    print(
        best_by_target[
            [
                "TargetPts",
                "StopATR",
                "Avg",
                "PF",
                "WinPct",
                "TotalPts",
                "MaxDD",
                "TargetHitPct",
                "StopHitPct",
                "PositiveYears",
                "WorstYearAvg"
            ]
        ].to_string(
            index=False
        )
    )


# ==================================================================================================
# BEST TARGET FOR EACH STOP
# ==================================================================================================

section("BEST TARGET FOR EACH ATR STOP")

best_stop_rows = []

for stop_atr in stop_grid:

    temp = results[
        np.isclose(
            results["StopATR"],
            stop_atr
        )
    ].copy()

    temp = temp[
        temp["PositiveYears"] >= 4
    ]

    if len(temp) == 0:
        continue

    temp = temp.sort_values(
        [
            "PF",
            "Avg"
        ],
        ascending=[
            False,
            False
        ]
    )

    best_stop_rows.append(
        temp.iloc[0]
    )


if best_stop_rows:

    best_by_stop = pd.DataFrame(
        best_stop_rows
    )

    print(
        best_by_stop[
            [
                "StopATR",
                "TargetPts",
                "Avg",
                "PF",
                "WinPct",
                "TotalPts",
                "MaxDD",
                "TargetHitPct",
                "StopHitPct",
                "PositiveYears",
                "WorstYearAvg"
            ]
        ].to_string(
            index=False
        )
    )


# ==================================================================================================
# TARGET HIT RATES WITHOUT REGARD TO STOP
# ==================================================================================================

section("RAW 90-MINUTE MFE / TARGET REACH RATE")

for target_points in target_grid:

    reach_pct = (
        trades["MFE90Points"] >= target_points
    ).mean() * 100.0

    print(
        f"Target {target_points:6.1f} pts | "
        f"Reached sometime within 90m: "
        f"{reach_pct:6.2f}%"
    )


# ==================================================================================================
# DONE
# ==================================================================================================

section("DONE")

print(
    f"Full matrix:\n{OUTPUT_MATRIX}"
)

print()

print(
    f"Year-by-year matrix:\n{OUTPUT_YEAR}"
)

print()

print(
    f"90-minute trade cohort:\n{OUTPUT_TRADES}"
)