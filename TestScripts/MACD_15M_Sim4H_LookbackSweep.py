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
    r"\MACD_15M_Sim4H_LookbackSweep"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

# 15m MACD signal
MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

# Simulated 4h MACD on 15m bars
SIM4H_FAST = 80
SIM4H_SLOW = 576
SIM4H_SIGNAL = 80

# Histogram-direction lookbacks to test
SIM4H_LOOKBACKS = [4, 6, 8, 10, 12]

# ATR
ATR_LENGTH = 14

# Filter 1: cross-bar move
PRICE_MOVE_MIN = 0.35
PRICE_MOVE_MAX = 0.90

# Filter 2: prior 5h move
PRIOR_LOOKBACK = 20
PRIOR_MIN = -2.0
PRIOR_MAX = 2.0

YEARS = [2022, 2023, 2024, 2025, 2026]

HOLDS = [120, 150, 180, 210, 240]


# ============================================================
# LOAD 1-MIN DATA
# ============================================================

print()
print("=" * 180)
print("LOADING 1-MIN DATA")
print("=" * 180)

df1 = pd.read_csv(ONE_MIN_FILE)

df1["DateTime"] = pd.to_datetime(df1["DateTime"])

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

print("Rows:", len(df1))
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
# MACD FUNCTION
# ============================================================

def macd(close, fast, slow, signal):

    ema_fast = close.ewm(
        span=fast,
        adjust=False
    ).mean()

    ema_slow = close.ewm(
        span=slow,
        adjust=False
    ).mean()

    macd_line = ema_fast - ema_slow

    signal_line = macd_line.ewm(
        span=signal,
        adjust=False
    ).mean()

    hist = macd_line - signal_line

    return macd_line, signal_line, hist


# ============================================================
# 15M MACD
# ============================================================

(
    df15["MACD15"],
    df15["Signal15"],
    df15["Hist15"]
) = macd(
    df15["Close"],
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL
)

df15["BullCross15"] = (
    (df15["Hist15"].shift(1) <= 0)
    &
    (df15["Hist15"] > 0)
)


# ============================================================
# SIMULATED 4H MACD
# ============================================================

(
    df15["Sim4H_MACD"],
    df15["Sim4H_Signal"],
    df15["Sim4H_Hist"]
) = macd(
    df15["Close"],
    SIM4H_FAST,
    SIM4H_SLOW,
    SIM4H_SIGNAL
)


# ============================================================
# WILDER ATR14
# ============================================================

prev_close = df15["Close"].shift(1)

tr1 = df15["High"] - df15["Low"]
tr2 = (df15["High"] - prev_close).abs()
tr3 = (df15["Low"] - prev_close).abs()

df15["TR"] = pd.concat(
    [tr1, tr2, tr3],
    axis=1
).max(axis=1)

df15["ATR14"] = df15["TR"].ewm(
    alpha=1.0 / ATR_LENGTH,
    adjust=False
).mean()


# ============================================================
# FILTER 1: PRICE MOVE
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
# FILTER 2: PRIOR MOVE
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
    (df15["PriorMoveATR14"] >= PRIOR_MIN)
    &
    (df15["PriorMoveATR14"] <= PRIOR_MAX)
)


# ============================================================
# QUALIFIED SIGNAL
# ============================================================

df15["QualifiedSignal"] = (
    df15["BullCross15"]
    &
    df15["PriceMovePass"]
    &
    df15["PriorMovePass"]
)


# ============================================================
# MASTER DATA
# ============================================================

master = df15[
    df15.index.year.isin(YEARS)
    &
    df15["ATR14"].notna()
    &
    df15["PriorMoveATR14"].notna()
].copy()

master["Year"] = master.index.year


# ============================================================
# BUILD STATE COLUMNS FOR EVERY LOOKBACK
# ============================================================

for lb in SIM4H_LOOKBACKS:

    hist_ago_col = f"Sim4H_Hist_{lb}Ago"
    change_col = f"Sim4H_HistChange_{lb}"
    state_col = f"Sim4H_State_{lb}"

    master[hist_ago_col] = (
        master["Sim4H_Hist"]
        .shift(lb)
    )

    master[change_col] = (
        master["Sim4H_Hist"]
        -
        master[hist_ago_col]
    )

    hist_up = (
        master[change_col] > 0
    )

    hist_down = (
        master[change_col] < 0
    )

    conditions = [
        (
            (master["Sim4H_MACD"] > 0)
            &
            hist_up
        ),
        (
            (master["Sim4H_MACD"] > 0)
            &
            hist_down
        ),
        (
            (master["Sim4H_MACD"] <= 0)
            &
            hist_up
        ),
        (
            (master["Sim4H_MACD"] <= 0)
            &
            hist_down
        )
    ]

    choices = [
        "A_POS_UP",
        "B_POS_DOWN",
        "C_NEG_UP",
        "D_NEG_DOWN"
    ]

    master[state_col] = np.select(
        conditions,
        choices,
        default="FLAT"
    )


# ============================================================
# SIGNAL COUNTS BY LOOKBACK
# ============================================================

print()
print("=" * 220)
print("QUALIFIED SIGNAL COUNTS BY SIMULATED 4H LOOKBACK")
print("=" * 220)

signals = master[
    master["QualifiedSignal"]
].copy()

print("Total qualified signals:", len(signals))

for lb in SIM4H_LOOKBACKS:

    state_col = f"Sim4H_State_{lb}"

    print()
    print(
        f"--- LOOKBACK {lb} BARS "
        f"({lb * 15 / 60:.1f} hours) ---"
    )

    print(
        signals[state_col]
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# 1-MIN EXECUTION ARRAYS
# ============================================================

minute_times = df1.index.values.astype(
    "datetime64[ns]"
)

minute_open = df1["Open"].to_numpy(
    dtype=float
)

minute_close = df1["Close"].to_numpy(
    dtype=float
)


# ============================================================
# ENTRY
# First following 1m Open after completed 15m bar
# ============================================================

bar_times_np = master.index.values.astype(
    "datetime64[ns]"
)

entry_positions = np.searchsorted(
    minute_times,
    bar_times_np,
    side="right"
)

valid_entry = (
    entry_positions
    <
    len(minute_times)
)

master = master.iloc[
    np.where(valid_entry)[0]
].copy()

entry_positions = entry_positions[
    valid_entry
]

master["EntryTime"] = pd.to_datetime(
    minute_times[
        entry_positions
    ]
)

master["EntryPrice"] = minute_open[
    entry_positions
]


# ============================================================
# PRECOMPUTE RETURNS
# ============================================================

print()
print("=" * 180)
print("PRECOMPUTING RETURNS")
print("=" * 180)

entry_times_np = master[
    "EntryTime"
].values.astype(
    "datetime64[ns]"
)

entry_prices_np = master[
    "EntryPrice"
].to_numpy(
    dtype=float
)

for hold in HOLDS:

    target_times = (
        entry_times_np
        +
        np.timedelta64(
            hold,
            "m"
        )
    )

    exit_positions = np.searchsorted(
        minute_times,
        target_times,
        side="left"
    )

    pnl = np.full(
        len(master),
        np.nan
    )

    valid_exit = (
        exit_positions
        <
        len(minute_times)
    )

    valid_idx = np.where(
        valid_exit
    )[0]

    actual_exit_times = minute_times[
        exit_positions[
            valid_idx
        ]
    ]

    tolerance_ok = (
        actual_exit_times
        <=
        target_times[
            valid_idx
        ]
        +
        np.timedelta64(
            2,
            "m"
        )
    )

    final_idx = valid_idx[
        tolerance_ok
    ]

    pnl[
        final_idx
    ] = (
        minute_close[
            exit_positions[
                final_idx
            ]
        ]
        -
        entry_prices_np[
            final_idx
        ]
    )

    master[f"PnL_{hold}"] = pnl

    print(
        f"{hold:>3} min -> "
        f"{np.isfinite(pnl).sum()} valid bars"
    )


# ============================================================
# STATS FUNCTIONS
# ============================================================

def profit_factor(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    pnl = pnl[np.isfinite(pnl)]

    gp = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()

    if gl == 0:

        if gp > 0:
            return np.inf

        return np.nan

    return gp / gl


def sqn(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    pnl = pnl[np.isfinite(pnl)]

    if len(pnl) < 2:
        return np.nan

    sd = pnl.std(ddof=1)

    if sd == 0:
        return np.nan

    return (
        np.sqrt(len(pnl))
        *
        pnl.mean()
        /
        sd
    )


def max_drawdown(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    pnl = pnl[np.isfinite(pnl)]

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

    return abs(
        (
            equity
            -
            peak
        ).min()
    )


def stats(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    pnl = pnl[np.isfinite(pnl)]

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
# TEST LOOKBACK x STATE x HOLD
# ============================================================

STATES = [
    "A_POS_UP",
    "B_POS_DOWN",
    "C_NEG_UP",
    "D_NEG_DOWN"
]

summary_rows = []
year_rows = []


for lb in SIM4H_LOOKBACKS:

    state_col = f"Sim4H_State_{lb}"

    for hold in HOLDS:

        pnl_col = f"PnL_{hold}"

        eligible = master[
            master[pnl_col].notna()
            &
            master[state_col].notna()
        ].copy()

        qualified = eligible[
            "QualifiedSignal"
        ].to_numpy(
            dtype=bool
        )

        states = eligible[
            state_col
        ].to_numpy()

        pnl_all = eligible[
            pnl_col
        ].to_numpy(
            dtype=float
        )

        years = eligible[
            "Year"
        ].to_numpy()


        for state in STATES:

            mask = (
                qualified
                &
                (states == state)
            )

            idx = np.where(mask)[0]

            pnl = pnl_all[idx]
            signal_years = years[idx]

            s = stats(pnl)

            year_pfs = []

            for year in YEARS:

                ypnl = pnl[
                    signal_years == year
                ]

                ys = stats(ypnl)

                year_rows.append({
                    "LookbackBars": lb,
                    "LookbackHours": lb * 15 / 60,
                    "State": state,
                    "HoldingMinutes": hold,
                    "Year": year,
                    **ys
                })

                if np.isfinite(
                    ys["ProfitFactor"]
                ):
                    year_pfs.append(
                        ys["ProfitFactor"]
                    )

            worst_pf = (
                min(year_pfs)
                if year_pfs
                else np.nan
            )

            best_pf = (
                max(year_pfs)
                if year_pfs
                else np.nan
            )

            profitable_years = sum(
                pf > 1
                for pf in year_pfs
            )

            summary_rows.append({
                "LookbackBars": lb,
                "LookbackHours": lb * 15 / 60,
                "State": state,
                "HoldingMinutes": hold,
                **s,
                "WorstYearPF": worst_pf,
                "BestYearPF": best_pf,
                "ProfitableYears": profitable_years
            })


# ============================================================
# OUTPUT DATAFRAMES
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

year_df = pd.DataFrame(
    year_rows
)

summary.to_csv(
    OUTPUT_DIR / "lookback_sweep_summary.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR / "lookback_sweep_year_by_year.csv",
    index=False
)

master[
    master["QualifiedSignal"]
].to_csv(
    OUTPUT_DIR / "signal_detail.csv"
)


# ============================================================
# DISPLAY SETTINGS
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.max_rows",
    5000
)

pd.set_option(
    "display.width",
    1000
)


# ============================================================
# PF SURFACE BY STATE
# ============================================================

for state in STATES:

    print()
    print("=" * 260)
    print(
        f"PROFIT FACTOR — {state}"
    )
    print("=" * 260)

    d = summary[
        summary["State"] == state
    ].pivot(
        index="LookbackBars",
        columns="HoldingMinutes",
        values="ProfitFactor"
    )

    print(
        d
        .round(4)
        .to_string()
    )


# ============================================================
# WORST-YEAR PF SURFACE
# ============================================================

for state in STATES:

    print()
    print("=" * 260)
    print(
        f"WORST-YEAR PF — {state}"
    )
    print("=" * 260)

    d = summary[
        summary["State"] == state
    ].pivot(
        index="LookbackBars",
        columns="HoldingMinutes",
        values="WorstYearPF"
    )

    print(
        d
        .round(4)
        .to_string()
    )


# ============================================================
# PROFITABLE YEARS
# ============================================================

for state in STATES:

    print()
    print("=" * 260)
    print(
        f"PROFITABLE YEARS — {state}"
    )
    print("=" * 260)

    d = summary[
        summary["State"] == state
    ].pivot(
        index="LookbackBars",
        columns="HoldingMinutes",
        values="ProfitableYears"
    )

    print(
        d.to_string()
    )


# ============================================================
# SAMPLE SIZE
# ============================================================

for state in STATES:

    print()
    print("=" * 260)
    print(
        f"SAMPLE SIZE — {state}"
    )
    print("=" * 260)

    d = summary[
        summary["State"] == state
    ].pivot(
        index="LookbackBars",
        columns="HoldingMinutes",
        values="N"
    )

    print(
        d.to_string()
    )


# ============================================================
# YEAR-BY-YEAR PF
# ============================================================

for state in STATES:

    print()
    print("=" * 280)
    print(
        f"YEAR-BY-YEAR PF — {state}"
    )
    print("=" * 280)

    d = year_df[
        year_df["State"] == state
    ].pivot_table(
        index=[
            "LookbackBars",
            "HoldingMinutes"
        ],
        columns="Year",
        values="ProfitFactor"
    )

    print(
        d
        .round(4)
        .to_string()
    )


# ============================================================
# TOP RESULTS OVERALL
# ============================================================

print()
print("=" * 300)
print("TOP RESULTS OVERALL — SORTED BY WORST-YEAR PF")
print("=" * 300)

top = summary.sort_values(
    [
        "WorstYearPF",
        "ProfitFactor",
        "N"
    ],
    ascending=[
        False,
        False,
        False
    ]
)

print(
    top[
        [
            "LookbackBars",
            "LookbackHours",
            "State",
            "HoldingMinutes",
            "N",
            "WinRatePct",
            "AveragePnL",
            "ProfitFactor",
            "SQN",
            "MaxDrawdown",
            "WorstYearPF",
            "BestYearPF",
            "ProfitableYears"
        ]
    ]
    .head(40)
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# BEST RESULT PER STATE
# ============================================================

print()
print("=" * 300)
print("BEST RESULTS PER STATE — SORTED BY WORST-YEAR PF")
print("=" * 300)

for state in STATES:

    print()
    print(
        f"--- {state} ---"
    )

    d = (
        summary[
            summary["State"] == state
        ]
        .sort_values(
            [
                "WorstYearPF",
                "ProfitFactor"
            ],
            ascending=[
                False,
                False
            ]
        )
    )

    print(
        d[
            [
                "LookbackBars",
                "LookbackHours",
                "HoldingMinutes",
                "N",
                "WinRatePct",
                "AveragePnL",
                "ProfitFactor",
                "SQN",
                "MaxDrawdown",
                "WorstYearPF",
                "BestYearPF",
                "ProfitableYears"
            ]
        ]
        .head(15)
        .round(4)
        .to_string(
            index=False
        )
    )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 260)
print("DONE")
print("=" * 260)

print()
print(
    "Signal = bullish 15m MACD(5,36,5) histogram zero-cross"
)

print(
    "Filter 1 = 0.35 <= PriceMoveATR14 <= 0.90"
)

print(
    "Filter 2 = -2 <= PriorMoveATR14 <= +2"
)

print()
print(
    "NO range filter"
)

print(
    "NO time-of-day filter"
)

print()
print(
    "Simulated 4h MACD = MACD(80,576,80) on 15m bars"
)

print(
    "Histogram direction lookbacks tested:"
)

for lb in SIM4H_LOOKBACKS:
    print(
        f"  {lb:>2} bars = {lb * 15 / 60:.1f} hours"
    )

print()
print(
    "A_POS_UP   = Sim4H MACD > 0 and histogram rising"
)

print(
    "B_POS_DOWN = Sim4H MACD > 0 and histogram falling"
)

print(
    "C_NEG_UP   = Sim4H MACD <= 0 and histogram rising"
)

print(
    "D_NEG_DOWN = Sim4H MACD <= 0 and histogram falling"
)

print()
print(
    "Holds = 120, 150, 180, 210, 240 minutes"
)

print()
print("Output folder:")
print(OUTPUT_DIR)