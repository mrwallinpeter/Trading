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
    r"\MACD_15M_PriceMove_PriorMove_HistUpBars"
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

PRIOR_LOOKBACK = 20

YEARS = [2022, 2023, 2024, 2025, 2026]

HOLDS = [
    120,
    150,
    180,
    210,
    240
]

MAX_COUNT_SEARCH = 50


# ============================================================
# LOAD 1-MIN DATA
# ============================================================

print()
print("=" * 180)
print("LOADING 1-MIN DATA")
print("=" * 180)

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

for c in [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]:
    df1[c] = pd.to_numeric(
        df1[c],
        errors="coerce"
    )

df1 = df1.dropna(
    subset=[
        "Open",
        "High",
        "Low",
        "Close"
    ]
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
    subset=[
        "Open",
        "High",
        "Low",
        "Close"
    ]
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
# PRE-CROSS HISTOGRAM UP-BAR COUNT
#
# Cross bar itself is NOT counted.
# ============================================================

hist = df15["Hist"].to_numpy(
    dtype=float
)

pre_cross_up_bars = np.zeros(
    len(df15),
    dtype=int
)

for i in range(2, len(df15)):

    count = 0
    j = i - 1

    while (
        j >= 1
        and
        count < MAX_COUNT_SEARCH
        and
        np.isfinite(hist[j])
        and
        np.isfinite(hist[j - 1])
        and
        hist[j] > hist[j - 1]
    ):

        count += 1
        j -= 1

    pre_cross_up_bars[i] = count


df15["PreCrossUpBars"] = (
    pre_cross_up_bars
)

df15["HistUpLT5"] = (
    df15["PreCrossUpBars"] < 5
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


# ============================================================
# MASTER
# ============================================================

master = df15[
    df15.index.year.isin(YEARS)
    &
    df15["ATR14"].notna()
    &
    df15["PriorMoveATR14"].notna()
].copy()

master["Year"] = (
    master.index.year
)


# ============================================================
# BASE SIGNAL
# ============================================================

master["BaseSignal"] = (
    master["BullCross"]
    &
    master["PriceMovePass"]
)


# ============================================================
# COUNTS
# ============================================================

base_signals = master[
    master["BaseSignal"]
].copy()

print()
print("=" * 180)
print("BASE SIGNAL COUNTS")
print("=" * 180)

print(
    "Bullish MACD cross + PriceMove:",
    len(base_signals)
)

print()
print(
    "PreCrossUpBars < 5:",
    (
        base_signals["PreCrossUpBars"] < 5
    ).sum()
)

print(
    "PreCrossUpBars >= 5:",
    (
        base_signals["PreCrossUpBars"] >= 5
    ).sum()
)

print()
print(
    "PriorMove -2 to +2:",
    (
        (base_signals["PriorMoveATR14"] >= -2.0)
        &
        (base_signals["PriorMoveATR14"] <= 2.0)
    ).sum()
)

print(
    "PriorMove -0.5 to 0:",
    (
        (base_signals["PriorMoveATR14"] >= -0.5)
        &
        (base_signals["PriorMoveATR14"] < 0.0)
    ).sum()
)

print(
    "PriorMove -0.5 to +0.5:",
    (
        (base_signals["PriorMoveATR14"] >= -0.5)
        &
        (base_signals["PriorMoveATR14"] <= 0.5)
    ).sum()
)


# ============================================================
# 1-MIN EXECUTION
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

entry_positions = (
    entry_positions[
        valid_entry
    ]
)

master["EntryTime"] = pd.to_datetime(
    minute_times[
        entry_positions
    ]
)

master["EntryPrice"] = (
    minute_open[
        entry_positions
    ]
)


# ============================================================
# RETURNS
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

    final_idx = (
        valid_idx[
            tolerance_ok
        ]
    )

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

    master[
        f"PnL_{hold}"
    ] = pnl

    print(
        f"{hold:>3} min -> "
        f"{np.isfinite(pnl).sum()} valid rows"
    )


# ============================================================
# STATS
# ============================================================

def profit_factor(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    pnl = pnl[
        np.isfinite(pnl)
    ]

    gp = pnl[
        pnl > 0
    ].sum()

    gl = -pnl[
        pnl < 0
    ].sum()

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

    pnl = pnl[
        np.isfinite(pnl)
    ]

    if len(pnl) < 2:
        return np.nan

    sd = pnl.std(
        ddof=1
    )

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

    pnl = pnl[
        np.isfinite(pnl)
    ]

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

    pnl = pnl[
        np.isfinite(pnl)
    ]

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
# GROUP DEFINITIONS
# ============================================================

GROUPS = [
    "BASE",
    "BASE_LT5",

    "M2_P2",
    "M2_P2_LT5",

    "NEG050_0",
    "NEG050_0_LT5",

    "CENTER050",
    "CENTER050_LT5",

    "HIST_5_PLUS"
]


# ============================================================
# MASK
# ============================================================

def make_mask(df, group):

    base = df[
        "BaseSignal"
    ].to_numpy(
        dtype=bool
    )

    pm = df[
        "PriorMoveATR14"
    ].to_numpy(
        dtype=float
    )

    upbars = df[
        "PreCrossUpBars"
    ].to_numpy(
        dtype=int
    )

    lt5 = (
        upbars < 5
    )

    ge5 = (
        upbars >= 5
    )

    m2_p2 = (
        (pm >= -2.0)
        &
        (pm <= 2.0)
    )

    neg050_0 = (
        (pm >= -0.5)
        &
        (pm < 0.0)
    )

    center050 = (
        (pm >= -0.5)
        &
        (pm <= 0.5)
    )

    if group == "BASE":
        return base

    if group == "BASE_LT5":
        return (
            base
            &
            lt5
        )

    if group == "M2_P2":
        return (
            base
            &
            m2_p2
        )

    if group == "M2_P2_LT5":
        return (
            base
            &
            m2_p2
            &
            lt5
        )

    if group == "NEG050_0":
        return (
            base
            &
            neg050_0
        )

    if group == "NEG050_0_LT5":
        return (
            base
            &
            neg050_0
            &
            lt5
        )

    if group == "CENTER050":
        return (
            base
            &
            center050
        )

    if group == "CENTER050_LT5":
        return (
            base
            &
            center050
            &
            lt5
        )

    if group == "HIST_5_PLUS":
        return (
            base
            &
            ge5
        )

    raise ValueError(
        f"Unknown group: {group}"
    )


# ============================================================
# RUN TEST
# ============================================================

summary_rows = []
year_rows = []


for hold in HOLDS:

    pnl_col = (
        f"PnL_{hold}"
    )

    eligible = master[
        master[pnl_col].notna()
    ].copy()

    pnl_all = eligible[
        pnl_col
    ].to_numpy(
        dtype=float
    )

    years = eligible[
        "Year"
    ].to_numpy()


    for group in GROUPS:

        mask = make_mask(
            eligible,
            group
        )

        idx = np.where(
            mask
        )[0]

        pnl = (
            pnl_all[
                idx
            ]
        )

        signal_years = (
            years[
                idx
            ]
        )

        s = stats(
            pnl
        )

        year_pfs = []

        for year in YEARS:

            ypnl = pnl[
                signal_years
                ==
                year
            ]

            ys = stats(
                ypnl
            )

            year_rows.append({
                "Group": group,
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

        profitable_years = sum(
            pf > 1
            for pf in year_pfs
        )

        summary_rows.append({
            "Group": group,
            "HoldingMinutes": hold,
            **s,
            "WorstYearPF": worst_pf,
            "ProfitableYears": profitable_years
        })


# ============================================================
# DATAFRAMES
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# SAVE
# ============================================================

summary.to_csv(
    OUTPUT_DIR / "summary.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR / "year_by_year.csv",
    index=False
)

master[
    master["BaseSignal"]
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
    1200
)


# ============================================================
# PROFIT FACTOR
# ============================================================

print()
print("=" * 300)
print(
    "PRICE MOVE + PRIOR MOVE + HISTOGRAM BUILDUP — PROFIT FACTOR"
)
print("=" * 300)

print(
    summary.pivot(
        index="HoldingMinutes",
        columns="Group",
        values="ProfitFactor"
    )
    .reindex(
        columns=GROUPS
    )
    .round(4)
    .to_string()
)


# ============================================================
# AVERAGE PNL
# ============================================================

print()
print("=" * 300)
print("AVERAGE PNL")
print("=" * 300)

print(
    summary.pivot(
        index="HoldingMinutes",
        columns="Group",
        values="AveragePnL"
    )
    .reindex(
        columns=GROUPS
    )
    .round(4)
    .to_string()
)


# ============================================================
# SAMPLE SIZE
# ============================================================

print()
print("=" * 300)
print("SAMPLE SIZE")
print("=" * 300)

print(
    summary.pivot(
        index="HoldingMinutes",
        columns="Group",
        values="N"
    )
    .reindex(
        columns=GROUPS
    )
    .to_string()
)


# ============================================================
# WORST YEAR
# ============================================================

print()
print("=" * 300)
print("WORST-YEAR PF")
print("=" * 300)

print(
    summary.pivot(
        index="HoldingMinutes",
        columns="Group",
        values="WorstYearPF"
    )
    .reindex(
        columns=GROUPS
    )
    .round(4)
    .to_string()
)


# ============================================================
# PROFITABLE YEARS
# ============================================================

print()
print("=" * 300)
print("PROFITABLE YEARS")
print("=" * 300)

print(
    summary.pivot(
        index="HoldingMinutes",
        columns="Group",
        values="ProfitableYears"
    )
    .reindex(
        columns=GROUPS
    )
    .to_string()
)


# ============================================================
# YEAR-BY-YEAR
# ============================================================

for group in GROUPS:

    print()
    print("=" * 240)
    print(
        f"YEAR-BY-YEAR PF — {group}"
    )
    print("=" * 240)

    d = year_df[
        year_df["Group"] == group
    ].pivot(
        index="HoldingMinutes",
        columns="Year",
        values="ProfitFactor"
    )

    print(
        d
        .round(4)
        .to_string()
    )


# ============================================================
# DIRECT COMPARISON TABLE
# ============================================================

print()
print("=" * 300)
print(
    "DIRECT COMPARISON — EFFECT OF EXCLUDING 5+ PRE-CROSS UP BARS"
)
print("=" * 300)

pairs = [
    ("BASE", "BASE_LT5"),
    ("M2_P2", "M2_P2_LT5"),
    ("NEG050_0", "NEG050_0_LT5"),
    ("CENTER050", "CENTER050_LT5")
]

comparison_rows = []

for hold in HOLDS:

    for before_group, after_group in pairs:

        before = summary[
            (summary["HoldingMinutes"] == hold)
            &
            (summary["Group"] == before_group)
        ].iloc[0]

        after = summary[
            (summary["HoldingMinutes"] == hold)
            &
            (summary["Group"] == after_group)
        ].iloc[0]

        comparison_rows.append({
            "HoldingMinutes": hold,
            "BeforeGroup": before_group,
            "AfterGroup": after_group,

            "N_Before": int(before["N"]),
            "N_After": int(after["N"]),
            "N_Removed": int(
                before["N"]
                -
                after["N"]
            ),

            "PF_Before": before["ProfitFactor"],
            "PF_After": after["ProfitFactor"],
            "PF_Change": (
                after["ProfitFactor"]
                -
                before["ProfitFactor"]
            ),

            "Avg_Before": before["AveragePnL"],
            "Avg_After": after["AveragePnL"],
            "Avg_Change": (
                after["AveragePnL"]
                -
                before["AveragePnL"]
            ),

            "WorstPF_Before": before["WorstYearPF"],
            "WorstPF_After": after["WorstYearPF"],
            "WorstPF_Change": (
                after["WorstYearPF"]
                -
                before["WorstYearPF"]
            ),

            "Years_Before": int(
                before["ProfitableYears"]
            ),
            "Years_After": int(
                after["ProfitableYears"]
            )
        })


comparison = pd.DataFrame(
    comparison_rows
)

comparison.to_csv(
    OUTPUT_DIR / "direct_comparison.csv",
    index=False
)

print(
    comparison.round(4).to_string(
        index=False
    )
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 240)
print("DONE")
print("=" * 240)

print()
print(
    "Base = bullish 15m MACD(5,36,5) histogram zero-cross"
)

print(
    "PriceMoveATR14 = 0.35 to 0.90"
)

print()
print(
    "PriorMoveATR14 = "
    "(Close[t-1] - Close[t-20]) / ATR14[t]"
)

print()
print(
    "Histogram filter = PreCrossUpBars < 5"
)

print(
    "Cross bar itself is NOT counted in PreCrossUpBars"
)

print()
print("Groups:")

print(
    "BASE = MACD cross + PriceMove"
)

print(
    "BASE_LT5 = BASE + PreCrossUpBars < 5"
)

print(
    "M2_P2 = BASE + PriorMove -2 to +2"
)

print(
    "M2_P2_LT5 = M2_P2 + PreCrossUpBars < 5"
)

print(
    "NEG050_0 = BASE + PriorMove -0.5 to 0"
)

print(
    "NEG050_0_LT5 = NEG050_0 + PreCrossUpBars < 5"
)

print(
    "CENTER050 = BASE + PriorMove -0.5 to +0.5"
)

print(
    "CENTER050_LT5 = CENTER050 + PreCrossUpBars < 5"
)

print(
    "HIST_5_PLUS = BASE + PreCrossUpBars >= 5"
)

print()
print(
    "NO range filter"
)

print(
    "NO simulated 4h filter"
)

print(
    "NO time filter"
)

print()
print(
    "Holds = 120, 150, 180, 210, 240 minutes"
)

print()
print("Output folder:")
print(OUTPUT_DIR)