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
    r"\MACD_30M_DNEGNEG_vs_Random"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

SESSION_START = 6
SESSION_END = 22

HOLDS = [
    120,
    150,
    180
]

YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026
]

N_RANDOM_RUNS = 1000
RANDOM_SEED = 42


# ============================================================
# LOAD 1-MIN DATA
# ============================================================

print()
print("=" * 180)
print("LOADING 1-MIN DATA")
print("=" * 180)

df1 = pd.read_csv(
    ONE_MIN_FILE
)

df1["DateTime"] = pd.to_datetime(
    df1["DateTime"]
)

df1 = (
    df1
    .sort_values("DateTime")
    .drop_duplicates("DateTime")
    .set_index("DateTime")
)

for col in [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]:
    df1[col] = pd.to_numeric(
        df1[col],
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
# RESAMPLE
# ============================================================

def build_bars(freq):

    d = df1.resample(
        freq,
        label="right",
        closed="right"
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })

    return d.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close"
        ]
    )


df30 = build_bars("30min")
df4h = build_bars("4h")

print()
print("30m bars:", len(df30))
print("4h bars: ", len(df4h))


# ============================================================
# MACD
# ============================================================

def add_macd(d):

    d = d.copy()

    fast = d["Close"].ewm(
        span=MACD_FAST,
        adjust=False
    ).mean()

    slow = d["Close"].ewm(
        span=MACD_SLOW,
        adjust=False
    ).mean()

    d["MACD"] = (
        fast
        -
        slow
    )

    d["SignalLine"] = d["MACD"].ewm(
        span=MACD_SIGNAL,
        adjust=False
    ).mean()

    d["Hist"] = (
        d["MACD"]
        -
        d["SignalLine"]
    )

    d["BullCross"] = (
        (d["Hist"].shift(1) <= 0)
        &
        (d["Hist"] > 0)
    )

    return d


df30 = add_macd(df30)
df4h = add_macd(df4h)


# ============================================================
# MAP LAST COMPLETED 4H STATE
# ============================================================

df30["MACD4H"] = (
    df4h["MACD"]
    .reindex(
        df30.index,
        method="ffill"
    )
)

df30["Hist4H"] = (
    df4h["Hist"]
    .reindex(
        df30.index,
        method="ffill"
    )
)


# ============================================================
# D_NEG_NEG REGIME
# ============================================================

df30["D_NEG_NEG"] = (
    (df30["MACD4H"] <= 0)
    &
    (df30["Hist4H"] <= 0)
)


# ============================================================
# MASTER SESSION UNIVERSE
# ============================================================

master = df30[
    df30.index.year.isin(YEARS)
    &
    (df30.index.hour >= SESSION_START)
    &
    (df30.index.hour < SESSION_END)
    &
    df30["MACD4H"].notna()
    &
    df30["Hist4H"].notna()
    &
    df30["D_NEG_NEG"]
].copy()

master["Year"] = master.index.year
master["Hour"] = master.index.hour


print()
print("=" * 180)
print("D_NEG_NEG UNIVERSE")
print("=" * 180)

print(
    "30m bars in D_NEG_NEG:",
    len(master)
)

print(
    "Bullish 30m MACD crosses:",
    int(
        master["BullCross"].sum()
    )
)

print()
print("Crosses by year:")

print(
    master[
        master["BullCross"]
    ]["Year"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# 1M ARRAYS
# ============================================================

minute_times = df1.index.values.astype(
    "datetime64[ns]"
)

minute_open = df1[
    "Open"
].to_numpy(
    dtype=float
)

minute_close = df1[
    "Close"
].to_numpy(
    dtype=float
)


# ============================================================
# ENTRY = FIRST 1M OPEN AFTER COMPLETED 30M BAR
# ============================================================

signal_times_np = master.index.values.astype(
    "datetime64[ns]"
)

entry_positions = np.searchsorted(
    minute_times,
    signal_times_np,
    side="right"
)

valid_entry = (
    entry_positions
    <
    len(minute_times)
)

master = master.iloc[
    np.where(
        valid_entry
    )[0]
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


print()
print("=" * 180)
print("PRECOMPUTING RETURNS")
print("=" * 180)


for hold in HOLDS:

    targets = (
        entry_times_np
        +
        np.timedelta64(
            hold,
            "m"
        )
    )

    exit_positions = np.searchsorted(
        minute_times,
        targets,
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
        targets[
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

    master[
        f"PnL_{hold}"
    ] = pnl

    print(
        f"{hold:>3} min -> "
        f"{np.isfinite(pnl).sum()} valid D_NEG_NEG bars"
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

    if len(pnl) == 0:
        return np.nan

    gross_profit = pnl[
        pnl > 0
    ].sum()

    gross_loss = -pnl[
        pnl < 0
    ].sum()

    if gross_loss == 0:

        if gross_profit > 0:
            return np.inf

        return np.nan

    return (
        gross_profit
        /
        gross_loss
    )


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
        np.sqrt(
            len(pnl)
        )
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

    peaks = np.maximum.accumulate(
        equity
    )

    dd = (
        equity
        -
        peaks
    )

    return abs(
        dd.min()
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

        "N":
            len(pnl),

        "WinRatePct":
            np.mean(
                pnl > 0
            ) * 100,

        "NetPoints":
            pnl.sum(),

        "AveragePnL":
            pnl.mean(),

        "MedianPnL":
            np.median(
                pnl
            ),

        "ProfitFactor":
            profit_factor(
                pnl
            ),

        "SQN":
            sqn(
                pnl
            ),

        "MaxDrawdown":
            max_drawdown(
                pnl
            )
    }


def percentile(
    random_values,
    actual
):

    r = np.asarray(
        random_values,
        dtype=float
    )

    r = r[
        np.isfinite(r)
    ]

    if len(r) == 0:
        return np.nan

    return (
        np.mean(
            r <= actual
        )
        *
        100
    )


# ============================================================
# TEST
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

summary_rows = []
year_rows = []


for hold in HOLDS:

    pnl_col = f"PnL_{hold}"

    eligible = master[
        master[pnl_col].notna()
    ].copy()

    pnl_all = eligible[
        pnl_col
    ].to_numpy(
        dtype=float
    )

    bull_all = eligible[
        "BullCross"
    ].to_numpy(
        dtype=bool
    )

    years_all = eligible[
        "Year"
    ].to_numpy()

    hours_all = eligible[
        "Hour"
    ].to_numpy()


    # ========================================================
    # ACTUAL MACD SIGNALS
    # ========================================================

    signal_idx = np.where(
        bull_all
    )[0]

    actual_pnl = pnl_all[
        signal_idx
    ]

    actual_years = years_all[
        signal_idx
    ]

    actual_hours = hours_all[
        signal_idx
    ]

    actual_stats = stats(
        actual_pnl
    )


    # ========================================================
    # RANDOM MATCHING STRATA
    #
    # Same:
    # - year
    # - hour
    # - D_NEG_NEG regime already guaranteed by master
    #
    # Exclude actual MACD cross bars.
    # ========================================================

    strata = []

    unique_pairs = sorted(
        set(
            zip(
                actual_years,
                actual_hours
            )
        )
    )


    for year, hour in unique_pairs:

        required_n = int(
            np.sum(
                (
                    actual_years
                    ==
                    year
                )
                &
                (
                    actual_hours
                    ==
                    hour
                )
            )
        )

        pool = np.where(
            (
                years_all
                ==
                year
            )
            &
            (
                hours_all
                ==
                hour
            )
            &
            (
                ~bull_all
            )
        )[0]

        if len(pool) < required_n:

            raise RuntimeError(
                f"Hold={hold}, "
                f"year={year}, "
                f"hour={hour}: "
                f"need {required_n}, "
                f"have {len(pool)}"
            )

        strata.append(
            (
                pool,
                required_n
            )
        )


    # ========================================================
    # RANDOM SIMULATION
    # ========================================================

    random_avg = []
    random_pf = []
    random_net = []
    random_sqn = []

    random_year_pf = {
        year: []
        for year in YEARS
    }

    random_year_avg = {
        year: []
        for year in YEARS
    }


    for _ in range(
        N_RANDOM_RUNS
    ):

        chosen = []


        for pool, required_n in strata:

            picked = rng.choice(
                pool,
                size=required_n,
                replace=False
            )

            chosen.extend(
                picked.tolist()
            )


        chosen = np.asarray(
            chosen,
            dtype=int
        )

        rpnl = pnl_all[
            chosen
        ]

        ryears = years_all[
            chosen
        ]


        random_avg.append(
            rpnl.mean()
        )

        random_net.append(
            rpnl.sum()
        )

        random_pf.append(
            profit_factor(
                rpnl
            )
        )

        random_sqn.append(
            sqn(
                rpnl
            )
        )


        for year in YEARS:

            ypnl = rpnl[
                ryears
                ==
                year
            ]

            if len(ypnl) > 0:

                random_year_pf[
                    year
                ].append(
                    profit_factor(
                        ypnl
                    )
                )

                random_year_avg[
                    year
                ].append(
                    ypnl.mean()
                )


    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    summary_rows.append({

        "HoldingMinutes":
            hold,

        "N":
            actual_stats["N"],

        "WinRatePct":
            actual_stats["WinRatePct"],

        "NetPoints":
            actual_stats["NetPoints"],

        "AveragePnL":
            actual_stats["AveragePnL"],

        "MedianPnL":
            actual_stats["MedianPnL"],

        "ProfitFactor":
            actual_stats["ProfitFactor"],

        "SQN":
            actual_stats["SQN"],

        "MaxDrawdown":
            actual_stats["MaxDrawdown"],

        "RandomMedianAvg":
            np.nanmedian(
                random_avg
            ),

        "RandomMedianPF":
            np.nanmedian(
                random_pf
            ),

        "RandomMedianSQN":
            np.nanmedian(
                random_sqn
            ),

        "MACDPercentile_Avg":
            percentile(
                random_avg,
                actual_stats[
                    "AveragePnL"
                ]
            ),

        "MACDPercentile_PF":
            percentile(
                random_pf,
                actual_stats[
                    "ProfitFactor"
                ]
            ),

        "MACDPercentile_SQN":
            percentile(
                random_sqn,
                actual_stats[
                    "SQN"
                ]
            ),

        "AvgEdgeVsRandom":
            (
                actual_stats[
                    "AveragePnL"
                ]
                -
                np.nanmedian(
                    random_avg
                )
            ),

        "PFEdgeVsRandom":
            (
                actual_stats[
                    "ProfitFactor"
                ]
                -
                np.nanmedian(
                    random_pf
                )
            )
    })


    # ========================================================
    # YEAR-BY-YEAR
    # ========================================================

    for year in YEARS:

        ypnl = actual_pnl[
            actual_years
            ==
            year
        ]

        ys = stats(
            ypnl
        )

        random_pf_year = np.asarray(
            random_year_pf[
                year
            ],
            dtype=float
        )

        random_avg_year = np.asarray(
            random_year_avg[
                year
            ],
            dtype=float
        )

        year_rows.append({

            "HoldingMinutes":
                hold,

            "Year":
                year,

            "N":
                ys["N"],

            "AveragePnL":
                ys["AveragePnL"],

            "ProfitFactor":
                ys["ProfitFactor"],

            "SQN":
                ys["SQN"],

            "RandomMedianAvg":
                np.nanmedian(
                    random_avg_year
                ),

            "RandomMedianPF":
                np.nanmedian(
                    random_pf_year
                ),

            "MACDPercentile_Avg":
                percentile(
                    random_avg_year,
                    ys["AveragePnL"]
                ),

            "MACDPercentile_PF":
                percentile(
                    random_pf_year,
                    ys["ProfitFactor"]
                ),

            "AvgEdgeVsRandom":
                (
                    ys["AveragePnL"]
                    -
                    np.nanmedian(
                        random_avg_year
                    )
                ),

            "PFEdgeVsRandom":
                (
                    ys["ProfitFactor"]
                    -
                    np.nanmedian(
                        random_pf_year
                    )
                )
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
    OUTPUT_DIR
    /
    "summary.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "year_by_year.csv",
    index=False
)

master[
    master["BullCross"]
].to_csv(
    OUTPUT_DIR
    /
    "signal_detail.csv"
)


# ============================================================
# DISPLAY
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.max_rows",
    1000
)

pd.set_option(
    "display.width",
    800
)


print()
print("=" * 300)
print("D_NEG_NEG — 30M MACD CROSS VS MATCHED RANDOM")
print("=" * 300)

print(
    summary
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 300)
print("YEAR-BY-YEAR — MACD VS RANDOM")
print("=" * 300)

print(
    year_df
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# PF MATRICES
# ============================================================

print()
print("=" * 240)
print("ACTUAL MACD PF BY YEAR")
print("=" * 240)

actual_pf_matrix = year_df.pivot(
    index="HoldingMinutes",
    columns="Year",
    values="ProfitFactor"
)

print(
    actual_pf_matrix
    .round(4)
    .to_string()
)


print()
print("=" * 240)
print("RANDOM MEDIAN PF BY YEAR")
print("=" * 240)

random_pf_matrix = year_df.pivot(
    index="HoldingMinutes",
    columns="Year",
    values="RandomMedianPF"
)

print(
    random_pf_matrix
    .round(4)
    .to_string()
)


print()
print("=" * 240)
print("MACD PF PERCENTILE BY YEAR")
print("=" * 240)

pct_pf_matrix = year_df.pivot(
    index="HoldingMinutes",
    columns="Year",
    values="MACDPercentile_PF"
)

print(
    pct_pf_matrix
    .round(1)
    .to_string()
)


print()
print("=" * 240)
print("PF EDGE VS RANDOM BY YEAR")
print("=" * 240)

edge_pf_matrix = year_df.pivot(
    index="HoldingMinutes",
    columns="Year",
    values="PFEdgeVsRandom"
)

print(
    edge_pf_matrix
    .round(4)
    .to_string()
)


print()
print("=" * 240)
print("DONE")
print("=" * 240)

print()
print(
    "Signal = bullish 30m MACD(5,36,5) histogram zero-cross"
)

print(
    "Context = 4h MACD <= 0 AND 4h histogram <= 0"
)

print(
    "Session = 06:00 <= signal hour < 22:00"
)

print(
    "Holds = 120, 150, 180 minutes"
)

print(
    "Random entries = same D_NEG_NEG regime, matched by Year + Hour"
)

print(
    "Actual MACD cross bars excluded from random pool"
)

print()
print("Output folder:")
print(OUTPUT_DIR)