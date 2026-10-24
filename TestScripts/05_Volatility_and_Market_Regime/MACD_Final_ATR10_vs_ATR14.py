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
    r"\MACD_Final_ATR10_vs_ATR14"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTHS = [
    10,
    14
]

PRICE_MOVE_MIN = 0.35
PRICE_MOVE_MAX = 0.90

PRIOR_MOVE_MIN = -2.0
PRIOR_MOVE_MAX = 2.0

PRIOR_LOOKBACK = 20

RANGE_LOOKBACK = 20
MAX_RANGE_HEIGHT = 25.0
MAX_RANGE_DRIFT = 10.0

START_HOUR = 14
END_HOUR = 18

HOLD_MINUTES = 210

MAIN_YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026
]


# ============================================================
# LOAD 1-MINUTE DATA
# ============================================================

print()
print("=" * 180)
print("LOADING DATA")
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
# BUILD 15-MIN DATA
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


# ============================================================
# MACD
# ============================================================

df15["FastEMA"] = (
    df15["Close"]
    .ewm(
        span=MACD_FAST,
        adjust=False
    )
    .mean()
)

df15["SlowEMA"] = (
    df15["Close"]
    .ewm(
        span=MACD_SLOW,
        adjust=False
    )
    .mean()
)

df15["MACD"] = (
    df15["FastEMA"]
    -
    df15["SlowEMA"]
)

df15["Signal"] = (
    df15["MACD"]
    .ewm(
        span=MACD_SIGNAL,
        adjust=False
    )
    .mean()
)

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
# TRUE RANGE + ATR10 / ATR14
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


for length in ATR_LENGTHS:

    df15[f"ATR{length}"] = tr.ewm(
        alpha=1.0 / length,
        adjust=False
    ).mean()


# ============================================================
# PRICE MOVE
# ============================================================

df15["PriceMovePoints"] = (
    df15["Close"]
    -
    df15["Close"].shift(1)
)

for length in ATR_LENGTHS:

    df15[f"PriceMoveATR{length}"] = (
        df15["PriceMovePoints"]
        /
        df15[f"ATR{length}"]
    )


# ============================================================
# STATS
# ============================================================

def profit_factor(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

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

    if len(pnl) < 2:
        return np.nan

    std = np.std(
        pnl,
        ddof=1
    )

    if std == 0:
        return np.nan

    return (
        np.sqrt(len(pnl))
        *
        np.mean(pnl)
        /
        std
    )


def max_drawdown(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

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

    drawdown = equity - peak

    return abs(
        drawdown.min()
    )


def calculate_stats(d):

    pnl = (
        d["PnL"]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

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


# ============================================================
# RANGE FILTER
#
# Previous exactly 20 bars.
#
# Valid range:
# height <= 25
# drift  <= 10
#
# Reject only if cross closes inside valid range.
# ============================================================

def get_range_info(pos):

    start_pos = (
        pos
        -
        RANGE_LOOKBACK
    )

    end_pos = (
        pos
        -
        1
    )

    window = df15.iloc[
        start_pos:end_pos + 1
    ]


    range_high = float(
        window["High"].max()
    )

    range_low = float(
        window["Low"].min()
    )

    range_height = (
        range_high
        -
        range_low
    )


    first_close = float(
        df15.iloc[
            start_pos
        ]["Close"]
    )

    last_close = float(
        df15.iloc[
            end_pos
        ]["Close"]
    )


    range_drift = abs(
        last_close
        -
        first_close
    )


    valid_range = (
        range_height
        <=
        MAX_RANGE_HEIGHT
        and
        range_drift
        <=
        MAX_RANGE_DRIFT
    )


    cross_close = float(
        df15.iloc[
            pos
        ]["Close"]
    )


    cross_inside = (
        valid_range
        and
        range_low
        <=
        cross_close
        <=
        range_high
    )


    return {

        "RangeHigh":
            range_high,

        "RangeLow":
            range_low,

        "RangeHeight":
            range_height,

        "RangeDrift":
            range_drift,

        "ValidRange":
            valid_range,

        "CrossInsideRange":
            cross_inside
    }


# ============================================================
# TRADE OUTCOME
# ============================================================

def build_trade(signal_time):

    entry_pos = df1.index.searchsorted(
        signal_time,
        side="right"
    )

    if entry_pos >= len(df1):
        return None


    entry_time = df1.index[
        entry_pos
    ]

    entry_price = float(
        df1.iloc[
            entry_pos
        ]["Open"]
    )


    target_time = (
        entry_time
        +
        pd.Timedelta(
            minutes=HOLD_MINUTES
        )
    )


    exit_pos = df1.index.searchsorted(
        target_time,
        side="left"
    )

    if exit_pos >= len(df1):
        return None


    exit_time = df1.index[
        exit_pos
    ]


    if (
        exit_time
        >
        target_time
        +
        pd.Timedelta(
            minutes=2
        )
    ):
        return None


    exit_price = float(
        df1.iloc[
            exit_pos
        ]["Close"]
    )


    return {

        "EntryTime":
            entry_time,

        "EntryPrice":
            entry_price,

        "ExitTime":
            exit_time,

        "ExitPrice":
            exit_price,

        "PnL":
            exit_price
            -
            entry_price
    }


# ============================================================
# BUILD CROSS DATASET
# ============================================================

print()
print("=" * 180)
print("BUILDING CROSS DATASET")
print("=" * 180)


signals = df15[
    df15["BullCross"]
]

rows = []


for signal_time in signals.index:

    pos = df15.index.get_loc(
        signal_time
    )


    if pos < max(
        RANGE_LOOKBACK,
        PRIOR_LOOKBACK
    ):
        continue


    # --------------------------------------------
    # ONLY 14:00 - 18:00
    # --------------------------------------------

    if not (
        START_HOUR
        <=
        signal_time.hour
        <
        END_HOUR
    ):
        continue


    trade = build_trade(
        signal_time
    )

    if trade is None:
        continue


    row = df15.iloc[
        pos
    ]


    range_info = get_range_info(
        pos
    )


    # Reject sideways-range crosses
    if range_info[
        "CrossInsideRange"
    ]:
        continue


    close_prev = float(
        df15.iloc[
            pos - 1
        ]["Close"]
    )


    prior_start_close = float(
        df15.iloc[
            pos - PRIOR_LOOKBACK
        ]["Close"]
    )


    result = {

        "SignalTime":
            signal_time,

        "Year":
            signal_time.year,

        "Hour":
            signal_time.hour,

        "PriceMovePoints":
            row[
                "PriceMovePoints"
            ],

        **range_info,

        **trade
    }


    # --------------------------------------------
    # Calculate features separately
    # for ATR10 and ATR14
    # --------------------------------------------

    for length in ATR_LENGTHS:

        atr = float(
            row[
                f"ATR{length}"
            ]
        )


        if (
            not np.isfinite(atr)
            or
            atr <= 0
        ):

            result[
                f"PriceMoveATR{length}"
            ] = np.nan

            result[
                f"PriorMoveATR{length}"
            ] = np.nan

            continue


        price_move_atr = (
            row[
                "PriceMovePoints"
            ]
            /
            atr
        )


        prior_move_atr = (
            close_prev
            -
            prior_start_close
        ) / atr


        result[
            f"ATR{length}"
        ] = atr

        result[
            f"PriceMoveATR{length}"
        ] = price_move_atr

        result[
            f"PriorMoveATR{length}"
        ] = prior_move_atr


    rows.append(
        result
    )


df = pd.DataFrame(
    rows
)


print(
    "14-18 completed non-range crosses:",
    len(df)
)


# ============================================================
# PASS FLAGS
# ============================================================

for length in ATR_LENGTHS:

    df[
        f"PassATR{length}"
    ] = (

        (
            df[
                f"PriceMoveATR{length}"
            ]
            >=
            PRICE_MOVE_MIN
        )

        &

        (
            df[
                f"PriceMoveATR{length}"
            ]
            <=
            PRICE_MOVE_MAX
        )

        &

        (
            df[
                f"PriorMoveATR{length}"
            ]
            >=
            PRIOR_MOVE_MIN
        )

        &

        (
            df[
                f"PriorMoveATR{length}"
            ]
            <=
            PRIOR_MOVE_MAX
        )
    )


# ============================================================
# ATR10 / ATR14 DATASETS
# ============================================================

atr10 = df[
    df["PassATR10"]
].copy()

atr14 = df[
    df["PassATR14"]
].copy()


# ============================================================
# OVERLAP GROUPS
# ============================================================

both = df[
    df["PassATR10"]
    &
    df["PassATR14"]
].copy()


atr10_only = df[
    df["PassATR10"]
    &
    ~df["PassATR14"]
].copy()


atr14_only = df[
    ~df["PassATR10"]
    &
    df["PassATR14"]
].copy()


neither = df[
    ~df["PassATR10"]
    &
    ~df["PassATR14"]
].copy()


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


groups = {

    "ATR10":
        atr10,

    "ATR14":
        atr14,

    "BOTH":
        both,

    "ATR10_ONLY":
        atr10_only,

    "ATR14_ONLY":
        atr14_only,

    "NEITHER":
        neither
}


for name, subset in groups.items():

    stats = calculate_stats(
        subset
    )


    year_pfs = []

    profitable_years = 0

    min_year_trades = np.inf


    for year in MAIN_YEARS:

        yd = subset[
            subset["Year"]
            ==
            year
        ]

        ys = calculate_stats(
            yd
        )


        if np.isfinite(
            ys["ProfitFactor"]
        ):

            year_pfs.append(
                ys["ProfitFactor"]
            )

            if (
                ys["ProfitFactor"]
                >
                1
            ):

                profitable_years += 1


        min_year_trades = min(
            min_year_trades,
            ys["N"]
        )


    summary_rows.append({

        "Group":
            name,

        **stats,

        "WorstYearPF":
            (
                min(year_pfs)
                if year_pfs
                else np.nan
            ),

        "ProfitableYears":
            profitable_years,

        "MinYearTrades":
            min_year_trades
    })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# YEAR-BY-YEAR
# ============================================================

year_rows = []


for name, subset in groups.items():

    for year in MAIN_YEARS:

        yd = subset[
            subset["Year"]
            ==
            year
        ]


        stats = calculate_stats(
            yd
        )


        year_rows.append({

            "Group":
                name,

            "Year":
                year,

            **stats
        })


year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# DIRECT ATR10 VS ATR14 YEAR MATRIX
# ============================================================

pf_matrix = (

    year_df[
        year_df[
            "Group"
        ].isin(
            [
                "ATR10",
                "ATR14"
            ]
        )
    ]

    .pivot(
        index="Year",
        columns="Group",
        values="ProfitFactor"
    )
)


avg_matrix = (

    year_df[
        year_df[
            "Group"
        ].isin(
            [
                "ATR10",
                "ATR14"
            ]
        )
    ]

    .pivot(
        index="Year",
        columns="Group",
        values="AveragePnL"
    )
)


n_matrix = (

    year_df[
        year_df[
            "Group"
        ].isin(
            [
                "ATR10",
                "ATR14"
            ]
        )
    ]

    .pivot(
        index="Year",
        columns="Group",
        values="N"
    )
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_DIR
    /
    "all_14_18_nonrange_crosses.csv",
    index=False
)

summary_df.to_csv(
    OUTPUT_DIR
    /
    "atr10_vs_atr14_summary.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "atr10_vs_atr14_year_by_year.csv",
    index=False
)

pf_matrix.to_csv(
    OUTPUT_DIR
    /
    "atr10_vs_atr14_pf_matrix.csv"
)

avg_matrix.to_csv(
    OUTPUT_DIR
    /
    "atr10_vs_atr14_average_pnl_matrix.csv"
)

n_matrix.to_csv(
    OUTPUT_DIR
    /
    "atr10_vs_atr14_trade_count_matrix.csv"
)


# ============================================================
# DISPLAY
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    500
)

pd.set_option(
    "display.max_rows",
    500
)


print()
print("=" * 220)
print("FINAL TEST")
print("14:00-18:00 | HOLD 210 MINUTES")
print("PriceMove 0.35-0.90 | PriorMove -2 to +2 | NO RANGE")
print("=" * 220)


print()
print("SUMMARY")
print("-" * 220)

print(
    summary_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("ATR10 VS ATR14 — PROFIT FACTOR BY YEAR")
print("=" * 220)

print(
    pf_matrix
    .round(4)
    .to_string()
)


print()
print("=" * 220)
print("ATR10 VS ATR14 — AVERAGE PNL BY YEAR")
print("=" * 220)

print(
    avg_matrix
    .round(4)
    .to_string()
)


print()
print("=" * 220)
print("ATR10 VS ATR14 — NUMBER OF TRADES BY YEAR")
print("=" * 220)

print(
    n_matrix
    .to_string()
)


print()
print("=" * 220)
print("OVERLAP")
print("=" * 220)

print(
    f"ATR10 trades:      {len(atr10)}"
)

print(
    f"ATR14 trades:      {len(atr14)}"
)

print(
    f"Both pass:         {len(both)}"
)

print(
    f"ATR10 only:        {len(atr10_only)}"
)

print(
    f"ATR14 only:        {len(atr14_only)}"
)


if len(atr10) > 0:

    print(
        "ATR10 overlap %:  "
        f"{len(both) / len(atr10) * 100:.2f}%"
    )


if len(atr14) > 0:

    print(
        "ATR14 overlap %:  "
        f"{len(both) / len(atr14) * 100:.2f}%"
    )


print()
print("=" * 220)
print("FULL YEAR-BY-YEAR DETAIL")
print("=" * 220)

print(
    year_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("DONE")
print("=" * 220)

print()
print("Output folder:")
print(OUTPUT_DIR)