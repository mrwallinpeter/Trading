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
    r"\MACD_TwoCandle_CombinedMomentum"
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

ATR_LENGTH = 14

START_HOUR = 10
END_HOUR = 18

FIRST_MIN = 0.35
FIRST_MAX = 0.90

HOLD_MINUTES = 240


# ============================================================
# SECOND-STAGE COMBINED MOVE RANGES
#
# Combined move:
#
# Close[t+1] - Close[t-1]
# -----------------------
# ATR from MACD cross bar
# ============================================================

SECOND_STAGE_TESTS = [

    (0.40, 0.80),
    (0.40, 1.00),
    (0.50, 0.90),
    (0.50, 1.00),
    (0.50, 1.10),
    (0.60, 1.00),
    (0.60, 1.10),
    (0.60, 1.20),
    (0.70, 1.10),
    (0.70, 1.20),
    (0.80, 1.20),
    (0.80, 1.30),
    (0.90, 1.30),
    (1.00, 1.40)
]


# ============================================================
# LOAD 1-MIN DATA
# ============================================================

print()
print("=" * 140)
print("LOADING DATA")
print("=" * 140)

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
# 15-MIN BARS
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
# ATR
# ============================================================

def wilder_atr(df, length):

    prev_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1.0 / length,
        adjust=False
    ).mean()


df15["ATR"] = wilder_atr(
    df15,
    ATR_LENGTH
)


# ============================================================
# TIME FILTER
# ============================================================

df15["TimeValid"] = (
    (df15.index.hour >= START_HOUR)
    &
    (df15.index.hour < END_HOUR)
)


# ============================================================
# STATS
# ============================================================

def profit_factor(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

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

    return gross_profit / gross_loss


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

    equity = np.cumsum(
        pnl
    )

    equity = np.concatenate(
        [
            [0.0],
            equity
        ]
    )

    running_max = np.maximum.accumulate(
        equity
    )

    dd = equity - running_max

    return abs(
        dd.min()
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
            "MaxDrawdownPoints": np.nan
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

        "MaxDrawdownPoints":
            max_drawdown(
                pnl
            )
    }


# ============================================================
# HELPER — BUILD 240-MIN TRADE
# ============================================================

def build_trade(entry_signal_time):

    entry_pos = df1.index.searchsorted(
        entry_signal_time,
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


    exit_target = (
        entry_time
        +
        pd.Timedelta(
            minutes=HOLD_MINUTES
        )
    )

    exit_pos = df1.index.searchsorted(
        exit_target,
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
        exit_target
        +
        pd.Timedelta(minutes=2)
    ):
        return None


    exit_price = float(
        df1.iloc[
            exit_pos
        ]["Close"]
    )

    pnl = (
        exit_price
        -
        entry_price
    )


    return {
        "EntryTime":
            entry_time,

        "ExitTime":
            exit_time,

        "EntryPrice":
            entry_price,

        "ExitPrice":
            exit_price,

        "PnL":
            pnl
    }


# ============================================================
# BUILD SIGNAL FEATURE DATASET
# ============================================================

signals = df15[
    df15["BullCross"]
    &
    df15["TimeValid"]
].copy()


print()
print("=" * 140)
print("BUILDING SIGNAL FEATURES")
print("=" * 140)

print(
    "Raw MACD crosses:",
    len(signals)
)


signal_rows = []


for signal_time in signals.index:

    pos = df15.index.get_loc(
        signal_time
    )

    if pos < 1:
        continue

    if pos + 1 >= len(df15):
        continue


    atr_cross = float(
        df15.iloc[pos]["ATR"]
    )

    if (
        not np.isfinite(atr_cross)
        or
        atr_cross <= 0
    ):
        continue


    close_prev = float(
        df15.iloc[pos - 1]["Close"]
    )

    close_cross = float(
        df15.iloc[pos]["Close"]
    )

    close_second = float(
        df15.iloc[pos + 1]["Close"]
    )


    # --------------------------------------------------------
    # FIRST CANDLE MOVE
    # --------------------------------------------------------

    move1 = (
        close_cross
        -
        close_prev
    ) / atr_cross


    # --------------------------------------------------------
    # SECOND CANDLE INDIVIDUAL MOVE
    # --------------------------------------------------------

    move2 = (
        close_second
        -
        close_cross
    ) / atr_cross


    # --------------------------------------------------------
    # COMBINED TWO-CANDLE MOVE
    #
    # Previous bar close -> second candle close
    # --------------------------------------------------------

    move12 = (
        close_second
        -
        close_prev
    ) / atr_cross


    first_valid = (
        FIRST_MIN
        <=
        move1
        <=
        FIRST_MAX
    )


    signal_rows.append({

        "SignalTime":
            signal_time,

        "Year":
            signal_time.year,

        "ATR":
            atr_cross,

        "Move1":
            move1,

        "Move2":
            move2,

        "Move12":
            move12,

        "FirstValid":
            first_valid,

        "SecondBarTime":
            df15.index[
                pos + 1
            ]
    })


signal_df = pd.DataFrame(
    signal_rows
)


# ============================================================
# BUILD FIRST-CANDLE TRADES
# ============================================================

first_rows = []


for _, row in signal_df[
    signal_df["FirstValid"]
].iterrows():

    trade = build_trade(
        row["SignalTime"]
    )

    if trade is None:
        continue


    first_rows.append({

        "SignalTime":
            row["SignalTime"],

        "Year":
            row["Year"],

        "EntryType":
            "ENTER_FIRST",

        "Move1":
            row["Move1"],

        "Move2":
            row["Move2"],

        "Move12":
            row["Move12"],

        **trade
    })


first_trades = pd.DataFrame(
    first_rows
)


# ============================================================
# CANDIDATES FOR SECOND-STAGE RESCUE
#
# Only cases where candle 1 FAILED.
# ============================================================

second_candidates = signal_df[
    ~signal_df["FirstValid"]
].copy()


print()
print(
    "First-candle trades:",
    len(first_trades)
)

print(
    "Second-stage candidates:",
    len(second_candidates)
)


# ============================================================
# TEST ALL COMBINED MOVE RANGES
# ============================================================

overall_rows = []
year_rows = []
trade_outputs = []


for lower, upper in SECOND_STAGE_TESTS:

    test_name = (
        f"Move12_{lower:.2f}_{upper:.2f}"
    )


    # --------------------------------------------------------
    # RESCUE CONDITIONS
    # --------------------------------------------------------

    rescue_candidates = second_candidates[
        (
            second_candidates["Move12"]
            >=
            lower
        )
        &
        (
            second_candidates["Move12"]
            <=
            upper
        )
    ].copy()


    # --------------------------------------------------------
    # BUILD SECOND-STAGE TRADES
    # --------------------------------------------------------

    second_rows = []


    for _, row in rescue_candidates.iterrows():

        trade = build_trade(
            row["SecondBarTime"]
        )

        if trade is None:
            continue


        second_rows.append({

            "SignalTime":
                row["SignalTime"],

            "Year":
                row["Year"],

            "EntryType":
                "ENTER_SECOND",

            "Test":
                test_name,

            "Move1":
                row["Move1"],

            "Move2":
                row["Move2"],

            "Move12":
                row["Move12"],

            **trade
        })


    second_trades = pd.DataFrame(
        second_rows
    )


    # --------------------------------------------------------
    # COMBINE WITH FIRST-STAGE TRADES
    # --------------------------------------------------------

    first_copy = first_trades.copy()

    first_copy["Test"] = (
        test_name
    )


    combined = pd.concat(
        [
            first_copy,
            second_trades
        ],
        ignore_index=True
    )


    combined = combined.sort_values(
        "EntryTime"
    )


    # --------------------------------------------------------
    # SECOND-STAGE STATS
    # --------------------------------------------------------

    second_stats = calculate_stats(
        second_trades
    )


    # --------------------------------------------------------
    # COMBINED STATS
    # --------------------------------------------------------

    combined_stats = calculate_stats(
        combined
    )


    overall_rows.append({

        "Test":
            test_name,

        "Move12Min":
            lower,

        "Move12Max":
            upper,

        "FirstTrades":
            len(first_trades),

        "SecondTrades":
            len(second_trades),

        "CombinedTrades":
            len(combined),

        "SecondAvgPnL":
            second_stats["AveragePnL"],

        "SecondPF":
            second_stats["ProfitFactor"],

        "SecondSQN":
            second_stats["SQN"],

        "CombinedAvgPnL":
            combined_stats["AveragePnL"],

        "CombinedNet":
            combined_stats["NetPoints"],

        "CombinedPF":
            combined_stats["ProfitFactor"],

        "CombinedSQN":
            combined_stats["SQN"],

        "CombinedMaxDD":
            combined_stats["MaxDrawdownPoints"]
    })


    # --------------------------------------------------------
    # YEAR-BY-YEAR
    # --------------------------------------------------------

    years = sorted(
        signal_df[
            "Year"
        ].unique()
    )


    for year in years:

        second_y = second_trades[
            second_trades["Year"]
            ==
            year
        ]

        combined_y = combined[
            combined["Year"]
            ==
            year
        ]


        second_y_stats = calculate_stats(
            second_y
        )

        combined_y_stats = calculate_stats(
            combined_y
        )


        year_rows.append({

            "Test":
                test_name,

            "Move12Min":
                lower,

            "Move12Max":
                upper,

            "Year":
                year,

            "SecondN":
                second_y_stats["N"],

            "SecondAvgPnL":
                second_y_stats["AveragePnL"],

            "SecondPF":
                second_y_stats["ProfitFactor"],

            "SecondSQN":
                second_y_stats["SQN"],

            "CombinedN":
                combined_y_stats["N"],

            "CombinedAvgPnL":
                combined_y_stats["AveragePnL"],

            "CombinedPF":
                combined_y_stats["ProfitFactor"],

            "CombinedSQN":
                combined_y_stats["SQN"]
        })


    # --------------------------------------------------------
    # SAVE EACH TRADE SET
    # --------------------------------------------------------

    trade_outputs.append(
        combined
    )


overall_df = pd.DataFrame(
    overall_rows
)

year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# STABILITY
# ============================================================

stability_rows = []


for _, row in overall_df.iterrows():

    test_name = row[
        "Test"
    ]

    y = year_df[
        (
            year_df["Test"]
            ==
            test_name
        )
        &
        (
            year_df["Year"]
            >=
            2022
        )
    ]


    second_pf = (
        y["SecondPF"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    combined_pf = (
        y["CombinedPF"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )


    stability_rows.append({

        **row.to_dict(),

        "SecondWorstYearPF":
            second_pf.min()
            if len(second_pf) > 0
            else np.nan,

        "SecondPFAbove1Years":
            int(
                np.sum(
                    y["SecondPF"]
                    >
                    1
                )
            ),

        "SecondProfitableYears":
            int(
                np.sum(
                    y["SecondAvgPnL"]
                    >
                    0
                )
            ),

        "CombinedWorstYearPF":
            combined_pf.min()
            if len(combined_pf) > 0
            else np.nan,

        "CombinedPFAbove1Years":
            int(
                np.sum(
                    y["CombinedPF"]
                    >
                    1
                )
            ),

        "CombinedProfitableYears":
            int(
                np.sum(
                    y["CombinedAvgPnL"]
                    >
                    0
                )
            ),

        "MinSecondYearTrades":
            int(
                y["SecondN"].min()
            )
    })


stability_df = pd.DataFrame(
    stability_rows
)


# ============================================================
# FIRST-ONLY BASELINE
# ============================================================

first_stats = calculate_stats(
    first_trades
)


print()
print("=" * 160)
print("FIRST-CANDLE BASELINE")
print("=" * 160)

print(
    "N:",
    first_stats["N"]
)

print(
    "Avg PnL:",
    round(
        first_stats["AveragePnL"],
        4
    )
)

print(
    "PF:",
    round(
        first_stats["ProfitFactor"],
        4
    )
)

print(
    "SQN:",
    round(
        first_stats["SQN"],
        4
    )
)

print(
    "Max DD:",
    round(
        first_stats["MaxDrawdownPoints"],
        2
    )
)


# ============================================================
# SAVE
# ============================================================

signal_df.to_csv(
    OUTPUT_DIR
    /
    "all_macd_signal_features.csv",
    index=False
)

first_trades.to_csv(
    OUTPUT_DIR
    /
    "first_candle_trades.csv",
    index=False
)

overall_df.to_csv(
    OUTPUT_DIR
    /
    "combined_move_range_overall.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "combined_move_range_year_by_year.csv",
    index=False
)

stability_df.to_csv(
    OUTPUT_DIR
    /
    "combined_move_range_stability.csv",
    index=False
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
    350
)


print()
print("=" * 190)
print(
    "TWO-CANDLE COMBINED MOMENTUM TESTS — SORTED BY SECOND-STAGE WORST YEAR PF"
)
print("=" * 190)

print(
    stability_df
    .sort_values(
        [
            "SecondWorstYearPF",
            "SecondPF"
        ],
        ascending=[
            False,
            False
        ]
    )
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 190)
print(
    "TWO-CANDLE COMBINED MOMENTUM TESTS — SORTED BY COMBINED WORST YEAR PF"
)
print("=" * 190)

print(
    stability_df
    .sort_values(
        [
            "CombinedWorstYearPF",
            "CombinedPF"
        ],
        ascending=[
            False,
            False
        ]
    )
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 190)
print("YEAR-BY-YEAR")
print("=" * 190)

print(
    year_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 190)
print("DONE")
print("=" * 190)

print()
print("Output folder:")
print(OUTPUT_DIR)