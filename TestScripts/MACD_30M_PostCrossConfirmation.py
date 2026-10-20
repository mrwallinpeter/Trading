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
    r"\MACD_30M_PostCrossConfirmation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TIMEFRAME = "30min"

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

HOLD_MINUTES_LIST = [
    60,
    90,
    120,
    150,
    180,
    240,
    300
]

YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026
]


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
# BUILD 30-MIN BARS
# ============================================================

print()
print("=" * 180)
print("BUILDING 30-MIN BARS")
print("=" * 180)

df30 = df1.resample(
    TIMEFRAME,
    label="right",
    closed="right"
).agg({
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum"
})

df30 = df30.dropna(
    subset=[
        "Open",
        "High",
        "Low",
        "Close"
    ]
)

print("30m bars:", len(df30))


# ============================================================
# MACD
# ============================================================

df30["FastEMA"] = df30["Close"].ewm(
    span=MACD_FAST,
    adjust=False
).mean()

df30["SlowEMA"] = df30["Close"].ewm(
    span=MACD_SLOW,
    adjust=False
).mean()

df30["MACD"] = (
    df30["FastEMA"]
    -
    df30["SlowEMA"]
)

df30["SignalLine"] = df30["MACD"].ewm(
    span=MACD_SIGNAL,
    adjust=False
).mean()

df30["Hist"] = (
    df30["MACD"]
    -
    df30["SignalLine"]
)

df30["BullCross"] = (
    (df30["Hist"].shift(1) <= 0)
    &
    (df30["Hist"] > 0)
)


# ============================================================
# STAT FUNCTIONS
# ============================================================

def profit_factor(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

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

    if len(pnl) < 2:
        return np.nan

    std = pnl.std(
        ddof=1
    )

    if std == 0:
        return np.nan

    return (
        np.sqrt(len(pnl))
        *
        pnl.mean()
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

    return abs(
        (
            equity
            -
            peak
        ).min()
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
# ENTRY RULES
# ============================================================

ENTRY_RULES = [
    "CROSS_NOW",
    "CONFIRM_1",
    "CONFIRM_2"
]


# ============================================================
# BUILD TRADES
# ============================================================

print()
print("=" * 180)
print("BUILDING POST-CROSS CONFIRMATION TRADES")
print("=" * 180)

trade_rows = []

cross_positions = np.where(
    df30["BullCross"]
    .to_numpy(dtype=bool)
)[0]


for cross_pos in cross_positions:

    cross_time = df30.index[
        cross_pos
    ]

    if cross_time.year not in YEARS:
        continue


    hist_cross = float(
        df30.iloc[
            cross_pos
        ]["Hist"]
    )


    # ========================================================
    # DEFINE ENTRY SIGNAL BAR FOR EACH RULE
    # ========================================================

    entry_signal_positions = {}


    # Immediate entry
    entry_signal_positions[
        "CROSS_NOW"
    ] = cross_pos


    # One confirming histogram expansion bar
    if (
        cross_pos + 1
        <
        len(df30)
    ):

        hist1 = float(
            df30.iloc[
                cross_pos + 1
            ]["Hist"]
        )

        if (
            np.isfinite(hist1)
            and
            hist1 > hist_cross
        ):

            entry_signal_positions[
                "CONFIRM_1"
            ] = (
                cross_pos + 1
            )


    # Two consecutive confirming expansion bars
    if (
        cross_pos + 2
        <
        len(df30)
    ):

        hist1 = float(
            df30.iloc[
                cross_pos + 1
            ]["Hist"]
        )

        hist2 = float(
            df30.iloc[
                cross_pos + 2
            ]["Hist"]
        )

        if (
            np.isfinite(hist1)
            and
            np.isfinite(hist2)
            and
            hist1 > hist_cross
            and
            hist2 > hist1
        ):

            entry_signal_positions[
                "CONFIRM_2"
            ] = (
                cross_pos + 2
            )


    # ========================================================
    # EXECUTE EACH VALID ENTRY RULE
    # ========================================================

    for rule, signal_pos in entry_signal_positions.items():

        entry_signal_time = df30.index[
            signal_pos
        ]


        # First 1m open after completed signal/confirmation bar
        entry_pos_1m = df1.index.searchsorted(
            entry_signal_time,
            side="right"
        )

        if entry_pos_1m >= len(df1):
            continue


        entry_time = df1.index[
            entry_pos_1m
        ]

        entry_price = float(
            df1.iloc[
                entry_pos_1m
            ]["Open"]
        )


        hist_entry_signal = float(
            df30.iloc[
                signal_pos
            ]["Hist"]
        )


        for hold_minutes in HOLD_MINUTES_LIST:

            target_exit = (
                entry_time
                +
                pd.Timedelta(
                    minutes=hold_minutes
                )
            )


            exit_pos_1m = df1.index.searchsorted(
                target_exit,
                side="left"
            )

            if exit_pos_1m >= len(df1):
                continue


            exit_time = df1.index[
                exit_pos_1m
            ]


            if (
                exit_time
                >
                target_exit
                +
                pd.Timedelta(
                    minutes=2
                )
            ):
                continue


            exit_price = float(
                df1.iloc[
                    exit_pos_1m
                ]["Close"]
            )


            pnl = (
                exit_price
                -
                entry_price
            )


            trade_rows.append({

                "Rule":
                    rule,

                "CrossTime":
                    cross_time,

                "Year":
                    cross_time.year,

                "Hour":
                    cross_time.hour,

                "CrossHist":
                    hist_cross,

                "EntrySignalTime":
                    entry_signal_time,

                "EntrySignalHist":
                    hist_entry_signal,

                "BarsWaited":
                    (
                        signal_pos
                        -
                        cross_pos
                    ),

                "EntryTime":
                    entry_time,

                "EntryPrice":
                    entry_price,

                "HoldingMinutes":
                    hold_minutes,

                "ExitTime":
                    exit_time,

                "ExitPrice":
                    exit_price,

                "PnL":
                    pnl
            })


trades = pd.DataFrame(
    trade_rows
)


# ============================================================
# SIGNAL / RETENTION COUNTS
# ============================================================

raw_cross_count = sum(
    (
        df30.index[
            cross_positions
        ].year
        >=
        min(YEARS)
    )
    &
    (
        df30.index[
            cross_positions
        ].year
        <=
        max(YEARS)
    )
)


count_rows = []


for rule in ENTRY_RULES:

    d = trades[
        (
            trades["Rule"]
            ==
            rule
        )
        &
        (
            trades[
                "HoldingMinutes"
            ]
            ==
            HOLD_MINUTES_LIST[0]
        )
    ]

    count_rows.append({

        "Rule":
            rule,

        "Trades":
            len(d),

        "RawCrosses":
            raw_cross_count,

        "RetentionPct":
            (
                len(d)
                /
                raw_cross_count
                *
                100
            )
            if raw_cross_count
            else np.nan
    })


count_df = pd.DataFrame(
    count_rows
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []
year_rows = []


for rule in ENTRY_RULES:

    for hold_minutes in HOLD_MINUTES_LIST:

        d = trades[
            (
                trades["Rule"]
                ==
                rule
            )
            &
            (
                trades[
                    "HoldingMinutes"
                ]
                ==
                hold_minutes
            )
        ].copy()


        s = calculate_stats(
            d
        )


        year_pfs = []


        for year in YEARS:

            yd = d[
                d["Year"]
                ==
                year
            ]

            ys = calculate_stats(
                yd
            )


            year_rows.append({

                "Rule":
                    rule,

                "HoldingMinutes":
                    hold_minutes,

                "Year":
                    year,

                **ys
            })


            if np.isfinite(
                ys[
                    "ProfitFactor"
                ]
            ):

                year_pfs.append(
                    ys[
                        "ProfitFactor"
                    ]
                )


        summary_rows.append({

            "Rule":
                rule,

            "HoldingMinutes":
                hold_minutes,

            **s,

            "WorstYearPF":
                (
                    min(
                        year_pfs
                    )
                    if year_pfs
                    else np.nan
                ),

            "BestYearPF":
                (
                    max(
                        year_pfs
                    )
                    if year_pfs
                    else np.nan
                ),

            "ProfitableYears":
                sum(
                    pf > 1
                    for pf in year_pfs
                )
        })


summary_df = pd.DataFrame(
    summary_rows
)

year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# YEAR PF MATRIX
# ============================================================

year_pf_matrix = year_df.pivot_table(
    index=[
        "Rule",
        "HoldingMinutes"
    ],
    columns="Year",
    values="ProfitFactor"
)


# ============================================================
# SAVE
# ============================================================

trades.to_csv(
    OUTPUT_DIR
    /
    "trade_detail.csv",
    index=False
)

count_df.to_csv(
    OUTPUT_DIR
    /
    "signal_retention.csv",
    index=False
)

summary_df.to_csv(
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

year_pf_matrix.to_csv(
    OUTPUT_DIR
    /
    "year_pf_matrix.csv"
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
    2000
)

pd.set_option(
    "display.width",
    700
)


print()
print("=" * 220)
print("SIGNAL RETENTION")
print("=" * 220)

print(
    count_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 260)
print("POST-CROSS CONFIRMATION SUMMARY")
print("=" * 260)

print(
    summary_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 260)
print("YEAR-BY-YEAR PROFIT FACTOR")
print("=" * 260)

print(
    year_pf_matrix
    .round(4)
    .to_string()
)


print()
print("=" * 260)
print("KEY COMPARISON")
print("=" * 260)

key_cols = [
    "Rule",
    "HoldingMinutes",
    "N",
    "WinRatePct",
    "AveragePnL",
    "MedianPnL",
    "ProfitFactor",
    "SQN",
    "MaxDrawdown",
    "WorstYearPF",
    "ProfitableYears"
]

print(
    summary_df[
        key_cols
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 260)
print("DONE")
print("=" * 260)

print()
print(
    "CROSS_NOW: enter immediately after bullish MACD cross."
)

print(
    "CONFIRM_1: require next 30m histogram bar > cross histogram."
)

print(
    "CONFIRM_2: require two consecutive post-cross histogram increases."
)

print()
print("Output folder:")
print(OUTPUT_DIR)