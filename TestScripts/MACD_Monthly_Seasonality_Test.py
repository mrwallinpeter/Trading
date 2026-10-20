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
    r"\MACD_Monthly_Seasonality_Test"
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

PRIOR_MOVE_MIN = -2.0
PRIOR_MOVE_MAX = 2.0
PRIOR_LOOKBACK = 20

RANGE_LOOKBACK = 20
MAX_RANGE_HEIGHT = 25.0
MAX_RANGE_DRIFT = 10.0

MAIN_YEARS = [2022, 2023, 2024, 2025, 2026]


# ============================================================
# SESSION DEFINITIONS
# ============================================================

SESSIONS = {
    "06_22": {
        "start_hour": 6,
        "end_hour": 22,
        "hold_minutes": 240
    },

    "14_18": {
        "start_hour": 14,
        "end_hour": 18,
        "hold_minutes": 210
    }
}


MONTH_NAMES = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec"
}


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 180)
print("LOADING DATA")
print("=" * 180)

df1 = pd.read_csv(ONE_MIN_FILE)

df1["DateTime"] = pd.to_datetime(df1["DateTime"])

df1 = (
    df1
    .sort_values("DateTime")
    .drop_duplicates("DateTime")
    .set_index("DateTime")
)

for col in ["Open", "High", "Low", "Close", "Volume"]:
    df1[col] = pd.to_numeric(
        df1[col],
        errors="coerce"
    )

df1 = df1.dropna(
    subset=["Open", "High", "Low", "Close"]
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
    subset=["Open", "High", "Low", "Close"]
)


# ============================================================
# MACD
# ============================================================

df15["FastEMA"] = df15["Close"].ewm(
    span=MACD_FAST,
    adjust=False
).mean()

df15["SlowEMA"] = df15["Close"].ewm(
    span=MACD_SLOW,
    adjust=False
).mean()

df15["MACD"] = (
    df15["FastEMA"]
    -
    df15["SlowEMA"]
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
# STAT FUNCTIONS
# ============================================================

def profit_factor(pnl):

    pnl = np.asarray(pnl, dtype=float)

    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()

    if gross_loss == 0:
        if gross_profit > 0:
            return np.inf
        return np.nan

    return gross_profit / gross_loss


def sqn(pnl):

    pnl = np.asarray(pnl, dtype=float)

    if len(pnl) < 2:
        return np.nan

    std = np.std(pnl, ddof=1)

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

    pnl = np.asarray(pnl, dtype=float)

    if len(pnl) == 0:
        return np.nan

    equity = np.concatenate(
        [[0.0], np.cumsum(pnl)]
    )

    peak = np.maximum.accumulate(equity)

    dd = equity - peak

    return abs(dd.min())


def calculate_stats(d):

    pnl = (
        d["PnL"]
        .dropna()
        .to_numpy(dtype=float)
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
# RANGE FILTER
# ============================================================

def get_range_info(pos):

    start_pos = pos - RANGE_LOOKBACK
    end_pos = pos - 1

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
        df15.iloc[start_pos]["Close"]
    )

    last_close = float(
        df15.iloc[end_pos]["Close"]
    )

    range_drift = abs(
        last_close
        -
        first_close
    )

    valid_range = (
        range_height <= MAX_RANGE_HEIGHT
        and
        range_drift <= MAX_RANGE_DRIFT
    )

    cross_close = float(
        df15.iloc[pos]["Close"]
    )

    cross_inside = (
        valid_range
        and
        range_low <= cross_close <= range_high
    )

    return {
        "RangeHigh": range_high,
        "RangeLow": range_low,
        "RangeHeight": range_height,
        "RangeDrift": range_drift,
        "ValidRange": valid_range,
        "CrossInsideRange": cross_inside
    }


# ============================================================
# BUILD FIXED QUALIFYING ENTRIES
# ============================================================

print()
print("=" * 180)
print("BUILDING FIXED ENTRY DATASET")
print("=" * 180)

signals = df15[
    df15["BullCross"]
]

entry_rows = []


for signal_time in signals.index:

    pos = df15.index.get_loc(signal_time)

    if pos < max(
        RANGE_LOOKBACK,
        PRIOR_LOOKBACK
    ):
        continue


    atr14 = float(
        df15.iloc[pos]["ATR14"]
    )

    if (
        not np.isfinite(atr14)
        or
        atr14 <= 0
    ):
        continue


    close_now = float(
        df15.iloc[pos]["Close"]
    )

    close_prev = float(
        df15.iloc[pos - 1]["Close"]
    )


    price_move_atr = (
        close_now
        -
        close_prev
    ) / atr14


    prior_start_close = float(
        df15.iloc[
            pos - PRIOR_LOOKBACK
        ]["Close"]
    )

    prior_move_atr = (
        close_prev
        -
        prior_start_close
    ) / atr14


    range_info = get_range_info(pos)


    # --------------------------------------------
    # FINAL ENTRY FILTERS
    # --------------------------------------------

    if not (
        PRICE_MOVE_MIN
        <= price_move_atr
        <= PRICE_MOVE_MAX
    ):
        continue


    if range_info["CrossInsideRange"]:
        continue


    if not (
        PRIOR_MOVE_MIN
        <= prior_move_atr
        <= PRIOR_MOVE_MAX
    ):
        continue


    entry_pos = df1.index.searchsorted(
        signal_time,
        side="right"
    )

    if entry_pos >= len(df1):
        continue


    entry_time = df1.index[
        entry_pos
    ]

    entry_price = float(
        df1.iloc[
            entry_pos
        ]["Open"]
    )


    entry_rows.append({

        "SignalTime":
            signal_time,

        "Year":
            signal_time.year,

        "Month":
            signal_time.month,

        "MonthName":
            MONTH_NAMES[
                signal_time.month
            ],

        "Hour":
            signal_time.hour,

        "EntryTime":
            entry_time,

        "EntryPrice":
            entry_price,

        "ATR14":
            atr14,

        "PriceMoveATR14":
            price_move_atr,

        "PriorMoveATR14":
            prior_move_atr,

        **range_info
    })


entries = pd.DataFrame(
    entry_rows
)

print(
    "Fixed qualifying entries:",
    len(entries)
)


# ============================================================
# BUILD SESSION-SPECIFIC TRADE OUTCOMES
# ============================================================

trade_rows = []


for session_name, settings in SESSIONS.items():

    start_hour = settings["start_hour"]
    end_hour = settings["end_hour"]
    hold_minutes = settings["hold_minutes"]


    session_entries = entries[
        (
            entries["Hour"] >= start_hour
        )
        &
        (
            entries["Hour"] < end_hour
        )
    ].copy()


    print(
        f"{session_name}: "
        f"{len(session_entries)} qualifying entries"
    )


    for _, row in session_entries.iterrows():

        target_time = (
            row["EntryTime"]
            +
            pd.Timedelta(
                minutes=hold_minutes
            )
        )


        exit_pos = df1.index.searchsorted(
            target_time,
            side="left"
        )

        if exit_pos >= len(df1):
            continue


        exit_time = df1.index[
            exit_pos
        ]


        if (
            exit_time
            >
            target_time
            +
            pd.Timedelta(minutes=2)
        ):
            continue


        exit_price = float(
            df1.iloc[
                exit_pos
            ]["Close"]
        )


        pnl = (
            exit_price
            -
            row["EntryPrice"]
        )


        trade_rows.append({

            **row.to_dict(),

            "Session":
                session_name,

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
# MONTH SUMMARY
# ============================================================

month_rows = []


for session_name in SESSIONS:

    session_trades = trades[
        trades["Session"]
        ==
        session_name
    ]


    for month in range(1, 13):

        md = session_trades[
            session_trades["Month"]
            ==
            month
        ]

        stats = calculate_stats(md)


        year_pfs = []
        profitable_years = 0
        losing_years = 0
        min_year_trades = np.inf


        for year in MAIN_YEARS:

            yd = md[
                md["Year"]
                ==
                year
            ]

            ys = calculate_stats(yd)

            min_year_trades = min(
                min_year_trades,
                ys["N"]
            )


            if np.isfinite(
                ys["ProfitFactor"]
            ):

                year_pfs.append(
                    ys["ProfitFactor"]
                )

                if ys["ProfitFactor"] > 1:
                    profitable_years += 1

                elif ys["ProfitFactor"] < 1:
                    losing_years += 1


        month_rows.append({

            "Session":
                session_name,

            "Month":
                month,

            "MonthName":
                MONTH_NAMES[month],

            **stats,

            "WorstYearPF":
                (
                    min(year_pfs)
                    if year_pfs
                    else np.nan
                ),

            "BestYearPF":
                (
                    max(year_pfs)
                    if year_pfs
                    else np.nan
                ),

            "ProfitableYears":
                profitable_years,

            "LosingYears":
                losing_years,

            "MinYearTrades":
                min_year_trades
        })


month_df = pd.DataFrame(
    month_rows
)


# ============================================================
# MONTH × YEAR DETAILS
# ============================================================

month_year_rows = []


for session_name in SESSIONS:

    session_trades = trades[
        trades["Session"]
        ==
        session_name
    ]


    for month in range(1, 13):

        for year in MAIN_YEARS:

            subset = session_trades[
                (
                    session_trades["Month"]
                    ==
                    month
                )
                &
                (
                    session_trades["Year"]
                    ==
                    year
                )
            ]

            stats = calculate_stats(
                subset
            )


            month_year_rows.append({

                "Session":
                    session_name,

                "Month":
                    month,

                "MonthName":
                    MONTH_NAMES[month],

                "Year":
                    year,

                **stats
            })


month_year_df = pd.DataFrame(
    month_year_rows
)


# ============================================================
# PF MATRICES
# ============================================================

pf_matrices = {}


for session_name in SESSIONS:

    temp = month_year_df[
        month_year_df["Session"]
        ==
        session_name
    ]

    matrix = temp.pivot(
        index="MonthName",
        columns="Year",
        values="ProfitFactor"
    )

    matrix = matrix.reindex(
        [
            "Jan", "Feb", "Mar", "Apr",
            "May", "Jun", "Jul", "Aug",
            "Sep", "Oct", "Nov", "Dec"
        ]
    )

    pf_matrices[
        session_name
    ] = matrix


# ============================================================
# MONTH RANKING
# ============================================================

ranking_df = month_df.copy()

ranking_df["All5Years"] = (
    ranking_df["ProfitableYears"]
    ==
    5
)

ranking_df = ranking_df.sort_values(
    [
        "Session",
        "All5Years",
        "ProfitableYears",
        "WorstYearPF",
        "ProfitFactor",
        "N"
    ],
    ascending=[
        True,
        False,
        False,
        False,
        False,
        False
    ]
)


# ============================================================
# SAVE
# ============================================================

entries.to_csv(
    OUTPUT_DIR
    /
    "fixed_entries.csv",
    index=False
)

trades.to_csv(
    OUTPUT_DIR
    /
    "monthly_session_trades.csv",
    index=False
)

month_df.to_csv(
    OUTPUT_DIR
    /
    "monthly_summary.csv",
    index=False
)

month_year_df.to_csv(
    OUTPUT_DIR
    /
    "monthly_year_by_year.csv",
    index=False
)

ranking_df.to_csv(
    OUTPUT_DIR
    /
    "monthly_ranking.csv",
    index=False
)


for session_name, matrix in pf_matrices.items():

    matrix.to_csv(
        OUTPUT_DIR
        /
        f"{session_name}_month_year_pf_matrix.csv"
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
    1000
)


print()
print("=" * 220)
print("MONTHLY SUMMARY")
print("=" * 220)

print(
    month_df
    .round(4)
    .to_string(
        index=False
    )
)


for session_name, matrix in pf_matrices.items():

    print()
    print("=" * 220)
    print(
        f"{session_name} — MONTH × YEAR PROFIT FACTOR"
    )
    print("=" * 220)

    print(
        matrix
        .round(4)
        .to_string()
    )


print()
print("=" * 220)
print("MONTH RANKING")
print("=" * 220)

print(
    ranking_df[
        [
            "Session",
            "MonthName",
            "N",
            "AveragePnL",
            "ProfitFactor",
            "SQN",
            "MaxDrawdown",
            "WorstYearPF",
            "ProfitableYears",
            "LosingYears",
            "MinYearTrades"
        ]
    ]
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