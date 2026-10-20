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
    r"\MACD_30M_TwoFilters_5H_Test"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TIMEFRAME = "30min"

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14

PRICE_MOVE_MIN = 0.35
PRICE_MOVE_MAX = 0.90

PRIOR_LOOKBACK = 20
PRIOR_MOVE_MIN = -2.0
PRIOR_MOVE_MAX = 2.0

# Fixed 5-hour holding period
HOLD_MINUTES = 300

YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026
]

SESSIONS = {
    "ALL_TIMES": (None, None),
    "06_22": (6, 22),
    "14_18": (14, 18)
}


# ============================================================
# LOAD 1-MIN DATA
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

print("1m rows:", len(df1))
print("Start:", df1.index.min())
print("End:  ", df1.index.max())


# ============================================================
# BUILD 30-MIN BARS
# ============================================================

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

print()
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

df30["Signal"] = df30["MACD"].ewm(
    span=MACD_SIGNAL,
    adjust=False
).mean()

df30["Hist"] = (
    df30["MACD"]
    -
    df30["Signal"]
)

df30["BullCross"] = (
    (df30["Hist"].shift(1) <= 0)
    &
    (df30["Hist"] > 0)
)


# ============================================================
# WILDER ATR14
# ============================================================

prev_close = df30["Close"].shift(1)

tr = pd.concat(
    [
        df30["High"] - df30["Low"],
        (df30["High"] - prev_close).abs(),
        (df30["Low"] - prev_close).abs()
    ],
    axis=1
).max(axis=1)

df30["ATR14"] = tr.ewm(
    alpha=1.0 / ATR_LENGTH,
    adjust=False
).mean()


# ============================================================
# FILTER VALUES
# ============================================================

df30["PriceMoveATR14"] = (
    df30["Close"]
    -
    df30["Close"].shift(1)
) / df30["ATR14"]


df30["PriorMoveATR14"] = (
    df30["Close"].shift(1)
    -
    df30["Close"].shift(PRIOR_LOOKBACK)
) / df30["ATR14"]


# ============================================================
# STAT FUNCTIONS
# ============================================================

def profit_factor(pnl):

    pnl = np.asarray(
        pnl,
        dtype=float
    )

    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()

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

        "N":
            len(pnl),

        "WinRatePct":
            np.mean(pnl > 0) * 100,

        "NetPoints":
            pnl.sum(),

        "AveragePnL":
            pnl.mean(),

        "MedianPnL":
            np.median(pnl),

        "ProfitFactor":
            profit_factor(pnl),

        "SQN":
            sqn(pnl),

        "MaxDrawdown":
            max_drawdown(pnl)
    }


# ============================================================
# BUILD SIGNAL TABLE
# ============================================================

print()
print("=" * 180)
print("BUILDING 30-MIN MACD SIGNALS")
print("=" * 180)


signal_rows = []


for signal_time in df30.index[
    df30["BullCross"]
]:

    pos = df30.index.get_loc(
        signal_time
    )

    if pos < PRIOR_LOOKBACK:
        continue


    atr14 = float(
        df30.iloc[pos]["ATR14"]
    )

    price_move = float(
        df30.iloc[pos]["PriceMoveATR14"]
    )

    prior_move = float(
        df30.iloc[pos]["PriorMoveATR14"]
    )


    if (
        not np.isfinite(atr14)
        or
        not np.isfinite(price_move)
        or
        not np.isfinite(prior_move)
        or
        atr14 <= 0
    ):
        continue


    signal_rows.append({

        "SignalTime":
            signal_time,

        "Year":
            signal_time.year,

        "Month":
            signal_time.month,

        "Hour":
            signal_time.hour,

        "ATR14":
            atr14,

        "PriceMoveATR14":
            price_move,

        "PriorMoveATR14":
            prior_move,

        "PassPriceMove":
            (
                PRICE_MOVE_MIN
                <= price_move
                <= PRICE_MOVE_MAX
            ),

        "PassPriorMove":
            (
                PRIOR_MOVE_MIN
                <= prior_move
                <= PRIOR_MOVE_MAX
            )
    })


signals = pd.DataFrame(
    signal_rows
)


print("Raw bullish 30m crosses:", len(signals))

print(
    "Pass PriceMove:",
    int(
        signals["PassPriceMove"].sum()
    )
)

print(
    "Pass PriceMove + PriorMove:",
    int(
        (
            signals["PassPriceMove"]
            &
            signals["PassPriorMove"]
        ).sum()
    )
)


# ============================================================
# FILTER STAGES
# ============================================================

FILTER_STAGES = {

    "RAW_MACD":
        lambda d:
            pd.Series(
                True,
                index=d.index
            ),

    "PRICE_MOVE":
        lambda d:
            d["PassPriceMove"],

    "PRICE_PLUS_PRIOR":
        lambda d:
            (
                d["PassPriceMove"]
                &
                d["PassPriorMove"]
            )
}


# ============================================================
# BUILD 5-HOUR TRADES
# ============================================================

trade_rows = []


for _, signal in signals.iterrows():

    signal_time = signal[
        "SignalTime"
    ]


    # Entry = first 1-minute open after completed 30m bar
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


    # Fixed 5-hour target exit
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
        continue


    exit_time = df1.index[
        exit_pos
    ]


    # Same execution tolerance as earlier tests
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
        entry_price
    )


    trade_rows.append({

        **signal.to_dict(),

        "EntryTime":
            entry_time,

        "EntryPrice":
            entry_price,

        "ExitTime":
            exit_time,

        "ExitPrice":
            exit_price,

        "HoldingMinutes":
            HOLD_MINUTES,

        "PnL":
            pnl
    })


trades = pd.DataFrame(
    trade_rows
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []
year_rows = []


for stage_name, stage_func in FILTER_STAGES.items():

    stage_mask = stage_func(
        trades
    )


    for session_name, (
        start_hour,
        end_hour
    ) in SESSIONS.items():


        if start_hour is None:

            session_mask = pd.Series(
                True,
                index=trades.index
            )

        else:

            session_mask = (
                (
                    trades["Hour"]
                    >=
                    start_hour
                )
                &
                (
                    trades["Hour"]
                    <
                    end_hour
                )
            )


        d = trades[
            stage_mask
            &
            session_mask
        ].copy()


        s = calculate_stats(
            d
        )


        year_pfs = []
        profitable_years = 0
        losing_years = 0
        min_year_trades = np.inf


        for year in YEARS:

            yd = d[
                d["Year"]
                ==
                year
            ]

            ys = calculate_stats(
                yd
            )

            min_year_trades = min(
                min_year_trades,
                ys["N"]
            )

            pf = ys[
                "ProfitFactor"
            ]

            if np.isfinite(pf):

                year_pfs.append(
                    pf
                )

                if pf > 1:
                    profitable_years += 1

                elif pf < 1:
                    losing_years += 1


            year_rows.append({

                "Stage":
                    stage_name,

                "Session":
                    session_name,

                "Year":
                    year,

                **ys
            })


        summary_rows.append({

            "Stage":
                stage_name,

            "Session":
                session_name,

            "HoldingMinutes":
                HOLD_MINUTES,

            **s,

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
                (
                    min_year_trades
                    if min_year_trades != np.inf
                    else 0
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
        "Stage",
        "Session"
    ],
    columns="Year",
    values="ProfitFactor"
)


# ============================================================
# SAVE
# ============================================================

signals.to_csv(
    OUTPUT_DIR
    /
    "30m_signals.csv",
    index=False
)

trades.to_csv(
    OUTPUT_DIR
    /
    "30m_5h_trades.csv",
    index=False
)

summary_df.to_csv(
    OUTPUT_DIR
    /
    "30m_5h_summary.csv",
    index=False
)

year_df.to_csv(
    OUTPUT_DIR
    /
    "30m_5h_year_by_year.csv",
    index=False
)

year_pf_matrix.to_csv(
    OUTPUT_DIR
    /
    "30m_5h_year_pf_matrix.csv"
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
    600
)


print()
print("=" * 220)
print("30M MACD — FIXED 5-HOUR HOLD")
print("=" * 220)

print(
    summary_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 220)
print("YEAR-BY-YEAR PROFIT FACTOR")
print("=" * 220)

print(
    year_pf_matrix
    .round(4)
    .to_string()
)


print()
print("=" * 220)
print("PRICE MOVE + PRIOR MOVE — YEAR BY YEAR")
print("=" * 220)

print(
    year_df[
        year_df["Stage"]
        ==
        "PRICE_PLUS_PRIOR"
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
print("Holding period:")
print(HOLD_MINUTES, "minutes = 5 hours")

print()
print("Output folder:")
print(OUTPUT_DIR)