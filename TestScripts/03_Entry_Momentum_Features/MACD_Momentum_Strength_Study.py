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
    r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\TestScripts\MACD_Momentum_Strength"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14

START_HOUR = 10
END_HOUR = 18

HOLD_MINUTES = 240

MAX_PRE_BARS = 200

STABILITY_YEARS = [
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
print("=" * 120)
print("LOADING 1-MIN DATA")
print("=" * 120)

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
# HELPER: PRICE EFFICIENCY
#
# efficiency =
#
# absolute net move
# -----------------------------
# sum of absolute bar-to-bar moves
#
# near 1 = directional
# near 0 = noisy
# ============================================================

def price_efficiency(close_series, pos, bars):

    if pos < bars:
        return np.nan

    start = pos - bars

    net_move = abs(
        float(close_series.iloc[pos])
        -
        float(close_series.iloc[start])
    )

    path = 0.0

    for i in range(start + 1, pos + 1):

        path += abs(
            float(close_series.iloc[i])
            -
            float(close_series.iloc[i - 1])
        )

    if path == 0:
        return 0.0

    return net_move / path


# ============================================================
# BUILD TRADE DATASET
# ============================================================

print()
print("=" * 120)
print("BUILDING MACD MOMENTUM FEATURE DATASET")
print("=" * 120)


signals = df15[
    df15["BullCross"]
    &
    df15["TimeValid"]
].copy()


rows = []


for signal_time in signals.index:

    pos = df15.index.get_loc(
        signal_time
    )


    if pos < 10:
        continue


    # --------------------------------------------------------
    # ENTRY
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

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
        continue


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


    # ========================================================
    # CURRENT BAR VALUES
    # ========================================================

    hist0 = float(
        df15.iloc[pos]["Hist"]
    )

    hist1 = float(
        df15.iloc[pos - 1]["Hist"]
    )

    hist2 = float(
        df15.iloc[pos - 2]["Hist"]
    )

    hist3 = float(
        df15.iloc[pos - 3]["Hist"]
    )

    hist4 = float(
        df15.iloc[pos - 4]["Hist"]
    )


    atr = float(
        df15.iloc[pos]["ATR"]
    )


    if not np.isfinite(atr) or atr <= 0:
        continue


    # ========================================================
    # HISTOGRAM MOMENTUM FEATURES
    # ========================================================

    hist_delta1 = (
        hist0
        -
        hist1
    )

    hist_slope2 = (
        hist0
        -
        hist2
    ) / 2.0

    hist_slope3 = (
        hist0
        -
        hist3
    ) / 3.0

    hist_slope4 = (
        hist0
        -
        hist4
    ) / 4.0


    # ATR normalized versions
    hist_cross_norm = (
        hist0
        /
        atr
    )

    hist_delta1_norm = (
        hist_delta1
        /
        atr
    )

    hist_slope2_norm = (
        hist_slope2
        /
        atr
    )

    hist_slope3_norm = (
        hist_slope3
        /
        atr
    )

    hist_slope4_norm = (
        hist_slope4
        /
        atr
    )


    # ========================================================
    # PRE-CROSS NEGATIVE HISTOGRAM AREA
    # ========================================================

    pre_area = 0.0
    pre_bars = 0

    j = pos - 1


    while (
        j >= 0
        and
        pre_bars < MAX_PRE_BARS
    ):

        h = float(
            df15.iloc[j]["Hist"]
        )


        if not np.isfinite(h):
            break


        if h >= 0:
            break


        pre_area += abs(h)

        pre_bars += 1

        j -= 1


    pre_area_norm = (
        pre_area
        /
        atr
    )


    pre_area_per_bar = (
        pre_area
        /
        pre_bars
        if pre_bars > 0
        else np.nan
    )


    pre_area_per_bar_norm = (
        pre_area_per_bar
        /
        atr
        if pre_bars > 0
        else np.nan
    )


    # ========================================================
    # PRICE MOMENTUM
    # ========================================================

    close0 = float(
        df15.iloc[pos]["Close"]
    )

    close1 = float(
        df15.iloc[pos - 1]["Close"]
    )

    close2 = float(
        df15.iloc[pos - 2]["Close"]
    )

    close4 = float(
        df15.iloc[pos - 4]["Close"]
    )


    price_move_15 = (
        close0
        -
        close1
    )

    price_move_30 = (
        close0
        -
        close2
    )

    price_move_60 = (
        close0
        -
        close4
    )


    price_move_15_atr = (
        price_move_15
        /
        atr
    )

    price_move_30_atr = (
        price_move_30
        /
        atr
    )

    price_move_60_atr = (
        price_move_60
        /
        atr
    )


    # ========================================================
    # PRICE EFFICIENCY
    # ========================================================

    efficiency4 = price_efficiency(
        df15["Close"],
        pos,
        4
    )

    efficiency8 = price_efficiency(
        df15["Close"],
        pos,
        8
    )


    # ========================================================
    # CROSS CANDLE STRUCTURE
    # ========================================================

    open0 = float(
        df15.iloc[pos]["Open"]
    )

    high0 = float(
        df15.iloc[pos]["High"]
    )

    low0 = float(
        df15.iloc[pos]["Low"]
    )


    body = (
        close0
        -
        open0
    )

    candle_range = (
        high0
        -
        low0
    )


    body_atr = (
        body
        /
        atr
    )

    range_atr = (
        candle_range
        /
        atr
    )


    if candle_range > 0:

        close_location = (
            close0
            -
            low0
        ) / candle_range

    else:

        close_location = 0.5


    # ========================================================
    # STORE
    # ========================================================

    rows.append({

        "SignalTime":
            signal_time,

        "EntryTime":
            entry_time,

        "ExitTime":
            exit_time,

        "Year":
            signal_time.year,

        "EntryPrice":
            entry_price,

        "ExitPrice":
            exit_price,

        "PnL":
            pnl,

        "ATR":
            atr,

        # Histogram
        "HistCross":
            hist0,

        "HistCrossNorm":
            hist_cross_norm,

        "HistDelta1":
            hist_delta1,

        "HistDelta1Norm":
            hist_delta1_norm,

        "HistSlope2":
            hist_slope2,

        "HistSlope2Norm":
            hist_slope2_norm,

        "HistSlope3":
            hist_slope3,

        "HistSlope3Norm":
            hist_slope3_norm,

        "HistSlope4":
            hist_slope4,

        "HistSlope4Norm":
            hist_slope4_norm,

        # Pre area
        "PreArea":
            pre_area,

        "PreAreaNorm":
            pre_area_norm,

        "PreBars":
            pre_bars,

        "PreAreaPerBar":
            pre_area_per_bar,

        "PreAreaPerBarNorm":
            pre_area_per_bar_norm,

        # Price momentum
        "PriceMove15":
            price_move_15,

        "PriceMove15ATR":
            price_move_15_atr,

        "PriceMove30":
            price_move_30,

        "PriceMove30ATR":
            price_move_30_atr,

        "PriceMove60":
            price_move_60,

        "PriceMove60ATR":
            price_move_60_atr,

        # Efficiency
        "Efficiency4":
            efficiency4,

        "Efficiency8":
            efficiency8,

        # Candle
        "CrossBody":
            body,

        "CrossBodyATR":
            body_atr,

        "CrossRange":
            candle_range,

        "CrossRangeATR":
            range_atr,

        "CloseLocation":
            close_location
    })


trades = pd.DataFrame(
    rows
)


print("Completed trades:", len(trades))


# ============================================================
# PERFORMANCE FUNCTIONS
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
            "SQN": np.nan
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
            )
    }


# ============================================================
# FEATURES TO TEST
#
# Prefer normalized versions where appropriate.
# ============================================================

FEATURES = [

    "HistCrossNorm",

    "HistDelta1Norm",

    "HistSlope2Norm",

    "HistSlope3Norm",

    "HistSlope4Norm",

    "PreAreaNorm",

    "PreBars",

    "PreAreaPerBarNorm",

    "PriceMove15ATR",

    "PriceMove30ATR",

    "PriceMove60ATR",

    "Efficiency4",

    "Efficiency8",

    "CrossBodyATR",

    "CrossRangeATR",

    "CloseLocation"
]


# ============================================================
# CREATE FULL-SAMPLE QUINTILES
#
# Exploratory only.
#
# Thresholds are determined from complete dataset.
# Later we can do walk-forward thresholds.
# ============================================================

for feature in FEATURES:

    quintile_col = (
        feature
        +
        "_Quintile"
    )


    valid = trades[
        feature
    ].notna()


    try:

        trades.loc[
            valid,
            quintile_col
        ] = pd.qcut(
            trades.loc[
                valid,
                feature
            ],
            q=5,
            labels=[
                "Q1",
                "Q2",
                "Q3",
                "Q4",
                "Q5"
            ],
            duplicates="drop"
        )

    except ValueError:

        trades[
            quintile_col
        ] = np.nan


# ============================================================
# QUINTILE RESULTS
# ============================================================

overall_rows = []
year_rows = []


for feature in FEATURES:

    quintile_col = (
        feature
        +
        "_Quintile"
    )


    for q in [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5"
    ]:

        d = trades[
            trades[
                quintile_col
            ]
            ==
            q
        ]


        stats = calculate_stats(
            d
        )


        overall_rows.append({

            "Feature":
                feature,

            "Quintile":
                q,

            "FeatureMin":
                d[feature].min()
                if len(d) > 0
                else np.nan,

            "FeatureMax":
                d[feature].max()
                if len(d) > 0
                else np.nan,

            **stats
        })


        for year in STABILITY_YEARS:

            yd = d[
                d["Year"]
                ==
                year
            ]


            year_stats = calculate_stats(
                yd
            )


            year_rows.append({

                "Feature":
                    feature,

                "Quintile":
                    q,

                "Year":
                    year,

                **year_stats
            })


overall_df = pd.DataFrame(
    overall_rows
)

year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# STABILITY SUMMARY
# ============================================================

stability_rows = []


for _, row in overall_df.iterrows():

    feature = row[
        "Feature"
    ]

    quintile = row[
        "Quintile"
    ]


    y = year_df[
        (year_df["Feature"] == feature)
        &
        (year_df["Quintile"] == quintile)
    ]


    pf = (
        y["ProfitFactor"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )


    stability_rows.append({

        "Feature":
            feature,

        "Quintile":
            quintile,

        "N":
            row["N"],

        "FeatureMin":
            row["FeatureMin"],

        "FeatureMax":
            row["FeatureMax"],

        "WinRatePct":
            row["WinRatePct"],

        "NetPoints":
            row["NetPoints"],

        "AveragePnL":
            row["AveragePnL"],

        "ProfitFactor":
            row["ProfitFactor"],

        "SQN":
            row["SQN"],

        "WorstYearPF":
            (
                pf.min()
                if len(pf) > 0
                else np.nan
            ),

        "MeanYearPF":
            (
                pf.mean()
                if len(pf) > 0
                else np.nan
            ),

        "ProfitableYears":
            int(
                np.sum(
                    y[
                        "NetPoints"
                    ]
                    > 0
                )
            ),

        "PFAbove1Years":
            int(
                np.sum(
                    y[
                        "ProfitFactor"
                    ]
                    > 1
                )
            ),

        "MinYearTrades":
            y[
                "N"
            ].min()
    })


stability_df = pd.DataFrame(
    stability_rows
)


# ============================================================
# FEATURE TREND SUMMARY
#
# This lets us see whether performance generally improves
# from Q1 -> Q5.
# ============================================================

trend_rows = []


for feature in FEATURES:

    f = overall_df[
        overall_df[
            "Feature"
        ]
        ==
        feature
    ].copy()


    f["QNum"] = (
        f["Quintile"]
        .str.replace(
            "Q",
            "",
            regex=False
        )
        .astype(int)
    )


    f = f.sort_values(
        "QNum"
    )


    avg_values = f[
        "AveragePnL"
    ].to_numpy()


    pf_values = f[
        "ProfitFactor"
    ].to_numpy()


    # Simple correlation between quintile number and result.
    qnums = f[
        "QNum"
    ].to_numpy()


    avg_corr = (
        np.corrcoef(
            qnums,
            avg_values
        )[0, 1]
        if len(f) == 5
        else np.nan
    )


    pf_corr = (
        np.corrcoef(
            qnums,
            pf_values
        )[0, 1]
        if len(f) == 5
        and
        np.all(
            np.isfinite(
                pf_values
            )
        )
        else np.nan
    )


    trend_rows.append({

        "Feature":
            feature,

        "Q1_AvgPnL":
            f.iloc[0]["AveragePnL"],

        "Q5_AvgPnL":
            f.iloc[-1]["AveragePnL"],

        "Q1_PF":
            f.iloc[0]["ProfitFactor"],

        "Q5_PF":
            f.iloc[-1]["ProfitFactor"],

        "AvgPnL_Q1_to_Q5_Change":
            (
                f.iloc[-1]["AveragePnL"]
                -
                f.iloc[0]["AveragePnL"]
            ),

        "PF_Q1_to_Q5_Change":
            (
                f.iloc[-1]["ProfitFactor"]
                -
                f.iloc[0]["ProfitFactor"]
            ),

        "AvgPnLQuintileCorrelation":
            avg_corr,

        "PFQuintileCorrelation":
            pf_corr
    })


trend_df = pd.DataFrame(
    trend_rows
)


# ============================================================
# SAVE
# ============================================================

trade_file = (
    OUTPUT_DIR
    /
    "macd_momentum_feature_dataset.csv"
)

overall_file = (
    OUTPUT_DIR
    /
    "feature_quintile_overall.csv"
)

year_file = (
    OUTPUT_DIR
    /
    "feature_quintile_year_by_year.csv"
)

stability_file = (
    OUTPUT_DIR
    /
    "feature_quintile_stability.csv"
)

trend_file = (
    OUTPUT_DIR
    /
    "feature_strength_trend_summary.csv"
)


trades.to_csv(
    trade_file,
    index=False
)

overall_df.to_csv(
    overall_file,
    index=False
)

year_df.to_csv(
    year_file,
    index=False
)

stability_df.to_csv(
    stability_file,
    index=False
)

trend_df.to_csv(
    trend_file,
    index=False
)


# ============================================================
# PRINT
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    320
)

pd.set_option(
    "display.max_rows",
    500
)


print()
print("=" * 180)
print("FEATURE TREND SUMMARY — Q1 VS Q5")
print("=" * 180)

print(
    trend_df
    .sort_values(
        "AvgPnL_Q1_to_Q5_Change",
        ascending=False
    )
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 180)
print("ALL FEATURE QUINTILES")
print("=" * 180)

print(
    stability_df
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 180)
print("TOP Q5 GROUPS BY PROFIT FACTOR")
print("=" * 180)

print(
    stability_df[
        stability_df[
            "Quintile"
        ]
        ==
        "Q5"
    ]
    .sort_values(
        "ProfitFactor",
        ascending=False
    )
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 180)
print("DONE")
print("=" * 180)

print()
print("Output folder:")
print(OUTPUT_DIR)