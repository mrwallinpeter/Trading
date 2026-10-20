import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==================================================================================================
# SETTINGS
# ==================================================================================================

CSV_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "MACD_RF_V2_PRICEMOVE_RSI_ATRPCT"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14
ATR_PERCENTILE_LOOKBACK = 100

RSI_LENGTH = 14

HOLD_MINUTES = 210

START_YEAR = 2022
END_YEAR = 2026

# Same basic RF architecture as V0
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = 5
RF_MIN_SAMPLES_LEAF = 25
RF_RANDOM_STATE = 42


# ==================================================================================================
# HELPERS
# ==================================================================================================

def section(title):
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


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


def wilder_rsi(close, length):
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(
        alpha=1.0 / length,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1.0 / length,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    return 100.0 - (100.0 / (1.0 + rs))


def rolling_percentile(series, lookback):
    return series.rolling(
        lookback,
        min_periods=lookback
    ).apply(
        lambda x: np.mean(x <= x[-1]),
        raw=True
    )


def profit_factor(pnls):
    gross_profit = pnls[pnls > 0].sum()
    gross_loss = -pnls[pnls < 0].sum()

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else np.nan

    return gross_profit / gross_loss


def max_drawdown(pnls):
    if len(pnls) == 0:
        return np.nan

    equity = pnls.cumsum()
    peak = equity.cummax()

    return (peak - equity).max()


def summarize_returns(df):
    if len(df) == 0:
        return {
            "N": 0,
            "AvgReturn": np.nan,
            "NetReturn": np.nan,
            "PF": np.nan,
            "WinPct": np.nan,
            "MaxDD": np.nan
        }

    r = df["ActualReturn"]

    return {
        "N": len(df),
        "AvgReturn": r.mean(),
        "NetReturn": r.sum(),
        "PF": profit_factor(r),
        "WinPct": (r > 0).mean() * 100,
        "MaxDD": max_drawdown(r)
    }


# ==================================================================================================
# LOAD DATA
# ==================================================================================================

section("LOADING DATA")

df1 = pd.read_csv(CSV_PATH)

possible_time_cols = [
    "DateTime",
    "Datetime",
    "datetime",
    "Timestamp",
    "timestamp",
    "Time",
    "time",
    "Date",
    "date"
]

time_col = None

for c in possible_time_cols:
    if c in df1.columns:
        time_col = c
        break

if time_col is None:
    raise ValueError("Could not find timestamp column.")

df1[time_col] = pd.to_datetime(df1[time_col])
df1 = df1.set_index(time_col).sort_index()

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

print(f"1m rows: {len(df1):,}")
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

print(f"15m bars: {len(df15):,}")


# ==================================================================================================
# MACD
# ==================================================================================================

section("CALCULATING MACD + FEATURES")

ema_fast = df15["Close"].ewm(
    span=MACD_FAST,
    adjust=False
).mean()

ema_slow = df15["Close"].ewm(
    span=MACD_SLOW,
    adjust=False
).mean()

macd = ema_fast - ema_slow

signal_line = macd.ewm(
    span=MACD_SIGNAL,
    adjust=False
).mean()

df15["MACDHist"] = macd - signal_line

df15["BullCross"] = (
    (df15["MACDHist"].shift(1) <= 0) &
    (df15["MACDHist"] > 0)
)


# ==================================================================================================
# ATR
# ==================================================================================================

df15["ATR14"] = wilder_atr(
    df15,
    ATR_LENGTH
)


# ==================================================================================================
# FEATURE 1 — PRICEMOVE ATR
# ==================================================================================================

df15["PriceMoveATR"] = (
    (df15["Close"] - df15["Close"].shift(1)) /
    df15["ATR14"]
)


# ==================================================================================================
# FEATURE 2 — RSI
# ==================================================================================================

df15["RSI14"] = wilder_rsi(
    df15["Close"],
    RSI_LENGTH
)


# ==================================================================================================
# FEATURE 3 — ATR PERCENTILE
# ==================================================================================================

df15["ATRPercentile100"] = rolling_percentile(
    df15["ATR14"],
    ATR_PERCENTILE_LOOKBACK
)


# ==================================================================================================
# EVENT UNIVERSE
# ==================================================================================================

events = df15[
    df15["BullCross"]
].copy()

events = events[
    (events.index.year >= START_YEAR) &
    (events.index.year <= END_YEAR)
].copy()

events = events.dropna(
    subset=[
        "PriceMoveATR",
        "RSI14",
        "ATRPercentile100"
    ]
)

section("MACD EVENT UNIVERSE")

print(f"Bullish MACD crosses: {len(events):,}")


# ==================================================================================================
# BUILD TARGET
# ==================================================================================================

section("CALCULATING 210-MINUTE FORWARD RETURN")

times = df1.index.values
opens = df1["Open"].to_numpy()

records = []

for signal_time, row in events.iterrows():

    entry_pos = np.searchsorted(
        times,
        np.datetime64(signal_time),
        side="right"
    )

    if entry_pos >= len(df1):
        continue

    entry_time = df1.index[entry_pos]
    entry_price = opens[entry_pos]

    target_exit_time = (
        entry_time +
        pd.Timedelta(minutes=HOLD_MINUTES)
    )

    exit_pos = np.searchsorted(
        times,
        np.datetime64(target_exit_time),
        side="left"
    )

    if exit_pos >= len(df1):
        continue

    exit_time = df1.index[exit_pos]

    if exit_time > target_exit_time + pd.Timedelta(minutes=2):
        continue

    exit_price = opens[exit_pos]

    records.append({
        "SignalTime": signal_time,
        "EntryTime": entry_time,
        "ExitTime": exit_time,
        "Year": signal_time.year,

        "PriceMoveATR": row["PriceMoveATR"],
        "RSI14": row["RSI14"],
        "ATRPercentile100": row["ATRPercentile100"],

        "ActualReturn": exit_price - entry_price
    })


dataset = pd.DataFrame(records)

print(f"Valid events with target: {len(dataset):,}")

print()
print("Events by year:")
print(dataset["Year"].value_counts().sort_index())


# ==================================================================================================
# FEATURES
# ==================================================================================================

FEATURES = [
    "PriceMoveATR",
    "RSI14",
    "ATRPercentile100"
]


# ==================================================================================================
# WALK-FORWARD
# ==================================================================================================

section("WALK-FORWARD OUT-OF-SAMPLE TEST")

walk_forward_tests = [
    ([2022], 2023),
    ([2022, 2023], 2024),
    ([2022, 2023, 2024], 2025),
    ([2022, 2023, 2024, 2025], 2026)
]

all_oos = []
fold_rows = []
importance_rows = []

for train_years, test_year in walk_forward_tests:

    train = dataset[
        dataset["Year"].isin(train_years)
    ].copy()

    test_df = dataset[
        dataset["Year"] == test_year
    ].copy()

    X_train = train[FEATURES]
    y_train = train["ActualReturn"]

    X_test = test_df[FEATURES]
    y_test = test_df["ActualReturn"]

    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,

        # all three features available at each split
        max_features=3,

        random_state=RF_RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    fold = test_df.copy()

    fold["PredictedReturn"] = pred
    fold["TestYear"] = test_year

    all_oos.append(
        fold
    )

    mae = mean_absolute_error(
        y_test,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )

    corr = np.corrcoef(
        y_test,
        pred
    )[0, 1]

    fold_rows.append({
        "TrainYears": f"{min(train_years)}-{max(train_years)}",
        "TestYear": test_year,
        "TrainN": len(train),
        "TestN": len(test_df),
        "MAE": mae,
        "RMSE": rmse,
        "PredictionActualCorr": corr,
        "AvgPrediction": pred.mean(),
        "AvgActual": y_test.mean()
    })

    for feature, importance in zip(
        FEATURES,
        model.feature_importances_
    ):

        importance_rows.append({
            "TestYear": test_year,
            "Feature": feature,
            "Importance": importance
        })

    print()
    print(
        f"Train {min(train_years)}-{max(train_years)} "
        f"-> Test {test_year}"
    )

    print(
        f"Train N: {len(train):,} | "
        f"Test N: {len(test_df):,}"
    )

    print(
        f"MAE: {mae:.3f} | "
        f"RMSE: {rmse:.3f} | "
        f"Corr: {corr:.3f}"
    )


# ==================================================================================================
# COMBINE TRUE OOS PREDICTIONS
# ==================================================================================================

oos = pd.concat(
    all_oos,
    ignore_index=True
)

fold_summary = pd.DataFrame(
    fold_rows
)

importance_summary = pd.DataFrame(
    importance_rows
)


# ==================================================================================================
# OVERALL QUINTILES
# ==================================================================================================

section("OUT-OF-SAMPLE PREDICTION QUINTILES")

quintile_labels = [
    "Q1 Lowest",
    "Q2",
    "Q3",
    "Q4",
    "Q5 Highest"
]

oos["PredictionQuintile"] = pd.qcut(
    oos["PredictedReturn"],
    q=5,
    labels=quintile_labels,
    duplicates="drop"
)

quintile_rows = []

for bucket in quintile_labels:

    g = oos[
        oos["PredictionQuintile"].astype(str) == bucket
    ]

    s = summarize_returns(g)

    quintile_rows.append({
        "Bucket": bucket,
        "N": s["N"],
        "AvgPrediction": g["PredictedReturn"].mean(),
        "AvgActualReturn": s["AvgReturn"],
        "NetReturn": s["NetReturn"],
        "PF": s["PF"],
        "WinPct": s["WinPct"],
        "MaxDD": s["MaxDD"]
    })


quintile_summary = pd.DataFrame(
    quintile_rows
)


pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 260)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")


print(
    quintile_summary.to_string(
        index=False
    )
)


# ==================================================================================================
# DECILES
# ==================================================================================================

section("OUT-OF-SAMPLE PREDICTION DECILES")

decile_labels = [
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
    "D8",
    "D9",
    "D10 Highest"
]

oos["PredictionDecile"] = pd.qcut(
    oos["PredictedReturn"],
    q=10,
    labels=decile_labels,
    duplicates="drop"
)

decile_rows = []

for bucket in decile_labels:

    g = oos[
        oos["PredictionDecile"].astype(str) == bucket
    ]

    s = summarize_returns(g)

    decile_rows.append({
        "Bucket": bucket,
        "N": s["N"],
        "AvgPrediction": g["PredictedReturn"].mean(),
        "AvgActualReturn": s["AvgReturn"],
        "NetReturn": s["NetReturn"],
        "PF": s["PF"],
        "WinPct": s["WinPct"],
        "MaxDD": s["MaxDD"]
    })


decile_summary = pd.DataFrame(
    decile_rows
)

print(
    decile_summary.to_string(
        index=False
    )
)


# ==================================================================================================
# YEAR-BY-YEAR QUINTILES
# ==================================================================================================

section("YEAR-BY-YEAR OOS QUINTILES")

year_rows = []

for year in sorted(
    oos["Year"].unique()
):

    gy = oos[
        oos["Year"] == year
    ].copy()

    gy["YearQuintile"] = pd.qcut(
        gy["PredictedReturn"],
        q=5,
        labels=quintile_labels,
        duplicates="drop"
    )

    for bucket in quintile_labels:

        g = gy[
            gy["YearQuintile"].astype(str) == bucket
        ]

        s = summarize_returns(g)

        year_rows.append({
            "Year": year,
            "Bucket": bucket,
            "N": s["N"],
            "AvgPrediction": g["PredictedReturn"].mean(),
            "AvgActualReturn": s["AvgReturn"],
            "NetReturn": s["NetReturn"],
            "PF": s["PF"],
            "WinPct": s["WinPct"]
        })


year_summary = pd.DataFrame(
    year_rows
)

print(
    year_summary.to_string(
        index=False
    )
)


# ==================================================================================================
# TOP PREDICTION GROUPS
# ==================================================================================================

section("TOP OOS PREDICTION GROUPS")

threshold_rows = []

for percentile in [
    50,
    60,
    70,
    80,
    90
]:

    cutoff = np.percentile(
        oos["PredictedReturn"],
        percentile
    )

    g = oos[
        oos["PredictedReturn"] >= cutoff
    ]

    s = summarize_returns(g)

    threshold_rows.append({
        "TopGroup": f"Top {100 - percentile}%",
        "PredictionCutoff": cutoff,
        "N": s["N"],
        "AvgActualReturn": s["AvgReturn"],
        "NetReturn": s["NetReturn"],
        "PF": s["PF"],
        "WinPct": s["WinPct"],
        "MaxDD": s["MaxDD"]
    })


threshold_summary = pd.DataFrame(
    threshold_rows
)

print(
    threshold_summary.to_string(
        index=False
    )
)


# ==================================================================================================
# FEATURE IMPORTANCE
# ==================================================================================================

section("RF FEATURE IMPORTANCE BY OOS FOLD")

print(
    importance_summary.to_string(
        index=False
    )
)

print()

avg_importance = (
    importance_summary
    .groupby("Feature")["Importance"]
    .mean()
    .sort_values(ascending=False)
)

print("Average importance:")
print(avg_importance)


# ==================================================================================================
# WALK-FORWARD SUMMARY
# ==================================================================================================

section("WALK-FORWARD SUMMARY")

print(
    fold_summary.to_string(
        index=False
    )
)


# ==================================================================================================
# SANITY CHECK
# ==================================================================================================

section("OOS SANITY CHECK")

overall_corr = np.corrcoef(
    oos["PredictedReturn"],
    oos["ActualReturn"]
)[0, 1]

lowest = quintile_summary.iloc[0]
highest = quintile_summary.iloc[-1]

print(f"Total OOS events: {len(oos):,}")
print(f"OOS prediction/actual correlation: {overall_corr:.4f}")

print()

print(
    f"Lowest quintile actual avg:  "
    f"{lowest['AvgActualReturn']:.3f}"
)

print(
    f"Highest quintile actual avg: "
    f"{highest['AvgActualReturn']:.3f}"
)

print(
    f"Q5 - Q1 spread:             "
    f"{highest['AvgActualReturn'] - lowest['AvgActualReturn']:.3f}"
)


# ==================================================================================================
# DIRECT V0 REFERENCE
# ==================================================================================================

section("REFERENCE — RF V0")

print("V0 = PriceMoveATR + RSI14")
print()
print("V0 OOS correlation: 0.0182")
print("V0 Q1 AvgReturn:     0.155")
print("V0 Q4 AvgReturn:     1.225")
print("V0 Q4 PF:            1.206")
print("V0 Q5 AvgReturn:     1.244")
print("V0 Q5 PF:            1.194")
print("V0 Q5-Q1 spread:     1.088")
print("V0 Top 10% PF:       1.212")


# ==================================================================================================
# SAVE
# ==================================================================================================

oos.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "oos_predictions.csv"
    ),
    index=False
)

fold_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "walk_forward_summary.csv"
    ),
    index=False
)

quintile_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "oos_quintiles.csv"
    ),
    index=False
)

decile_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "oos_deciles.csv"
    ),
    index=False
)

year_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "oos_year_quintiles.csv"
    ),
    index=False
)

threshold_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "oos_top_groups.csv"
    ),
    index=False
)

importance_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_importance.csv"
    ),
    index=False
)


# ==================================================================================================
# DONE
# ==================================================================================================

section("DONE")

print()
print("RF V2")
print("Event universe: ALL bullish MACD zero-crosses")

print()
print("Features:")
print("  1. PriceMoveATR")
print("  2. RSI14")
print("  3. ATRPercentile100")

print()
print("Target:")
print("  210-minute forward return")

print()
print("Question:")
print("Does volatility state add incremental OOS information to V0?")

print()
print("Output directory:")
print(OUTPUT_DIR)