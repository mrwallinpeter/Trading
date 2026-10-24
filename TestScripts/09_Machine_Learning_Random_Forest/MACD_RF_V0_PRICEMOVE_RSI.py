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
    "MACD_RF_V0_PRICEMOVE_RSI"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# MACD
MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

# Features
ATR_LENGTH = 14
RSI_LENGTH = 14

# Target
HOLD_MINUTES = 210

# Research years
START_YEAR = 2022
END_YEAR = 2026

# Random Forest
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = 5
RF_MIN_SAMPLES_LEAF = 25
RF_MAX_FEATURES = 2
RF_RANDOM_STATE = 42

# Prediction ranking
N_QUINTILES = 5
N_DECILES = 10


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


def summarize_returns(df, return_col="ActualReturn"):
    if len(df) == 0:
        return {
            "N": 0,
            "AvgReturn": np.nan,
            "NetReturn": np.nan,
            "PF": np.nan,
            "WinPct": np.nan,
            "MaxDD": np.nan
        }

    r = df[return_col]

    return {
        "N": len(df),
        "AvgReturn": r.mean(),
        "NetReturn": r.sum(),
        "PF": profit_factor(r),
        "WinPct": (r > 0).mean() * 100,
        "MaxDD": max_drawdown(r)
    }


# ==================================================================================================
# LOAD 1-MINUTE DATA
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
# BUILD 15-MINUTE DATA
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
# FEATURE 1: PRICEMOVE ATR
# ==================================================================================================

df15["ATR14"] = wilder_atr(
    df15,
    ATR_LENGTH
)

df15["PriceMoveATR"] = (
    (df15["Close"] - df15["Close"].shift(1)) /
    df15["ATR14"]
)


# ==================================================================================================
# FEATURE 2: RSI
# ==================================================================================================

df15["RSI14"] = wilder_rsi(
    df15["Close"],
    RSI_LENGTH
)


# ==================================================================================================
# BUILD EVENT DATASET
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
        "RSI14"
    ]
)

section("MACD EVENT UNIVERSE")

print(f"Bullish MACD crosses: {len(events):,}")


# ==================================================================================================
# BUILD 210-MINUTE TARGET FROM 1-MINUTE DATA
# ==================================================================================================

section("CALCULATING 210-MINUTE FORWARD RETURN")

times = df1.index.values
opens = df1["Open"].to_numpy()

records = []

for signal_time, row in events.iterrows():

    # Entry = first 1m open strictly after completed 15m signal bar
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

    actual_exit_time = df1.index[exit_pos]

    if actual_exit_time > target_exit_time + pd.Timedelta(minutes=2):
        continue

    exit_price = opens[exit_pos]

    actual_return = (
        exit_price -
        entry_price
    )

    records.append({
        "SignalTime": signal_time,
        "EntryTime": entry_time,
        "ExitTime": actual_exit_time,
        "Year": signal_time.year,
        "PriceMoveATR": row["PriceMoveATR"],
        "RSI14": row["RSI14"],
        "ActualReturn": actual_return
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
    "RSI14"
]


# ==================================================================================================
# WALK-FORWARD
# ==================================================================================================

section("WALK-FORWARD OUT-OF-SAMPLE TEST")

walk_forward_tests = [
    {
        "TrainYears": [2022],
        "TestYear": 2023
    },
    {
        "TrainYears": [2022, 2023],
        "TestYear": 2024
    },
    {
        "TrainYears": [2022, 2023, 2024],
        "TestYear": 2025
    },
    {
        "TrainYears": [2022, 2023, 2024, 2025],
        "TestYear": 2026
    }
]

all_oos = []
fold_rows = []
importance_rows = []

for test in walk_forward_tests:

    train_years = test["TrainYears"]
    test_year = test["TestYear"]

    train = dataset[
        dataset["Year"].isin(train_years)
    ].copy()

    test_df = dataset[
        dataset["Year"] == test_year
    ].copy()

    if len(train) == 0 or len(test_df) == 0:
        continue

    X_train = train[FEATURES]
    y_train = train["ActualReturn"]

    X_test = test_df[FEATURES]
    y_test = test_df["ActualReturn"]

    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features=RF_MAX_FEATURES,
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
    fold["TrainYears"] = f"{min(train_years)}-{max(train_years)}"
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

    correlation = np.corrcoef(
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
        "PredictionActualCorr": correlation,
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
        f"Corr: {correlation:.3f}"
    )


# ==================================================================================================
# COMBINE ALL TRUE OOS PREDICTIONS
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
# OOS QUINTILES
# ==================================================================================================

section("OUT-OF-SAMPLE PREDICTION QUINTILES")

oos["PredictionQuintile"] = pd.qcut(
    oos["PredictedReturn"],
    q=N_QUINTILES,
    labels=[
        "Q1 Lowest",
        "Q2",
        "Q3",
        "Q4",
        "Q5 Highest"
    ],
    duplicates="drop"
)

quintile_rows = []

for bucket in [
    "Q1 Lowest",
    "Q2",
    "Q3",
    "Q4",
    "Q5 Highest"
]:

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
# OOS DECILES
# ==================================================================================================

section("OUT-OF-SAMPLE PREDICTION DECILES")

oos["PredictionDecile"] = pd.qcut(
    oos["PredictedReturn"],
    q=N_DECILES,
    labels=[
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
    ],
    duplicates="drop"
)

decile_rows = []

decile_order = [
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

for bucket in decile_order:

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
# YEAR × QUINTILE
# ==================================================================================================

section("YEAR-BY-YEAR OOS QUINTILES")

year_quintile_rows = []

for year in sorted(
    oos["Year"].unique()
):

    gy = oos[
        oos["Year"] == year
    ].copy()

    # Important:
    # quintiles are formed inside each OOS year,
    # so we compare high vs low predictions within that year.
    gy["YearQuintile"] = pd.qcut(
        gy["PredictedReturn"],
        q=5,
        labels=[
            "Q1 Lowest",
            "Q2",
            "Q3",
            "Q4",
            "Q5 Highest"
        ],
        duplicates="drop"
    )

    for bucket in [
        "Q1 Lowest",
        "Q2",
        "Q3",
        "Q4",
        "Q5 Highest"
    ]:

        g = gy[
            gy["YearQuintile"].astype(str) == bucket
        ]

        s = summarize_returns(g)

        year_quintile_rows.append({
            "Year": year,
            "Bucket": bucket,
            "N": s["N"],
            "AvgPrediction": g["PredictedReturn"].mean(),
            "AvgActualReturn": s["AvgReturn"],
            "NetReturn": s["NetReturn"],
            "PF": s["PF"],
            "WinPct": s["WinPct"]
        })

year_quintile_summary = pd.DataFrame(
    year_quintile_rows
)

print(
    year_quintile_summary.to_string(
        index=False
    )
)


# ==================================================================================================
# TOP-PREDICTION THRESHOLDS
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
# SIMPLE SANITY CHECK
# ==================================================================================================

section("OOS SANITY CHECK")

overall_corr = np.corrcoef(
    oos["PredictedReturn"],
    oos["ActualReturn"]
)[0, 1]

print(f"Total OOS events: {len(oos):,}")
print(f"OOS prediction/actual correlation: {overall_corr:.4f}")

lowest = quintile_summary.iloc[0]
highest = quintile_summary.iloc[-1]

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
    f"Difference:                  "
    f"{highest['AvgActualReturn'] - lowest['AvgActualReturn']:.3f}"
)


# ==================================================================================================
# SAVE OUTPUTS
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

year_quintile_summary.to_csv(
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
print("RF V0")
print("Event universe: ALL bullish MACD zero-crosses")
print()
print("Features:")
print("  1. PriceMoveATR")
print("  2. RSI14")
print()
print("Target:")
print("  210-minute forward return")
print()
print("Validation:")
print("  expanding chronological walk-forward")
print("  predictions evaluated only on unseen years")
print()
print("Output directory:")
print(OUTPUT_DIR)