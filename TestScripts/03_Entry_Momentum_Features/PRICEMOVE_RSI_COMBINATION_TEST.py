import numpy as np
import pandas as pd
from itertools import combinations


# ==================================================================================================
# SETTINGS
# ==================================================================================================

CSV_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"

OUTPUT_MAIN = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\PRICEMOVE_RSI_COMBINATION_TEST.csv"
OUTPUT_YEAR = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\PRICEMOVE_RSI_COMBINATION_YEAR.csv"

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

ATR_LENGTH = 14
RSI_LENGTH = 14

HOLD_MINUTES = 210

START_YEAR = 2022
END_YEAR = 2026


# Same broad range for all three variants
FILTER_MIN = 0.35
FILTER_MAX = 0.90


# RSI buckets
RSI_EDGES = [
    0,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100.0001
]


# ==================================================================================================
# HELPERS
# ==================================================================================================

def section(title):

    print()
    print("=" * 125)
    print(title)
    print("=" * 125)


def profit_factor(series):

    r = pd.Series(series).dropna()

    gross_profit = r[r > 0].sum()
    gross_loss = -r[r < 0].sum()

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else np.nan

    return gross_profit / gross_loss


def stats(df):

    if len(df) == 0:

        return {
            "N": 0,
            "Avg": np.nan,
            "PF": np.nan,
            "WinPct": np.nan,
            "TotalPts": np.nan
        }

    r = df["Return210"].dropna()

    return {
        "N": len(r),
        "Avg": r.mean(),
        "PF": profit_factor(r),
        "WinPct": (r > 0).mean() * 100.0,
        "TotalPts": r.sum()
    }


def calculate_year_stability(subset):

    positive_years = 0
    tested_years = 0
    yearly_avgs = []
    year_rows = []

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        y = subset[
            subset["Year"] == year
        ]

        if len(y) == 0:
            continue

        ys = stats(y)

        tested_years += 1

        if ys["Avg"] > 0:
            positive_years += 1

        yearly_avgs.append(
            ys["Avg"]
        )

        year_rows.append(
            (
                year,
                ys
            )
        )

    worst = (
        min(yearly_avgs)
        if yearly_avgs
        else np.nan
    )

    best = (
        max(yearly_avgs)
        if yearly_avgs
        else np.nan
    )

    return (
        positive_years,
        tested_years,
        worst,
        best,
        year_rows
    )


# ==================================================================================================
# LOAD DATA
# ==================================================================================================

section("LOADING DATA")

df1 = pd.read_csv(
    CSV_PATH
)


time_col = None

for c in [
    "DateTime",
    "Datetime",
    "datetime",
    "Timestamp",
    "timestamp",
    "Time",
    "time",
    "Date",
    "date"
]:

    if c in df1.columns:

        time_col = c
        break


if time_col is None:

    raise ValueError(
        "Could not find timestamp column."
    )


df1[time_col] = pd.to_datetime(
    df1[time_col]
)


df1 = (
    df1
    .set_index(time_col)
    .sort_index()
)


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


df1 = df1.rename(
    columns=rename_map
)


print(
    f"1m rows: {len(df1):,}"
)

print(
    f"Start: {df1.index.min()}"
)

print(
    f"End:   {df1.index.max()}"
)


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
).agg(
    agg
)


df15 = df15.dropna(
    subset=[
        "Open",
        "High",
        "Low",
        "Close"
    ]
)


print(
    f"15m bars: {len(df15):,}"
)


# ==================================================================================================
# MACD
# ==================================================================================================

section("MACD")


ema_fast = df15[
    "Close"
].ewm(
    span=MACD_FAST,
    adjust=False
).mean()


ema_slow = df15[
    "Close"
].ewm(
    span=MACD_SLOW,
    adjust=False
).mean()


macd = (
    ema_fast -
    ema_slow
)


macd_signal_line = macd.ewm(
    span=MACD_SIGNAL,
    adjust=False
).mean()


df15["MACDHist"] = (
    macd -
    macd_signal_line
)


df15["BullCross"] = (
    (df15["MACDHist"].shift(1) <= 0) &
    (df15["MACDHist"] > 0)
)


# ==================================================================================================
# ATR
# ==================================================================================================

section("ATR + THREE PRICE-MOVE FEATURES + RSI")


prev_close = df15[
    "Close"
].shift(1)


true_range = pd.concat(
    [
        df15["High"] - df15["Low"],

        (
            df15["High"] -
            prev_close
        ).abs(),

        (
            df15["Low"] -
            prev_close
        ).abs()
    ],
    axis=1
).max(
    axis=1
)


df15["ATR14"] = true_range.ewm(
    alpha=1 / ATR_LENGTH,
    adjust=False
).mean()


# ==================================================================================================
# A, B, C
# ==================================================================================================

# A:
# Current close minus previous close
df15["A"] = (
    df15["Close"] -
    df15["Close"].shift(1)
) / df15["ATR14"]


# B:
# Current close minus previous bar open
df15["B"] = (
    df15["Close"] -
    df15["Open"].shift(1)
) / df15["ATR14"]


# C:
# Current candle body
df15["C"] = (
    df15["Close"] -
    df15["Open"]
) / df15["ATR14"]


# ==================================================================================================
# RSI 14
# ==================================================================================================

delta = df15[
    "Close"
].diff()


gain = delta.clip(
    lower=0
)

loss = -delta.clip(
    upper=0
)


avg_gain = gain.ewm(
    alpha=1 / RSI_LENGTH,
    adjust=False
).mean()


avg_loss = loss.ewm(
    alpha=1 / RSI_LENGTH,
    adjust=False
).mean()


rs = (
    avg_gain /
    avg_loss
)


df15["RSI14"] = (
    100 -
    (
        100 /
        (1 + rs)
    )
)


# ==================================================================================================
# BOOLEAN FILTERS
# ==================================================================================================

df15["A_PASS"] = (
    (df15["A"] >= FILTER_MIN) &
    (df15["A"] <= FILTER_MAX)
)


df15["B_PASS"] = (
    (df15["B"] >= FILTER_MIN) &
    (df15["B"] <= FILTER_MAX)
)


df15["C_PASS"] = (
    (df15["C"] >= FILTER_MIN) &
    (df15["C"] <= FILTER_MAX)
)


# ==================================================================================================
# SIGNALS
# ==================================================================================================

signals = df15[
    df15["BullCross"]
].copy()


signals = signals[
    (signals.index.year >= START_YEAR) &
    (signals.index.year <= END_YEAR)
]


signals = signals.dropna(
    subset=[
        "A",
        "B",
        "C",
        "RSI14"
    ]
)


print(
    f"Bullish MACD crosses: {len(signals):,}"
)


# ==================================================================================================
# 210-MINUTE RETURNS
# ==================================================================================================

section("CALCULATING 210-MINUTE RETURNS")


times = df1.index.values

opens = df1[
    "Open"
].to_numpy()


valid_times = []
entry_times = []
entry_prices = []
exit_times = []
exit_prices = []
returns = []


for signal_time in signals.index:


    entry_pos = np.searchsorted(
        times,
        np.datetime64(signal_time),
        side="right"
    )


    if entry_pos >= len(df1):
        continue


    entry_time = df1.index[
        entry_pos
    ]

    entry_price = opens[
        entry_pos
    ]


    target_exit_time = (
        entry_time +
        pd.Timedelta(
            minutes=HOLD_MINUTES
        )
    )


    exit_pos = np.searchsorted(
        times,
        np.datetime64(target_exit_time),
        side="left"
    )


    if exit_pos >= len(df1):
        continue


    exit_time = df1.index[
        exit_pos
    ]


    if (
        exit_time >
        target_exit_time +
        pd.Timedelta(minutes=2)
    ):
        continue


    exit_price = opens[
        exit_pos
    ]


    valid_times.append(
        signal_time
    )

    entry_times.append(
        entry_time
    )

    entry_prices.append(
        entry_price
    )

    exit_times.append(
        exit_time
    )

    exit_prices.append(
        exit_price
    )

    returns.append(
        exit_price -
        entry_price
    )


signals = signals.loc[
    valid_times
].copy()


signals["EntryTime"] = entry_times
signals["EntryPrice"] = entry_prices

signals["ExitTime"] = exit_times
signals["ExitPrice"] = exit_prices

signals["Return210"] = returns

signals["Year"] = (
    signals.index.year
)


# ==================================================================================================
# BASELINE
# ==================================================================================================

section("BASELINE")


baseline = stats(
    signals
)


print(
    f"N:     {baseline['N']:,}"
)

print(
    f"Avg:   {baseline['Avg']:.3f}"
)

print(
    f"PF:    {baseline['PF']:.3f}"
)

print(
    f"Win %: {baseline['WinPct']:.2f}"
)


# ==================================================================================================
# RSI BUCKET
# ==================================================================================================

signals["RSIBucket"] = pd.cut(
    signals["RSI14"],
    bins=RSI_EDGES,
    right=False,
    include_lowest=True
)


result_rows = []
year_output_rows = []


# ==================================================================================================
# FUNCTION FOR RESULT
# ==================================================================================================

def record_result(
    feature_combo,
    rsi_bucket,
    subset
):

    s = stats(
        subset
    )


    (
        positive_years,
        tested_years,
        worst_year,
        best_year,
        year_details
    ) = calculate_year_stability(
        subset
    )


    result_rows.append(
        {
            "Combination":
                feature_combo,

            "RSIBucket":
                rsi_bucket,

            "N":
                s["N"],

            "Avg":
                s["Avg"],

            "PF":
                s["PF"],

            "WinPct":
                s["WinPct"],

            "TotalPts":
                s["TotalPts"],

            "AvgVsBaseline":
                s["Avg"] -
                baseline["Avg"],

            "PositiveYears":
                positive_years,

            "TestedYears":
                tested_years,

            "WorstYearAvg":
                worst_year,

            "BestYearAvg":
                best_year
        }
    )


    for year, ys in year_details:

        year_output_rows.append(
            {
                "Combination":
                    feature_combo,

                "RSIBucket":
                    rsi_bucket,

                "Year":
                    year,

                **ys
            }
        )


# ==================================================================================================
# RSI ALONE
# ==================================================================================================

section("RSI BUCKETS - NO A/B/C FILTER")


for bucket, subset in signals.groupby(
    "RSIBucket",
    observed=True
):


    record_result(
        "NONE",
        str(bucket),
        subset
    )


    s = stats(
        subset
    )


    (
        positive_years,
        tested_years,
        worst_year,
        best_year,
        _
    ) = calculate_year_stability(
        subset
    )


    print(
        f"{str(bucket):>14} | "
        f"N {s['N']:4d} | "
        f"Avg {s['Avg']:7.3f} | "
        f"PF {s['PF']:6.3f} | "
        f"Win {s['WinPct']:6.2f}% | "
        f"Years {positive_years}/{tested_years}"
    )


# ==================================================================================================
# ALL A/B/C COMBINATIONS
# ==================================================================================================

combo_definitions = {

    "A":
        signals["A_PASS"],

    "B":
        signals["B_PASS"],

    "C":
        signals["C_PASS"],

    "A+B":
        signals["A_PASS"] &
        signals["B_PASS"],

    "A+C":
        signals["A_PASS"] &
        signals["C_PASS"],

    "B+C":
        signals["B_PASS"] &
        signals["C_PASS"],

    "A+B+C":
        signals["A_PASS"] &
        signals["B_PASS"] &
        signals["C_PASS"]
}


# ==================================================================================================
# OVERALL COMBINATION PERFORMANCE
# ==================================================================================================

section("A/B/C COMBINATIONS - WITHOUT RSI FILTER")


for combo_name, mask in combo_definitions.items():


    subset = signals[
        mask
    ]


    record_result(
        combo_name,
        "ALL",
        subset
    )


    s = stats(
        subset
    )


    (
        positive_years,
        tested_years,
        worst_year,
        best_year,
        _
    ) = calculate_year_stability(
        subset
    )


    print(
        f"{combo_name:<8} | "
        f"N {s['N']:4d} | "
        f"Avg {s['Avg']:7.3f} | "
        f"PF {s['PF']:6.3f} | "
        f"Win {s['WinPct']:6.2f}% | "
        f"Years {positive_years}/{tested_years} | "
        f"Worst {worst_year:+7.3f}"
    )


# ==================================================================================================
# COMBINATIONS + RSI BUCKETS
# ==================================================================================================

for combo_name, mask in combo_definitions.items():


    section(
        f"{combo_name} + RSI BUCKETS"
    )


    filtered = signals[
        mask
    ]


    for bucket, subset in filtered.groupby(
        "RSIBucket",
        observed=True
    ):


        record_result(
            combo_name,
            str(bucket),
            subset
        )


        s = stats(
            subset
        )


        (
            positive_years,
            tested_years,
            worst_year,
            best_year,
            _
        ) = calculate_year_stability(
            subset
        )


        print(
            f"{str(bucket):>14} | "
            f"N {s['N']:4d} | "
            f"Avg {s['Avg']:7.3f} | "
            f"PF {s['PF']:6.3f} | "
            f"Win {s['WinPct']:6.2f}% | "
            f"Years {positive_years}/{tested_years}"
        )


# ==================================================================================================
# RESULTS
# ==================================================================================================

results = pd.DataFrame(
    result_rows
)


year_results = pd.DataFrame(
    year_output_rows
)


# ==================================================================================================
# RANK USEFUL RESULTS
# ==================================================================================================

section("BEST COMBINATIONS + RSI BUCKETS - N >= 100")


ranked = results[
    (results["N"] >= 100) &
    (results["RSIBucket"] != "ALL")
].copy()


ranked = ranked.sort_values(
    [
        "PositiveYears",
        "PF",
        "Avg"
    ],
    ascending=[
        False,
        False,
        False
    ]
)


print(
    ranked[
        [
            "Combination",
            "RSIBucket",
            "N",
            "Avg",
            "PF",
            "WinPct",
            "PositiveYears",
            "WorstYearAvg"
        ]
    ].head(
        40
    ).to_string(
        index=False
    )
)


# ==================================================================================================
# EACH COMBINATION'S BEST RSI BUCKET
# ==================================================================================================

section("BEST RSI BUCKET FOR EACH A/B/C COMBINATION")


for combo_name in [
    "NONE",
    "A",
    "B",
    "C",
    "A+B",
    "A+C",
    "B+C",
    "A+B+C"
]:


    temp = results[
        (results["Combination"] == combo_name) &
        (results["RSIBucket"] != "ALL") &
        (results["N"] >= 100)
    ].copy()


    if len(temp) == 0:

        continue


    temp = temp.sort_values(
        [
            "PositiveYears",
            "PF",
            "Avg"
        ],
        ascending=[
            False,
            False,
            False
        ]
    )


    best = temp.iloc[0]


    print(
        f"{combo_name:<8} | "
        f"RSI {best['RSIBucket']:<14} | "
        f"N {int(best['N']):4d} | "
        f"Avg {best['Avg']:7.3f} | "
        f"PF {best['PF']:6.3f} | "
        f"Years {int(best['PositiveYears'])}/{int(best['TestedYears'])} | "
        f"Worst {best['WorstYearAvg']:+7.3f}"
    )


# ==================================================================================================
# SPECIAL BROAD RSI RANGES
# ==================================================================================================

section("BROAD RSI RANGES")


rsi_ranges = {

    "RSI 30-40":
        (
            (signals["RSI14"] >= 30) &
            (signals["RSI14"] < 40)
        ),

    "RSI 40-50":
        (
            (signals["RSI14"] >= 40) &
            (signals["RSI14"] < 50)
        ),

    "RSI 50-60":
        (
            (signals["RSI14"] >= 50) &
            (signals["RSI14"] < 60)
        ),

    "RSI 30-50":
        (
            (signals["RSI14"] >= 30) &
            (signals["RSI14"] < 50)
        ),

    "RSI 40-60":
        (
            (signals["RSI14"] >= 40) &
            (signals["RSI14"] < 60)
        ),

    "RSI 30-60":
        (
            (signals["RSI14"] >= 30) &
            (signals["RSI14"] < 60)
        )
}


for combo_name, combo_mask in combo_definitions.items():

    print()
    print(
        f"--- {combo_name} ---"
    )


    for rsi_name, rsi_mask in rsi_ranges.items():

        subset = signals[
            combo_mask &
            rsi_mask
        ]


        s = stats(
            subset
        )


        (
            positive_years,
            tested_years,
            worst_year,
            best_year,
            _
        ) = calculate_year_stability(
            subset
        )


        print(
            f"{rsi_name:<12} | "
            f"N {s['N']:4d} | "
            f"Avg {s['Avg']:7.3f} | "
            f"PF {s['PF']:6.3f} | "
            f"Years {positive_years}/{tested_years}"
        )


# ==================================================================================================
# CORRELATIONS
# ==================================================================================================

section("FEATURE CORRELATIONS")


corr = signals[
    [
        "A",
        "B",
        "C",
        "RSI14"
    ]
].corr()


print(
    corr.to_string()
)


# ==================================================================================================
# SAVE
# ==================================================================================================

results.to_csv(
    OUTPUT_MAIN,
    index=False
)


year_results.to_csv(
    OUTPUT_YEAR,
    index=False
)


section("DONE")


print(
    f"Main results:\n{OUTPUT_MAIN}"
)

print()

print(
    f"Year results:\n{OUTPUT_YEAR}"
)