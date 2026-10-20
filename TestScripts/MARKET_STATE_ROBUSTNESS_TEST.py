import numpy as np
import pandas as pd


# ==================================================================================================
# SETTINGS
# ==================================================================================================

INPUT_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\MARKET_STATE_SIGNALS.csv"

OUTPUT_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\MARKET_STATE_ROBUSTNESS_RESULTS.csv"

MONTH_OUTPUT_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\MARKET_STATE_ROBUSTNESS_MONTHLY.csv"


# Recent-period test
RECENT_START = "2026-05-01"
RECENT_END   = "2026-09-18 23:59:59"


# ==================================================================================================
# HELPERS
# ==================================================================================================

def section(title):
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def profit_factor(series):
    r = pd.Series(series).dropna()

    gp = r[r > 0].sum()
    gl = -r[r < 0].sum()

    if gl == 0:
        return np.inf if gp > 0 else np.nan

    return gp / gl


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


def print_stats(name, df):
    s = stats(df)

    print(
        f"{name:<30} "
        f"N {s['N']:4d} | "
        f"Avg {s['Avg']:7.3f} | "
        f"PF {s['PF']:6.3f} | "
        f"Win {s['WinPct']:6.2f}% | "
        f"Total {s['TotalPts']:9.2f}"
    )

    return s


# ==================================================================================================
# LOAD SIGNAL DATA
# ==================================================================================================

section("LOADING SIGNAL DATA")

df = pd.read_csv(
    INPUT_PATH
)

# First column may contain the original signal timestamp
first_col = df.columns[0]

if first_col not in [
    "SignalTime",
    "EntryTime",
    "ExitTime"
]:
    df = df.rename(
        columns={
            first_col: "SignalTime"
        }
    )


for c in [
    "SignalTime",
    "EntryTime",
    "ExitTime"
]:
    if c in df.columns:
        df[c] = pd.to_datetime(
            df[c]
        )


if "SignalTime" not in df.columns:
    raise ValueError(
        "SignalTime column not found."
    )


df["Year"] = df[
    "SignalTime"
].dt.year

df["Month"] = df[
    "SignalTime"
].dt.to_period(
    "M"
).astype(str)


print(
    f"Signals loaded: {len(df):,}"
)

print(
    f"Start: {df['SignalTime'].min()}"
)

print(
    f"End:   {df['SignalTime'].max()}"
)


# ==================================================================================================
# DEFINE CANDIDATE STATES
# ==================================================================================================

baseline = df.copy()


# --------------------------------------------------------------------------------------------------
# Candidate 1:
# Moderate downtrend
# --------------------------------------------------------------------------------------------------

down23 = df[
    (df["TrendDirection"] == "DOWN") &
    (df["TrendCode"].isin([2, 3]))
].copy()


# --------------------------------------------------------------------------------------------------
# Candidate 2:
# Moderate downtrend + BB width 2-4
# --------------------------------------------------------------------------------------------------

down23_bb24 = df[
    (df["TrendDirection"] == "DOWN") &
    (df["TrendCode"].isin([2, 3])) &
    (df["BBWidthCode"].isin([2, 3, 4]))
].copy()


# --------------------------------------------------------------------------------------------------
# Comparison:
# Downtrend too strong
# --------------------------------------------------------------------------------------------------

down45 = df[
    (df["TrendDirection"] == "DOWN") &
    (df["TrendCode"].isin([4, 5]))
].copy()


# --------------------------------------------------------------------------------------------------
# Comparison:
# BB extremes while down 2-3
# --------------------------------------------------------------------------------------------------

down23_bb1 = df[
    (df["TrendDirection"] == "DOWN") &
    (df["TrendCode"].isin([2, 3])) &
    (df["BBWidthCode"] == 1)
].copy()


down23_bb5 = df[
    (df["TrendDirection"] == "DOWN") &
    (df["TrendCode"].isin([2, 3])) &
    (df["BBWidthCode"] == 5)
].copy()


groups = {
    "BASELINE": baseline,
    "DOWN 2-3": down23,
    "DOWN 2-3 + BB 2-4": down23_bb24,
    "DOWN 4-5": down45,
    "DOWN 2-3 + BB 1": down23_bb1,
    "DOWN 2-3 + BB 5": down23_bb5
}


# ==================================================================================================
# OVERALL RESULTS
# ==================================================================================================

section("OVERALL")

overall_rows = []

for name, subset in groups.items():

    s = print_stats(
        name,
        subset
    )

    overall_rows.append(
        {
            "Period": "ALL",
            "Group": name,
            **s
        }
    )


# ==================================================================================================
# YEAR-BY-YEAR
# ==================================================================================================

section("YEAR-BY-YEAR")

year_rows = []

for year in range(
    2022,
    2027
):

    print()
    print(
        f"--- {year} ---"
    )

    for name, subset in groups.items():

        y = subset[
            subset["Year"] == year
        ]

        s = print_stats(
            name,
            y
        )

        year_rows.append(
            {
                "Period": str(year),
                "Group": name,
                **s
            }
        )


# ==================================================================================================
# POSITIVE YEAR COUNTS
# ==================================================================================================

section("YEAR STABILITY")

for name in groups.keys():

    rows = [
        r
        for r in year_rows
        if r["Group"] == name
    ]

    positive = sum(
        1
        for r in rows
        if r["Avg"] > 0
    )

    profitable_pf = sum(
        1
        for r in rows
        if r["PF"] > 1
    )

    avgs = [
        r["Avg"]
        for r in rows
        if not pd.isna(r["Avg"])
    ]

    print(
        f"{name:<30} "
        f"Positive Avg years {positive}/5 | "
        f"PF>1 years {profitable_pf}/5 | "
        f"Worst year {min(avgs):7.3f} | "
        f"Best year {max(avgs):7.3f}"
    )


# ==================================================================================================
# RECENT 2026 PERIOD
# ==================================================================================================

section(
    f"RECENT PERIOD: {RECENT_START} TO {RECENT_END}"
)

recent_rows = []

recent_start = pd.Timestamp(
    RECENT_START
)

recent_end = pd.Timestamp(
    RECENT_END
)


for name, subset in groups.items():

    r = subset[
        (subset["SignalTime"] >= recent_start) &
        (subset["SignalTime"] <= recent_end)
    ]

    s = print_stats(
        name,
        r
    )

    recent_rows.append(
        {
            "Period":
                f"{RECENT_START} to {RECENT_END}",

            "Group":
                name,

            **s
        }
    )


# ==================================================================================================
# 2026 MONTH-BY-MONTH
# ==================================================================================================

section("2026 MONTH-BY-MONTH")

month_rows = []

months_2026 = sorted(
    df[
        df["Year"] == 2026
    ]["Month"].unique()
)


for month in months_2026:

    print()
    print(
        f"--- {month} ---"
    )

    for name in [
        "BASELINE",
        "DOWN 2-3",
        "DOWN 2-3 + BB 2-4"
    ]:

        subset = groups[
            name
        ]

        m = subset[
            subset["Month"] == month
        ]

        s = print_stats(
            name,
            m
        )

        month_rows.append(
            {
                "Month": month,
                "Group": name,
                **s
            }
        )


# ==================================================================================================
# HALF-YEAR / RECENT WINDOWS
# ==================================================================================================

section("2026 SUBPERIODS")

windows = [
    (
        "2026 Jan-Mar",
        "2026-01-01",
        "2026-03-31 23:59:59"
    ),
    (
        "2026 Apr-Jun",
        "2026-04-01",
        "2026-06-30 23:59:59"
    ),
    (
        "2026 Jul-Sep18",
        "2026-07-01",
        "2026-09-18 23:59:59"
    ),
    (
        "2026 May-Sep18",
        "2026-05-01",
        "2026-09-18 23:59:59"
    )
]


window_rows = []


for label, start, end in windows:

    print()
    print(
        f"--- {label} ---"
    )

    start = pd.Timestamp(
        start
    )

    end = pd.Timestamp(
        end
    )

    for name in [
        "BASELINE",
        "DOWN 2-3",
        "DOWN 2-3 + BB 2-4"
    ]:

        subset = groups[
            name
        ]

        w = subset[
            (subset["SignalTime"] >= start) &
            (subset["SignalTime"] <= end)
        ]

        s = print_stats(
            name,
            w
        )

        window_rows.append(
            {
                "Period": label,
                "Group": name,
                **s
            }
        )


# ==================================================================================================
# CHECK BB CONTRIBUTION INSIDE DOWN 2-3
# ==================================================================================================

section("BB WIDTH INSIDE DOWN 2-3")

for bb_level in range(
    1,
    6
):

    subset = df[
        (df["TrendDirection"] == "DOWN") &
        (df["TrendCode"].isin([2, 3])) &
        (df["BBWidthCode"] == bb_level)
    ]

    print_stats(
        f"BB {bb_level}",
        subset
    )


# ==================================================================================================
# CHECK TREND LEVELS INSIDE BB 2-4
# ==================================================================================================

section("DOWN TREND LEVEL INSIDE BB 2-4")

for trend_level in range(
    1,
    6
):

    subset = df[
        (df["TrendDirection"] == "DOWN") &
        (df["TrendCode"] == trend_level) &
        (df["BBWidthCode"].isin([2, 3, 4]))
    ]

    print_stats(
        f"DOWN {trend_level}",
        subset
    )


# ==================================================================================================
# SAVE
# ==================================================================================================

all_rows = (
    overall_rows +
    year_rows +
    recent_rows +
    window_rows
)

results = pd.DataFrame(
    all_rows
)

results.to_csv(
    OUTPUT_PATH,
    index=False
)


monthly = pd.DataFrame(
    month_rows
)

monthly.to_csv(
    MONTH_OUTPUT_PATH,
    index=False
)


# ==================================================================================================
# DONE
# ==================================================================================================

section("DONE")

print(
    f"Robustness results:\n{OUTPUT_PATH}"
)

print()

print(
    f"Monthly results:\n{MONTH_OUTPUT_PATH}"
)