import numpy as np
import pandas as pd


# ==================================================================================================
# SETTINGS
# ==================================================================================================

INPUT_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\MARKET_STATE_SIGNALS.csv"

OUTPUT_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\MARKET_STATE_DOWN3_TEST.csv"

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


def print_stats(name, df):
    s = stats(df)

    print(
        f"{name:<28} "
        f"N {s['N']:4d} | "
        f"Avg {s['Avg']:7.3f} | "
        f"PF {s['PF']:6.3f} | "
        f"Win {s['WinPct']:6.2f}% | "
        f"Total {s['TotalPts']:9.2f}"
    )

    return s


# ==================================================================================================
# LOAD DATA
# ==================================================================================================

section("LOADING SIGNAL DATA")

df = pd.read_csv(INPUT_PATH)

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
    raise ValueError("SignalTime column not found.")

df["Year"] = df["SignalTime"].dt.year
df["Month"] = df["SignalTime"].dt.to_period("M").astype(str)

print(f"Signals loaded: {len(df):,}")
print(f"Start: {df['SignalTime'].min()}")
print(f"End:   {df['SignalTime'].max()}")


# ==================================================================================================
# DEFINE TEST GROUPS
# ==================================================================================================

baseline = df.copy()


# --------------------------------------------------------------------------------------------------
# DOWN 3
# --------------------------------------------------------------------------------------------------

down3 = df[
    (df["TrendDirection"] == "DOWN") &
    (df["TrendCode"] == 3)
].copy()


# --------------------------------------------------------------------------------------------------
# DOWN 3 + BB 1-4
# --------------------------------------------------------------------------------------------------

down3_bb14 = df[
    (df["TrendDirection"] == "DOWN") &
    (df["TrendCode"] == 3) &
    (df["BBWidthCode"].isin([1, 2, 3, 4]))
].copy()


# --------------------------------------------------------------------------------------------------
# DOWN 3 + BB 5
# --------------------------------------------------------------------------------------------------

down3_bb5 = df[
    (df["TrendDirection"] == "DOWN") &
    (df["TrendCode"] == 3) &
    (df["BBWidthCode"] == 5)
].copy()


# --------------------------------------------------------------------------------------------------
# DOWN 2
# --------------------------------------------------------------------------------------------------

down2 = df[
    (df["TrendDirection"] == "DOWN") &
    (df["TrendCode"] == 2)
].copy()


# --------------------------------------------------------------------------------------------------
# DOWN 2 + BB 1-4
# --------------------------------------------------------------------------------------------------

down2_bb14 = df[
    (df["TrendDirection"] == "DOWN") &
    (df["TrendCode"] == 2) &
    (df["BBWidthCode"].isin([1, 2, 3, 4]))
].copy()


# --------------------------------------------------------------------------------------------------
# DOWN 2-3 + BB 1-4
# --------------------------------------------------------------------------------------------------

down23_bb14 = df[
    (df["TrendDirection"] == "DOWN") &
    (df["TrendCode"].isin([2, 3])) &
    (df["BBWidthCode"].isin([1, 2, 3, 4]))
].copy()


groups = {
    "BASELINE": baseline,
    "DOWN 3": down3,
    "DOWN 3 + BB 1-4": down3_bb14,
    "DOWN 3 + BB 5": down3_bb5,
    "DOWN 2": down2,
    "DOWN 2 + BB 1-4": down2_bb14,
    "DOWN 2-3 + BB 1-4": down23_bb14
}


# ==================================================================================================
# OVERALL
# ==================================================================================================

section("OVERALL")

rows = []

for name, subset in groups.items():

    s = print_stats(
        name,
        subset
    )

    rows.append(
        {
            "Period": "ALL",
            "Group": name,
            **s
        }
    )


# ==================================================================================================
# YEAR BY YEAR
# ==================================================================================================

section("YEAR-BY-YEAR")

for year in range(2022, 2027):

    print()
    print(f"--- {year} ---")

    for name, subset in groups.items():

        y = subset[
            subset["Year"] == year
        ]

        s = print_stats(
            name,
            y
        )

        rows.append(
            {
                "Period": str(year),
                "Group": name,
                **s
            }
        )


# ==================================================================================================
# YEAR STABILITY
# ==================================================================================================

section("YEAR STABILITY")

for name in groups.keys():

    group_years = [
        r
        for r in rows
        if r["Group"] == name
        and r["Period"] in [
            "2022",
            "2023",
            "2024",
            "2025",
            "2026"
        ]
    ]

    positive_avg = sum(
        1
        for r in group_years
        if r["Avg"] > 0
    )

    positive_pf = sum(
        1
        for r in group_years
        if r["PF"] > 1
    )

    avgs = [
        r["Avg"]
        for r in group_years
        if not pd.isna(r["Avg"])
    ]

    print(
        f"{name:<28} "
        f"Positive Avg {positive_avg}/5 | "
        f"PF>1 {positive_pf}/5 | "
        f"Worst {min(avgs):7.3f} | "
        f"Best {max(avgs):7.3f}"
    )


# ==================================================================================================
# RECENT PERIOD
# ==================================================================================================

section("RECENT 2026 MAY-SEP18")

recent_start = pd.Timestamp(
    RECENT_START
)

recent_end = pd.Timestamp(
    RECENT_END
)

for name, subset in groups.items():

    recent = subset[
        (subset["SignalTime"] >= recent_start) &
        (subset["SignalTime"] <= recent_end)
    ]

    s = print_stats(
        name,
        recent
    )

    rows.append(
        {
            "Period": "2026 May-Sep18",
            "Group": name,
            **s
        }
    )


# ==================================================================================================
# 2026 SUBPERIODS
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
    )
]


for label, start, end in windows:

    print()
    print(f"--- {label} ---")

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    for name, subset in groups.items():

        w = subset[
            (subset["SignalTime"] >= start) &
            (subset["SignalTime"] <= end)
        ]

        s = print_stats(
            name,
            w
        )

        rows.append(
            {
                "Period": label,
                "Group": name,
                **s
            }
        )


# ==================================================================================================
# BB LEVELS INSIDE DOWN 3
# ==================================================================================================

section("BB LEVELS INSIDE DOWN 3")

for bb_level in range(1, 6):

    subset = df[
        (df["TrendDirection"] == "DOWN") &
        (df["TrendCode"] == 3) &
        (df["BBWidthCode"] == bb_level)
    ]

    print_stats(
        f"DOWN 3 + BB {bb_level}",
        subset
    )


# ==================================================================================================
# BB LEVELS INSIDE DOWN 2
# ==================================================================================================

section("BB LEVELS INSIDE DOWN 2")

for bb_level in range(1, 6):

    subset = df[
        (df["TrendDirection"] == "DOWN") &
        (df["TrendCode"] == 2) &
        (df["BBWidthCode"] == bb_level)
    ]

    print_stats(
        f"DOWN 2 + BB {bb_level}",
        subset
    )


# ==================================================================================================
# TREND 2 VS 3 DIRECT COMPARISON
# ==================================================================================================

section("DOWN 2 VS DOWN 3")

for bb_group_name, bb_mask in [
    (
        "ALL BB",
        pd.Series(True, index=df.index)
    ),
    (
        "BB 1-4",
        df["BBWidthCode"].isin([1, 2, 3, 4])
    ),
    (
        "BB 2-4",
        df["BBWidthCode"].isin([2, 3, 4])
    )
]:

    print()
    print(f"--- {bb_group_name} ---")

    for trend_level in [2, 3]:

        subset = df[
            (df["TrendDirection"] == "DOWN") &
            (df["TrendCode"] == trend_level) &
            bb_mask
        ]

        print_stats(
            f"DOWN {trend_level}",
            subset
        )


# ==================================================================================================
# SAVE
# ==================================================================================================

results = pd.DataFrame(
    rows
)

results.to_csv(
    OUTPUT_PATH,
    index=False
)


section("DONE")

print(
    f"Results saved:\n{OUTPUT_PATH}"
)