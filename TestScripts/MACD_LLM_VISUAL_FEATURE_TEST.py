import os
import json
import time
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import mplfinance as mpf

from openai import OpenAI


# ==================================================================================================
# SETTINGS
# ==================================================================================================

CSV_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"

OUTPUT_DIR = Path(
    r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\LLM_IMAGE_VS_JSON"
)

RESULTS_CSV = OUTPUT_DIR / "IMAGE_VS_JSON_RESULTS.csv"
CHART_DIR = OUTPUT_DIR / "charts"

MAX_SIGNALS = 50

LOOKBACK_BARS = 50

MACD_FAST = 5
MACD_SLOW = 36
MACD_SIGNAL = 5

BB_LENGTH = 20
BB_MULT = 2.0

ATR_LENGTH = 14

START_YEAR = 2022
END_YEAR = 2026

MODEL = "gpt-5.6"

MAX_RETRIES = 4
API_DELAY_SECONDS = 0.3


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================================================================
# OPENAI
# ==================================================================================================

api_key = os.environ["OPENAI_API_KEY"].strip()

client = OpenAI(api_key=api_key)


# ==================================================================================================
# HELPERS
# ==================================================================================================

def section(title):
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def clean_json_response(text):
    return (
        text
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_llm(content):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=MODEL,
                input=[
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )

            text = clean_json_response(
                response.output_text
            )

            return json.loads(text)

        except Exception as e:
            last_error = e

            print(
                f"Attempt {attempt}/{MAX_RETRIES} failed: {e}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)

    raise RuntimeError(last_error)


# ==================================================================================================
# COMMON CLASSIFICATION PROMPT
# ==================================================================================================

CLASSIFICATION_TEXT = """
Classify the current market structure.

The final observation is the MACD signal bar.
Do not predict future price.
Do not provide trading advice.

Use the full historical context, but place greater emphasis on approximately the final 15-20 bars when judging the CURRENT state.

Return ONLY valid JSON:

{
  "trend_direction": "up | down | sideways",
  "trend_strength": "weak | medium | strong",
  "trend_character": "steady | volatile | mixed",
  "bollinger_width_state": "contracting | stable | expanding | mixed",
  "band_behavior": "riding_upper_band | riding_lower_band | bouncing_between_bands | center_band | mixed",
  "trend_score": 0,
  "volatility_score": 0,
  "steadiness_score": 0,
  "confidence_score": 0
}

Definitions:

trend_direction:
up = persistent upward structure
down = persistent downward structure
sideways = no clear directional structure

trend_strength:
weak = mild directional bias
medium = clear direction with meaningful pullbacks
strong = persistent directional progression

trend_character:
steady = smooth, persistent directional movement
volatile = abrupt and irregular movement with large swings
mixed = neither clearly steady nor clearly volatile

bollinger_width_state:
contracting = bands narrowing in the recent region
stable = width remains relatively similar
expanding = bands widening in the recent region
mixed = alternating expansion and contraction

band_behavior:
riding_upper_band = price repeatedly stays near the upper Bollinger band
riding_lower_band = price repeatedly stays near the lower Bollinger band
bouncing_between_bands = price repeatedly moves from one side of the band structure to the other
center_band = price spends most time around the middle band
mixed = no dominant behavior

Scores must be integers from 0 to 10.
"""


# ==================================================================================================
# LOAD DATA
# ==================================================================================================

section("LOADING DATA")

df1 = pd.read_csv(CSV_PATH)

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
    raise ValueError("Could not find timestamp column.")

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

df1 = df1.rename(columns=rename_map)


# ==================================================================================================
# BUILD 15M
# ==================================================================================================

section("BUILDING 15-MINUTE DATA")

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
    subset=[
        "Open",
        "High",
        "Low",
        "Close"
    ]
)


# ==================================================================================================
# MACD
# ==================================================================================================

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

df15["MACDHist"] = (
    macd -
    signal_line
)

df15["BullCross"] = (
    (df15["MACDHist"].shift(1) <= 0) &
    (df15["MACDHist"] > 0)
)


# ==================================================================================================
# ATR
# ==================================================================================================

prev_close = df15["Close"].shift(1)

tr = pd.concat(
    [
        df15["High"] - df15["Low"],
        (df15["High"] - prev_close).abs(),
        (df15["Low"] - prev_close).abs()
    ],
    axis=1
).max(axis=1)

df15["ATR"] = tr.ewm(
    alpha=1 / ATR_LENGTH,
    adjust=False
).mean()


# ==================================================================================================
# BOLLINGER
# ==================================================================================================

df15["BB_Mid"] = (
    df15["Close"]
    .rolling(BB_LENGTH)
    .mean()
)

std = (
    df15["Close"]
    .rolling(BB_LENGTH)
    .std(ddof=0)
)

df15["BB_Upper"] = (
    df15["BB_Mid"] +
    BB_MULT * std
)

df15["BB_Lower"] = (
    df15["BB_Mid"] -
    BB_MULT * std
)

df15["BB_Width"] = (
    df15["BB_Upper"] -
    df15["BB_Lower"]
)


# ==================================================================================================
# SIGNAL UNIVERSE
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
        "ATR",
        "BB_Mid",
        "BB_Upper",
        "BB_Lower"
    ]
)

print(
    f"Signals available: {len(signals):,}"
)


# ==================================================================================================
# IMAGE CLASSIFICATION
# ==================================================================================================

def analyze_image(chart, chart_path):

    additional_plots = [
        mpf.make_addplot(
            chart["BB_Mid"],
            width=1
        ),
        mpf.make_addplot(
            chart["BB_Upper"],
            width=1
        ),
        mpf.make_addplot(
            chart["BB_Lower"],
            width=1
        )
    ]

    mpf.plot(
        chart,
        type="candle",
        volume="Volume" in chart.columns,
        addplot=additional_plots,
        style="yahoo",
        figsize=(9, 5),
        title="MES 15-minute market structure",
        tight_layout=True,
        savefig=dict(
            fname=str(chart_path),
            dpi=90,
            bbox_inches="tight"
        )
    )

    img = encode_image(
        chart_path
    )

    content = [
        {
            "type": "input_text",
            "text":
                "You are viewing a 15-minute MES chart with Bollinger Bands.\n\n"
                + CLASSIFICATION_TEXT
        },
        {
            "type": "input_image",
            "image_url":
                f"data:image/png;base64,{img}"
        }
    ]

    return call_llm(
        content
    )


# ==================================================================================================
# JSON CLASSIFICATION
# ==================================================================================================

def build_numeric_payload(chart):

    c = chart.copy()

    atr = c["ATR"].replace(
        0,
        np.nan
    )

    bb_width = (
        c["BB_Upper"] -
        c["BB_Lower"]
    ).replace(
        0,
        np.nan
    )

    features = pd.DataFrame(
        index=c.index
    )

    # bar-to-bar price movement normalized by ATR
    features["return_atr"] = (
        c["Close"].diff() /
        atr
    )

    # candle range normalized by ATR
    features["range_atr"] = (
        (c["High"] - c["Low"]) /
        atr
    )

    # candle body normalized by ATR
    features["body_atr"] = (
        (c["Close"] - c["Open"]) /
        atr
    )

    # Bollinger position:
    # 0 = lower band
    # 0.5 = middle
    # 1 = upper band
    features["bb_position"] = (
        (c["Close"] - c["BB_Lower"]) /
        bb_width
    )

    # total BB width normalized by ATR
    features["bb_width_atr"] = (
        bb_width /
        atr
    )

    # close relative to middle band
    features["close_vs_mid_atr"] = (
        (c["Close"] - c["BB_Mid"]) /
        atr
    )

    # BB middle-band slope
    features["bb_mid_slope_atr"] = (
        c["BB_Mid"].diff() /
        atr
    )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # round aggressively to keep request compact
    result = {}

    for column in features.columns:

        values = []

        for x in features[column]:

            if pd.isna(x):
                values.append(None)
            else:
                values.append(
                    round(float(x), 3)
                )

        result[column] = values

    return result


def analyze_json(chart):

    payload = build_numeric_payload(
        chart
    )

    explanation = """
The numerical arrays are chronological from oldest to newest.

Each array contains the same bars.

bb_position:
0.0 = lower Bollinger band
0.5 = Bollinger middle
1.0 = upper Bollinger band
values below 0 are below the lower band
values above 1 are above the upper band

return_atr:
close-to-close movement divided by ATR

range_atr:
high-low range divided by ATR

body_atr:
close-open candle body divided by ATR

bb_width_atr:
total Bollinger width divided by ATR

close_vs_mid_atr:
distance from Bollinger middle divided by ATR

bb_mid_slope_atr:
change in Bollinger middle line divided by ATR

Infer the same visual concepts that a trader would infer from a chart.
Do not impose fixed thresholds.
Judge the overall pattern and the recent market state.
"""

    text = (
        explanation
        + "\n\n"
        + CLASSIFICATION_TEXT
        + "\n\nDATA:\n"
        + json.dumps(
            payload,
            separators=(",", ":")
        )
    )

    content = [
        {
            "type": "input_text",
            "text": text
        }
    ]

    return call_llm(
        content
    )


# ==================================================================================================
# PROCESS SIGNALS
# ==================================================================================================

section("COMPARING IMAGE VS JSON")

rows = []

processed = 0

for signal_number, signal_time in enumerate(
    signals.index,
    start=1
):

    signal_pos = df15.index.get_loc(
        signal_time
    )

    start_pos = (
        signal_pos -
        LOOKBACK_BARS +
        1
    )

    if start_pos < 0:
        continue

    chart = df15.iloc[
        start_pos:
        signal_pos + 1
    ].copy()

    chart = chart.dropna(
        subset=[
            "ATR",
            "BB_Mid",
            "BB_Upper",
            "BB_Lower"
        ]
    )

    if len(chart) < LOOKBACK_BARS:
        continue

    processed += 1

    safe_time = signal_time.strftime(
        "%Y%m%d_%H%M"
    )

    chart_path = (
        CHART_DIR /
        f"MES_{safe_time}.png"
    )

    print()
    print(
        f"{processed}/{MAX_SIGNALS} | "
        f"{signal_time}"
    )


    # ------------------------------------------------------------------------------------------------
    # IMAGE METHOD
    # ------------------------------------------------------------------------------------------------

    image_result = analyze_image(
        chart,
        chart_path
    )


    # ------------------------------------------------------------------------------------------------
    # JSON METHOD
    # ------------------------------------------------------------------------------------------------

    json_result = analyze_json(
        chart
    )


    # ------------------------------------------------------------------------------------------------
    # COMPARE
    # ------------------------------------------------------------------------------------------------

    fields = [
        "trend_direction",
        "trend_strength",
        "trend_character",
        "bollinger_width_state",
        "band_behavior"
    ]

    agreement = {}

    for field in fields:

        agreement[field] = int(
            image_result.get(field) ==
            json_result.get(field)
        )


    categorical_agreement = (
        sum(agreement.values()) /
        len(fields)
    )


    # score differences
    trend_score_diff = abs(
        float(image_result.get("trend_score", np.nan)) -
        float(json_result.get("trend_score", np.nan))
    )

    volatility_score_diff = abs(
        float(image_result.get("volatility_score", np.nan)) -
        float(json_result.get("volatility_score", np.nan))
    )

    steadiness_score_diff = abs(
        float(image_result.get("steadiness_score", np.nan)) -
        float(json_result.get("steadiness_score", np.nan))
    )


    row = {
        "SignalTime":
            signal_time,

        # image
        "ImageTrendDirection":
            image_result.get("trend_direction"),

        "ImageTrendStrength":
            image_result.get("trend_strength"),

        "ImageTrendCharacter":
            image_result.get("trend_character"),

        "ImageBBWidthState":
            image_result.get("bollinger_width_state"),

        "ImageBandBehavior":
            image_result.get("band_behavior"),

        "ImageTrendScore":
            image_result.get("trend_score"),

        "ImageVolatilityScore":
            image_result.get("volatility_score"),

        "ImageSteadinessScore":
            image_result.get("steadiness_score"),

        "ImageConfidence":
            image_result.get("confidence_score"),

        # json
        "JSONTrendDirection":
            json_result.get("trend_direction"),

        "JSONTrendStrength":
            json_result.get("trend_strength"),

        "JSONTrendCharacter":
            json_result.get("trend_character"),

        "JSONBBWidthState":
            json_result.get("bollinger_width_state"),

        "JSONBandBehavior":
            json_result.get("band_behavior"),

        "JSONTrendScore":
            json_result.get("trend_score"),

        "JSONVolatilityScore":
            json_result.get("volatility_score"),

        "JSONSteadinessScore":
            json_result.get("steadiness_score"),

        "JSONConfidence":
            json_result.get("confidence_score"),

        # agreements
        "AgreeTrendDirection":
            agreement["trend_direction"],

        "AgreeTrendStrength":
            agreement["trend_strength"],

        "AgreeTrendCharacter":
            agreement["trend_character"],

        "AgreeBBWidthState":
            agreement["bollinger_width_state"],

        "AgreeBandBehavior":
            agreement["band_behavior"],

        "CategoricalAgreement":
            categorical_agreement,

        "TrendScoreDiff":
            trend_score_diff,

        "VolatilityScoreDiff":
            volatility_score_diff,

        "SteadinessScoreDiff":
            steadiness_score_diff
    }

    rows.append(
        row
    )


    # save after every signal
    pd.DataFrame(
        rows
    ).to_csv(
        RESULTS_CSV,
        index=False
    )


    print(
        "IMAGE:",
        image_result.get("trend_direction"),
        image_result.get("trend_strength"),
        image_result.get("trend_character"),
        image_result.get("bollinger_width_state"),
        image_result.get("band_behavior")
    )

    print(
        "JSON: ",
        json_result.get("trend_direction"),
        json_result.get("trend_strength"),
        json_result.get("trend_character"),
        json_result.get("bollinger_width_state"),
        json_result.get("band_behavior")
    )

    print(
        f"Categorical agreement: "
        f"{categorical_agreement:.0%}"
    )


    if processed >= MAX_SIGNALS:
        break


    time.sleep(
        API_DELAY_SECONDS
    )


# ==================================================================================================
# FINAL COMPARISON
# ==================================================================================================

section("FINAL COMPARISON")

result = pd.DataFrame(
    rows
)

print(
    f"Signals compared: {len(result)}"
)

print()

comparison_cols = [
    "AgreeTrendDirection",
    "AgreeTrendStrength",
    "AgreeTrendCharacter",
    "AgreeBBWidthState",
    "AgreeBandBehavior"
]

for col in comparison_cols:

    pct = (
        result[col].mean() *
        100
    )

    print(
        f"{col:<28} {pct:6.1f}%"
    )


print()

print(
    "Overall categorical agreement: "
    f"{result['CategoricalAgreement'].mean() * 100:.1f}%"
)

print()

print(
    "Mean absolute TrendScore difference: "
    f"{result['TrendScoreDiff'].mean():.2f}"
)

print(
    "Mean absolute VolatilityScore difference: "
    f"{result['VolatilityScoreDiff'].mean():.2f}"
)

print(
    "Mean absolute SteadinessScore difference: "
    f"{result['SteadinessScoreDiff'].mean():.2f}"
)


print()
print(
    f"Results saved to:"
)
print(
    RESULTS_CSV
)