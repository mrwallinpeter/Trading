import os
import json
import base64

import pandas as pd
import mplfinance as mpf

from openai import OpenAI


# ==================================================================================================
# SETTINGS
# ==================================================================================================

CSV_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\MES_SEP26_1MIN.csv"

CHART_PATH = r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\trend_regime_chart.png"

# Change these to the period you want to analyze
START = "2026-09-17 09:00"
END   = "2026-09-17 16:00"

TIMEFRAME = "15min"

BB_LENGTH = 20
BB_MULT = 2.0

MODEL = "gpt-5.6"


# ==================================================================================================
# LOAD DATA
# ==================================================================================================

print()
print("=" * 100)
print("LOADING DATA")
print("=" * 100)

df = pd.read_csv(CSV_PATH)

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
    if c in df.columns:
        time_col = c
        break

if time_col is None:
    raise ValueError("Could not find timestamp column.")

df[time_col] = pd.to_datetime(df[time_col])

df = (
    df
    .set_index(time_col)
    .sort_index()
)

rename_map = {}

for c in df.columns:

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

df = df.rename(columns=rename_map)

print(f"Rows:  {len(df):,}")
print(f"Start: {df.index.min()}")
print(f"End:   {df.index.max()}")


# ==================================================================================================
# RESAMPLE TO 15-MINUTE BARS
# ==================================================================================================

print()
print("=" * 100)
print("BUILDING 15-MINUTE BARS")
print("=" * 100)

agg = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last"
}

if "Volume" in df.columns:
    agg["Volume"] = "sum"

df15 = df.resample(
    TIMEFRAME,
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

print(f"15m bars: {len(df15):,}")


# ==================================================================================================
# BOLLINGER BANDS
# ==================================================================================================

basis = df15["Close"].rolling(
    BB_LENGTH
).mean()

std = df15["Close"].rolling(
    BB_LENGTH
).std(
    ddof=0
)

df15["BB_Mid"] = basis

df15["BB_Upper"] = (
    basis +
    BB_MULT * std
)

df15["BB_Lower"] = (
    basis -
    BB_MULT * std
)

df15["BB_Width"] = (
    df15["BB_Upper"] -
    df15["BB_Lower"]
)


# ==================================================================================================
# SELECT PERIOD
# ==================================================================================================

chart = df15.loc[
    START:END
].copy()

chart = chart.dropna(
    subset=[
        "BB_Mid",
        "BB_Upper",
        "BB_Lower"
    ]
)

if len(chart) == 0:
    raise ValueError(
        "No data found in selected period."
    )

print()
print("=" * 100)
print("CHART PERIOD")
print("=" * 100)

print(f"Start: {chart.index.min()}")
print(f"End:   {chart.index.max()}")
print(f"Bars:  {len(chart)}")


# ==================================================================================================
# CREATE CHART
# ==================================================================================================

print()
print("=" * 100)
print("CREATING CHART")
print("=" * 100)

additional_plots = [

    mpf.make_addplot(
        chart["BB_Mid"],
        width=1.0
    ),

    mpf.make_addplot(
        chart["BB_Upper"],
        width=1.0
    ),

    mpf.make_addplot(
        chart["BB_Lower"],
        width=1.0
    )
]

mpf.plot(
    chart,
    type="candle",
    volume="Volume" in chart.columns,
    addplot=additional_plots,
    style="yahoo",
    figsize=(15, 9),
    title="MES 15-minute chart with Bollinger Bands",
    tight_layout=True,
    savefig=dict(
        fname=CHART_PATH,
        dpi=160,
        bbox_inches="tight"
    )
)

print(f"Saved chart: {CHART_PATH}")


# ==================================================================================================
# ENCODE IMAGE
# ==================================================================================================

with open(
    CHART_PATH,
    "rb"
) as f:

    image_base64 = base64.b64encode(
        f.read()
    ).decode("utf-8")


# ==================================================================================================
# OPENAI CLIENT
# ==================================================================================================

api_key = os.environ[
    "OPENAI_API_KEY"
].strip()

client = OpenAI(
    api_key=api_key
)


# ==================================================================================================
# PROMPT
# ==================================================================================================

prompt = """
You are analyzing a 15-minute MES futures chart.

The chart contains candlesticks and Bollinger Bands:
- upper band
- middle band
- lower band

Your job is NOT to predict whether a trade will win.

Your job is only to describe the observable market structure.

Analyze the full visible chart, with somewhat greater emphasis on the most recent third of the chart.

Return ONLY valid JSON.

Use exactly this structure:

{
  "trend_direction": "up | down | sideways",
  "trend_strength": "weak | medium | strong",
  "trend_character": "steady | volatile | mixed",
  "bollinger_width_state": "contracting | stable | expanding | mixed",
  "band_behavior": "riding_upper_band | riding_lower_band | bouncing_between_bands | center_band | mixed",
  "trend_score": 0,
  "volatility_score": 0,
  "steadiness_score": 0,
  "explanation": ""
}

Definitions:

trend_direction:
- up = market has a clearly upward directional structure
- down = market has a clearly downward directional structure
- sideways = no persistent directional structure

trend_strength:
- weak = slight directional bias but frequent failure or reversal
- medium = clearly directional but with meaningful pullbacks
- strong = persistent directional movement with clear structure

trend_character:
- steady = price progresses consistently in one direction and Bollinger width changes gradually
- volatile = large swings, abrupt expansion/contraction, irregular movement
- mixed = neither clearly steady nor clearly volatile

bollinger_width_state:
- contracting = bands are visibly narrowing
- stable = band width stays relatively similar through the analyzed region
- expanding = bands are visibly widening
- mixed = width changes substantially between contraction and expansion

band_behavior:
- riding_upper_band = price repeatedly stays close to or touches the upper Bollinger band while trending upward
- riding_lower_band = price repeatedly stays close to or touches the lower Bollinger band while trending downward
- bouncing_between_bands = price repeatedly travels from one side of the Bollinger structure to the other
- center_band = price spends much of its time near the Bollinger middle line
- mixed = no single behavior clearly dominates

Scores:
trend_score:
0 = no trend
10 = extremely strong trend

volatility_score:
0 = extremely quiet
10 = extremely volatile

steadiness_score:
0 = extremely erratic
10 = extremely smooth/persistent

The explanation should be 2-4 concise sentences and describe WHY you chose the classifications.

Do not mention future price action.
Do not provide trading advice.
Do not predict profit or loss.
"""


# ==================================================================================================
# SEND CHART TO MODEL
# ==================================================================================================

print()
print("=" * 100)
print("SENDING CHART TO LLM")
print("=" * 100)

response = client.responses.create(

    model=MODEL,

    input=[
        {
            "role": "user",

            "content": [

                {
                    "type": "input_text",
                    "text": prompt
                },

                {
                    "type": "input_image",
                    "image_url":
                        f"data:image/png;base64,{image_base64}"
                }
            ]
        }
    ]
)


# ==================================================================================================
# PARSE RESPONSE
# ==================================================================================================

raw_text = response.output_text.strip()

print()
print("=" * 100)
print("RAW LLM RESPONSE")
print("=" * 100)

print(raw_text)


# Remove accidental markdown fences if model ever adds them
clean_text = raw_text

clean_text = clean_text.replace(
    "```json",
    ""
)

clean_text = clean_text.replace(
    "```",
    ""
)

clean_text = clean_text.strip()


try:

    result = json.loads(
        clean_text
    )

except json.JSONDecodeError:

    print()
    print("ERROR: Response was not valid JSON.")

    raise


# ==================================================================================================
# PRETTY RESULT
# ==================================================================================================

print()
print("=" * 100)
print("MARKET STRUCTURE")
print("=" * 100)

print(
    f"Trend direction:       "
    f"{result.get('trend_direction')}"
)

print(
    f"Trend strength:        "
    f"{result.get('trend_strength')}"
)

print(
    f"Trend character:       "
    f"{result.get('trend_character')}"
)

print(
    f"Bollinger width:       "
    f"{result.get('bollinger_width_state')}"
)

print(
    f"Band behavior:         "
    f"{result.get('band_behavior')}"
)

print(
    f"Trend score:           "
    f"{result.get('trend_score')}/10"
)

print(
    f"Volatility score:      "
    f"{result.get('volatility_score')}/10"
)

print(
    f"Steadiness score:      "
    f"{result.get('steadiness_score')}/10"
)

print()
print("Explanation:")

print(
    result.get(
        "explanation"
    )
)


# ==================================================================================================
# SAVE RESULT
# ==================================================================================================

result_df = pd.DataFrame(
    [
        {
            "Start": START,
            "End": END,

            "TrendDirection":
                result.get(
                    "trend_direction"
                ),

            "TrendStrength":
                result.get(
                    "trend_strength"
                ),

            "TrendCharacter":
                result.get(
                    "trend_character"
                ),

            "BBWidthState":
                result.get(
                    "bollinger_width_state"
                ),

            "BandBehavior":
                result.get(
                    "band_behavior"
                ),

            "TrendScore":
                result.get(
                    "trend_score"
                ),

            "VolatilityScore":
                result.get(
                    "volatility_score"
                ),

            "SteadinessScore":
                result.get(
                    "steadiness_score"
                ),

            "Explanation":
                result.get(
                    "explanation"
                )
        }
    ]
)

CSV_OUTPUT = (
    r"C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts"
    r"\LLM_MARKET_STRUCTURE_RESULT.csv"
)

result_df.to_csv(
    CSV_OUTPUT,
    index=False
)

print()
print("=" * 100)
print("DONE")
print("=" * 100)

print(
    f"Result saved to: "
    f"{CSV_OUTPUT}"
)