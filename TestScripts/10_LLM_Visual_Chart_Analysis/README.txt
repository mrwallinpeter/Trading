================================================================================
10  MULTIMODAL LLM AS A MARKET-STATE FEATURE EXTRACTOR          (Exploratory)
================================================================================

PURPOSE
Discretionary traders read charts visually. Can a multimodal large
language model reproduce such qualitative judgements consistently?

FILES
MACD_LLM_VISUAL_FEATURE_TEST.py
    For 50 MACD signals, renders a candlestick chart with Bollinger bands
    (mplfinance) and sends it to an OpenAI vision model with a fixed
    classification prompt: trend direction, strength and character,
    BB-width state, band behaviour, numeric scores and confidence. The
    SAME prompt is also given the underlying numbers as JSON. The study
    measures IMAGE-vs-JSON agreement, i.e. whether the model's judgement
    is consistent across input modalities.

MACD_BBWIDTH_EXPANSION_WAIT2_TEST.py
    (Legacy file name.) Single-chart market-structure classifier: renders a
    chosen intraday period and asks the model to describe the regime. Used
    to develop and debug the classification prompt.

REQUIREMENTS
pip install openai mplfinance ; environment variable OPENAI_API_KEY.

RESULT (50 signals, agreement between image and JSON input)
  Trend direction      88 %
  Trend character      74 %
  Band behaviour       74 %
  Trend strength       68 %
  BB-width state       52 %

The model is fairly consistent on coarse labels (trend direction), but
much less consistent on finer judgements, and close to chance on
volatility state. LLM-derived labels might be usable as coarse features,
but they need validation before entering a quantitative model. The sample
is small and the study is exploratory.
