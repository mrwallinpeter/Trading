================================================================================
03  ENTRY MOMENTUM FEATURES                                 (Research question 2)
================================================================================

PURPOSE
Study whether the STRENGTH of the impulse that creates the MACD cross
predicts the outcome. Features are normalised by ATR(14), so they are
comparable across volatility regimes and years.

CENTRAL FEATURE
PriceMoveATR = (Close[t] - Close[t-1]) / ATR14[t]
A moderate impulse (0.35-0.90 ATR) performed better than a weak one (no
real momentum) or a very strong one (exhaustion). This band became
"Filter A" in the rest of the project.

FILES
Price-move definition and robustness
  MACD_PRICEMOVE_PLATEAU_TEST.py          Is the 0.35-0.90 band a plateau?
  MACD_PRICEMOVE_BODY_INTERSECTION.py     Close-to-close vs candle-body move
  MACD_BODYMOVE_PLATEAU_TEST.py           Plateau test of the body version
  PRICEMOVE_VARIANTS_TEST.py              Alternative definitions, 10 buckets
  PRICEMOVE_RAW_VS_PERCENTILE_TEST.py     Raw ATR units vs rolling percentile
  MACD_PriceMoveATR_NoRange_Test.py       Price-move filter without the
                                          range filter
  MACD_LongShort_PriceMoveATR_Test.py     Symmetry: longs vs shorts
  MACD_Short_PriceMoveATR_Plateau_Test.py Plateau test for short signals

Candle and histogram shape
  MACD_GREEN_CANDLE_TEST.py               Must the cross bar be a green candle?
  MACD_CROSS_CANDLE_COMPARE.py            Old vs new cross-candle filters
  MACD_Histogram_Height_Increase_Test.py  Size of the histogram jump
  MACD_15M_Cross_PreCrossUpBars.py        Consecutive rising histogram bars
                                          before the cross
  MACD_15M_Cross_PreCrossUpBars_Grouped.py  Same, grouped 0 / 1-2 / 3-4 / 5+
  MACD_15M_PriceMove_PriorMove_HistUpBars.py  Histogram build-up combined
                                          with Filters A and B
  MACD_FirstCandle_Context_Study.py       Close location, histogram slope and
                                          ATR expansion on the signal bar

Multi-bar momentum
  MACD_TwoCandle_CombinedMomentum_Test.py Combined move over two bars
  MACD_TwoStage_Momentum_Entry.py         Wait for confirmation from the
                                          next bar
  MACD_Momentum_Strength_Study.py         Quintile analysis of all momentum
                                          features
  MACD_Momentum_Range_Test.py             Year-by-year stability of momentum
                                          filters

Comparison with other oscillators
  MACD_General_Indicator_Comparison.py    RSI, ATR-based and other indicators
                                          ranked as conditioning variables
  MACD_NOPRICEMOVE_RSI_BUCKET_TEST.py     RSI buckets without the price filter
  MACD_STRATEGY_RSI_BUCKET_TEST.py        RSI buckets within the strategy
  MACD_PRICEMOVE_RSI_OVERLAP_TEST.py      Do PriceMove and RSI capture the
                                          same information?
  MACD_PRICEMOVE_VS_RSI40_50.py           Head-to-head: PriceMove vs RSI 40-50
  PRICEMOVE_RSI_COMBINATION_TEST.py       Joint PriceMove x RSI grid

METHOD NOTE
Every filter is judged on (i) profit factor and average P&L, (ii) the
number of profitable years, and (iii) whether neighbouring parameter values
give similar results. Single-value optima are treated as likely overfitting.

NOTE
Several scripts in this folder write their results to a sub-folder next to
the script; the others write to the central results directory.
