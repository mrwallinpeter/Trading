================================================================================
07  SESSION TIMING, HOLDING PERIOD AND WALK-FORWARD VALIDATION
                                                       (Research questions 2-3)
================================================================================

PURPOSE
Measure how the edge depends on time of day and holding period, and
validate the selection out-of-sample.

FILES
  MACD_TimeBlock_HoldingPeriod_Test.py   PF / average P&L / profitable-year
                     matrices: time block x holding period (60-240 min)
  MACD_FinalSetup_TimeBlock_Test.py      Ranking of time blocks for the
                     final setup
  MACD_0614_vs_1418_Diagnostic_Study.py  Why do 06-14 and 14-18 signals
                     differ? Feature distributions, winners vs losers and
                     feature-P&L correlations per time block
  MES_ORIGINAL_TWO_FILTERS_90MIN_TIMEWINDOWS.py  Filters A+B with a short
                     90-min hold over many time windows
  MACD_30M_06_22_HoldSweep_30Min.py      Holding-period sweep in 30-min steps,
                     ranked by percentile vs matched random
  MACD_30M_HistogramDeclineExit.py       Dynamic exit (histogram turns down)
                     vs fixed holding time
  MACD_Monthly_Seasonality_Test.py       Month-of-year effects
  MACD_HoldingPeriod_WalkForward.py      WALK-FORWARD VALIDATION:
                     the holding period is chosen on past data only
                     (train 2022 -> test 2023, train 2022-23 -> test 2024,
                     train 2022-24 -> test 2025, train 2022-25 -> test 2026)
                     and evaluated on the following, unseen year

KEY RESULT (pure out-of-sample, 2023-2026; times are exchange time)
  Time block   OOS trades   Win %   Profit factor   Profitable years
  06-10            51        56.9       1.28             3 / 4
  10-14            49        59.2       0.98             2 / 4
  14-18           104        62.5       2.14             4 / 4
  18-22           102        59.8       1.89             4 / 4
  14-22           208        60.1       1.94             4 / 4

Afternoon and evening signals kept their edge out-of-sample, and the
selected holding period was stable (about 180-210 min).
Caveat: the time blocks themselves were defined after exploratory
analysis, so this is evidence of stability, not a fully independent test.
