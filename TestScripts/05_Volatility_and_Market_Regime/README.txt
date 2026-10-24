================================================================================
05  VOLATILITY AND MARKET REGIME                            (Research question 2)
================================================================================

PURPOSE
Determine whether the signal's edge depends on the volatility regime and
on a broader classification of market state.

FILES
ATR-based volatility
  MACD_15M_ATR_PERCENTILE_EXCLUSION_TEST.py  Excluding the lowest ATR
                     percentiles; 5-percentile buckets; robustness to the
                     percentile lookback (50-400 bars)
  MACD_ATR14_Absolute_Level_Study.py   Absolute ATR level (points) vs
                     quintiles
  MACD_ATR14_LowRange_Study.py         Very-low-volatility environments
  MACD_Final_ATR10_vs_ATR14.py         Sensitivity to the ATR length
  MACD_PriceMove_ATR_Length_Test.py    Filter A computed with different ATR
                     lengths; broad percentile bands

Bollinger-band width (volatility compression / expansion)
  MACD_2FILTER_BBWIDTH_SMA200_TEST.py  BB width relative to its 200-bar
                     average as a third filter

Market activity
  MACD_PRICEMOVE_ACTIVITY_TEST.py      Activity diagnostics on the price-move
                     strategy
  MACD_STRATEGY_ACTIVITY_EXACT_TEST.py Exact activity variants on the final
                     strategy, aligned with the Pine Script version

Market-state taxonomy
  MARKET_STATE_CATEGORY_TEST.py        Classifies every signal by trend
                     block, ATR rank, BB width, volatility variation and
                     volume rank
  MARKET_STATE_BROAD_REGIME_TEST.py    Collapses states into broad regimes
  MARKET_STATE_DOWN3_TEST.py           Moderate-downtrend states combined
                     with BB-width buckets
  MARKET_STATE_LONG_SHORT_TEST.py      Same states for long vs short signals
  MARKET_STATE_ROBUSTNESS_TEST.py      Candidate states re-tested year by
                     year, on a recent hold-out period and by quarter

KEY RESULT (market-state robustness test, all MACD crosses, 210-min hold)
  State                                   Signals    Profit factor
  All signals (baseline)                    6,813        1.06
  Moderate prior decline + mid BB width       897        1.28
  Strong prior decline                      1,110        0.95

The moderate-decline / mid-BB-width state had PF above 1 in 4 of 5 years
(2022-2026; the exception was 2023 with PF 0.96). In the most recent
period (May-Sep 2026) it reached PF 2.24 on 78 signals; this sample is
small. Signals after a strong decline were weaker than baseline,
consistent with Filter B (folder 04).

NOTE
Two scripts in this folder write their results to a sub-folder next to
the script; the others write to the central results directory.
