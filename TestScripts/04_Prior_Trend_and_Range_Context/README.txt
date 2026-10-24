================================================================================
04  PRIOR TREND AND RANGE CONTEXT                           (Research question 2)
================================================================================

PURPOSE
Test whether what happened BEFORE the signal (the preceding trend, or a
sideways consolidation) changes the information content of the cross.

CENTRAL FEATURE
PriorMoveATR = (Close[t-1] - Close[t-20]) / ATR14[t]    (15-min bars = 5 h)
Keeping -2 <= PriorMoveATR <= +2 (no extended move before the signal)
became "Filter B".

FILES
Prior move (trend before the signal)
  MACD_15M_PriorMove_Bucket_Test.py        Buckets of PriorMoveATR
  MACD_15M_PriceMove_PriorMove_Buckets.py  Joint Filter A x prior-move grid
  MACD_PriorMoveATR_Band_Test.py           Broad bands around centre and tails
  MACD_PriorMoveATR_Plateau_Test.py        Plateau test of the -2..+2 band
  MACD_Main_PriorMove_Lookback_Sweep.py    Sensitivity to the lookback length
  MACD_30M_PriorMoveATR_Test.py            Replication on 30-min bars (same 5 h)
  MACD_30M_PriorMoveATR_Tails.py           Three states: down tail / neutral /
                                           up tail
  MACD_PriceMove_NoRange_WithAgainst_Test.py  Signal with vs against the
                                           prior direction
  TREND_RAW_VS_PERCENTILE_TEST.py          Raw trend vs rolling-percentile trend
  MACD_NoTimeFilter_PriceMove_PriorMove_NoRange.py  Filters A+B with no time
                                           or range filters (clean ablation)

Sideways ranges and breakouts
  MACD_30M_NoRange_Test.py                 Excluding signals inside a range
  MACD_Range_Breakout_Strategy_Test.py     Cross that breaks out of a range
  MACD_Range_ExactBreakoutTiming_Test.py   Which bar after the range breaks?
  MACD_Range_WaitThenBreakout_Test.py      Wait N bars, then require breakout
  MACD_RecentRange_Breakout_Test.py        Range ending 1-4 bars before the
                                           cross; required breakout distance
  MACD_InsideRange_DelayedBreakout_Test.py Cross inside a range that breaks
                                           out later
  MACD_Combined_Main_Range_Strategy.py     Main strategy + range-continuation
                                           strategy; overlap analysis

INTERPRETATION
The MACD cross works best as a fresh impulse from a balanced state:
moderate momentum (Filter A) arriving after no extended trend (Filter B).
This is supported independently by the market-state analysis in folder 05,
where signals after a strong prior decline had PF below 1.
