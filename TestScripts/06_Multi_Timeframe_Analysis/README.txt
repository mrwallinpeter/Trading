================================================================================
06  MULTI-TIMEFRAME ANALYSIS                                (Research question 2)
================================================================================

PURPOSE
Practitioners often require agreement between timeframes. These scripts
test that belief quantitatively, and the choice between 15- and 30-minute
signal bars.

CAUSALITY
Higher-timeframe values are mapped onto lower-timeframe bars using only
the LAST COMPLETED higher-timeframe bar, so no future information leaks
into a signal. A "simulated 4h MACD" (EMA lengths scaled x16 on 15-min
bars: 80/576/80) is also used, giving a smooth, always-current
higher-timeframe state.

FILES
15m vs 30m
  MACD_15M_vs_30M_SameExit.py          Same exit rule, different signal
                                       timeframe
  MACD_15M_Setup_WaitFor30M_Cross.py   15m setup that waits for a 30m
                                       cross: signal retention vs quality
  MACD_30M_TwoFilters_5H_Test.py       Filters A+B moved to 30-min bars
  MACD_30M_TwoFilters_5H_TimeMatched.py  Same, with lookbacks matched in
                                       clock time (5 h)
  MACD_30M_PostCrossConfirmation.py    Require confirmation after the cross

4h context
  MACD_30M_4H_TrendContext.py          30m signals split by 4h trend state
  MACD_30M_4H_FourState.py             Four 4h states (MACD above/below 0 x
                                       histogram above/below 0); PF surfaces
                                       across holding periods vs matched
                                       random
  MACD_30M_DNEGNEG_vs_Random.py        The "both negative" 4h state vs
                                       random, year by year
  MACD_15M_Sim4H_LookbackSweep.py      Simulated-4h histogram direction,
                                       lookback sweep
  MACD_15M_TwoFilters_Sim4H_Direction.py  Filters A+B x 4h direction
  MACD_15M_TwoFilters_Sim4H_FourState.py  Filters A+B x four 4h states
  MACD_15M_POSUP_HistCrossAge.py       Bars since the 4h histogram crossed
                                       zero (the "age" of the larger trend)

KEY RESULT
4h-state effects were unstable. For example, signals in the fully
bullish 4h state (MACD > 0 and histogram > 0) ranked at the 92nd
percentile of matched random portfolios with a 30-minute hold, but only
at the 35th percentile with a 60-minute hold. No 4h state was
consistently better than random across holding periods, so the common
rule "only trade in the direction of the higher-timeframe trend" was not
supported for this signal.
