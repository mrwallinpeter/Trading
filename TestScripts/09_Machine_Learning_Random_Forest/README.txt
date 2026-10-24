================================================================================
09  MACHINE LEARNING: RANDOM FOREST SIGNAL RANKING      (Research questions 2-3)
================================================================================

PURPOSE
Test whether a non-linear model can combine conditioning features better
than hand-set thresholds, under strict out-of-sample evaluation.

DESIGN
* Sample: every bullish MACD(5,36,5) cross on 15-min MES bars, 2022-2026.
* Target: 210-minute forward return (index points).
* Model: RandomForestRegressor (500 trees, max_depth 5,
  min_samples_leaf 25, max_features 2), deliberately shallow and heavily
  regularised.
* Validation: expanding-window walk-forward by year; only out-of-sample
  predictions are evaluated.
* Evaluation: OOS predictions are ranked into quintiles and deciles. A
  useful model should show realised returns and profit factor increasing
  from Q1 to Q5. Feature importances are saved.

INCREMENTAL FEATURE STUDY (one feature added per version)
  V0  PriceMoveATR + RSI                                 (baseline)
  V1  + PriorMoveATR
  V2  + ATR percentile (volatility regime)
  V3  + change in MACD
  V4  + distance from SMA50                     <- best version
  V5  + time of day
  V6  + 4-bar return
  V7  + candle body/range ratio

KEY RESULT (V4, pure out-of-sample)
  Quintile      N     Avg return (pts)   Profit factor   Win %
  Q1 lowest   1071        -0.00              1.00         50.4
  Q5 highest  1071        +1.88              1.30         55.8

Top-quintile PF across versions: V0 1.19, V2 1.24, V4 1.30. Adding
features after V4 did not improve on it. This is consistent with a low
signal-to-noise problem and supports a parsimonious model. The middle
quintiles and individual years are not perfectly monotonic, a reminder
that the predictive signal is weak.

NOTE
These scripts write their results to a sub-folder next to the script.
