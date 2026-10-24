================================================================================
02  BASELINE SIGNAL VS MATCHED RANDOM BENCHMARK            (Research question 1)
================================================================================

PURPOSE
Test whether the raw MACD histogram zero-cross carries information about
future returns, against a null hypothesis of random entry timing.

DESIGN
1. Compute the forward return for EVERY eligible bar in the sample.
2. The MACD strategy is the subset of bars where a bullish cross occurred.
3. Draw 1,000 random subsets of the same size, matched on year and hour
   of day (actual signal bars are excluded from the random pool).
4. Report where the MACD result falls in the random distribution
   (percentile) for PF, average P&L, win rate, SQN and drawdown.
Because signal and random trades share exactly the same execution logic,
any difference is due to signal timing, not to the backtest mechanics.

FILES
RandomCrosses_vs_MACD_Crosses.py
    Core test on 15-minute bars, 240-minute holding period.
RandomCrosses_vs_MACD_Crosses_NoSideways.py
    Same test after removing signals that occur inside a sideways range
    (20-bar range height <= 25 points and drift <= 10 points).
MACD_30M_vs_Random_5H.py
    Replication on 30-minute bars with a 5-hour holding period.
MACD_30M_vs_Random_HoldSweep.py
    Repeats the comparison across holding periods, to check that the
    conclusion does not depend on one horizon.
MACD_30M_Path_vs_Random.py
    Compares the whole forward price path (MAE, MFE, return at several
    horizons) with the random paths, not just the end point.
MACD_30M_ParamSweep_vs_Random_FAST.py
    Sweeps MACD parameter sets (3/18/3 ... 8/36/5) against the null, to
    test sensitivity to parameter choice.
MACD_30M_LocalRobustness_vs_Random_FAST.py
    Local neighbourhood test around the chosen parameters: is the result
    a plateau or an isolated peak? Vectorised NumPy implementation.
MACD_PriceMoveATR_vs_Random.py
    Applies the same random benchmark to the filtered (PriceMoveATR)
    strategy, to measure the added value of the filter.

KEY RESULT
Raw signal with sideways setups removed (2021-2026): 1,716 trades,
PF 1.09, at about the 72nd percentile of matched random portfolios
(random median PF 1.05; 95th percentile PF 1.17).
The unconditional signal is NOT statistically distinguishable from random
timing. This motivates the conditional analysis in folders 03-07.
