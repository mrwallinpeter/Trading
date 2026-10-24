================================================================================
01  DATA VALIDATION AND MARKET CHARACTERISATION
================================================================================

PURPOSE
Make sure the research infrastructure is correct before any hypothesis is
tested, and describe the basic intraday structure of the market.

FILES
ATR14_Intraday_Cycle.py
    Builds 15-minute bars from 1-minute MES data and computes Wilder's
    ATR(14). For each trading day it finds when volatility peaks and
    troughs, and aggregates these times into 15-minute and hourly
    distributions, year by year (2022-2026). It answers when the market
    is "awake", which later motivates the time-of-day analysis (folder 07).

MACD_15M_Pine_Count_Validation.py
    Independent re-implementation check. Recomputes the signal (MACD 5/36/5
    zero-cross with the PriceMove and PriorMove filters) in Python over a
    fixed window and exports every signal so it can be compared one by one
    with the TradingView Pine Script implementation. Agreement between two
    independent codebases guards against look-ahead bias and off-by-one
    errors in bar alignment.

WHY IT MATTERS
Intraday backtests are very sensitive to bar construction, timestamp
conventions and execution timing. This step establishes that signals are
computed on completed bars only and that entries happen at the next
available price.
