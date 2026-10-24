================================================================================
CONDITIONAL PREDICTABILITY OF MOMENTUM SIGNALS IN INTRADAY EQUITY-INDEX FUTURES
An empirical study of MACD histogram zero-crosses in Micro E-mini S&P 500 (MES)
================================================================================

Author:   Peter Wallin  (mrwallinpeter@gmail.com)
Type:     Independent quantitative research
Period:   2025-2026
Code:     Python 3 (pandas, NumPy, scikit-learn, mplfinance, OpenAI API)


--------------------------------------------------------------------------------
1. RESEARCH QUESTION
--------------------------------------------------------------------------------
Technical momentum indicators are widely used by practitioners, yet their
predictive value is rarely tested against a proper null hypothesis.
This project asks:

  (Q1) Does a bullish MACD histogram zero-cross on intraday futures data
       contain any information about future returns beyond what is obtained
       by entering at random times drawn from the same population?

  (Q2) If the unconditional signal is weak, can the edge be made conditional
       on observable market state: the momentum of the signal bar, the
       preceding trend, volatility regime, time of day and higher-timeframe
       context?

  (Q3) Do such conditional effects survive out-of-sample (walk-forward)
       evaluation, and do they transfer to related but different markets?


--------------------------------------------------------------------------------
2. DATA
--------------------------------------------------------------------------------
* Primary: 1-minute OHLCV bars for CME Micro E-mini S&P 500 futures (MES),
  continuous contract, 2021-2026 (main statistical window 2022-2026).
* Portability: Micro E-mini Nasdaq-100 (MNQ) and Micro E-mini Dow (MYM),
  1-minute bars, same period.
* Signal bars (15 min, 30 min, 4 h) are resampled from the 1-minute series.
  Entries are executed at the first 1-minute open AFTER the signal bar has
  completed, so no information from the signal bar's future is used.
* Raw data files are not included in this repository (licensing and size).
  Every script reads them from a single path defined at the top of the file.


--------------------------------------------------------------------------------
3. METHODOLOGY
--------------------------------------------------------------------------------
Signal:    MACD(5, 36, 5) histogram crossing from <= 0 to > 0 (long) or
           from >= 0 to < 0 (short) on a completed 15- or 30-minute bar.

Outcome:   Fixed-horizon forward return in index points (typically 60-240
           min), plus path statistics (MAE/MFE) and stop/target simulations.

Metrics:   Profit factor (PF), average P&L, win rate, SQN / t-statistic,
           maximum drawdown, and the number of profitable years
           (a stability criterion).

Null model (core of the design):
           For every eligible timestamp the forward return is precomputed.
           MACD trades are a subset of these timestamps; the null
           distribution is built from 1,000 random subsets of equal size,
           matched on year and hour of day (actual signal bars excluded
           from the random pool). The MACD result is reported as a
           percentile of that distribution. Because execution is identical
           for signal and random trades, any difference can be attributed to
           the signal rather than to the backtest mechanics.

Robustness controls:
           * Parameter plateaus rather than single optimal values
             ("plateau" tests: an effect must hold across neighbouring
             parameter ranges).
           * Raw vs rolling-percentile versions of every feature, to separate
             level effects from regime effects.
           * Year-by-year decomposition of every result.
           * Walk-forward validation with expanding training windows
             (train 2022 -> test 2023, ..., train 2022-2025 -> test 2026).
           * Cross-market portability (MNQ, MYM).
           * Independent re-implementation check against the TradingView
             (Pine Script) version of the signal.


--------------------------------------------------------------------------------
4. REPOSITORY STRUCTURE (follows the order of the research)
--------------------------------------------------------------------------------
01_Data_Validation_and_Market_Characterisation   Data checks, intraday
                                                 volatility cycle
02_Baseline_Signal_vs_Random_Benchmark           Q1: raw signal vs matched
                                                 random null
03_Entry_Momentum_Features                       Q2: strength of the signal
                                                 bar (price move, candle,
                                                 histogram, RSI)
04_Prior_Trend_and_Range_Context                 Q2: preceding trend,
                                                 sideways ranges, breakouts
05_Volatility_and_Market_Regime                  Q2: ATR, Bollinger width,
                                                 market-state taxonomy
06_Multi_Timeframe_Analysis                      Q2: 15m / 30m / 4h
                                                 interaction
07_Session_Timing_and_Holding_Period             Q2/Q3: time of day, holding
                                                 period, walk-forward
08_Risk_Management_MAE_MFE_Stops                 Trade path, stops, targets
09_Machine_Learning_Random_Forest                Q2/Q3: non-linear feature
                                                 combination, walk-forward
10_LLM_Visual_Chart_Analysis                     Can a multimodal LLM extract
                                                 market-state features from
                                                 chart images?
11_Cross_Market_Portability                      Q3: transfer to MNQ and MYM

Each folder contains a README.txt describing its scripts and findings.
All results (CSV tables and charts) are written to a separate results
directory; the scripts print summary tables to the console.


--------------------------------------------------------------------------------
5. SUMMARY OF FINDINGS
--------------------------------------------------------------------------------
(Q1) The unconditional MACD signal is weak. Over 1,716 filtered trades
     (2021-2026) its profit factor (1.09) lies at roughly the 72nd
     percentile of the matched-random distribution: above the median, but
     far from the 95th percentile required to reject the null. Much of the
     apparent profitability of the raw strategy is explained by market
     drift during the sample.

(Q2) Conditioning matters. Two simple, ATR-normalised filters raise the
     edge substantially:
       A: the move of the signal bar, (Close[t] - Close[t-1]) / ATR14,
          between 0.35 and 0.90 (a moderate, not exhausted, impulse);
       B: the prior 5-hour move, (Close[t-1] - Close[t-20]) / ATR14,
          between -2 and +2 (no extended trend before the signal).
     Strong time-of-day dependence was found: afternoon/evening signals
     (14:00-22:00) behave very differently from morning signals.

(Q3) Out-of-sample: in walk-forward testing (holding period chosen on past
     years only), the 14:00-22:00 block gave PF 1.94 over 208 OOS trades,
     profitable in 4 of 4 test years. A Random Forest ranking model (V4)
     separated OOS outcomes monotonically enough that the top prediction
     quintile reached PF 1.30, against roughly 1.00 for the bottom quintile.
     Portability is partial: the A+B combination improved MNQ (PF 1.22,
     positive in 5/5 years) but failed on MYM (PF 0.96): the effect is not
     universal, and what determines where it holds is an open question.

     Further results: a "moderate prior decline + mid Bollinger width"
     market state raised PF from 1.06 to 1.28 (897 signals, 4 of 5 years
     above 1); 4h higher-timeframe filters were unstable across holding
     periods; and image-based market-state labels from a multimodal LLM
     agreed with number-based labels 88% of the time for trend direction
     but only 52% of the time for volatility state.


--------------------------------------------------------------------------------
6. LIMITATIONS AND THREATS TO VALIDITY
--------------------------------------------------------------------------------
* Multiple testing: many hypotheses were explored on the same data, so
  in-sample results must be discounted. Walk-forward and cross-market tests
  were added specifically to address this, but a formal correction
  (e.g. White's Reality Check / Hansen's SPA test, or the deflated Sharpe
  ratio) is a natural next step.
* Sample sizes for some conditional subsets are small (tens of trades per
  year), so year-level estimates are noisy.
* Transaction costs and slippage are not modelled in most scripts
  (results are gross, in index points).
* A single asset class (US equity-index futures) and a period with
  specific regimes (2022 bear market, 2023-2025 bull market).


--------------------------------------------------------------------------------
7. POSSIBLE EXTENSIONS (RESEARCH DIRECTIONS)
--------------------------------------------------------------------------------
* Formal data-snooping-robust inference across the full family of tests.
* Regime-switching / hidden-Markov models instead of hand-defined states.
* Modelling the full return distribution (quantile regression,
  gradient-boosted distributional models) instead of mean returns.
* Extension to other asset classes (rates, FX, commodities) and to
  limit-order-book data to study the microstructure behind momentum
  ignition.
* Systematic evaluation of multimodal LLMs as feature extractors for
  financial time series.


--------------------------------------------------------------------------------
8. REPRODUCING THE RESULTS
--------------------------------------------------------------------------------
    pip install pandas numpy scikit-learn matplotlib mplfinance openai

1. Set the data path (CSV_PATH / ONE_MIN_FILE) at the top of a script.
2. Set the output path (OUTPUT_DIR / OUTPUT_PATH) if needed.
3. python <folder>/<script>.py
4. The LLM scripts (folder 10) need the environment variable OPENAI_API_KEY.

Scripts are independent of each other and can be run in any order.
