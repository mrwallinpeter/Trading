================================================================================
08  RISK MANAGEMENT: TRADE PATH (MAE/MFE), STOPS AND TARGETS
================================================================================

PURPOSE
Analyse the path of each trade, not just its final result, so that
stop-loss and profit-target rules are designed from data rather than
from convention.

DEFINITIONS
MAE = maximum adverse excursion (worst unrealised loss during the trade)
MFE = maximum favourable excursion (best unrealised gain during the trade)

FILES
  MACD_ATR_BUCKET_MAE_MFE_60MIN.py     Pure MACD crosses (long and short, no
                     filters): MAE, MFE and return at 15-60 min, by ATR
                     bucket
  MACD_Combined_MAE_MFE_StopStudy.py   Full MAE/MFE distributions for winners
                     vs losers; ATR-based stop simulation
  MACD_Main_Winner_MAE_Timing.py       WHEN, during a winning trade, does the
                     deepest drawdown occur? (informs how long a stop must
                     survive)
  STOPLOSS_MAE_TWO_FILTER_TEST.py      Grids of fixed-point and ATR-multiple
                     stops on the Filter A+B strategy
  TARGET_STOP_MATRIX_90MIN.py          Joint target x stop matrix for a
                     90-minute hold, year by year

KEY RESULT (timing of the worst drawdown in winning trades, main strategy)
Of 89 winning trades, 52% had reached their maximum adverse excursion
within 30 minutes, 69% within 60 minutes and 94% within 120 minutes.
Winners whose worst point came 30-120 minutes into the trade had a median
MAE of about 1.7-2.2 ATR before recovering. A stop tighter than this
would have turned many eventual winners into losses. Stops are therefore
evaluated as ATR multiples, reporting both their cost (winners stopped
out) and their benefit (losses avoided).
