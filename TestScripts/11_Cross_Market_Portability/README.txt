================================================================================
11  CROSS-MARKET PORTABILITY                                (Research question 3)
================================================================================

PURPOSE
An effect found on one instrument may be a statistical accident. A strong
test is to apply the unchanged rules to related markets that were NOT
used during development.

FILES
MNQ_AB_PORTABILITY_TEST.py   Micro E-mini Nasdaq-100 futures (MNQ)
MYM_AB_PORTABILITY_TEST.py   Micro E-mini Dow futures (MYM)

DESIGN
The same MACD(5,36,5) signal, the same ATR-normalised Filters A and B and
the same 210-minute hold are applied without any re-optimisation. Results
are reported for BASELINE, A only, B only and A+B, overall and per year
(2022-2026).

RESULTS
                 Baseline PF   A+B PF   A+B trades   A+B positive years
  MNQ (Nasdaq)      1.09        1.22       1,218           5 / 5
  MYM (Dow)         1.06        0.96       1,178           2 / 5

INTERPRETATION
The filters transferred to the Nasdaq contract but failed on the Dow
contract. The conditioning effect is therefore not universal and appears
to depend on the instrument. This negative result is reported
deliberately: it limits the claims that can be made, and it points to a
concrete follow-up question: which market characteristics determine
whether momentum conditioning works?
