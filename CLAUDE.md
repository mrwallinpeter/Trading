# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Quantitative research on MACD histogram zero-cross signals in Micro E-mini S&P 500 (MES) futures, 2022-2026. All code lives in `TestScripts/`, organized as a research portfolio. `TestScripts/README.txt` describes the research questions, methodology, findings and limitations, and each numbered folder has its own `README.txt`.

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib mplfinance openai
```

The LLM scripts (`10_LLM_Visual_Chart_Analysis/`) read `OPENAI_API_KEY` from the environment.

## Running Code

Every script is standalone and can be run in any order:

```bash
python TestScripts/<folder>/<script>.py
```

## Architecture

- **Data:** 1-minute OHLCV CSVs (MES, plus MNQ and MYM for portability tests) are not in the repo. Each script reads them from an absolute path set at the top of the file (`CSV_PATH` / `ONE_MIN_FILE`), under `C:\Users\mrwal\Dropbox\Trading\NinjaTrader\`.
- **Outputs:** most scripts write CSV results to `C:\Users\mrwal\Dropbox\Trading\NinjaTrader\Testscripts\` (`OUTPUT_DIR` / `OUTPUT_PATH`). Scripts that use `os.path.dirname(__file__)` instead write to a sub-folder next to the script (all of folder 09, plus some in 03 and 05).
- **Shared conventions:** MACD(5, 36, 5); Wilder ATR(14); signal bars resampled from 1-minute data; entry at the first 1-minute open after the signal bar completes; higher-timeframe values come from the last completed bar only, which avoids look-ahead bias.
- **Core filters:** A = PriceMoveATR `(Close[t] - Close[t-1]) / ATR14` in 0.35-0.90; B = PriorMoveATR `(Close[t-1] - Close[t-20]) / ATR14` in -2..+2.
- **Null model:** signals are benchmarked against 1,000 random entry sets matched on year and hour.

## Folder layout

`01` data validation, `02` signal vs random benchmark, `03` entry momentum features, `04` prior trend and range context, `05` volatility and market regime, `06` multi-timeframe, `07` session timing, holding period and walk-forward, `08` risk management (MAE/MFE, stops), `09` Random Forest, `10` LLM chart analysis, `11` cross-market portability. New scripts go in the matching folder, and that folder's `README.txt` should be updated.
