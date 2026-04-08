# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Install Python dependencies:
```bash
pip install pandas numpy matplotlib ta requests
```

## Running Code

Run the backtesting script:
```bash
python Rulebased/Research_01/Backtest_01.py
```

## Architecture Overview

This is a personal trading research repository combining backtesting code, platform scripts, and analysis files.

### Core Backtesting (`Rulebased/Research_01/Backtest_01.py`)

Multi-timeframe MACD strategy backtester that:
- Fetches OHLC data from the Polygon.io REST API
- Tests signals across 5m, 15m, 1h, and 4h timeframes
- Generates equity curves via matplotlib
- Reads ticker lists from text files

The Polygon.io API key is stored directly in the script. If rotating the key, update it there.

### Trading Platform Scripts

- **NinjaTrader/**: Script specifications (in `.docx` files) describing stop-loss, profit target, and entry logic for NinjaScript strategies
- **TradingView_Files/PositionCandidates/**: Trade candidate notes and analysis
- **Trendspider/**: Scanner setup screenshots

### Web Tool (`TradingCalculation-filer/`)

A standalone HTML/JS/CSS trading calculator. The JS is obfuscated (Cloudflare-protected) and uses LZ string compression. Treat this as a static artifact — edits should focus on the unminified portions if any exist.

### Reference Files

- `Futures.xlsx`, `SectorMatch.xlsx`, `TradingCalc.xlsx` — spreadsheet-based analysis tools
- `TradingHours.docx`, `TradingFirms.docx` — reference documentation
