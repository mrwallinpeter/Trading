//
// MultiTFMACDLong.cs
//
// Multi-timeframe MACD long strategy for NinjaTrader 8.
//
// Logic summary
// ─────────────
// Trend filter  : 1-hour  MACD line   (5,36,5) > 0
// Signal        : 5-min   MACD hist   (5,36,5) crosses above 0 on bar close
// 15-min confirm: 15-min  MACD hist   (5,36,5) closed < 0 AND hist[1] > hist[2]
// Entry         : 1-min   MACD hist   (5,36,5) > 0, 5-min hist still > 0,
//                 within 5 one-minute candles of the 5-min signal
// Stop loss     : 1.5 × ATR(14) on 1-minute bars below the entry price
// Exit          : 5-min MACD histogram turns down on bar close (current < previous)
//
// Primary chart timeframe must be 1 minute.
// Place this file in:
//   Documents\NinjaTrader 8\bin\Custom\Strategies\
// then compile via NinjaTrader → Tools → Edit NinjaScript → Compile.
//

#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.DrawingTools;
using NinjaTrader.NinjaScript.Indicators;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class MultiTFMACDLong : Strategy
    {
        // ── Indicators ────────────────────────────────────────────────────────
        private MACD macd1Min;
        private MACD macd5Min;
        private MACD macd15Min;
        private MACD macd1Hour;
        private ATR  atr1Min;

        // ── Signal state ──────────────────────────────────────────────────────
        private bool fiveMinSignalActive;
        private DateTime fiveMinSignalTime;  // timestamp of the 5-min bar that triggered the signal

        // ── Guard flag set when primary series is not 1-minute ────────────────
        private bool invalidTimeframe;

        // ── Constants ─────────────────────────────────────────────────────────
        private const int    MACDFast        = 5;
        private const int    MACDSlow        = 36;
        private const int    MACDSmooth      = 5;
        private const int    ATRPeriod       = 14;
        private const double ATRMultiplier   = 1.5;


        // ─────────────────────────────────────────────────────────────────────
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Long-only multi-timeframe MACD strategy (1m/5m/15m/1h).";
                Name        = "MultiTFMACDLong";
                Calculate   = Calculate.OnEachTick;   // intrabar detection on 1-min series

                EntriesPerDirection          = 1;
                EntryHandling                = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds    = 30;
                BarsRequiredToTrade          = 50;
                MaximumBarsLookBack          = MaximumBarsLookBack.TwoHundredFiftySix;

                StartBehavior         = StartBehavior.WaitUntilFlat;
                TimeInForce           = TimeInForce.Gtc;
                TraceOrders           = false;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling    = StopTargetHandling.PerEntryExecution;
            }
            else if (State == State.Configure)
            {
                // Additional data series (primary 1-min series is index 0)
                AddDataSeries(BarsPeriodType.Minute,  5);  // BarsArray[1] – 5-min  → BarsInProgress == 1
                AddDataSeries(BarsPeriodType.Minute, 15);  // BarsArray[2] – 15-min → BarsInProgress == 2
                AddDataSeries(BarsPeriodType.Minute, 60);  // BarsArray[3] – 1-hour → BarsInProgress == 3
            }
            else if (State == State.DataLoaded)
            {
                // Enforce 1-minute primary chart
                if (BarsPeriod.BarsPeriodType != BarsPeriodType.Minute || BarsPeriod.Value != 1)
                {
                    invalidTimeframe = true;
                    Log("MultiTFMACDLong requires a 1-minute primary chart. Strategy will not trade.", LogLevel.Error);
                    return;
                }

                macd1Min  = MACD(BarsArray[0], MACDFast, MACDSlow, MACDSmooth);
                macd5Min  = MACD(BarsArray[1], MACDFast, MACDSlow, MACDSmooth);
                macd15Min = MACD(BarsArray[2], MACDFast, MACDSlow, MACDSmooth);
                macd1Hour = MACD(BarsArray[3], MACDFast, MACDSlow, MACDSmooth);
                atr1Min   = ATR(BarsArray[0], ATRPeriod);

                fiveMinSignalActive = false;
                fiveMinSignalTime   = DateTime.MinValue;
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        protected override void OnBarUpdate()
        {
            if (invalidTimeframe)
                return;

            // Wait until all series have enough history
            if (CurrentBars[0] < BarsRequiredToTrade ||
                CurrentBars[1] < BarsRequiredToTrade ||
                CurrentBars[2] < BarsRequiredToTrade ||
                CurrentBars[3] < BarsRequiredToTrade)
                return;

            // ── 5-MINUTE SERIES: signal detection and exit ───────────────────
            if (BarsInProgress == 1)
            {
                // Only process once per new 5-minute bar — all logic uses closed bars
                if (!IsFirstTickOfBar)
                    return;

                // EXIT on 5-min bar close: histogram turned down
                if (Position.MarketPosition == MarketPosition.Long)
                {
                    if (macd5Min.Diff[1] < macd5Min.Diff[2])
                    {
                        ExitLong("Exit_5mHistDown", "Long");
                        fiveMinSignalActive = false;
                    }
                    return;
                }

                // 15-min confirmation uses only closed 15-min bars (Diff[1] and Diff[2])
                bool confirm15m = macd15Min.Diff[1] < 0 &&
                                  macd15Min.Diff[1] > macd15Min.Diff[2];

                // True cross: just-closed 5-min bar above zero, prior bar was at or below zero
                bool fiveMinCrossUp = macd5Min.Diff[1] > 0 && macd5Min.Diff[2] <= 0;

                if (!fiveMinSignalActive && fiveMinCrossUp && confirm15m)
                {
                    fiveMinSignalActive = true;
                    fiveMinSignalTime   = Times[1][1];   // timestamp of the just-closed 5-min bar
                }

                return;
            }

            // ── 1-MINUTE SERIES: entry management ────────────────────────────
            if (BarsInProgress != 0)
                return;

            // Already in a position; exit is handled on the 5-min series above
            if (Position.MarketPosition == MarketPosition.Long)
                return;

            // Trend filter: 1-hour MACD line must be above zero
            if (macd1Hour[0] <= 0)
            {
                fiveMinSignalActive = false;
                return;
            }

            if (!fiveMinSignalActive)
                return;

            // Expire signal 5 minutes after the 5-minute signal timestamp
            if (Time[0] >= fiveMinSignalTime.AddMinutes(5))
            {
                fiveMinSignalActive = false;
                return;
            }

            // Enter when 1-min and 5-min MACD histograms are both above zero
            if (macd1Min.Diff[0] > 0 && macd5Min.Diff[0] > 0)
            {
                double stopPrice = Close[0] - ATRMultiplier * atr1Min[0];

                // Stop must be set before the entry order is submitted
                SetStopLoss("Long", CalculationMode.Price, stopPrice, false);
                EnterLong("Long");

                fiveMinSignalActive = false;  // one entry per signal
            }
        }
    }
}
