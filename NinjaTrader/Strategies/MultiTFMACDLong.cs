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

        // ── Session filter properties (HHMMSS format, exchange time zone) ─────
        [NinjaScriptProperty]
        [Display(Name = "Session Start (HHMMSS)", GroupName = "Session Filter", Order = 1,
                 Description = "Earliest time entries are allowed, in HHMMSS exchange-time format. For MES1! use CT: 84500 = 8:45 AM CT (9:45 AM ET).")]
        public int SessionStart { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Session End (HHMMSS)", GroupName = "Session Filter", Order = 2,
                 Description = "Last time a new entry is allowed, in HHMMSS exchange-time format. For MES1! use CT: 143000 = 2:30 PM CT (3:30 PM ET). RTH closes at 3:15 PM CT.")]
        public int SessionEnd { get; set; }


        // ─────────────────────────────────────────────────────────────────────
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description  = "Long-only multi-timeframe MACD strategy (1m/5m/15m/1h).";
                Name         = "MultiTFMACDLong";
                Calculate    = Calculate.OnEachTick;   // intrabar detection on 1-min series
                SessionStart = 84500;    // 8:45 AM CT (= 9:45 AM ET) — avoids noisy open
                SessionEnd   = 143000;   // 2:30 PM CT (= 3:30 PM ET) — avoids last 45 min before RTH close

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
                // EXIT: checked on every tick using closed bars — closed bars don't
                // change mid-bar so this is stable, but fires as soon as the new bar opens
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

                // Intrabar cross: current forming 5-min bar just crossed above zero
                bool fiveMinCrossUp = macd5Min.Diff[0] > 0 && macd5Min.Diff[1] <= 0;

                if (!fiveMinSignalActive && fiveMinCrossUp && confirm15m)
                {
                    fiveMinSignalActive = true;
                    fiveMinSignalTime   = Times[1][0];   // timestamp of the current forming 5-min bar
                }

                // Invalidate intrabar if the histogram drops back to zero or below
                if (fiveMinSignalActive && macd5Min.Diff[0] <= 0)
                    fiveMinSignalActive = false;

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

            // Session time filter — no new entries outside liquid hours
            int now = ToTime(Time[0]);
            if (now < SessionStart || now >= SessionEnd)
                return;

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
