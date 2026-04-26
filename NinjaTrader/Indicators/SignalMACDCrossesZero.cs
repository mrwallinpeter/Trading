//
// SignalMACDCrossesZero.cs
//
// Multi-timeframe MACD histogram zero-line cross indicator for NinjaTrader 8.
//
// Logic summary
// ─────────────
// 5-min signals  : 5-min MACD histogram (5,36,5) crosses zero on the current forming bar
//                  AND 15-min MACD histogram last closed bar > bar before it
// 15-min signals : 15-min MACD histogram (5,36,5) crosses zero on the current forming bar
//                  AND 1-hour MACD histogram last closed bar > bar before it
//
// Crosses are detected tick-by-tick on the current (forming) bar — no bar-close delay.
// One arrow is drawn per bar per signal type; if the histogram re-crosses on the same bar
// no duplicate arrow is added.
//
// Visual output (on price chart, IsOverlay = true)
// ─────────────────────────────────────────────────
//   Lime   arrow up   – 5-min bullish cross  (hist crosses zero from below, 15m rising)
//   Red    arrow down – 5-min bearish cross  (hist crosses zero from above, 15m rising)
//   Cyan   arrow up   – 15-min bullish cross (hist crosses zero from below, 1h rising)
//   Orange arrow down – 15-min bearish cross (hist crosses zero from above, 1h rising)
//
// Primary chart MUST be 5-minute.
// Place this file in:
//   Documents\NinjaTrader 8\bin\Custom\Indicators\
// then compile via NinjaTrader → Tools → Edit NinjaScript → Compile.
//

#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Windows.Media;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.DrawingTools;
using NinjaTrader.NinjaScript.Indicators;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class SignalMACDCrossesZero : Indicator
    {
        // ── Indicators ────────────────────────────────────────────────────────
        private MACD macd5m;
        private MACD macd15m;
        private MACD macd1h;

        // ── Per-bar de-duplication: track bar index of last drawn signal ──────
        private int last5mBullBar  = -1;
        private int last5mBearBar  = -1;
        private int last15mBullBar = -1;
        private int last15mBearBar = -1;

        // ── Minimum bars needed before calculating ────────────────────────────
        private const int MinBars = 50;

        // ── Arrow offset in ticks relative to bar high/low ────────────────────
        private const int ArrowOffset5m  = 3;
        private const int ArrowOffset15m = 7;  // larger so 15m arrows don't overlap 5m ones

        // ─────────────────────────────────────────────────────────────────────
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Tick-based 5-min and 15-min MACD (5,36,5) histogram zero-line cross signals with higher timeframe confirmation. Primary chart must be 5-minute.";
                Name        = "SignalMACDCrossesZero";
                Calculate   = Calculate.OnEachTick;
                IsOverlay   = false;

                MACDFast               = 5;
                MACDSlow               = 36;
                MACDSmooth             = 5;
                RecoveryThreshold      = 80;
                BelowZeroBarThreshold  = 4;

                // Sub-panel histogram: ±1 for 5m signals, ±2 for 15m signals
                AddPlot(new Stroke(Brushes.Lime, 6), PlotStyle.Bar, "Signal");
            }
            else if (State == State.Configure)
            {
                // Additional series — primary (index 0) is the 5-min chart
                AddDataSeries(BarsPeriodType.Minute, 15);  // BarsArray[1] – 15-min
                AddDataSeries(BarsPeriodType.Minute, 60);  // BarsArray[2] – 1-hour
            }
            else if (State == State.DataLoaded)
            {
                if (BarsPeriod.BarsPeriodType != BarsPeriodType.Minute || BarsPeriod.Value != 5)
                    Log("SignalMACDCrossesZero: primary chart should be 5-minute for correct signal timing.", LogLevel.Warning);

                macd5m  = MACD(BarsArray[0], MACDFast, MACDSlow, MACDSmooth);
                macd15m = MACD(BarsArray[1], MACDFast, MACDSlow, MACDSmooth);
                macd1h  = MACD(BarsArray[2], MACDFast, MACDSlow, MACDSmooth);
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        protected override void OnBarUpdate()
        {
            // All signal detection runs on primary (5-min) series ticks only
            if (BarsInProgress != 0)
                return;

            if (IsFirstTickOfBar)
            {
                Values[0][0]      = 0;
                PlotBrushes[0][0] = Brushes.Transparent;
            }

            if (CurrentBars[0] < MinBars ||
                CurrentBars[1] < MinBars ||
                CurrentBars[2] < MinBars)
                return;

            // 5m cross detection: forming bar vs last closed bar (tick-based)
            double h5      = macd5m.Diff[0];
            double h5Prev  = macd5m.Diff[1];

            // 15m cross detection: forming bar vs last closed bar (tick-based)
            double h15     = macd15m.Diff[0];
            double h15Prev = macd15m.Diff[1];

            // Higher-timeframe trend confirmation (closed bars only)
            bool confirm15mBull = IsRecovering(macd15m, CurrentBars[1], RecoveryThreshold);
            bool confirm15mBear = IsDeclining(macd15m,  CurrentBars[1], RecoveryThreshold);
            bool confirm1hBull  = !IsRecentlyBelowZero(macd1h, CurrentBars[2], BelowZeroBarThreshold);
            bool confirm1hBear  = !IsRecentlyBelowZero(macd1h, CurrentBars[2], BelowZeroBarThreshold);

            string alertSound = NinjaTrader.Core.Globals.InstallDir + @"\sounds\Alert1.wav";

            // ── 5-min bullish cross: histogram crosses zero from below ──────────
            if (h5 > 0 && h5Prev <= 0 && confirm15mBull && CurrentBar != last5mBullBar)
            {
                last5mBullBar = CurrentBar;
                Draw.ArrowUp(this, "5mBull_" + CurrentBar, false, 0,
                             Low[0] - ArrowOffset5m * TickSize, Brushes.Lime);
                Alert("5mBull", Priority.High, "5m MACD histogram crossed above zero (15m rising)",
                      alertSound, 1, Brushes.Lime, Brushes.Black);
                Values[0][0]      = 1;
                PlotBrushes[0][0] = Brushes.Lime;
            }

            // ── 5-min bearish cross: histogram crosses zero from above ──────────
            if (h5 < 0 && h5Prev >= 0 && confirm15mBear && CurrentBar != last5mBearBar)
            {
                last5mBearBar = CurrentBar;
                Draw.ArrowDown(this, "5mBear_" + CurrentBar, false, 0,
                               High[0] + ArrowOffset5m * TickSize, Brushes.Red);
                Alert("5mBear", Priority.High, "5m MACD histogram crossed below zero (15m rising)",
                      alertSound, 1, Brushes.Red, Brushes.White);
                Values[0][0]      = -1;
                PlotBrushes[0][0] = Brushes.Red;
            }

            // ── 15-min bullish cross: histogram crosses zero from below ─────────
            if (h15 > 0 && h15Prev <= 0 && confirm1hBull && CurrentBar != last15mBullBar)
            {
                last15mBullBar = CurrentBar;
                Draw.ArrowUp(this, "15mBull_" + CurrentBar, false, 0,
                             Low[0] - ArrowOffset15m * TickSize, Brushes.Cyan);
                Alert("15mBull", Priority.High, "15m MACD histogram crossed above zero (1h rising)",
                      alertSound, 1, Brushes.Cyan, Brushes.Black);
                Values[0][0]      = 2;
                PlotBrushes[0][0] = Brushes.Cyan;
            }

            // ── 15-min bearish cross: histogram crosses zero from above ─────────
            if (h15 < 0 && h15Prev >= 0 && confirm1hBear && CurrentBar != last15mBearBar)
            {
                last15mBearBar = CurrentBar;
                Draw.ArrowDown(this, "15mBear_" + CurrentBar, false, 0,
                               High[0] + ArrowOffset15m * TickSize, Brushes.Orange);
                Alert("15mBear", Priority.High, "15m MACD histogram crossed below zero (1h rising)",
                      alertSound, 1, Brushes.Orange, Brushes.Black);
                Values[0][0]      = -2;
                PlotBrushes[0][0] = Brushes.Orange;
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Histogram below zero, found its minimum, and has been rising >= threshold% since
        private bool IsRecovering(MACD macd, int barsCount, int thresholdPct)
        {
            if (barsCount < 3)          return false;
            if (macd.Diff[1] >= 0)      return false; // last closed bar must be below zero

            int lookback = Math.Min(barsCount - 1, 100);
            int minIdx   = 1;
            double minVal = macd.Diff[1];

            for (int i = 1; i <= lookback; i++)
            {
                if (macd.Diff[i] >= 0) break; // end of below-zero run
                if (macd.Diff[i] < minVal) { minVal = macd.Diff[i]; minIdx = i; }
            }

            int recoveryBars = minIdx - 1; // closed bars between minimum and last closed
            if (recoveryBars <= 0) return false;

            int risingCount = 0;
            for (int i = minIdx - 1; i >= 1; i--)
                if (macd.Diff[i] > macd.Diff[i + 1]) risingCount++;

            return (risingCount * 100.0 / recoveryBars) >= thresholdPct;
        }

        // Block entry if histogram is below zero and crossed there only recently (within maxBars)
        private bool IsRecentlyBelowZero(MACD macd, int barsCount, int maxBars)
        {
            if (macd.Diff[1] >= 0) return false; // above zero — no restriction

            int lookback = Math.Min(barsCount - 1, maxBars + 1);
            int count    = 0;
            for (int i = 1; i <= lookback; i++)
            {
                if (macd.Diff[i] < 0) count++;
                else break;
            }
            return count <= maxBars;
        }

        // Histogram above zero, found its peak, and has been falling >= threshold% since
        private bool IsDeclining(MACD macd, int barsCount, int thresholdPct)
        {
            if (barsCount < 3)          return false;
            if (macd.Diff[1] <= 0)      return false; // last closed bar must be above zero

            int lookback = Math.Min(barsCount - 1, 100);
            int maxIdx   = 1;
            double maxVal = macd.Diff[1];

            for (int i = 1; i <= lookback; i++)
            {
                if (macd.Diff[i] <= 0) break; // end of above-zero run
                if (macd.Diff[i] > maxVal) { maxVal = macd.Diff[i]; maxIdx = i; }
            }

            int declineBars = maxIdx - 1;
            if (declineBars <= 0) return false;

            int fallingCount = 0;
            for (int i = maxIdx - 1; i >= 1; i--)
                if (macd.Diff[i] < macd.Diff[i + 1]) fallingCount++;

            return (fallingCount * 100.0 / declineBars) >= thresholdPct;
        }

        // ─────────────────────────────────────────────────────────────────────
        #region Properties

        [NinjaScriptProperty]
        [Display(Name = "Fast Period", GroupName = "MACD Parameters", Order = 1)]
        public int MACDFast { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Slow Period", GroupName = "MACD Parameters", Order = 2)]
        public int MACDSlow { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Smooth Period", GroupName = "MACD Parameters", Order = 3)]
        public int MACDSmooth { get; set; }

        [NinjaScriptProperty]
        [Range(1, 100)]
        [Display(Name = "Recovery Threshold %", GroupName = "Signal Parameters", Order = 1)]
        public int RecoveryThreshold { get; set; }

        [NinjaScriptProperty]
        [Range(1, 50)]
        [Display(Name = "1h Below-Zero Bar Threshold", GroupName = "Signal Parameters", Order = 2)]
        public int BelowZeroBarThreshold { get; set; }

        #endregion
    }
}
