---
name: rsi-divergence-v1
description: Trade bullish and bearish RSI divergences from price action as early reversal signals
entry_logic: >
  Bullish divergence entry: price makes a lower low while RSI(14) makes a higher low, with the RSI
  second low above 25 (not deeply oversold, which may indicate a strong trend rather than a reversal).
  Require at least 5 and no more than 30 trading days between the two RSI lows. Enter long when price
  closes above the high of the bar where the second RSI low occurred. Bearish divergence: price makes
  a higher high while RSI(14) makes a lower high, with the RSI second high below 75. Enter short when
  price closes below the low of the bar where the second RSI high occurred. Confirm with MACD histogram
  showing momentum shift in the divergence direction. Avoid signals if ADX(14) is above 40 (very strong
  trends can show divergence for extended periods before actually reversing).
exit_logic: >
  For bullish divergence longs: stop-loss at the lower of the two price lows that formed the divergence
  pattern, minus 0.5x ATR(14) for buffer. First target at the swing high between the two divergence
  lows (the "neckline"). Second target at 1.5x the distance from entry to the first target. Trail the
  final portion at 2x ATR(14). For bearish divergence shorts: mirror logic with stop above the higher
  of the two highs plus 0.5x ATR buffer. Time-based exit: if the position has not reached the first
  target within 15 trading days, exit at market. Exit immediately if a new divergence forms in the
  opposite direction.
position_sizing_default: "1.5% of portfolio per position"
target_regime: any with clear divergence
expected_sharpe: 0.80
expected_max_drawdown: 0.11
expected_win_rate: 0.48
risk_parameters:
  max_position_pct: 0.03
  max_concurrent_divergence_trades: 5
  time_stop_days: 15
  adx_max_threshold: 40
---

# RSI Divergence

## Reasoning Framework

RSI divergence is a classic technical signal that detects waning momentum before it manifests in price. When price makes a new extreme (higher high or lower low) but the RSI indicator fails to confirm, it indicates that the underlying buying or selling pressure is weakening despite the price move. This divergence between momentum and price often precedes reversals because the final price extreme is driven by the last marginal participants (often retail traders chasing the move) rather than strong institutional conviction. The MACD confirmation filter adds a second momentum measure to reduce the noise inherent in single-indicator divergence signals.

The key weakness of RSI divergence trading is that divergences can persist for extended periods in strong trends. A stock in a powerful uptrend may show bearish divergence repeatedly while continuing to make new highs -- this is known as "divergence in a trend" and is one of the most common traps for reversal traders. The ADX filter at 40 helps avoid the strongest trends, but intermediate-strength trends (ADX 25-40) can still produce persistent false divergences. Additionally, the signal is inherently discretionary in its identification: the spacing between the two reference points significantly affects signal quality, and algorithmic detection requires careful parameterization to match what experienced technical analysts would identify visually.

The evolution engine should focus on optimizing the RSI period (14 is standard but may not be optimal for all timeframes), the minimum and maximum spacing between divergence points, and the ADX threshold for trend filtering. Testing alternative momentum indicators alongside RSI (such as stochastic oscillator, CCI, or Williams %R) to create a multi-indicator divergence composite could improve signal reliability. The profit target methodology (swing-based targets vs. fixed ATR multiples) is another area where evolution could find regime-dependent optimal approaches.
