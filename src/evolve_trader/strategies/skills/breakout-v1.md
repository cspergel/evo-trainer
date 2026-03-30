---
name: breakout-v1
description: Trade price breakouts from consolidation ranges with volume confirmation and volatility expansion
entry_logic: >
  Identify stocks in consolidation: price range over the past 20 trading days is less than 60% of the
  range over the prior 60 days (volatility contraction). Enter long when price closes above the 20-day
  high with volume at least 1.5x the 20-day average volume. Enter short when price closes below the
  20-day low with the same volume confirmation. Additional filter: Bollinger Band width (20-period)
  must be below its 50-day average, confirming the squeeze condition prior to breakout. Exclude stocks
  with earnings announcements within the next 5 trading days (breakout may be noise from pre-earnings
  positioning). Require a minimum of 3 days of contraction before the breakout to ensure a genuine
  consolidation pattern rather than a brief pause in a volatile move.
exit_logic: >
  Initial stop-loss placed at the midpoint of the consolidation range (approximately 50% of the way
  back into the range). Trail stop to breakeven once the position moves 1x ATR(14) in the favorable
  direction. Trail further at 2x ATR(14) below the highest close for longs. Take 33% profit at 1.5x
  ATR target, another 33% at 3x ATR, and trail the final 33%. Time-based exit: close the position if
  it has not moved at least 1x ATR in the breakout direction within 5 trading days (failed breakout).
  Exit immediately if price reverses and closes back inside the consolidation range with above-average
  volume (false breakout confirmation).
position_sizing_default: "1.5% of portfolio per position"
target_regime: transitional, volatility expansion
expected_sharpe: 0.90
expected_max_drawdown: 0.13
expected_win_rate: 0.42
risk_parameters:
  max_position_pct: 0.04
  max_concurrent_breakout_trades: 6
  time_stop_days: 5
  volume_confirmation_multiplier: 1.5
---

# Breakout

## Reasoning Framework

Breakout trading is based on the observation that periods of low volatility (consolidation) tend to precede periods of high volatility (expansion), and the direction of the initial expansion often persists. This pattern reflects the accumulation/distribution dynamics of institutional investors: during consolidation, large players quietly build positions, and when the price moves beyond the range boundary, it triggers stop orders, momentum algorithms, and attention-driven retail buying that creates a self-reinforcing move. Volume confirmation is critical because breakouts on low volume are far more likely to be false signals that quickly reverse back into the range.

The strategy's primary challenge is the high rate of false breakouts: studies suggest that 50-70% of breakouts fail to sustain momentum beyond the initial move. This is why the win rate is expected to be relatively low at 42%, and the strategy depends on favorable risk-reward asymmetry (cutting losses quickly on false breakouts while letting winners run on genuine ones). The tiered profit-taking approach (33% at each target) ensures some gains are locked in even on breakouts that stall partway through the expected move. False breakouts are particularly common in low-liquidity stocks and during choppy market conditions, which is why the volume filter and regime awareness are essential.

The evolution engine should experiment with alternative consolidation detection methods beyond simple range contraction, such as Keltner Channel squeezes, declining ATR patterns, or fractal dimension measures. The volume confirmation threshold (1.5x) and the time-stop duration (5 days) are high-impact parameters for optimization. Additionally, incorporating order flow data or level 2 market depth information could help distinguish between genuine institutional accumulation breakouts and noise-driven false signals before the position is fully entered.
