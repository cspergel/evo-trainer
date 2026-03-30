---
name: trend-following-v1
description: Ride momentum in established trends using moving average alignment and ADX confirmation
entry_logic: >
  Enter long when the 20-day EMA crosses above the 50-day EMA, the 50-day EMA is above the 200-day SMA,
  and ADX(14) is above 25 indicating a strong trend. Price must be above the 20-day EMA at time of entry.
  Confirm with volume being at least 1.2x the 20-day average volume on the crossover day.
  Enter short (or exit all longs) when the 20-day EMA crosses below the 50-day EMA and ADX remains above 25.
  Avoid entry if ADX is below 20 (choppy, non-trending market).
exit_logic: >
  Exit long positions when price closes below the 50-day EMA for two consecutive days, or when ADX drops
  below 20 signaling trend exhaustion. Place a trailing stop at 2.5x ATR(14) below the highest close since
  entry. Hard stop-loss at 3.5x ATR(14) below entry price. Take partial profits (50%) when position gains
  2x ATR(14), and trail the remainder. Exit immediately if the 50-day EMA crosses below the 200-day SMA
  (death cross).
position_sizing_default: "2% of portfolio per position"
target_regime: risk-on, trending
expected_sharpe: 1.05
expected_max_drawdown: 0.15
expected_win_rate: 0.45
risk_parameters:
  max_position_pct: 0.05
  max_correlated_exposure: 0.15
  trailing_stop_atr_multiplier: 2.5
  hard_stop_atr_multiplier: 3.5
---

# Trend Following

## Reasoning Framework

Trend following exploits the well-documented tendency of financial assets to exhibit serial correlation in returns over medium-term horizons. This persistence arises from behavioral biases such as anchoring (investors underreact to new information), herding (momentum attracts more buyers), and the gradual diffusion of information across heterogeneous market participants. By aligning multiple moving averages of different timeframes, the strategy filters out noise and only enters when short, medium, and long-term momentum are in agreement, reducing the probability of whipsaws.

The primary failure mode of trend following is range-bound or choppy markets where price oscillates around the moving averages, generating frequent false crossover signals. During these regimes, the strategy will experience consecutive small losses as stops are triggered repeatedly. The ADX filter mitigates this by requiring a minimum trend strength before entry, but it cannot eliminate all whipsaws. Trend following also suffers during sharp V-shaped reversals where the trailing stop is hit after giving back a significant portion of unrealized gains.

The evolution engine should focus on optimizing the ADX threshold for trend confirmation, the ATR multiplier for trailing stops (balancing between being stopped out too early vs. giving back too much profit), and the regime detection mechanism that determines when to activate or deactivate this strategy. Additionally, experimenting with adaptive moving average periods that adjust based on recent volatility could improve responsiveness without sacrificing noise filtering.
