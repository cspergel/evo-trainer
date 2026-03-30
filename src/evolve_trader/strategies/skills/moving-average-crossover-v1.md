---
name: moving-average-crossover-v1
description: Trade 50/200 day moving average golden cross and death cross signals with trend filters
entry_logic: >
  Enter long when the 50-day SMA crosses above the 200-day SMA (golden cross) AND price is above both
  moving averages at the time of crossover. Confirm with weekly MACD(12,26,9) being positive and rising.
  Volume on the crossover week should be above the 10-week average volume. Enter short or exit all longs
  when the 50-day SMA crosses below the 200-day SMA (death cross) with price below both averages.
  Additional filter: ignore crossover signals if the spread between the 50-day and 200-day SMA at the
  time of cross is less than 0.3% (very tight intertwining suggests a range-bound market likely to
  produce whipsaws). Prefer signals where the 200-day SMA itself is sloping in the direction of the
  trade (rising for golden cross, falling for death cross).
exit_logic: >
  Primary exit on the opposite crossover signal (death cross exits golden cross longs, and vice versa).
  Trailing stop at 3x ATR(20) below the highest weekly close for long positions. Hard stop-loss at 7%
  below entry price. Take partial profits: sell 25% at 10% gain, another 25% at 20% gain, trail the
  remaining 50%. Exit if the 200-day SMA flattens and begins curling in the opposite direction even
  before a formal crossover occurs (early exit signal). Monthly review: if position is less than 2%
  profitable after 60 trading days, tighten the trailing stop to 2x ATR(20) to reduce the risk of
  a slow round-trip.
position_sizing_default: "2.5% of portfolio per position"
target_regime: trending, medium-term
expected_sharpe: 0.70
expected_max_drawdown: 0.16
expected_win_rate: 0.43
risk_parameters:
  max_position_pct: 0.06
  max_correlated_exposure: 0.18
  trailing_stop_atr_multiplier: 3.0
  hard_stop_pct: 0.07
---

# Moving Average Crossover

## Reasoning Framework

The 50/200 day moving average crossover is one of the most widely followed technical signals in financial markets, monitored by retail traders, institutional investors, and algorithmic systems alike. Its effectiveness derives not only from its trend-detection properties but also from its self-fulfilling nature: because so many participants watch for golden and death crosses, the signals trigger substantial order flow that reinforces the directional move. The 50-day and 200-day periods represent approximately one quarter and one year of trading data respectively, capturing the transition between medium-term and long-term trend direction in a way that filters most short-term noise.

The primary weakness of this strategy is its lagging nature. Moving average crossovers are inherently late signals: by the time the 50-day SMA crosses the 200-day, a significant portion of the trend move has already occurred. In the best case, the crossover captures the middle portion of a sustained trend; in the worst case, it triggers near the end of a trend, resulting in an entry near a top or bottom. The strategy generates poor returns in range-bound markets where the moving averages intertwine and produce repeated false crossover signals. The 0.3% minimum spread filter helps reduce these whipsaws but cannot eliminate them entirely. The expected win rate of 43% reflects these challenges, with profitability depending on large winners outweighing frequent small losses.

The evolution engine should investigate adaptive moving average periods that adjust based on market volatility (shorter periods in high-volatility environments to reduce lag, longer periods in low-volatility environments to reduce noise). Testing exponential or weighted moving averages versus simple moving averages could improve signal timing. The profit-taking schedule (25% at 10%, 25% at 20%) is a candidate for optimization, as the optimal scaling-out strategy depends heavily on the distribution of trend lengths and magnitudes in the current market regime. Combining the crossover signal with volume profile analysis or market breadth indicators could also improve signal filtering.
