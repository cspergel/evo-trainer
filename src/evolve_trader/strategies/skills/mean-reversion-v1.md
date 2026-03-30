---
name: mean-reversion-v1
description: Buy dips in range-bound markets using Bollinger Bands and RSI oversold confirmation
entry_logic: >
  Enter long when price touches or closes below the lower Bollinger Band (20-period, 2 standard deviations)
  AND RSI(14) is below 30 AND the 200-day SMA slope is flat (absolute slope less than 0.1% per day),
  confirming a range-bound market. Require that the stock has reverted to the mean at least twice in the
  past 60 days to confirm mean-reverting behavior. Enter short when price touches or closes above the
  upper Bollinger Band AND RSI(14) is above 70 under the same regime conditions. Scale into positions:
  50% at first touch, additional 50% if price moves another 0.5 standard deviations against the position.
exit_logic: >
  Exit long positions when price reaches the 20-period moving average (middle Bollinger Band) for the
  first partial target (50% of position). Exit the remaining 50% at the upper Bollinger Band. Stop-loss
  at 1.5 standard deviations below the lower Bollinger Band at time of entry (approximately 3.5 standard
  deviations from the mean). Time-based stop: exit if position has not reverted to the mean within 10
  trading days. Exit all positions immediately if ADX(14) rises above 30, indicating a potential trend
  regime shift that invalidates the mean-reversion thesis.
position_sizing_default: "1.5% of portfolio per position"
target_regime: range-bound, stable
expected_sharpe: 0.95
expected_max_drawdown: 0.10
expected_win_rate: 0.62
risk_parameters:
  max_position_pct: 0.04
  max_sector_exposure: 0.12
  time_stop_days: 10
  bb_stop_std_devs: 3.5
---

# Mean Reversion

## Reasoning Framework

Mean reversion strategies are grounded in the statistical property that asset prices tend to oscillate around a central value during range-bound market conditions. This behavior is driven by market microstructure dynamics: when prices deviate significantly from fair value, contrarian institutional investors and market makers step in to provide liquidity, pushing prices back toward equilibrium. Bollinger Bands provide a dynamic measure of "overextension" that adapts to current volatility, while RSI confirmation ensures that the price deviation is accompanied by genuine oversold or overbought momentum conditions rather than a structural break.

The strategy's primary vulnerability is regime change: when a range-bound market transitions into a trending market, mean-reversion entries become counter-trend trades that suffer escalating losses. A stock touching the lower Bollinger Band during a genuine downtrend will continue falling, turning what was expected to be a dip into a sustained decline. The ADX-based regime filter and time-based stop are critical safeguards, but regime transitions are inherently difficult to detect in real-time. The strategy also underperforms during periods of expanding volatility when Bollinger Bands widen rapidly, reducing the signal quality of band touches.

The evolution engine should prioritize improving the regime classification accuracy to better distinguish mean-reverting environments from early-stage trends. Experimenting with alternative mean-reversion indicators (such as z-scores of returns, Hurst exponent estimates, or variance ratio tests) could complement Bollinger Bands. The scaling-in logic and time-stop parameters are also strong candidates for optimization, as the optimal holding period and entry staging will vary across different market conditions and asset classes.
