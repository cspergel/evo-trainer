---
name: pairs-trading-v1
description: Trade mean-reversion of price spreads between historically correlated stock pairs
entry_logic: >
  Identify pairs with Pearson correlation above 0.80 over a 252-day lookback and confirmed cointegration
  via the Engle-Granger test (p-value < 0.05). Calculate the z-score of the price spread (log price
  ratio) using a 60-day rolling mean and standard deviation. Enter a long-short position when the
  z-score exceeds +/- 2.0: go long the underperforming stock and short the outperforming stock. Scale
  into the position: 50% at z-score 2.0, additional 50% at z-score 2.5 if the divergence continues.
  Dollar-neutralize the pair: match position notional values so net market exposure is approximately zero.
  Revalidate cointegration monthly; do not enter new trades if the pair fails the cointegration test
  in the most recent validation. Limit to pairs within the same GICS sector to ensure a fundamental
  linkage supports the statistical relationship.
exit_logic: >
  Primary exit when the z-score reverts to 0 (full mean reversion). Take 50% profit at z-score 0.5
  (partial reversion). Stop-loss if the z-score exceeds +/- 3.5 (spread blowout indicating potential
  structural break in the relationship). Time-based stop: exit if the spread has not reverted to
  z-score 1.0 or below within 30 trading days. Exit immediately if one stock in the pair announces
  a merger, acquisition, or major restructuring that would fundamentally alter the correlation
  relationship. Exit all positions in a pair if the rolling 60-day correlation drops below 0.60,
  indicating a breakdown of the statistical relationship.
position_sizing_default: "2% of portfolio per pair (1% each leg)"
target_regime: range-bound, stable correlations
expected_sharpe: 1.10
expected_max_drawdown: 0.08
expected_win_rate: 0.63
risk_parameters:
  max_position_pct: 0.04
  max_concurrent_pairs: 8
  max_single_stock_exposure: 0.03
  correlation_min_threshold: 0.80
  cointegration_p_value_max: 0.05
---

# Pairs Trading

## Reasoning Framework

Pairs trading is a market-neutral strategy that exploits temporary dislocations in the price relationship between two fundamentally linked securities. The theoretical foundation rests on the concept of cointegration: while individual stock prices may be non-stationary and unpredictable, the spread between two cointegrated stocks is stationary and mean-reverting. This mean-reversion occurs because the same economic forces (sector demand, input costs, regulatory environment) drive both stocks, and temporary divergences caused by idiosyncratic events (analyst coverage, index rebalancing, short-term sentiment) tend to correct as fundamental value reasserts itself. By being simultaneously long and short, the strategy hedges out broad market risk and isolates the relative value component.

The primary risk is a structural break in the pair relationship: a merger announcement, regulatory change, competitive disruption, or business model pivot can permanently alter the correlation between two stocks, causing the spread to diverge without reverting. The z-score 3.5 stop-loss and the monthly cointegration revalidation are safeguards against this, but structural breaks are often sudden and unpredictable. The strategy also faces crowding risk: pairs trading is widely practiced by quantitative hedge funds, and popular pairs can become crowded, reducing the available spread and increasing the speed and magnitude of spread blowouts when multiple funds exit simultaneously. Transaction costs from maintaining two positions per trade and frequent rebalancing can also erode returns, particularly for lower-spread pairs.

The evolution engine should explore alternative spread models beyond simple log price ratios, such as Kalman filter-based dynamic hedge ratios that adapt to time-varying relationships. Optimizing the z-score entry and exit thresholds based on the historical spread distribution of each specific pair (rather than using universal thresholds) could improve performance. Incorporating fundamental similarity metrics (revenue correlation, customer overlap, supply chain linkage) alongside statistical cointegration could help pre-filter pairs that are more likely to maintain their relationship, reducing the structural break risk. Testing different lookback periods for the cointegration test and the rolling spread statistics is also a high-priority parameter for evolution.
