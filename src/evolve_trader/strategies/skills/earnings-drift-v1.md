---
name: earnings-drift-v1
description: Trade post-earnings announcement drift by entering after significant earnings surprises
entry_logic: >
  Enter long within the first 2 trading days after an earnings report if the company reports EPS that
  beats consensus estimates by at least 10% AND revenue beats by at least 3%. Require the stock to gap
  up at least 3% on the earnings day (confirming market recognition of the surprise). Additionally,
  check that at least 2 analyst estimates have been revised upward within 48 hours of the report.
  Enter short within 2 trading days if EPS misses by at least 10% AND revenue misses by at least 3%,
  with a gap down of at least 3%. Avoid entry if implied volatility (IV) rank is above the 90th
  percentile pre-earnings (priced-in move too large, limited drift potential). Limit to stocks with
  market cap above $2B and average daily volume above 1M shares.
exit_logic: >
  Hold positions for 15-40 trading days to capture the post-earnings drift window. Take 50% profit at
  the 15-day mark if the position is profitable. Trail the remaining 50% with a stop at 1.5x ATR(14)
  below the highest close. Hard stop-loss at 5% below entry for longs and 5% above entry for shorts.
  Exit all remaining positions by the 40th trading day regardless of P&L (drift effect dissipates).
  Exit immediately if the company issues a guidance revision that contradicts the earnings surprise
  direction (e.g., beat on earnings but guides below consensus for next quarter).
position_sizing_default: "1.5% of portfolio per position"
target_regime: earnings season, high-volatility
expected_sharpe: 1.15
expected_max_drawdown: 0.12
expected_win_rate: 0.58
risk_parameters:
  max_position_pct: 0.03
  max_concurrent_earnings_trades: 8
  max_holding_days: 40
  min_market_cap_billions: 2.0
---

# Earnings Drift

## Reasoning Framework

Post-earnings announcement drift (PEAD) is one of the most robust market anomalies, documented consistently since the 1960s. It arises because investors systematically underreact to earnings surprises: the initial price move on the announcement captures only a fraction of the total information content, and the remaining adjustment occurs gradually over the following 30-60 trading days. This underreaction is driven by anchoring bias (analysts and investors adjust estimates too slowly), attention limitations (not all market participants process the information simultaneously), and institutional constraints (many funds rebalance on fixed schedules rather than immediately after new information).

The strategy's main failure mode is when an earnings surprise is already fully priced in due to pre-earnings anticipation or when the surprise is driven by one-time items rather than sustainable improvements. Stocks with very high pre-earnings IV have often already moved significantly into the report, leaving limited drift potential. Additionally, in highly efficient large-cap markets, the drift window has compressed over time as more quantitative funds exploit this anomaly. The strategy also faces crowding risk during peak earnings seasons when many PEAD strategies compete for the same trades, reducing the available alpha.

The evolution engine should experiment with the surprise threshold parameters (the 10% EPS beat minimum may be too high or too low depending on the sector and market cap tier). Optimizing the holding period and profit-taking schedule based on the magnitude of the surprise could capture more of the drift for strong surprises while cutting weaker signals earlier. Incorporating natural language processing of earnings call transcripts to assess management tone and forward guidance quality would add a qualitative dimension that pure quantitative surprise metrics miss.
