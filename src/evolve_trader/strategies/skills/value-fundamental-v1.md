---
name: value-fundamental-v1
description: Buy undervalued stocks trading below sector-average P/E and P/B ratios with quality filters
entry_logic: >
  Screen for stocks with trailing P/E ratio at least 25% below their GICS industry group median AND
  price-to-book ratio at least 20% below the industry group median. Apply quality filters: require
  positive free cash flow for the trailing 12 months, debt-to-equity below the industry median, and
  return on equity above 8%. Exclude stocks with declining revenue for 3 consecutive quarters (value
  traps). Rank qualifying stocks by a composite value score (equal-weight P/E percentile + P/B percentile
  + EV/EBITDA percentile within sector). Enter positions in the top 10 ranked stocks. Rebalance quarterly
  after earnings seasons complete (mid-February, mid-May, mid-August, mid-November).
exit_logic: >
  Exit when the stock's P/E ratio rises above the industry group median (valuation gap has closed).
  Exit if the quality filters are violated: free cash flow turns negative for two consecutive quarters
  or debt-to-equity rises above 1.5x the industry median. Stop-loss at 15% below entry price to protect
  against value traps that continue deteriorating. Time-based review: if a position has not appreciated
  by at least 5% after two full quarters (approximately 6 months), reassess the thesis and exit if the
  fundamental catalyst has not materialized. Trim 50% of position if it appreciates 30% or more to lock
  in gains.
position_sizing_default: "2% of portfolio per position"
target_regime: any, long-term
expected_sharpe: 0.75
expected_max_drawdown: 0.18
expected_win_rate: 0.53
risk_parameters:
  max_position_pct: 0.04
  max_sector_exposure: 0.15
  max_positions: 10
  rebalance_frequency_days: 63
---

# Value Fundamental

## Reasoning Framework

Value investing is one of the most extensively documented factor premiums in academic finance, rooted in the behavioral tendency of investors to overextrapolate recent poor performance and underweight long-term mean reversion in corporate fundamentals. Stocks trading at low multiples relative to their sector peers often reflect excessive pessimism that creates a margin of safety. The quality filters (positive free cash flow, reasonable leverage, minimum ROE) are essential to distinguish genuine undervaluation from "value traps" -- companies that are cheap because their fundamentals are in structural decline. By screening within industry groups rather than the broad market, the strategy controls for legitimate differences in valuation norms across sectors.

The strategy's key risk is the "value trap" problem: some stocks are cheap for good reason and continue to underperform or go to zero. The 15% stop-loss and the declining revenue exclusion filter are imperfect safeguards. Value strategies also experience prolonged drawdowns during growth-dominated market regimes (such as 2017-2020) where expensive high-growth stocks persistently outperform cheap stocks. During these periods, the strategy requires patience and conviction, and the expected holding period of 3-12 months means capital is tied up for extended periods with uncertain timing of the catalyst.

The evolution engine should focus on improving value trap detection by incorporating alternative data signals such as insider buying activity, short interest trends, and analyst revision momentum as confirming or disconfirming indicators. Optimizing the relative weights of different valuation metrics based on sector characteristics (e.g., P/B is more relevant for financials, EV/EBITDA for industrials) could improve stock selection. The quarterly rebalance timing relative to earnings releases is also a key parameter to evolve, as early or late rebalancing captures different information sets.
