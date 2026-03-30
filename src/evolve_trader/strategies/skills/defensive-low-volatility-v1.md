---
name: defensive-low-volatility-v1
description: Shift to low-beta dividend-paying stocks and defensive sectors during risk-off regimes
entry_logic: >
  Activate when the regime classifier signals risk-off conditions: VIX above 25, S&P 500 below its
  200-day SMA, and credit spreads (ICE BofA High Yield OAS) widening by more than 50bps over the past
  30 days. Screen for stocks with 3-year beta below 0.7, dividend yield above 2.5%, payout ratio below
  75% (sustainable dividend), and positive free cash flow. Prioritize utilities, consumer staples, and
  healthcare sectors. Rank by a composite defensive score: 40% weight on low beta, 30% weight on
  dividend yield, 30% weight on earnings stability (low coefficient of variation of quarterly EPS over
  8 quarters). Enter the top 15 ranked stocks. Transition gradually: rotate 20% of portfolio per week
  over 5 weeks to avoid market impact.
exit_logic: >
  Begin unwinding defensive positions when risk-off conditions ease: VIX drops below 20 for 5
  consecutive days AND S&P 500 reclaims its 200-day SMA AND credit spreads narrow by at least 30bps
  from their recent peak. Unwind at the same gradual pace (20% per week). Stop-loss at 10% below entry
  for individual positions (even defensive stocks can fall in severe bear markets). Exit any position
  where the dividend is cut or suspended. Exit if a stock's beta rises above 1.0 over a rolling 6-month
  window, indicating it has lost its defensive characteristics. Maintain at least 10% cash allocation
  during the defensive regime for opportunistic rebalancing.
position_sizing_default: "2.5% of portfolio per position"
target_regime: risk-off, high uncertainty
expected_sharpe: 0.65
expected_max_drawdown: 0.09
expected_win_rate: 0.55
risk_parameters:
  max_position_pct: 0.06
  max_positions: 15
  min_cash_allocation: 0.10
  transition_weeks: 5
---

# Defensive Low Volatility

## Reasoning Framework

The low-volatility anomaly is one of the most counterintuitive findings in empirical finance: historically, low-beta stocks have delivered risk-adjusted returns comparable to or better than high-beta stocks, directly contradicting the CAPM prediction that higher risk should be compensated with higher return. This anomaly persists because institutional investors face benchmarking incentives that drive them toward high-beta stocks (lottery-ticket preference), leaving low-volatility stocks systematically under-owned and underpriced. During risk-off periods specifically, the flight to quality amplifies this effect as panicked investors sell volatile names indiscriminately, creating relative outperformance for defensive holdings.

The strategy's weakness is opportunity cost during strong bull markets and risk-on rallies, where low-beta stocks significantly underperform. If the regime detection triggers too early or produces false positives, the portfolio will miss substantial upside by being defensively positioned. The gradual transition mechanism (5-week rotation) is a deliberate trade-off: it reduces market impact and the damage from false signals but means the portfolio is only partially protected during the first few weeks of a genuine downturn. In extreme crash scenarios (e.g., March 2020), even defensive stocks suffer significant drawdowns as correlations spike.

The evolution engine should focus on refining the regime detection triggers, particularly the thresholds for VIX, credit spreads, and the SMA filter. A more nuanced multi-signal regime classifier that incorporates yield curve dynamics, put/call ratios, and fund flow data could reduce false positive rates. The transition speed is another key parameter to evolve: faster transitions provide better downside protection but higher transaction costs and more exposure to signal noise. Exploring dynamic position sizing within the defensive portfolio (overweighting the most stable names during the highest-stress periods) could also improve risk-adjusted returns.
