---
name: momentum-sector-rotation-v1
description: Rotate into strongest sectors based on relative strength rankings over multiple lookback periods
entry_logic: >
  Rank all 11 GICS sectors by a composite relative strength score: 40% weight on 1-month return, 35%
  weight on 3-month return, and 25% weight on 6-month return, all measured relative to the S&P 500.
  Go long the top 3 sectors via sector ETFs. Rebalance monthly on the first trading day. Only enter a
  sector if its composite RS score is positive (outperforming the benchmark). If fewer than 3 sectors
  have positive RS scores, allocate the remaining capital to short-term treasuries. Confirm sector
  momentum with positive money flow index (MFI > 50) for each selected sector ETF to validate
  institutional participation.
exit_logic: >
  Exit a sector position when it drops out of the top 4 sectors in the monthly ranking (buffer of 1
  rank to avoid excessive churn). Exit immediately if the sector's 1-month relative strength turns
  negative AND its 3-month RS is declining. Stop-loss at 8% below the entry price for each sector
  position. Reduce all equity exposure by 50% if the S&P 500 is below its 200-day SMA (broad risk-off
  signal). Full exit of all sector positions if 8 or more of 11 sectors show negative composite RS
  scores, indicating broad market weakness.
position_sizing_default: "3% of portfolio per sector position"
target_regime: risk-on, sector divergence
expected_sharpe: 0.85
expected_max_drawdown: 0.14
expected_win_rate: 0.52
risk_parameters:
  max_position_pct: 0.10
  max_sectors_held: 3
  rebalance_frequency_days: 21
  broad_market_filter: true
---

# Momentum Sector Rotation

## Reasoning Framework

Sector rotation momentum exploits the tendency for economic and market forces to favor different industry sectors at different phases of the business cycle and sentiment cycle. Strong sectors tend to remain strong over 1-6 month horizons due to persistent capital flows from institutional investors executing thematic trades, earnings revision cycles that unfold gradually across a sector, and herding behavior among fund managers who chase recent outperformers. By using a composite lookback that blends short, medium, and long-term relative strength, the strategy captures sectors with both recent acceleration and sustained outperformance, filtering out short-lived spikes.

The strategy struggles during broad market selloffs where correlations spike to 1.0 and sector differentiation collapses. In panic-driven drawdowns, the previously strongest sectors often fall the hardest as crowded momentum trades unwind simultaneously. The 200-day SMA market filter provides a coarse but effective safeguard against this scenario. The strategy also underperforms during rapid sector rotation caused by sudden policy shifts or macro surprises (e.g., an unexpected rate hike devastating rate-sensitive sectors overnight), where the monthly rebalance frequency is too slow to adapt.

The evolution engine should explore adaptive rebalance frequencies that accelerate during periods of rising cross-sector dispersion and decelerate when dispersion is low. Optimizing the relative weights assigned to each lookback period based on the current volatility regime could improve signal quality. Additionally, incorporating forward-looking signals such as earnings revision breadth or credit spreads by sector could provide early warning of rotation shifts before they appear in price-based momentum measures.
