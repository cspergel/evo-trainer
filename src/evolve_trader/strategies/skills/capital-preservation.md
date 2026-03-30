---
name: capital-preservation
description: Hold cash and make no trades when conditions are uncertain
entry_logic: Never enter positions — this skill's purpose is to avoid trading
exit_logic: Not applicable — no positions to exit
position_sizing_default: 100% cash
target_regime: uncertain, low-confidence, conflicting signals
expected_sharpe: 0.0
expected_max_drawdown: 0.0
expected_win_rate: 1.0
risk_parameters:
  max_position_pct: 0.0
---

# Capital Preservation

## Reasoning Framework
Many of the best real-world traders attribute a large portion of their returns
to NOT trading when conditions are unclear. This skill represents the explicit
decision to hold cash.

## When This Skill Wins
- Regime classifier confidence below threshold (default: 0.6)
- Signal source conflicts are unresolved
- The evolution engine may adjust the confidence threshold over time

## Why This Matters
Without an explicit "do nothing" option, the system will always deploy some
strategy, introducing unnecessary risk during ambiguous periods.
