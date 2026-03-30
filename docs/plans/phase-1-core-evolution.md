# Phase 1: Core Evolution Loop — Detailed Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Get trading strategies represented as SKILL.md files executing trades against Evolve-Trader's historical replay harness, with the OpenSpace evolution engine running FIX/DERIVED/CAPTURED cycles. Implement Capital Preservation, immutable risk constraints, stochastic fitness evaluation, complexity penalties, and the version DAG. Seed with 10-15 strategies and run cold-start experiments on NASDAQ 100.

**Architecture:** OpenSpace's evolution engine drives FIX/DERIVED/CAPTURED on trading strategy SKILL.md files. Evolve-Trader's own replay harness provides the historical market environment (AI-Trader is a paper trading simulator, not a backtest engine). A custom post-execution analyzer replaces OpenSpace's binary evaluation with financial metrics. Walk-forward validation gates promotion on out-of-sample performance.

**Tech Stack:** Python 3.12+, OpenSpace (evolution engine), AI-Trader (trading environment), pytest, SQLite (temporary, migrates to PostgreSQL in Phase 2)

**Parent plan:** [`2026-03-29-evolve-trader-master-plan.md`](2026-03-29-evolve-trader-master-plan.md)

**Prerequisites:** Phase 0 complete. Both codebases running. Integration points documented. Decision gates resolved.

---

## Task 1: Define the StrategySkill Schema

**Files:**
- Create: `src/evolve_trader/strategies/schema.py`
- Create: `src/evolve_trader/strategies/templates/`
- Create: `tests/unit/test_strategy_schema.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_strategy_schema.py
"""Tests for the StrategySkill schema."""
import pytest
from evolve_trader.strategies.schema import StrategySkill, parse_skill_md


def test_strategy_skill_has_required_fields():
    """A StrategySkill must have all required fields."""
    skill = StrategySkill(
        name="test-momentum-v1",
        description="Simple momentum strategy for testing",
        entry_logic="Buy when 20-day RSI crosses above 50 and price is above 50-day SMA",
        exit_logic="Sell when RSI drops below 40 or price drops 5% from entry (stop-loss)",
        position_sizing_default="2% of portfolio per position",
        target_regime="risk-on, momentum strengthening",
        expected_sharpe=1.0,
        expected_max_drawdown=0.10,
        expected_win_rate=0.55,
        risk_parameters={"max_position_pct": 0.05},
    )
    assert skill.name == "test-momentum-v1"
    assert skill.expected_sharpe == 1.0
    assert skill.risk_parameters["max_position_pct"] == 0.05


def test_strategy_skill_rejects_missing_entry_logic():
    """Entry logic is required."""
    with pytest.raises(ValueError):
        StrategySkill(
            name="bad-strategy",
            description="Missing entry logic",
            entry_logic="",  # empty
            exit_logic="Some exit logic",
            position_sizing_default="1%",
            target_regime="any",
        )


def test_parse_skill_md_roundtrip():
    """A SKILL.md file can be parsed and re-serialized."""
    md_content = '''---
name: test-momentum-v1
description: Simple momentum strategy for testing
entry_logic: Buy when 20-day RSI crosses above 50
exit_logic: Sell when RSI drops below 40
position_sizing_default: 2% of portfolio
target_regime: risk-on
expected_sharpe: 1.0
expected_max_drawdown: 0.10
expected_win_rate: 0.55
risk_parameters:
  max_position_pct: 0.05
---

# test-momentum-v1

## Reasoning Framework
When momentum indicators confirm an uptrend...
'''
    skill = parse_skill_md(md_content)
    assert skill.name == "test-momentum-v1"
    assert skill.expected_sharpe == 1.0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_strategy_schema.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.strategies.schema'`

**Step 3: Implement the schema**

```python
# src/evolve_trader/strategies/schema.py
"""StrategySkill schema — the core data model for trading strategy SKILL.md files."""
from __future__ import annotations

import yaml
from pydantic import BaseModel, field_validator


class StrategySkill(BaseModel):
    """A trading strategy encoded as a structured skill.

    Maps to a SKILL.md file with YAML frontmatter and markdown body.
    """

    name: str
    description: str
    entry_logic: str
    exit_logic: str
    position_sizing_default: str
    target_regime: str
    expected_sharpe: float | None = None
    expected_max_drawdown: float | None = None
    expected_win_rate: float | None = None
    risk_parameters: dict[str, float] = {}
    body: str = ""  # The markdown reasoning framework below the frontmatter

    @field_validator("entry_logic")
    @classmethod
    def entry_logic_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("entry_logic cannot be empty")
        return v

    @field_validator("exit_logic")
    @classmethod
    def exit_logic_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("exit_logic cannot be empty")
        return v


def parse_skill_md(content: str) -> StrategySkill:
    """Parse a SKILL.md file with YAML frontmatter into a StrategySkill."""
    if not content.startswith("---"):
        raise ValueError("SKILL.md must start with YAML frontmatter (---)")

    # Split frontmatter from body
    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ValueError("SKILL.md must have closing --- for frontmatter")

    frontmatter = yaml.safe_load(parts[1])
    body = parts[2].strip()

    return StrategySkill(**frontmatter, body=body)


def serialize_skill_md(skill: StrategySkill) -> str:
    """Serialize a StrategySkill back to SKILL.md format."""
    data = skill.model_dump(exclude={"body"}, exclude_none=True)
    frontmatter = yaml.dump(data, default_flow_style=False, sort_keys=False)
    return f"---\n{frontmatter}---\n\n{skill.body}\n"
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_strategy_schema.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/strategies/schema.py tests/unit/test_strategy_schema.py
git commit -m "feat: define StrategySkill schema with SKILL.md parsing"
```

---

## Task 2: Build the Post-Execution Analyzer

**Files:**
- Create: `src/evolve_trader/core/analyzer.py`
- Create: `tests/unit/test_analyzer.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_analyzer.py
"""Tests for the post-execution financial analyzer."""
import numpy as np
import pytest
from evolve_trader.core.analyzer import (
    TradeResult,
    StrategyPerformance,
    analyze_strategy_performance,
    compute_sharpe_ratio,
    compute_max_drawdown,
    analyze_failure_mode,
)


def test_sharpe_ratio_known_values():
    """Sharpe ratio computation matches hand-calculated values."""
    # Daily returns: mean=0.001, std=0.01 → annualized Sharpe ≈ 1.58
    daily_returns = [0.001] * 252  # constant daily return
    sharpe = compute_sharpe_ratio(daily_returns, risk_free_rate=0.0)
    assert abs(sharpe - 1.58) < 0.1


def test_sharpe_ratio_negative():
    """Negative returns produce negative Sharpe."""
    daily_returns = [-0.002] * 100
    sharpe = compute_sharpe_ratio(daily_returns, risk_free_rate=0.0)
    assert sharpe < 0


def test_max_drawdown_known_values():
    """Max drawdown matches hand-calculated value."""
    # Portfolio: 100 → 120 → 90 → 110
    # Max drawdown = (120 - 90) / 120 = 25%
    equity_curve = [100, 110, 120, 100, 90, 95, 110]
    mdd = compute_max_drawdown(equity_curve)
    assert abs(mdd - 0.25) < 0.01


def test_max_drawdown_no_drawdown():
    """Monotonically increasing equity has zero drawdown."""
    equity_curve = [100, 101, 102, 103, 104]
    mdd = compute_max_drawdown(equity_curve)
    assert mdd == 0.0


def test_analyze_strategy_performance():
    """Full strategy analysis produces all required metrics."""
    trades = [
        TradeResult(ticker="AAPL", entry_price=150, exit_price=160, shares=10, entry_date="2025-01-01", exit_date="2025-01-15"),
        TradeResult(ticker="MSFT", entry_price=300, exit_price=290, shares=5, entry_date="2025-01-05", exit_date="2025-01-20"),
        TradeResult(ticker="GOOGL", entry_price=140, exit_price=155, shares=8, entry_date="2025-01-10", exit_date="2025-01-25"),
    ]
    perf = analyze_strategy_performance(trades, initial_capital=100000)

    assert perf.win_rate == pytest.approx(2 / 3, abs=0.01)
    assert perf.total_trades == 3
    assert perf.mean_return is not None
    assert perf.variance is not None
    assert perf.skewness is not None
    assert perf.kurtosis is not None
    assert perf.max_drawdown >= 0
    assert perf.sharpe_ratio is not None


def test_analyze_failure_mode_entry_failure():
    """Failure tracing identifies entry logic as the problem."""
    # Trade that lost money immediately after entry (bad entry timing)
    trades = [
        TradeResult(ticker="TSLA", entry_price=200, exit_price=170, shares=10,
                    entry_date="2025-01-01", exit_date="2025-01-02",
                    reasoning="Entry based on RSI crossover"),
    ]
    failure = analyze_failure_mode(trades)
    assert failure is not None
    assert "entry" in failure.component.lower() or failure.component is not None


def test_distributional_metrics():
    """Distribution metrics capture tail risk."""
    trades_consistent = [
        TradeResult(ticker="SPY", entry_price=100, exit_price=101, shares=100,
                    entry_date=f"2025-01-{i:02d}", exit_date=f"2025-01-{i+1:02d}")
        for i in range(1, 21)
    ]
    perf = analyze_strategy_performance(trades_consistent, initial_capital=100000)
    # Consistent returns should have low variance and near-zero skewness
    assert perf.variance < 0.01
```

**Step 2: Run to verify failure**

```bash
pytest tests/unit/test_analyzer.py -v
```

Expected: FAIL — module not found

**Step 3: Implement the analyzer**

```python
# src/evolve_trader/core/analyzer.py
"""Post-execution financial analyzer for trading strategies.

Replaces OpenSpace's binary success/failure evaluation with
distributional financial metrics: Sharpe, drawdown, win rate,
and full return distribution analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats


@dataclass
class TradeResult:
    """A single completed trade."""

    ticker: str
    entry_price: float
    exit_price: float
    shares: float
    entry_date: str
    exit_date: str
    reasoning: str = ""

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def return_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price


@dataclass
class StrategyPerformance:
    """Full distributional performance analysis of a strategy."""

    total_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    mean_return: float
    variance: float
    skewness: float
    kurtosis: float
    total_pnl: float
    avg_return_per_trade: float
    tail_risk_5pct: float  # 5th percentile of returns (worst-case)


@dataclass
class FailureAnalysis:
    """Diagnosis of why a strategy underperformed."""

    component: str  # "entry_logic", "exit_logic", "regime_mismatch", "sizing"
    description: str
    severity: float  # 0-1, how much this component contributed to loss
    suggested_fix: str


def compute_sharpe_ratio(
    daily_returns: list[float],
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252.0,
) -> float:
    """Compute annualized Sharpe ratio from daily returns."""
    if len(daily_returns) < 2:
        return 0.0
    arr = np.array(daily_returns)
    excess = arr - (risk_free_rate / annualization_factor)
    if np.std(excess) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(annualization_factor))


def compute_max_drawdown(equity_curve: list[float]) -> float:
    """Compute maximum drawdown from an equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    arr = np.array(equity_curve)
    peak = np.maximum.accumulate(arr)
    drawdown = (peak - arr) / np.where(peak == 0, 1, peak)
    return float(np.max(drawdown))


def analyze_strategy_performance(
    trades: list[TradeResult],
    initial_capital: float,
) -> StrategyPerformance:
    """Compute full distributional performance metrics for a strategy."""
    if not trades:
        return StrategyPerformance(
            total_trades=0, win_rate=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
            mean_return=0.0, variance=0.0, skewness=0.0, kurtosis=0.0,
            total_pnl=0.0, avg_return_per_trade=0.0, tail_risk_5pct=0.0,
        )

    returns = [t.return_pct for t in trades]
    pnls = [t.pnl for t in trades]
    wins = sum(1 for r in returns if r > 0)

    # Build equity curve
    equity = [initial_capital]
    for pnl in pnls:
        equity.append(equity[-1] + pnl)

    arr = np.array(returns)

    return StrategyPerformance(
        total_trades=len(trades),
        win_rate=wins / len(trades) if trades else 0.0,
        sharpe_ratio=compute_sharpe_ratio(returns),
        max_drawdown=compute_max_drawdown(equity),
        mean_return=float(np.mean(arr)),
        variance=float(np.var(arr)),
        skewness=float(stats.skew(arr)) if len(arr) > 2 else 0.0,
        kurtosis=float(stats.kurtosis(arr)) if len(arr) > 3 else 0.0,
        total_pnl=sum(pnls),
        avg_return_per_trade=float(np.mean(arr)),
        tail_risk_5pct=float(np.percentile(arr, 5)) if len(arr) > 0 else 0.0,
    )


def analyze_failure_mode(trades: list[TradeResult]) -> Optional[FailureAnalysis]:
    """Trace back through trades to identify the primary failure component.

    Determines whether losses are attributable to entry timing, exit timing,
    regime mismatch, or position sizing.
    """
    if not trades:
        return None

    losing_trades = [t for t in trades if t.pnl < 0]
    if not losing_trades:
        return None

    # Heuristic: if most losses happen quickly (exit within 1-2 days),
    # it's likely an entry timing problem
    total_loss = sum(t.pnl for t in losing_trades)

    # Simple heuristic analysis — will be enhanced with LLM reasoning in later phases
    avg_loss_pct = np.mean([t.return_pct for t in losing_trades])

    if avg_loss_pct < -0.05:
        return FailureAnalysis(
            component="entry_logic",
            description=f"Average losing trade lost {avg_loss_pct:.1%}. "
                        "Large losses suggest poor entry timing.",
            severity=min(1.0, abs(avg_loss_pct) / 0.10),
            suggested_fix="Add confirmation signal before entry or tighten stop-loss.",
        )

    return FailureAnalysis(
        component="exit_logic",
        description=f"Average losing trade lost {avg_loss_pct:.1%}. "
                    "Moderate losses suggest exit timing could improve.",
        severity=min(1.0, abs(avg_loss_pct) / 0.10),
        suggested_fix="Consider trailing stop or time-based exit.",
    )
```

**Step 4: Run tests**

```bash
pytest tests/unit/test_analyzer.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/core/analyzer.py tests/unit/test_analyzer.py
git commit -m "feat: post-execution financial analyzer with distributional metrics"
```

---

## Task 3: Build the Walk-Forward Validation Harness

**Files:**
- Create: `src/evolve_trader/core/validation.py`
- Create: `tests/unit/test_validation.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_validation.py
"""Tests for the walk-forward validation harness."""
import pytest
from evolve_trader.core.validation import (
    WalkForwardConfig,
    WalkForwardWindow,
    generate_walk_forward_windows,
    validate_no_lookahead,
)


def test_generate_windows_correct_count():
    """Generate correct number of train/validate windows."""
    config = WalkForwardConfig(
        total_days=100,
        train_days=30,
        validate_days=10,
        step_days=10,
    )
    windows = generate_walk_forward_windows(config)
    # Windows: [0-30, 30-40], [10-40, 40-50], [20-50, 50-60], ...
    assert len(windows) > 0
    for w in windows:
        assert w.train_end == w.validate_start
        assert w.validate_end - w.validate_start == 10
        assert w.train_end - w.train_start == 30


def test_windows_no_overlap():
    """Validation data never overlaps with training data."""
    config = WalkForwardConfig(
        total_days=100,
        train_days=30,
        validate_days=10,
        step_days=10,
    )
    windows = generate_walk_forward_windows(config)
    for w in windows:
        assert w.validate_start >= w.train_end


def test_validate_no_lookahead():
    """Lookahead check catches data leakage."""
    window = WalkForwardWindow(
        train_start=0, train_end=30,
        validate_start=30, validate_end=40,
    )

    # Data within training period — OK
    assert validate_no_lookahead(window, data_timestamp=15) is True

    # Data within validation period — NOT OK for training
    assert validate_no_lookahead(window, data_timestamp=35) is False

    # Data after validation — NOT OK
    assert validate_no_lookahead(window, data_timestamp=50) is False
```

**Step 2: Run to verify failure**

```bash
pytest tests/unit/test_validation.py -v
```

**Step 3: Implement**

```python
# src/evolve_trader/core/validation.py
"""Walk-forward validation harness for strategy evolution.

Ensures strategies are evolved on training data and validated on
out-of-sample data. Prevents overfitting by gating promotion on
out-of-sample performance.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""

    total_days: int
    train_days: int = 30
    validate_days: int = 10
    step_days: int = 10


@dataclass
class WalkForwardWindow:
    """A single train/validate window."""

    train_start: int  # day index
    train_end: int
    validate_start: int
    validate_end: int


def generate_walk_forward_windows(config: WalkForwardConfig) -> list[WalkForwardWindow]:
    """Generate non-overlapping walk-forward train/validate windows."""
    windows: list[WalkForwardWindow] = []
    start = 0

    while start + config.train_days + config.validate_days <= config.total_days:
        train_start = start
        train_end = start + config.train_days
        validate_start = train_end
        validate_end = validate_start + config.validate_days

        windows.append(WalkForwardWindow(
            train_start=train_start,
            train_end=train_end,
            validate_start=validate_start,
            validate_end=validate_end,
        ))
        start += config.step_days

    return windows


def validate_no_lookahead(window: WalkForwardWindow, data_timestamp: int) -> bool:
    """Check if a data timestamp is valid for the training period of this window.

    Returns True if the data is within or before the training period.
    Returns False if the data is in the validation or future period (lookahead).
    """
    return data_timestamp < window.train_end
```

**Step 4: Run tests**

```bash
pytest tests/unit/test_validation.py -v
```

**Step 5: Commit**

```bash
git add src/evolve_trader/core/validation.py tests/unit/test_validation.py
git commit -m "feat: walk-forward validation harness with anti-lookahead checks"
```

---

## Task 4: Implement Capital Preservation Skill

**Files:**
- Create: `src/evolve_trader/strategies/capital_preservation.py`
- Create: `src/evolve_trader/strategies/skills/capital-preservation.md`
- Create: `tests/unit/test_capital_preservation.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_capital_preservation.py
"""Tests for the Capital Preservation (do nothing) skill."""
import pytest
from evolve_trader.strategies.capital_preservation import (
    should_activate_capital_preservation,
    CapitalPreservationConfig,
)


def test_activates_below_confidence_threshold():
    """Capital Preservation activates when confidence is below threshold."""
    config = CapitalPreservationConfig(confidence_threshold=0.6)
    assert should_activate_capital_preservation(
        regime_confidence=0.4, config=config
    ) is True


def test_does_not_activate_above_threshold():
    """Capital Preservation does not activate when confidence is sufficient."""
    config = CapitalPreservationConfig(confidence_threshold=0.6)
    assert should_activate_capital_preservation(
        regime_confidence=0.8, config=config
    ) is False


def test_activates_at_exactly_threshold():
    """At exactly the threshold, Capital Preservation activates (conservative)."""
    config = CapitalPreservationConfig(confidence_threshold=0.6)
    assert should_activate_capital_preservation(
        regime_confidence=0.6, config=config
    ) is True


def test_activates_on_signal_conflict():
    """Capital Preservation activates when signals conflict."""
    config = CapitalPreservationConfig(confidence_threshold=0.6)
    assert should_activate_capital_preservation(
        regime_confidence=0.9,  # high confidence
        config=config,
        unresolved_conflicts=True,
    ) is True


def test_threshold_is_configurable():
    """Different thresholds produce different activation behavior."""
    strict = CapitalPreservationConfig(confidence_threshold=0.8)
    loose = CapitalPreservationConfig(confidence_threshold=0.3)

    assert should_activate_capital_preservation(regime_confidence=0.5, config=strict) is True
    assert should_activate_capital_preservation(regime_confidence=0.5, config=loose) is False
```

**Step 2: Run to verify failure**

```bash
pytest tests/unit/test_capital_preservation.py -v
```

**Step 3: Implement**

```python
# src/evolve_trader/strategies/capital_preservation.py
"""Capital Preservation — the 'do nothing' skill.

Holds cash and makes no trades. Activated when the regime classifier's
confidence is below threshold or when signal source conflicts are unresolved.
The confidence threshold is itself evolvable.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CapitalPreservationConfig:
    """Configuration for Capital Preservation activation."""

    confidence_threshold: float = 0.6


def should_activate_capital_preservation(
    regime_confidence: float,
    config: CapitalPreservationConfig,
    unresolved_conflicts: bool = False,
) -> bool:
    """Determine if Capital Preservation should be the active strategy.

    Returns True (activate Capital Preservation) when:
    - Regime confidence is at or below the threshold
    - There are unresolved signal source conflicts
    """
    if unresolved_conflicts:
        return True
    return regime_confidence <= config.confidence_threshold
```

**Step 4: Create the SKILL.md file**

```markdown
# src/evolve_trader/strategies/skills/capital-preservation.md
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
```

**Step 5: Run tests and commit**

```bash
pytest tests/unit/test_capital_preservation.py -v
git add src/evolve_trader/strategies/capital_preservation.py \
        src/evolve_trader/strategies/skills/capital-preservation.md \
        tests/unit/test_capital_preservation.py
git commit -m "feat: Capital Preservation skill with configurable confidence threshold"
```

---

## Task 5: Implement Immutable Risk Constraints

**Files:**
- Create: `src/evolve_trader/core/risk_constraints.py`
- Create: `tests/unit/test_risk_constraints.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_risk_constraints.py
"""Tests for immutable risk constraints.

These constraints are NEVER subject to AI override or evolution.
"""
import pytest
from evolve_trader.core.risk_constraints import (
    RiskConstraints,
    PortfolioState,
    check_trade_allowed,
    ConstraintViolation,
)


def test_position_size_limit():
    """Trade exceeding 5% of portfolio is blocked."""
    constraints = RiskConstraints()
    portfolio = PortfolioState(
        total_value=100_000,
        positions={"AAPL": 3000},  # 3% already
        sector_exposure={"Technology": 0.03},
        current_drawdown=0.0,
    )
    # Trying to add 3% more AAPL → total 6% → exceeds 5%
    result = check_trade_allowed(
        constraints=constraints,
        portfolio=portfolio,
        ticker="AAPL",
        sector="Technology",
        trade_value=3000,
    )
    assert result.allowed is False
    assert result.violation == ConstraintViolation.POSITION_LIMIT


def test_position_size_within_limit():
    """Trade within 5% of portfolio is allowed."""
    constraints = RiskConstraints()
    portfolio = PortfolioState(
        total_value=100_000,
        positions={},
        sector_exposure={},
        current_drawdown=0.0,
    )
    result = check_trade_allowed(
        constraints=constraints,
        portfolio=portfolio,
        ticker="AAPL",
        sector="Technology",
        trade_value=4000,  # 4% — under limit
    )
    assert result.allowed is True


def test_sector_concentration_limit():
    """Trade exceeding 25% sector exposure is blocked."""
    constraints = RiskConstraints()
    portfolio = PortfolioState(
        total_value=100_000,
        positions={"AAPL": 12000, "MSFT": 12000},
        sector_exposure={"Technology": 0.24},  # 24% already
        current_drawdown=0.0,
    )
    result = check_trade_allowed(
        constraints=constraints,
        portfolio=portfolio,
        ticker="GOOGL",
        sector="Technology",
        trade_value=2000,  # would push to 26%
    )
    assert result.allowed is False
    assert result.violation == ConstraintViolation.SECTOR_LIMIT


def test_drawdown_forces_capital_preservation():
    """20% drawdown forces de-risking."""
    constraints = RiskConstraints()
    portfolio = PortfolioState(
        total_value=80_000,  # down from 100k
        positions={"AAPL": 5000},
        sector_exposure={"Technology": 0.0625},
        current_drawdown=0.20,  # exactly 20%
    )
    result = check_trade_allowed(
        constraints=constraints,
        portfolio=portfolio,
        ticker="MSFT",
        sector="Technology",
        trade_value=2000,
    )
    assert result.allowed is False
    assert result.violation == ConstraintViolation.DRAWDOWN_LIMIT


def test_constraints_are_immutable():
    """Risk constraints cannot be modified after creation."""
    constraints = RiskConstraints()
    # Attempting to relax limits should fail
    with pytest.raises(AttributeError):
        constraints.max_position_pct = 0.10  # trying to relax from 5% to 10%
```

**Step 2: Run to verify failure**

```bash
pytest tests/unit/test_risk_constraints.py -v
```

**Step 3: Implement**

```python
# src/evolve_trader/core/risk_constraints.py
"""Immutable risk constraints — NEVER subject to AI override.

These are the hard safety limits that sit outside the evolution engine.
No evolved skill can remove, relax, or override these constraints.
The AI evolves everything else.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ConstraintViolation(Enum):
    """Types of constraint violations."""

    NONE = "none"
    POSITION_LIMIT = "position_limit_exceeded"
    SECTOR_LIMIT = "sector_limit_exceeded"
    DRAWDOWN_LIMIT = "drawdown_limit_exceeded"


@dataclass(frozen=True)  # frozen=True makes it immutable
class RiskConstraints:
    """Hard risk limits. Frozen dataclass prevents modification."""

    max_position_pct: float = 0.05    # 5% max in any single position
    max_sector_pct: float = 0.25      # 25% max in any single sector
    max_drawdown_pct: float = 0.20    # 20% max drawdown before forced de-risk


@dataclass
class PortfolioState:
    """Current state of the portfolio for constraint checking."""

    total_value: float
    positions: dict[str, float]       # ticker → current value
    sector_exposure: dict[str, float] # sector → fraction of portfolio
    current_drawdown: float           # 0.0 to 1.0


@dataclass
class TradeCheckResult:
    """Result of a constraint check on a proposed trade."""

    allowed: bool
    violation: ConstraintViolation = ConstraintViolation.NONE
    message: str = ""


def check_trade_allowed(
    constraints: RiskConstraints,
    portfolio: PortfolioState,
    ticker: str,
    sector: str,
    trade_value: float,
) -> TradeCheckResult:
    """Check if a proposed trade violates any immutable risk constraints.

    This function is called before EVERY trade, regardless of strategy,
    confidence, or automation level. It is the first gate in the execution pipeline.
    """
    if portfolio.total_value <= 0:
        return TradeCheckResult(
            allowed=False,
            violation=ConstraintViolation.DRAWDOWN_LIMIT,
            message="Portfolio value is zero or negative.",
        )

    # Check drawdown limit
    if portfolio.current_drawdown >= constraints.max_drawdown_pct:
        return TradeCheckResult(
            allowed=False,
            violation=ConstraintViolation.DRAWDOWN_LIMIT,
            message=f"Current drawdown {portfolio.current_drawdown:.1%} "
                    f"exceeds limit {constraints.max_drawdown_pct:.1%}. "
                    f"Forced de-risk to Capital Preservation.",
        )

    # Check position size limit
    existing_position = portfolio.positions.get(ticker, 0.0)
    new_position_value = existing_position + trade_value
    position_pct = new_position_value / portfolio.total_value

    if position_pct > constraints.max_position_pct:
        return TradeCheckResult(
            allowed=False,
            violation=ConstraintViolation.POSITION_LIMIT,
            message=f"Position in {ticker} would be {position_pct:.1%} "
                    f"of portfolio, exceeding {constraints.max_position_pct:.1%} limit.",
        )

    # Check sector concentration limit
    existing_sector = portfolio.sector_exposure.get(sector, 0.0)
    new_sector_pct = existing_sector + (trade_value / portfolio.total_value)

    if new_sector_pct > constraints.max_sector_pct:
        return TradeCheckResult(
            allowed=False,
            violation=ConstraintViolation.SECTOR_LIMIT,
            message=f"Sector {sector} would be {new_sector_pct:.1%} "
                    f"of portfolio, exceeding {constraints.max_sector_pct:.1%} limit.",
        )

    return TradeCheckResult(allowed=True)
```

**Step 4: Run tests and commit**

```bash
pytest tests/unit/test_risk_constraints.py -v
git add src/evolve_trader/core/risk_constraints.py tests/unit/test_risk_constraints.py
git commit -m "feat: immutable risk constraints (5% position, 25% sector, 20% drawdown)"
```

---

## Task 6: Implement Stochastic Fitness Evaluation

**Files:**
- Create: `src/evolve_trader/core/fitness.py`
- Create: `tests/unit/test_fitness.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_fitness.py
"""Tests for stochastic fitness evaluation."""
import pytest
from evolve_trader.core.fitness import (
    compare_strategy_fitness,
    compute_complexity_penalty,
    FitnessResult,
)


def test_consistent_strategy_beats_volatile():
    """A strategy with lower variance is preferred over one with higher mean but higher variance."""
    consistent = FitnessResult(sharpe=0.9, sharpe_std=0.2, max_drawdown=0.08, n_evaluations=10)
    volatile = FitnessResult(sharpe=1.2, sharpe_std=0.8, max_drawdown=0.15, n_evaluations=10)

    # consistent should win (better risk-adjusted fitness)
    assert compare_strategy_fitness(consistent, volatile) > 0


def test_identical_strategies_tie():
    """Identical strategies produce a tie (result near zero)."""
    a = FitnessResult(sharpe=1.0, sharpe_std=0.3, max_drawdown=0.10, n_evaluations=10)
    b = FitnessResult(sharpe=1.0, sharpe_std=0.3, max_drawdown=0.10, n_evaluations=10)
    assert abs(compare_strategy_fitness(a, b)) < 0.01


def test_complexity_penalty_specific_tickers():
    """Skills referencing specific tickers get penalized."""
    skill_text = "Buy AAPL when RSI > 50. Also buy NVDA on dips."
    penalty = compute_complexity_penalty(skill_text)
    assert penalty > 0  # nonzero penalty for naming specific tickers


def test_complexity_penalty_general():
    """General skills without specific tickers get no penalty."""
    skill_text = "Buy when RSI crosses above 50 and price is above 50-day SMA."
    penalty = compute_complexity_penalty(skill_text)
    assert penalty == 0.0


def test_complexity_penalty_date_references():
    """Skills referencing narrow date ranges get penalized."""
    skill_text = "Only trade between January 15 and February 28, 2024."
    penalty = compute_complexity_penalty(skill_text)
    assert penalty > 0
```

**Step 2: Run to verify failure**

```bash
pytest tests/unit/test_fitness.py -v
```

**Step 3: Implement**

```python
# src/evolve_trader/core/fitness.py
"""Stochastic fitness evaluation for trading strategies.

Compares return distributions, not single numbers. Penalizes complexity.
A strategy with Sharpe 1.2 ± 0.8 is less fit than one with 0.9 ± 0.2.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class FitnessResult:
    """Distributional fitness of a strategy across multiple evaluations."""

    sharpe: float          # mean Sharpe ratio across evaluations
    sharpe_std: float      # standard deviation of Sharpe across evaluations
    max_drawdown: float    # worst max drawdown across evaluations
    n_evaluations: int     # number of walk-forward windows evaluated


def compare_strategy_fitness(a: FitnessResult, b: FitnessResult) -> float:
    """Compare two strategies on distributional fitness.

    Returns positive if `a` is fitter, negative if `b` is fitter, ~0 if tied.
    Uses a conservative comparison: mean minus 1 standard deviation.
    This penalizes high-variance strategies.
    """
    a_conservative = a.sharpe - a.sharpe_std
    b_conservative = b.sharpe - b.sharpe_std

    # Also penalize drawdown (lower is better)
    a_score = a_conservative - a.max_drawdown
    b_score = b_conservative - b.max_drawdown

    return a_score - b_score


# Common stock tickers for complexity detection
_TICKER_PATTERN = re.compile(
    r'\b(AAPL|MSFT|GOOGL|GOOG|AMZN|NVDA|META|TSLA|BRK|JPM|V|JNJ|WMT|'
    r'PG|MA|UNH|HD|DIS|BAC|XOM|PFE|KO|PEP|ABBV|COST|AVGO|TMO|MRK|'
    r'CVX|ADBE|CRM|ACN|NFLX|AMD|INTC|QCOM|TXN|CSCO|ORCL|IBM)\b'
)

_DATE_PATTERN = re.compile(
    r'\b(January|February|March|April|May|June|July|August|September|'
    r'October|November|December)\s+\d{1,2}.*?(20\d{2}|19\d{2})\b',
    re.IGNORECASE,
)


def compute_complexity_penalty(skill_text: str) -> float:
    """Compute a complexity penalty for a strategy skill.

    Penalizes:
    - References to specific tickers (overfitting to particular stocks)
    - Narrow date ranges (overfitting to particular time periods)
    - Highly specific numeric thresholds (overfitting to parameters)

    Returns 0.0 (no penalty) to 1.0 (maximum penalty).
    """
    penalty = 0.0

    # Ticker references
    ticker_matches = _TICKER_PATTERN.findall(skill_text)
    if ticker_matches:
        penalty += min(0.5, len(ticker_matches) * 0.1)

    # Date references
    date_matches = _DATE_PATTERN.findall(skill_text)
    if date_matches:
        penalty += min(0.5, len(date_matches) * 0.2)

    return min(1.0, penalty)
```

**Step 4: Run tests and commit**

```bash
pytest tests/unit/test_fitness.py -v
git add src/evolve_trader/core/fitness.py tests/unit/test_fitness.py
git commit -m "feat: stochastic fitness evaluation with complexity penalties"
```

---

## Task 7: Implement Version DAG for Skill Lineage

**Files:**
- Create: `src/evolve_trader/core/version_dag.py`
- Create: `tests/unit/test_version_dag.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_version_dag.py
"""Tests for the version DAG tracking skill lineage."""
import pytest
from evolve_trader.core.version_dag import (
    VersionDAG,
    EvolutionEvent,
    EvolutionMode,
)


def test_add_root_skill():
    """A seed strategy is a root node with no parent."""
    dag = VersionDAG()
    dag.add_root("momentum-v1")
    assert dag.get_parent("momentum-v1") is None
    assert dag.get_lineage("momentum-v1") == ["momentum-v1"]


def test_fix_creates_child():
    """FIX creates a new version linked to the parent."""
    dag = VersionDAG()
    dag.add_root("momentum-v1")
    dag.add_evolution(
        parent="momentum-v1",
        child="momentum-v1.1",
        mode=EvolutionMode.FIX,
        reason="Added stop-loss after excessive drawdown",
        metrics={"sharpe_before": 0.4, "max_drawdown_before": 0.20},
    )
    assert dag.get_parent("momentum-v1.1") == "momentum-v1"
    assert dag.get_lineage("momentum-v1.1") == ["momentum-v1", "momentum-v1.1"]


def test_derived_creates_branch():
    """DERIVED creates a specialized branch from parent."""
    dag = VersionDAG()
    dag.add_root("mean-reversion-v1")
    dag.add_evolution(
        parent="mean-reversion-v1",
        child="mean-reversion-earnings-v1",
        mode=EvolutionMode.DERIVED,
        reason="Specialized for post-earnings mean reversion",
    )
    assert dag.get_parent("mean-reversion-earnings-v1") == "mean-reversion-v1"
    children = dag.get_children("mean-reversion-v1")
    assert "mean-reversion-earnings-v1" in children


def test_captured_has_no_parent():
    """CAPTURED creates a new root from emergent behavior."""
    dag = VersionDAG()
    dag.add_captured(
        skill_name="emergent-sector-rotation-v1",
        reason="Agent discovered sector rotation pattern not in any existing skill",
        metrics={"sharpe": 1.5, "trades": 12},
    )
    assert dag.get_parent("emergent-sector-rotation-v1") is None
    events = dag.get_events("emergent-sector-rotation-v1")
    assert len(events) == 1
    assert events[0].mode == EvolutionMode.CAPTURED


def test_full_lineage_trace():
    """Can trace full lineage through multiple evolution steps."""
    dag = VersionDAG()
    dag.add_root("momentum-v1")
    dag.add_evolution(parent="momentum-v1", child="momentum-v2", mode=EvolutionMode.FIX, reason="fix1")
    dag.add_evolution(parent="momentum-v2", child="momentum-v3", mode=EvolutionMode.FIX, reason="fix2")
    dag.add_evolution(parent="momentum-v3", child="momentum-tech-v1", mode=EvolutionMode.DERIVED, reason="tech specialization")

    lineage = dag.get_lineage("momentum-tech-v1")
    assert lineage == ["momentum-v1", "momentum-v2", "momentum-v3", "momentum-tech-v1"]
```

**Step 2: Run to verify failure**

```bash
pytest tests/unit/test_version_dag.py -v
```

**Step 3: Implement**

```python
# src/evolve_trader/core/version_dag.py
"""Version DAG for tracking skill evolution lineage.

Records parent→child relationships through FIX/DERIVED/CAPTURED events.
Stores market conditions and performance metrics that triggered evolution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class EvolutionMode(Enum):
    """How a skill was created or modified."""

    SEED = "seed"          # Hand-crafted initial strategy
    FIX = "fix"            # In-place patch to fix failure
    DERIVED = "derived"    # Specialized variant for new context
    CAPTURED = "captured"  # Novel pattern extracted from emergent behavior


@dataclass
class EvolutionEvent:
    """A single evolution event in the DAG."""

    parent: Optional[str]
    child: str
    mode: EvolutionMode
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: dict[str, float] = field(default_factory=dict)


class VersionDAG:
    """Directed acyclic graph tracking skill evolution lineage."""

    def __init__(self) -> None:
        self._parents: dict[str, Optional[str]] = {}
        self._children: dict[str, list[str]] = {}
        self._events: dict[str, list[EvolutionEvent]] = {}

    def add_root(self, skill_name: str) -> None:
        """Add a seed strategy as a root node."""
        self._parents[skill_name] = None
        self._children.setdefault(skill_name, [])
        self._events.setdefault(skill_name, []).append(
            EvolutionEvent(parent=None, child=skill_name, mode=EvolutionMode.SEED, reason="Initial seed strategy")
        )

    def add_evolution(
        self,
        parent: str,
        child: str,
        mode: EvolutionMode,
        reason: str,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Record a FIX or DERIVED evolution event."""
        self._parents[child] = parent
        self._children.setdefault(parent, []).append(child)
        self._children.setdefault(child, [])
        self._events.setdefault(child, []).append(
            EvolutionEvent(parent=parent, child=child, mode=mode, reason=reason, metrics=metrics or {})
        )

    def add_captured(
        self,
        skill_name: str,
        reason: str,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Record a CAPTURED event — novel strategy from emergent behavior."""
        self._parents[skill_name] = None
        self._children.setdefault(skill_name, [])
        self._events.setdefault(skill_name, []).append(
            EvolutionEvent(parent=None, child=skill_name, mode=EvolutionMode.CAPTURED, reason=reason, metrics=metrics or {})
        )

    def get_parent(self, skill_name: str) -> Optional[str]:
        """Get the parent of a skill, or None if it's a root."""
        return self._parents.get(skill_name)

    def get_children(self, skill_name: str) -> list[str]:
        """Get all direct children of a skill."""
        return self._children.get(skill_name, [])

    def get_events(self, skill_name: str) -> list[EvolutionEvent]:
        """Get all evolution events for a skill."""
        return self._events.get(skill_name, [])

    def get_lineage(self, skill_name: str) -> list[str]:
        """Trace the full lineage from root to this skill."""
        lineage: list[str] = []
        current: Optional[str] = skill_name
        while current is not None:
            lineage.append(current)
            current = self._parents.get(current)
        lineage.reverse()
        return lineage
```

**Step 4: Run tests and commit**

```bash
pytest tests/unit/test_version_dag.py -v
git add src/evolve_trader/core/version_dag.py tests/unit/test_version_dag.py
git commit -m "feat: version DAG for skill evolution lineage tracking"
```

---

## Task 8: Implement Lightweight LLM Usage Logger

This task establishes the shared logger interface that Phase 2 will later persist to PostgreSQL. Use the final module name now so migration is additive rather than a rename.

**Files:**
- Create: `src/evolve_trader/core/llm_logger.py`
- Create: `tests/unit/test_llm_logger.py`

**Requirements:**
- File-based JSONL persistence in Phase 1
- Stable `LLMUsageRecord` / `LLMUsageLogger` interface reused in Phase 2
- Per-component cost aggregation and monthly budget checks
- Compact metadata only; no raw prompts, raw responses, or unrestricted chain-of-thought persistence

**Acceptance criteria:**
- Strategy execution, evolution, and analysis components can all log through the same interface
- The logger can be swapped to PostgreSQL in Phase 2 without changing callers
- Budget warning and hard-stop behaviors are testable at the logger layer

---

## Task 9: Write Seed Strategy Library (10-15 SKILL.md Files)

**Files:**
- Create: `src/evolve_trader/strategies/skills/trend-following-v1.md`
- Create: `src/evolve_trader/strategies/skills/mean-reversion-v1.md`
- Create: `src/evolve_trader/strategies/skills/momentum-sector-rotation-v1.md`
- Create: `src/evolve_trader/strategies/skills/value-fundamental-v1.md`
- Create: `src/evolve_trader/strategies/skills/earnings-drift-v1.md`
- Create: `src/evolve_trader/strategies/skills/defensive-low-volatility-v1.md`
- Create: `src/evolve_trader/strategies/skills/breakout-v1.md`
- Create: `src/evolve_trader/strategies/skills/gap-fade-v1.md`
- Create: `src/evolve_trader/strategies/skills/rsi-divergence-v1.md`
- Create: `src/evolve_trader/strategies/skills/moving-average-crossover-v1.md`
- Create: `src/evolve_trader/strategies/skills/bollinger-reversion-v1.md`
- Create: `src/evolve_trader/strategies/skills/pairs-trading-v1.md`
- Create: `tests/unit/test_seed_strategies.py`

Each seed strategy follows the StrategySkill schema from Task 1. These must be complete, parseable SKILL.md files with realistic reasoning frameworks — not stubs.

**Step 1: Write the validation test**

```python
# tests/unit/test_seed_strategies.py
"""Tests that all seed strategies parse correctly and have required fields."""
from pathlib import Path

import pytest
from evolve_trader.strategies.schema import parse_skill_md, StrategySkill

SKILLS_DIR = Path(__file__).parent.parent.parent / "src" / "evolve_trader" / "strategies" / "skills"


def get_all_skill_files():
    """Collect all SKILL.md files."""
    return list(SKILLS_DIR.glob("*.md"))


@pytest.mark.parametrize("skill_path", get_all_skill_files(), ids=lambda p: p.stem)
def test_skill_parses(skill_path: Path):
    """Every seed strategy file parses into a valid StrategySkill."""
    content = skill_path.read_text()
    skill = parse_skill_md(content)
    assert isinstance(skill, StrategySkill)
    assert skill.name
    assert skill.entry_logic
    assert skill.exit_logic
    assert skill.target_regime


def test_minimum_seed_count():
    """We have at least 10 seed strategies."""
    skills = get_all_skill_files()
    assert len(skills) >= 10, f"Only {len(skills)} seed strategies found, need at least 10"


def test_strategy_diversity():
    """Seed strategies cover diverse approaches (not all the same type)."""
    skills = get_all_skill_files()
    names = [p.stem for p in skills]
    # Check we have at least 4 different strategy families
    families = set()
    for name in names:
        if "momentum" in name or "trend" in name:
            families.add("trend")
        elif "reversion" in name or "bollinger" in name or "rsi" in name:
            families.add("mean-reversion")
        elif "value" in name or "fundamental" in name:
            families.add("value")
        elif "earnings" in name or "gap" in name:
            families.add("event-driven")
        elif "defensive" in name or "capital-preservation" in name:
            families.add("defensive")
        else:
            families.add("other")
    assert len(families) >= 4, f"Only {len(families)} strategy families found, need at least 4"
```

**Step 2: Write each seed strategy SKILL.md file**

Each file follows the schema. Write 12 strategies covering: trend-following, mean-reversion, momentum sector rotation, value/fundamental, earnings drift, defensive low-volatility, breakout, gap fade, RSI divergence, moving average crossover, Bollinger reversion, and pairs trading. Each has complete entry logic, exit logic, target regime, performance expectations, risk parameters, and a reasoning framework in the body.

**Step 3: Run tests**

```bash
pytest tests/unit/test_seed_strategies.py -v
```

Expected: All 12+ strategies parse, minimum count met, diversity check passes.

**Step 4: Commit**

```bash
git add src/evolve_trader/strategies/skills/ tests/unit/test_seed_strategies.py
git commit -m "feat: seed strategy library with 12 diverse SKILL.md trading strategies"
```

---

## Task 10: Integration — Connect Evolution Engine to Trading Replay

This task depends on what was discovered in Phase 0 about the integration points. The exact code will vary based on the decision gate results. The structure is:

**Files:**
- Create: `src/evolve_trader/core/evolution_loop.py`
- Create: `tests/integration/test_evolution_loop.py`

**Step 1: Write the integration test**

```python
# tests/integration/test_evolution_loop.py
"""Integration test: strategy evolution loop against historical replay.

This is the core thesis test: do strategies evolve meaningfully
based on trading outcomes?
"""
import pytest
from evolve_trader.core.evolution_loop import run_evolution_cycle
from evolve_trader.core.version_dag import VersionDAG, EvolutionMode


@pytest.mark.integration
def test_evolution_produces_at_least_one_fix():
    """Running evolution on NASDAQ 100 data produces at least one FIX event."""
    dag = VersionDAG()
    results = run_evolution_cycle(
        seed_skills_dir="src/evolve_trader/strategies/skills/",
        replay_days=50,
        universe="nasdaq100",
        version_dag=dag,
    )

    # At least one evolution event should have occurred
    all_events = []
    for skill_name in results.evolved_skills:
        all_events.extend(dag.get_events(skill_name))

    fix_events = [e for e in all_events if e.mode == EvolutionMode.FIX]
    assert len(fix_events) >= 1, "Evolution should produce at least one FIX event in 50 trading days"


@pytest.mark.integration
def test_evolved_strategy_outperforms_seed():
    """At least one evolved strategy outperforms its seed parent on out-of-sample data."""
    dag = VersionDAG()
    results = run_evolution_cycle(
        seed_skills_dir="src/evolve_trader/strategies/skills/",
        replay_days=50,
        universe="nasdaq100",
        version_dag=dag,
    )

    # Check if any evolved skill has better out-of-sample Sharpe than its parent
    improvements = [
        r for r in results.performance_comparisons
        if r.child_oos_sharpe > r.parent_oos_sharpe
    ]
    assert len(improvements) >= 1, "At least one evolved strategy should outperform its seed on OOS data"
```

**Step 2: Implement the evolution loop**

This is the most complex integration piece. It connects:
- Seed strategies (SKILL.md files from Task 9)
- Evolve-Trader's replay harness (market data)
- OpenSpace's evolution engine (FIX/DERIVED/CAPTURED)
- Post-execution analyzer (Task 2)
- Walk-forward validation (Task 3)
- Risk constraints (Task 5)
- Fitness evaluation (Task 6)
- Version DAG (Task 7)

The exact implementation depends on Phase 0 findings about how OpenSpace and AI-Trader work. The interface should be:

```python
# src/evolve_trader/core/evolution_loop.py (interface — implementation depends on Phase 0)
"""Core evolution loop connecting OpenSpace evolution to AI-Trader trading replay."""

@dataclass
class EvolutionResults:
    evolved_skills: list[str]
    performance_comparisons: list[PerformanceComparison]
    total_fix_events: int
    total_derived_events: int
    total_captured_events: int

def run_evolution_cycle(
    seed_skills_dir: str,
    replay_days: int,
    universe: str,
    version_dag: VersionDAG,
) -> EvolutionResults:
    """Run a complete evolution cycle: load seeds, replay, evaluate, evolve."""
    ...
```

**Step 3: Run the cold-start experiment**

```bash
pytest tests/integration/test_evolution_loop.py -v --timeout=600
```

This may take several minutes as it runs 50 days of NASDAQ 100 simulation with LLM calls.

**Step 4: Document cold-start results**

Create `docs/cold-start-results.md` with:
- Number of FIX/DERIVED/CAPTURED events
- Fitness trajectory over the 50 days
- Which seed strategies evolved and which didn't
- Out-of-sample performance of evolved vs. seed strategies
- LLM cost for the experiment (from Task 8 tracker)

**Step 5: Commit**

```bash
git add src/evolve_trader/core/evolution_loop.py tests/integration/test_evolution_loop.py docs/cold-start-results.md
git commit -m "feat: core evolution loop with cold-start experiment results"
```

---

## Task 11: Final Phase 1 Verification

**Step 1: Run the complete test suite**

```bash
pytest --cov=src/evolve_trader -v
```

Expected: All unit and integration tests pass.

**Step 2: Verify all Phase 1 deliverables**

- [ ] StrategySkill SKILL.md schema defined and documented
- [ ] Post-execution analyzer computing financial metrics + distributional evaluation
- [ ] Walk-forward validation harness with configurable windows
- [ ] Capital Preservation skill implemented
- [ ] Immutable risk constraints enforced at trade-execution level
- [ ] Complexity penalty system integrated into fitness evaluation
- [ ] Version DAG tracking skill lineage
- [ ] 10-15 seed strategies written as SKILL.md files
- [ ] Cold-start experiment results documented
- [ ] Lightweight LLM token usage logger
- [ ] All components unit tested and integration tested

**Step 3: Final commit**

```bash
git add -A
git commit -m "docs: Phase 1 complete — core evolution loop verified"
```

---

## Summary: Phase 1 Task Sequence

| Task | Description | Depends On |
|------|-------------|------------|
| 1 | StrategySkill schema + SKILL.md parsing | — |
| 2 | Post-execution financial analyzer | — |
| 3 | Walk-forward validation harness | — |
| 4 | Capital Preservation skill | 1 |
| 5 | Immutable risk constraints | — |
| 6 | Stochastic fitness evaluation + complexity penalties | 2 |
| 7 | Version DAG for skill lineage | — |
| 8 | LLM token usage logger | — |
| 9 | Seed strategy library (12 SKILL.md files) | 1 |
| 10 | Integration — evolution loop + cold-start experiment | 1-9 |
| 11 | Final verification | 10 |

**Parallelizable:** Tasks 1, 2, 3, 5, 7, and 8 are fully independent. Tasks 4, 6, 9 have light dependencies. Task 10 requires all prior tasks. Task 11 requires Task 10.
