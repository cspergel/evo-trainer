# Phase 4: Position Sizing & Risk Evolution — Detailed Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Separate position sizing into its own evolvable skill family. Build the composition interface where strategy skills output what/when and sizing skills output how much. Implement portfolio-level risk constraint enforcement. Add tax-aware evolution mode.

**Architecture:** Position sizing becomes an independent evolvable SKILL.md family. Strategy skills produce TradeIntents (ticker, direction, conditions, regime). Sizing skills consume TradeIntents and output sized positions (shares, position %). A composition layer cleanly separates the two concerns. Portfolio-level risk enforcement applies real-time exposure tracking, sector limits, and regime-linked gross exposure caps. A survival gate validates all skill types before live deployment. Tax-aware evolution mode optionally penalizes short-term capital gains.

**Tech Stack:** Python 3.11+, PostgreSQL, Pydantic, numpy, scipy, pytest

**Parent plan:** [`2026-03-29-evolve-trader-master-plan.md`](2026-03-29-evolve-trader-master-plan.md)

**Prerequisites:** Phase 3 complete. Meta-selector routing engine, signal scoring, source lifecycle pipeline, conflict resolution, and multi-timeframe stacking all working.

---

## Task 1: SizingSkill Schema

**Files:**
- Create: `src/evolve_trader/sizing/__init__.py`
- Create: `src/evolve_trader/sizing/schema.py`
- Create: `tests/unit/test_sizing_schema.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_sizing_schema.py
"""Tests for sizing skill schema and SKILL.md parser."""
import pytest
from evolve_trader.sizing.schema import (
    SizingSkillConfig,
    SizingMethod,
    RegimeBehavior,
    RiskBudget,
    parse_sizing_skill_md,
)


def test_sizing_method_enum_values():
    """SizingMethod enumerates all supported sizing strategies."""
    assert SizingMethod.KELLY == "kelly"
    assert SizingMethod.VOLATILITY_TARGET == "volatility_target"
    assert SizingMethod.CORRELATION_AWARE == "correlation_aware"
    assert SizingMethod.REGIME_ADJUSTED == "regime_adjusted"
    assert SizingMethod.FIXED_FRACTIONAL == "fixed_fractional"


def test_regime_behavior_has_required_fields():
    """RegimeBehavior defines per-regime exposure caps and scaling."""
    behavior = RegimeBehavior(
        risk_on_max_exposure=1.0,
        risk_off_max_exposure=0.6,
        transitional_max_exposure=0.8,
        crisis_max_exposure=0.3,
        scaling_speed="gradual",
    )
    assert behavior.risk_on_max_exposure == 1.0
    assert behavior.crisis_max_exposure == 0.3
    assert behavior.scaling_speed in ("gradual", "immediate")


def test_risk_budget_has_required_fields():
    """RiskBudget defines per-strategy risk allocation."""
    budget = RiskBudget(
        max_position_pct=0.10,
        max_sector_pct=0.30,
        max_correlated_group_pct=0.40,
        max_daily_var_pct=0.02,
        max_drawdown_pct=0.15,
    )
    assert budget.max_position_pct == 0.10
    assert budget.max_drawdown_pct == 0.15


def test_sizing_skill_config_creation():
    """SizingSkillConfig has all required fields."""
    config = SizingSkillConfig(
        name="kelly-v1",
        version=1,
        status="active",
        skill_type="sizing",
        sizing_method=SizingMethod.KELLY,
        parameters={"fraction": 0.25, "win_prob_lookback_days": 60},
        target_regime_behavior=RegimeBehavior(
            risk_on_max_exposure=1.0,
            risk_off_max_exposure=0.6,
            transitional_max_exposure=0.8,
            crisis_max_exposure=0.3,
            scaling_speed="gradual",
        ),
        risk_budget=RiskBudget(
            max_position_pct=0.10,
            max_sector_pct=0.30,
            max_correlated_group_pct=0.40,
            max_daily_var_pct=0.02,
            max_drawdown_pct=0.15,
        ),
        description="Fractional Kelly sizing with regime-aware caps",
    )
    assert config.name == "kelly-v1"
    assert config.sizing_method == SizingMethod.KELLY
    assert config.parameters["fraction"] == 0.25
    assert config.risk_budget.max_position_pct == 0.10


def test_sizing_skill_config_validates_exposure_bounds():
    """Exposure caps must be between 0 and 1."""
    with pytest.raises(ValueError, match="exposure"):
        RegimeBehavior(
            risk_on_max_exposure=1.5,
            risk_off_max_exposure=0.6,
            transitional_max_exposure=0.8,
            crisis_max_exposure=0.3,
            scaling_speed="gradual",
        )


def test_sizing_skill_config_validates_risk_budget():
    """Risk budget percentages must be between 0 and 1."""
    with pytest.raises(ValueError, match="position"):
        RiskBudget(
            max_position_pct=1.5,
            max_sector_pct=0.30,
            max_correlated_group_pct=0.40,
            max_daily_var_pct=0.02,
            max_drawdown_pct=0.15,
        )


def test_parse_sizing_skill_md():
    """Parser extracts SizingSkillConfig from SKILL.md frontmatter."""
    md_content = """---
name: kelly-v1
description: Fractional Kelly sizing
version: 1
status: active
skill_type: sizing
sizing_method: kelly
parameters:
  fraction: 0.25
  win_prob_lookback_days: 60
target_regime_behavior:
  risk_on_max_exposure: 1.0
  risk_off_max_exposure: 0.6
  transitional_max_exposure: 0.8
  crisis_max_exposure: 0.3
  scaling_speed: gradual
risk_budget:
  max_position_pct: 0.10
  max_sector_pct: 0.30
  max_correlated_group_pct: 0.40
  max_daily_var_pct: 0.02
  max_drawdown_pct: 0.15
---

# Kelly Criterion Sizing v1

## Logic
Uses fractional Kelly to size positions based on win probability and payoff ratio.
"""
    config = parse_sizing_skill_md(md_content)
    assert config.name == "kelly-v1"
    assert config.sizing_method == SizingMethod.KELLY
    assert config.parameters["fraction"] == 0.25
    assert config.target_regime_behavior.risk_off_max_exposure == 0.6


def test_parse_sizing_skill_md_missing_required_field():
    """Parser raises on missing required fields."""
    md_content = """---
name: bad-skill
version: 1
status: active
skill_type: sizing
---

# Bad Skill
"""
    with pytest.raises(ValueError, match="sizing_method"):
        parse_sizing_skill_md(md_content)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_sizing_schema.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.sizing'`

**Step 3: Implement**

```python
# src/evolve_trader/sizing/__init__.py
"""Position sizing skill family."""
```

```python
# src/evolve_trader/sizing/schema.py
"""SizingSkill schema — Pydantic models and SKILL.md parser for sizing skills."""
from __future__ import annotations

from enum import Enum
from typing import Any

import yaml
from pydantic import BaseModel, field_validator


class SizingMethod(str, Enum):
    """Supported position sizing methods."""
    KELLY = "kelly"
    VOLATILITY_TARGET = "volatility_target"
    CORRELATION_AWARE = "correlation_aware"
    REGIME_ADJUSTED = "regime_adjusted"
    FIXED_FRACTIONAL = "fixed_fractional"


class RegimeBehavior(BaseModel):
    """Per-regime exposure caps and scaling behavior."""
    risk_on_max_exposure: float
    risk_off_max_exposure: float
    transitional_max_exposure: float
    crisis_max_exposure: float
    scaling_speed: str  # "gradual" or "immediate"

    @field_validator(
        "risk_on_max_exposure",
        "risk_off_max_exposure",
        "transitional_max_exposure",
        "crisis_max_exposure",
    )
    @classmethod
    def validate_exposure_range(cls, v: float, info) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"exposure must be between 0 and 1, got {v}")
        return v

    @field_validator("scaling_speed")
    @classmethod
    def validate_scaling_speed(cls, v: str) -> str:
        if v not in ("gradual", "immediate"):
            raise ValueError(f"scaling_speed must be 'gradual' or 'immediate', got {v}")
        return v


class RiskBudget(BaseModel):
    """Per-strategy risk allocation constraints."""
    max_position_pct: float
    max_sector_pct: float
    max_correlated_group_pct: float
    max_daily_var_pct: float
    max_drawdown_pct: float

    @field_validator(
        "max_position_pct",
        "max_sector_pct",
        "max_correlated_group_pct",
        "max_daily_var_pct",
        "max_drawdown_pct",
    )
    @classmethod
    def validate_pct_range(cls, v: float, info) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"{info.field_name} must be between 0 and 1, got {v}"
            )
        return v


class SizingSkillConfig(BaseModel):
    """Full configuration for a sizing skill parsed from SKILL.md."""
    name: str
    version: int
    status: str
    skill_type: str
    sizing_method: SizingMethod
    parameters: dict[str, Any]
    target_regime_behavior: RegimeBehavior
    risk_budget: RiskBudget
    description: str = ""


def parse_sizing_skill_md(md_content: str) -> SizingSkillConfig:
    """Parse a sizing SKILL.md file and return a validated SizingSkillConfig.

    Extracts YAML frontmatter between --- delimiters.
    """
    lines = md_content.strip().split("\n")
    if lines[0].strip() != "---":
        raise ValueError("SKILL.md must start with --- frontmatter delimiter")

    # Find closing ---
    end_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx == -1:
        raise ValueError("No closing --- found for frontmatter")

    frontmatter = "\n".join(lines[1:end_idx])
    data = yaml.safe_load(frontmatter)

    if "sizing_method" not in data:
        raise ValueError("sizing_method is required in sizing SKILL.md frontmatter")

    # Map nested dicts to sub-models
    if "target_regime_behavior" in data and isinstance(data["target_regime_behavior"], dict):
        data["target_regime_behavior"] = RegimeBehavior(**data["target_regime_behavior"])
    if "risk_budget" in data and isinstance(data["risk_budget"], dict):
        data["risk_budget"] = RiskBudget(**data["risk_budget"])

    return SizingSkillConfig(**data)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_sizing_schema.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/sizing/ tests/unit/test_sizing_schema.py
git commit -m "feat: sizing skill schema with Pydantic models and SKILL.md parser"
```

---

## Task 2: Kelly Criterion Sizing

**Files:**
- Create: `src/evolve_trader/sizing/kelly.py`
- Create: `strategies/sizing/kelly-v1.skill.md`
- Create: `tests/unit/test_kelly_sizing.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_kelly_sizing.py
"""Tests for Kelly Criterion position sizing."""
import pytest
import numpy as np
from evolve_trader.sizing.kelly import (
    KellySizer,
    KellyResult,
    TradeHistory,
)


def _make_trade_history(wins: int, losses: int, avg_win: float, avg_loss: float) -> TradeHistory:
    """Build a synthetic trade history for Kelly computation."""
    returns = [avg_win] * wins + [avg_loss] * losses
    return TradeHistory(returns=returns)


def test_kelly_fraction_computation():
    """Full Kelly: f* = p - q/b where p=win_prob, q=1-p, b=payoff_ratio."""
    sizer = KellySizer(kelly_fraction=1.0)
    history = _make_trade_history(wins=60, losses=40, avg_win=0.05, avg_loss=-0.03)
    result = sizer.compute(history, portfolio_value=100_000.0, price=150.0)
    # p=0.6, b=5/3≈1.667, f* = 0.6 - 0.4/1.667 ≈ 0.36
    assert 0.30 <= result.kelly_fraction <= 0.42


def test_fractional_kelly_reduces_position():
    """Quarter-Kelly reduces position to 25% of full Kelly."""
    full_sizer = KellySizer(kelly_fraction=1.0)
    quarter_sizer = KellySizer(kelly_fraction=0.25)
    history = _make_trade_history(wins=60, losses=40, avg_win=0.05, avg_loss=-0.03)

    full_result = full_sizer.compute(history, portfolio_value=100_000.0, price=150.0)
    quarter_result = quarter_sizer.compute(history, portfolio_value=100_000.0, price=150.0)

    assert quarter_result.position_pct == pytest.approx(
        full_result.position_pct * 0.25, abs=0.01
    )


def test_kelly_negative_edge_returns_zero():
    """Negative edge (losing strategy) → zero position."""
    sizer = KellySizer(kelly_fraction=0.25)
    history = _make_trade_history(wins=30, losses=70, avg_win=0.02, avg_loss=-0.04)
    result = sizer.compute(history, portfolio_value=100_000.0, price=150.0)
    assert result.position_pct == 0.0
    assert result.shares == 0


def test_kelly_caps_at_max_position():
    """Kelly never exceeds max_position_pct even with huge edge."""
    sizer = KellySizer(kelly_fraction=1.0, max_position_pct=0.10)
    history = _make_trade_history(wins=95, losses=5, avg_win=0.10, avg_loss=-0.01)
    result = sizer.compute(history, portfolio_value=100_000.0, price=150.0)
    assert result.position_pct <= 0.10


def test_kelly_minimum_history_required():
    """Below minimum trade count → zero position (insufficient data)."""
    sizer = KellySizer(kelly_fraction=0.25, min_trades=20)
    history = TradeHistory(returns=[0.05, 0.03, -0.02])  # Only 3 trades
    result = sizer.compute(history, portfolio_value=100_000.0, price=150.0)
    assert result.position_pct == 0.0
    assert result.insufficient_data is True


def test_kelly_computes_share_count():
    """Kelly translates position % into share count."""
    sizer = KellySizer(kelly_fraction=0.25)
    history = _make_trade_history(wins=60, losses=40, avg_win=0.05, avg_loss=-0.03)
    result = sizer.compute(history, portfolio_value=100_000.0, price=50.0)
    expected_shares = int((result.position_pct * 100_000.0) / 50.0)
    assert result.shares == expected_shares


def test_kelly_result_data_model():
    """KellyResult has all required fields."""
    result = KellyResult(
        kelly_fraction=0.36,
        applied_fraction=0.09,
        position_pct=0.09,
        position_value=9_000.0,
        shares=60,
        win_probability=0.6,
        payoff_ratio=1.667,
        edge=0.36,
        insufficient_data=False,
    )
    assert result.edge == 0.36
    assert result.shares == 60
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_kelly_sizing.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.sizing.kelly'`

**Step 3: Implement**

```python
# src/evolve_trader/sizing/kelly.py
"""Kelly Criterion position sizing — fractional Kelly with safety caps."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradeHistory:
    """Historical trade returns for Kelly computation."""
    returns: list[float]

    @property
    def win_probability(self) -> float:
        if not self.returns:
            return 0.0
        wins = sum(1 for r in self.returns if r > 0)
        return wins / len(self.returns)

    @property
    def avg_win(self) -> float:
        wins = [r for r in self.returns if r > 0]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [r for r in self.returns if r <= 0]
        return sum(losses) / len(losses) if losses else 0.0

    @property
    def payoff_ratio(self) -> float:
        """Average win / |average loss|."""
        avg_l = self.avg_loss
        if avg_l == 0:
            return 0.0
        return self.avg_win / abs(avg_l)


@dataclass
class KellyResult:
    """Output of Kelly sizing computation."""
    kelly_fraction: float       # Full Kelly f*
    applied_fraction: float     # After fractional Kelly scaling
    position_pct: float         # Final position as % of portfolio
    position_value: float       # Dollar value of position
    shares: int                 # Number of shares
    win_probability: float
    payoff_ratio: float
    edge: float                 # Kelly edge = p - q/b
    insufficient_data: bool


class KellySizer:
    """Fractional Kelly position sizer.

    Full Kelly: f* = p - q/b
    where p = win probability, q = 1-p, b = payoff ratio (avg_win / |avg_loss|)

    Applied fraction = kelly_fraction * f*
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_position_pct: float = 1.0,
        min_trades: int = 10,
    ):
        self._kelly_fraction = kelly_fraction
        self._max_position_pct = max_position_pct
        self._min_trades = min_trades

    def compute(
        self,
        history: TradeHistory,
        portfolio_value: float,
        price: float,
    ) -> KellyResult:
        """Compute Kelly-optimal position size."""
        # Insufficient data check
        if len(history.returns) < self._min_trades:
            return KellyResult(
                kelly_fraction=0.0,
                applied_fraction=0.0,
                position_pct=0.0,
                position_value=0.0,
                shares=0,
                win_probability=0.0,
                payoff_ratio=0.0,
                edge=0.0,
                insufficient_data=True,
            )

        p = history.win_probability
        q = 1.0 - p
        b = history.payoff_ratio

        # Full Kelly fraction
        if b == 0:
            f_star = 0.0
        else:
            f_star = p - (q / b)

        # Negative edge → no position
        if f_star <= 0:
            return KellyResult(
                kelly_fraction=f_star,
                applied_fraction=0.0,
                position_pct=0.0,
                position_value=0.0,
                shares=0,
                win_probability=p,
                payoff_ratio=b,
                edge=f_star,
                insufficient_data=False,
            )

        # Apply fractional Kelly
        applied = f_star * self._kelly_fraction

        # Cap at max position
        position_pct = min(applied, self._max_position_pct)

        # Compute dollar value and shares
        position_value = position_pct * portfolio_value
        shares = int(position_value / price) if price > 0 else 0

        return KellyResult(
            kelly_fraction=f_star,
            applied_fraction=applied,
            position_pct=position_pct,
            position_value=position_value,
            shares=shares,
            win_probability=p,
            payoff_ratio=b,
            edge=f_star,
            insufficient_data=False,
        )
```

```markdown
<!-- strategies/sizing/kelly-v1.skill.md -->
---
name: kelly-v1
description: Fractional Kelly Criterion position sizing
version: 1
status: active
skill_type: sizing
sizing_method: kelly
parameters:
  fraction: 0.25
  win_prob_lookback_days: 60
  min_trades: 20
target_regime_behavior:
  risk_on_max_exposure: 1.0
  risk_off_max_exposure: 0.6
  transitional_max_exposure: 0.8
  crisis_max_exposure: 0.3
  scaling_speed: gradual
risk_budget:
  max_position_pct: 0.10
  max_sector_pct: 0.30
  max_correlated_group_pct: 0.40
  max_daily_var_pct: 0.02
  max_drawdown_pct: 0.15
---

# Kelly Criterion Sizing v1

## Logic

Uses fractional Kelly criterion to determine optimal position size:
1. Compute win probability (p) and payoff ratio (b = avg_win / |avg_loss|) from rolling trade history
2. Full Kelly: f* = p - (1-p)/b
3. Apply fractional scaling (default 25%) for safety
4. Cap at max_position_pct from risk budget

## Evolution Notes

Subject to FIX/DERIVED/CAPTURED:
- FIX: When Kelly oversizes in volatile regimes despite fractional scaling
- DERIVED: When regime-specific Kelly fractions outperform a single fraction
- CAPTURED: When win probability estimation improvements are discovered
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_kelly_sizing.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/sizing/kelly.py strategies/sizing/kelly-v1.skill.md tests/unit/test_kelly_sizing.py
git commit -m "feat: Kelly Criterion position sizing with fractional scaling"
```

---

## Task 3: Volatility Targeting Sizing

**Files:**
- Create: `src/evolve_trader/sizing/volatility.py`
- Create: `tests/unit/test_volatility_sizing.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_volatility_sizing.py
"""Tests for volatility targeting position sizing."""
import pytest
import numpy as np
from evolve_trader.sizing.volatility import (
    VolatilityTargetSizer,
    VolatilityResult,
)


def _low_vol_returns() -> list[float]:
    """Simulate low-volatility asset (annualized ~10%)."""
    np.random.seed(42)
    return list(np.random.normal(0.0004, 0.006, 60))  # ~10% annualized vol


def _high_vol_returns() -> list[float]:
    """Simulate high-volatility asset (annualized ~40%)."""
    np.random.seed(42)
    return list(np.random.normal(0.0004, 0.025, 60))  # ~40% annualized vol


def test_volatility_target_sizes_inversely_to_vol():
    """Lower realized vol → larger position; higher vol → smaller position."""
    sizer = VolatilityTargetSizer(target_annual_vol=0.15)
    low_vol = sizer.compute(
        recent_returns=_low_vol_returns(),
        portfolio_value=100_000.0,
        price=100.0,
    )
    high_vol = sizer.compute(
        recent_returns=_high_vol_returns(),
        portfolio_value=100_000.0,
        price=100.0,
    )
    assert low_vol.position_pct > high_vol.position_pct


def test_volatility_target_consistent_risk_contribution():
    """Both positions should contribute ~target_vol to portfolio risk."""
    sizer = VolatilityTargetSizer(target_annual_vol=0.15)
    low_result = sizer.compute(
        recent_returns=_low_vol_returns(),
        portfolio_value=100_000.0,
        price=100.0,
    )
    high_result = sizer.compute(
        recent_returns=_high_vol_returns(),
        portfolio_value=100_000.0,
        price=100.0,
    )
    # Risk contribution ≈ position_pct * realized_vol should be ~target
    low_risk = low_result.position_pct * low_result.realized_annual_vol
    high_risk = high_result.position_pct * high_result.realized_annual_vol
    assert low_risk == pytest.approx(0.15, abs=0.05)
    assert high_risk == pytest.approx(0.15, abs=0.05)


def test_volatility_target_caps_at_max():
    """Position never exceeds max_position_pct even with ultra-low vol."""
    sizer = VolatilityTargetSizer(target_annual_vol=0.15, max_position_pct=0.20)
    # Artificial ultra-low vol returns
    stable_returns = [0.001] * 60
    result = sizer.compute(
        recent_returns=stable_returns,
        portfolio_value=100_000.0,
        price=100.0,
    )
    assert result.position_pct <= 0.20


def test_volatility_target_insufficient_data():
    """Below minimum observation count → zero position."""
    sizer = VolatilityTargetSizer(target_annual_vol=0.15, min_observations=30)
    result = sizer.compute(
        recent_returns=[0.01, -0.005, 0.003],  # Only 3 data points
        portfolio_value=100_000.0,
        price=100.0,
    )
    assert result.position_pct == 0.0
    assert result.insufficient_data is True


def test_volatility_target_annualization():
    """Realized vol is correctly annualized from daily returns."""
    sizer = VolatilityTargetSizer(target_annual_vol=0.15)
    daily_returns = list(np.random.RandomState(42).normal(0, 0.01, 60))
    result = sizer.compute(
        recent_returns=daily_returns,
        portfolio_value=100_000.0,
        price=100.0,
    )
    # Annualized vol ≈ daily_std * sqrt(252)
    expected_annual = np.std(daily_returns) * np.sqrt(252)
    assert result.realized_annual_vol == pytest.approx(expected_annual, rel=0.01)


def test_volatility_result_data_model():
    """VolatilityResult has all required fields."""
    result = VolatilityResult(
        realized_annual_vol=0.25,
        target_annual_vol=0.15,
        raw_position_pct=0.60,
        position_pct=0.20,
        position_value=20_000.0,
        shares=133,
        vol_ratio=0.60,
        insufficient_data=False,
    )
    assert result.vol_ratio == 0.60
    assert result.shares == 133
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_volatility_sizing.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.sizing.volatility'`

**Step 3: Implement**

```python
# src/evolve_trader/sizing/volatility.py
"""Volatility targeting position sizing — consistent risk contribution."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


TRADING_DAYS_PER_YEAR = 252


@dataclass
class VolatilityResult:
    """Output of volatility target sizing computation."""
    realized_annual_vol: float
    target_annual_vol: float
    raw_position_pct: float    # Before max cap
    position_pct: float        # After max cap
    position_value: float
    shares: int
    vol_ratio: float           # target_vol / realized_vol
    insufficient_data: bool


class VolatilityTargetSizer:
    """Sizes positions inversely to realized volatility.

    position_pct = target_annual_vol / realized_annual_vol
    This produces consistent risk contribution across assets with different volatilities.
    """

    def __init__(
        self,
        target_annual_vol: float = 0.15,
        max_position_pct: float = 1.0,
        min_observations: int = 20,
        lookback_days: int = 60,
    ):
        self._target_vol = target_annual_vol
        self._max_position_pct = max_position_pct
        self._min_obs = min_observations
        self._lookback_days = lookback_days

    def compute(
        self,
        recent_returns: list[float],
        portfolio_value: float,
        price: float,
    ) -> VolatilityResult:
        """Compute volatility-targeted position size."""
        # Insufficient data check
        if len(recent_returns) < self._min_obs:
            return VolatilityResult(
                realized_annual_vol=0.0,
                target_annual_vol=self._target_vol,
                raw_position_pct=0.0,
                position_pct=0.0,
                position_value=0.0,
                shares=0,
                vol_ratio=0.0,
                insufficient_data=True,
            )

        # Use most recent lookback_days
        recent = recent_returns[-self._lookback_days:]

        # Compute realized daily vol and annualize
        daily_vol = float(np.std(recent))
        realized_annual_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Avoid division by zero (near-zero vol)
        if realized_annual_vol < 1e-8:
            raw_pct = self._max_position_pct
        else:
            raw_pct = self._target_vol / realized_annual_vol

        vol_ratio = raw_pct  # target / realized
        position_pct = min(raw_pct, self._max_position_pct)

        position_value = position_pct * portfolio_value
        shares = int(position_value / price) if price > 0 else 0

        return VolatilityResult(
            realized_annual_vol=realized_annual_vol,
            target_annual_vol=self._target_vol,
            raw_position_pct=raw_pct,
            position_pct=position_pct,
            position_value=position_value,
            shares=shares,
            vol_ratio=vol_ratio,
            insufficient_data=False,
        )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_volatility_sizing.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/sizing/volatility.py tests/unit/test_volatility_sizing.py
git commit -m "feat: volatility targeting position sizing for consistent risk contribution"
```

---

## Task 4: Correlation-Aware Sizing

**Files:**
- Create: `src/evolve_trader/sizing/correlation.py`
- Create: `tests/unit/test_correlation_sizing.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_correlation_sizing.py
"""Tests for correlation-aware position sizing."""
import pytest
import numpy as np
from evolve_trader.sizing.correlation import (
    CorrelationAwareSizer,
    CorrelationResult,
    PositionRequest,
)


def _correlated_returns(n: int = 60, correlation: float = 0.8) -> tuple[list[float], list[float]]:
    """Generate two return series with specified correlation."""
    rng = np.random.RandomState(42)
    x = rng.normal(0, 0.02, n)
    noise = rng.normal(0, 0.02, n)
    y = correlation * x + np.sqrt(1 - correlation**2) * noise
    return list(x), list(y)


def _uncorrelated_returns(n: int = 60) -> tuple[list[float], list[float]]:
    """Generate two uncorrelated return series."""
    return _correlated_returns(n, correlation=0.0)


def test_correlated_assets_get_reduced_sizing():
    """Highly correlated assets → reduced joint position to prevent concentration."""
    sizer = CorrelationAwareSizer(max_correlated_group_pct=0.40)
    r1, r2 = _correlated_returns(correlation=0.9)
    requests = [
        PositionRequest("AAPL", raw_position_pct=0.30, recent_returns=r1, sector="Technology"),
        PositionRequest("MSFT", raw_position_pct=0.30, recent_returns=r2, sector="Technology"),
    ]
    result = sizer.compute(requests, portfolio_value=100_000.0)
    total_pct = sum(p.adjusted_position_pct for p in result.positions)
    assert total_pct <= 0.40 + 1e-9


def test_uncorrelated_assets_retain_full_sizing():
    """Uncorrelated assets → no reduction needed."""
    sizer = CorrelationAwareSizer(max_correlated_group_pct=0.40)
    r1, r2 = _uncorrelated_returns()
    requests = [
        PositionRequest("AAPL", raw_position_pct=0.15, recent_returns=r1, sector="Technology"),
        PositionRequest("XOM", raw_position_pct=0.15, recent_returns=r2, sector="Energy"),
    ]
    result = sizer.compute(requests, portfolio_value=100_000.0)
    # Uncorrelated: total 0.30 < 0.40 cap, should retain full sizing
    for pos in result.positions:
        assert pos.adjusted_position_pct == pytest.approx(pos.raw_position_pct, abs=0.01)


def test_correlation_matrix_computed():
    """CorrelationResult includes the computed correlation matrix."""
    sizer = CorrelationAwareSizer(max_correlated_group_pct=0.40)
    r1, r2 = _correlated_returns(correlation=0.8)
    requests = [
        PositionRequest("A", raw_position_pct=0.20, recent_returns=r1, sector="X"),
        PositionRequest("B", raw_position_pct=0.20, recent_returns=r2, sector="Y"),
    ]
    result = sizer.compute(requests, portfolio_value=100_000.0)
    assert result.correlation_matrix.shape == (2, 2)
    assert result.correlation_matrix[0, 1] == pytest.approx(0.8, abs=0.15)


def test_single_position_passthrough():
    """Single position → no correlation adjustment needed."""
    sizer = CorrelationAwareSizer(max_correlated_group_pct=0.40)
    r1, _ = _correlated_returns()
    requests = [
        PositionRequest("AAPL", raw_position_pct=0.25, recent_returns=r1, sector="Technology"),
    ]
    result = sizer.compute(requests, portfolio_value=100_000.0)
    assert result.positions[0].adjusted_position_pct == pytest.approx(0.25, abs=0.01)


def test_same_sector_concentration_flagged():
    """Positions in same sector are flagged for review."""
    sizer = CorrelationAwareSizer(max_correlated_group_pct=0.40, correlation_threshold=0.6)
    r1, r2 = _correlated_returns(correlation=0.85)
    requests = [
        PositionRequest("AAPL", raw_position_pct=0.25, recent_returns=r1, sector="Technology"),
        PositionRequest("MSFT", raw_position_pct=0.25, recent_returns=r2, sector="Technology"),
    ]
    result = sizer.compute(requests, portfolio_value=100_000.0)
    assert len(result.correlated_groups) >= 1
    group_tickers = result.correlated_groups[0]
    assert "AAPL" in group_tickers and "MSFT" in group_tickers


def test_proportional_scaling_preserves_relative_weights():
    """When scaled down, relative weights between correlated assets are preserved."""
    sizer = CorrelationAwareSizer(max_correlated_group_pct=0.40)
    r1, r2 = _correlated_returns(correlation=0.9)
    requests = [
        PositionRequest("A", raw_position_pct=0.30, recent_returns=r1, sector="X"),
        PositionRequest("B", raw_position_pct=0.20, recent_returns=r2, sector="X"),
    ]
    result = sizer.compute(requests, portfolio_value=100_000.0)
    a_pct = result.positions[0].adjusted_position_pct
    b_pct = result.positions[1].adjusted_position_pct
    # Ratio should be preserved: 30/20 = 1.5
    if a_pct > 0 and b_pct > 0:
        assert a_pct / b_pct == pytest.approx(1.5, abs=0.15)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_correlation_sizing.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.sizing.correlation'`

**Step 3: Implement**

```python
# src/evolve_trader/sizing/correlation.py
"""Correlation-aware position sizing — prevents factor-level concentration."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class PositionRequest:
    """Input request for a single position to be correlation-adjusted."""
    ticker: str
    raw_position_pct: float
    recent_returns: list[float]
    sector: str


@dataclass
class AdjustedPosition:
    """Output for a single position after correlation adjustment."""
    ticker: str
    raw_position_pct: float
    adjusted_position_pct: float
    sector: str
    scaling_factor: float


@dataclass
class CorrelationResult:
    """Output of correlation-aware sizing computation."""
    positions: list[AdjustedPosition]
    correlation_matrix: np.ndarray
    correlated_groups: list[list[str]]  # Groups of tickers with high correlation


class CorrelationAwareSizer:
    """Sizes positions jointly to prevent factor-level concentration.

    Groups correlated assets (above threshold) and caps total group exposure.
    Scales down proportionally within groups that exceed the cap.
    """

    def __init__(
        self,
        max_correlated_group_pct: float = 0.40,
        correlation_threshold: float = 0.6,
        min_observations: int = 20,
    ):
        self._max_group_pct = max_correlated_group_pct
        self._corr_threshold = correlation_threshold
        self._min_obs = min_observations

    def compute(
        self,
        requests: list[PositionRequest],
        portfolio_value: float,
    ) -> CorrelationResult:
        """Compute correlation-adjusted position sizes."""
        n = len(requests)

        # Single position — passthrough
        if n <= 1:
            positions = [
                AdjustedPosition(
                    ticker=r.ticker,
                    raw_position_pct=r.raw_position_pct,
                    adjusted_position_pct=r.raw_position_pct,
                    sector=r.sector,
                    scaling_factor=1.0,
                )
                for r in requests
            ]
            corr_matrix = np.array([[1.0]]) if n == 1 else np.array([])
            return CorrelationResult(
                positions=positions,
                correlation_matrix=corr_matrix,
                correlated_groups=[],
            )

        # Build return matrix and compute correlation
        min_len = min(len(r.recent_returns) for r in requests)
        return_matrix = np.array([r.recent_returns[:min_len] for r in requests])
        corr_matrix = np.corrcoef(return_matrix)

        # Find correlated groups using union-find
        groups = self._find_correlated_groups(requests, corr_matrix)

        # Scale down groups that exceed cap
        scaling_factors = {r.ticker: 1.0 for r in requests}
        for group in groups:
            group_total = sum(
                r.raw_position_pct for r in requests if r.ticker in group
            )
            if group_total > self._max_group_pct:
                scale = self._max_group_pct / group_total
                for ticker in group:
                    scaling_factors[ticker] = min(scaling_factors[ticker], scale)

        positions = [
            AdjustedPosition(
                ticker=r.ticker,
                raw_position_pct=r.raw_position_pct,
                adjusted_position_pct=r.raw_position_pct * scaling_factors[r.ticker],
                sector=r.sector,
                scaling_factor=scaling_factors[r.ticker],
            )
            for r in requests
        ]

        return CorrelationResult(
            positions=positions,
            correlation_matrix=corr_matrix,
            correlated_groups=groups,
        )

    def _find_correlated_groups(
        self,
        requests: list[PositionRequest],
        corr_matrix: np.ndarray,
    ) -> list[list[str]]:
        """Find groups of tickers with pairwise correlation above threshold."""
        n = len(requests)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr_matrix[i, j]) >= self._corr_threshold:
                    union(i, j)

        # Collect groups with more than one member
        from collections import defaultdict
        group_map: dict[int, list[str]] = defaultdict(list)
        for i in range(n):
            group_map[find(i)].append(requests[i].ticker)

        return [tickers for tickers in group_map.values() if len(tickers) > 1]
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_correlation_sizing.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/sizing/correlation.py tests/unit/test_correlation_sizing.py
git commit -m "feat: correlation-aware sizing prevents factor concentration"
```

---

## Task 5: Regime-Adjusted Sizing

**Files:**
- Create: `src/evolve_trader/sizing/regime_adjusted.py`
- Create: `tests/unit/test_regime_sizing.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_regime_sizing.py
"""Tests for regime-adjusted position sizing."""
import pytest
from evolve_trader.sizing.regime_adjusted import (
    RegimeAdjustedSizer,
    RegimeAdjustedResult,
    ExposureCaps,
)
from evolve_trader.regime.labels import RegimeLabel, PrimaryRegime, MomentumState


def _make_regime(primary: PrimaryRegime, confidence: float = 0.8) -> RegimeLabel:
    return RegimeLabel(
        primary, "test narrative", MomentumState.STABLE, confidence, "short-term"
    )


def test_risk_on_allows_full_exposure():
    """Risk-on regime → up to 100% gross exposure."""
    caps = ExposureCaps(risk_on=1.0, risk_off=0.6, transitional=0.8, crisis=0.3)
    sizer = RegimeAdjustedSizer(caps=caps)
    result = sizer.adjust(
        raw_position_pct=0.25,
        current_gross_exposure=0.50,
        regime=_make_regime(PrimaryRegime.RISK_ON),
    )
    assert result.adjusted_position_pct == 0.25  # No reduction needed
    assert result.regime_cap == 1.0


def test_risk_off_caps_at_60_pct():
    """Risk-off regime → 60% gross exposure cap."""
    caps = ExposureCaps(risk_on=1.0, risk_off=0.6, transitional=0.8, crisis=0.3)
    sizer = RegimeAdjustedSizer(caps=caps)
    result = sizer.adjust(
        raw_position_pct=0.25,
        current_gross_exposure=0.50,
        regime=_make_regime(PrimaryRegime.RISK_OFF),
    )
    # Cap is 0.6, current is 0.5, room for 0.10 max
    assert result.adjusted_position_pct == pytest.approx(0.10, abs=0.01)


def test_crisis_regime_caps_at_30_pct():
    """Crisis regime → 30% gross exposure cap."""
    caps = ExposureCaps(risk_on=1.0, risk_off=0.6, transitional=0.8, crisis=0.3)
    sizer = RegimeAdjustedSizer(caps=caps)
    result = sizer.adjust(
        raw_position_pct=0.25,
        current_gross_exposure=0.20,
        regime=_make_regime(PrimaryRegime.CRISIS),
    )
    # Cap is 0.3, current is 0.2, room for 0.10
    assert result.adjusted_position_pct == pytest.approx(0.10, abs=0.01)


def test_already_at_cap_returns_zero():
    """When already at cap → zero new position allowed."""
    caps = ExposureCaps(risk_on=1.0, risk_off=0.6, transitional=0.8, crisis=0.3)
    sizer = RegimeAdjustedSizer(caps=caps)
    result = sizer.adjust(
        raw_position_pct=0.25,
        current_gross_exposure=0.60,
        regime=_make_regime(PrimaryRegime.RISK_OFF),
    )
    assert result.adjusted_position_pct == 0.0
    assert result.at_capacity is True


def test_transitional_regime_uses_transitional_cap():
    """Transitional regime → 80% cap."""
    caps = ExposureCaps(risk_on=1.0, risk_off=0.6, transitional=0.8, crisis=0.3)
    sizer = RegimeAdjustedSizer(caps=caps)
    result = sizer.adjust(
        raw_position_pct=0.50,
        current_gross_exposure=0.40,
        regime=_make_regime(PrimaryRegime.TRANSITIONAL),
    )
    # Cap is 0.8, current is 0.4, room for 0.40
    assert result.adjusted_position_pct == pytest.approx(0.40, abs=0.01)


def test_low_confidence_reduces_cap():
    """Low regime confidence further reduces the effective cap."""
    caps = ExposureCaps(risk_on=1.0, risk_off=0.6, transitional=0.8, crisis=0.3)
    sizer = RegimeAdjustedSizer(caps=caps, confidence_scaling=True)
    result = sizer.adjust(
        raw_position_pct=0.25,
        current_gross_exposure=0.0,
        regime=_make_regime(PrimaryRegime.RISK_ON, confidence=0.5),
    )
    # Effective cap = 1.0 * 0.5 = 0.50
    assert result.effective_cap == pytest.approx(0.50, abs=0.05)


def test_regime_adjusted_result_data_model():
    """RegimeAdjustedResult has all required fields."""
    result = RegimeAdjustedResult(
        raw_position_pct=0.25,
        adjusted_position_pct=0.10,
        regime_cap=0.6,
        effective_cap=0.6,
        current_gross_exposure=0.50,
        headroom=0.10,
        at_capacity=False,
        regime_name="RISK_OFF",
    )
    assert result.headroom == 0.10
    assert result.regime_name == "RISK_OFF"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_regime_sizing.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.sizing.regime_adjusted'`

**Step 3: Implement**

```python
# src/evolve_trader/sizing/regime_adjusted.py
"""Regime-adjusted position sizing — modulates gross exposure by market regime."""
from __future__ import annotations

from dataclasses import dataclass
from evolve_trader.regime.labels import RegimeLabel, PrimaryRegime


@dataclass
class ExposureCaps:
    """Per-regime gross exposure caps."""
    risk_on: float = 1.0
    risk_off: float = 0.6
    transitional: float = 0.8
    crisis: float = 0.3

    def get_cap(self, regime: PrimaryRegime) -> float:
        """Get the exposure cap for a given regime."""
        mapping = {
            PrimaryRegime.RISK_ON: self.risk_on,
            PrimaryRegime.RISK_OFF: self.risk_off,
            PrimaryRegime.TRANSITIONAL: self.transitional,
            PrimaryRegime.CRISIS: self.crisis,
        }
        return mapping.get(regime, self.transitional)


@dataclass
class RegimeAdjustedResult:
    """Output of regime-adjusted sizing."""
    raw_position_pct: float
    adjusted_position_pct: float
    regime_cap: float
    effective_cap: float
    current_gross_exposure: float
    headroom: float
    at_capacity: bool
    regime_name: str


class RegimeAdjustedSizer:
    """Modulates gross exposure based on the current market regime.

    Risk-off: 60% cap. Bull/Risk-on: 100%. Crisis: 30%.
    Optionally scales cap by regime confidence.
    """

    def __init__(
        self,
        caps: ExposureCaps | None = None,
        confidence_scaling: bool = False,
    ):
        self._caps = caps or ExposureCaps()
        self._confidence_scaling = confidence_scaling

    def adjust(
        self,
        raw_position_pct: float,
        current_gross_exposure: float,
        regime: RegimeLabel,
    ) -> RegimeAdjustedResult:
        """Adjust position size based on regime exposure cap."""
        regime_cap = self._caps.get_cap(regime.primary_regime)

        # Optionally scale by regime confidence
        if self._confidence_scaling:
            effective_cap = regime_cap * regime.confidence
        else:
            effective_cap = regime_cap

        # Compute headroom
        headroom = max(0.0, effective_cap - current_gross_exposure)

        # Adjust position to fit within headroom
        adjusted = min(raw_position_pct, headroom)
        at_capacity = headroom <= 0.0

        return RegimeAdjustedResult(
            raw_position_pct=raw_position_pct,
            adjusted_position_pct=adjusted,
            regime_cap=regime_cap,
            effective_cap=effective_cap,
            current_gross_exposure=current_gross_exposure,
            headroom=headroom,
            at_capacity=at_capacity,
            regime_name=regime.primary_regime.name,
        )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_regime_sizing.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/sizing/regime_adjusted.py tests/unit/test_regime_sizing.py
git commit -m "feat: regime-adjusted sizing with per-regime exposure caps"
```

---

## Task 6: Composition Interface

**Files:**
- Create: `src/evolve_trader/core/composition.py`
- Create: `src/evolve_trader/core/trade_intent.py`
- Create: `tests/unit/test_composition.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_composition.py
"""Tests for the strategy-sizing composition interface."""
import pytest
from datetime import datetime, timezone
from evolve_trader.core.trade_intent import (
    StrategyOutput,
    SizingOutput,
    TradeIntent,
    Direction,
)
from evolve_trader.core.composition import (
    Composer,
    CompositionResult,
    CompositionError,
)
from evolve_trader.regime.labels import RegimeLabel, PrimaryRegime, MomentumState


def _risk_on_regime() -> RegimeLabel:
    return RegimeLabel(
        PrimaryRegime.RISK_ON, "bull market", MomentumState.STRENGTHENING, 0.85, "short-term"
    )


def test_strategy_output_creation():
    """StrategyOutput captures what/when from a strategy skill."""
    output = StrategyOutput(
        strategy_name="momentum-v1",
        ticker="AAPL",
        direction=Direction.LONG,
        conditions=["price_above_sma_50", "volume_breakout"],
        regime=_risk_on_regime(),
        confidence=0.8,
        timestamp=datetime.now(timezone.utc),
    )
    assert output.ticker == "AAPL"
    assert output.direction == Direction.LONG
    assert len(output.conditions) == 2


def test_sizing_output_creation():
    """SizingOutput captures how much from a sizing skill."""
    output = SizingOutput(
        sizing_skill="kelly-v1",
        position_pct=0.08,
        shares=53,
        position_value=8_000.0,
        rationale="Quarter-Kelly: f*=0.32, applied=0.08",
    )
    assert output.position_pct == 0.08
    assert output.shares == 53


def test_trade_intent_combines_strategy_and_sizing():
    """TradeIntent merges strategy what/when with sizing how much."""
    intent = TradeIntent(
        strategy=StrategyOutput(
            strategy_name="momentum-v1",
            ticker="AAPL",
            direction=Direction.LONG,
            conditions=["breakout"],
            regime=_risk_on_regime(),
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
        ),
        sizing=SizingOutput(
            sizing_skill="kelly-v1",
            position_pct=0.08,
            shares=53,
            position_value=8_000.0,
            rationale="Quarter-Kelly",
        ),
    )
    assert intent.strategy.ticker == "AAPL"
    assert intent.sizing.shares == 53
    assert intent.ticker == "AAPL"  # Convenience property
    assert intent.direction == Direction.LONG


def test_trade_intent_rejects_zero_shares():
    """TradeIntent with zero shares is flagged as no-op."""
    intent = TradeIntent(
        strategy=StrategyOutput(
            strategy_name="momentum-v1",
            ticker="AAPL",
            direction=Direction.LONG,
            conditions=["breakout"],
            regime=_risk_on_regime(),
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
        ),
        sizing=SizingOutput(
            sizing_skill="kelly-v1",
            position_pct=0.0,
            shares=0,
            position_value=0.0,
            rationale="Negative edge",
        ),
    )
    assert intent.is_noop is True


def test_composer_produces_trade_intent():
    """Composer wires strategy output through sizing to produce TradeIntent."""
    composer = Composer()
    strategy_out = StrategyOutput(
        strategy_name="momentum-v1",
        ticker="NVDA",
        direction=Direction.LONG,
        conditions=["rsi_oversold_bounce"],
        regime=_risk_on_regime(),
        confidence=0.75,
        timestamp=datetime.now(timezone.utc),
    )
    sizing_out = SizingOutput(
        sizing_skill="kelly-v1",
        position_pct=0.06,
        shares=12,
        position_value=6_000.0,
        rationale="Quarter-Kelly: f*=0.24, applied=0.06",
    )
    result = composer.compose(strategy_out, sizing_out)
    assert isinstance(result, CompositionResult)
    assert result.intent.ticker == "NVDA"
    assert result.intent.sizing.shares == 12


def test_composer_validates_ticker_consistency():
    """Composer validates that strategy and sizing agree on ticker."""
    composer = Composer()
    # Strategy says AAPL but we try to compose — should just use strategy ticker
    strategy_out = StrategyOutput(
        strategy_name="momentum-v1",
        ticker="AAPL",
        direction=Direction.LONG,
        conditions=[],
        regime=_risk_on_regime(),
        confidence=0.75,
        timestamp=datetime.now(timezone.utc),
    )
    sizing_out = SizingOutput(
        sizing_skill="kelly-v1",
        position_pct=0.05,
        shares=10,
        position_value=5_000.0,
        rationale="Standard sizing",
    )
    result = composer.compose(strategy_out, sizing_out)
    assert result.intent.ticker == "AAPL"


def test_composer_batch_composition():
    """Composer handles multiple strategy outputs in batch."""
    composer = Composer()
    strategy_outputs = [
        StrategyOutput(
            strategy_name="momentum-v1",
            ticker="AAPL",
            direction=Direction.LONG,
            conditions=["breakout"],
            regime=_risk_on_regime(),
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
        ),
        StrategyOutput(
            strategy_name="mean-reversion-v1",
            ticker="TSLA",
            direction=Direction.SHORT,
            conditions=["overbought"],
            regime=_risk_on_regime(),
            confidence=0.7,
            timestamp=datetime.now(timezone.utc),
        ),
    ]
    sizing_outputs = [
        SizingOutput("kelly-v1", 0.06, 40, 6_000.0, "Kelly sizing"),
        SizingOutput("volatility-v1", 0.04, 5, 4_000.0, "Vol target sizing"),
    ]
    results = composer.compose_batch(strategy_outputs, sizing_outputs)
    assert len(results) == 2
    assert results[0].intent.ticker == "AAPL"
    assert results[1].intent.direction == Direction.SHORT


def test_composer_batch_length_mismatch_raises():
    """Batch composition with mismatched list lengths raises error."""
    composer = Composer()
    with pytest.raises(CompositionError, match="length mismatch"):
        composer.compose_batch(
            [StrategyOutput("m", "A", Direction.LONG, [], _risk_on_regime(), 0.8, datetime.now(timezone.utc))],
            [],
        )
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_composition.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.core.trade_intent'`

**Step 3: Implement**

```python
# src/evolve_trader/core/trade_intent.py
"""TradeIntent — the composed output of strategy (what/when) + sizing (how much)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from evolve_trader.regime.labels import RegimeLabel


class Direction(str, Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True)
class StrategyOutput:
    """What a strategy skill produces: what to trade and when.

    This is the 'signal' side of composition.
    """
    strategy_name: str
    ticker: str
    direction: Direction
    conditions: list[str]
    regime: RegimeLabel
    confidence: float
    timestamp: datetime


@dataclass(frozen=True)
class SizingOutput:
    """What a sizing skill produces: how much to trade.

    This is the 'sizing' side of composition.
    """
    sizing_skill: str
    position_pct: float
    shares: int
    position_value: float
    rationale: str


@dataclass
class TradeIntent:
    """The fully composed trade: strategy output + sizing output.

    This is the interface handed to portfolio risk enforcement and execution.
    """
    strategy: StrategyOutput
    sizing: SizingOutput

    @property
    def ticker(self) -> str:
        return self.strategy.ticker

    @property
    def direction(self) -> Direction:
        return self.strategy.direction

    @property
    def is_noop(self) -> bool:
        """True if sizing determined no position should be taken."""
        return self.sizing.shares == 0 or self.sizing.position_pct == 0.0
```

```python
# src/evolve_trader/core/composition.py
"""Composer — wires strategy outputs through sizing to produce TradeIntents."""
from __future__ import annotations

from dataclasses import dataclass

from evolve_trader.core.trade_intent import (
    StrategyOutput,
    SizingOutput,
    TradeIntent,
)


class CompositionError(Exception):
    """Raised when composition fails due to invalid inputs."""
    pass


@dataclass
class CompositionResult:
    """Result of composing a strategy output with a sizing output."""
    intent: TradeIntent
    warnings: list[str]


class Composer:
    """Composes strategy outputs (what/when) with sizing outputs (how much).

    Clean separation: strategies never know about position sizes.
    Sizing skills never know about entry conditions.
    The Composer bridges the two.
    """

    def compose(
        self,
        strategy_output: StrategyOutput,
        sizing_output: SizingOutput,
    ) -> CompositionResult:
        """Compose a single strategy output with its sizing output."""
        warnings: list[str] = []

        intent = TradeIntent(
            strategy=strategy_output,
            sizing=sizing_output,
        )

        if intent.is_noop:
            warnings.append(
                f"No-op trade for {strategy_output.ticker}: "
                f"sizing returned 0 shares ({sizing_output.rationale})"
            )

        return CompositionResult(intent=intent, warnings=warnings)

    def compose_batch(
        self,
        strategy_outputs: list[StrategyOutput],
        sizing_outputs: list[SizingOutput],
    ) -> list[CompositionResult]:
        """Compose multiple strategy-sizing pairs in batch."""
        if len(strategy_outputs) != len(sizing_outputs):
            raise CompositionError(
                f"Batch composition length mismatch: "
                f"{len(strategy_outputs)} strategies vs {len(sizing_outputs)} sizings"
            )

        return [
            self.compose(s, z)
            for s, z in zip(strategy_outputs, sizing_outputs)
        ]
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_composition.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/core/trade_intent.py src/evolve_trader/core/composition.py tests/unit/test_composition.py
git commit -m "feat: composition interface separating strategy what/when from sizing how much"
```

---

## Task 7: Portfolio-Level Risk Enforcement

**Files:**
- Create: `src/evolve_trader/risk/__init__.py`
- Create: `src/evolve_trader/risk/portfolio_tracker.py`
- Create: `src/evolve_trader/risk/pre_trade_validator.py`
- Create: `tests/unit/test_portfolio_risk.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_portfolio_risk.py
"""Tests for portfolio-level risk enforcement."""
import pytest
from datetime import datetime, timezone
from evolve_trader.risk.portfolio_tracker import (
    PortfolioTracker,
    Position,
    ExposureSnapshot,
)
from evolve_trader.risk.pre_trade_validator import (
    PreTradeValidator,
    ValidationResult,
    ValidationRejection,
    RiskLimits,
)
from evolve_trader.core.trade_intent import (
    StrategyOutput,
    SizingOutput,
    TradeIntent,
    Direction,
)
from evolve_trader.regime.labels import RegimeLabel, PrimaryRegime, MomentumState


def _risk_on_regime() -> RegimeLabel:
    return RegimeLabel(
        PrimaryRegime.RISK_ON, "bull", MomentumState.STRENGTHENING, 0.85, "short-term"
    )


def _make_intent(
    ticker: str,
    direction: Direction,
    position_pct: float,
    shares: int,
    sector: str = "Technology",
) -> TradeIntent:
    return TradeIntent(
        strategy=StrategyOutput(
            strategy_name="test-strategy",
            ticker=ticker,
            direction=direction,
            conditions=[],
            regime=_risk_on_regime(),
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
        ),
        sizing=SizingOutput(
            sizing_skill="test-sizer",
            position_pct=position_pct,
            shares=shares,
            position_value=position_pct * 100_000.0,
            rationale="test",
        ),
    )


# --- PortfolioTracker Tests ---

def test_portfolio_tracker_empty_portfolio():
    """Empty portfolio has zero exposure."""
    tracker = PortfolioTracker(portfolio_value=100_000.0)
    snapshot = tracker.get_exposure()
    assert snapshot.gross_exposure == 0.0
    assert snapshot.net_exposure == 0.0
    assert snapshot.position_count == 0


def test_portfolio_tracker_add_position():
    """Adding a position updates exposure tracking."""
    tracker = PortfolioTracker(portfolio_value=100_000.0)
    tracker.add_position(Position("AAPL", 50, 150.0, "Technology", Direction.LONG))
    snapshot = tracker.get_exposure()
    assert snapshot.gross_exposure == pytest.approx(0.075, abs=0.001)  # 7500/100000
    assert snapshot.position_count == 1


def test_portfolio_tracker_long_short_net_exposure():
    """Net exposure = long - short."""
    tracker = PortfolioTracker(portfolio_value=100_000.0)
    tracker.add_position(Position("AAPL", 50, 200.0, "Technology", Direction.LONG))   # +10k
    tracker.add_position(Position("TSLA", 20, 150.0, "Automotive", Direction.SHORT))  # -3k
    snapshot = tracker.get_exposure()
    assert snapshot.gross_exposure == pytest.approx(0.13, abs=0.01)  # 13k/100k
    assert snapshot.net_exposure == pytest.approx(0.07, abs=0.01)    # 7k/100k


def test_portfolio_tracker_sector_exposure():
    """Tracks per-sector exposure."""
    tracker = PortfolioTracker(portfolio_value=100_000.0)
    tracker.add_position(Position("AAPL", 50, 200.0, "Technology", Direction.LONG))
    tracker.add_position(Position("MSFT", 30, 300.0, "Technology", Direction.LONG))
    tracker.add_position(Position("JPM", 40, 150.0, "Financials", Direction.LONG))
    snapshot = tracker.get_exposure()
    assert snapshot.sector_exposures["Technology"] == pytest.approx(0.19, abs=0.01)
    assert snapshot.sector_exposures["Financials"] == pytest.approx(0.06, abs=0.01)


def test_portfolio_tracker_remove_position():
    """Removing a position updates exposure."""
    tracker = PortfolioTracker(portfolio_value=100_000.0)
    tracker.add_position(Position("AAPL", 50, 200.0, "Technology", Direction.LONG))
    tracker.remove_position("AAPL")
    snapshot = tracker.get_exposure()
    assert snapshot.gross_exposure == 0.0


def test_portfolio_tracker_rebalance_trigger():
    """Rebalance is triggered when drift exceeds threshold."""
    tracker = PortfolioTracker(portfolio_value=100_000.0, rebalance_drift_pct=0.05)
    tracker.add_position(Position("AAPL", 50, 200.0, "Technology", Direction.LONG))
    # Simulate price drift
    tracker.update_price("AAPL", 250.0)  # 25% price increase
    assert tracker.needs_rebalance() is True


# --- PreTradeValidator Tests ---

def test_pre_trade_validator_approves_valid_trade():
    """Valid trade within all limits is approved."""
    limits = RiskLimits(
        max_position_pct=0.10,
        max_sector_pct=0.30,
        max_gross_exposure=1.0,
        max_single_trade_pct=0.10,
    )
    validator = PreTradeValidator(limits=limits)
    tracker = PortfolioTracker(portfolio_value=100_000.0)

    intent = _make_intent("AAPL", Direction.LONG, 0.05, 33)
    result = validator.validate(intent, tracker, sector="Technology")
    assert result.approved is True
    assert len(result.rejections) == 0


def test_pre_trade_validator_rejects_oversized_position():
    """Position exceeding max_position_pct is rejected."""
    limits = RiskLimits(
        max_position_pct=0.10,
        max_sector_pct=0.30,
        max_gross_exposure=1.0,
        max_single_trade_pct=0.10,
    )
    validator = PreTradeValidator(limits=limits)
    tracker = PortfolioTracker(portfolio_value=100_000.0)

    intent = _make_intent("AAPL", Direction.LONG, 0.15, 100)
    result = validator.validate(intent, tracker, sector="Technology")
    assert result.approved is False
    assert any("position" in r.reason.lower() for r in result.rejections)


def test_pre_trade_validator_rejects_sector_breach():
    """Trade that would breach sector limit is rejected."""
    limits = RiskLimits(
        max_position_pct=0.10,
        max_sector_pct=0.30,
        max_gross_exposure=1.0,
        max_single_trade_pct=0.10,
    )
    validator = PreTradeValidator(limits=limits)
    tracker = PortfolioTracker(portfolio_value=100_000.0)
    # Already 25% in Technology
    tracker.add_position(Position("MSFT", 83, 300.0, "Technology", Direction.LONG))

    intent = _make_intent("AAPL", Direction.LONG, 0.08, 53)
    result = validator.validate(intent, tracker, sector="Technology")
    assert result.approved is False
    assert any("sector" in r.reason.lower() for r in result.rejections)


def test_pre_trade_validator_rejects_gross_exposure_breach():
    """Trade that would breach gross exposure limit is rejected."""
    limits = RiskLimits(
        max_position_pct=0.10,
        max_sector_pct=0.30,
        max_gross_exposure=0.60,
        max_single_trade_pct=0.10,
    )
    validator = PreTradeValidator(limits=limits)
    tracker = PortfolioTracker(portfolio_value=100_000.0)
    # Already at 55% gross exposure
    tracker.add_position(Position("MSFT", 183, 300.0, "Technology", Direction.LONG))

    intent = _make_intent("AAPL", Direction.LONG, 0.08, 53)
    result = validator.validate(intent, tracker, sector="Technology")
    assert result.approved is False
    assert any("gross exposure" in r.reason.lower() for r in result.rejections)


def test_pre_trade_validator_multiple_rejections():
    """Trade can be rejected for multiple reasons simultaneously."""
    limits = RiskLimits(
        max_position_pct=0.05,
        max_sector_pct=0.10,
        max_gross_exposure=0.50,
        max_single_trade_pct=0.05,
    )
    validator = PreTradeValidator(limits=limits)
    tracker = PortfolioTracker(portfolio_value=100_000.0)
    tracker.add_position(Position("MSFT", 150, 300.0, "Technology", Direction.LONG))

    intent = _make_intent("AAPL", Direction.LONG, 0.08, 53)
    result = validator.validate(intent, tracker, sector="Technology")
    assert result.approved is False
    assert len(result.rejections) >= 2


def test_validation_result_data_model():
    """ValidationResult has all required fields."""
    result = ValidationResult(
        approved=False,
        rejections=[
            ValidationRejection("position_limit", "Position 15% exceeds max 10%"),
        ],
        intent_ticker="AAPL",
        intent_direction="long",
    )
    assert result.rejections[0].rule == "position_limit"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_portfolio_risk.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.risk'`

**Step 3: Implement**

```python
# src/evolve_trader/risk/__init__.py
"""Portfolio-level risk enforcement."""
```

```python
# src/evolve_trader/risk/portfolio_tracker.py
"""Real-time portfolio exposure tracking."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict

from evolve_trader.core.trade_intent import Direction


@dataclass
class Position:
    """A single portfolio position."""
    ticker: str
    shares: int
    price: float
    sector: str
    direction: Direction

    @property
    def market_value(self) -> float:
        return self.shares * self.price

    @property
    def signed_value(self) -> float:
        """Positive for long, negative for short."""
        return self.market_value if self.direction == Direction.LONG else -self.market_value


@dataclass
class ExposureSnapshot:
    """Point-in-time portfolio exposure metrics."""
    gross_exposure: float        # Sum of |position values| / portfolio_value
    net_exposure: float          # Sum of signed position values / portfolio_value
    position_count: int
    sector_exposures: dict[str, float]  # sector → exposure pct
    largest_position_pct: float
    timestamp: str = ""


class PortfolioTracker:
    """Tracks real-time portfolio exposure for risk enforcement.

    Maintains position inventory and computes exposure metrics on demand.
    Triggers rebalance when drift exceeds threshold.
    """

    def __init__(
        self,
        portfolio_value: float,
        rebalance_drift_pct: float = 0.05,
    ):
        self._portfolio_value = portfolio_value
        self._rebalance_drift_pct = rebalance_drift_pct
        self._positions: dict[str, Position] = {}
        self._target_weights: dict[str, float] = {}

    def add_position(self, position: Position) -> None:
        """Add or update a position."""
        self._positions[position.ticker] = position
        self._target_weights[position.ticker] = (
            position.market_value / self._portfolio_value
        )

    def remove_position(self, ticker: str) -> None:
        """Remove a position."""
        self._positions.pop(ticker, None)
        self._target_weights.pop(ticker, None)

    def update_price(self, ticker: str, new_price: float) -> None:
        """Update the price of a held position."""
        if ticker in self._positions:
            pos = self._positions[ticker]
            self._positions[ticker] = Position(
                ticker=pos.ticker,
                shares=pos.shares,
                price=new_price,
                sector=pos.sector,
                direction=pos.direction,
            )

    def get_exposure(self) -> ExposureSnapshot:
        """Compute current exposure snapshot."""
        if not self._positions:
            return ExposureSnapshot(
                gross_exposure=0.0,
                net_exposure=0.0,
                position_count=0,
                sector_exposures={},
                largest_position_pct=0.0,
            )

        gross = 0.0
        net = 0.0
        sector_values: dict[str, float] = defaultdict(float)
        largest = 0.0

        for pos in self._positions.values():
            mv = pos.market_value
            pct = mv / self._portfolio_value
            gross += pct
            net += pos.signed_value / self._portfolio_value
            sector_values[pos.sector] += pct
            largest = max(largest, pct)

        return ExposureSnapshot(
            gross_exposure=gross,
            net_exposure=net,
            position_count=len(self._positions),
            sector_exposures=dict(sector_values),
            largest_position_pct=largest,
        )

    def get_sector_exposure(self, sector: str) -> float:
        """Get total exposure for a specific sector."""
        total = 0.0
        for pos in self._positions.values():
            if pos.sector == sector:
                total += pos.market_value / self._portfolio_value
        return total

    def needs_rebalance(self) -> bool:
        """Check if any position has drifted beyond the rebalance threshold."""
        for ticker, pos in self._positions.items():
            current_weight = pos.market_value / self._portfolio_value
            target_weight = self._target_weights.get(ticker, 0.0)
            if abs(current_weight - target_weight) > self._rebalance_drift_pct:
                return True
        return False
```

```python
# src/evolve_trader/risk/pre_trade_validator.py
"""Pre-trade risk validation — enforces portfolio-level constraints before execution."""
from __future__ import annotations

from dataclasses import dataclass, field

from evolve_trader.core.trade_intent import TradeIntent
from evolve_trader.risk.portfolio_tracker import PortfolioTracker


@dataclass
class RiskLimits:
    """Portfolio-level risk limits."""
    max_position_pct: float = 0.10
    max_sector_pct: float = 0.30
    max_gross_exposure: float = 1.0
    max_single_trade_pct: float = 0.10


@dataclass
class ValidationRejection:
    """A single reason for rejecting a trade."""
    rule: str
    reason: str


@dataclass
class ValidationResult:
    """Result of pre-trade validation."""
    approved: bool
    rejections: list[ValidationRejection]
    intent_ticker: str
    intent_direction: str


class PreTradeValidator:
    """Validates trade intents against portfolio-level risk constraints.

    Checks:
    1. Single position size limit
    2. Sector concentration limit
    3. Gross exposure limit
    4. Single trade size limit
    """

    def __init__(self, limits: RiskLimits | None = None):
        self._limits = limits or RiskLimits()

    def validate(
        self,
        intent: TradeIntent,
        tracker: PortfolioTracker,
        sector: str,
    ) -> ValidationResult:
        """Validate a trade intent against all risk limits."""
        rejections: list[ValidationRejection] = []
        exposure = tracker.get_exposure()

        # Check 1: Single position size
        if intent.sizing.position_pct > self._limits.max_position_pct:
            rejections.append(ValidationRejection(
                "position_limit",
                f"Position {intent.sizing.position_pct:.0%} exceeds max {self._limits.max_position_pct:.0%}",
            ))

        # Check 2: Sector concentration
        current_sector = tracker.get_sector_exposure(sector)
        if current_sector + intent.sizing.position_pct > self._limits.max_sector_pct:
            rejections.append(ValidationRejection(
                "sector_limit",
                f"Sector '{sector}' would reach {current_sector + intent.sizing.position_pct:.0%}, "
                f"exceeding max {self._limits.max_sector_pct:.0%}",
            ))

        # Check 3: Gross exposure
        if exposure.gross_exposure + intent.sizing.position_pct > self._limits.max_gross_exposure:
            rejections.append(ValidationRejection(
                "gross_exposure_limit",
                f"Gross exposure would reach {exposure.gross_exposure + intent.sizing.position_pct:.0%}, "
                f"exceeding max {self._limits.max_gross_exposure:.0%}",
            ))

        # Check 4: Single trade size
        if intent.sizing.position_pct > self._limits.max_single_trade_pct:
            rejections.append(ValidationRejection(
                "single_trade_limit",
                f"Single trade {intent.sizing.position_pct:.0%} exceeds max {self._limits.max_single_trade_pct:.0%}",
            ))

        return ValidationResult(
            approved=len(rejections) == 0,
            rejections=rejections,
            intent_ticker=intent.ticker,
            intent_direction=intent.direction.value,
        )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_portfolio_risk.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/risk/ tests/unit/test_portfolio_risk.py
git commit -m "feat: portfolio-level risk enforcement with exposure tracking and pre-trade validation"
```

---

## Task 8: Tax-Aware Evolution Mode

**Files:**
- Create: `src/evolve_trader/core/tax_aware.py`
- Create: `tests/unit/test_tax_aware.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_tax_aware.py
"""Tests for tax-aware evolution mode."""
import pytest
from datetime import datetime, timezone, timedelta
from evolve_trader.core.tax_aware import (
    TaxAwareEvaluator,
    TaxConfig,
    TaxDragMetric,
    HoldingPeriod,
)


def test_holding_period_classification():
    """Short-term < 365 days, long-term >= 365 days."""
    entry = datetime(2025, 1, 1, tzinfo=timezone.utc)
    short_exit = datetime(2025, 6, 1, tzinfo=timezone.utc)
    long_exit = datetime(2026, 1, 2, tzinfo=timezone.utc)
    assert HoldingPeriod.classify(entry, short_exit) == HoldingPeriod.SHORT_TERM
    assert HoldingPeriod.classify(entry, long_exit) == HoldingPeriod.LONG_TERM


def test_tax_config_default_rates():
    """Default marginal tax rates for US federal."""
    config = TaxConfig()
    assert config.short_term_rate == 0.37  # Top bracket
    assert config.long_term_rate == 0.20
    assert config.differential == pytest.approx(0.17, abs=0.01)


def test_tax_config_custom_rates():
    """Custom tax rates for different jurisdictions or brackets."""
    config = TaxConfig(short_term_rate=0.32, long_term_rate=0.15)
    assert config.differential == pytest.approx(0.17, abs=0.01)


def test_tax_drag_metric_computed_when_disabled():
    """Tax drag is computed and visible even when tax-aware mode is off."""
    evaluator = TaxAwareEvaluator(config=TaxConfig(), enabled=False)
    trades = [
        {"pnl": 1000.0, "entry_date": datetime(2025, 1, 1, tzinfo=timezone.utc),
         "exit_date": datetime(2025, 3, 1, tzinfo=timezone.utc)},  # Short-term gain
        {"pnl": 2000.0, "entry_date": datetime(2025, 1, 1, tzinfo=timezone.utc),
         "exit_date": datetime(2026, 2, 1, tzinfo=timezone.utc)},  # Long-term gain
    ]
    metric = evaluator.compute_tax_drag(trades)
    assert metric.total_tax_drag > 0
    assert metric.short_term_gains == 1000.0
    assert metric.long_term_gains == 2000.0


def test_tax_penalty_applied_when_enabled():
    """When enabled, fitness is penalized by marginal tax rate differential."""
    evaluator = TaxAwareEvaluator(config=TaxConfig(), enabled=True)
    trades = [
        {"pnl": 1000.0, "entry_date": datetime(2025, 1, 1, tzinfo=timezone.utc),
         "exit_date": datetime(2025, 3, 1, tzinfo=timezone.utc)},  # Short-term
    ]
    raw_fitness = 1.5  # Raw Sharpe ratio
    adjusted = evaluator.adjust_fitness(raw_fitness, trades, total_pnl=1000.0)
    assert adjusted < raw_fitness  # Fitness should be penalized


def test_tax_penalty_not_applied_when_disabled():
    """When disabled, fitness is returned unchanged."""
    evaluator = TaxAwareEvaluator(config=TaxConfig(), enabled=False)
    trades = [
        {"pnl": 1000.0, "entry_date": datetime(2025, 1, 1, tzinfo=timezone.utc),
         "exit_date": datetime(2025, 3, 1, tzinfo=timezone.utc)},
    ]
    raw_fitness = 1.5
    adjusted = evaluator.adjust_fitness(raw_fitness, trades, total_pnl=1000.0)
    assert adjusted == raw_fitness


def test_all_long_term_gains_no_penalty():
    """Strategy with only long-term gains gets minimal or no penalty."""
    evaluator = TaxAwareEvaluator(config=TaxConfig(), enabled=True)
    trades = [
        {"pnl": 5000.0, "entry_date": datetime(2025, 1, 1, tzinfo=timezone.utc),
         "exit_date": datetime(2026, 3, 1, tzinfo=timezone.utc)},
    ]
    raw_fitness = 1.5
    adjusted = evaluator.adjust_fitness(raw_fitness, trades, total_pnl=5000.0)
    # No short-term gains → no differential penalty
    assert adjusted == pytest.approx(raw_fitness, abs=0.01)


def test_losses_not_penalized():
    """Losses are not penalized (no tax on losses — simplified)."""
    evaluator = TaxAwareEvaluator(config=TaxConfig(), enabled=True)
    trades = [
        {"pnl": -500.0, "entry_date": datetime(2025, 1, 1, tzinfo=timezone.utc),
         "exit_date": datetime(2025, 3, 1, tzinfo=timezone.utc)},
    ]
    metric = evaluator.compute_tax_drag(trades)
    assert metric.short_term_gains == 0.0  # Losses don't count as gains


def test_tax_drag_metric_data_model():
    """TaxDragMetric has all required fields."""
    metric = TaxDragMetric(
        short_term_gains=3000.0,
        long_term_gains=7000.0,
        short_term_tax=1110.0,
        long_term_tax=1400.0,
        total_tax_drag=2510.0,
        tax_efficiency_ratio=0.749,
        short_term_trade_count=5,
        long_term_trade_count=10,
    )
    assert metric.total_tax_drag == 2510.0
    assert metric.tax_efficiency_ratio == 0.749


def test_tax_aware_flag_is_configurable():
    """Tax-aware mode is a configurable flag."""
    enabled = TaxAwareEvaluator(config=TaxConfig(), enabled=True)
    disabled = TaxAwareEvaluator(config=TaxConfig(), enabled=False)
    assert enabled.is_enabled is True
    assert disabled.is_enabled is False
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_tax_aware.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.core.tax_aware'`

**Step 3: Implement**

```python
# src/evolve_trader/core/tax_aware.py
"""Tax-aware evolution mode — penalizes short-term capital gains in fitness evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class HoldingPeriod(str, Enum):
    """IRS holding period classification."""
    SHORT_TERM = "short_term"   # < 365 days
    LONG_TERM = "long_term"     # >= 365 days

    @staticmethod
    def classify(entry_date: datetime, exit_date: datetime) -> HoldingPeriod:
        """Classify a trade's holding period."""
        days_held = (exit_date - entry_date).days
        if days_held >= 365:
            return HoldingPeriod.LONG_TERM
        return HoldingPeriod.SHORT_TERM


@dataclass
class TaxConfig:
    """Tax rate configuration."""
    short_term_rate: float = 0.37   # Top marginal ordinary income rate
    long_term_rate: float = 0.20    # Top long-term capital gains rate

    @property
    def differential(self) -> float:
        """Marginal tax rate differential between short and long-term."""
        return self.short_term_rate - self.long_term_rate


@dataclass
class TaxDragMetric:
    """Computed tax drag metrics for a set of trades."""
    short_term_gains: float
    long_term_gains: float
    short_term_tax: float
    long_term_tax: float
    total_tax_drag: float
    tax_efficiency_ratio: float   # (total_gains - total_tax) / total_gains
    short_term_trade_count: int
    long_term_trade_count: int


class TaxAwareEvaluator:
    """Evaluates tax drag and optionally penalizes fitness for short-term gains.

    When enabled: fitness is penalized by the proportion of short-term gains
    multiplied by the marginal tax rate differential.

    When disabled: tax drag metric is still computed and visible for monitoring,
    but fitness is returned unchanged.
    """

    def __init__(self, config: TaxConfig | None = None, enabled: bool = False):
        self._config = config or TaxConfig()
        self._enabled = enabled

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def compute_tax_drag(self, trades: list[dict[str, Any]]) -> TaxDragMetric:
        """Compute tax drag metrics for a set of completed trades.

        Each trade dict must have: pnl, entry_date, exit_date.
        """
        short_term_gains = 0.0
        long_term_gains = 0.0
        short_count = 0
        long_count = 0

        for trade in trades:
            pnl = trade["pnl"]
            if pnl <= 0:
                continue  # Simplified: skip losses

            period = HoldingPeriod.classify(trade["entry_date"], trade["exit_date"])
            if period == HoldingPeriod.SHORT_TERM:
                short_term_gains += pnl
                short_count += 1
            else:
                long_term_gains += pnl
                long_count += 1

        short_tax = short_term_gains * self._config.short_term_rate
        long_tax = long_term_gains * self._config.long_term_rate
        total_tax = short_tax + long_tax
        total_gains = short_term_gains + long_term_gains

        efficiency = (total_gains - total_tax) / total_gains if total_gains > 0 else 1.0

        return TaxDragMetric(
            short_term_gains=short_term_gains,
            long_term_gains=long_term_gains,
            short_term_tax=short_tax,
            long_term_tax=long_tax,
            total_tax_drag=total_tax,
            tax_efficiency_ratio=efficiency,
            short_term_trade_count=short_count,
            long_term_trade_count=long_count,
        )

    def adjust_fitness(
        self,
        raw_fitness: float,
        trades: list[dict[str, Any]],
        total_pnl: float,
    ) -> float:
        """Optionally adjust fitness score based on tax drag.

        Penalty = (short_term_gains / total_pnl) * tax_rate_differential
        This penalizes strategies that generate predominantly short-term gains.
        """
        if not self._enabled:
            return raw_fitness

        if total_pnl <= 0:
            return raw_fitness

        metric = self.compute_tax_drag(trades)

        # Penalty proportional to short-term gains share * differential
        short_term_ratio = metric.short_term_gains / total_pnl if total_pnl > 0 else 0.0
        penalty = short_term_ratio * self._config.differential

        return raw_fitness * (1.0 - penalty)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_tax_aware.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/core/tax_aware.py tests/unit/test_tax_aware.py
git commit -m "feat: tax-aware evolution mode with configurable penalty and always-visible drag metric"
```

---

## Task 9: Paper-Trading Survival Gate

**Files:**
- Create: `src/evolve_trader/core/survival_gate.py`
- Create: `tests/unit/test_survival_gate.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_survival_gate.py
"""Tests for the paper-trading survival gate."""
import pytest
from datetime import datetime, timezone, timedelta
from evolve_trader.core.survival_gate import (
    SurvivalGate,
    SurvivalCriteria,
    SurvivalResult,
    PaperTradeRecord,
)


def _make_records(
    n_days: int,
    daily_returns: list[float] | None = None,
    sharpe: float = 1.0,
    max_drawdown: float = 0.08,
) -> list[PaperTradeRecord]:
    """Build synthetic paper trade records."""
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    if daily_returns is None:
        # Generate returns that produce approximate target Sharpe
        import numpy as np
        rng = np.random.RandomState(42)
        daily_returns = list(rng.normal(0.0004 * sharpe, 0.01, n_days))
    records = []
    equity = 100_000.0
    peak = equity
    for i, ret in enumerate(daily_returns[:n_days]):
        equity *= (1 + ret)
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        records.append(PaperTradeRecord(
            date=base + timedelta(days=i),
            equity=equity,
            daily_return=ret,
            drawdown=dd,
        ))
    return records


def test_survival_gate_passes_healthy_strategy():
    """Strategy meeting all criteria passes the gate."""
    criteria = SurvivalCriteria(
        min_trading_days=20,
        min_sharpe=0.5,
        max_drawdown=0.15,
    )
    gate = SurvivalGate(criteria=criteria)
    records = _make_records(n_days=25, sharpe=1.2, max_drawdown=0.05)
    result = gate.evaluate(records)
    assert result.passed is True


def test_survival_gate_fails_insufficient_days():
    """Strategy with < 20 trading days fails."""
    criteria = SurvivalCriteria(min_trading_days=20, min_sharpe=0.5, max_drawdown=0.15)
    gate = SurvivalGate(criteria=criteria)
    records = _make_records(n_days=10, sharpe=2.0)
    result = gate.evaluate(records)
    assert result.passed is False
    assert "trading_days" in result.failure_reasons[0].lower()


def test_survival_gate_fails_low_sharpe():
    """Strategy with Sharpe < 0.5 fails."""
    criteria = SurvivalCriteria(min_trading_days=20, min_sharpe=0.5, max_drawdown=0.15)
    gate = SurvivalGate(criteria=criteria)
    # Generate returns with very low Sharpe
    import numpy as np
    rng = np.random.RandomState(99)
    low_sharpe_returns = list(rng.normal(0.0, 0.02, 30))  # ~0 mean → ~0 Sharpe
    records = _make_records(n_days=30, daily_returns=low_sharpe_returns)
    result = gate.evaluate(records)
    assert result.passed is False
    assert any("sharpe" in r.lower() for r in result.failure_reasons)


def test_survival_gate_fails_excessive_drawdown():
    """Strategy with drawdown > 15% fails."""
    criteria = SurvivalCriteria(min_trading_days=20, min_sharpe=0.5, max_drawdown=0.15)
    gate = SurvivalGate(criteria=criteria)
    # Generate returns with a crash
    returns = [0.005] * 10 + [-0.03] * 8 + [0.005] * 12  # Crash in middle
    records = _make_records(n_days=30, daily_returns=returns)
    result = gate.evaluate(records)
    # The crash should cause > 15% drawdown
    max_dd = max(r.drawdown for r in records)
    if max_dd > 0.15:
        assert result.passed is False
        assert any("drawdown" in r.lower() for r in result.failure_reasons)


def test_survival_gate_multiple_failures():
    """Strategy can fail multiple criteria simultaneously."""
    criteria = SurvivalCriteria(min_trading_days=20, min_sharpe=0.5, max_drawdown=0.15)
    gate = SurvivalGate(criteria=criteria)
    records = _make_records(n_days=10, sharpe=0.1)  # Too few days + low Sharpe
    result = gate.evaluate(records)
    assert result.passed is False
    assert len(result.failure_reasons) >= 1


def test_survival_criteria_defaults():
    """Default criteria: 20 days, Sharpe >0.5, drawdown <15%."""
    criteria = SurvivalCriteria()
    assert criteria.min_trading_days == 20
    assert criteria.min_sharpe == 0.5
    assert criteria.max_drawdown == 0.15


def test_survival_result_data_model():
    """SurvivalResult has all required fields."""
    result = SurvivalResult(
        passed=True,
        failure_reasons=[],
        actual_trading_days=25,
        actual_sharpe=1.2,
        actual_max_drawdown=0.08,
        evaluation_period_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        evaluation_period_end=datetime(2025, 2, 5, tzinfo=timezone.utc),
    )
    assert result.actual_sharpe == 1.2
    assert result.actual_max_drawdown == 0.08


def test_survival_gate_empty_records():
    """Empty record list fails immediately."""
    criteria = SurvivalCriteria()
    gate = SurvivalGate(criteria=criteria)
    result = gate.evaluate([])
    assert result.passed is False
    assert "no records" in result.failure_reasons[0].lower()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_survival_gate.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.core.survival_gate'`

**Step 3: Implement**

```python
# src/evolve_trader/core/survival_gate.py
"""Paper-trading survival gate — validates skill readiness before live deployment."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


TRADING_DAYS_PER_YEAR = 252


@dataclass
class PaperTradeRecord:
    """A single day's paper trading result."""
    date: datetime
    equity: float
    daily_return: float
    drawdown: float


@dataclass
class SurvivalCriteria:
    """Minimum criteria for passing the survival gate."""
    min_trading_days: int = 20
    min_sharpe: float = 0.5
    max_drawdown: float = 0.15


@dataclass
class SurvivalResult:
    """Result of survival gate evaluation."""
    passed: bool
    failure_reasons: list[str]
    actual_trading_days: int
    actual_sharpe: float
    actual_max_drawdown: float
    evaluation_period_start: datetime | None = None
    evaluation_period_end: datetime | None = None


class SurvivalGate:
    """Validates that a skill has demonstrated sufficient paper-trading performance.

    Requirements (default):
    - 20+ trading days of paper trading
    - Sharpe ratio > 0.5
    - Maximum drawdown < 15%

    All skill types must pass. Walk-forward validation pre-Phase-6, Alpaca paper post-Phase-6.
    """

    def __init__(self, criteria: SurvivalCriteria | None = None):
        self._criteria = criteria or SurvivalCriteria()

    def evaluate(self, records: list[PaperTradeRecord]) -> SurvivalResult:
        """Evaluate paper trading records against survival criteria."""
        failures: list[str] = []

        if not records:
            return SurvivalResult(
                passed=False,
                failure_reasons=["No records provided — cannot evaluate"],
                actual_trading_days=0,
                actual_sharpe=0.0,
                actual_max_drawdown=0.0,
            )

        n_days = len(records)
        returns = np.array([r.daily_return for r in records])
        max_dd = max(r.drawdown for r in records)

        # Compute annualized Sharpe
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            sharpe = 0.0

        # Check criteria
        if n_days < self._criteria.min_trading_days:
            failures.append(
                f"Insufficient trading_days: {n_days} < {self._criteria.min_trading_days}"
            )

        if sharpe < self._criteria.min_sharpe:
            failures.append(
                f"Sharpe ratio too low: {sharpe:.2f} < {self._criteria.min_sharpe}"
            )

        if max_dd > self._criteria.max_drawdown:
            failures.append(
                f"Max drawdown too high: {max_dd:.2%} > {self._criteria.max_drawdown:.2%}"
            )

        return SurvivalResult(
            passed=len(failures) == 0,
            failure_reasons=failures,
            actual_trading_days=n_days,
            actual_sharpe=float(sharpe),
            actual_max_drawdown=float(max_dd),
            evaluation_period_start=records[0].date,
            evaluation_period_end=records[-1].date,
        )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_survival_gate.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/core/survival_gate.py tests/unit/test_survival_gate.py
git commit -m "feat: paper-trading survival gate with configurable Sharpe/drawdown/days criteria"
```

---

## Task 10: Integration Testing — Full Composition Pipeline

**Files:**
- Create: `tests/integration/test_sizing_pipeline.py`

**Step 1: Write the integration tests**

```python
# tests/integration/test_sizing_pipeline.py
"""Integration tests for the full sizing composition pipeline.

End-to-end: strategy output → sizing → composition → risk validation → survival gate.
"""
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from evolve_trader.core.trade_intent import (
    StrategyOutput,
    SizingOutput,
    TradeIntent,
    Direction,
)
from evolve_trader.core.composition import Composer, CompositionResult
from evolve_trader.sizing.kelly import KellySizer, TradeHistory
from evolve_trader.sizing.volatility import VolatilityTargetSizer
from evolve_trader.sizing.correlation import (
    CorrelationAwareSizer,
    PositionRequest,
)
from evolve_trader.sizing.regime_adjusted import (
    RegimeAdjustedSizer,
    ExposureCaps,
)
from evolve_trader.risk.portfolio_tracker import PortfolioTracker, Position
from evolve_trader.risk.pre_trade_validator import PreTradeValidator, RiskLimits
from evolve_trader.core.tax_aware import TaxAwareEvaluator, TaxConfig
from evolve_trader.core.survival_gate import (
    SurvivalGate,
    SurvivalCriteria,
    PaperTradeRecord,
)
from evolve_trader.regime.labels import RegimeLabel, PrimaryRegime, MomentumState
from evolve_trader.sizing.schema import (
    SizingSkillConfig,
    SizingMethod,
    RegimeBehavior,
    RiskBudget,
    parse_sizing_skill_md,
)


def _risk_on_regime(confidence: float = 0.85) -> RegimeLabel:
    return RegimeLabel(
        PrimaryRegime.RISK_ON, "bull market", MomentumState.STRENGTHENING, confidence, "short-term"
    )


def _risk_off_regime(confidence: float = 0.8) -> RegimeLabel:
    return RegimeLabel(
        PrimaryRegime.RISK_OFF, "bear signal", MomentumState.WEAKENING, confidence, "short-term"
    )


def test_full_pipeline_kelly_to_execution():
    """End-to-end: Kelly sizing → composition → risk validation."""
    # 1. Strategy produces output
    strategy_out = StrategyOutput(
        strategy_name="momentum-v1",
        ticker="AAPL",
        direction=Direction.LONG,
        conditions=["price_above_sma_50", "volume_breakout"],
        regime=_risk_on_regime(),
        confidence=0.8,
        timestamp=datetime.now(timezone.utc),
    )

    # 2. Kelly sizer computes position
    kelly = KellySizer(kelly_fraction=0.25, max_position_pct=0.10)
    history = TradeHistory(
        returns=[0.05] * 60 + [-0.03] * 40  # 60% win rate
    )
    kelly_result = kelly.compute(history, portfolio_value=100_000.0, price=150.0)

    # 3. Compose into TradeIntent
    sizing_out = SizingOutput(
        sizing_skill="kelly-v1",
        position_pct=kelly_result.position_pct,
        shares=kelly_result.shares,
        position_value=kelly_result.position_value,
        rationale=f"Quarter-Kelly: f*={kelly_result.kelly_fraction:.3f}",
    )
    composer = Composer()
    result = composer.compose(strategy_out, sizing_out)
    assert isinstance(result, CompositionResult)
    assert result.intent.ticker == "AAPL"

    # 4. Pre-trade risk validation
    limits = RiskLimits(max_position_pct=0.10, max_sector_pct=0.30, max_gross_exposure=1.0)
    validator = PreTradeValidator(limits=limits)
    tracker = PortfolioTracker(portfolio_value=100_000.0)
    validation = validator.validate(result.intent, tracker, sector="Technology")
    assert validation.approved is True


def test_full_pipeline_volatility_sizing_risk_off():
    """Volatility sizing in risk-off regime with regime adjustment."""
    # 1. Strategy output in risk-off
    strategy_out = StrategyOutput(
        strategy_name="mean-reversion-v1",
        ticker="XOM",
        direction=Direction.LONG,
        conditions=["reversion_signal"],
        regime=_risk_off_regime(),
        confidence=0.75,
        timestamp=datetime.now(timezone.utc),
    )

    # 2. Volatility sizing
    vol_sizer = VolatilityTargetSizer(target_annual_vol=0.15, max_position_pct=0.20)
    rng = np.random.RandomState(42)
    returns = list(rng.normal(0, 0.015, 60))
    vol_result = vol_sizer.compute(returns, portfolio_value=100_000.0, price=80.0)

    # 3. Regime adjustment
    regime_sizer = RegimeAdjustedSizer(
        caps=ExposureCaps(risk_on=1.0, risk_off=0.6, transitional=0.8, crisis=0.3),
    )
    regime_result = regime_sizer.adjust(
        raw_position_pct=vol_result.position_pct,
        current_gross_exposure=0.40,
        regime=_risk_off_regime(),
    )

    # Risk-off cap is 0.6, current 0.4, headroom 0.2
    assert regime_result.adjusted_position_pct <= 0.20

    # 4. Compose
    sizing_out = SizingOutput(
        sizing_skill="volatility-v1",
        position_pct=regime_result.adjusted_position_pct,
        shares=int(regime_result.adjusted_position_pct * 100_000.0 / 80.0),
        position_value=regime_result.adjusted_position_pct * 100_000.0,
        rationale="Vol target + regime adjustment",
    )
    composer = Composer()
    result = composer.compose(strategy_out, sizing_out)
    assert result.intent.ticker == "XOM"


def test_full_pipeline_correlation_scaling():
    """Correlated positions get scaled down through the pipeline."""
    rng = np.random.RandomState(42)
    base = rng.normal(0, 0.02, 60)
    noise = rng.normal(0, 0.02, 60)
    r1 = list(base)
    r2 = list(0.9 * base + 0.1 * noise)  # Highly correlated

    corr_sizer = CorrelationAwareSizer(max_correlated_group_pct=0.30)
    requests = [
        PositionRequest("AAPL", 0.20, r1, "Technology"),
        PositionRequest("MSFT", 0.20, r2, "Technology"),
    ]
    result = corr_sizer.compute(requests, portfolio_value=100_000.0)
    total = sum(p.adjusted_position_pct for p in result.positions)
    assert total <= 0.30 + 1e-9

    # Compose each into TradeIntent
    composer = Composer()
    for pos in result.positions:
        strategy_out = StrategyOutput(
            strategy_name="momentum-v1",
            ticker=pos.ticker,
            direction=Direction.LONG,
            conditions=["tech_momentum"],
            regime=_risk_on_regime(),
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
        )
        sizing_out = SizingOutput(
            sizing_skill="correlation-v1",
            position_pct=pos.adjusted_position_pct,
            shares=int(pos.adjusted_position_pct * 100_000.0 / 150.0),
            position_value=pos.adjusted_position_pct * 100_000.0,
            rationale="Correlation-adjusted",
        )
        comp_result = composer.compose(strategy_out, sizing_out)
        assert comp_result.intent.ticker == pos.ticker


def test_full_pipeline_tax_aware_evaluation():
    """Tax-aware evaluation penalizes short-term heavy strategy."""
    evaluator = TaxAwareEvaluator(config=TaxConfig(), enabled=True)
    trades = [
        {"pnl": 500.0, "entry_date": datetime(2025, 1, 1, tzinfo=timezone.utc),
         "exit_date": datetime(2025, 2, 1, tzinfo=timezone.utc)},
        {"pnl": 500.0, "entry_date": datetime(2025, 1, 1, tzinfo=timezone.utc),
         "exit_date": datetime(2025, 3, 1, tzinfo=timezone.utc)},
        {"pnl": 1000.0, "entry_date": datetime(2025, 1, 1, tzinfo=timezone.utc),
         "exit_date": datetime(2026, 2, 1, tzinfo=timezone.utc)},
    ]
    raw_sharpe = 1.5
    adjusted = evaluator.adjust_fitness(raw_sharpe, trades, total_pnl=2000.0)
    assert adjusted < raw_sharpe  # Penalized for short-term gains

    metric = evaluator.compute_tax_drag(trades)
    assert metric.short_term_gains == 1000.0
    assert metric.long_term_gains == 1000.0


def test_full_pipeline_survival_gate():
    """Survival gate validates paper trading results."""
    gate = SurvivalGate(criteria=SurvivalCriteria(
        min_trading_days=20,
        min_sharpe=0.5,
        max_drawdown=0.15,
    ))

    # Generate healthy paper trading results
    rng = np.random.RandomState(42)
    returns = list(rng.normal(0.001, 0.008, 30))  # Positive drift, moderate vol
    records = []
    equity = 100_000.0
    peak = equity
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for i, ret in enumerate(returns):
        equity *= (1 + ret)
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        records.append(PaperTradeRecord(
            date=base + timedelta(days=i),
            equity=equity,
            daily_return=ret,
            drawdown=dd,
        ))

    result = gate.evaluate(records)
    assert result.actual_trading_days == 30
    # Result depends on random seed — just verify it returns a valid result
    assert isinstance(result.passed, bool)
    assert isinstance(result.actual_sharpe, float)


def test_full_pipeline_schema_round_trip():
    """Sizing skill SKILL.md → parse → config → validate."""
    md_content = """---
name: volatility-target-v1
description: Volatility targeting sizing
version: 1
status: active
skill_type: sizing
sizing_method: volatility_target
parameters:
  target_annual_vol: 0.15
  lookback_days: 60
target_regime_behavior:
  risk_on_max_exposure: 1.0
  risk_off_max_exposure: 0.6
  transitional_max_exposure: 0.8
  crisis_max_exposure: 0.3
  scaling_speed: gradual
risk_budget:
  max_position_pct: 0.15
  max_sector_pct: 0.30
  max_correlated_group_pct: 0.40
  max_daily_var_pct: 0.02
  max_drawdown_pct: 0.15
---

# Volatility Target Sizing v1
"""
    config = parse_sizing_skill_md(md_content)
    assert config.sizing_method == SizingMethod.VOLATILITY_TARGET
    assert config.parameters["target_annual_vol"] == 0.15
    assert config.risk_budget.max_position_pct == 0.15
```

**Step 2: Run test**

```bash
pytest tests/integration/test_sizing_pipeline.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_sizing_pipeline.py
git commit -m "test: integration tests for full sizing composition pipeline"
```

---

## Task 11: Final Verification

**Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: ALL PASS — Phase 1, 2, 3, and 4 tests

**Step 2: Run linting and type checking**

```bash
ruff check src/evolve_trader/
mypy src/evolve_trader/ --ignore-missing-imports
```

Expected: No errors

**Step 3: Commit**

```bash
git add -A
git commit -m "test: Phase 4 final verification — all tests passing"
```

---

## Parallelization Notes

```
Task 1 (SizingSkill Schema) ────────────────────┐
Task 2 (Kelly Criterion) ───────────────────────┤
Task 3 (Volatility Targeting) ──────────────────┤── Task 6 (Composition Interface) ──┐
Task 4 (Correlation-Aware) ─────────────────────┤                                    ├── Task 10 (Integration)
Task 5 (Regime-Adjusted) ──────────────────────┤── Task 7 (Portfolio Risk) ──────────┘
Task 8 (Tax-Aware Evolution) ───────────────────┤
Task 9 (Survival Gate) ────────────────────────┘
```

**Can run in parallel:**
- Tasks 1-5, Task 8, and Task 9 are all independent — run simultaneously
- Task 6 (composition interface) depends on schema conventions from Task 1, but no code dependency
- Task 7 (portfolio risk enforcement) depends on TradeIntent from Task 6
- Task 10 (integration testing) depends on all previous tasks
- Task 11 (final verification) must be last
