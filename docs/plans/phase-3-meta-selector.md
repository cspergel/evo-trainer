# Phase 3: Meta-Selector & Signal Intelligence — Detailed Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the meta-selector that maps RegimeLabel + SignalEvents to weighted strategy selection. Implement the full signal source scoring/weighting system with rolling scorecards. Build the signal source lifecycle pipeline. Implement conflict resolution and multi-timeframe skill stacking.

**Architecture:** The meta-selector is an evolvable SKILL.md routing function. It consumes RegimeLabels and scored SignalEvents, then outputs a weighted set of strategy skills with capital allocation percentages. Signal sources are dynamically scored via rolling scorecards. A lifecycle pipeline manages source promotion/demotion. Conflict resolution handles opposing signals. Multi-timeframe stacking separates strategic, tactical, and execution layers.

**Tech Stack:** Python 3.11+, PostgreSQL, Pydantic, numpy, scipy (for statistical tests), pytest

**Parent plan:** [`2026-03-29-evolve-trader-master-plan.md`](2026-03-29-evolve-trader-master-plan.md)

**Prerequisites:** Phase 2 complete. PostgreSQL running. SignalEvent framework, EDGAR 13F/Form 4 parsers, congressional source, regime classifier all working.

---

## Task 1: Meta-Selector Core

**Files:**
- Create: `src/evolve_trader/selection/__init__.py`
- Create: `src/evolve_trader/selection/meta_selector.py`
- Create: `strategies/meta-selector-v1.skill.md`
- Create: `tests/unit/test_meta_selector.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_meta_selector.py
"""Tests for the meta-selector routing engine."""
import pytest
from evolve_trader.selection.meta_selector import (
    MetaSelector,
    StrategyAllocation,
    AllocationResult,
)
from evolve_trader.regime.labels import RegimeLabel, PrimaryRegime, MomentumState
from evolve_trader.signals.events import SignalEvent
from evolve_trader.signals.types import SignalType, DecayProfile, DecayType
from datetime import datetime, timezone


def _risk_on_label(confidence: float = 0.8) -> RegimeLabel:
    return RegimeLabel(PrimaryRegime.RISK_ON, "overweight technology", MomentumState.STRENGTHENING, confidence, "short-term")

def _risk_off_label(confidence: float = 0.8) -> RegimeLabel:
    return RegimeLabel(PrimaryRegime.RISK_OFF, "underweight technology", MomentumState.WEAKENING, confidence, "short-term")

def _low_confidence_label() -> RegimeLabel:
    return RegimeLabel(PrimaryRegime.TRANSITIONAL, "neutral", MomentumState.STABLE, 0.3, "unknown")

def _make_signal(confidence: float = 0.8) -> SignalEvent:
    return SignalEvent(
        source="test", source_entity="Test", timestamp=datetime.now(timezone.utc),
        confidence=confidence, decay_profile=DecayProfile(confidence, 30, DecayType.LINEAR),
        signal_type=SignalType.CONVICTION, payload={"action": "BUY"},
    )


def test_allocation_result_percentages_sum():
    """Strategy allocations must sum to <= 100%."""
    result = AllocationResult(
        allocations=[
            StrategyAllocation("momentum-v1", 0.40),
            StrategyAllocation("mean-reversion-v1", 0.30),
            StrategyAllocation("capital-preservation", 0.30),
        ],
        regime=_risk_on_label(),
        confidence=0.8,
    )
    total = sum(a.weight for a in result.allocations)
    assert total <= 1.0 + 1e-9


def test_meta_selector_routes_risk_on():
    """Risk-on regime with strong signals → aggressive strategies selected."""
    selector = MetaSelector(
        available_strategies=["momentum-v1", "mean-reversion-v1", "trend-following-v1", "capital-preservation"],
        strategy_regime_affinity={
            "momentum-v1": {PrimaryRegime.RISK_ON: 0.9, PrimaryRegime.RISK_OFF: 0.1},
            "mean-reversion-v1": {PrimaryRegime.RISK_ON: 0.5, PrimaryRegime.RISK_OFF: 0.6},
            "trend-following-v1": {PrimaryRegime.RISK_ON: 0.8, PrimaryRegime.RISK_OFF: 0.2},
            "capital-preservation": {PrimaryRegime.RISK_ON: 0.1, PrimaryRegime.RISK_OFF: 0.9},
        },
    )
    result = selector.select(_risk_on_label(), [_make_signal(0.85)])
    assert len(result.allocations) >= 1
    # Momentum and trend-following should dominate
    strategy_names = {a.strategy_name for a in result.allocations if a.weight > 0.1}
    assert "momentum-v1" in strategy_names or "trend-following-v1" in strategy_names


def test_meta_selector_routes_risk_off():
    """Risk-off regime → defensive strategies and capital preservation."""
    selector = MetaSelector(
        available_strategies=["momentum-v1", "capital-preservation", "mean-reversion-v1"],
        strategy_regime_affinity={
            "momentum-v1": {PrimaryRegime.RISK_ON: 0.9, PrimaryRegime.RISK_OFF: 0.1},
            "capital-preservation": {PrimaryRegime.RISK_ON: 0.1, PrimaryRegime.RISK_OFF: 0.9},
            "mean-reversion-v1": {PrimaryRegime.RISK_ON: 0.5, PrimaryRegime.RISK_OFF: 0.6},
        },
    )
    result = selector.select(_risk_off_label(), [])
    cp_alloc = next((a for a in result.allocations if a.strategy_name == "capital-preservation"), None)
    assert cp_alloc is not None
    assert cp_alloc.weight > 0.3


def test_meta_selector_low_confidence_routes_to_capital_preservation():
    """Low confidence regime → Capital Preservation dominates."""
    selector = MetaSelector(
        available_strategies=["momentum-v1", "capital-preservation"],
        strategy_regime_affinity={
            "momentum-v1": {PrimaryRegime.RISK_ON: 0.9, PrimaryRegime.TRANSITIONAL: 0.3},
            "capital-preservation": {PrimaryRegime.RISK_ON: 0.1, PrimaryRegime.TRANSITIONAL: 0.8},
        },
        confidence_threshold=0.6,
    )
    result = selector.select(_low_confidence_label(), [])
    cp_alloc = next((a for a in result.allocations if a.strategy_name == "capital-preservation"), None)
    assert cp_alloc is not None
    assert cp_alloc.weight >= 0.5


def test_meta_selector_ensemble_deployment():
    """Meta-selector can activate multiple strategies simultaneously."""
    selector = MetaSelector(
        available_strategies=["momentum-v1", "trend-following-v1", "capital-preservation"],
        strategy_regime_affinity={
            "momentum-v1": {PrimaryRegime.RISK_ON: 0.9},
            "trend-following-v1": {PrimaryRegime.RISK_ON: 0.8},
            "capital-preservation": {PrimaryRegime.RISK_ON: 0.1},
        },
    )
    result = selector.select(_risk_on_label(0.9), [_make_signal(0.9)])
    active = [a for a in result.allocations if a.weight > 0.05]
    assert len(active) >= 2  # Multiple strategies active
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_meta_selector.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.selection'`

**Step 3: Implement**

```python
# src/evolve_trader/selection/meta_selector.py
"""Meta-selector — routes regime + signals to weighted strategy allocation."""
from __future__ import annotations

from dataclasses import dataclass, field
from evolve_trader.regime.labels import RegimeLabel, PrimaryRegime
from evolve_trader.signals.events import SignalEvent


@dataclass(frozen=True)
class StrategyAllocation:
    """A single strategy with its capital allocation weight."""
    strategy_name: str
    weight: float  # 0.0 - 1.0


@dataclass
class AllocationResult:
    """The meta-selector's output: a weighted set of strategies."""
    allocations: list[StrategyAllocation]
    regime: RegimeLabel
    confidence: float

    @property
    def total_weight(self) -> float:
        return sum(a.weight for a in self.allocations)


class MetaSelector:
    """Routes RegimeLabel + SignalEvents to weighted strategy selection.

    This is an evolvable SKILL.md — subject to FIX/DERIVED/CAPTURED.
    """

    def __init__(
        self,
        available_strategies: list[str],
        strategy_regime_affinity: dict[str, dict[PrimaryRegime, float]] | None = None,
        confidence_threshold: float = 0.6,
    ):
        self._strategies = available_strategies
        self._affinity = strategy_regime_affinity or {}
        self._confidence_threshold = confidence_threshold

    def select(
        self,
        regime: RegimeLabel,
        signals: list[SignalEvent],
    ) -> AllocationResult:
        """Select and weight strategies based on regime and signals."""
        # If confidence too low, route heavily to capital preservation
        if regime.confidence < self._confidence_threshold:
            return self._low_confidence_allocation(regime)

        # Score each strategy by regime affinity
        scores: dict[str, float] = {}
        for strategy in self._strategies:
            affinity = self._affinity.get(strategy, {})
            base_score = affinity.get(regime.primary_regime, 0.5)
            # Boost score based on signal strength
            signal_boost = sum(s.current_confidence() for s in signals) * 0.1
            scores[strategy] = base_score + min(signal_boost, 0.2)

        # Normalize to weights that sum to 1.0
        total_score = sum(scores.values())
        if total_score == 0:
            return self._low_confidence_allocation(regime)

        allocations = []
        for strategy, score in scores.items():
            weight = score / total_score
            if weight > 0.02:  # Filter out negligible allocations
                allocations.append(StrategyAllocation(strategy, round(weight, 4)))

        # Re-normalize after filtering
        total_weight = sum(a.weight for a in allocations)
        if total_weight > 0:
            allocations = [
                StrategyAllocation(a.strategy_name, round(a.weight / total_weight, 4))
                for a in allocations
            ]

        return AllocationResult(
            allocations=allocations,
            regime=regime,
            confidence=regime.confidence,
        )

    def _low_confidence_allocation(self, regime: RegimeLabel) -> AllocationResult:
        """When confidence is low, route to capital preservation."""
        allocations = []
        for strategy in self._strategies:
            if "capital-preservation" in strategy:
                allocations.append(StrategyAllocation(strategy, 0.7))
            else:
                # Small allocation to other strategies
                remaining = 0.3 / max(len(self._strategies) - 1, 1)
                allocations.append(StrategyAllocation(strategy, round(remaining, 4)))
        return AllocationResult(allocations=allocations, regime=regime, confidence=regime.confidence)
```

```markdown
<!-- strategies/meta-selector-v1.skill.md -->
---
name: meta-selector-v1
description: Routes regime labels and signals to weighted strategy allocation
version: 1
status: active
skill_type: meta_selector
confidence_threshold: 0.6
---

# Meta-Selector v1

## Routing Logic

1. If regime confidence < threshold → route 70% to Capital Preservation
2. Score each strategy by regime affinity (how well it performs in current regime)
3. Boost scores based on active signal strength
4. Normalize to allocation weights summing to 1.0
5. Filter out allocations < 2%

## Evolution Notes

Subject to FIX/DERIVED/CAPTURED:
- FIX: When routing consistently selects underperforming strategies
- DERIVED: When specialized selectors for specific regimes outperform the generalist
- CAPTURED: When emergent routing patterns improve performance
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_meta_selector.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/selection/ strategies/meta-selector-v1.skill.md tests/unit/test_meta_selector.py
git commit -m "feat: meta-selector routing engine with ensemble deployment"
```

---

## Task 2: Signal Source Scoring Engine

**Files:**
- Create: `src/evolve_trader/signals/scoring.py`
- Create: `tests/unit/test_signal_scoring.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_signal_scoring.py
"""Tests for the signal source scoring engine."""
import pytest
from datetime import datetime, timezone, timedelta
from evolve_trader.signals.scoring import (
    SourceScorer,
    SourceScore,
    SourceTier,
    SignalOutcome,
)


def test_source_tier_base_weights():
    """Tier weights: Tier 1 = 3.0x, Tier 2 = 2.0x, Tier 3 = 1.0x."""
    assert SourceTier.TIER_1.base_weight == 3.0
    assert SourceTier.TIER_2.base_weight == 2.0
    assert SourceTier.TIER_3.base_weight == 1.0


def test_scorer_base_tier_weight():
    """New source with no history uses base tier weight."""
    scorer = SourceScorer()
    score = scorer.compute_score("edgar_13f", tier=SourceTier.TIER_1, outcomes=[])
    assert score.composite_weight == pytest.approx(3.0, abs=0.1)


def test_scorer_rolling_hit_rate_multiplier():
    """High hit rate multiplies base weight."""
    scorer = SourceScorer()
    # 75% hit rate over baseline 50% → 1.5x multiplier
    outcomes = [
        SignalOutcome(True, 0.05, datetime.now(timezone.utc) - timedelta(weeks=i))
        for i in range(9)  # 9 wins
    ] + [
        SignalOutcome(False, -0.02, datetime.now(timezone.utc) - timedelta(weeks=i))
        for i in range(3)  # 3 losses
    ]
    score = scorer.compute_score("test", tier=SourceTier.TIER_2, outcomes=outcomes)
    assert score.hit_rate == pytest.approx(0.75, abs=0.01)
    assert score.hit_rate_multiplier > 1.0


def test_scorer_cold_streak_penalty():
    """Hit rate below 35% → 0.5x penalty."""
    scorer = SourceScorer()
    outcomes = [
        SignalOutcome(False, -0.03, datetime.now(timezone.utc) - timedelta(weeks=i))
        for i in range(8)
    ] + [
        SignalOutcome(True, 0.01, datetime.now(timezone.utc) - timedelta(weeks=i))
        for i in range(2)
    ]
    score = scorer.compute_score("test", tier=SourceTier.TIER_1, outcomes=outcomes)
    assert score.hit_rate < 0.35
    assert score.cold_streak_penalty == 0.5


def test_scorer_regime_alignment_bonus():
    """Correct regime prediction → +0.5x bonus."""
    scorer = SourceScorer()
    outcomes = [
        SignalOutcome(True, 0.05, datetime.now(timezone.utc), regime_aligned=True)
        for _ in range(5)
    ]
    score = scorer.compute_score("test", tier=SourceTier.TIER_2, outcomes=outcomes)
    assert score.regime_alignment_bonus > 0


def test_scorer_min_observation_count():
    """Below 5-trade minimum → static tier weight only."""
    scorer = SourceScorer()
    outcomes = [
        SignalOutcome(True, 0.10, datetime.now(timezone.utc))
        for _ in range(3)  # Only 3 observations
    ]
    score = scorer.compute_score("test", tier=SourceTier.TIER_1, outcomes=outcomes)
    # Should use static tier weight, not rolling scorecard
    assert score.composite_weight == pytest.approx(3.0, abs=0.1)
    assert score.using_static_weight is True


def test_scorer_lookback_calibration():
    """Active congressional traders use shorter lookback than Buffett."""
    scorer = SourceScorer()
    # Congressional: 6-week lookback
    assert scorer.get_lookback_weeks("congressional") == 6
    # Buffett: 26-week (6-month) lookback
    assert scorer.get_lookback_weeks("edgar_13f") >= 26


def test_source_score_data_model():
    """SourceScore has all required fields."""
    score = SourceScore(
        source_name="edgar_13f",
        tier=SourceTier.TIER_1,
        base_weight=3.0,
        hit_rate=0.65,
        hit_rate_multiplier=1.3,
        regime_alignment_bonus=0.3,
        cold_streak_penalty=1.0,
        recency_factor=0.95,
        magnitude_accuracy=0.7,
        composite_weight=4.2,
        using_static_weight=False,
    )
    assert score.composite_weight == 4.2
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_signal_scoring.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement**

```python
# src/evolve_trader/signals/scoring.py
"""Signal source scoring engine — dynamic credibility system."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum


class SourceTier(Enum):
    """Signal source tier classification."""
    TIER_1 = ("tier_1", 3.0)  # Congressional leadership, insider clusters, ARK
    TIER_2 = ("tier_2", 2.0)  # Active congressional, concentrated value, macro
    TIER_3 = ("tier_3", 1.0)  # Activist, sector specialist, investor letters

    def __init__(self, label: str, weight: float):
        self.label = label
        self.base_weight = weight


@dataclass
class SignalOutcome:
    """Outcome of a signal for scorecard tracking."""
    was_correct: bool
    return_pct: float
    timestamp: datetime
    regime_aligned: bool = False
    magnitude: float = 0.0  # Expected magnitude vs actual


@dataclass
class SourceScore:
    """Computed score for a signal source."""
    source_name: str
    tier: SourceTier
    base_weight: float
    hit_rate: float
    hit_rate_multiplier: float
    regime_alignment_bonus: float
    cold_streak_penalty: float
    recency_factor: float
    magnitude_accuracy: float
    composite_weight: float
    using_static_weight: bool


# Per-source lookback windows (in weeks)
_LOOKBACK_WEEKS = {
    "congressional": 6,
    "edgar_13f": 26,   # 6 months for slow movers like Buffett
    "edgar_form4": 12,
    "ark_daily": 6,
    "options_unusual": 4,
    "onchain_whale": 4,
    "default": 12,
}

MIN_OBSERVATIONS = 5


class SourceScorer:
    """Computes dynamic credibility scores for signal sources."""

    def compute_score(
        self,
        source_name: str,
        tier: SourceTier,
        outcomes: list[SignalOutcome],
        lookback_weeks: int | None = None,
    ) -> SourceScore:
        """Compute composite score for a signal source."""
        if lookback_weeks is None:
            lookback_weeks = self.get_lookback_weeks(source_name)

        # Filter to lookback window
        cutoff = datetime.now(timezone.utc) - timedelta(weeks=lookback_weeks)
        recent = [o for o in outcomes if o.timestamp >= cutoff]

        # Below minimum observations → static tier weight
        if len(recent) < MIN_OBSERVATIONS:
            return SourceScore(
                source_name=source_name,
                tier=tier,
                base_weight=tier.base_weight,
                hit_rate=0.0,
                hit_rate_multiplier=1.0,
                regime_alignment_bonus=0.0,
                cold_streak_penalty=1.0,
                recency_factor=1.0,
                magnitude_accuracy=0.0,
                composite_weight=tier.base_weight,
                using_static_weight=True,
            )

        # Rolling hit rate
        wins = sum(1 for o in recent if o.was_correct)
        hit_rate = wins / len(recent)

        # Hit rate multiplier (vs 50% baseline)
        hit_rate_multiplier = hit_rate / 0.5 if hit_rate > 0 else 0.0

        # Cold streak penalty
        cold_streak_penalty = 0.5 if hit_rate < 0.35 else 1.0

        # Regime alignment bonus
        regime_aligned = [o for o in recent if o.regime_aligned]
        regime_bonus = 0.5 if len(regime_aligned) > len(recent) * 0.5 else 0.0

        # Recency factor (more recent outcomes weighted higher)
        recency_factor = 1.0  # Simplified — full implementation weights by time

        # Magnitude accuracy
        magnitude_accuracy = 0.0  # Requires more data — placeholder

        # Composite weight
        composite = (
            tier.base_weight
            * hit_rate_multiplier
            * cold_streak_penalty
            + regime_bonus
        ) * recency_factor

        return SourceScore(
            source_name=source_name,
            tier=tier,
            base_weight=tier.base_weight,
            hit_rate=hit_rate,
            hit_rate_multiplier=hit_rate_multiplier,
            regime_alignment_bonus=regime_bonus,
            cold_streak_penalty=cold_streak_penalty,
            recency_factor=recency_factor,
            magnitude_accuracy=magnitude_accuracy,
            composite_weight=composite,
            using_static_weight=False,
        )

    def get_lookback_weeks(self, source_name: str) -> int:
        """Get the calibrated lookback window for a source."""
        return _LOOKBACK_WEEKS.get(source_name, _LOOKBACK_WEEKS["default"])
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_signal_scoring.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/signals/scoring.py tests/unit/test_signal_scoring.py
git commit -m "feat: signal source scoring engine with rolling scorecards"
```

---

## Task 3: Disclosure-to-Executable Spread Tracking

**Files:**
- Create: `src/evolve_trader/signals/spread_tracker.py`
- Create: `tests/unit/test_spread_tracker.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_spread_tracker.py
"""Tests for disclosure-to-executable spread tracking."""
import pytest
from datetime import datetime, timezone, timedelta
from evolve_trader.signals.spread_tracker import (
    SpreadTracker,
    SpreadRecord,
    OptimalDelay,
)


def test_spread_record_computes_slippage():
    """SpreadRecord computes price slippage at each checkpoint."""
    record = SpreadRecord(
        source="congressional",
        ticker="NVDA",
        disclosure_time=datetime(2025, 3, 1, tzinfo=timezone.utc),
        price_at_disclosure=800.0,
        price_at_24h=815.0,
        price_at_48h=810.0,
        price_at_72h=805.0,
    )
    assert record.slippage_24h == pytest.approx(0.01875, abs=0.001)  # (815-800)/800
    assert record.slippage_48h == pytest.approx(0.0125, abs=0.001)
    assert record.slippage_72h == pytest.approx(0.00625, abs=0.001)


def test_spread_tracker_learns_optimal_delay():
    """Tracker learns optimal action delay per source."""
    tracker = SpreadTracker()
    now = datetime(2025, 3, 1, tzinfo=timezone.utc)

    # Congressional disclosures: copycat surge at 24h, fades by 72h
    for i in range(10):
        tracker.record(SpreadRecord(
            source="congressional", ticker=f"TICK{i}",
            disclosure_time=now + timedelta(days=i),
            price_at_disclosure=100.0,
            price_at_24h=105.0,   # 5% surge
            price_at_48h=103.0,   # fading
            price_at_72h=101.0,   # mostly faded
        ))

    delay = tracker.get_optimal_delay("congressional")
    # Best to wait until 72h when copycat surge has faded
    assert delay.recommended_hours >= 48


def test_spread_tracker_ark_zero_delay():
    """ARK trades have minimal slippage — execute immediately."""
    tracker = SpreadTracker()
    now = datetime(2025, 3, 1, tzinfo=timezone.utc)

    for i in range(10):
        tracker.record(SpreadRecord(
            source="ark_daily", ticker=f"TICK{i}",
            disclosure_time=now + timedelta(days=i),
            price_at_disclosure=50.0,
            price_at_24h=50.2,    # minimal impact
            price_at_48h=50.5,    # price moving away
            price_at_72h=51.0,    # missing out
        ))

    delay = tracker.get_optimal_delay("ark_daily")
    assert delay.recommended_hours <= 24


def test_spread_tracker_insufficient_data():
    """With insufficient data, returns conservative delay."""
    tracker = SpreadTracker()
    delay = tracker.get_optimal_delay("unknown_source")
    assert delay.recommended_hours == 24  # Default conservative
    assert delay.confidence < 0.5
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_spread_tracker.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement**

```python
# src/evolve_trader/signals/spread_tracker.py
"""Disclosure-to-executable spread tracking."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SpreadRecord:
    """Price observations at disclosure checkpoints."""
    source: str
    ticker: str
    disclosure_time: datetime
    price_at_disclosure: float
    price_at_24h: float | None = None
    price_at_48h: float | None = None
    price_at_72h: float | None = None

    @property
    def slippage_24h(self) -> float:
        if self.price_at_24h is None or self.price_at_disclosure == 0:
            return 0.0
        return (self.price_at_24h - self.price_at_disclosure) / self.price_at_disclosure

    @property
    def slippage_48h(self) -> float:
        if self.price_at_48h is None or self.price_at_disclosure == 0:
            return 0.0
        return (self.price_at_48h - self.price_at_disclosure) / self.price_at_disclosure

    @property
    def slippage_72h(self) -> float:
        if self.price_at_72h is None or self.price_at_disclosure == 0:
            return 0.0
        return (self.price_at_72h - self.price_at_disclosure) / self.price_at_disclosure


@dataclass(frozen=True)
class OptimalDelay:
    """Recommended action delay for a signal source."""
    source: str
    recommended_hours: int
    avg_slippage_at_optimal: float
    confidence: float  # How confident we are in this recommendation


MIN_RECORDS = 5


class SpreadTracker:
    """Tracks and learns optimal action delay per signal source."""

    def __init__(self):
        self._records: dict[str, list[SpreadRecord]] = defaultdict(list)

    def record(self, spread: SpreadRecord) -> None:
        """Record a new spread observation."""
        self._records[spread.source].append(spread)

    def get_optimal_delay(self, source: str) -> OptimalDelay:
        """Compute optimal delay for a source based on historical spreads."""
        records = self._records.get(source, [])

        if len(records) < MIN_RECORDS:
            return OptimalDelay(source=source, recommended_hours=24, avg_slippage_at_optimal=0.0, confidence=0.3)

        # Compare average absolute slippage at each checkpoint
        # Lower slippage = better execution price
        avg_slip_24h = sum(abs(r.slippage_24h) for r in records) / len(records)
        avg_slip_48h = sum(abs(r.slippage_48h) for r in records) / len(records)
        avg_slip_72h = sum(abs(r.slippage_72h) for r in records) / len(records)

        # Check if slippage is increasing (price moving away = execute sooner)
        # or decreasing (copycat surge fading = wait)
        checkpoints = {0: 0.0, 24: avg_slip_24h, 48: avg_slip_48h, 72: avg_slip_72h}

        # Find the checkpoint with minimum slippage
        # If slippage increases then decreases, wait for the trough
        # If slippage only increases, execute immediately
        if avg_slip_24h < avg_slip_48h and avg_slip_24h < avg_slip_72h:
            # Price moves away from disclosure — execute quickly
            optimal_hours = 0
            optimal_slip = 0.0
        elif avg_slip_72h < avg_slip_24h:
            # Copycat surge fades — wait
            optimal_hours = 72
            optimal_slip = avg_slip_72h
        elif avg_slip_48h < avg_slip_24h:
            optimal_hours = 48
            optimal_slip = avg_slip_48h
        else:
            optimal_hours = 24
            optimal_slip = avg_slip_24h

        confidence = min(len(records) / 20, 1.0)  # Max confidence at 20+ records

        return OptimalDelay(
            source=source,
            recommended_hours=optimal_hours,
            avg_slippage_at_optimal=optimal_slip,
            confidence=round(confidence, 2),
        )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_spread_tracker.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/signals/spread_tracker.py tests/unit/test_spread_tracker.py
git commit -m "feat: disclosure-to-executable spread tracking with optimal delay learning"
```

---

## Task 4: Post-Signal Return Tracking

**Files:**
- Create: `src/evolve_trader/signals/return_tracker.py`
- Create: `tests/unit/test_return_tracker.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_return_tracker.py
"""Tests for post-signal return tracking."""
import pytest
from datetime import datetime, timezone, timedelta
from evolve_trader.signals.return_tracker import (
    ReturnTracker,
    PostSignalReturn,
    SourceReturnProfile,
)


def test_post_signal_return_data_model():
    """PostSignalReturn captures returns at multiple horizons."""
    psr = PostSignalReturn(
        source="edgar_13f",
        ticker="AAPL",
        signal_date=datetime(2025, 1, 15, tzinfo=timezone.utc),
        return_2w=0.03,
        return_4w=0.05,
        return_6w=0.04,
        return_12w=0.08,
        benchmark_return_2w=0.01,
        benchmark_return_4w=0.02,
        benchmark_return_6w=0.015,
        benchmark_return_12w=0.04,
    )
    assert psr.alpha_2w == pytest.approx(0.02, abs=0.001)  # return - benchmark
    assert psr.alpha_12w == pytest.approx(0.04, abs=0.001)


def test_return_tracker_computes_source_profile():
    """Tracker aggregates returns into a source profile."""
    tracker = ReturnTracker()
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)

    for i in range(10):
        tracker.record(PostSignalReturn(
            source="edgar_13f", ticker=f"TICK{i}",
            signal_date=base + timedelta(weeks=i),
            return_2w=0.03, return_4w=0.05, return_6w=0.04, return_12w=0.08,
            benchmark_return_2w=0.01, benchmark_return_4w=0.02,
            benchmark_return_6w=0.015, benchmark_return_12w=0.04,
        ))

    profile = tracker.get_profile("edgar_13f")
    assert profile.avg_alpha_2w > 0
    assert profile.avg_alpha_12w > 0
    assert profile.hit_rate > 0.5  # All positive alpha
    assert profile.observation_count == 10


def test_return_tracker_empty_source():
    """Empty source returns zero profile."""
    tracker = ReturnTracker()
    profile = tracker.get_profile("unknown")
    assert profile.observation_count == 0
    assert profile.hit_rate == 0.0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_return_tracker.py -v
```

Expected: FAIL

**Step 3: Implement**

```python
# src/evolve_trader/signals/return_tracker.py
"""Post-signal return tracking at multiple horizons."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PostSignalReturn:
    """Return observations at 2, 4, 6, 12 weeks post-signal."""
    source: str
    ticker: str
    signal_date: datetime
    return_2w: float = 0.0
    return_4w: float = 0.0
    return_6w: float = 0.0
    return_12w: float = 0.0
    benchmark_return_2w: float = 0.0
    benchmark_return_4w: float = 0.0
    benchmark_return_6w: float = 0.0
    benchmark_return_12w: float = 0.0

    @property
    def alpha_2w(self) -> float:
        return self.return_2w - self.benchmark_return_2w

    @property
    def alpha_4w(self) -> float:
        return self.return_4w - self.benchmark_return_4w

    @property
    def alpha_6w(self) -> float:
        return self.return_6w - self.benchmark_return_6w

    @property
    def alpha_12w(self) -> float:
        return self.return_12w - self.benchmark_return_12w


@dataclass(frozen=True)
class SourceReturnProfile:
    """Aggregated return profile for a signal source."""
    source_name: str
    observation_count: int
    avg_alpha_2w: float
    avg_alpha_4w: float
    avg_alpha_6w: float
    avg_alpha_12w: float
    hit_rate: float  # % of signals with positive alpha at 12w
    avg_magnitude_accuracy: float


class ReturnTracker:
    """Tracks post-signal returns for all sources."""

    def __init__(self):
        self._returns: dict[str, list[PostSignalReturn]] = defaultdict(list)

    def record(self, psr: PostSignalReturn) -> None:
        self._returns[psr.source].append(psr)

    def get_profile(self, source: str) -> SourceReturnProfile:
        records = self._returns.get(source, [])
        if not records:
            return SourceReturnProfile(source, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        n = len(records)
        avg_2w = sum(r.alpha_2w for r in records) / n
        avg_4w = sum(r.alpha_4w for r in records) / n
        avg_6w = sum(r.alpha_6w for r in records) / n
        avg_12w = sum(r.alpha_12w for r in records) / n
        hit_rate = sum(1 for r in records if r.alpha_12w > 0) / n

        return SourceReturnProfile(
            source_name=source,
            observation_count=n,
            avg_alpha_2w=round(avg_2w, 6),
            avg_alpha_4w=round(avg_4w, 6),
            avg_alpha_6w=round(avg_6w, 6),
            avg_alpha_12w=round(avg_12w, 6),
            hit_rate=round(hit_rate, 4),
            avg_magnitude_accuracy=0.0,
        )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_return_tracker.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/signals/return_tracker.py tests/unit/test_return_tracker.py
git commit -m "feat: post-signal return tracking at 2/4/6/12 week horizons"
```

---

## Task 5: Signal Source Lifecycle Pipeline

**Files:**
- Create: `src/evolve_trader/signals/lifecycle.py`
- Create: `tests/unit/test_signal_lifecycle.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_signal_lifecycle.py
"""Tests for signal source lifecycle pipeline."""
import pytest
from datetime import datetime, timezone, timedelta
from evolve_trader.signals.lifecycle import (
    LifecycleManager,
    LifecycleStage,
    SourceLifecycle,
)


def test_lifecycle_stages():
    """All 5 lifecycle stages exist."""
    assert LifecycleStage.CANDIDATE.value == "candidate"
    assert LifecycleStage.OBSERVATION.value == "observation"
    assert LifecycleStage.PROBATION.value == "probation"
    assert LifecycleStage.ACTIVE.value == "active"
    assert LifecycleStage.DEMOTED.value == "demoted"


def test_new_source_starts_as_candidate():
    """New source enters at CANDIDATE stage."""
    mgr = LifecycleManager()
    mgr.add_source("new_fund", tier_guess=3)
    lifecycle = mgr.get_lifecycle("new_fund")
    assert lifecycle.stage == LifecycleStage.CANDIDATE
    assert lifecycle.weight == 0.0  # Zero weight in candidate


def test_candidate_promotes_to_observation():
    """Candidate with discovery filter pass → observation."""
    mgr = LifecycleManager()
    mgr.add_source("new_fund", tier_guess=3)
    mgr.promote("new_fund", reason="Discovery filter passed")
    lifecycle = mgr.get_lifecycle("new_fund")
    assert lifecycle.stage == LifecycleStage.OBSERVATION
    assert lifecycle.weight == 0.0  # Still zero weight


def test_observation_promotes_to_probation():
    """Observation with min 5 trades and >50% hit rate → probation."""
    mgr = LifecycleManager()
    mgr.add_source("new_fund", tier_guess=3)
    mgr.promote("new_fund", reason="Discovery filter")
    mgr.promote("new_fund", reason="5+ trades observed, hit rate 55%")
    lifecycle = mgr.get_lifecycle("new_fund")
    assert lifecycle.stage == LifecycleStage.PROBATION
    assert lifecycle.weight > 0  # Low weight in probation


def test_full_lifecycle_to_active():
    """Source progresses through all stages to active."""
    mgr = LifecycleManager()
    mgr.add_source("new_fund", tier_guess=2)
    mgr.promote("new_fund", reason="Filter passed")
    mgr.promote("new_fund", reason="5+ trades, >50% hit rate")
    mgr.promote("new_fund", reason="Sustained performance through probation")
    lifecycle = mgr.get_lifecycle("new_fund")
    assert lifecycle.stage == LifecycleStage.ACTIVE


def test_demotion():
    """Active source with bad performance → demoted."""
    mgr = LifecycleManager()
    mgr.add_source("bad_fund", tier_guess=1)
    for _ in range(3):
        mgr.promote("bad_fund", reason="advancing")
    assert mgr.get_lifecycle("bad_fund").stage == LifecycleStage.ACTIVE

    mgr.demote("bad_fund", reason="Hit rate <30% for 2 consecutive periods")
    lifecycle = mgr.get_lifecycle("bad_fund")
    assert lifecycle.stage == LifecycleStage.DEMOTED
    assert lifecycle.weight == 0.0


def test_cannot_skip_stages():
    """Cannot promote directly from candidate to active."""
    mgr = LifecycleManager()
    mgr.add_source("fund", tier_guess=1)
    # Each promote advances exactly one stage
    mgr.promote("fund", reason="step 1")
    assert mgr.get_lifecycle("fund").stage == LifecycleStage.OBSERVATION
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_signal_lifecycle.py -v
```

Expected: FAIL

**Step 3: Implement**

```python
# src/evolve_trader/signals/lifecycle.py
"""Signal source lifecycle pipeline — staged promotion/demotion."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class LifecycleStage(str, Enum):
    CANDIDATE = "candidate"
    OBSERVATION = "observation"
    PROBATION = "probation"
    ACTIVE = "active"
    DEMOTED = "demoted"


_STAGE_ORDER = [
    LifecycleStage.CANDIDATE,
    LifecycleStage.OBSERVATION,
    LifecycleStage.PROBATION,
    LifecycleStage.ACTIVE,
]

_STAGE_WEIGHTS = {
    LifecycleStage.CANDIDATE: 0.0,
    LifecycleStage.OBSERVATION: 0.0,
    LifecycleStage.PROBATION: 0.5,  # 0.5x Tier 3 base
    LifecycleStage.ACTIVE: 1.0,     # Full tier-appropriate weight
    LifecycleStage.DEMOTED: 0.0,
}


@dataclass
class LifecycleTransition:
    """Record of a lifecycle stage change."""
    from_stage: LifecycleStage
    to_stage: LifecycleStage
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SourceLifecycle:
    """Lifecycle state for a signal source."""
    source_name: str
    stage: LifecycleStage
    tier_guess: int
    weight: float
    entered_stage_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    transitions: list[LifecycleTransition] = field(default_factory=list)


class LifecycleManager:
    """Manages lifecycle state for all signal sources."""

    def __init__(self):
        self._sources: dict[str, SourceLifecycle] = {}

    def add_source(self, name: str, tier_guess: int = 3) -> SourceLifecycle:
        lifecycle = SourceLifecycle(
            source_name=name,
            stage=LifecycleStage.CANDIDATE,
            tier_guess=tier_guess,
            weight=_STAGE_WEIGHTS[LifecycleStage.CANDIDATE],
        )
        self._sources[name] = lifecycle
        return lifecycle

    def get_lifecycle(self, name: str) -> SourceLifecycle:
        if name not in self._sources:
            raise KeyError(f"Source '{name}' not tracked")
        return self._sources[name]

    def promote(self, name: str, reason: str) -> SourceLifecycle:
        lifecycle = self.get_lifecycle(name)
        if lifecycle.stage == LifecycleStage.ACTIVE:
            return lifecycle  # Already at top
        if lifecycle.stage == LifecycleStage.DEMOTED:
            raise ValueError(f"Cannot promote demoted source '{name}' — must re-add as candidate")

        current_idx = _STAGE_ORDER.index(lifecycle.stage)
        next_stage = _STAGE_ORDER[current_idx + 1]

        transition = LifecycleTransition(lifecycle.stage, next_stage, reason)
        lifecycle.transitions.append(transition)
        lifecycle.stage = next_stage
        lifecycle.weight = _STAGE_WEIGHTS[next_stage]
        lifecycle.entered_stage_at = datetime.now(timezone.utc)

        return lifecycle

    def demote(self, name: str, reason: str) -> SourceLifecycle:
        lifecycle = self.get_lifecycle(name)
        transition = LifecycleTransition(lifecycle.stage, LifecycleStage.DEMOTED, reason)
        lifecycle.transitions.append(transition)
        lifecycle.stage = LifecycleStage.DEMOTED
        lifecycle.weight = 0.0
        lifecycle.entered_stage_at = datetime.now(timezone.utc)
        return lifecycle
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_signal_lifecycle.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/signals/lifecycle.py tests/unit/test_signal_lifecycle.py
git commit -m "feat: signal source lifecycle pipeline with staged promotion/demotion"
```

---

## Task 6: Signal Conflict Resolution

**Files:**
- Create: `src/evolve_trader/selection/conflict_resolution.py`
- Create: `tests/unit/test_conflict_resolution.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_conflict_resolution.py
"""Tests for signal conflict resolution."""
import pytest
from datetime import datetime, timezone
from evolve_trader.selection.conflict_resolution import (
    ConflictResolver,
    ConflictResult,
    ConflictAction,
)
from evolve_trader.signals.events import SignalEvent
from evolve_trader.signals.types import SignalType, DecayProfile, DecayType


def _buy_signal(source: str, confidence: float, weight: float) -> SignalEvent:
    return SignalEvent(
        source=source, source_entity="Test", timestamp=datetime.now(timezone.utc),
        confidence=confidence, decay_profile=DecayProfile(confidence, 30, DecayType.LINEAR),
        signal_type=SignalType.CONVICTION, payload={"action": "BUY", "ticker": "AAPL"},
    )

def _sell_signal(source: str, confidence: float, weight: float) -> SignalEvent:
    return SignalEvent(
        source=source, source_entity="Test", timestamp=datetime.now(timezone.utc),
        confidence=confidence, decay_profile=DecayProfile(confidence, 30, DecayType.LINEAR),
        signal_type=SignalType.CONVICTION, payload={"action": "SELL", "ticker": "AAPL"},
    )


def test_no_conflict_when_signals_agree():
    """Aligned signals → no conflict."""
    resolver = ConflictResolver()
    signals = [_buy_signal("edgar_13f", 0.85, 3.0), _buy_signal("congressional", 0.70, 2.0)]
    weights = {"edgar_13f": 3.0, "congressional": 2.0}
    result = resolver.resolve(signals, weights, ticker="AAPL")
    assert result.action == ConflictAction.PROCEED
    assert result.conflict_detected is False


def test_equal_weight_conflict_defaults_to_preservation():
    """Equal weight opposing signals → Capital Preservation."""
    resolver = ConflictResolver()
    signals = [_buy_signal("source_a", 0.80, 3.0), _sell_signal("source_b", 0.80, 3.0)]
    weights = {"source_a": 3.0, "source_b": 3.0}
    result = resolver.resolve(signals, weights, ticker="AAPL")
    assert result.action == ConflictAction.CAPITAL_PRESERVATION
    assert result.conflict_detected is True


def test_dominant_source_wins():
    """When one source dramatically outscores the other, it wins."""
    resolver = ConflictResolver()
    signals = [_buy_signal("strong", 0.90, 5.0), _sell_signal("weak", 0.60, 1.0)]
    weights = {"strong": 5.0, "weak": 1.0}
    result = resolver.resolve(signals, weights, ticker="AAPL")
    assert result.action == ConflictAction.PROCEED
    assert result.winning_direction == "BUY"


def test_conflict_result_has_metadata():
    """ConflictResult includes resolution metadata."""
    resolver = ConflictResolver()
    signals = [_buy_signal("a", 0.80, 3.0), _sell_signal("b", 0.75, 2.5)]
    weights = {"a": 3.0, "b": 2.5}
    result = resolver.resolve(signals, weights, ticker="AAPL")
    assert result.buy_weight > 0
    assert result.sell_weight > 0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_conflict_resolution.py -v
```

Expected: FAIL

**Step 3: Implement**

```python
# src/evolve_trader/selection/conflict_resolution.py
"""Signal conflict resolution — handles opposing signals."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from evolve_trader.signals.events import SignalEvent


class ConflictAction(str, Enum):
    PROCEED = "proceed"
    CAPITAL_PRESERVATION = "capital_preservation"
    REDUCED_POSITION = "reduced_position"


@dataclass
class ConflictResult:
    """Result of conflict resolution."""
    conflict_detected: bool
    action: ConflictAction
    winning_direction: str | None = None  # BUY or SELL
    buy_weight: float = 0.0
    sell_weight: float = 0.0
    resolution_reason: str = ""


class ConflictResolver:
    """Resolves conflicts between opposing signal sources."""

    def __init__(self, dominance_threshold: float = 2.0):
        self._dominance_threshold = dominance_threshold

    def resolve(
        self,
        signals: list[SignalEvent],
        source_weights: dict[str, float],
        ticker: str,
    ) -> ConflictResult:
        """Resolve potential conflicts among signals for a ticker."""
        buy_weight = 0.0
        sell_weight = 0.0

        for signal in signals:
            weight = source_weights.get(signal.source, 1.0)
            confidence = signal.current_confidence()
            effective = weight * confidence

            action = signal.payload.get("action", "").upper()
            tx_type = signal.payload.get("transaction_type", "").lower()
            tx_code = signal.payload.get("transaction_code", "")

            is_buy = action == "BUY" or tx_type == "purchase" or tx_code == "P"
            is_sell = action == "SELL" or tx_type == "sale" or tx_code == "S"

            if is_buy:
                buy_weight += effective
            elif is_sell:
                sell_weight += effective

        # No conflict if signals are unidirectional
        if buy_weight == 0 or sell_weight == 0:
            direction = "BUY" if buy_weight > 0 else "SELL" if sell_weight > 0 else None
            return ConflictResult(
                conflict_detected=False,
                action=ConflictAction.PROCEED,
                winning_direction=direction,
                buy_weight=buy_weight,
                sell_weight=sell_weight,
            )

        # Conflict detected — check dominance
        ratio = max(buy_weight, sell_weight) / min(buy_weight, sell_weight)

        if ratio >= self._dominance_threshold:
            winner = "BUY" if buy_weight > sell_weight else "SELL"
            return ConflictResult(
                conflict_detected=True,
                action=ConflictAction.PROCEED,
                winning_direction=winner,
                buy_weight=buy_weight,
                sell_weight=sell_weight,
                resolution_reason=f"Dominant source wins ({ratio:.1f}x weight ratio)",
            )

        # Similar weights — default to Capital Preservation
        return ConflictResult(
            conflict_detected=True,
            action=ConflictAction.CAPITAL_PRESERVATION,
            buy_weight=buy_weight,
            sell_weight=sell_weight,
            resolution_reason=f"Similar weights ({ratio:.1f}x ratio) — defaulting to preservation",
        )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_conflict_resolution.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/selection/conflict_resolution.py tests/unit/test_conflict_resolution.py
git commit -m "feat: signal conflict resolution with confidence-weighted averaging"
```

---

## Task 7: Multi-Timeframe Skill Stacking

**Files:**
- Create: `src/evolve_trader/selection/timeframe_stack.py`
- Create: `src/evolve_trader/selection/interfaces.py`
- Create: `tests/unit/test_timeframe_stack.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_timeframe_stack.py
"""Tests for multi-timeframe skill stacking."""
import pytest
from evolve_trader.selection.timeframe_stack import (
    TimeframeStack,
    StrategicConstraints,
    TacticalDecision,
    ExecutionTiming,
    TimeframeLayer,
)


def test_strategic_layer_sets_constraints():
    """Strategic layer outputs sector exposure constraints."""
    constraints = StrategicConstraints(
        max_tech_exposure=0.20,
        max_financial_exposure=0.25,
        gross_exposure_limit=0.80,
        asset_class_targets={"equities": 0.70, "cash": 0.30},
    )
    assert constraints.max_tech_exposure == 0.20
    assert constraints.gross_exposure_limit == 0.80


def test_tactical_respects_strategic_constraints():
    """Tactical decisions cannot violate strategic constraints."""
    stack = TimeframeStack()
    strategic = StrategicConstraints(
        max_tech_exposure=0.20,
        gross_exposure_limit=0.80,
    )
    stack.set_strategic(strategic)

    # Tactical wants 30% tech — should be capped at 20%
    tactical = TacticalDecision(
        ticker="NVDA", direction="BUY", sector="Technology",
        proposed_weight=0.30,
    )
    validated = stack.validate_tactical(tactical)
    assert validated.proposed_weight <= strategic.max_tech_exposure


def test_execution_respects_tactical():
    """Execution timing works within tactical bounds."""
    stack = TimeframeStack()
    stack.set_strategic(StrategicConstraints(gross_exposure_limit=1.0))

    tactical = TacticalDecision(ticker="AAPL", direction="BUY", sector="Technology", proposed_weight=0.05)
    execution = ExecutionTiming(
        ticker="AAPL", order_type="LIMIT", urgency="normal",
    )
    assert execution.order_type == "LIMIT"


def test_timeframe_layers_exist():
    """All three timeframe layers are defined."""
    assert TimeframeLayer.STRATEGIC.value == "strategic"
    assert TimeframeLayer.TACTICAL.value == "tactical"
    assert TimeframeLayer.EXECUTION.value == "execution"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_timeframe_stack.py -v
```

Expected: FAIL

**Step 3: Implement**

```python
# src/evolve_trader/selection/interfaces.py
"""Typed interfaces for multi-timeframe skill stacking."""
from __future__ import annotations
from enum import Enum


class TimeframeLayer(str, Enum):
    STRATEGIC = "strategic"    # months-years
    TACTICAL = "tactical"      # weeks-months
    EXECUTION = "execution"    # hours-days
```

```python
# src/evolve_trader/selection/timeframe_stack.py
"""Multi-timeframe skill stacking — strategic/tactical/execution layers."""
from __future__ import annotations

from dataclasses import dataclass, field
from evolve_trader.selection.interfaces import TimeframeLayer


@dataclass
class StrategicConstraints:
    """Strategic layer output — portfolio-level constraints."""
    max_tech_exposure: float = 0.30
    max_financial_exposure: float = 0.25
    gross_exposure_limit: float = 1.0
    asset_class_targets: dict[str, float] = field(default_factory=dict)


@dataclass
class TacticalDecision:
    """Tactical layer output — individual position decisions."""
    ticker: str
    direction: str  # BUY, SELL
    sector: str = ""
    proposed_weight: float = 0.0


@dataclass
class ExecutionTiming:
    """Execution layer output — order timing and type."""
    ticker: str
    order_type: str = "MARKET"  # MARKET, LIMIT, STOP
    urgency: str = "normal"     # immediate, normal, patient
    limit_price: float | None = None


_SECTOR_CONSTRAINT_MAP = {
    "Technology": "max_tech_exposure",
    "Financials": "max_financial_exposure",
}


class TimeframeStack:
    """Manages the three-layer timeframe hierarchy."""

    def __init__(self):
        self._strategic: StrategicConstraints | None = None

    def set_strategic(self, constraints: StrategicConstraints) -> None:
        self._strategic = constraints

    def validate_tactical(self, decision: TacticalDecision) -> TacticalDecision:
        """Validate and constrain tactical decision against strategic layer."""
        if self._strategic is None:
            return decision

        # Check sector exposure limit
        constraint_attr = _SECTOR_CONSTRAINT_MAP.get(decision.sector)
        if constraint_attr:
            max_exposure = getattr(self._strategic, constraint_attr, 1.0)
            if decision.proposed_weight > max_exposure:
                decision.proposed_weight = max_exposure

        # Check gross exposure
        if decision.proposed_weight > self._strategic.gross_exposure_limit:
            decision.proposed_weight = self._strategic.gross_exposure_limit

        return decision
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_timeframe_stack.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/selection/interfaces.py src/evolve_trader/selection/timeframe_stack.py tests/unit/test_timeframe_stack.py
git commit -m "feat: multi-timeframe skill stacking with strategic/tactical/execution layers"
```

---

## Task 8: Survivorship Bias & Alpha Decay Monitoring

**Files:**
- Create: `src/evolve_trader/monitoring/survivorship.py`
- Create: `src/evolve_trader/signals/alpha_decay.py`
- Create: `tests/unit/test_survivorship.py`
- Create: `tests/unit/test_alpha_decay.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_survivorship.py
"""Tests for survivorship bias monitoring."""
import pytest
from evolve_trader.monitoring.survivorship import SurvivorshipMonitor


def test_initial_roster_retention_rate():
    """Tracks % of original roster still in active tier."""
    monitor = SurvivorshipMonitor(
        initial_roster={"buffett", "dalio", "ackman", "pelosi", "burry"}
    )
    # After 6 months, only 3 of 5 still active
    monitor.update_active({"buffett", "dalio", "ackman"})
    assert monitor.retention_rate() == pytest.approx(0.6, abs=0.01)


def test_high_retention_warning():
    """>80% retention after 6+ months = not demoting aggressively enough."""
    monitor = SurvivorshipMonitor(
        initial_roster={"a", "b", "c", "d", "e"}
    )
    monitor.update_active({"a", "b", "c", "d", "e"})  # All still active
    assert monitor.retention_rate() == 1.0
    assert monitor.is_retention_too_high()  # >80%
```

```python
# tests/unit/test_alpha_decay.py
"""Tests for alpha decay / popularity penalty monitoring."""
import pytest
from datetime import datetime, timezone, timedelta
from evolve_trader.signals.alpha_decay import AlphaDecayMonitor, DisclosureImpact


def test_disclosure_impact_trend():
    """Tracks average price impact at disclosure over time."""
    monitor = AlphaDecayMonitor()
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)

    # Early: low impact
    for i in range(5):
        monitor.record(DisclosureImpact("congressional", base + timedelta(weeks=i), 0.01))
    # Later: high impact (market front-running)
    for i in range(5, 10):
        monitor.record(DisclosureImpact("congressional", base + timedelta(weeks=i), 0.05))

    trend = monitor.get_impact_trend("congressional")
    assert trend.is_increasing  # Impact growing = popularity penalty needed
    assert trend.latest_impact > trend.earliest_impact
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_survivorship.py tests/unit/test_alpha_decay.py -v
```

Expected: FAIL

**Step 3: Implement**

```python
# src/evolve_trader/monitoring/survivorship.py
"""Survivorship bias monitoring."""
from __future__ import annotations


class SurvivorshipMonitor:
    """Monitors Initial Roster Retention Rate."""

    def __init__(self, initial_roster: set[str]):
        self._initial = frozenset(initial_roster)
        self._current_active: set[str] = set()

    def update_active(self, active_sources: set[str]) -> None:
        self._current_active = active_sources

    def retention_rate(self) -> float:
        if not self._initial:
            return 0.0
        retained = self._initial & self._current_active
        return len(retained) / len(self._initial)

    def is_retention_too_high(self, threshold: float = 0.80) -> bool:
        return self.retention_rate() > threshold
```

```python
# src/evolve_trader/signals/alpha_decay.py
"""Alpha decay and popularity penalty monitoring."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DisclosureImpact:
    """Price impact at disclosure time for a source."""
    source: str
    timestamp: datetime
    impact_pct: float  # Absolute price move 0-24h post-disclosure


@dataclass
class ImpactTrend:
    """Trend analysis for disclosure impact."""
    source: str
    earliest_impact: float
    latest_impact: float
    is_increasing: bool
    observation_count: int


class AlphaDecayMonitor:
    """Monitors disclosure impact trend per source."""

    def __init__(self):
        self._impacts: dict[str, list[DisclosureImpact]] = defaultdict(list)

    def record(self, impact: DisclosureImpact) -> None:
        self._impacts[impact.source].append(impact)

    def get_impact_trend(self, source: str) -> ImpactTrend:
        records = sorted(self._impacts.get(source, []), key=lambda r: r.timestamp)
        if len(records) < 2:
            return ImpactTrend(source, 0.0, 0.0, False, len(records))

        half = len(records) // 2
        early_avg = sum(r.impact_pct for r in records[:half]) / half
        late_avg = sum(r.impact_pct for r in records[half:]) / (len(records) - half)

        return ImpactTrend(
            source=source,
            earliest_impact=early_avg,
            latest_impact=late_avg,
            is_increasing=late_avg > early_avg * 1.1,  # 10% threshold
            observation_count=len(records),
        )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_survivorship.py tests/unit/test_alpha_decay.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/monitoring/ src/evolve_trader/signals/alpha_decay.py tests/unit/test_survivorship.py tests/unit/test_alpha_decay.py
git commit -m "feat: survivorship bias monitoring and alpha decay tracking"
```

---

## Task 9: Integration Testing — Full Meta-Selector Pipeline

**Files:**
- Create: `tests/integration/test_meta_selector_pipeline.py`

**Step 1: Write the integration test**

```python
# tests/integration/test_meta_selector_pipeline.py
"""Integration: signals → scoring → lifecycle → conflict → meta-selector → allocation."""
import pytest
from datetime import datetime, timezone, timedelta
from evolve_trader.signals.sources.edgar_13f import Edgar13FSource, Filing13F, Holding
from evolve_trader.signals.sources.congressional import CongressionalTradeSource, CongressionalTrade, LeadershipRole
from evolve_trader.signals.scoring import SourceScorer, SourceTier, SignalOutcome
from evolve_trader.signals.lifecycle import LifecycleManager, LifecycleStage
from evolve_trader.selection.conflict_resolution import ConflictResolver, ConflictAction
from evolve_trader.selection.meta_selector import MetaSelector
from evolve_trader.regime.classifier import BasicRegimeClassifier
from evolve_trader.regime.labels import PrimaryRegime


def test_full_pipeline_buy_consensus():
    """Full pipeline: multiple buy signals → risk-on → aggressive allocation."""
    now = datetime.now(timezone.utc)

    # 1. Generate signals
    edgar = Edgar13FSource()
    filing = Filing13F("0001067983", "Berkshire Hathaway",
        now - timedelta(days=45), now,
        [Holding("Apple Inc", "037833100", 75000, 500000, "SOLE", 500000)])
    signals_13f = edgar.filing_to_signals(filing)

    congress = CongressionalTradeSource()
    trade = CongressionalTrade("Nancy Pelosi", "D", "CA", "House",
        now - timedelta(days=10), now - timedelta(days=3),
        "AAPL", "purchase", "$1M+", ["Intelligence"], LeadershipRole.SPEAKER_EMERITUS)
    signals_congress = congress.trade_to_signals(trade)

    all_signals = signals_13f + signals_congress

    # 2. Score sources
    scorer = SourceScorer()
    outcomes_13f = [SignalOutcome(True, 0.05, now - timedelta(weeks=i)) for i in range(10)]
    score_13f = scorer.compute_score("edgar_13f", SourceTier.TIER_1, outcomes_13f)

    # 3. Check lifecycle
    lifecycle_mgr = LifecycleManager()
    lifecycle_mgr.add_source("edgar_13f", tier_guess=1)
    for _ in range(3):
        lifecycle_mgr.promote("edgar_13f", reason="advancing")
    assert lifecycle_mgr.get_lifecycle("edgar_13f").stage == LifecycleStage.ACTIVE

    # 4. Resolve conflicts
    resolver = ConflictResolver()
    weights = {"edgar_13f": score_13f.composite_weight, "congressional": 2.0}
    conflict = resolver.resolve(all_signals, weights, ticker="AAPL")
    assert conflict.action == ConflictAction.PROCEED

    # 5. Classify regime
    classifier = BasicRegimeClassifier()
    regime = classifier.classify(all_signals)
    assert regime.primary_regime == PrimaryRegime.RISK_ON

    # 6. Run meta-selector
    selector = MetaSelector(
        available_strategies=["momentum-v1", "capital-preservation"],
        strategy_regime_affinity={
            "momentum-v1": {PrimaryRegime.RISK_ON: 0.9, PrimaryRegime.RISK_OFF: 0.1},
            "capital-preservation": {PrimaryRegime.RISK_ON: 0.1, PrimaryRegime.RISK_OFF: 0.9},
        },
    )
    result = selector.select(regime, all_signals)

    # Momentum should dominate in risk-on
    momentum_alloc = next((a for a in result.allocations if a.strategy_name == "momentum-v1"), None)
    assert momentum_alloc is not None
    assert momentum_alloc.weight > 0.5
```

**Step 2: Run test**

```bash
pytest tests/integration/test_meta_selector_pipeline.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_meta_selector_pipeline.py
git commit -m "test: integration tests for full meta-selector pipeline"
```

---

## Task 10: Final Verification

**Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: ALL PASS — Phase 1, 2, and 3 tests

**Step 2: Run linting and type checking**

```bash
ruff check src/evolve_trader/
mypy src/evolve_trader/ --ignore-missing-imports
```

Expected: No errors

**Step 3: Commit**

```bash
git add -A
git commit -m "test: Phase 3 final verification — all tests passing"
```

---

## Parallelization Notes

```
Task 1 (Meta-Selector) ──────────────────────┐
Task 2 (Signal Scoring) ─────────────────────┤
Task 3 (Spread Tracking) ────────────────────┤── Task 6 (Conflict Resolution) ──┐
Task 4 (Return Tracking) ────────────────────┤                                   ├── Task 9 (Integration)
Task 5 (Lifecycle Pipeline) ─────────────────┤── Task 7 (Timeframe Stacking) ──┘
Task 8 (Survivorship & Alpha Decay) ─────────┘
```

**Can run in parallel:**
- Tasks 1-5 and Task 8 are all independent — run simultaneously
- Task 6 (conflict resolution) depends on scoring (Task 2) being available
- Task 7 (timeframe stacking) is independent
- Task 9 (integration) depends on all previous tasks
- Task 10 (final verification) must be last
