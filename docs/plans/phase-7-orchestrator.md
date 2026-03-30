# Phase 7: Meta-Evolution Orchestrator — Detailed Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the meta-evolution orchestrator — an LLM-powered agent running on a weekly/bi-weekly cadence that detects cross-layer interactions, runs counterfactual analysis, and tunes how the system evolves. THIS IS THE CORE IP. The orchestrator sits above all other layers and adjusts evolution pace, detection thresholds, discovery aggressiveness, and inter-layer tension — all subject to immutable risk constraints.

**Architecture:** The orchestrator operates as a periodic agent that ingests a structured metrics report covering every system layer (returns, drawdowns, evolution events, regime classifications, signal performance, constraint proximity, trade frequency, conflict resolutions). It uses LLM reasoning (via LiteLLM) to identify tensions, propose adjustments, validate them via counterfactual replay, and log all decisions with structured rationales, cited metrics, and compact evidence. All adjustments pass through immutable risk constraints before application. The orchestrator may tune discovery parameters introduced in Phase 9, but Phase 9 source integrations do not depend on the orchestrator existing first.

**Tech Stack:** Python 3.12+, LiteLLM, PostgreSQL (via existing DAL), numpy, pytest

**Parent plan:** [`2026-03-29-evolve-trader-master-plan.md`](2026-03-29-evolve-trader-master-plan.md)

**Prerequisites:** Phase 6 complete. Portfolio construction, position sizing, multi-strategy coordination, risk overlay, and allocation optimization all verified. All prior evolution layers (strategy evolution, signal framework, meta-selector, monitoring, discovery) operational.

---

## Task 1: Orchestrator Agent Core

**Files:**
- Create: `src/evolve_trader/orchestrator/__init__.py`
- Create: `src/evolve_trader/orchestrator/agent.py`
- Create: `src/evolve_trader/orchestrator/config.py`
- Create: `tests/unit/test_orchestrator_agent.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_orchestrator_agent.py
"""Tests for the meta-evolution orchestrator agent."""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from evolve_trader.orchestrator.agent import OrchestratorAgent, OrchestratorResult
from evolve_trader.orchestrator.config import OrchestratorConfig


def test_orchestrator_config_defaults():
    """OrchestratorConfig has sensible defaults for cadence and constraints."""
    config = OrchestratorConfig()
    assert config.cadence_days == 7
    assert config.max_adjustments_per_cycle <= 5
    assert config.llm_model is not None
    assert config.immutable_constraints_enforced is True
    assert config.dry_run is False


def test_orchestrator_config_custom_cadence():
    """OrchestratorConfig accepts custom cadence."""
    config = OrchestratorConfig(cadence_days=14, max_adjustments_per_cycle=3)
    assert config.cadence_days == 14
    assert config.max_adjustments_per_cycle == 3


def test_orchestrator_config_rejects_zero_cadence():
    """OrchestratorConfig rejects invalid cadence values."""
    with pytest.raises(ValueError, match="cadence_days"):
        OrchestratorConfig(cadence_days=0)


def test_orchestrator_config_immutable_constraints_cannot_be_disabled():
    """Immutable risk constraints are always enforced — no override."""
    config = OrchestratorConfig(immutable_constraints_enforced=False)
    assert config.immutable_constraints_enforced is True


def test_orchestrator_agent_creation():
    """OrchestratorAgent initializes with config and dependencies."""
    config = OrchestratorConfig()
    agent = OrchestratorAgent(config=config)
    assert agent.config.cadence_days == 7
    assert agent.last_run is None
    assert agent.cycle_count == 0


def test_orchestrator_agent_is_due_for_run():
    """Agent correctly determines if a run is due based on cadence."""
    config = OrchestratorConfig(cadence_days=7)
    agent = OrchestratorAgent(config=config)

    # Never run before — always due
    assert agent.is_due(as_of=datetime.now(timezone.utc)) is True

    # Just ran — not due
    agent.last_run = datetime.now(timezone.utc)
    assert agent.is_due(as_of=datetime.now(timezone.utc)) is False

    # Ran 8 days ago — due
    agent.last_run = datetime.now(timezone.utc) - timedelta(days=8)
    assert agent.is_due(as_of=datetime.now(timezone.utc)) is True


@pytest.mark.asyncio
async def test_orchestrator_agent_run_returns_result():
    """Agent run produces an OrchestratorResult with adjustments and reasoning."""
    config = OrchestratorConfig(dry_run=True)
    agent = OrchestratorAgent(config=config)

    # Mock the metrics aggregator and LLM
    mock_metrics = MagicMock()
    mock_metrics.aggregate.return_value = {
        "period_start": datetime(2026, 3, 1, tzinfo=timezone.utc),
        "period_end": datetime(2026, 3, 15, tzinfo=timezone.utc),
        "portfolio_return": 0.03,
        "max_drawdown": 0.05,
        "evolution_events": 4,
        "regime_label": "risk-on",
        "constraint_proximity": {"max_drawdown": 0.4},
    }

    with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "reasoning": "Portfolio performing well. Minor oscillation in momentum layer.",
            "proposed_adjustments": [
                {
                    "layer": "evolution_pace",
                    "target": "momentum",
                    "action": "slow",
                    "magnitude": 0.2,
                    "rationale": "FIX->un-FIX->re-FIX cycle detected",
                }
            ],
        }
        result = await agent.run(metrics=mock_metrics)

    assert isinstance(result, OrchestratorResult)
    assert result.cycle_number == 1
    assert len(result.proposed_adjustments) == 1
    assert result.reasoning is not None
    assert result.applied is False  # dry_run=True


@pytest.mark.asyncio
async def test_orchestrator_agent_enforces_max_adjustments():
    """Agent caps adjustments at max_adjustments_per_cycle."""
    config = OrchestratorConfig(max_adjustments_per_cycle=2, dry_run=True)
    agent = OrchestratorAgent(config=config)

    mock_metrics = MagicMock()
    mock_metrics.aggregate.return_value = {
        "period_start": datetime(2026, 3, 1, tzinfo=timezone.utc),
        "period_end": datetime(2026, 3, 15, tzinfo=timezone.utc),
        "portfolio_return": -0.02,
        "max_drawdown": 0.12,
        "evolution_events": 12,
        "regime_label": "risk-off",
        "constraint_proximity": {"max_drawdown": 0.8},
    }

    with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "reasoning": "System under stress.",
            "proposed_adjustments": [
                {"layer": "pace", "target": "a", "action": "slow", "magnitude": 0.1, "rationale": "r1"},
                {"layer": "threshold", "target": "b", "action": "tighten", "magnitude": 0.2, "rationale": "r2"},
                {"layer": "discovery", "target": "c", "action": "accelerate", "magnitude": 0.3, "rationale": "r3"},
                {"layer": "pace", "target": "d", "action": "slow", "magnitude": 0.1, "rationale": "r4"},
            ],
        }
        result = await agent.run(metrics=mock_metrics)

    assert len(result.proposed_adjustments) <= 2


@pytest.mark.asyncio
async def test_orchestrator_agent_increments_cycle_count():
    """Agent tracks cycle count across runs."""
    config = OrchestratorConfig(dry_run=True)
    agent = OrchestratorAgent(config=config)

    mock_metrics = MagicMock()
    mock_metrics.aggregate.return_value = {
        "period_start": datetime(2026, 3, 1, tzinfo=timezone.utc),
        "period_end": datetime(2026, 3, 15, tzinfo=timezone.utc),
        "portfolio_return": 0.01,
        "max_drawdown": 0.03,
        "evolution_events": 2,
        "regime_label": "risk-on",
        "constraint_proximity": {"max_drawdown": 0.2},
    }

    with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {"reasoning": "Stable.", "proposed_adjustments": []}
        await agent.run(metrics=mock_metrics)
        await agent.run(metrics=mock_metrics)

    assert agent.cycle_count == 2
    assert agent.last_run is not None
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_orchestrator_agent.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.orchestrator'`

**Step 3: Implement the orchestrator agent core**

```python
# src/evolve_trader/orchestrator/__init__.py
"""Meta-evolution orchestrator — the core IP layer."""
```

```python
# src/evolve_trader/orchestrator/config.py
"""Configuration for the meta-evolution orchestrator."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OrchestratorConfig:
    """Orchestrator configuration with immutable safety guarantees.

    The orchestrator runs on a configurable cadence (default weekly),
    proposes up to max_adjustments_per_cycle changes, and always
    enforces immutable risk constraints regardless of input.
    """

    cadence_days: int = 7
    max_adjustments_per_cycle: int = 5
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.3
    immutable_constraints_enforced: bool = True
    dry_run: bool = False
    counterfactual_enabled: bool = True
    lookback_periods: int = 3

    def __post_init__(self):
        if self.cadence_days < 1:
            raise ValueError("cadence_days must be >= 1")
        # Immutable constraints can NEVER be disabled
        self.immutable_constraints_enforced = True
```

```python
# src/evolve_trader/orchestrator/agent.py
"""Meta-evolution orchestrator agent — LLM-powered system tuner."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

import litellm

from evolve_trader.orchestrator.config import OrchestratorConfig


@dataclass
class ProposedAdjustment:
    """A single proposed adjustment to a system layer."""

    layer: str
    target: str
    action: str
    magnitude: float
    rationale: str
    counterfactual_result: dict[str, Any] | None = None
    applied: bool = False


@dataclass
class OrchestratorResult:
    """Result of a single orchestrator cycle."""

    cycle_number: int
    period_start: datetime
    period_end: datetime
    reasoning: str
    proposed_adjustments: list[ProposedAdjustment]
    metrics_snapshot: dict[str, Any]
    applied: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class OrchestratorAgent:
    """Meta-evolution orchestrator agent.

    Runs on a configurable cadence, ingests system-wide metrics,
    uses LLM reasoning to propose adjustments, validates via
    counterfactual replay, and applies changes subject to immutable
    risk constraints.
    """

    def __init__(self, config: OrchestratorConfig | None = None):
        self.config = config or OrchestratorConfig()
        self.last_run: datetime | None = None
        self.cycle_count: int = 0

    def is_due(self, as_of: datetime) -> bool:
        """Check if an orchestrator run is due."""
        if self.last_run is None:
            return True
        return (as_of - self.last_run) >= timedelta(days=self.config.cadence_days)

    async def run(self, metrics: Any) -> OrchestratorResult:
        """Execute a single orchestrator cycle.

        1. Aggregate metrics across all layers
        2. Call LLM for analysis and proposed adjustments
        3. Cap adjustments at max_adjustments_per_cycle
        4. Apply (or defer if dry_run)
        5. Log everything
        """
        report = metrics.aggregate()

        llm_response = await self._call_llm(report)

        raw_adjustments = llm_response.get("proposed_adjustments", [])
        capped = raw_adjustments[: self.config.max_adjustments_per_cycle]

        adjustments = [
            ProposedAdjustment(
                layer=adj["layer"],
                target=adj["target"],
                action=adj["action"],
                magnitude=adj["magnitude"],
                rationale=adj["rationale"],
            )
            for adj in capped
        ]

        self.cycle_count += 1
        self.last_run = datetime.now(timezone.utc)

        applied = not self.config.dry_run and len(adjustments) > 0

        return OrchestratorResult(
            cycle_number=self.cycle_count,
            period_start=report["period_start"],
            period_end=report["period_end"],
            reasoning=llm_response["reasoning"],
            proposed_adjustments=adjustments,
            metrics_snapshot=report,
            applied=applied,
        )

    async def _call_llm(self, metrics_report: dict[str, Any]) -> dict[str, Any]:
        """Call LLM with structured metrics for analysis.

        Returns dict with 'reasoning' and 'proposed_adjustments' keys.
        """
        prompt = self._build_prompt(metrics_report)
        response = await litellm.acompletion(
            model=self.config.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the meta-evolution orchestrator for a trading system. "
                        "Analyze system metrics and propose targeted adjustments. "
                        "All proposals must respect immutable risk constraints. "
                        "Be conservative — fewer high-confidence changes beat many uncertain ones."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.llm_temperature,
            response_format={"type": "json_object"},
        )
        import json

        return json.loads(response.choices[0].message.content)

    def _build_prompt(self, report: dict[str, Any]) -> str:
        """Build structured prompt from metrics report."""
        import json

        return (
            "Analyze the following system metrics report and propose adjustments.\n\n"
            f"Metrics Report:\n{json.dumps(report, indent=2, default=str)}\n\n"
            "Respond with JSON: {\"reasoning\": \"...\", \"proposed_adjustments\": ["
            "{\"layer\": \"...\", \"target\": \"...\", \"action\": \"...\", "
            "\"magnitude\": 0.0, \"rationale\": \"...\"}]}"
        )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_orchestrator_agent.py -v
```

Expected: PASS — all orchestrator agent core tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/orchestrator/__init__.py src/evolve_trader/orchestrator/agent.py src/evolve_trader/orchestrator/config.py tests/unit/test_orchestrator_agent.py
git commit -m "feat: orchestrator agent core with LLM-powered cycle, cadence, and immutable constraints"
```

---

## Task 2: Metrics Aggregator

**Files:**
- Create: `src/evolve_trader/orchestrator/metrics_aggregator.py`
- Create: `tests/unit/test_metrics_aggregator.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_metrics_aggregator.py
"""Tests for the orchestrator metrics aggregator."""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
from evolve_trader.orchestrator.metrics_aggregator import (
    MetricsAggregator,
    AggregatedReport,
    LayerMetrics,
)


def test_aggregated_report_has_required_fields():
    """AggregatedReport contains all cross-layer metric categories."""
    report = AggregatedReport(
        period_start=datetime(2026, 3, 1, tzinfo=timezone.utc),
        period_end=datetime(2026, 3, 15, tzinfo=timezone.utc),
        portfolio_return=0.03,
        max_drawdown=0.05,
        sharpe_ratio=1.2,
        evolution_events=[],
        regime_classifications=[],
        signal_performance={},
        constraint_proximity={},
        trade_frequency=12,
        conflict_resolutions=[],
        layer_metrics={},
    )
    assert report.portfolio_return == 0.03
    assert report.trade_frequency == 12


def test_layer_metrics_structure():
    """LayerMetrics captures per-layer health data."""
    lm = LayerMetrics(
        layer_name="strategy_evolution",
        event_count=5,
        fix_count=2,
        derive_count=2,
        capture_count=1,
        performance_delta=0.02,
        oscillation_score=0.1,
        convergence_score=0.7,
    )
    assert lm.layer_name == "strategy_evolution"
    assert lm.fix_count == 2
    assert lm.oscillation_score == 0.1


def test_metrics_aggregator_creation():
    """MetricsAggregator initializes with data source references."""
    aggregator = MetricsAggregator()
    assert aggregator is not None


def test_metrics_aggregator_aggregate_returns_report():
    """MetricsAggregator.aggregate() produces a structured AggregatedReport."""
    aggregator = MetricsAggregator()

    # Inject mock data sources
    aggregator.portfolio_source = MagicMock()
    aggregator.portfolio_source.get_returns.return_value = [0.01, 0.02, -0.005, 0.015]
    aggregator.portfolio_source.get_max_drawdown.return_value = 0.05

    aggregator.evolution_source = MagicMock()
    aggregator.evolution_source.get_events.return_value = [
        {"type": "FIX", "layer": "momentum", "timestamp": "2026-03-05"},
        {"type": "DERIVED", "layer": "mean_reversion", "timestamp": "2026-03-10"},
    ]

    aggregator.regime_source = MagicMock()
    aggregator.regime_source.get_classifications.return_value = [
        {"label": "risk-on", "confidence": 0.8, "timestamp": "2026-03-01"},
    ]

    aggregator.signal_source = MagicMock()
    aggregator.signal_source.get_performance.return_value = {
        "edgar_13f": {"accuracy": 0.65, "signal_count": 12},
        "form4": {"accuracy": 0.58, "signal_count": 8},
    }

    aggregator.constraint_source = MagicMock()
    aggregator.constraint_source.get_proximity.return_value = {
        "max_drawdown": 0.4,
        "position_concentration": 0.3,
    }

    aggregator.trade_source = MagicMock()
    aggregator.trade_source.get_frequency.return_value = 15

    aggregator.conflict_source = MagicMock()
    aggregator.conflict_source.get_resolutions.return_value = []

    period_start = datetime(2026, 3, 1, tzinfo=timezone.utc)
    period_end = datetime(2026, 3, 15, tzinfo=timezone.utc)

    report = aggregator.aggregate(period_start=period_start, period_end=period_end)

    assert isinstance(report, AggregatedReport)
    assert report.portfolio_return == pytest.approx(0.04, abs=0.001)
    assert report.max_drawdown == 0.05
    assert len(report.evolution_events) == 2
    assert report.trade_frequency == 15
    assert "max_drawdown" in report.constraint_proximity


def test_metrics_aggregator_computes_sharpe():
    """MetricsAggregator computes annualized Sharpe from period returns."""
    aggregator = MetricsAggregator()

    aggregator.portfolio_source = MagicMock()
    aggregator.portfolio_source.get_returns.return_value = [0.01] * 14  # 2 weeks of daily returns
    aggregator.portfolio_source.get_max_drawdown.return_value = 0.02

    aggregator.evolution_source = MagicMock()
    aggregator.evolution_source.get_events.return_value = []
    aggregator.regime_source = MagicMock()
    aggregator.regime_source.get_classifications.return_value = []
    aggregator.signal_source = MagicMock()
    aggregator.signal_source.get_performance.return_value = {}
    aggregator.constraint_source = MagicMock()
    aggregator.constraint_source.get_proximity.return_value = {}
    aggregator.trade_source = MagicMock()
    aggregator.trade_source.get_frequency.return_value = 0
    aggregator.conflict_source = MagicMock()
    aggregator.conflict_source.get_resolutions.return_value = []

    period_start = datetime(2026, 3, 1, tzinfo=timezone.utc)
    period_end = datetime(2026, 3, 15, tzinfo=timezone.utc)

    report = aggregator.aggregate(period_start=period_start, period_end=period_end)

    # Constant positive returns → very high Sharpe (std ≈ 0 but we floor it)
    assert report.sharpe_ratio > 0


def test_metrics_aggregator_to_dict():
    """AggregatedReport serializes to dict for LLM prompt construction."""
    report = AggregatedReport(
        period_start=datetime(2026, 3, 1, tzinfo=timezone.utc),
        period_end=datetime(2026, 3, 15, tzinfo=timezone.utc),
        portfolio_return=0.03,
        max_drawdown=0.05,
        sharpe_ratio=1.2,
        evolution_events=[],
        regime_classifications=[],
        signal_performance={},
        constraint_proximity={"max_drawdown": 0.4},
        trade_frequency=12,
        conflict_resolutions=[],
        layer_metrics={},
    )
    d = report.to_dict()
    assert isinstance(d, dict)
    assert d["portfolio_return"] == 0.03
    assert "period_start" in d
    assert "constraint_proximity" in d
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_metrics_aggregator.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.orchestrator.metrics_aggregator'`

**Step 3: Implement the metrics aggregator**

```python
# src/evolve_trader/orchestrator/metrics_aggregator.py
"""Metrics aggregator for the orchestrator — collects cross-layer data."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np


@dataclass
class LayerMetrics:
    """Per-layer health metrics."""

    layer_name: str
    event_count: int = 0
    fix_count: int = 0
    derive_count: int = 0
    capture_count: int = 0
    performance_delta: float = 0.0
    oscillation_score: float = 0.0
    convergence_score: float = 0.0


@dataclass
class AggregatedReport:
    """Structured report aggregating metrics across all system layers.

    This is the primary input to the orchestrator's LLM reasoning step.
    """

    period_start: datetime
    period_end: datetime
    portfolio_return: float
    max_drawdown: float
    sharpe_ratio: float
    evolution_events: list[dict[str, Any]]
    regime_classifications: list[dict[str, Any]]
    signal_performance: dict[str, Any]
    constraint_proximity: dict[str, float]
    trade_frequency: int
    conflict_resolutions: list[dict[str, Any]]
    layer_metrics: dict[str, LayerMetrics]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for LLM prompt construction."""
        d = asdict(self)
        d["period_start"] = self.period_start.isoformat()
        d["period_end"] = self.period_end.isoformat()
        return d


class MetricsAggregator:
    """Collects returns, drawdowns, evolution events, regime classifications,
    signal performance, constraint proximity, trade frequency, and conflict
    resolutions into a structured AggregatedReport.

    Each data source is injected as an attribute with a known interface.
    """

    def __init__(self):
        self.portfolio_source: Any = None
        self.evolution_source: Any = None
        self.regime_source: Any = None
        self.signal_source: Any = None
        self.constraint_source: Any = None
        self.trade_source: Any = None
        self.conflict_source: Any = None

    def aggregate(
        self,
        period_start: datetime,
        period_end: datetime,
    ) -> AggregatedReport:
        """Aggregate metrics from all sources for the given period."""
        returns = self.portfolio_source.get_returns()
        max_dd = self.portfolio_source.get_max_drawdown()
        total_return = sum(returns)
        sharpe = self._compute_sharpe(returns)

        evolution_events = self.evolution_source.get_events()
        regime_classifications = self.regime_source.get_classifications()
        signal_performance = self.signal_source.get_performance()
        constraint_proximity = self.constraint_source.get_proximity()
        trade_frequency = self.trade_source.get_frequency()
        conflict_resolutions = self.conflict_source.get_resolutions()

        layer_metrics = self._compute_layer_metrics(evolution_events)

        return AggregatedReport(
            period_start=period_start,
            period_end=period_end,
            portfolio_return=total_return,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            evolution_events=evolution_events,
            regime_classifications=regime_classifications,
            signal_performance=signal_performance,
            constraint_proximity=constraint_proximity,
            trade_frequency=trade_frequency,
            conflict_resolutions=conflict_resolutions,
            layer_metrics=layer_metrics,
        )

    def _compute_sharpe(
        self, returns: list[float], annualization_factor: float = 252.0
    ) -> float:
        """Compute annualized Sharpe ratio from daily returns."""
        if len(returns) < 2:
            return 0.0
        arr = np.array(returns)
        mean_ret = np.mean(arr)
        std_ret = np.std(arr, ddof=1)
        if std_ret < 1e-10:
            # Near-zero vol — return sign-adjusted large value
            return float(np.sign(mean_ret) * 10.0)
        return float((mean_ret / std_ret) * np.sqrt(annualization_factor))

    def _compute_layer_metrics(
        self, events: list[dict[str, Any]]
    ) -> dict[str, LayerMetrics]:
        """Compute per-layer metrics from evolution events."""
        layers: dict[str, LayerMetrics] = {}
        for event in events:
            layer = event.get("layer", "unknown")
            if layer not in layers:
                layers[layer] = LayerMetrics(layer_name=layer)
            lm = layers[layer]
            lm.event_count += 1
            event_type = event.get("type", "").upper()
            if event_type == "FIX":
                lm.fix_count += 1
            elif event_type == "DERIVED":
                lm.derive_count += 1
            elif event_type == "CAPTURED":
                lm.capture_count += 1
        return layers
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_metrics_aggregator.py -v
```

Expected: PASS — all metrics aggregator tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/orchestrator/metrics_aggregator.py tests/unit/test_metrics_aggregator.py
git commit -m "feat: metrics aggregator collecting cross-layer data for orchestrator"
```

---

## Task 3: Evolution Pace Control

**Files:**
- Create: `src/evolve_trader/orchestrator/pace_control.py`
- Create: `tests/unit/test_pace_control.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_pace_control.py
"""Tests for evolution pace control — detecting cycles, oscillations, convergence."""
import pytest
from datetime import datetime, timezone, timedelta
from evolve_trader.orchestrator.pace_control import (
    PaceController,
    PaceAssessment,
    EvolutionPattern,
)


def test_pace_assessment_structure():
    """PaceAssessment captures layer-level pace analysis."""
    assessment = PaceAssessment(
        layer="momentum",
        pattern=EvolutionPattern.OSCILLATING,
        event_count=8,
        cycle_detected=True,
        cycle_description="FIX->un-FIX->re-FIX on stop-loss parameter",
        recommended_action="slow",
        pace_multiplier=0.5,
    )
    assert assessment.pattern == EvolutionPattern.OSCILLATING
    assert assessment.pace_multiplier == 0.5


def test_evolution_pattern_enum():
    """EvolutionPattern covers key states."""
    assert EvolutionPattern.CONVERGING.value == "converging"
    assert EvolutionPattern.OSCILLATING.value == "oscillating"
    assert EvolutionPattern.STAGNANT.value == "stagnant"
    assert EvolutionPattern.HEALTHY.value == "healthy"
    assert EvolutionPattern.DIVERGING.value == "diverging"


def test_pace_controller_detect_oscillation():
    """PaceController detects FIX->un-FIX->re-FIX oscillation patterns."""
    controller = PaceController()

    # Simulate oscillating evolution log: FIX, revert, FIX, revert, FIX
    events = [
        {"type": "FIX", "layer": "momentum", "param": "stop_loss", "value": 0.02,
         "timestamp": datetime(2026, 3, 1, tzinfo=timezone.utc)},
        {"type": "FIX", "layer": "momentum", "param": "stop_loss", "value": 0.05,
         "timestamp": datetime(2026, 3, 3, tzinfo=timezone.utc)},
        {"type": "FIX", "layer": "momentum", "param": "stop_loss", "value": 0.02,
         "timestamp": datetime(2026, 3, 5, tzinfo=timezone.utc)},
        {"type": "FIX", "layer": "momentum", "param": "stop_loss", "value": 0.05,
         "timestamp": datetime(2026, 3, 7, tzinfo=timezone.utc)},
    ]

    assessment = controller.assess_layer("momentum", events)
    assert assessment.pattern == EvolutionPattern.OSCILLATING
    assert assessment.cycle_detected is True
    assert assessment.pace_multiplier < 1.0  # Recommendation: slow down


def test_pace_controller_detect_convergence():
    """PaceController detects convergence — changes getting smaller."""
    controller = PaceController()

    events = [
        {"type": "FIX", "layer": "mean_reversion", "param": "lookback", "value": 20,
         "timestamp": datetime(2026, 3, 1, tzinfo=timezone.utc)},
        {"type": "FIX", "layer": "mean_reversion", "param": "lookback", "value": 22,
         "timestamp": datetime(2026, 3, 5, tzinfo=timezone.utc)},
        {"type": "FIX", "layer": "mean_reversion", "param": "lookback", "value": 21,
         "timestamp": datetime(2026, 3, 10, tzinfo=timezone.utc)},
    ]

    assessment = controller.assess_layer("mean_reversion", events)
    assert assessment.pattern == EvolutionPattern.CONVERGING
    assert assessment.pace_multiplier >= 0.8  # Nearly done — maintain or slightly slow


def test_pace_controller_detect_stagnation():
    """PaceController detects stagnation — no evolution events in lookback."""
    controller = PaceController()

    events = []  # No events at all

    assessment = controller.assess_layer("stat_arb", events)
    assert assessment.pattern == EvolutionPattern.STAGNANT
    assert assessment.pace_multiplier > 1.0  # Recommendation: speed up


def test_pace_controller_healthy_evolution():
    """PaceController identifies healthy evolution — steady FIX/DERIVE without cycling."""
    controller = PaceController()

    events = [
        {"type": "FIX", "layer": "momentum", "param": "entry_threshold", "value": 0.5,
         "timestamp": datetime(2026, 3, 1, tzinfo=timezone.utc)},
        {"type": "DERIVED", "layer": "momentum", "param": None, "value": None,
         "timestamp": datetime(2026, 3, 8, tzinfo=timezone.utc)},
        {"type": "FIX", "layer": "momentum", "param": "exit_threshold", "value": 0.3,
         "timestamp": datetime(2026, 3, 15, tzinfo=timezone.utc)},
    ]

    assessment = controller.assess_layer("momentum", events)
    assert assessment.pattern == EvolutionPattern.HEALTHY
    assert assessment.pace_multiplier == pytest.approx(1.0, abs=0.2)


def test_pace_controller_assess_all_layers():
    """PaceController.assess_all returns assessments for every active layer."""
    controller = PaceController()

    all_events = {
        "momentum": [
            {"type": "FIX", "layer": "momentum", "param": "x", "value": 1,
             "timestamp": datetime(2026, 3, 1, tzinfo=timezone.utc)},
        ],
        "mean_reversion": [],
    }

    assessments = controller.assess_all(all_events)
    assert "momentum" in assessments
    assert "mean_reversion" in assessments
    assert assessments["mean_reversion"].pattern == EvolutionPattern.STAGNANT
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_pace_control.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.orchestrator.pace_control'`

**Step 3: Implement evolution pace control**

```python
# src/evolve_trader/orchestrator/pace_control.py
"""Evolution pace control — detect cycles, oscillations, convergence, stagnation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class EvolutionPattern(Enum):
    """Detected evolution pattern for a layer."""

    CONVERGING = "converging"
    OSCILLATING = "oscillating"
    STAGNANT = "stagnant"
    HEALTHY = "healthy"
    DIVERGING = "diverging"


@dataclass
class PaceAssessment:
    """Assessment of evolution pace for a single layer."""

    layer: str
    pattern: EvolutionPattern
    event_count: int
    cycle_detected: bool
    cycle_description: str | None = None
    recommended_action: str = "maintain"
    pace_multiplier: float = 1.0


class PaceController:
    """Monitors evolution logs per layer to detect patterns and recommend pace changes.

    Detects:
    - Oscillations: FIX->un-FIX->re-FIX cycles (same param toggling between values)
    - Convergence: changes getting smaller over time
    - Stagnation: no events in lookback period
    - Healthy: steady, non-repeating evolution
    """

    def __init__(self, oscillation_threshold: int = 3, min_events_for_analysis: int = 2):
        self.oscillation_threshold = oscillation_threshold
        self.min_events_for_analysis = min_events_for_analysis

    def assess_layer(self, layer: str, events: list[dict[str, Any]]) -> PaceAssessment:
        """Assess evolution pace for a single layer."""
        if not events:
            return PaceAssessment(
                layer=layer,
                pattern=EvolutionPattern.STAGNANT,
                event_count=0,
                cycle_detected=False,
                recommended_action="accelerate",
                pace_multiplier=1.5,
            )

        # Check for oscillation: same param alternating between values
        oscillation = self._detect_oscillation(events)
        if oscillation:
            return PaceAssessment(
                layer=layer,
                pattern=EvolutionPattern.OSCILLATING,
                event_count=len(events),
                cycle_detected=True,
                cycle_description=oscillation,
                recommended_action="slow",
                pace_multiplier=0.5,
            )

        # Check for convergence: param changes getting smaller
        convergence = self._detect_convergence(events)
        if convergence:
            return PaceAssessment(
                layer=layer,
                pattern=EvolutionPattern.CONVERGING,
                event_count=len(events),
                cycle_detected=False,
                recommended_action="maintain",
                pace_multiplier=0.9,
            )

        # Default: healthy evolution
        return PaceAssessment(
            layer=layer,
            pattern=EvolutionPattern.HEALTHY,
            event_count=len(events),
            cycle_detected=False,
            recommended_action="maintain",
            pace_multiplier=1.0,
        )

    def assess_all(
        self, events_by_layer: dict[str, list[dict[str, Any]]]
    ) -> dict[str, PaceAssessment]:
        """Assess all layers at once."""
        return {
            layer: self.assess_layer(layer, events)
            for layer, events in events_by_layer.items()
        }

    def _detect_oscillation(self, events: list[dict[str, Any]]) -> str | None:
        """Detect parameter value oscillation patterns.

        Looks for the same parameter alternating between values
        (e.g., stop_loss: 0.02 -> 0.05 -> 0.02 -> 0.05).
        """
        # Group by param
        param_values: dict[str, list[Any]] = {}
        for event in events:
            param = event.get("param")
            if param is None:
                continue
            param_values.setdefault(param, []).append(event.get("value"))

        for param, values in param_values.items():
            if len(values) < self.oscillation_threshold:
                continue
            # Check for alternating pattern
            unique_vals = set(values)
            if len(unique_vals) <= 2 and len(values) >= self.oscillation_threshold:
                return f"FIX->un-FIX->re-FIX on {param} between {unique_vals}"

        return None

    def _detect_convergence(self, events: list[dict[str, Any]]) -> bool:
        """Detect convergence — parameter changes getting progressively smaller."""
        param_values: dict[str, list[float]] = {}
        for event in events:
            param = event.get("param")
            value = event.get("value")
            if param is None or value is None:
                continue
            try:
                param_values.setdefault(param, []).append(float(value))
            except (TypeError, ValueError):
                continue

        for param, values in param_values.items():
            if len(values) < 3:
                continue
            deltas = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
            # Convergence: each delta smaller than previous
            if all(deltas[i] <= deltas[i - 1] for i in range(1, len(deltas))):
                return True

        return False
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_pace_control.py -v
```

Expected: PASS — all pace control tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/orchestrator/pace_control.py tests/unit/test_pace_control.py
git commit -m "feat: evolution pace control detecting oscillations, convergence, stagnation"
```

---

## Task 4: Inter-Layer Tension Detection

**Files:**
- Create: `src/evolve_trader/orchestrator/tension_detector.py`
- Create: `tests/unit/test_tension_detector.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_tension_detector.py
"""Tests for inter-layer tension detection."""
import pytest
from evolve_trader.orchestrator.tension_detector import (
    TensionDetector,
    Tension,
    TensionSeverity,
)


def test_tension_structure():
    """Tension captures a detected inter-layer conflict."""
    tension = Tension(
        layer_a="strategy_evolution",
        layer_b="regime_classifier",
        description="Aggressive momentum strategy paired with oversensitive regime classifier causes whipsaw",
        severity=TensionSeverity.HIGH,
        evidence={"whipsaw_count": 12, "regime_switches": 8},
        suggested_resolution="Reduce classifier sensitivity or add regime-switch cooldown",
    )
    assert tension.severity == TensionSeverity.HIGH
    assert "whipsaw" in tension.description


def test_tension_severity_ordering():
    """TensionSeverity has LOW < MEDIUM < HIGH < CRITICAL ordering."""
    assert TensionSeverity.LOW.value < TensionSeverity.MEDIUM.value
    assert TensionSeverity.MEDIUM.value < TensionSeverity.HIGH.value
    assert TensionSeverity.HIGH.value < TensionSeverity.CRITICAL.value


def test_tension_detector_identifies_strategy_classifier_whipsaw():
    """Detect tension when aggressive strategies + sensitive classifier = whipsaw."""
    detector = TensionDetector()

    layer_states = {
        "strategy_evolution": {
            "active_strategies": ["aggressive_momentum"],
            "avg_trade_frequency": 25,  # High frequency
            "direction_bias": "long",
        },
        "regime_classifier": {
            "regime_switches": 8,  # Many switches in period
            "avg_switch_interval_days": 1.5,
            "current_regime": "risk-off",
        },
    }

    tensions = detector.detect(layer_states)
    assert len(tensions) >= 1
    whipsaw_tensions = [t for t in tensions if "whipsaw" in t.description.lower()
                        or "switch" in t.description.lower()]
    assert len(whipsaw_tensions) >= 1
    assert whipsaw_tensions[0].severity.value >= TensionSeverity.MEDIUM.value


def test_tension_detector_no_tension_in_aligned_system():
    """No tension when layers are well-aligned."""
    detector = TensionDetector()

    layer_states = {
        "strategy_evolution": {
            "active_strategies": ["conservative_value"],
            "avg_trade_frequency": 3,
            "direction_bias": "neutral",
        },
        "regime_classifier": {
            "regime_switches": 1,
            "avg_switch_interval_days": 14.0,
            "current_regime": "risk-on",
        },
    }

    tensions = detector.detect(layer_states)
    assert len(tensions) == 0


def test_tension_detector_constraint_proximity_tension():
    """Detect tension when strategy pushes toward constraint limits."""
    detector = TensionDetector()

    layer_states = {
        "strategy_evolution": {
            "active_strategies": ["aggressive_momentum"],
            "avg_trade_frequency": 20,
            "direction_bias": "long",
        },
        "risk_constraints": {
            "max_drawdown_proximity": 0.85,  # 85% toward limit
            "position_concentration_proximity": 0.7,
        },
    }

    tensions = detector.detect(layer_states)
    constraint_tensions = [t for t in tensions if "constraint" in t.description.lower()
                           or "drawdown" in t.description.lower()]
    assert len(constraint_tensions) >= 1
    assert constraint_tensions[0].severity.value >= TensionSeverity.HIGH.value


def test_tension_detector_discovery_vs_stability():
    """Detect tension when discovery engine promotes too aggressively during drawdown."""
    detector = TensionDetector()

    layer_states = {
        "discovery_engine": {
            "promotion_rate": 0.3,  # 30% of candidates promoted — high
            "candidates_in_trial": 5,
        },
        "portfolio": {
            "current_drawdown": 0.10,
            "portfolio_return": -0.05,
        },
    }

    tensions = detector.detect(layer_states)
    assert len(tensions) >= 1


def test_tension_detector_returns_sorted_by_severity():
    """Tensions are returned sorted by severity, highest first."""
    detector = TensionDetector()

    layer_states = {
        "strategy_evolution": {
            "active_strategies": ["aggressive_momentum"],
            "avg_trade_frequency": 25,
            "direction_bias": "long",
        },
        "regime_classifier": {
            "regime_switches": 10,
            "avg_switch_interval_days": 1.0,
            "current_regime": "risk-off",
        },
        "risk_constraints": {
            "max_drawdown_proximity": 0.9,
            "position_concentration_proximity": 0.8,
        },
        "discovery_engine": {
            "promotion_rate": 0.4,
            "candidates_in_trial": 8,
        },
        "portfolio": {
            "current_drawdown": 0.12,
            "portfolio_return": -0.08,
        },
    }

    tensions = detector.detect(layer_states)
    if len(tensions) >= 2:
        for i in range(len(tensions) - 1):
            assert tensions[i].severity.value >= tensions[i + 1].severity.value
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_tension_detector.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.orchestrator.tension_detector'`

**Step 3: Implement inter-layer tension detection**

```python
# src/evolve_trader/orchestrator/tension_detector.py
"""Inter-layer tension detection — identify layers pulling opposite directions."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class TensionSeverity(IntEnum):
    """Severity levels for detected tensions."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Tension:
    """A detected inter-layer tension/conflict."""

    layer_a: str
    layer_b: str
    description: str
    severity: TensionSeverity
    evidence: dict[str, Any] = field(default_factory=dict)
    suggested_resolution: str = ""


class TensionDetector:
    """Identifies inter-layer tensions where components pull in opposite directions.

    Detects patterns like:
    - Aggressive strategies + sensitive classifier = whipsaw
    - High trade frequency + near constraint limits = risk
    - Aggressive discovery + drawdown = instability
    """

    def __init__(
        self,
        regime_switch_threshold: int = 5,
        constraint_proximity_threshold: float = 0.75,
        promotion_rate_threshold: float = 0.2,
        drawdown_concern_threshold: float = 0.05,
        trade_frequency_threshold: int = 15,
    ):
        self.regime_switch_threshold = regime_switch_threshold
        self.constraint_proximity_threshold = constraint_proximity_threshold
        self.promotion_rate_threshold = promotion_rate_threshold
        self.drawdown_concern_threshold = drawdown_concern_threshold
        self.trade_frequency_threshold = trade_frequency_threshold

    def detect(self, layer_states: dict[str, dict[str, Any]]) -> list[Tension]:
        """Detect all tensions across provided layer states.

        Returns tensions sorted by severity (highest first).
        """
        tensions: list[Tension] = []

        tensions.extend(self._check_strategy_classifier_whipsaw(layer_states))
        tensions.extend(self._check_constraint_proximity(layer_states))
        tensions.extend(self._check_discovery_stability(layer_states))

        tensions.sort(key=lambda t: t.severity.value, reverse=True)
        return tensions

    def _check_strategy_classifier_whipsaw(
        self, states: dict[str, dict[str, Any]]
    ) -> list[Tension]:
        """Check for whipsaw from aggressive strategies + sensitive classifier."""
        strat = states.get("strategy_evolution")
        classifier = states.get("regime_classifier")
        if not strat or not classifier:
            return []

        trade_freq = strat.get("avg_trade_frequency", 0)
        regime_switches = classifier.get("regime_switches", 0)

        if (
            trade_freq >= self.trade_frequency_threshold
            and regime_switches >= self.regime_switch_threshold
        ):
            severity = TensionSeverity.HIGH if regime_switches >= 8 else TensionSeverity.MEDIUM
            return [
                Tension(
                    layer_a="strategy_evolution",
                    layer_b="regime_classifier",
                    description=(
                        f"High trade frequency ({trade_freq}) combined with frequent "
                        f"regime switches ({regime_switches}) causes whipsaw risk"
                    ),
                    severity=severity,
                    evidence={
                        "trade_frequency": trade_freq,
                        "regime_switches": regime_switches,
                    },
                    suggested_resolution=(
                        "Reduce classifier sensitivity, add regime-switch cooldown, "
                        "or reduce strategy aggressiveness"
                    ),
                )
            ]
        return []

    def _check_constraint_proximity(
        self, states: dict[str, dict[str, Any]]
    ) -> list[Tension]:
        """Check for strategies pushing toward constraint limits."""
        strat = states.get("strategy_evolution")
        constraints = states.get("risk_constraints")
        if not strat or not constraints:
            return []

        tensions = []
        for constraint_name, proximity in constraints.items():
            if proximity >= self.constraint_proximity_threshold:
                severity = (
                    TensionSeverity.CRITICAL
                    if proximity >= 0.9
                    else TensionSeverity.HIGH
                )
                tensions.append(
                    Tension(
                        layer_a="strategy_evolution",
                        layer_b="risk_constraints",
                        description=(
                            f"Strategy pushing toward {constraint_name} constraint "
                            f"limit ({proximity:.0%} proximity, drawdown risk)"
                        ),
                        severity=severity,
                        evidence={
                            "constraint": constraint_name,
                            "proximity": proximity,
                            "strategies": strat.get("active_strategies", []),
                        },
                        suggested_resolution=(
                            f"Reduce exposure or tighten {constraint_name} monitoring threshold"
                        ),
                    )
                )
        return tensions

    def _check_discovery_stability(
        self, states: dict[str, dict[str, Any]]
    ) -> list[Tension]:
        """Check for aggressive discovery during drawdown periods."""
        discovery = states.get("discovery_engine")
        portfolio = states.get("portfolio")
        if not discovery or not portfolio:
            return []

        promotion_rate = discovery.get("promotion_rate", 0)
        drawdown = portfolio.get("current_drawdown", 0)

        if (
            promotion_rate > self.promotion_rate_threshold
            and drawdown > self.drawdown_concern_threshold
        ):
            severity = TensionSeverity.HIGH if drawdown > 0.10 else TensionSeverity.MEDIUM
            return [
                Tension(
                    layer_a="discovery_engine",
                    layer_b="portfolio",
                    description=(
                        f"Discovery engine promoting aggressively ({promotion_rate:.0%}) "
                        f"during drawdown period ({drawdown:.1%})"
                    ),
                    severity=severity,
                    evidence={
                        "promotion_rate": promotion_rate,
                        "drawdown": drawdown,
                        "candidates_in_trial": discovery.get("candidates_in_trial", 0),
                    },
                    suggested_resolution=(
                        "Reduce discovery promotion rate until drawdown recovers"
                    ),
                )
            ]
        return []
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_tension_detector.py -v
```

Expected: PASS — all tension detection tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/orchestrator/tension_detector.py tests/unit/test_tension_detector.py
git commit -m "feat: inter-layer tension detection for strategy/classifier whipsaw, constraint proximity, discovery stability"
```

---

## Task 5: Counterfactual Replay Engine

**Files:**
- Create: `src/evolve_trader/orchestrator/counterfactual.py`
- Create: `tests/unit/test_counterfactual.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_counterfactual.py
"""Tests for the counterfactual replay engine."""
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock
from evolve_trader.orchestrator.counterfactual import (
    CounterfactualEngine,
    ReplayResult,
    ProposedChange,
    TradeRecord,
)


def test_trade_record_structure():
    """TradeRecord captures a historical trade for replay."""
    trade = TradeRecord(
        timestamp=datetime(2026, 3, 5, tzinfo=timezone.utc),
        ticker="AAPL",
        direction="BUY",
        quantity=10,
        entry_price=180.0,
        exit_price=185.0,
        pnl=50.0,
        strategy="momentum-v3",
        regime_label="risk-on",
    )
    assert trade.pnl == 50.0
    assert trade.strategy == "momentum-v3"


def test_proposed_change_structure():
    """ProposedChange captures an adjustment to replay against history."""
    change = ProposedChange(
        layer="evolution_pace",
        target="momentum",
        parameter="pace_multiplier",
        current_value=1.0,
        proposed_value=0.5,
    )
    assert change.proposed_value == 0.5


def test_replay_result_structure():
    """ReplayResult captures the outcome of a counterfactual replay."""
    result = ReplayResult(
        change=ProposedChange(
            layer="threshold",
            target="drawdown_monitor",
            parameter="threshold",
            current_value=0.10,
            proposed_value=0.08,
        ),
        original_pnl=500.0,
        counterfactual_pnl=420.0,
        original_max_drawdown=0.08,
        counterfactual_max_drawdown=0.06,
        original_trade_count=15,
        counterfactual_trade_count=12,
        improvement_pnl=-80.0,
        improvement_drawdown=0.02,
    )
    assert result.improvement_pnl == -80.0
    assert result.improvement_drawdown == 0.02  # Positive = less drawdown = better


def test_counterfactual_engine_creation():
    """CounterfactualEngine initializes."""
    engine = CounterfactualEngine()
    assert engine is not None


def test_counterfactual_engine_replay_single_change():
    """Engine replays history with a single proposed change."""
    engine = CounterfactualEngine()

    trades = [
        TradeRecord(
            timestamp=datetime(2026, 3, i, tzinfo=timezone.utc),
            ticker="AAPL",
            direction="BUY",
            quantity=10,
            entry_price=180.0,
            exit_price=180.0 + (i * 0.5 if i % 2 == 0 else -i * 0.3),
            pnl=(i * 0.5 if i % 2 == 0 else -i * 0.3) * 10,
            strategy="momentum-v3",
            regime_label="risk-on",
        )
        for i in range(1, 11)
    ]

    change = ProposedChange(
        layer="threshold",
        target="drawdown_monitor",
        parameter="max_drawdown_threshold",
        current_value=0.10,
        proposed_value=0.07,
    )

    # Provide a trade filter function
    def tighter_drawdown_filter(trade: TradeRecord, change: ProposedChange) -> bool:
        """Simulates filtering out trades that would breach tighter drawdown."""
        return trade.pnl >= -20.0  # Only keep trades with modest losses

    result = engine.replay(trades=trades, change=change, trade_filter=tighter_drawdown_filter)

    assert isinstance(result, ReplayResult)
    assert result.original_trade_count == 10
    assert result.counterfactual_trade_count <= 10
    assert result.change == change


def test_counterfactual_engine_batch_replay():
    """Engine batch-replays multiple changes efficiently."""
    engine = CounterfactualEngine()

    trades = [
        TradeRecord(
            timestamp=datetime(2026, 3, 1, tzinfo=timezone.utc),
            ticker="AAPL",
            direction="BUY",
            quantity=10,
            entry_price=180.0,
            exit_price=185.0,
            pnl=50.0,
            strategy="momentum-v3",
            regime_label="risk-on",
        ),
    ]

    changes = [
        ProposedChange(
            layer="threshold", target="a", parameter="x",
            current_value=0.10, proposed_value=0.08,
        ),
        ProposedChange(
            layer="pace", target="b", parameter="y",
            current_value=1.0, proposed_value=0.5,
        ),
    ]

    # Default filter: keep all trades
    results = engine.batch_replay(trades=trades, changes=changes)

    assert len(results) == 2
    assert all(isinstance(r, ReplayResult) for r in results)


def test_counterfactual_engine_empty_trades():
    """Engine handles empty trade history gracefully."""
    engine = CounterfactualEngine()

    change = ProposedChange(
        layer="threshold", target="a", parameter="x",
        current_value=0.10, proposed_value=0.08,
    )

    result = engine.replay(trades=[], change=change)
    assert result.original_trade_count == 0
    assert result.counterfactual_trade_count == 0
    assert result.original_pnl == 0.0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_counterfactual.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.orchestrator.counterfactual'`

**Step 3: Implement the counterfactual replay engine**

```python
# src/evolve_trader/orchestrator/counterfactual.py
"""Counterfactual replay engine — estimate impact of proposed adjustments."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import numpy as np


@dataclass
class TradeRecord:
    """A historical trade record for replay."""

    timestamp: datetime
    ticker: str
    direction: str
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    strategy: str
    regime_label: str


@dataclass
class ProposedChange:
    """A proposed adjustment to replay against history."""

    layer: str
    target: str
    parameter: str
    current_value: float
    proposed_value: float


@dataclass
class ReplayResult:
    """Outcome of a counterfactual replay."""

    change: ProposedChange
    original_pnl: float
    counterfactual_pnl: float
    original_max_drawdown: float
    counterfactual_max_drawdown: float
    original_trade_count: int
    counterfactual_trade_count: int
    improvement_pnl: float
    improvement_drawdown: float


# Default filter: keep all trades
def _default_filter(trade: TradeRecord, change: ProposedChange) -> bool:
    return True


class CounterfactualEngine:
    """Replays last period's trades with proposed adjustments to estimate impact.

    Supports single and batch replay for efficiency. Each replay applies
    a trade_filter function that simulates the effect of the proposed change
    on historical trades.
    """

    def replay(
        self,
        trades: list[TradeRecord],
        change: ProposedChange,
        trade_filter: Callable[[TradeRecord, ProposedChange], bool] | None = None,
    ) -> ReplayResult:
        """Replay trades with a single proposed change.

        Args:
            trades: Historical trade records for the period.
            change: The proposed adjustment.
            trade_filter: Function that returns True if the trade would
                          still have been taken under the proposed change.
                          Defaults to keeping all trades.
        """
        if trade_filter is None:
            trade_filter = _default_filter

        original_pnl = sum(t.pnl for t in trades)
        original_dd = self._compute_max_drawdown([t.pnl for t in trades])

        filtered_trades = [t for t in trades if trade_filter(t, change)]
        counterfactual_pnl = sum(t.pnl for t in filtered_trades)
        counterfactual_dd = self._compute_max_drawdown([t.pnl for t in filtered_trades])

        return ReplayResult(
            change=change,
            original_pnl=original_pnl,
            counterfactual_pnl=counterfactual_pnl,
            original_max_drawdown=original_dd,
            counterfactual_max_drawdown=counterfactual_dd,
            original_trade_count=len(trades),
            counterfactual_trade_count=len(filtered_trades),
            improvement_pnl=counterfactual_pnl - original_pnl,
            improvement_drawdown=original_dd - counterfactual_dd,
        )

    def batch_replay(
        self,
        trades: list[TradeRecord],
        changes: list[ProposedChange],
        trade_filters: dict[int, Callable[[TradeRecord, ProposedChange], bool]] | None = None,
    ) -> list[ReplayResult]:
        """Batch-replay multiple proposed changes against the same trade history.

        Args:
            trades: Historical trade records.
            changes: List of proposed adjustments.
            trade_filters: Optional per-change filters keyed by index.
                           Defaults to keeping all trades for each change.
        """
        results = []
        for i, change in enumerate(changes):
            filt = (trade_filters or {}).get(i, _default_filter)
            results.append(self.replay(trades, change, trade_filter=filt))
        return results

    @staticmethod
    def _compute_max_drawdown(pnls: list[float]) -> float:
        """Compute max drawdown from a sequence of PnL values."""
        if not pnls:
            return 0.0
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_counterfactual.py -v
```

Expected: PASS — all counterfactual replay tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/orchestrator/counterfactual.py tests/unit/test_counterfactual.py
git commit -m "feat: counterfactual replay engine for estimating adjustment impact before applying"
```

---

## Task 6: Monitoring Threshold Calibration

**Files:**
- Create: `src/evolve_trader/orchestrator/threshold_calibrator.py`
- Create: `tests/unit/test_threshold_calibrator.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_threshold_calibrator.py
"""Tests for monitoring threshold calibration."""
import pytest
from evolve_trader.orchestrator.threshold_calibrator import (
    ThresholdCalibrator,
    ThresholdAdjustment,
    CalibrationContext,
)


def test_threshold_adjustment_structure():
    """ThresholdAdjustment captures a proposed threshold change."""
    adj = ThresholdAdjustment(
        metric_name="max_drawdown",
        current_threshold=0.10,
        proposed_threshold=0.08,
        reason="Recent drawdown approaching limit — tighten for safety",
        confidence=0.85,
    )
    assert adj.proposed_threshold == 0.08
    assert adj.confidence == 0.85


def test_calibration_context_structure():
    """CalibrationContext provides performance data for calibration decisions."""
    ctx = CalibrationContext(
        portfolio_return=0.05,
        max_drawdown=0.04,
        sharpe_ratio=1.5,
        period_days=14,
        constraint_proximity={"max_drawdown": 0.3, "concentration": 0.2},
        recent_violations=0,
    )
    assert ctx.portfolio_return == 0.05
    assert ctx.recent_violations == 0


def test_calibrator_relaxes_thresholds_when_profitable():
    """When profitable with low risk, calibrator relaxes thresholds slightly."""
    calibrator = ThresholdCalibrator()

    ctx = CalibrationContext(
        portfolio_return=0.08,
        max_drawdown=0.02,
        sharpe_ratio=2.0,
        period_days=14,
        constraint_proximity={"max_drawdown": 0.15},
        recent_violations=0,
    )

    current_thresholds = {"max_drawdown": 0.08}
    adjustments = calibrator.calibrate(current_thresholds, ctx)

    # Should recommend relaxing (increasing) the threshold slightly
    dd_adj = [a for a in adjustments if a.metric_name == "max_drawdown"]
    if dd_adj:
        assert dd_adj[0].proposed_threshold >= dd_adj[0].current_threshold


def test_calibrator_tightens_thresholds_during_drawdown():
    """When in drawdown, calibrator tightens thresholds."""
    calibrator = ThresholdCalibrator()

    ctx = CalibrationContext(
        portfolio_return=-0.05,
        max_drawdown=0.09,
        sharpe_ratio=-0.3,
        period_days=14,
        constraint_proximity={"max_drawdown": 0.75},
        recent_violations=2,
    )

    current_thresholds = {"max_drawdown": 0.10}
    adjustments = calibrator.calibrate(current_thresholds, ctx)

    dd_adj = [a for a in adjustments if a.metric_name == "max_drawdown"]
    assert len(dd_adj) >= 1
    assert dd_adj[0].proposed_threshold < dd_adj[0].current_threshold


def test_calibrator_respects_min_max_bounds():
    """Calibrator never pushes thresholds beyond safety bounds."""
    calibrator = ThresholdCalibrator(
        min_thresholds={"max_drawdown": 0.03},
        max_thresholds={"max_drawdown": 0.15},
    )

    # Very profitable — wants to relax a lot
    ctx = CalibrationContext(
        portfolio_return=0.20,
        max_drawdown=0.01,
        sharpe_ratio=4.0,
        period_days=14,
        constraint_proximity={"max_drawdown": 0.05},
        recent_violations=0,
    )

    current_thresholds = {"max_drawdown": 0.14}
    adjustments = calibrator.calibrate(current_thresholds, ctx)

    for adj in adjustments:
        if adj.metric_name == "max_drawdown":
            assert adj.proposed_threshold <= 0.15
            assert adj.proposed_threshold >= 0.03


def test_calibrator_no_change_when_stable():
    """Calibrator makes no changes when system is in a stable band."""
    calibrator = ThresholdCalibrator()

    ctx = CalibrationContext(
        portfolio_return=0.02,
        max_drawdown=0.04,
        sharpe_ratio=1.0,
        period_days=14,
        constraint_proximity={"max_drawdown": 0.40},
        recent_violations=0,
    )

    current_thresholds = {"max_drawdown": 0.10}
    adjustments = calibrator.calibrate(current_thresholds, ctx)

    # Either no adjustments or very small ones
    for adj in adjustments:
        delta = abs(adj.proposed_threshold - adj.current_threshold)
        assert delta < 0.02  # Less than 2% change
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_threshold_calibrator.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.orchestrator.threshold_calibrator'`

**Step 3: Implement monitoring threshold calibration**

```python
# src/evolve_trader/orchestrator/threshold_calibrator.py
"""Monitoring threshold calibration — meta-evolution loop for system thresholds."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CalibrationContext:
    """Performance context for threshold calibration decisions."""

    portfolio_return: float
    max_drawdown: float
    sharpe_ratio: float
    period_days: int
    constraint_proximity: dict[str, float]
    recent_violations: int


@dataclass
class ThresholdAdjustment:
    """A proposed threshold change."""

    metric_name: str
    current_threshold: float
    proposed_threshold: float
    reason: str
    confidence: float


class ThresholdCalibrator:
    """Tunes monitoring thresholds based on performance.

    Profitable periods with low risk → relax thresholds slightly.
    Drawdown periods or constraint proximity → tighten thresholds.
    Stable performance → no changes.

    All adjustments are bounded by min/max safety limits.
    """

    def __init__(
        self,
        relax_rate: float = 0.05,
        tighten_rate: float = 0.10,
        stability_band: float = 0.3,
        min_thresholds: dict[str, float] | None = None,
        max_thresholds: dict[str, float] | None = None,
    ):
        self.relax_rate = relax_rate
        self.tighten_rate = tighten_rate
        self.stability_band = stability_band
        self.min_thresholds = min_thresholds or {}
        self.max_thresholds = max_thresholds or {}

    def calibrate(
        self,
        current_thresholds: dict[str, float],
        context: CalibrationContext,
    ) -> list[ThresholdAdjustment]:
        """Calibrate all thresholds based on recent performance context.

        Returns list of proposed adjustments (may be empty if stable).
        """
        adjustments = []

        for metric_name, current_value in current_thresholds.items():
            proximity = context.constraint_proximity.get(metric_name, 0.0)
            adjustment = self._calibrate_single(
                metric_name, current_value, proximity, context
            )
            if adjustment is not None:
                adjustments.append(adjustment)

        return adjustments

    def _calibrate_single(
        self,
        metric_name: str,
        current_value: float,
        proximity: float,
        context: CalibrationContext,
    ) -> ThresholdAdjustment | None:
        """Calibrate a single threshold."""
        # Determine direction
        should_tighten = (
            context.portfolio_return < 0
            or context.recent_violations > 0
            or proximity > 0.6
        )
        should_relax = (
            context.portfolio_return > 0.03
            and context.sharpe_ratio > 1.0
            and proximity < 0.3
            and context.recent_violations == 0
        )

        if should_tighten:
            # Tighten: reduce threshold
            magnitude = self.tighten_rate * (1 + proximity)
            proposed = current_value * (1 - magnitude)
            proposed = self._clamp(metric_name, proposed)
            delta = abs(proposed - current_value)
            if delta < current_value * 0.005:
                return None
            return ThresholdAdjustment(
                metric_name=metric_name,
                current_threshold=current_value,
                proposed_threshold=round(proposed, 6),
                reason=f"Tightening due to {'drawdown' if context.portfolio_return < 0 else 'constraint proximity'}",
                confidence=min(0.9, 0.5 + proximity),
            )

        if should_relax:
            # Relax: increase threshold
            proposed = current_value * (1 + self.relax_rate)
            proposed = self._clamp(metric_name, proposed)
            delta = abs(proposed - current_value)
            if delta < current_value * 0.005:
                return None
            return ThresholdAdjustment(
                metric_name=metric_name,
                current_threshold=current_value,
                proposed_threshold=round(proposed, 6),
                reason="Relaxing — strong performance with low risk",
                confidence=0.7,
            )

        # Stable — no change
        return None

    def _clamp(self, metric_name: str, value: float) -> float:
        """Clamp threshold to min/max safety bounds."""
        min_val = self.min_thresholds.get(metric_name, 0.0)
        max_val = self.max_thresholds.get(metric_name, float("inf"))
        return max(min_val, min(max_val, value))
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_threshold_calibrator.py -v
```

Expected: PASS — all threshold calibration tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/orchestrator/threshold_calibrator.py tests/unit/test_threshold_calibrator.py
git commit -m "feat: monitoring threshold calibration with meta-evolution loop"
```

---

## Task 7: Discovery Engine Tuning Interface

**Files:**
- Create: `src/evolve_trader/orchestrator/discovery_tuner.py`
- Create: `tests/unit/test_discovery_tuner.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_discovery_tuner.py
"""Tests for discovery engine tuning interface."""
import pytest
from evolve_trader.orchestrator.discovery_tuner import (
    DiscoveryTuner,
    DiscoveryTuningResult,
    RosterHealth,
)


def test_roster_health_structure():
    """RosterHealth captures current strategy roster state."""
    health = RosterHealth(
        active_count=8,
        avg_sharpe=1.3,
        avg_return=0.04,
        underperformers=1,
        total_candidates_in_trial=3,
        promotion_rate=0.15,
        recent_promotions=2,
        recent_demotions=1,
    )
    assert health.active_count == 8
    assert health.avg_sharpe == 1.3


def test_discovery_tuning_result_structure():
    """DiscoveryTuningResult captures the tuning recommendation."""
    result = DiscoveryTuningResult(
        current_aggressiveness=0.5,
        recommended_aggressiveness=0.3,
        reason="Roster performing well — slow discovery to maintain stability",
        promotion_rate_adjustment=-0.05,
        trial_duration_adjustment=2,
    )
    assert result.recommended_aggressiveness == 0.3
    assert result.trial_duration_adjustment == 2


def test_tuner_slows_discovery_for_good_roster():
    """Good roster (high Sharpe, few underperformers) → slow discovery."""
    tuner = DiscoveryTuner()

    health = RosterHealth(
        active_count=10,
        avg_sharpe=1.8,
        avg_return=0.06,
        underperformers=0,
        total_candidates_in_trial=4,
        promotion_rate=0.2,
        recent_promotions=3,
        recent_demotions=0,
    )

    result = tuner.tune(health, current_aggressiveness=0.5)
    assert result.recommended_aggressiveness < 0.5
    assert result.promotion_rate_adjustment <= 0


def test_tuner_accelerates_discovery_for_poor_roster():
    """Poor roster (low Sharpe, many underperformers) → accelerate discovery."""
    tuner = DiscoveryTuner()

    health = RosterHealth(
        active_count=5,
        avg_sharpe=0.3,
        avg_return=-0.01,
        underperformers=3,
        total_candidates_in_trial=1,
        promotion_rate=0.05,
        recent_promotions=0,
        recent_demotions=2,
    )

    result = tuner.tune(health, current_aggressiveness=0.3)
    assert result.recommended_aggressiveness > 0.3
    assert result.promotion_rate_adjustment >= 0


def test_tuner_maintains_discovery_for_average_roster():
    """Average roster → maintain current discovery pace."""
    tuner = DiscoveryTuner()

    health = RosterHealth(
        active_count=8,
        avg_sharpe=0.9,
        avg_return=0.02,
        underperformers=1,
        total_candidates_in_trial=2,
        promotion_rate=0.10,
        recent_promotions=1,
        recent_demotions=1,
    )

    result = tuner.tune(health, current_aggressiveness=0.5)
    delta = abs(result.recommended_aggressiveness - 0.5)
    assert delta < 0.15  # Small change


def test_tuner_clamps_aggressiveness():
    """Aggressiveness is clamped to [0.0, 1.0] range."""
    tuner = DiscoveryTuner()

    # Very bad roster — wants to push aggressiveness very high
    health = RosterHealth(
        active_count=2,
        avg_sharpe=-0.5,
        avg_return=-0.10,
        underperformers=2,
        total_candidates_in_trial=0,
        promotion_rate=0.0,
        recent_promotions=0,
        recent_demotions=2,
    )

    result = tuner.tune(health, current_aggressiveness=0.9)
    assert 0.0 <= result.recommended_aggressiveness <= 1.0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_discovery_tuner.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.orchestrator.discovery_tuner'`

**Step 3: Implement the discovery engine tuning interface**

```python
# src/evolve_trader/orchestrator/discovery_tuner.py
"""Discovery engine tuning — adjust promotion aggressiveness based on roster health."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RosterHealth:
    """Current strategy roster health metrics."""

    active_count: int
    avg_sharpe: float
    avg_return: float
    underperformers: int
    total_candidates_in_trial: int
    promotion_rate: float
    recent_promotions: int
    recent_demotions: int


@dataclass
class DiscoveryTuningResult:
    """Result of discovery tuning analysis."""

    current_aggressiveness: float
    recommended_aggressiveness: float
    reason: str
    promotion_rate_adjustment: float
    trial_duration_adjustment: int  # Days to add/subtract from trial period


class DiscoveryTuner:
    """Adjusts discovery engine promotion aggressiveness based on roster health.

    Good roster (high Sharpe, few underperformers) → slow discovery.
    Underperforming roster → accelerate discovery and promotion.
    """

    def __init__(
        self,
        sharpe_good_threshold: float = 1.2,
        sharpe_poor_threshold: float = 0.5,
        underperformer_ratio_threshold: float = 0.3,
        adjustment_step: float = 0.15,
    ):
        self.sharpe_good_threshold = sharpe_good_threshold
        self.sharpe_poor_threshold = sharpe_poor_threshold
        self.underperformer_ratio_threshold = underperformer_ratio_threshold
        self.adjustment_step = adjustment_step

    def tune(
        self,
        health: RosterHealth,
        current_aggressiveness: float,
    ) -> DiscoveryTuningResult:
        """Analyze roster health and recommend discovery aggressiveness changes."""
        underperformer_ratio = (
            health.underperformers / health.active_count
            if health.active_count > 0
            else 1.0
        )

        is_good = (
            health.avg_sharpe >= self.sharpe_good_threshold
            and underperformer_ratio < self.underperformer_ratio_threshold
        )
        is_poor = (
            health.avg_sharpe < self.sharpe_poor_threshold
            or underperformer_ratio >= self.underperformer_ratio_threshold
            or health.active_count < 3
        )

        if is_good:
            new_aggressiveness = current_aggressiveness - self.adjustment_step
            promotion_adj = -0.05
            trial_adj = 2  # Longer trials when stable
            reason = "Roster performing well — slow discovery to maintain stability"
        elif is_poor:
            new_aggressiveness = current_aggressiveness + self.adjustment_step
            promotion_adj = 0.05
            trial_adj = -1  # Shorter trials to find replacements faster
            reason = "Roster underperforming — accelerate discovery and promotion"
        else:
            new_aggressiveness = current_aggressiveness
            promotion_adj = 0.0
            trial_adj = 0
            reason = "Roster at acceptable level — maintain current discovery pace"

        new_aggressiveness = max(0.0, min(1.0, new_aggressiveness))

        return DiscoveryTuningResult(
            current_aggressiveness=current_aggressiveness,
            recommended_aggressiveness=round(new_aggressiveness, 4),
            reason=reason,
            promotion_rate_adjustment=promotion_adj,
            trial_duration_adjustment=trial_adj,
        )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_discovery_tuner.py -v
```

Expected: PASS — all discovery tuner tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/orchestrator/discovery_tuner.py tests/unit/test_discovery_tuner.py
git commit -m "feat: discovery engine tuning interface adjusting promotion aggressiveness"
```

---

## Task 8: Cross-Layer Correlation Analysis

**Files:**
- Create: `src/evolve_trader/orchestrator/cross_layer.py`
- Create: `tests/unit/test_cross_layer.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_cross_layer.py
"""Tests for cross-layer correlation analysis."""
import pytest
import numpy as np
from evolve_trader.orchestrator.cross_layer import (
    CrossLayerAnalyzer,
    LayerPerformanceSeries,
    CorrelationResult,
    SystemHealthState,
)


def test_layer_performance_series_structure():
    """LayerPerformanceSeries holds time-series performance for a layer."""
    series = LayerPerformanceSeries(
        layer_name="momentum",
        timestamps=["2026-03-01", "2026-03-08", "2026-03-15"],
        values=[0.02, 0.03, 0.01],
    )
    assert series.layer_name == "momentum"
    assert len(series.values) == 3


def test_correlation_result_structure():
    """CorrelationResult captures pairwise layer correlation."""
    result = CorrelationResult(
        layer_a="momentum",
        layer_b="mean_reversion",
        correlation=0.85,
        is_offsetting=False,
        is_healthy=True,
        description="Layers improving together — healthy alignment",
    )
    assert result.correlation == 0.85
    assert result.is_healthy is True


def test_system_health_state_enum():
    """SystemHealthState covers key system states."""
    assert SystemHealthState.HEALTHY.value == "healthy"
    assert SystemHealthState.DEGRADING.value == "degrading"
    assert SystemHealthState.OFFSETTING.value == "offsetting"
    assert SystemHealthState.UNSTABLE.value == "unstable"


def test_cross_layer_analyzer_detects_positive_correlation():
    """Analyzer detects layers improving together."""
    analyzer = CrossLayerAnalyzer()

    series = [
        LayerPerformanceSeries("momentum", ["t1", "t2", "t3", "t4", "t5"],
                               [0.01, 0.02, 0.03, 0.04, 0.05]),
        LayerPerformanceSeries("mean_reversion", ["t1", "t2", "t3", "t4", "t5"],
                               [0.02, 0.03, 0.04, 0.05, 0.06]),
    ]

    results = analyzer.analyze(series)
    assert len(results) == 1  # One pair
    assert results[0].correlation > 0.9
    assert results[0].is_healthy is True


def test_cross_layer_analyzer_detects_offsetting():
    """Analyzer detects one layer improving while another degrades."""
    analyzer = CrossLayerAnalyzer()

    series = [
        LayerPerformanceSeries("momentum", ["t1", "t2", "t3", "t4", "t5"],
                               [0.05, 0.04, 0.03, 0.02, 0.01]),
        LayerPerformanceSeries("signal_quality", ["t1", "t2", "t3", "t4", "t5"],
                               [0.01, 0.02, 0.03, 0.04, 0.05]),
    ]

    results = analyzer.analyze(series)
    assert len(results) == 1
    assert results[0].correlation < -0.9
    assert results[0].is_offsetting is True
    assert results[0].is_healthy is False


def test_cross_layer_analyzer_overall_health():
    """Analyzer computes overall system health from all correlations."""
    analyzer = CrossLayerAnalyzer()

    series = [
        LayerPerformanceSeries("a", ["t1", "t2", "t3", "t4", "t5"],
                               [0.01, 0.02, 0.03, 0.04, 0.05]),
        LayerPerformanceSeries("b", ["t1", "t2", "t3", "t4", "t5"],
                               [0.02, 0.03, 0.04, 0.05, 0.06]),
        LayerPerformanceSeries("c", ["t1", "t2", "t3", "t4", "t5"],
                               [0.03, 0.04, 0.05, 0.06, 0.07]),
    ]

    health = analyzer.assess_system_health(series)
    assert health == SystemHealthState.HEALTHY


def test_cross_layer_analyzer_detects_unstable_system():
    """Analyzer flags unstable system when multiple layers offset."""
    analyzer = CrossLayerAnalyzer()

    series = [
        LayerPerformanceSeries("a", ["t1", "t2", "t3", "t4", "t5"],
                               [0.05, 0.04, 0.03, 0.02, 0.01]),
        LayerPerformanceSeries("b", ["t1", "t2", "t3", "t4", "t5"],
                               [0.01, 0.02, 0.03, 0.04, 0.05]),
        LayerPerformanceSeries("c", ["t1", "t2", "t3", "t4", "t5"],
                               [0.05, 0.03, 0.01, -0.01, -0.03]),
    ]

    health = analyzer.assess_system_health(series)
    assert health in (SystemHealthState.OFFSETTING, SystemHealthState.UNSTABLE)


def test_cross_layer_analyzer_handles_single_layer():
    """Analyzer handles single layer gracefully — no pairs to correlate."""
    analyzer = CrossLayerAnalyzer()

    series = [
        LayerPerformanceSeries("momentum", ["t1", "t2", "t3"], [0.01, 0.02, 0.03]),
    ]

    results = analyzer.analyze(series)
    assert len(results) == 0

    health = analyzer.assess_system_health(series)
    assert health == SystemHealthState.HEALTHY
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_cross_layer.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.orchestrator.cross_layer'`

**Step 3: Implement cross-layer correlation analysis**

```python
# src/evolve_trader/orchestrator/cross_layer.py
"""Cross-layer correlation analysis — monitor improvement offsets between layers."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations

import numpy as np


class SystemHealthState(Enum):
    """Overall system health assessment."""

    HEALTHY = "healthy"
    DEGRADING = "degrading"
    OFFSETTING = "offsetting"
    UNSTABLE = "unstable"


@dataclass
class LayerPerformanceSeries:
    """Time-series performance data for a single layer."""

    layer_name: str
    timestamps: list[str]
    values: list[float]


@dataclass
class CorrelationResult:
    """Pairwise correlation result between two layers."""

    layer_a: str
    layer_b: str
    correlation: float
    is_offsetting: bool
    is_healthy: bool
    description: str


class CrossLayerAnalyzer:
    """Monitors cross-layer correlations to detect offsetting improvements.

    When one layer improves while another degrades (strong negative correlation),
    that is an unhealthy offsetting pattern. When layers improve together
    (positive correlation), the system is healthy.
    """

    def __init__(
        self,
        offsetting_threshold: float = -0.5,
        healthy_threshold: float = 0.0,
        unstable_ratio_threshold: float = 0.5,
    ):
        self.offsetting_threshold = offsetting_threshold
        self.healthy_threshold = healthy_threshold
        self.unstable_ratio_threshold = unstable_ratio_threshold

    def analyze(self, series: list[LayerPerformanceSeries]) -> list[CorrelationResult]:
        """Compute pairwise correlations between all layers."""
        results = []

        for a, b in combinations(series, 2):
            corr = self._compute_correlation(a.values, b.values)
            is_offsetting = corr < self.offsetting_threshold
            is_healthy = corr >= self.healthy_threshold

            if is_offsetting:
                desc = (
                    f"{a.layer_name} and {b.layer_name} are offsetting "
                    f"(correlation={corr:.2f}) — improvement in one degrades the other"
                )
            elif is_healthy:
                desc = (
                    f"{a.layer_name} and {b.layer_name} are aligned "
                    f"(correlation={corr:.2f}) — healthy co-movement"
                )
            else:
                desc = (
                    f"{a.layer_name} and {b.layer_name} weakly correlated "
                    f"(correlation={corr:.2f}) — monitoring"
                )

            results.append(
                CorrelationResult(
                    layer_a=a.layer_name,
                    layer_b=b.layer_name,
                    correlation=round(corr, 4),
                    is_offsetting=is_offsetting,
                    is_healthy=is_healthy,
                    description=desc,
                )
            )

        return results

    def assess_system_health(
        self, series: list[LayerPerformanceSeries]
    ) -> SystemHealthState:
        """Assess overall system health from cross-layer correlations."""
        if len(series) < 2:
            return SystemHealthState.HEALTHY

        results = self.analyze(series)
        if not results:
            return SystemHealthState.HEALTHY

        offsetting_count = sum(1 for r in results if r.is_offsetting)
        total_pairs = len(results)
        offsetting_ratio = offsetting_count / total_pairs

        if offsetting_ratio >= self.unstable_ratio_threshold:
            return SystemHealthState.UNSTABLE
        elif offsetting_count > 0:
            return SystemHealthState.OFFSETTING
        elif all(r.is_healthy for r in results):
            return SystemHealthState.HEALTHY
        else:
            return SystemHealthState.DEGRADING

    @staticmethod
    def _compute_correlation(a: list[float], b: list[float]) -> float:
        """Compute Pearson correlation between two series."""
        if len(a) < 2 or len(b) < 2:
            return 0.0
        min_len = min(len(a), len(b))
        arr_a = np.array(a[:min_len])
        arr_b = np.array(b[:min_len])
        std_a = np.std(arr_a)
        std_b = np.std(arr_b)
        if std_a < 1e-10 or std_b < 1e-10:
            return 0.0
        corr_matrix = np.corrcoef(arr_a, arr_b)
        return float(corr_matrix[0, 1])
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_cross_layer.py -v
```

Expected: PASS — all cross-layer analysis tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/orchestrator/cross_layer.py tests/unit/test_cross_layer.py
git commit -m "feat: cross-layer correlation analysis detecting offsetting improvements"
```

---

## Task 9: Regime Classifier Decomposition

**Files:**
- Create: `src/evolve_trader/orchestrator/classifier_decomp.py`
- Create: `tests/unit/test_classifier_decomp.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_classifier_decomp.py
"""Tests for regime classifier decomposition and specialized variant derivation."""
import pytest
from evolve_trader.orchestrator.classifier_decomp import (
    ClassifierDecomposer,
    SignalContribution,
    ClassifierDiagnostic,
    SpecializedVariant,
)


def test_signal_contribution_structure():
    """SignalContribution captures a signal's role in classification."""
    contrib = SignalContribution(
        signal_name="vix_level",
        weight=0.35,
        accuracy_contribution=0.12,
        failure_contribution=0.08,
        trend="stable",
    )
    assert contrib.weight == 0.35
    assert contrib.trend == "stable"


def test_classifier_diagnostic_structure():
    """ClassifierDiagnostic captures overall classifier health analysis."""
    diag = ClassifierDiagnostic(
        overall_accuracy=0.72,
        accuracy_trend="declining",
        signal_contributions=[
            SignalContribution("vix", 0.3, 0.10, 0.05, "stable"),
            SignalContribution("yield_curve", 0.2, 0.08, 0.12, "declining"),
        ],
        failure_modes=["Misclassifies during regime transitions"],
        recommended_variants=[],
    )
    assert diag.overall_accuracy == 0.72
    assert len(diag.signal_contributions) == 2


def test_specialized_variant_structure():
    """SpecializedVariant defines a proposed specialized classifier."""
    variant = SpecializedVariant(
        name="volatility_regime_classifier",
        focus_dimension="volatility",
        primary_signals=["vix_level", "realized_vol", "vol_term_structure"],
        rationale="Current classifier weak on vol regime transitions — specialize",
        expected_improvement=0.15,
    )
    assert variant.focus_dimension == "volatility"
    assert len(variant.primary_signals) == 3


def test_decomposer_analyzes_signal_contributions():
    """Decomposer analyzes which signals the classifier relies on."""
    decomposer = ClassifierDecomposer()

    classifier_state = {
        "accuracy": 0.72,
        "accuracy_history": [0.80, 0.78, 0.75, 0.72],
        "signal_weights": {
            "vix_level": 0.30,
            "yield_curve": 0.25,
            "credit_spread": 0.20,
            "momentum_breadth": 0.15,
            "volume_profile": 0.10,
        },
        "signal_accuracies": {
            "vix_level": 0.80,
            "yield_curve": 0.60,
            "credit_spread": 0.75,
            "momentum_breadth": 0.70,
            "volume_profile": 0.65,
        },
        "recent_misclassifications": [
            {"actual": "risk-off", "predicted": "risk-on", "signals": ["yield_curve", "volume_profile"]},
            {"actual": "risk-off", "predicted": "risk-on", "signals": ["yield_curve"]},
        ],
    }

    diagnostic = decomposer.diagnose(classifier_state)

    assert isinstance(diagnostic, ClassifierDiagnostic)
    assert diagnostic.overall_accuracy == 0.72
    assert diagnostic.accuracy_trend == "declining"
    assert len(diagnostic.signal_contributions) == 5


def test_decomposer_identifies_failing_signals():
    """Decomposer identifies signals contributing most to failures."""
    decomposer = ClassifierDecomposer()

    classifier_state = {
        "accuracy": 0.55,
        "accuracy_history": [0.75, 0.70, 0.65, 0.60, 0.55],
        "signal_weights": {
            "vix_level": 0.40,
            "yield_curve": 0.35,
            "credit_spread": 0.25,
        },
        "signal_accuracies": {
            "vix_level": 0.80,
            "yield_curve": 0.40,
            "credit_spread": 0.45,
        },
        "recent_misclassifications": [
            {"actual": "risk-off", "predicted": "risk-on", "signals": ["yield_curve", "credit_spread"]},
            {"actual": "crisis", "predicted": "risk-off", "signals": ["yield_curve"]},
            {"actual": "risk-on", "predicted": "risk-off", "signals": ["credit_spread"]},
        ],
    }

    diagnostic = decomposer.diagnose(classifier_state)

    # yield_curve should have highest failure contribution
    yc = [s for s in diagnostic.signal_contributions if s.signal_name == "yield_curve"]
    assert len(yc) == 1
    assert yc[0].failure_contribution > 0


def test_decomposer_recommends_specialized_variants_when_failing():
    """When classifier is failing, decomposer recommends specialized variants."""
    decomposer = ClassifierDecomposer(accuracy_threshold=0.65)

    classifier_state = {
        "accuracy": 0.55,
        "accuracy_history": [0.70, 0.65, 0.60, 0.55],
        "signal_weights": {
            "vix_level": 0.30,
            "realized_vol": 0.20,
            "yield_curve": 0.25,
            "credit_spread": 0.25,
        },
        "signal_accuracies": {
            "vix_level": 0.80,
            "realized_vol": 0.75,
            "yield_curve": 0.40,
            "credit_spread": 0.45,
        },
        "recent_misclassifications": [
            {"actual": "crisis", "predicted": "risk-off", "signals": ["yield_curve", "credit_spread"]},
        ],
    }

    diagnostic = decomposer.diagnose(classifier_state)

    assert len(diagnostic.recommended_variants) >= 1
    # Should group related signals into a specialized variant
    variant_signals = set()
    for v in diagnostic.recommended_variants:
        variant_signals.update(v.primary_signals)
    # High-performing signals should be in recommended variants
    assert "vix_level" in variant_signals or "realized_vol" in variant_signals


def test_decomposer_no_variants_when_healthy():
    """Healthy classifier (above threshold) gets no variant recommendations."""
    decomposer = ClassifierDecomposer(accuracy_threshold=0.65)

    classifier_state = {
        "accuracy": 0.82,
        "accuracy_history": [0.78, 0.80, 0.81, 0.82],
        "signal_weights": {"vix_level": 0.5, "yield_curve": 0.5},
        "signal_accuracies": {"vix_level": 0.85, "yield_curve": 0.80},
        "recent_misclassifications": [],
    }

    diagnostic = decomposer.diagnose(classifier_state)
    assert len(diagnostic.recommended_variants) == 0
    assert diagnostic.accuracy_trend == "improving"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_classifier_decomp.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.orchestrator.classifier_decomp'`

**Step 3: Implement regime classifier decomposition**

```python
# src/evolve_trader/orchestrator/classifier_decomp.py
"""Regime classifier decomposition — monitor signal usage and derive specialized variants."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SignalContribution:
    """A signal's contribution to classifier performance."""

    signal_name: str
    weight: float
    accuracy_contribution: float
    failure_contribution: float
    trend: str  # "improving", "stable", "declining"


@dataclass
class SpecializedVariant:
    """A proposed specialized classifier variant."""

    name: str
    focus_dimension: str
    primary_signals: list[str]
    rationale: str
    expected_improvement: float


@dataclass
class ClassifierDiagnostic:
    """Full diagnostic of classifier health with decomposed signal analysis."""

    overall_accuracy: float
    accuracy_trend: str
    signal_contributions: list[SignalContribution]
    failure_modes: list[str]
    recommended_variants: list[SpecializedVariant]


class ClassifierDecomposer:
    """Monitors which signals the regime classifier uses, identifies failure modes,
    and derives specialized variants when accuracy drops.

    When the classifier is failing, groups high-performing signals into
    specialized variants focused on specific regime dimensions (volatility,
    credit, momentum, etc.).
    """

    def __init__(
        self,
        accuracy_threshold: float = 0.65,
        signal_accuracy_threshold: float = 0.60,
    ):
        self.accuracy_threshold = accuracy_threshold
        self.signal_accuracy_threshold = signal_accuracy_threshold

    def diagnose(self, classifier_state: dict[str, Any]) -> ClassifierDiagnostic:
        """Produce a full diagnostic of classifier health.

        Args:
            classifier_state: Dict with keys:
                - accuracy: float, current accuracy
                - accuracy_history: list[float], recent accuracy values
                - signal_weights: dict[str, float], weight per signal
                - signal_accuracies: dict[str, float], per-signal accuracy
                - recent_misclassifications: list[dict], failure details
        """
        accuracy = classifier_state["accuracy"]
        accuracy_history = classifier_state["accuracy_history"]
        signal_weights = classifier_state["signal_weights"]
        signal_accuracies = classifier_state["signal_accuracies"]
        misclassifications = classifier_state.get("recent_misclassifications", [])

        trend = self._compute_trend(accuracy_history)
        failure_counts = self._count_failures(misclassifications)
        total_failures = sum(failure_counts.values()) if failure_counts else 1

        contributions = []
        for signal_name, weight in signal_weights.items():
            sig_accuracy = signal_accuracies.get(signal_name, 0.5)
            accuracy_contrib = weight * sig_accuracy
            failure_contrib = failure_counts.get(signal_name, 0) / max(total_failures, 1)

            contributions.append(
                SignalContribution(
                    signal_name=signal_name,
                    weight=weight,
                    accuracy_contribution=round(accuracy_contrib, 4),
                    failure_contribution=round(failure_contrib, 4),
                    trend="stable",  # Could be extended with per-signal history
                )
            )

        failure_modes = self._identify_failure_modes(misclassifications)

        variants = []
        if accuracy < self.accuracy_threshold:
            variants = self._derive_specialized_variants(
                signal_weights, signal_accuracies, failure_counts
            )

        return ClassifierDiagnostic(
            overall_accuracy=accuracy,
            accuracy_trend=trend,
            signal_contributions=contributions,
            failure_modes=failure_modes,
            recommended_variants=variants,
        )

    def _compute_trend(self, history: list[float]) -> str:
        """Determine accuracy trend from history."""
        if len(history) < 2:
            return "stable"
        recent = history[-3:] if len(history) >= 3 else history
        if all(recent[i] >= recent[i - 1] for i in range(1, len(recent))):
            return "improving"
        if all(recent[i] <= recent[i - 1] for i in range(1, len(recent))):
            return "declining"
        return "stable"

    def _count_failures(
        self, misclassifications: list[dict[str, Any]]
    ) -> Counter[str]:
        """Count how often each signal appears in misclassifications."""
        counter: Counter[str] = Counter()
        for mis in misclassifications:
            for signal in mis.get("signals", []):
                counter[signal] += 1
        return counter

    def _identify_failure_modes(
        self, misclassifications: list[dict[str, Any]]
    ) -> list[str]:
        """Identify common failure mode descriptions."""
        modes = []
        transition_failures = [
            m for m in misclassifications
            if m.get("actual") != m.get("predicted")
        ]
        if transition_failures:
            actual_predicted = set()
            for m in transition_failures:
                actual_predicted.add(f"{m.get('predicted')} instead of {m.get('actual')}")
            for ap in actual_predicted:
                modes.append(f"Misclassified: predicted {ap}")
        return modes

    def _derive_specialized_variants(
        self,
        signal_weights: dict[str, float],
        signal_accuracies: dict[str, float],
        failure_counts: Counter[str],
    ) -> list[SpecializedVariant]:
        """Derive specialized classifier variants from signal analysis.

        Groups high-performing signals into focused variants.
        """
        # Separate good and bad signals
        good_signals = [
            name
            for name, acc in signal_accuracies.items()
            if acc >= self.signal_accuracy_threshold
        ]
        bad_signals = [
            name
            for name, acc in signal_accuracies.items()
            if acc < self.signal_accuracy_threshold
        ]

        variants = []
        if good_signals:
            # Group good signals into a specialized variant
            focus = self._infer_dimension(good_signals)
            avg_acc = sum(signal_accuracies[s] for s in good_signals) / len(good_signals)
            variants.append(
                SpecializedVariant(
                    name=f"{focus}_regime_classifier",
                    focus_dimension=focus,
                    primary_signals=good_signals,
                    rationale=(
                        f"Current classifier failing (below {self.accuracy_threshold:.0%}). "
                        f"Signals {good_signals} performing well ({avg_acc:.0%} avg accuracy) — "
                        f"specialize into {focus}-focused variant"
                    ),
                    expected_improvement=round(avg_acc - 0.5, 2),
                )
            )

        return variants

    @staticmethod
    def _infer_dimension(signal_names: list[str]) -> str:
        """Infer the regime dimension from signal names."""
        name_str = " ".join(signal_names).lower()
        if any(kw in name_str for kw in ["vix", "vol", "realized"]):
            return "volatility"
        if any(kw in name_str for kw in ["yield", "rate", "bond"]):
            return "rates"
        if any(kw in name_str for kw in ["credit", "spread"]):
            return "credit"
        if any(kw in name_str for kw in ["momentum", "breadth", "trend"]):
            return "momentum"
        return "multi_signal"
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_classifier_decomp.py -v
```

Expected: PASS — all classifier decomposition tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/orchestrator/classifier_decomp.py tests/unit/test_classifier_decomp.py
git commit -m "feat: regime classifier decomposition with signal analysis and specialized variant derivation"
```

---

## Task 10: Orchestrator Adjustment Log

**Files:**
- Create: `src/evolve_trader/orchestrator/adjustment_log.py`
- Create: `tests/unit/test_adjustment_log.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_adjustment_log.py
"""Tests for the orchestrator adjustment log."""
import pytest
from datetime import datetime, timezone
from evolve_trader.orchestrator.adjustment_log import (
    AdjustmentLog,
    AdjustmentEntry,
    AdjustmentStatus,
)


def test_adjustment_status_enum():
    """AdjustmentStatus covers all decision states."""
    assert AdjustmentStatus.PROPOSED.value == "proposed"
    assert AdjustmentStatus.VALIDATED.value == "validated"
    assert AdjustmentStatus.APPLIED.value == "applied"
    assert AdjustmentStatus.DEFERRED.value == "deferred"
    assert AdjustmentStatus.REJECTED.value == "rejected"


def test_adjustment_entry_structure():
    """AdjustmentEntry captures a full decision record."""
    entry = AdjustmentEntry(
        cycle_number=5,
        timestamp=datetime(2026, 3, 15, tzinfo=timezone.utc),
        layer="evolution_pace",
        target="momentum",
        action="slow",
        magnitude=0.3,
        reasoning="FIX->un-FIX->re-FIX cycle detected on stop_loss parameter",
        counterfactual_pnl_delta=-20.0,
        counterfactual_drawdown_delta=0.02,
        status=AdjustmentStatus.APPLIED,
        deferred_reason=None,
    )
    assert entry.cycle_number == 5
    assert entry.status == AdjustmentStatus.APPLIED
    assert entry.counterfactual_drawdown_delta == 0.02


def test_adjustment_log_creation():
    """AdjustmentLog initializes empty."""
    log = AdjustmentLog()
    assert len(log) == 0


def test_adjustment_log_append():
    """AdjustmentLog.append adds an entry."""
    log = AdjustmentLog()

    entry = AdjustmentEntry(
        cycle_number=1,
        timestamp=datetime.now(timezone.utc),
        layer="threshold",
        target="drawdown_monitor",
        action="tighten",
        magnitude=0.02,
        reasoning="Approaching constraint limit",
        counterfactual_pnl_delta=None,
        counterfactual_drawdown_delta=None,
        status=AdjustmentStatus.PROPOSED,
    )
    log.append(entry)
    assert len(log) == 1


def test_adjustment_log_query_by_cycle():
    """AdjustmentLog supports querying by cycle number."""
    log = AdjustmentLog()

    for i in range(1, 4):
        log.append(
            AdjustmentEntry(
                cycle_number=i,
                timestamp=datetime.now(timezone.utc),
                layer="pace",
                target=f"layer_{i}",
                action="slow",
                magnitude=0.1,
                reasoning=f"Reason {i}",
                status=AdjustmentStatus.APPLIED,
            )
        )

    cycle_2 = log.get_by_cycle(2)
    assert len(cycle_2) == 1
    assert cycle_2[0].target == "layer_2"


def test_adjustment_log_query_by_status():
    """AdjustmentLog supports querying by status."""
    log = AdjustmentLog()

    log.append(AdjustmentEntry(
        cycle_number=1, timestamp=datetime.now(timezone.utc),
        layer="pace", target="a", action="slow", magnitude=0.1,
        reasoning="r1", status=AdjustmentStatus.APPLIED,
    ))
    log.append(AdjustmentEntry(
        cycle_number=1, timestamp=datetime.now(timezone.utc),
        layer="threshold", target="b", action="tighten", magnitude=0.05,
        reasoning="r2", status=AdjustmentStatus.DEFERRED, deferred_reason="Low confidence",
    ))
    log.append(AdjustmentEntry(
        cycle_number=2, timestamp=datetime.now(timezone.utc),
        layer="discovery", target="c", action="accelerate", magnitude=0.2,
        reasoning="r3", status=AdjustmentStatus.APPLIED,
    ))

    applied = log.get_by_status(AdjustmentStatus.APPLIED)
    assert len(applied) == 2

    deferred = log.get_by_status(AdjustmentStatus.DEFERRED)
    assert len(deferred) == 1
    assert deferred[0].deferred_reason == "Low confidence"


def test_adjustment_log_query_by_layer():
    """AdjustmentLog supports querying by layer."""
    log = AdjustmentLog()

    log.append(AdjustmentEntry(
        cycle_number=1, timestamp=datetime.now(timezone.utc),
        layer="pace", target="a", action="slow", magnitude=0.1,
        reasoning="r1", status=AdjustmentStatus.APPLIED,
    ))
    log.append(AdjustmentEntry(
        cycle_number=2, timestamp=datetime.now(timezone.utc),
        layer="pace", target="b", action="slow", magnitude=0.2,
        reasoning="r2", status=AdjustmentStatus.APPLIED,
    ))
    log.append(AdjustmentEntry(
        cycle_number=3, timestamp=datetime.now(timezone.utc),
        layer="threshold", target="c", action="tighten", magnitude=0.05,
        reasoning="r3", status=AdjustmentStatus.APPLIED,
    ))

    pace_entries = log.get_by_layer("pace")
    assert len(pace_entries) == 2


def test_adjustment_log_to_records():
    """AdjustmentLog serializes all entries to list of dicts."""
    log = AdjustmentLog()

    log.append(AdjustmentEntry(
        cycle_number=1, timestamp=datetime(2026, 3, 15, tzinfo=timezone.utc),
        layer="pace", target="a", action="slow", magnitude=0.1,
        reasoning="r1", status=AdjustmentStatus.APPLIED,
    ))

    records = log.to_records()
    assert len(records) == 1
    assert records[0]["layer"] == "pace"
    assert records[0]["status"] == "applied"
    assert "timestamp" in records[0]


def test_adjustment_log_latest_n():
    """AdjustmentLog.latest(n) returns the most recent n entries."""
    log = AdjustmentLog()

    for i in range(10):
        log.append(AdjustmentEntry(
            cycle_number=i, timestamp=datetime.now(timezone.utc),
            layer="pace", target=f"t{i}", action="slow", magnitude=0.1,
            reasoning=f"r{i}", status=AdjustmentStatus.APPLIED,
        ))

    latest = log.latest(3)
    assert len(latest) == 3
    assert latest[0].cycle_number == 9  # Most recent first
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_adjustment_log.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.orchestrator.adjustment_log'`

**Step 3: Implement the adjustment log**

```python
# src/evolve_trader/orchestrator/adjustment_log.py
"""Orchestrator adjustment log — full audit trail of all decisions."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any


class AdjustmentStatus(Enum):
    """Status of an orchestrator adjustment decision."""

    PROPOSED = "proposed"
    VALIDATED = "validated"
    APPLIED = "applied"
    DEFERRED = "deferred"
    REJECTED = "rejected"


@dataclass
class AdjustmentEntry:
    """A single orchestrator decision record with full reasoning chain."""

    cycle_number: int
    timestamp: datetime
    layer: str
    target: str
    action: str
    magnitude: float
    reasoning: str
    status: AdjustmentStatus
    counterfactual_pnl_delta: float | None = None
    counterfactual_drawdown_delta: float | None = None
    deferred_reason: str | None = None


class AdjustmentLog:
    """Append-only log of all orchestrator decisions.

    Every proposed change, whether applied, deferred, or rejected,
    is recorded with full reasoning, counterfactual results, and status.
    Supports querying by cycle, status, and layer for analysis.
    """

    def __init__(self):
        self._entries: list[AdjustmentEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    def append(self, entry: AdjustmentEntry) -> None:
        """Add an entry to the log."""
        self._entries.append(entry)

    def get_by_cycle(self, cycle_number: int) -> list[AdjustmentEntry]:
        """Get all entries for a specific cycle."""
        return [e for e in self._entries if e.cycle_number == cycle_number]

    def get_by_status(self, status: AdjustmentStatus) -> list[AdjustmentEntry]:
        """Get all entries with a specific status."""
        return [e for e in self._entries if e.status == status]

    def get_by_layer(self, layer: str) -> list[AdjustmentEntry]:
        """Get all entries targeting a specific layer."""
        return [e for e in self._entries if e.layer == layer]

    def latest(self, n: int = 10) -> list[AdjustmentEntry]:
        """Get the most recent n entries, newest first."""
        return list(reversed(self._entries[-n:]))

    def to_records(self) -> list[dict[str, Any]]:
        """Serialize all entries to list of dicts."""
        records = []
        for entry in self._entries:
            d = asdict(entry)
            d["status"] = entry.status.value
            d["timestamp"] = entry.timestamp.isoformat()
            records.append(d)
        return records
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_adjustment_log.py -v
```

Expected: PASS — all adjustment log tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/orchestrator/adjustment_log.py tests/unit/test_adjustment_log.py
git commit -m "feat: orchestrator adjustment log with full audit trail of all decisions"
```

---

## Task 11: Integration Testing & Final Verification

**Files:**
- Create: `tests/integration/test_orchestrator_pipeline.py`

**Step 1: Write the integration tests**

```python
# tests/integration/test_orchestrator_pipeline.py
"""Integration tests for the full orchestrator pipeline."""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from evolve_trader.orchestrator.agent import OrchestratorAgent, OrchestratorResult
from evolve_trader.orchestrator.config import OrchestratorConfig
from evolve_trader.orchestrator.metrics_aggregator import MetricsAggregator, AggregatedReport
from evolve_trader.orchestrator.pace_control import PaceController, EvolutionPattern
from evolve_trader.orchestrator.tension_detector import TensionDetector, TensionSeverity
from evolve_trader.orchestrator.counterfactual import (
    CounterfactualEngine,
    TradeRecord,
    ProposedChange,
)
from evolve_trader.orchestrator.threshold_calibrator import (
    ThresholdCalibrator,
    CalibrationContext,
)
from evolve_trader.orchestrator.discovery_tuner import DiscoveryTuner, RosterHealth
from evolve_trader.orchestrator.cross_layer import (
    CrossLayerAnalyzer,
    LayerPerformanceSeries,
    SystemHealthState,
)
from evolve_trader.orchestrator.classifier_decomp import ClassifierDecomposer
from evolve_trader.orchestrator.adjustment_log import (
    AdjustmentLog,
    AdjustmentEntry,
    AdjustmentStatus,
)


def test_full_orchestrator_components_initialize():
    """All orchestrator components can be instantiated together."""
    config = OrchestratorConfig(cadence_days=7, dry_run=True)
    agent = OrchestratorAgent(config=config)
    aggregator = MetricsAggregator()
    pace = PaceController()
    tension = TensionDetector()
    counterfactual = CounterfactualEngine()
    calibrator = ThresholdCalibrator()
    tuner = DiscoveryTuner()
    cross_layer = CrossLayerAnalyzer()
    decomposer = ClassifierDecomposer()
    log = AdjustmentLog()

    assert agent is not None
    assert aggregator is not None
    assert pace is not None
    assert tension is not None
    assert counterfactual is not None
    assert calibrator is not None
    assert tuner is not None
    assert cross_layer is not None
    assert decomposer is not None
    assert log is not None


def test_pace_to_tension_pipeline():
    """Pace assessments feed into tension detection context."""
    pace = PaceController()

    oscillating_events = [
        {"type": "FIX", "layer": "momentum", "param": "stop_loss", "value": 0.02,
         "timestamp": datetime(2026, 3, 1, tzinfo=timezone.utc)},
        {"type": "FIX", "layer": "momentum", "param": "stop_loss", "value": 0.05,
         "timestamp": datetime(2026, 3, 3, tzinfo=timezone.utc)},
        {"type": "FIX", "layer": "momentum", "param": "stop_loss", "value": 0.02,
         "timestamp": datetime(2026, 3, 5, tzinfo=timezone.utc)},
    ]

    assessment = pace.assess_layer("momentum", oscillating_events)
    assert assessment.pattern == EvolutionPattern.OSCILLATING

    # Feed pace assessment into a broader layer state for tension detection
    tension = TensionDetector()
    layer_states = {
        "strategy_evolution": {
            "active_strategies": ["momentum"],
            "avg_trade_frequency": 20,
            "direction_bias": "long",
            "pace_assessment": assessment.pattern.value,
        },
        "regime_classifier": {
            "regime_switches": 7,
            "avg_switch_interval_days": 2.0,
            "current_regime": "risk-on",
        },
    }

    tensions = tension.detect(layer_states)
    assert len(tensions) >= 1


def test_counterfactual_validates_calibration():
    """Counterfactual engine validates threshold calibration proposals."""
    calibrator = ThresholdCalibrator()
    counterfactual = CounterfactualEngine()

    ctx = CalibrationContext(
        portfolio_return=-0.03,
        max_drawdown=0.08,
        sharpe_ratio=0.2,
        period_days=14,
        constraint_proximity={"max_drawdown": 0.65},
        recent_violations=1,
    )

    adjustments = calibrator.calibrate({"max_drawdown": 0.10}, ctx)
    assert len(adjustments) >= 1

    # Convert calibration adjustment to ProposedChange for replay
    adj = adjustments[0]
    change = ProposedChange(
        layer="threshold",
        target=adj.metric_name,
        parameter="threshold",
        current_value=adj.current_threshold,
        proposed_value=adj.proposed_threshold,
    )

    trades = [
        TradeRecord(
            timestamp=datetime(2026, 3, i, tzinfo=timezone.utc),
            ticker="AAPL", direction="BUY", quantity=10,
            entry_price=180.0, exit_price=180.0 + (-3 if i % 3 == 0 else 2),
            pnl=(-3 if i % 3 == 0 else 2) * 10,
            strategy="momentum", regime_label="risk-off",
        )
        for i in range(1, 8)
    ]

    result = counterfactual.replay(trades=trades, change=change)
    assert result.original_trade_count == 7


def test_discovery_tuner_with_cross_layer_health():
    """Cross-layer health informs discovery tuning decisions."""
    cross_layer = CrossLayerAnalyzer()
    tuner = DiscoveryTuner()

    series = [
        LayerPerformanceSeries("strategy_perf", ["t1", "t2", "t3", "t4", "t5"],
                               [0.05, 0.04, 0.03, 0.02, 0.01]),
        LayerPerformanceSeries("signal_quality", ["t1", "t2", "t3", "t4", "t5"],
                               [0.01, 0.02, 0.03, 0.04, 0.05]),
    ]

    health = cross_layer.assess_system_health(series)
    assert health in (SystemHealthState.OFFSETTING, SystemHealthState.UNSTABLE)

    # During unhealthy state, roster is also degrading
    roster = RosterHealth(
        active_count=6, avg_sharpe=0.4, avg_return=-0.01,
        underperformers=3, total_candidates_in_trial=1,
        promotion_rate=0.05, recent_promotions=0, recent_demotions=2,
    )

    result = tuner.tune(roster, current_aggressiveness=0.3)
    assert result.recommended_aggressiveness > 0.3  # Should accelerate


def test_full_cycle_logged():
    """Full orchestrator cycle results are captured in adjustment log."""
    log = AdjustmentLog()

    # Simulate a cycle producing adjustments
    entries = [
        AdjustmentEntry(
            cycle_number=1,
            timestamp=datetime.now(timezone.utc),
            layer="evolution_pace",
            target="momentum",
            action="slow",
            magnitude=0.3,
            reasoning="Oscillation detected in stop_loss parameter",
            counterfactual_pnl_delta=-15.0,
            counterfactual_drawdown_delta=0.02,
            status=AdjustmentStatus.APPLIED,
        ),
        AdjustmentEntry(
            cycle_number=1,
            timestamp=datetime.now(timezone.utc),
            layer="threshold",
            target="drawdown_monitor",
            action="tighten",
            magnitude=0.02,
            reasoning="Approaching constraint limit at 75% proximity",
            counterfactual_pnl_delta=-40.0,
            counterfactual_drawdown_delta=0.03,
            status=AdjustmentStatus.DEFERRED,
            deferred_reason="PnL impact too high — defer to next cycle for re-evaluation",
        ),
    ]

    for entry in entries:
        log.append(entry)

    assert len(log) == 2
    assert len(log.get_by_status(AdjustmentStatus.APPLIED)) == 1
    assert len(log.get_by_status(AdjustmentStatus.DEFERRED)) == 1

    records = log.to_records()
    assert all("reasoning" in r for r in records)
    assert all("counterfactual_pnl_delta" in r for r in records)


@pytest.mark.asyncio
async def test_agent_end_to_end_dry_run():
    """End-to-end: agent ingests metrics, calls LLM, produces logged result."""
    config = OrchestratorConfig(cadence_days=7, dry_run=True, max_adjustments_per_cycle=3)
    agent = OrchestratorAgent(config=config)
    log = AdjustmentLog()

    mock_metrics = MagicMock()
    mock_metrics.aggregate.return_value = {
        "period_start": datetime(2026, 3, 1, tzinfo=timezone.utc),
        "period_end": datetime(2026, 3, 15, tzinfo=timezone.utc),
        "portfolio_return": -0.02,
        "max_drawdown": 0.09,
        "sharpe_ratio": 0.3,
        "evolution_events": 8,
        "regime_label": "risk-off",
        "constraint_proximity": {"max_drawdown": 0.7},
    }

    with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "reasoning": "System under stress. Momentum oscillating, drawdown approaching limit.",
            "proposed_adjustments": [
                {"layer": "pace", "target": "momentum", "action": "slow",
                 "magnitude": 0.3, "rationale": "Oscillation detected"},
                {"layer": "threshold", "target": "drawdown", "action": "tighten",
                 "magnitude": 0.02, "rationale": "Approaching limit"},
            ],
        }
        result = await agent.run(metrics=mock_metrics)

    assert isinstance(result, OrchestratorResult)
    assert result.applied is False  # dry_run
    assert len(result.proposed_adjustments) == 2

    # Log the results
    for adj in result.proposed_adjustments:
        log.append(AdjustmentEntry(
            cycle_number=result.cycle_number,
            timestamp=result.timestamp,
            layer=adj.layer,
            target=adj.target,
            action=adj.action,
            magnitude=adj.magnitude,
            reasoning=adj.rationale,
            status=AdjustmentStatus.PROPOSED,
        ))

    assert len(log) == 2
    assert all(e.status == AdjustmentStatus.PROPOSED for e in log.latest(2))
```

**Step 2: Run the integration tests**

```bash
pytest tests/integration/test_orchestrator_pipeline.py -v
```

Expected: PASS (all components already implemented)

**Step 3: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: ALL PASS — all prior phases and Phase 7 tests green

**Step 4: Run linting and type checking**

```bash
ruff check src/evolve_trader/orchestrator/
mypy src/evolve_trader/orchestrator/ --ignore-missing-imports
```

Expected: No errors

**Step 5: Commit**

```bash
git add tests/integration/test_orchestrator_pipeline.py
git commit -m "test: integration tests for full orchestrator pipeline"
```

**Step 6: Final commit**

```bash
git add -A
git commit -m "test: Phase 7 final verification — all orchestrator tests passing"
```

---

## Parallelization Notes

Tasks in this phase have the following dependency structure:

```
Task 1 (Agent Core + Config) ────────┐
Task 2 (Metrics Aggregator) ──────────┤
                                       ├── Task 11 (Integration Tests)
Task 3 (Pace Control) ────────────────┤
Task 4 (Tension Detection) ───────────┤
Task 5 (Counterfactual Engine) ───────┤
Task 6 (Threshold Calibration) ───────┤
Task 7 (Discovery Tuner) ─────────────┤
Task 8 (Cross-Layer Analysis) ────────┤
Task 9 (Classifier Decomposition) ────┤
Task 10 (Adjustment Log) ─────────────┘
```

**Can run in parallel:**
- Tasks 2-10 are independent of each other — all can run simultaneously after Task 1
- Task 1 (Agent Core) must come first as it defines `OrchestratorConfig` used by integration tests
- Task 11 (Integration) depends on all other tasks being complete
- Within each task, the TDD cycle is strictly sequential: test → fail → implement → pass → commit

**Target:** ~1200-1800 lines across all source files and test files combined.
