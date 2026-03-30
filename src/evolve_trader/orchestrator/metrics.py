"""Cross-layer metrics aggregation.

Collects metrics from all system layers into a unified snapshot
for the orchestrator to reason about.

Per profitability contract: orchestrator is advisory-only until
it proves causal value. All adjustments are logged, not auto-applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class StrategyMetrics:
    """Performance metrics for a single strategy."""

    name: str
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    recent_pnl: float = 0.0


@dataclass
class SignalSourceMetrics:
    """Health and performance metrics for a signal source."""

    name: str
    hit_rate: float = 0.0
    total_signals: int = 0
    lifecycle_stage: str = "unknown"
    is_healthy: bool = True


@dataclass
class SystemSnapshot:
    """Cross-layer metrics snapshot at a point in time."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Portfolio
    portfolio_value: float = 0.0
    cash: float = 0.0
    drawdown: float = 0.0
    overall_sharpe: float = 0.0

    # Strategies
    active_strategies: list[StrategyMetrics] = field(default_factory=list)
    total_evolution_events: int = 0

    # Signals
    signal_sources: list[SignalSourceMetrics] = field(default_factory=list)
    regime_label: str = "unknown"
    regime_confidence: float = 0.0

    # Costs
    llm_cost_total: float = 0.0
    llm_cost_this_month: float = 0.0

    # Promotion
    promotion_stage: str = "paper_training"
    paper_live_correlation: float | None = None

    @property
    def strategy_count(self) -> int:
        return len(self.active_strategies)

    @property
    def healthy_source_count(self) -> int:
        return sum(1 for s in self.signal_sources if s.is_healthy)
