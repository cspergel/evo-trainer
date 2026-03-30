"""Orchestrator — advisory supervisory layer.

Ingests cross-layer metrics, proposes bounded adjustments,
validates via counterfactual replay, logs all decisions.

Per profitability contract section 7: advisory-only until
proven causal value. Does not auto-apply adjustments.
"""

from __future__ import annotations

from evolve_trader.orchestrator.adjustment_log import (
    Adjustment,
    AdjustmentLog,
    AdjustmentType,
)
from evolve_trader.orchestrator.counterfactual import (
    CounterfactualResult,
    run_counterfactual,
)
from evolve_trader.orchestrator.metrics import SystemSnapshot


class Orchestrator:
    """Cross-layer supervisory agent.

    Analyzes system metrics, proposes adjustments, validates
    them via counterfactual replay, and logs all decisions.
    Advisory-only — does not auto-apply changes.
    """

    def __init__(self) -> None:
        self.log = AdjustmentLog()
        self._snapshots: list[SystemSnapshot] = []

    def ingest_snapshot(self, snapshot: SystemSnapshot) -> None:
        """Record a system metrics snapshot."""
        self._snapshots.append(snapshot)

    def analyze_and_propose(self, snapshot: SystemSnapshot) -> list[Adjustment]:
        """Analyze metrics and propose adjustments.

        Returns proposed adjustments. Caller decides whether to apply.
        """
        self.ingest_snapshot(snapshot)
        proposals: list[Adjustment] = []

        # Check for high drawdown — propose risk tightening
        if snapshot.drawdown > 0.15:
            proposals.append(
                self.log.propose(
                    AdjustmentType.RISK_TIGHTENING,
                    rationale=(
                        f"Drawdown at {snapshot.drawdown:.1%}, approaching 20% limit. "
                        f"Propose reducing gross exposure."
                    ),
                    metrics_cited={"drawdown": snapshot.drawdown},
                )
            )

        # Check for low overall Sharpe — propose evolution pace increase
        if snapshot.overall_sharpe < 0.3 and snapshot.total_evolution_events < 5:
            proposals.append(
                self.log.propose(
                    AdjustmentType.EVOLUTION_PACE,
                    rationale=(
                        f"Overall Sharpe {snapshot.overall_sharpe:.2f} is low "
                        f"with only {snapshot.total_evolution_events} evolution events. "
                        f"Propose increasing evolution frequency."
                    ),
                    metrics_cited={
                        "overall_sharpe": snapshot.overall_sharpe,
                        "evolution_events": float(snapshot.total_evolution_events),
                    },
                )
            )

        # Check for unhealthy signal sources — propose freeze
        unhealthy = [s for s in snapshot.signal_sources if not s.is_healthy]
        if unhealthy:
            names = ", ".join(s.name for s in unhealthy)
            proposals.append(
                self.log.propose(
                    AdjustmentType.FREEZE_COMPONENT,
                    rationale=(
                        f"Unhealthy signal sources: {names}. " f"Propose freezing until recovery."
                    ),
                    metrics_cited={"unhealthy_sources": float(len(unhealthy))},
                )
            )

        # Check paper/live divergence
        if snapshot.paper_live_correlation is not None and snapshot.paper_live_correlation < 0.85:
            proposals.append(
                self.log.propose(
                    AdjustmentType.PROMOTION_HOLD,
                    rationale=(
                        f"Paper/live correlation {snapshot.paper_live_correlation:.2f} "
                        f"is declining. Hold promotion until stabilized."
                    ),
                    metrics_cited={"paper_live_correlation": snapshot.paper_live_correlation},
                )
            )

        return proposals

    def validate_adjustment(
        self,
        adjustment: Adjustment,
        baseline_sharpe: list[float],
        adjusted_sharpe: list[float],
    ) -> CounterfactualResult:
        """Validate an adjustment via counterfactual replay.

        Stores the result on the adjustment for audit trail.
        """
        result = run_counterfactual(baseline_sharpe, adjusted_sharpe)
        adjustment.counterfactual_result = {
            "baseline_sharpe": result.baseline_sharpe,
            "adjusted_sharpe": result.adjusted_sharpe,
            "improvement": result.improvement,
            "passes": result.passes_simplicity_tax,
        }

        if result.passes_simplicity_tax:
            self.log.apply(adjustment)
        else:
            self.log.defer(
                adjustment,
                f"Counterfactual improvement {result.improvement:+.3f} below threshold",
            )

        return result
