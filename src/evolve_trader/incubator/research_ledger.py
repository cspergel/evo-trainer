"""Research ledger — tracks all incubator experiments.

Per profitability contract section 5: every candidate, mutation,
parameter set must be logged. Total experiment count is tracked
and penalizes promotion thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class ExperimentRecord:
    """A single incubator experiment."""

    candidate_id: str
    parent_id: str | None  # None for novel generation
    hypothesis: str
    sharpe_result: float | None = None
    evaluation_windows: int = 0
    status: str = "pending"  # pending, evaluated, promoted, discarded
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class ResearchLedger:
    """Tracks all incubator experiments for multiple-testing discipline.

    The more experiments run, the stricter the promotion bar becomes.
    No silent cherry-picking — every candidate is logged.
    """

    def __init__(self) -> None:
        self._experiments: list[ExperimentRecord] = []

    def log_experiment(
        self,
        candidate_id: str,
        hypothesis: str,
        parent_id: str | None = None,
    ) -> ExperimentRecord:
        """Log a new experiment before evaluation."""
        record = ExperimentRecord(
            candidate_id=candidate_id,
            parent_id=parent_id,
            hypothesis=hypothesis,
        )
        self._experiments.append(record)
        return record

    def record_result(
        self,
        record: ExperimentRecord,
        sharpe: float,
        windows: int,
        promoted: bool = False,
    ) -> None:
        """Record evaluation result for an experiment."""
        record.sharpe_result = sharpe
        record.evaluation_windows = windows
        record.status = "promoted" if promoted else "evaluated"

    def discard(self, record: ExperimentRecord) -> None:
        """Mark an experiment as discarded."""
        record.status = "discarded"

    @property
    def total_experiments(self) -> int:
        return len(self._experiments)

    @property
    def total_promoted(self) -> int:
        return sum(1 for e in self._experiments if e.status == "promoted")

    @property
    def total_discarded(self) -> int:
        return sum(1 for e in self._experiments if e.status == "discarded")

    def get_all(self) -> list[ExperimentRecord]:
        return list(self._experiments)

    def get_by_family(self, parent_id: str) -> list[ExperimentRecord]:
        """Get all experiments derived from a parent."""
        return [e for e in self._experiments if e.parent_id == parent_id]
