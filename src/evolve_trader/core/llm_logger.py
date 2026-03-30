"""Shared LLM usage logger — file-backed in Phase 1, PostgreSQL in Phase 2.

Tracks model usage, token counts, costs, and budget utilization.
No raw prompts or responses stored — compact metadata only.

Note: Phase 1 is single-process. No file locking on writes.
Phase 2 PostgreSQL backend will handle concurrency.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class LLMUsageRecord:
    """A single LLM API call record."""

    model: str
    component: str  # "strategy_execution", "evolution", "analysis", "orchestrator"
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "component": self.component,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "timestamp": self.timestamp,
        }


class LLMUsageLogger:
    """File-backed LLM usage logger with budget tracking.

    Persists to JSONL for Phase 1. Phase 2 swaps the backend to PostgreSQL
    without changing callers.
    """

    def __init__(
        self,
        log_path: Path | str = "data/llm_usage.jsonl",
        monthly_budget_usd: float = 100.0,
    ) -> None:
        self._log_path = Path(log_path)
        self._monthly_budget_usd = monthly_budget_usd
        self._records: list[LLMUsageRecord] = []
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing records from disk, skipping malformed lines."""
        if not self._log_path.exists():
            return
        for line in self._log_path.read_text().strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._records.append(
                LLMUsageRecord(
                    model=data["model"],
                    component=data["component"],
                    input_tokens=data["input_tokens"],
                    output_tokens=data["output_tokens"],
                    cost_usd=data["cost_usd"],
                    timestamp=data.get("timestamp", ""),
                )
            )

    def log(
        self,
        model: str,
        component: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> LLMUsageRecord:
        """Log an LLM API call."""
        record = LLMUsageRecord(
            model=model,
            component=component,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
        self._records.append(record)
        self._persist(record)
        return record

    def _persist(self, record: LLMUsageRecord) -> None:
        """Append a record to the JSONL file."""
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def _current_month_cost(self) -> float:
        """Sum costs for the current calendar month only."""
        prefix = datetime.now(UTC).strftime("%Y-%m")
        return sum(r.cost_usd for r in self._records if r.timestamp.startswith(prefix))

    def total_cost(self) -> float:
        """Total cost across all logged records (all time)."""
        return sum(r.cost_usd for r in self._records)

    def cost_by_component(self) -> dict[str, float]:
        """Aggregate costs by component."""
        totals: dict[str, float] = defaultdict(float)
        for r in self._records:
            totals[r.component] += r.cost_usd
        return dict(totals)

    def budget_utilization(self) -> float:
        """Current month's budget utilization as a fraction (0.0-1.0+)."""
        if self._monthly_budget_usd <= 0:
            return 0.0
        return self._current_month_cost() / self._monthly_budget_usd

    def is_budget_warning(self) -> bool:
        """True if current month's budget utilization >= 80%."""
        return self.budget_utilization() >= 0.80

    def is_budget_exceeded(self) -> bool:
        """True if current month's budget utilization >= 100%."""
        return self.budget_utilization() >= 1.0
