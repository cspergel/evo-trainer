"""Signal source registration and discovery.

All signal sources register through this framework.
The registry tracks source health and provides discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from evolve_trader.signals.types import SignalEvent


class SignalSource(Protocol):
    """Protocol that all signal sources must implement."""

    @property
    def name(self) -> str:
        """Unique source identifier (e.g., 'edgar_13f', 'congressional')."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of this source."""
        ...

    async def fetch_signals(self) -> list[SignalEvent]:
        """Fetch latest signals from this source."""
        ...


@dataclass
class SourceHealth:
    """Health status of a signal source."""

    last_success: datetime | None = None
    last_failure: datetime | None = None
    consecutive_failures: int = 0
    total_signals_fetched: int = 0

    @property
    def is_healthy(self) -> bool:
        return self.consecutive_failures < 5

    @property
    def status(self) -> str:
        if self.last_success is None:
            return "unknown"
        if self.consecutive_failures == 0:
            return "healthy"
        if self.consecutive_failures < 5:
            return "degraded"
        return "unhealthy"


class SourceRegistry:
    """Registry for all signal sources with health tracking."""

    def __init__(self) -> None:
        self._sources: dict[str, SignalSource] = {}
        self._health: dict[str, SourceHealth] = {}

    def register(self, source: SignalSource) -> None:
        """Register a signal source."""
        self._sources[source.name] = source
        self._health[source.name] = SourceHealth()

    def get(self, name: str) -> SignalSource | None:
        """Get a registered source by name."""
        return self._sources.get(name)

    def list_sources(self) -> list[str]:
        """List all registered source names."""
        return list(self._sources.keys())

    def get_health(self, name: str) -> SourceHealth | None:
        """Get health status for a source."""
        return self._health.get(name)

    def record_success(self, name: str, signals_count: int) -> None:
        """Record a successful fetch for a source."""
        health = self._health.get(name)
        if health:
            health.last_success = datetime.now(UTC)
            health.consecutive_failures = 0
            health.total_signals_fetched += signals_count

    def record_failure(self, name: str) -> None:
        """Record a failed fetch for a source."""
        health = self._health.get(name)
        if health:
            health.last_failure = datetime.now(UTC)
            health.consecutive_failures += 1

    def get_healthy_sources(self) -> list[str]:
        """List sources that are currently healthy."""
        return [name for name, health in self._health.items() if health.is_healthy]
