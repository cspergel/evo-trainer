"""Version DAG for tracking skill evolution lineage.

Records parent->child relationships through FIX/DERIVED/CAPTURED events.
Stores market conditions and performance metrics that triggered evolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class EvolutionMode(Enum):
    """How a skill was created or modified."""

    SEED = "seed"
    FIX = "fix"
    DERIVED = "derived"
    CAPTURED = "captured"


@dataclass
class EvolutionEvent:
    """A single evolution event in the DAG."""

    parent: str | None
    child: str
    mode: EvolutionMode
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metrics: dict[str, float] = field(default_factory=dict)


class VersionDAG:
    """Directed acyclic graph tracking skill evolution lineage."""

    def __init__(self) -> None:
        self._parents: dict[str, str | None] = {}
        self._children: dict[str, list[str]] = {}
        self._events: dict[str, list[EvolutionEvent]] = {}

    def add_root(self, skill_name: str) -> None:
        """Add a seed strategy as a root node."""
        self._parents[skill_name] = None
        self._children.setdefault(skill_name, [])
        self._events.setdefault(skill_name, []).append(
            EvolutionEvent(
                parent=None,
                child=skill_name,
                mode=EvolutionMode.SEED,
                reason="Initial seed strategy",
            )
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
            EvolutionEvent(
                parent=parent,
                child=child,
                mode=mode,
                reason=reason,
                metrics=metrics or {},
            )
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
            EvolutionEvent(
                parent=None,
                child=skill_name,
                mode=EvolutionMode.CAPTURED,
                reason=reason,
                metrics=metrics or {},
            )
        )

    def get_parent(self, skill_name: str) -> str | None:
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
        current: str | None = skill_name
        while current is not None:
            lineage.append(current)
            current = self._parents.get(current)
        lineage.reverse()
        return lineage
