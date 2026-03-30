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
        if parent not in self._parents:
            raise ValueError(f"Parent '{parent}' not found in DAG")
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
        visited: set[str] = set()
        while current is not None:
            if current in visited:
                raise ValueError(f"Cycle detected in lineage at '{current}'")
            visited.add(current)
            lineage.append(current)
            current = self._parents.get(current)
        lineage.reverse()
        return lineage

    def to_dict(self) -> dict[str, object]:
        """Serialize the DAG to a dictionary for persistence."""
        return {
            "parents": dict(self._parents),
            "events": {
                name: [
                    {
                        "parent": e.parent,
                        "child": e.child,
                        "mode": e.mode.value,
                        "reason": e.reason,
                        "timestamp": e.timestamp.isoformat(),
                        "metrics": e.metrics,
                    }
                    for e in events
                ]
                for name, events in self._events.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> VersionDAG:
        """Deserialize a DAG from a dictionary."""
        dag = cls()
        parents = data.get("parents", {})
        events_data = data.get("events", {})

        assert isinstance(parents, dict)
        assert isinstance(events_data, dict)

        # Rebuild parents and children
        for child, parent in parents.items():
            assert isinstance(child, str)
            dag._parents[child] = parent if isinstance(parent, str) else None
            dag._children.setdefault(child, [])
            if isinstance(parent, str):
                dag._children.setdefault(parent, []).append(child)

        # Rebuild events
        for name, event_list in events_data.items():
            assert isinstance(name, str)
            assert isinstance(event_list, list)
            dag._events[name] = [
                EvolutionEvent(
                    parent=e["parent"] if isinstance(e.get("parent"), str) else None,
                    child=str(e["child"]),
                    mode=EvolutionMode(str(e["mode"])),
                    reason=str(e["reason"]),
                    metrics=dict(e.get("metrics", {})),
                )
                for e in event_list
                if isinstance(e, dict)
            ]

        return dag
