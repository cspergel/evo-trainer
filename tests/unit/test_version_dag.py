"""Tests for the version DAG tracking skill lineage."""

from evolve_trader.core.version_dag import (
    EvolutionMode,
    VersionDAG,
)


def test_add_root_skill():
    """A seed strategy is a root node with no parent."""
    dag = VersionDAG()
    dag.add_root("momentum-v1")
    assert dag.get_parent("momentum-v1") is None
    assert dag.get_lineage("momentum-v1") == ["momentum-v1"]


def test_fix_creates_child():
    """FIX creates a new version linked to the parent."""
    dag = VersionDAG()
    dag.add_root("momentum-v1")
    dag.add_evolution(
        parent="momentum-v1",
        child="momentum-v1.1",
        mode=EvolutionMode.FIX,
        reason="Added stop-loss after excessive drawdown",
        metrics={"sharpe_before": 0.4, "max_drawdown_before": 0.20},
    )
    assert dag.get_parent("momentum-v1.1") == "momentum-v1"
    assert dag.get_lineage("momentum-v1.1") == ["momentum-v1", "momentum-v1.1"]


def test_derived_creates_branch():
    """DERIVED creates a specialized branch from parent."""
    dag = VersionDAG()
    dag.add_root("mean-reversion-v1")
    dag.add_evolution(
        parent="mean-reversion-v1",
        child="mean-reversion-earnings-v1",
        mode=EvolutionMode.DERIVED,
        reason="Specialized for post-earnings mean reversion",
    )
    assert dag.get_parent("mean-reversion-earnings-v1") == "mean-reversion-v1"
    children = dag.get_children("mean-reversion-v1")
    assert "mean-reversion-earnings-v1" in children


def test_captured_has_no_parent():
    """CAPTURED creates a new root from emergent behavior."""
    dag = VersionDAG()
    dag.add_captured(
        skill_name="emergent-sector-rotation-v1",
        reason="Agent discovered sector rotation pattern not in any existing skill",
        metrics={"sharpe": 1.5, "trades": 12},
    )
    assert dag.get_parent("emergent-sector-rotation-v1") is None
    events = dag.get_events("emergent-sector-rotation-v1")
    assert len(events) == 1
    assert events[0].mode == EvolutionMode.CAPTURED


def test_full_lineage_trace():
    """Can trace full lineage through multiple evolution steps."""
    dag = VersionDAG()
    dag.add_root("momentum-v1")
    dag.add_evolution(
        parent="momentum-v1", child="momentum-v2", mode=EvolutionMode.FIX, reason="fix1"
    )
    dag.add_evolution(
        parent="momentum-v2", child="momentum-v3", mode=EvolutionMode.FIX, reason="fix2"
    )
    dag.add_evolution(
        parent="momentum-v3",
        child="momentum-tech-v1",
        mode=EvolutionMode.DERIVED,
        reason="tech specialization",
    )

    lineage = dag.get_lineage("momentum-tech-v1")
    assert lineage == ["momentum-v1", "momentum-v2", "momentum-v3", "momentum-tech-v1"]


def test_get_children_empty():
    """A leaf node has no children."""
    dag = VersionDAG()
    dag.add_root("leaf-skill")
    assert dag.get_children("leaf-skill") == []


def test_get_events_unknown_skill():
    """Unknown skill returns empty events list."""
    dag = VersionDAG()
    assert dag.get_events("nonexistent") == []
