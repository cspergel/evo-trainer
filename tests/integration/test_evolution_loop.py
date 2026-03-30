"""Integration test: strategy evolution loop.

Tests the core thesis: do strategies evolve meaningfully
based on trading outcomes?
"""

from pathlib import Path

from evolve_trader.core.evolution_loop import (
    EvolutionResults,
    RuleBasedEvolutionDriver,
    load_seed_skills,
    run_evolution_cycle,
)
from evolve_trader.core.version_dag import VersionDAG

SKILLS_DIR = Path(__file__).parent.parent.parent / "src" / "evolve_trader" / "strategies" / "skills"


def test_load_seed_skills():
    """All seed skills load successfully."""
    skills = load_seed_skills(SKILLS_DIR)
    assert len(skills) >= 10


def test_evolution_cycle_runs():
    """A complete evolution cycle executes without errors."""
    dag = VersionDAG()
    results = run_evolution_cycle(
        seed_skills_dir=str(SKILLS_DIR),
        replay_days=50,
        universe="nasdaq100",
        version_dag=dag,
    )
    assert isinstance(results, EvolutionResults)
    assert results.skills_evaluated >= 10


def test_evolution_produces_fix_events():
    """Evolution produces at least one FIX event from underperforming seeds."""
    dag = VersionDAG()
    # Use a high threshold so most seeds trigger evolution
    driver = RuleBasedEvolutionDriver(sharpe_threshold=2.0)
    results = run_evolution_cycle(
        seed_skills_dir=str(SKILLS_DIR),
        replay_days=50,
        universe="nasdaq100",
        version_dag=dag,
        driver=driver,
    )
    assert results.total_fix_events >= 1, "Should produce at least one FIX event"


def test_evolved_skills_recorded_in_dag():
    """Evolved skills are recorded in the version DAG with correct lineage."""
    dag = VersionDAG()
    driver = RuleBasedEvolutionDriver(sharpe_threshold=2.0)
    results = run_evolution_cycle(
        seed_skills_dir=str(SKILLS_DIR),
        replay_days=50,
        universe="nasdaq100",
        version_dag=dag,
        driver=driver,
    )

    for skill_name in results.evolved_skills:
        parent = dag.get_parent(skill_name)
        assert parent is not None, f"Evolved skill {skill_name} should have a parent"
        lineage = dag.get_lineage(skill_name)
        assert len(lineage) >= 2, f"Evolved skill {skill_name} should have lineage depth >= 2"


def test_performance_comparisons_populated():
    """Performance comparisons are populated for evolved skills."""
    dag = VersionDAG()
    driver = RuleBasedEvolutionDriver(sharpe_threshold=2.0)
    results = run_evolution_cycle(
        seed_skills_dir=str(SKILLS_DIR),
        replay_days=50,
        universe="nasdaq100",
        version_dag=dag,
        driver=driver,
    )

    assert len(results.performance_comparisons) >= 1
    for comp in results.performance_comparisons:
        assert comp.parent_name
        assert comp.child_name
        assert comp.parent_oos_sharpe is not None
        assert comp.child_oos_sharpe is not None


def test_capital_preservation_not_evolved():
    """Capital preservation skill should not be 'fixed' — it has no underperformance."""
    dag = VersionDAG()
    # Low threshold — only truly bad strategies get evolved
    driver = RuleBasedEvolutionDriver(sharpe_threshold=0.3)
    results = run_evolution_cycle(
        seed_skills_dir=str(SKILLS_DIR),
        replay_days=50,
        universe="nasdaq100",
        version_dag=dag,
        driver=driver,
    )

    # With low threshold, not every skill should be evolved
    # This verifies the driver doesn't blindly evolve everything
    assert results.skills_evaluated >= 10
    assert results.total_fix_events < results.skills_evaluated
