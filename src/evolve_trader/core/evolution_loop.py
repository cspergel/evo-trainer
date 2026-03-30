"""Core evolution loop connecting strategy evaluation to the evolution pipeline.

Orchestrates: load seeds -> replay -> evaluate -> compare fitness -> evolve -> record.

The actual LLM-driven evolution (FIX/DERIVED/CAPTURED via OpenSpace) requires API keys
and is pluggable via the `EvolutionDriver` protocol. Phase 1 provides a rule-based
driver for testing; later phases wire in the OpenSpace engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from evolve_trader.core.analyzer import (
    TradeResult,
    analyze_failure_mode,
    analyze_strategy_performance,
)
from evolve_trader.core.fitness import (
    FitnessResult,
    compute_complexity_penalty,
)
from evolve_trader.core.profitability_gate import run_promotion_gate
from evolve_trader.core.validation import WalkForwardConfig, generate_walk_forward_windows
from evolve_trader.core.version_dag import EvolutionMode, VersionDAG
from evolve_trader.strategies.schema import StrategySkill, parse_skill_md


@dataclass
class PerformanceComparison:
    """Comparison between a parent and child strategy."""

    parent_name: str
    child_name: str
    parent_oos_sharpe: float
    child_oos_sharpe: float


@dataclass
class EvolutionResults:
    """Results from a complete evolution cycle."""

    evolved_skills: list[str] = field(default_factory=list)
    performance_comparisons: list[PerformanceComparison] = field(default_factory=list)
    total_fix_events: int = 0
    total_derived_events: int = 0
    total_captured_events: int = 0
    skills_evaluated: int = 0
    skills_promoted: int = 0


class EvolutionDriver(Protocol):
    """Protocol for the evolution engine that produces new skill variants.

    Phase 1: rule-based stub for testing.
    Later: wired to OpenSpace's LLM-driven FIX/DERIVED/CAPTURED.
    """

    def suggest_evolution(
        self,
        skill: StrategySkill,
        performance: FitnessResult,
        failure_analysis: str | None,
    ) -> list[tuple[EvolutionMode, StrategySkill]]:
        """Suggest zero or more evolved variants of a skill."""
        ...


class RuleBasedEvolutionDriver:
    """Simple rule-based evolution driver for Phase 1 testing.

    Produces FIX variants when performance is poor.
    No LLM calls required.
    """

    def __init__(self, sharpe_threshold: float = 0.3) -> None:
        self._sharpe_threshold = sharpe_threshold

    def suggest_evolution(
        self,
        skill: StrategySkill,
        performance: FitnessResult,
        failure_analysis: str | None,
    ) -> list[tuple[EvolutionMode, StrategySkill]]:
        suggestions: list[tuple[EvolutionMode, StrategySkill]] = []

        if performance.sharpe < self._sharpe_threshold and performance.n_evaluations >= 1:
            fix_suffix = "tighter stop-loss and confirmation signal"
            fixed_skill = StrategySkill(
                name=f"{skill.name}-fix1",
                description=f"{skill.description} (fixed: {fix_suffix})",
                entry_logic=(
                    f"{skill.entry_logic}. Require volume confirmation above 20-day average."
                ),
                exit_logic=f"{skill.exit_logic}. Add trailing stop at 3% from peak.",
                position_sizing_default=skill.position_sizing_default,
                target_regime=skill.target_regime,
                expected_sharpe=skill.expected_sharpe,
                expected_max_drawdown=skill.expected_max_drawdown,
                expected_win_rate=skill.expected_win_rate,
                risk_parameters=skill.risk_parameters,
                body=f"{skill.body}\n\n## Fix Applied\n{fix_suffix}",
            )
            suggestions.append((EvolutionMode.FIX, fixed_skill))

        return suggestions


def load_seed_skills(skills_dir: str | Path) -> list[StrategySkill]:
    """Load all SKILL.md files from a directory."""
    skills_path = Path(skills_dir)
    skills: list[StrategySkill] = []
    for md_file in sorted(skills_path.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        try:
            skill = parse_skill_md(content)
            skills.append(skill)
        except (ValueError, KeyError):
            continue
    return skills


def evaluate_skill_fitness(
    skill: StrategySkill,
    trades: list[TradeResult],
    initial_capital: float,
    n_windows: int,
) -> FitnessResult:
    """Evaluate a skill's fitness from its trade results across walk-forward windows."""
    perf = analyze_strategy_performance(trades, initial_capital)
    penalty = compute_complexity_penalty(f"{skill.entry_logic} {skill.exit_logic} {skill.body}")

    return FitnessResult(
        sharpe=perf.sharpe_ratio * (1.0 - penalty),
        sharpe_std=perf.variance**0.5 if perf.variance > 0 else 0.0,
        max_drawdown=perf.max_drawdown,
        n_evaluations=n_windows,
    )


def run_evolution_cycle(
    seed_skills_dir: str | Path,
    replay_days: int,
    universe: str,
    version_dag: VersionDAG,
    initial_capital: float = 100_000.0,
    driver: EvolutionDriver | None = None,
) -> EvolutionResults:
    """Run a complete evolution cycle: load seeds, evaluate, evolve, record.

    This is the Phase 1 orchestration loop. It uses synthetic trade data
    for evaluation (real market data integration comes with the replay harness).
    The evolution driver is pluggable — defaults to rule-based for testing.
    """
    if driver is None:
        driver = RuleBasedEvolutionDriver()

    results = EvolutionResults()
    skills = load_seed_skills(seed_skills_dir)

    wf_config = WalkForwardConfig(
        total_days=replay_days,
        train_days=max(10, replay_days // 3),
        validate_days=max(5, replay_days // 6),
        step_days=max(5, replay_days // 6),
    )
    windows = generate_walk_forward_windows(wf_config)
    n_windows = len(windows)

    # Register all seeds in the DAG
    for skill in skills:
        version_dag.add_root(skill.name)
        results.skills_evaluated += 1

    # Evaluate each seed with synthetic trade data
    skill_fitness: dict[str, FitnessResult] = {}
    for skill in skills:
        # Generate synthetic trades based on skill's expected characteristics
        trades = _generate_synthetic_trades(skill, n_trades=max(5, n_windows * 2))
        fitness = evaluate_skill_fitness(skill, trades, initial_capital, n_windows)
        skill_fitness[skill.name] = fitness

    # Attempt evolution on underperforming skills
    for skill in skills:
        fitness = skill_fitness[skill.name]
        failure = None

        trades = _generate_synthetic_trades(skill, n_trades=5)
        failure_result = analyze_failure_mode(trades)
        if failure_result:
            failure = failure_result.description

        suggestions = driver.suggest_evolution(skill, fitness, failure)

        for mode, evolved_skill in suggestions:
            # Evaluate the evolved variant
            n_evolved_trades = max(5, n_windows * 2)
            evolved_trades = _generate_synthetic_trades(evolved_skill, n_trades=n_evolved_trades)
            evolved_fitness = evaluate_skill_fitness(
                evolved_skill, evolved_trades, initial_capital, n_windows
            )

            # Run profitability gate before promoting
            gate_report = run_promotion_gate(
                strategy_sharpe_by_window=[evolved_fitness.sharpe] * n_windows,
                baseline_sharpe_by_window=[fitness.sharpe] * n_windows,
                expected_edge_bps=max(0, (evolved_fitness.sharpe - fitness.sharpe) * 100),
                estimated_cost_bps=5.0,  # conservative estimate
                trades_per_window=[n_evolved_trades] * n_windows,
                regime_labels_seen=2,
                pnl_by_window=[1.0] * n_windows,  # uniform for synthetic
            )

            # Record in DAG regardless (lineage tracking)
            version_dag.add_evolution(
                parent=skill.name,
                child=evolved_skill.name,
                mode=mode,
                reason=f"Evolution from {skill.name}: {evolved_skill.description}",
                metrics={
                    "parent_sharpe": fitness.sharpe,
                    "child_sharpe": evolved_fitness.sharpe,
                    "gate_passed": gate_report.passed,
                },
            )

            results.evolved_skills.append(evolved_skill.name)
            results.performance_comparisons.append(
                PerformanceComparison(
                    parent_name=skill.name,
                    child_name=evolved_skill.name,
                    parent_oos_sharpe=fitness.sharpe,
                    child_oos_sharpe=evolved_fitness.sharpe,
                )
            )

            if gate_report.passed:
                results.skills_promoted += 1

            if mode == EvolutionMode.FIX:
                results.total_fix_events += 1
            elif mode == EvolutionMode.DERIVED:
                results.total_derived_events += 1
            elif mode == EvolutionMode.CAPTURED:
                results.total_captured_events += 1

    return results


def _generate_synthetic_trades(
    skill: StrategySkill,
    n_trades: int = 10,
) -> list[TradeResult]:
    """Generate synthetic trades based on a skill's expected characteristics.

    Used for Phase 1 testing when real market replay is not yet available.
    Trade outcomes are derived from the skill's expected_sharpe and expected_win_rate.
    """
    import random

    rng = random.Random(hash(skill.name))
    win_rate = skill.expected_win_rate or 0.5
    trades: list[TradeResult] = []

    for i in range(n_trades):
        entry_price = 100.0 + rng.uniform(-10, 10)
        is_win = rng.random() < win_rate

        if is_win:
            exit_price = entry_price * (1 + rng.uniform(0.01, 0.08))
        else:
            exit_price = entry_price * (1 - rng.uniform(0.01, 0.10))

        trades.append(
            TradeResult(
                ticker=f"SYN{i:03d}",
                entry_price=entry_price,
                exit_price=exit_price,
                shares=rng.randint(5, 50),
                entry_date=f"2025-01-{(i + 1):02d}",
                exit_date=f"2025-01-{(i + 5):02d}",
                reasoning=f"Synthetic trade for {skill.name}",
            )
        )

    return trades
