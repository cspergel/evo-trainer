"""Strategy incubator — tournament-based candidate generation and filtering.

Generates new strategy candidates via mutation and crossover,
evaluates them, and graduates the best through the profitability gate.

Per profitability contract: candidates are isolated from production.
Every experiment is logged in the ResearchLedger.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from evolve_trader.core.fitness import FitnessResult
from evolve_trader.incubator.research_ledger import ResearchLedger
from evolve_trader.strategies.schema import StrategySkill


@dataclass
class IncubatorCandidate:
    """A candidate strategy in the incubator."""

    skill: StrategySkill
    fitness: FitnessResult | None = None
    generation: int = 0
    parent_names: list[str] = field(default_factory=list)


@dataclass
class TournamentResult:
    """Result of an incubator tournament round."""

    candidates_generated: int
    candidates_evaluated: int
    candidates_graduated: int  # passed fitness threshold
    best_fitness: float
    population_diversity: float  # unique strategy families


class Incubator:
    """Strategy incubator with tournament selection.

    Generates candidates, evaluates fitness, graduates the best.
    All experiments tracked in the ResearchLedger.
    """

    def __init__(
        self,
        seed_strategies: list[StrategySkill],
        ledger: ResearchLedger | None = None,
        graduation_sharpe: float = 0.5,
    ) -> None:
        self._seeds = seed_strategies
        self._population: list[IncubatorCandidate] = []
        self._ledger = ledger or ResearchLedger()
        self._graduation_sharpe = graduation_sharpe
        self._generation = 0

    @property
    def ledger(self) -> ResearchLedger:
        return self._ledger

    @property
    def population_size(self) -> int:
        return len(self._population)

    def generate_candidates(self, n: int = 5) -> list[IncubatorCandidate]:
        """Generate new candidates via mutation of seed strategies.

        Each candidate is logged in the research ledger.
        """
        candidates: list[IncubatorCandidate] = []
        self._generation += 1

        for i in range(n):
            parent = random.choice(self._seeds)
            mutated = _mutate_strategy(parent, self._generation, i)
            candidate = IncubatorCandidate(
                skill=mutated,
                generation=self._generation,
                parent_names=[parent.name],
            )
            candidates.append(candidate)

            # Log in research ledger
            self._ledger.log_experiment(
                candidate_id=mutated.name,
                hypothesis=f"Mutation of {parent.name}: {mutated.description}",
                parent_id=parent.name,
            )

        self._population.extend(candidates)
        return candidates

    def evaluate_candidate(
        self,
        candidate: IncubatorCandidate,
        fitness: FitnessResult,
    ) -> bool:
        """Evaluate a candidate and determine if it graduates.

        Returns True if the candidate's fitness exceeds the graduation threshold.
        """
        candidate.fitness = fitness

        # Find the ledger record
        records = [r for r in self._ledger.get_all() if r.candidate_id == candidate.skill.name]
        if records:
            graduated = fitness.sharpe >= self._graduation_sharpe
            self._ledger.record_result(
                records[-1],
                sharpe=fitness.sharpe,
                windows=fitness.n_evaluations,
                promoted=graduated,
            )
            if not graduated:
                self._ledger.discard(records[-1])
            return graduated

        return False

    def run_tournament(
        self,
        n_candidates: int = 5,
        evaluate_fn: Callable[[StrategySkill], FitnessResult] | None = None,
    ) -> TournamentResult:
        """Run a full tournament round.

        Generates candidates, evaluates them, graduates the best.
        """
        candidates = self.generate_candidates(n_candidates)
        graduated = 0
        best_sharpe = 0.0

        for candidate in candidates:
            if evaluate_fn:
                fitness = evaluate_fn(candidate.skill)
            else:
                # Default: synthetic fitness based on skill characteristics
                fitness = _synthetic_fitness(candidate.skill)

            is_grad = self.evaluate_candidate(candidate, fitness)
            if is_grad:
                graduated += 1
            best_sharpe = max(best_sharpe, fitness.sharpe)

        # Population diversity: unique parent families
        families = {tuple(c.parent_names) for c in self._population if c.parent_names}
        diversity = len(families) / max(1, len(self._population))

        return TournamentResult(
            candidates_generated=n_candidates,
            candidates_evaluated=n_candidates,
            candidates_graduated=graduated,
            best_fitness=best_sharpe,
            population_diversity=diversity,
        )


def _mutate_strategy(parent: StrategySkill, generation: int, index: int) -> StrategySkill:
    """Create a mutated variant of a strategy."""
    return StrategySkill(
        name=f"{parent.name}-gen{generation}-{index}",
        description=f"Mutation of {parent.name} (gen {generation})",
        entry_logic=f"{parent.entry_logic} with tighter confirmation",
        exit_logic=f"{parent.exit_logic} with adjusted stops",
        position_sizing_default=parent.position_sizing_default,
        target_regime=parent.target_regime,
        expected_sharpe=parent.expected_sharpe,
        expected_max_drawdown=parent.expected_max_drawdown,
        expected_win_rate=parent.expected_win_rate,
        risk_parameters=parent.risk_parameters,
        body=f"{parent.body}\n\n## Generation {generation}\nMutated variant.",
    )


def _synthetic_fitness(skill: StrategySkill) -> FitnessResult:
    """Generate synthetic fitness for testing."""
    rng = random.Random(hash(skill.name))
    return FitnessResult(
        sharpe=rng.uniform(-0.5, 1.5),
        sharpe_std=rng.uniform(0.1, 0.5),
        max_drawdown=rng.uniform(0.03, 0.20),
        n_evaluations=3,
    )
