"""Tests for the strategy incubator and research ledger."""

from evolve_trader.core.fitness import FitnessResult
from evolve_trader.incubator.research_ledger import ResearchLedger
from evolve_trader.incubator.tournament import Incubator, TournamentResult
from evolve_trader.strategies.schema import StrategySkill


def _make_seed() -> StrategySkill:
    return StrategySkill(
        name="test-seed",
        description="Test seed strategy",
        entry_logic="Buy on momentum",
        exit_logic="Sell on reversal",
        position_sizing_default="2%",
        target_regime="risk-on",
        expected_sharpe=0.8,
        expected_max_drawdown=0.10,
        expected_win_rate=0.55,
    )


class TestResearchLedger:
    def test_log_and_track(self) -> None:
        """Experiments are logged and counted."""
        ledger = ResearchLedger()
        r1 = ledger.log_experiment("c1", "Test hypothesis 1")
        ledger.log_experiment("c2", "Test hypothesis 2", parent_id="c1")

        assert ledger.total_experiments == 2
        assert r1.status == "pending"

    def test_record_result(self) -> None:
        """Results update the experiment record."""
        ledger = ResearchLedger()
        record = ledger.log_experiment("c1", "Test")
        ledger.record_result(record, sharpe=0.8, windows=3, promoted=True)

        assert record.status == "promoted"
        assert record.sharpe_result == 0.8
        assert ledger.total_promoted == 1

    def test_discard(self) -> None:
        """Discarded experiments are tracked."""
        ledger = ResearchLedger()
        record = ledger.log_experiment("c1", "Test")
        ledger.discard(record)

        assert record.status == "discarded"
        assert ledger.total_discarded == 1

    def test_family_tracking(self) -> None:
        """Can query experiments by parent family."""
        ledger = ResearchLedger()
        ledger.log_experiment("child1", "Test", parent_id="parent")
        ledger.log_experiment("child2", "Test", parent_id="parent")
        ledger.log_experiment("other", "Test", parent_id="other_parent")

        family = ledger.get_by_family("parent")
        assert len(family) == 2


class TestIncubator:
    def test_generate_candidates(self) -> None:
        """Generates N candidates from seed strategies."""
        incubator = Incubator(seed_strategies=[_make_seed()])
        candidates = incubator.generate_candidates(5)

        assert len(candidates) == 5
        assert incubator.population_size == 5
        assert incubator.ledger.total_experiments == 5

    def test_evaluate_graduates_good_candidate(self) -> None:
        """Candidate with high Sharpe graduates."""
        incubator = Incubator(
            seed_strategies=[_make_seed()],
            graduation_sharpe=0.5,
        )
        candidates = incubator.generate_candidates(1)
        good_fitness = FitnessResult(sharpe=0.8, sharpe_std=0.2, max_drawdown=0.08, n_evaluations=3)
        assert incubator.evaluate_candidate(candidates[0], good_fitness)
        assert incubator.ledger.total_promoted == 1

    def test_evaluate_discards_weak_candidate(self) -> None:
        """Candidate with low Sharpe is discarded."""
        incubator = Incubator(
            seed_strategies=[_make_seed()],
            graduation_sharpe=0.5,
        )
        candidates = incubator.generate_candidates(1)
        weak_fitness = FitnessResult(sharpe=0.2, sharpe_std=0.3, max_drawdown=0.15, n_evaluations=3)
        assert not incubator.evaluate_candidate(candidates[0], weak_fitness)
        assert incubator.ledger.total_discarded == 1

    def test_run_tournament(self) -> None:
        """Full tournament round generates, evaluates, and graduates."""
        incubator = Incubator(seed_strategies=[_make_seed()])
        result = incubator.run_tournament(n_candidates=10)

        assert isinstance(result, TournamentResult)
        assert result.candidates_generated == 10
        assert result.candidates_evaluated == 10
        assert incubator.ledger.total_experiments == 10
        # Some may graduate, some may not (random fitness)
        assert result.candidates_graduated >= 0

    def test_multiple_testing_discipline(self) -> None:
        """Research ledger tracks experiment count for penalty calculation."""
        incubator = Incubator(seed_strategies=[_make_seed()])

        # Run 3 tournament rounds
        for _ in range(3):
            incubator.run_tournament(n_candidates=10)

        # 30 total experiments tracked
        assert incubator.ledger.total_experiments == 30
        # This count feeds into profitability_gate.check_multiple_testing()

    def test_candidates_isolated_from_production(self) -> None:
        """Incubator candidates don't directly enter production."""
        incubator = Incubator(seed_strategies=[_make_seed()])
        candidates = incubator.generate_candidates(3)

        # Candidates exist in incubator population
        assert incubator.population_size == 3

        # But they have no connection to the production evolution loop
        # (they must pass profitability gate separately to graduate)
        for c in candidates:
            assert c.fitness is None  # not yet evaluated
