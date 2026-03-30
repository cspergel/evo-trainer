"""Tests for stochastic fitness evaluation."""

from evolve_trader.core.fitness import (
    FitnessResult,
    compare_strategy_fitness,
    compute_complexity_penalty,
)


def test_consistent_strategy_beats_volatile():
    """A strategy with lower variance is preferred over higher mean but higher variance."""
    consistent = FitnessResult(sharpe=0.9, sharpe_std=0.2, max_drawdown=0.08, n_evaluations=10)
    volatile = FitnessResult(sharpe=1.2, sharpe_std=0.8, max_drawdown=0.15, n_evaluations=10)
    assert compare_strategy_fitness(consistent, volatile) > 0


def test_identical_strategies_tie():
    """Identical strategies produce a tie (result near zero)."""
    a = FitnessResult(sharpe=1.0, sharpe_std=0.3, max_drawdown=0.10, n_evaluations=10)
    b = FitnessResult(sharpe=1.0, sharpe_std=0.3, max_drawdown=0.10, n_evaluations=10)
    assert abs(compare_strategy_fitness(a, b)) < 0.01


def test_complexity_penalty_specific_tickers():
    """Skills referencing specific tickers get penalized."""
    skill_text = "Buy AAPL when RSI > 50. Also buy NVDA on dips."
    penalty = compute_complexity_penalty(skill_text)
    assert penalty > 0


def test_complexity_penalty_general():
    """General skills without specific tickers get no penalty."""
    skill_text = "Buy when RSI crosses above 50 and price is above 50-day SMA."
    penalty = compute_complexity_penalty(skill_text)
    assert penalty == 0.0


def test_complexity_penalty_date_references():
    """Skills referencing narrow date ranges get penalized."""
    skill_text = "Only trade between January 15 and February 28, 2024."
    penalty = compute_complexity_penalty(skill_text)
    assert penalty > 0


def test_clearly_better_strategy_wins():
    """A strategy that is better on all dimensions wins decisively."""
    good = FitnessResult(sharpe=1.5, sharpe_std=0.1, max_drawdown=0.05, n_evaluations=20)
    bad = FitnessResult(sharpe=0.3, sharpe_std=0.5, max_drawdown=0.25, n_evaluations=20)
    assert compare_strategy_fitness(good, bad) > 0.5


def test_complexity_penalty_capped_at_one():
    """Penalty is capped at 1.0 even with many tickers and dates."""
    skill_text = (
        "Buy AAPL MSFT GOOGL AMZN NVDA META TSLA JPM "
        "between January 1, 2024 and February 1, 2024 "
        "and March 1, 2024 and April 1, 2024."
    )
    penalty = compute_complexity_penalty(skill_text)
    assert penalty <= 1.0
