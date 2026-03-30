"""Integration tests for LLM-driven evolution.

These tests make real LLM API calls and require API keys in the environment.
Mark with @pytest.mark.llm so they can be skipped in CI without keys.
"""

import os

import pytest

from evolve_trader.core.fitness import FitnessResult
from evolve_trader.core.llm_evolution_driver import LLMEvolutionConfig, LLMEvolutionDriver
from evolve_trader.core.version_dag import EvolutionMode
from evolve_trader.strategies.schema import StrategySkill

HAS_API_KEY = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))

skip_no_key = pytest.mark.skipif(not HAS_API_KEY, reason="No LLM API key available")


def _make_test_skill() -> StrategySkill:
    return StrategySkill(
        name="test-momentum-v1",
        description="Simple momentum strategy for testing",
        entry_logic="Buy when 20-day RSI crosses above 50 and price is above 50-day SMA",
        exit_logic="Sell when RSI drops below 40 or price drops 5% from entry",
        position_sizing_default="2% of portfolio per position",
        target_regime="risk-on, trending",
        expected_sharpe=0.8,
        expected_max_drawdown=0.15,
        expected_win_rate=0.50,
        risk_parameters={"max_position_pct": 0.05},
        body="# Momentum Strategy\n\nBuy uptrends, sell reversals.",
    )


@skip_no_key
def test_llm_driver_produces_fix():
    """LLM driver produces a valid FIX variant for an underperforming strategy."""
    config = LLMEvolutionConfig(model="gpt-4o-mini", max_tokens=1500)
    driver = LLMEvolutionDriver(config=config)
    skill = _make_test_skill()
    poor_performance = FitnessResult(sharpe=0.1, sharpe_std=0.5, max_drawdown=0.25, n_evaluations=5)

    suggestions = driver.suggest_evolution(
        skill, poor_performance, "Large drawdowns from poor entry timing"
    )

    assert len(suggestions) >= 1
    mode, evolved = suggestions[0]
    assert mode == EvolutionMode.FIX
    assert isinstance(evolved, StrategySkill)
    assert evolved.name != skill.name
    assert evolved.entry_logic
    assert evolved.exit_logic


@skip_no_key
def test_llm_driver_produces_derived():
    """LLM driver produces a DERIVED variant for a decent strategy."""
    config = LLMEvolutionConfig(
        model="gpt-4o-mini",
        max_tokens=1500,
        sharpe_fix_threshold=0.3,
        sharpe_derive_threshold=1.5,
    )
    driver = LLMEvolutionDriver(config=config)
    skill = _make_test_skill()
    ok_performance = FitnessResult(sharpe=0.7, sharpe_std=0.2, max_drawdown=0.10, n_evaluations=10)

    suggestions = driver.suggest_evolution(skill, ok_performance, None)

    assert len(suggestions) >= 1
    mode, evolved = suggestions[0]
    assert mode == EvolutionMode.DERIVED
    assert isinstance(evolved, StrategySkill)


@skip_no_key
def test_llm_driver_no_evolution_when_strong():
    """Strong strategies don't get evolved."""
    config = LLMEvolutionConfig(model="gpt-4o-mini")
    driver = LLMEvolutionDriver(config=config)
    skill = _make_test_skill()
    strong_performance = FitnessResult(
        sharpe=1.5, sharpe_std=0.1, max_drawdown=0.05, n_evaluations=20
    )

    suggestions = driver.suggest_evolution(skill, strong_performance, None)
    assert len(suggestions) == 0
