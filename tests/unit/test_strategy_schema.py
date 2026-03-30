"""Tests for the StrategySkill schema."""

import pytest

from evolve_trader.strategies.schema import StrategySkill, parse_skill_md, serialize_skill_md


def test_strategy_skill_has_required_fields():
    """A StrategySkill must have all required fields."""
    skill = StrategySkill(
        name="test-momentum-v1",
        description="Simple momentum strategy for testing",
        entry_logic="Buy when 20-day RSI crosses above 50 and price is above 50-day SMA",
        exit_logic="Sell when RSI drops below 40 or price drops 5% from entry (stop-loss)",
        position_sizing_default="2% of portfolio per position",
        target_regime="risk-on, momentum strengthening",
        expected_sharpe=1.0,
        expected_max_drawdown=0.10,
        expected_win_rate=0.55,
        risk_parameters={"max_position_pct": 0.05},
    )
    assert skill.name == "test-momentum-v1"
    assert skill.expected_sharpe == 1.0
    assert skill.risk_parameters["max_position_pct"] == 0.05


def test_strategy_skill_rejects_missing_entry_logic():
    """Entry logic is required."""
    with pytest.raises(ValueError):
        StrategySkill(
            name="bad-strategy",
            description="Missing entry logic",
            entry_logic="",
            exit_logic="Some exit logic",
            position_sizing_default="1%",
            target_regime="any",
        )


def test_strategy_skill_rejects_missing_exit_logic():
    """Exit logic is required."""
    with pytest.raises(ValueError):
        StrategySkill(
            name="bad-strategy",
            description="Missing exit logic",
            entry_logic="Some entry logic",
            exit_logic="",
            position_sizing_default="1%",
            target_regime="any",
        )


def test_parse_skill_md_roundtrip():
    """A SKILL.md file can be parsed and re-serialized."""
    md_content = """---
name: test-momentum-v1
description: Simple momentum strategy for testing
entry_logic: Buy when 20-day RSI crosses above 50
exit_logic: Sell when RSI drops below 40
position_sizing_default: 2% of portfolio
target_regime: risk-on
expected_sharpe: 1.0
expected_max_drawdown: 0.10
expected_win_rate: 0.55
risk_parameters:
  max_position_pct: 0.05
---

# test-momentum-v1

## Reasoning Framework
When momentum indicators confirm an uptrend...
"""
    skill = parse_skill_md(md_content)
    assert skill.name == "test-momentum-v1"
    assert skill.expected_sharpe == 1.0
    assert "momentum indicators" in skill.body


def test_parse_skill_md_rejects_no_frontmatter():
    """SKILL.md without frontmatter is rejected."""
    with pytest.raises(ValueError, match="frontmatter"):
        parse_skill_md("Just some markdown without frontmatter")


def test_serialize_roundtrip():
    """Serialize then parse produces equivalent skill."""
    original = StrategySkill(
        name="roundtrip-test",
        description="Test serialization",
        entry_logic="Buy signal",
        exit_logic="Sell signal",
        position_sizing_default="1%",
        target_regime="any",
        body="# Reasoning\nTest body content.",
    )
    md = serialize_skill_md(original)
    parsed = parse_skill_md(md)
    assert parsed.name == original.name
    assert parsed.entry_logic == original.entry_logic
    assert parsed.body == original.body
    assert parsed.expected_sharpe == original.expected_sharpe
    assert parsed.risk_parameters == original.risk_parameters


def test_parse_skill_md_handles_bom():
    """SKILL.md with UTF-8 BOM still parses."""
    md_content = (
        "\ufeff---\n"
        "name: bom-test\n"
        "description: test\n"
        "entry_logic: buy\n"
        "exit_logic: sell\n"
        "position_sizing_default: 1%\n"
        "target_regime: any\n"
        "---\n"
        "body"
    )
    skill = parse_skill_md(md_content)
    assert skill.name == "bom-test"


def test_parse_skill_md_rejects_empty_frontmatter():
    """Empty frontmatter raises ValueError."""
    with pytest.raises(ValueError, match="key-value"):
        parse_skill_md("---\n---\nbody")
