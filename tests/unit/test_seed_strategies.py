"""Tests that all seed strategies parse correctly and have required fields."""

from pathlib import Path

from evolve_trader.strategies.schema import StrategySkill, parse_skill_md

SKILLS_DIR = Path(__file__).parent.parent.parent / "src" / "evolve_trader" / "strategies" / "skills"


def get_all_skill_files() -> list[Path]:
    """Collect all SKILL.md files."""
    return sorted(SKILLS_DIR.glob("*.md"))


class TestSeedStrategies:
    """Validate all seed strategy SKILL.md files."""

    def test_minimum_seed_count(self) -> None:
        """We have at least 10 seed strategies (including capital-preservation)."""
        skills = get_all_skill_files()
        assert len(skills) >= 10, f"Only {len(skills)} seed strategies found, need at least 10"

    def test_all_skills_parse(self) -> None:
        """Every seed strategy file parses into a valid StrategySkill."""
        for skill_path in get_all_skill_files():
            content = skill_path.read_text(encoding="utf-8")
            skill = parse_skill_md(content)
            assert isinstance(skill, StrategySkill), f"{skill_path.name} failed to parse"
            assert skill.name, f"{skill_path.name} has no name"
            assert skill.entry_logic, f"{skill_path.name} has no entry_logic"
            assert skill.exit_logic, f"{skill_path.name} has no exit_logic"
            assert skill.target_regime, f"{skill_path.name} has no target_regime"

    def test_strategy_diversity(self) -> None:
        """Seed strategies cover diverse approaches (at least 4 families)."""
        skills = get_all_skill_files()
        names = [p.stem for p in skills]
        families: set[str] = set()
        for name in names:
            if "momentum" in name or "trend" in name or "crossover" in name:
                families.add("trend")
            elif "reversion" in name or "bollinger" in name or "rsi" in name or "pairs" in name:
                families.add("mean-reversion")
            elif "value" in name or "fundamental" in name:
                families.add("value")
            elif "earnings" in name or "gap" in name or "breakout" in name:
                families.add("event-driven")
            elif "defensive" in name or "capital-preservation" in name:
                families.add("defensive")
            else:
                families.add("other")
        assert len(families) >= 4, f"Only {len(families)} strategy families: {families}"

    def test_all_have_body_content(self) -> None:
        """Every strategy has a non-empty reasoning framework in the body."""
        for skill_path in get_all_skill_files():
            content = skill_path.read_text(encoding="utf-8")
            skill = parse_skill_md(content)
            assert len(skill.body) > 50, f"{skill_path.name} has insufficient body content"
