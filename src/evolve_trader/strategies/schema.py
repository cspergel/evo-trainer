"""StrategySkill schema — the core data model for trading strategy SKILL.md files."""

from __future__ import annotations

import yaml
from pydantic import BaseModel, field_validator


class StrategySkill(BaseModel):
    """A trading strategy encoded as a structured skill.

    Maps to a SKILL.md file with YAML frontmatter and markdown body.
    """

    name: str
    description: str
    entry_logic: str
    exit_logic: str
    position_sizing_default: str
    target_regime: str
    expected_sharpe: float | None = None
    expected_max_drawdown: float | None = None
    expected_win_rate: float | None = None
    risk_parameters: dict[str, float] = {}
    body: str = ""

    @field_validator("entry_logic")
    @classmethod
    def entry_logic_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("entry_logic cannot be empty")
        return v

    @field_validator("exit_logic")
    @classmethod
    def exit_logic_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("exit_logic cannot be empty")
        return v


def parse_skill_md(content: str) -> StrategySkill:
    """Parse a SKILL.md file with YAML frontmatter into a StrategySkill."""
    # Handle BOM and leading whitespace (common on Windows editors)
    content = content.lstrip("\ufeff").lstrip()

    if not content.startswith("---"):
        raise ValueError("SKILL.md must start with YAML frontmatter (---)")

    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ValueError("SKILL.md must have closing --- for frontmatter")

    frontmatter = yaml.safe_load(parts[1])
    if not isinstance(frontmatter, dict):
        raise ValueError("SKILL.md frontmatter must contain valid YAML key-value pairs")

    body = parts[2].strip()

    return StrategySkill(**frontmatter, body=body)


def serialize_skill_md(skill: StrategySkill) -> str:
    """Serialize a StrategySkill back to SKILL.md format."""
    data = skill.model_dump(exclude={"body"}, exclude_none=True)
    frontmatter = yaml.dump(data, default_flow_style=False, sort_keys=False)
    return f"---\n{frontmatter}---\n\n{skill.body}\n"
