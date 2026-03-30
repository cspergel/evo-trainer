"""LLM-driven evolution driver using LiteLLM.

Replaces the rule-based stub with actual LLM reasoning for strategy evolution.
Uses the same FIX/DERIVED/CAPTURED patterns as OpenSpace but through our own
interface, keeping us decoupled from OpenSpace's internal types.

Requires ANTHROPIC_API_KEY or OPENAI_API_KEY in environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import litellm

from evolve_trader.core.fitness import FitnessResult
from evolve_trader.core.version_dag import EvolutionMode
from evolve_trader.strategies.schema import StrategySkill, parse_skill_md

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True

_FIX_PROMPT = """You are a trading strategy evolution engine. A strategy has been underperforming.

## Current Strategy
Name: {name}
Description: {description}
Entry Logic: {entry_logic}
Exit Logic: {exit_logic}
Position Sizing: {position_sizing}
Target Regime: {target_regime}

## Performance Data
Mean Sharpe Ratio: {sharpe:.3f}
Sharpe Std Dev: {sharpe_std:.3f}
Max Drawdown: {max_drawdown:.1%}
Evaluations: {n_evaluations}

## Failure Analysis
{failure_analysis}

## Your Task
Produce an improved version of this strategy. Focus on fixing the identified weakness.
Keep the general approach but improve entry/exit logic, add risk controls, or refine conditions.

Output ONLY a valid SKILL.md with YAML frontmatter. No explanations outside the SKILL.md format.
The format MUST be:
---
name: {name}-fix1
description: [improved description]
entry_logic: [improved entry logic]
exit_logic: [improved exit logic]
position_sizing_default: [sizing]
target_regime: [regime]
expected_sharpe: [float]
expected_max_drawdown: [float]
expected_win_rate: [float]
risk_parameters:
  max_position_pct: 0.05
---

# [Strategy Name]

## Reasoning Framework
[2-3 paragraphs explaining the improved approach]
"""

_DERIVED_PROMPT = """\
You are a trading strategy evolution engine. \
Create a specialized variant of this strategy.

## Parent Strategy
Name: {name}
Description: {description}
Entry Logic: {entry_logic}
Exit Logic: {exit_logic}
Target Regime: {target_regime}

## Performance Data
Mean Sharpe Ratio: {sharpe:.3f}
Max Drawdown: {max_drawdown:.1%}

## Specialization Direction
{direction}

## Your Task
Create a NEW specialized strategy derived from the parent. It should target a narrower
market condition where the parent's approach can be refined for better performance.

Output ONLY a valid SKILL.md with YAML frontmatter. No explanations outside the SKILL.md format.
The format MUST be exactly:
---
name: {name}-derived1
description: [specialized description]
entry_logic: [specialized entry logic]
exit_logic: [specialized exit logic]
position_sizing_default: [sizing approach]
target_regime: [narrower regime]
expected_sharpe: [float between 0.5 and 2.0]
expected_max_drawdown: [float between 0.05 and 0.20]
expected_win_rate: [float between 0.40 and 0.65]
risk_parameters:
  max_position_pct: 0.05
---

# [Strategy Name]

## Reasoning Framework
[2-3 paragraphs explaining the specialized approach]
"""


@dataclass
class LLMEvolutionConfig:
    """Configuration for LLM-driven evolution."""

    model: str = "anthropic/claude-sonnet-4-20250514"
    max_tokens: int = 2000
    temperature: float = 0.7
    sharpe_fix_threshold: float = 0.5
    sharpe_derive_threshold: float = 1.0


class LLMEvolutionDriver:
    """LLM-driven evolution driver using LiteLLM.

    Produces FIX variants when performance is poor (sharpe < fix_threshold).
    Produces DERIVED variants when performance is decent but could specialize
    (sharpe >= fix_threshold but < derive_threshold).
    """

    def __init__(self, config: LLMEvolutionConfig | None = None) -> None:
        self._config = config or LLMEvolutionConfig()
        self._ensure_api_key()

    def _ensure_api_key(self) -> None:
        """Verify at least one LLM API key is available."""
        has_key = bool(
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
        )
        if not has_key:
            raise OSError(
                "No LLM API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, "
                "or OPENROUTER_API_KEY in environment or .env file."
            )

    def suggest_evolution(
        self,
        skill: StrategySkill,
        performance: FitnessResult,
        failure_analysis: str | None,
    ) -> list[tuple[EvolutionMode, StrategySkill]]:
        """Suggest evolved variants using LLM reasoning."""
        suggestions: list[tuple[EvolutionMode, StrategySkill]] = []

        if performance.n_evaluations < 1:
            return suggestions

        # FIX: underperforming strategies
        if performance.sharpe < self._config.sharpe_fix_threshold:
            fixed = self._generate_fix(skill, performance, failure_analysis)
            if fixed:
                suggestions.append((EvolutionMode.FIX, fixed))

        # DERIVED: decent strategies that could specialize
        elif performance.sharpe < self._config.sharpe_derive_threshold:
            derived = self._generate_derived(skill, performance)
            if derived:
                suggestions.append((EvolutionMode.DERIVED, derived))

        return suggestions

    def _generate_fix(
        self,
        skill: StrategySkill,
        performance: FitnessResult,
        failure_analysis: str | None,
    ) -> StrategySkill | None:
        """Use LLM to generate a FIX variant."""
        prompt = _FIX_PROMPT.format(
            name=skill.name,
            description=skill.description,
            entry_logic=skill.entry_logic,
            exit_logic=skill.exit_logic,
            position_sizing=skill.position_sizing_default,
            target_regime=skill.target_regime,
            sharpe=performance.sharpe,
            sharpe_std=performance.sharpe_std,
            max_drawdown=performance.max_drawdown,
            n_evaluations=performance.n_evaluations,
            failure_analysis=failure_analysis or "No specific failure identified.",
        )
        return self._call_llm_and_parse(prompt)

    def _generate_derived(
        self,
        skill: StrategySkill,
        performance: FitnessResult,
    ) -> StrategySkill | None:
        """Use LLM to generate a DERIVED variant."""
        prompt = _DERIVED_PROMPT.format(
            name=skill.name,
            description=skill.description,
            entry_logic=skill.entry_logic,
            exit_logic=skill.exit_logic,
            target_regime=skill.target_regime,
            sharpe=performance.sharpe,
            max_drawdown=performance.max_drawdown,
            direction=(
                "Specialize this strategy for a specific market regime or sector "
                "where its core logic would perform better."
            ),
        )
        return self._call_llm_and_parse(prompt)

    def _call_llm_and_parse(self, prompt: str, retries: int = 2) -> StrategySkill | None:
        """Call LLM and parse response into a StrategySkill. Retries on parse failure."""
        for _attempt in range(retries + 1):
            result = self._single_llm_call(prompt)
            if result is not None:
                return result
        return None

    def _single_llm_call(self, prompt: str) -> StrategySkill | None:
        """Single LLM call attempt."""
        try:
            response = litellm.completion(
                model=self._config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            )
            msg = response.choices[0].message
            content = msg.content if msg else None
            if not content:
                return None

            # Extract SKILL.md content — LLM might wrap in markdown code block
            skill_md = _extract_skill_md(content)
            return parse_skill_md(skill_md)

        except Exception:
            return None


def _extract_skill_md(content: str) -> str:
    """Extract SKILL.md content from LLM response.

    Handles cases where LLM wraps output in ```markdown blocks.
    """
    # Strip markdown code fences if present
    if "```" in content:
        lines = content.split("\n")
        inside_fence = False
        extracted: list[str] = []
        for line in lines:
            if line.strip().startswith("```"):
                inside_fence = not inside_fence
                continue
            if inside_fence:
                extracted.append(line)
        if extracted:
            return "\n".join(extracted)

    return content.strip()
