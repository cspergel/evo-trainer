"""Counterfactual replay — validates proposed adjustments.

Replays historical data with and without a proposed change,
compares outcomes. Required by Phase 7 acceptance criteria.

Per profitability contract section 7: each new adjustment must
prove incremental value over the system without it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CounterfactualResult:
    """Result of a counterfactual comparison."""

    baseline_sharpe: float  # System without the change
    adjusted_sharpe: float  # System with the change
    improvement: float  # adjusted - baseline
    passes_simplicity_tax: bool  # improvement >= 0.1
    evaluation_windows: int
    description: str


def run_counterfactual(
    baseline_sharpe_by_window: list[float],
    adjusted_sharpe_by_window: list[float],
    min_improvement: float = 0.1,
) -> CounterfactualResult:
    """Compare system performance with and without a proposed change.

    The change must improve Sharpe by at least min_improvement
    to pass the simplicity tax.
    """
    if not baseline_sharpe_by_window or not adjusted_sharpe_by_window:
        return CounterfactualResult(
            baseline_sharpe=0.0,
            adjusted_sharpe=0.0,
            improvement=0.0,
            passes_simplicity_tax=False,
            evaluation_windows=0,
            description="No data for counterfactual comparison",
        )

    import math

    n = min(len(baseline_sharpe_by_window), len(adjusted_sharpe_by_window))
    avg_baseline = sum(baseline_sharpe_by_window[:n]) / n
    avg_adjusted = sum(adjusted_sharpe_by_window[:n]) / n
    improvement = avg_adjusted - avg_baseline

    # Per-window win count (contract section 1: majority of windows must improve)
    wins = sum(
        1
        for b, a in zip(
            baseline_sharpe_by_window[:n],
            adjusted_sharpe_by_window[:n],
            strict=True,
        )
        if a > b
    )
    majority_wins = wins >= math.ceil(n / 2)

    passes = improvement >= min_improvement and majority_wins
    win_desc = f"{wins}/{n} windows improved"

    return CounterfactualResult(
        baseline_sharpe=round(avg_baseline, 4),
        adjusted_sharpe=round(avg_adjusted, 4),
        improvement=round(improvement, 4),
        passes_simplicity_tax=passes,
        evaluation_windows=n,
        description=(
            f"Avg Sharpe: {avg_baseline:.3f} → {avg_adjusted:.3f} "
            f"(Δ{improvement:+.3f}, {win_desc}, "
            f"{'PASS' if passes else 'FAIL'})"
        ),
    )
