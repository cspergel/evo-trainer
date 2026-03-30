"""Profitability Gate — runtime enforcement of the profitability contract.

Every strategy promotion, signal source activation, and live deployment
must pass through this gate. It codifies docs/implementation/profitability-contract.md
as executable checks.

This is the governing authority. If a check fails, the component is blocked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GateResult(Enum):
    """Outcome of a profitability gate check."""

    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class GateCheck:
    """A single gate check result."""

    name: str
    result: GateResult
    message: str
    value: float | None = None
    threshold: float | None = None


@dataclass
class GateReport:
    """Full gate evaluation report."""

    checks: list[GateCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.result == GateResult.PASS for c in self.checks)

    @property
    def failed_checks(self) -> list[GateCheck]:
        return [c for c in self.checks if c.result == GateResult.FAIL]

    @property
    def summary(self) -> str:
        passed = sum(1 for c in self.checks if c.result == GateResult.PASS)
        failed = sum(1 for c in self.checks if c.result == GateResult.FAIL)
        insufficient = sum(1 for c in self.checks if c.result == GateResult.INSUFFICIENT_DATA)
        return f"{passed} passed, {failed} failed, {insufficient} insufficient data"


# --- Contract constants ---

# Section 1: Baseline
MIN_WINDOWS_FOR_PROMOTION = 3
MIN_WINDOWS_BEATING_BASELINE = 2

# Section 2: Executable alpha
MIN_EDGE_TO_COST_RATIO = 2.0

# Section 3: Statistical bar
MIN_TRADES_PER_WINDOW = 30
MIN_TOTAL_TRADES_LOW_FREQ = 90
MIN_REGIME_LABELS = 2
MAX_SINGLE_WINDOW_PNL_FRACTION = 0.50

# Section 4: Capacity
MAX_ADV_PARTICIPATION = 0.01  # 1% of average daily volume

# Section 6: Scope
ALLOWED_UNIVERSE = "sp500"
MAX_ACTIVE_STRATEGIES = 3  # plus capital preservation
MAX_ACTIVE_SIGNAL_SOURCES = 3

# Section 7: Simplicity tax
MIN_INCREMENTAL_SHARPE = 0.1

# Section 10: Paper/live deviation
MIN_PAPER_LIVE_CORRELATION = 0.80
PAPER_LIVE_SHARPE_DEVIATION_THRESHOLD = 1.0  # std devs


def check_baseline_beating(
    strategy_sharpe_by_window: list[float],
    baseline_sharpe_by_window: list[float],
) -> GateCheck:
    """Section 1: Strategy must beat baseline in at least 2 of 3 windows."""
    n_windows = len(strategy_sharpe_by_window)
    if n_windows < MIN_WINDOWS_FOR_PROMOTION:
        return GateCheck(
            name="baseline_beating",
            result=GateResult.INSUFFICIENT_DATA,
            message=f"Need {MIN_WINDOWS_FOR_PROMOTION} windows, have {n_windows}",
        )

    wins = sum(
        1
        for s, b in zip(strategy_sharpe_by_window, baseline_sharpe_by_window, strict=True)
        if s > b
    )

    passed = wins >= MIN_WINDOWS_BEATING_BASELINE
    return GateCheck(
        name="baseline_beating",
        result=GateResult.PASS if passed else GateResult.FAIL,
        message=(
            f"Beat baseline in {wins}/{n_windows} windows " f"(need {MIN_WINDOWS_BEATING_BASELINE})"
        ),
        value=float(wins),
        threshold=float(MIN_WINDOWS_BEATING_BASELINE),
    )


def check_executable_alpha(
    expected_edge_bps: float,
    estimated_round_trip_cost_bps: float,
) -> GateCheck:
    """Section 2: Edge must be at least 2x the round-trip cost."""
    if estimated_round_trip_cost_bps <= 0:
        return GateCheck(
            name="executable_alpha",
            result=GateResult.PASS,
            message="Zero estimated cost",
        )

    ratio = expected_edge_bps / estimated_round_trip_cost_bps
    passed = ratio >= MIN_EDGE_TO_COST_RATIO

    return GateCheck(
        name="executable_alpha",
        result=GateResult.PASS if passed else GateResult.FAIL,
        message=f"Edge/cost ratio: {ratio:.1f}x (need {MIN_EDGE_TO_COST_RATIO}x)",
        value=ratio,
        threshold=MIN_EDGE_TO_COST_RATIO,
    )


def check_statistical_bar(
    trades_per_window: list[int],
    regime_labels_seen: int,
    pnl_by_window: list[float],
) -> GateCheck:
    """Section 3: Minimum sample size, regime diversity, PnL distribution."""
    issues: list[str] = []

    # Minimum trades per window
    for i, count in enumerate(trades_per_window):
        if count < MIN_TRADES_PER_WINDOW:
            issues.append(f"Window {i}: {count} trades (need {MIN_TRADES_PER_WINDOW})")

    # Total trades for low-frequency
    total = sum(trades_per_window)
    if total < MIN_TOTAL_TRADES_LOW_FREQ:
        issues.append(f"Total trades: {total} (need {MIN_TOTAL_TRADES_LOW_FREQ})")

    # Regime diversity
    if regime_labels_seen < MIN_REGIME_LABELS:
        issues.append(f"Regime labels: {regime_labels_seen} (need {MIN_REGIME_LABELS})")

    # PnL concentration
    if pnl_by_window:
        total_pnl = sum(abs(p) for p in pnl_by_window)
        if total_pnl > 0:
            for i, pnl in enumerate(pnl_by_window):
                fraction = abs(pnl) / total_pnl
                if fraction > MAX_SINGLE_WINDOW_PNL_FRACTION:
                    issues.append(
                        f"Window {i} accounts for {fraction:.0%} of PnL "
                        f"(max {MAX_SINGLE_WINDOW_PNL_FRACTION:.0%})"
                    )

    if issues:
        return GateCheck(
            name="statistical_bar",
            result=GateResult.FAIL,
            message="; ".join(issues),
        )

    return GateCheck(
        name="statistical_bar",
        result=GateResult.PASS,
        message=(
            f"{total} trades across {len(trades_per_window)} windows, "
            f"{regime_labels_seen} regimes"
        ),
    )


def check_capacity(
    order_value: float,
    average_daily_volume_dollars: float,
) -> GateCheck:
    """Section 4: ADV participation must not exceed 1%."""
    if average_daily_volume_dollars <= 0:
        return GateCheck(
            name="capacity",
            result=GateResult.FAIL,
            message="No volume data available",
        )

    participation = order_value / average_daily_volume_dollars
    passed = participation <= MAX_ADV_PARTICIPATION

    return GateCheck(
        name="capacity",
        result=GateResult.PASS if passed else GateResult.FAIL,
        message=f"ADV participation: {participation:.2%} (max {MAX_ADV_PARTICIPATION:.0%})",
        value=participation,
        threshold=MAX_ADV_PARTICIPATION,
    )


def check_simplicity_tax(
    sharpe_with_component: float,
    sharpe_without_component: float,
) -> GateCheck:
    """Section 7: Component must improve Sharpe by at least 0.1."""
    improvement = sharpe_with_component - sharpe_without_component
    passed = improvement >= MIN_INCREMENTAL_SHARPE

    return GateCheck(
        name="simplicity_tax",
        result=GateResult.PASS if passed else GateResult.FAIL,
        message=f"Sharpe improvement: {improvement:+.3f} (need +{MIN_INCREMENTAL_SHARPE})",
        value=improvement,
        threshold=MIN_INCREMENTAL_SHARPE,
    )


def check_paper_live_correlation(
    correlation: float,
) -> GateCheck:
    """Section 10: Paper/live correlation must exceed 0.8."""
    passed = correlation >= MIN_PAPER_LIVE_CORRELATION

    return GateCheck(
        name="paper_live_correlation",
        result=GateResult.PASS if passed else GateResult.FAIL,
        message=f"Paper/live correlation: {correlation:.3f} (need {MIN_PAPER_LIVE_CORRELATION})",
        value=correlation,
        threshold=MIN_PAPER_LIVE_CORRELATION,
    )


def check_scope_constraints(
    active_strategies: int,
    active_signal_sources: int,
    universe: str,
) -> GateCheck:
    """Section 6: Enforce narrow initial scope."""
    issues: list[str] = []

    if active_strategies > MAX_ACTIVE_STRATEGIES:
        issues.append(f"{active_strategies} active strategies (max {MAX_ACTIVE_STRATEGIES})")

    if active_signal_sources > MAX_ACTIVE_SIGNAL_SOURCES:
        issues.append(f"{active_signal_sources} active sources (max {MAX_ACTIVE_SIGNAL_SOURCES})")

    if universe != ALLOWED_UNIVERSE:
        issues.append(f"Universe '{universe}' not allowed (must be '{ALLOWED_UNIVERSE}')")

    if issues:
        return GateCheck(
            name="scope_constraints",
            result=GateResult.FAIL,
            message="; ".join(issues),
        )

    return GateCheck(
        name="scope_constraints",
        result=GateResult.PASS,
        message=f"{active_strategies} strategies, {active_signal_sources} sources, {universe}",
    )


def run_promotion_gate(
    strategy_sharpe_by_window: list[float],
    baseline_sharpe_by_window: list[float],
    expected_edge_bps: float,
    estimated_cost_bps: float,
    trades_per_window: list[int],
    regime_labels_seen: int,
    pnl_by_window: list[float],
) -> GateReport:
    """Run all promotion checks for a strategy.

    This is the single entry point for strategy promotion decisions.
    A strategy must pass ALL checks to be promoted.
    """
    report = GateReport()

    report.checks.append(
        check_baseline_beating(strategy_sharpe_by_window, baseline_sharpe_by_window)
    )
    report.checks.append(check_executable_alpha(expected_edge_bps, estimated_cost_bps))
    report.checks.append(
        check_statistical_bar(trades_per_window, regime_labels_seen, pnl_by_window)
    )

    return report
