"""Post-execution financial analyzer for trading strategies.

Replaces OpenSpace's binary success/failure evaluation with
distributional financial metrics: Sharpe, drawdown, win rate,
and full return distribution analysis.

Note: compute_sharpe_ratio currently receives per-trade returns,
not daily portfolio returns. The annualization is approximate until
the replay harness (Task 3) produces actual daily return series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TradeResult:
    """A single completed trade."""

    ticker: str
    entry_price: float
    exit_price: float
    shares: float
    entry_date: str
    exit_date: str
    reasoning: str = ""

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def return_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price


@dataclass
class StrategyPerformance:
    """Full distributional performance analysis of a strategy."""

    total_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    mean_return: float
    variance: float
    skewness: float
    kurtosis: float
    total_pnl: float
    tail_risk_5pct: float


@dataclass
class FailureAnalysis:
    """Diagnosis of why a strategy underperformed."""

    component: str  # "entry_logic", "exit_logic", "regime_mismatch", "sizing"
    description: str
    severity: float  # 0-1
    suggested_fix: str


def compute_sharpe_ratio(
    daily_returns: list[float],
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252.0,
) -> float:
    """Compute annualized Sharpe ratio from daily returns.

    Uses sample standard deviation (ddof=1) per financial convention.
    """
    if len(daily_returns) < 2:
        return 0.0
    arr = np.array(daily_returns)
    excess = arr - (risk_free_rate / annualization_factor)
    std = float(np.std(excess, ddof=1))
    if std < 1e-12:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(annualization_factor))


def compute_max_drawdown(equity_curve: list[float]) -> float:
    """Compute maximum drawdown from an equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    arr = np.array(equity_curve)
    peak = np.maximum.accumulate(arr)
    drawdown = (peak - arr) / np.where(peak == 0, 1, peak)
    return float(np.max(drawdown))


def analyze_strategy_performance(
    trades: list[TradeResult],
    initial_capital: float,
) -> StrategyPerformance:
    """Compute full distributional performance metrics for a strategy."""
    if not trades:
        return StrategyPerformance(
            total_trades=0,
            win_rate=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            mean_return=0.0,
            variance=0.0,
            skewness=0.0,
            kurtosis=0.0,
            total_pnl=0.0,
            tail_risk_5pct=0.0,
        )

    returns = [t.return_pct for t in trades]
    pnls = [t.pnl for t in trades]
    wins = sum(1 for r in returns if r > 0)

    equity = [initial_capital]
    for pnl in pnls:
        equity.append(equity[-1] + pnl)

    arr = np.array(returns)

    return StrategyPerformance(
        total_trades=len(trades),
        win_rate=wins / len(trades),
        sharpe_ratio=compute_sharpe_ratio(returns),
        max_drawdown=compute_max_drawdown(equity),
        mean_return=float(np.mean(arr)),
        variance=float(np.var(arr)),
        skewness=float(stats.skew(arr)) if len(arr) > 2 else 0.0,
        kurtosis=float(stats.kurtosis(arr)) if len(arr) > 3 else 0.0,
        total_pnl=sum(pnls),
        tail_risk_5pct=float(np.percentile(arr, 5)) if len(arr) > 0 else 0.0,
    )


def analyze_failure_mode(trades: list[TradeResult]) -> FailureAnalysis | None:
    """Trace back through trades to identify the primary failure component.

    Simple heuristic for Phase 1. Will be enhanced with LLM reasoning
    and regime_mismatch/sizing failure modes in later phases.
    """
    if not trades:
        return None

    losing_trades = [t for t in trades if t.pnl < 0]
    if not losing_trades:
        return None

    avg_loss_pct = float(np.mean([t.return_pct for t in losing_trades]))

    if avg_loss_pct < -0.05:
        return FailureAnalysis(
            component="entry_logic",
            description=(
                f"Average losing trade lost {avg_loss_pct:.1%}. "
                "Large losses suggest poor entry timing."
            ),
            severity=min(1.0, abs(avg_loss_pct) / 0.10),
            suggested_fix="Add confirmation signal before entry or tighten stop-loss.",
        )

    return FailureAnalysis(
        component="exit_logic",
        description=(
            f"Average losing trade lost {avg_loss_pct:.1%}. "
            "Moderate losses suggest exit timing could improve."
        ),
        severity=min(1.0, abs(avg_loss_pct) / 0.10),
        suggested_fix="Consider trailing stop or time-based exit.",
    )
