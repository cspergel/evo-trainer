"""Stochastic fitness evaluation for trading strategies.

Compares return distributions, not single numbers. Penalizes complexity.
A strategy with Sharpe 1.2 +/- 0.8 is less fit than one with 0.9 +/- 0.2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class FitnessResult:
    """Distributional fitness of a strategy across multiple evaluations."""

    sharpe: float  # mean Sharpe ratio across evaluations
    sharpe_std: float  # standard deviation of Sharpe across evaluations
    max_drawdown: float  # worst max drawdown across evaluations
    n_evaluations: int  # number of walk-forward windows evaluated


def compare_strategy_fitness(a: FitnessResult, b: FitnessResult) -> float:
    """Compare two strategies on distributional fitness.

    Returns positive if `a` is fitter, negative if `b` is fitter, ~0 if tied.
    Uses a conservative comparison: mean minus 1 standard deviation.
    This penalizes high-variance strategies.
    """
    a_conservative = a.sharpe - a.sharpe_std
    b_conservative = b.sharpe - b.sharpe_std

    # Also penalize drawdown (lower is better)
    a_score = a_conservative - a.max_drawdown
    b_score = b_conservative - b.max_drawdown

    return a_score - b_score


# Common stock tickers for complexity detection
_TICKER_PATTERN = re.compile(
    r"\b(AAPL|MSFT|GOOGL|GOOG|AMZN|NVDA|META|TSLA|BRK|JPM|V|JNJ|WMT|"
    r"PG|MA|UNH|HD|DIS|BAC|XOM|PFE|KO|PEP|ABBV|COST|AVGO|TMO|MRK|"
    r"CVX|ADBE|CRM|ACN|NFLX|AMD|INTC|QCOM|TXN|CSCO|ORCL|IBM)\b"
)

_DATE_PATTERN = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2}.*?(20\d{2}|19\d{2})\b",
    re.IGNORECASE,
)


def compute_complexity_penalty(skill_text: str) -> float:
    """Compute a complexity penalty for a strategy skill.

    Penalizes:
    - References to specific tickers (overfitting to particular stocks)
    - Narrow date ranges (overfitting to particular time periods)

    Returns 0.0 (no penalty) to 1.0 (maximum penalty).
    """
    penalty = 0.0

    ticker_matches = _TICKER_PATTERN.findall(skill_text)
    if ticker_matches:
        penalty += min(0.5, len(ticker_matches) * 0.1)

    date_matches = _DATE_PATTERN.findall(skill_text)
    if date_matches:
        penalty += min(0.5, len(date_matches) * 0.2)

    return min(1.0, penalty)
