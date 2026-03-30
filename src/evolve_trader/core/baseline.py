"""BaselineComparator — computes benchmark returns for strategy evaluation.

Per profitability contract section 1: nothing ships to live unless it beats
the baseline after costs. Supports multiple benchmark types matched to
strategy exposure profile.

Uses yfinance for historical price data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum


class BenchmarkType(Enum):
    """Benchmark types matched to strategy exposure profiles."""

    SPY = "spy"  # Directional long-only
    SECTOR_ETF = "sector_etf"  # Sector-relative
    CASH = "cash"  # Low-net or defensive


@dataclass
class BaselineResult:
    """Benchmark performance over a period."""

    benchmark_type: BenchmarkType
    ticker: str
    start_date: date
    end_date: date
    total_return: float  # Fractional (e.g., 0.05 = 5%)
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float


def compute_baseline(
    benchmark: BenchmarkType = BenchmarkType.SPY,
    start_date: date | None = None,
    end_date: date | None = None,
    sector_etf: str = "SPY",
    risk_free_rate: float = 0.05,
) -> BaselineResult:
    """Compute benchmark returns for a given period.

    Uses yfinance for price data. Falls back to synthetic returns
    if yfinance is unavailable or data fetch fails.
    """
    end = end_date or date.today()
    start = start_date or (end - timedelta(days=365))

    ticker = "SPY"
    if benchmark == BenchmarkType.SECTOR_ETF:
        ticker = sector_etf
    elif benchmark == BenchmarkType.CASH:
        # Cash benchmark: risk-free rate
        days = (end - start).days
        annual_return = risk_free_rate
        total_return = annual_return * days / 365
        return BaselineResult(
            benchmark_type=benchmark,
            ticker="CASH",
            start_date=start,
            end_date=end,
            total_return=total_return,
            annualized_return=annual_return,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
        )

    # Try yfinance, fall back to historical average
    try:
        return _compute_from_yfinance(ticker, start, end, benchmark, risk_free_rate)
    except Exception:
        return _compute_synthetic(ticker, start, end, benchmark, risk_free_rate)


def _compute_from_yfinance(
    ticker: str,
    start: date,
    end: date,
    benchmark: BenchmarkType,
    risk_free_rate: float,
) -> BaselineResult:
    """Compute baseline from yfinance price data."""
    import numpy as np
    import yfinance as yf

    data = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)
    if data.empty:
        raise ValueError(f"No data for {ticker}")

    prices = data["Close"].values.flatten()
    if len(prices) < 2:
        raise ValueError(f"Insufficient data for {ticker}")

    # Daily returns
    daily_returns = np.diff(prices) / prices[:-1]

    # Total return
    total_return = float((prices[-1] / prices[0]) - 1)

    # Annualized return
    trading_days = len(daily_returns)
    annualized = float((1 + total_return) ** (252 / max(trading_days, 1)) - 1)

    # Sharpe ratio
    excess = daily_returns - (risk_free_rate / 252)
    std = float(np.std(excess, ddof=1))
    sharpe = float(np.mean(excess) / std * np.sqrt(252)) if std > 1e-12 else 0.0

    # Max drawdown
    cumulative = np.cumprod(1 + daily_returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / np.where(peak == 0, 1, peak)
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    return BaselineResult(
        benchmark_type=benchmark,
        ticker=ticker,
        start_date=start,
        end_date=end,
        total_return=total_return,
        annualized_return=annualized,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
    )


def _compute_synthetic(
    ticker: str,
    start: date,
    end: date,
    benchmark: BenchmarkType,
    risk_free_rate: float,
) -> BaselineResult:
    """Synthetic baseline when yfinance is unavailable.

    Uses long-term SPY averages: ~10% annualized return, ~15% volatility.
    """
    days = (end - start).days
    annual_return = 0.10  # Historical SPY average
    total_return = annual_return * days / 365

    # Historical SPY Sharpe ~ 0.6-0.7
    sharpe = 0.65

    return BaselineResult(
        benchmark_type=benchmark,
        ticker=ticker,
        start_date=start,
        end_date=end,
        total_return=total_return,
        annualized_return=annual_return,
        sharpe_ratio=sharpe,
        max_drawdown=0.15,
    )
