"""Real market data via yfinance.

Replaces synthetic trade generation with actual historical prices.
Strategies are evaluated against real S&P 500 stock movements.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np


@dataclass
class PriceBar:
    """A single OHLCV bar."""

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class PriceSeries:
    """Historical price data for one ticker."""

    ticker: str
    bars: list[PriceBar]

    @property
    def closes(self) -> list[float]:
        return [b.close for b in self.bars]

    @property
    def daily_returns(self) -> list[float]:
        closes = self.closes
        if len(closes) < 2:
            return []
        arr = np.array(closes)
        return list((arr[1:] - arr[:-1]) / arr[:-1])

    @property
    def dates(self) -> list[date]:
        return [b.date for b in self.bars]


def fetch_prices(
    ticker: str,
    start: date | None = None,
    end: date | None = None,
    period_days: int = 252,
) -> PriceSeries:
    """Fetch historical daily prices from yfinance.

    Args:
        ticker: Stock symbol (e.g., "AAPL")
        start: Start date. Defaults to period_days ago.
        end: End date. Defaults to today.
        period_days: Number of calendar days if start not specified.
    """
    import yfinance as yf

    end_date = end or date.today()
    start_date = start or (end_date - timedelta(days=period_days))

    data = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        progress=False,
    )

    if data.empty:
        return PriceSeries(ticker=ticker, bars=[])

    def _val(cell: object) -> float:
        """Extract float from a yfinance cell (may be scalar or Series)."""
        if hasattr(cell, "iloc"):
            return float(cell.iloc[0])  # type: ignore[union-attr]
        return float(cell)  # type: ignore[arg-type]

    bars: list[PriceBar] = []
    for idx, row in data.iterrows():
        bar_date = idx.date() if hasattr(idx, "date") else idx  # type: ignore[union-attr]
        bars.append(
            PriceBar(
                date=bar_date,
                open=_val(row["Open"]),
                high=_val(row["High"]),
                low=_val(row["Low"]),
                close=_val(row["Close"]),
                volume=_val(row["Volume"]),
            )
        )

    return PriceSeries(ticker=ticker, bars=bars)


# S&P 500 large-cap tickers for initial scope
SP500_SAMPLE = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "BRK-B",
    "JPM",
    "V",
    "UNH",
    "XOM",
    "JNJ",
    "PG",
    "MA",
    "HD",
    "COST",
    "ABBV",
    "MRK",
    "PFE",
    "KO",
]


def fetch_universe(
    tickers: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> dict[str, PriceSeries]:
    """Fetch prices for a universe of tickers.

    Defaults to a sample of S&P 500 large-caps.
    """
    tickers = tickers or SP500_SAMPLE[:10]
    universe: dict[str, PriceSeries] = {}
    for ticker in tickers:
        try:
            series = fetch_prices(ticker, start=start, end=end)
            if series.bars:
                universe[ticker] = series
        except Exception:
            continue
    return universe
