"""Alpaca paper trading client.

Wraps alpaca-py SDK for paper order submission, position tracking,
and account reconciliation. Market-session aware.

Per profitability contract: all paper trades are logged for
paper/live deviation tracking.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass
class OrderResult:
    """Result of an order submission."""

    order_id: str
    ticker: str
    side: OrderSide
    quantity: float
    status: OrderStatus
    filled_price: float | None = None
    filled_at: datetime | None = None
    message: str = ""


@dataclass
class Position:
    """A current portfolio position."""

    ticker: str
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


@dataclass
class AccountInfo:
    """Alpaca account summary."""

    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    positions_count: int


class AlpacaPaperClient:
    """Paper trading client wrapping alpaca-py.

    Uses paper-api.alpaca.markets. Never connects to live endpoint.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self._client: Any = None

    def _get_client(self) -> Any:  # noqa: ANN401
        """Lazy init of Alpaca TradingClient."""
        if self._client is None:
            from alpaca.trading.client import TradingClient

            self._client = TradingClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
                paper=True,  # ALWAYS paper
            )
        return self._client

    def get_account(self) -> AccountInfo:
        """Get current account info."""
        client = self._get_client()
        account = client.get_account()
        return AccountInfo(
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            portfolio_value=float(account.portfolio_value),
            positions_count=len(client.get_all_positions()),
        )

    def get_positions(self) -> list[Position]:
        """Get all current positions."""
        client = self._get_client()
        positions = client.get_all_positions()
        return [
            Position(
                ticker=str(p.symbol),
                quantity=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                market_value=float(p.market_value),
                unrealized_pnl=float(p.unrealized_pl),
                unrealized_pnl_pct=float(p.unrealized_plpc),
            )
            for p in positions
        ]

    def submit_market_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: float,
    ) -> OrderResult:
        """Submit a market order.

        Per profitability contract: only S&P 500 large-cap in initial scope.
        """
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        client = self._get_client()

        request = MarketOrderRequest(
            symbol=ticker,
            qty=quantity,
            side=AlpacaSide.BUY if side == OrderSide.BUY else AlpacaSide.SELL,
            time_in_force=TimeInForce.DAY,
        )

        try:
            order = client.submit_order(request)
            return OrderResult(
                order_id=str(order.id),
                ticker=ticker,
                side=side,
                quantity=quantity,
                status=OrderStatus.PENDING,
                message="Order submitted",
            )
        except Exception as e:
            return OrderResult(
                order_id="",
                ticker=ticker,
                side=side,
                quantity=quantity,
                status=OrderStatus.REJECTED,
                message=str(e),
            )

    def get_order_status(self, order_id: str) -> OrderResult:
        """Check the status of a submitted order."""
        client = self._get_client()
        order = client.get_order_by_id(order_id)
        return OrderResult(
            order_id=str(order.id),
            ticker=str(order.symbol),
            side=OrderSide.BUY if str(order.side) == "buy" else OrderSide.SELL,
            quantity=float(order.qty),
            status=_map_status(str(order.status)),
            filled_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            filled_at=order.filled_at,
        )

    def is_market_open(self) -> bool:
        """Check if US equity market is currently open."""
        client = self._get_client()
        clock = client.get_clock()
        return bool(clock.is_open)


def _map_status(alpaca_status: str) -> OrderStatus:
    """Map Alpaca order status string to our enum."""
    mapping = {
        "new": OrderStatus.PENDING,
        "accepted": OrderStatus.PENDING,
        "pending_new": OrderStatus.PENDING,
        "filled": OrderStatus.FILLED,
        "partially_filled": OrderStatus.PARTIALLY_FILLED,
        "canceled": OrderStatus.CANCELED,
        "expired": OrderStatus.CANCELED,
        "rejected": OrderStatus.REJECTED,
    }
    return mapping.get(alpaca_status, OrderStatus.PENDING)
