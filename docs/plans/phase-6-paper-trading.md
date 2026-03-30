# Phase 6: Paper Trading & Notifications — Detailed Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Connect to Alpaca paper trading. Implement TradeIntent with 3 execution gates (immutable risk, paper shadow, graduated approval). Build the single shared promotion protocol with 5 stages and bidirectional demotion. Trade notifications with approval workflow across Slack, Telegram, and email.

**Architecture:** Every trade begins as a TradeIntent dataclass capturing full provenance (strategy skill, lineage, regime, signals, confidence, structured rationale summary, and evidence). The intent passes through three sequential gates: Gate 1 (immutable risk constraints) blocks violations outright, Gate 2 (paper shadow) always executes on paper for counterfactual tracking, and Gate 3 (graduated approval) routes to manual or auto-approval based on track record. A single promotion protocol governs the transition from paper-only to full live trading across 5 stages and is reused unchanged by Phase 11 rather than redefined there. A multi-channel notification dispatcher formats and delivers TradeIntent details with interactive approval workflows. Order idempotency, market-session awareness, and broker reconciliation are mandatory parts of the execution layer.

**Tech Stack:** Python 3.11+, alpaca-py, WebSocket, httpx, pytest

**Parent plan:** [`2026-03-29-evolve-trader-master-plan.md`](2026-03-29-evolve-trader-master-plan.md)

**Prerequisites:** Phase 5 complete. Portfolio construction, position sizing, regime-aware allocation, and rebalancing engine all verified.

---

## Task 1: Alpaca Paper Trading Client

**Files:**
- Create: `src/evolve_trader/execution/alpaca_client.py`
- Create: `src/evolve_trader/execution/order_mapper.py`
- Create: `tests/unit/test_alpaca_client.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_alpaca_client.py
"""Tests for Alpaca paper trading client and order mapping."""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal

from evolve_trader.execution.alpaca_client import (
    AlpacaPaperClient,
    AlpacaConfig,
    AlpacaPosition,
    AlpacaOrder,
    OrderStatus,
    ConnectionState,
)
from evolve_trader.execution.order_mapper import (
    OrderMapper,
    OrderRequest,
    OrderType,
    OrderSide,
    TimeInForce,
)


# ─── AlpacaConfig ───────────────────────────────────────────────────


def test_alpaca_config_defaults_to_paper():
    """Config defaults to paper endpoint, never live."""
    config = AlpacaConfig(api_key="test-key", secret_key="test-secret")
    assert config.base_url == "https://paper-api.alpaca.markets"
    assert config.is_paper is True


def test_alpaca_config_rejects_live_endpoint():
    """Config raises if someone tries to pass a live URL."""
    with pytest.raises(ValueError, match="paper"):
        AlpacaConfig(
            api_key="test-key",
            secret_key="test-secret",
            base_url="https://api.alpaca.markets",
            is_paper=False,
        )


def test_alpaca_config_from_env(monkeypatch):
    """Config can load from environment variables."""
    monkeypatch.setenv("ALPACA_API_KEY", "env-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "env-secret")
    config = AlpacaConfig.from_env()
    assert config.api_key == "env-key"
    assert config.secret_key == "env-secret"
    assert config.is_paper is True


# ─── AlpacaPaperClient ──────────────────────────────────────────────


@pytest.fixture
def mock_config():
    return AlpacaConfig(api_key="test-key", secret_key="test-secret")


@pytest.fixture
def client(mock_config):
    return AlpacaPaperClient(config=mock_config)


def test_client_initial_state(client):
    """Client starts disconnected."""
    assert client.connection_state == ConnectionState.DISCONNECTED
    assert client.positions == {}
    assert client.account_equity is None


@pytest.mark.asyncio
async def test_client_connect(client):
    """Client connects to Alpaca paper endpoint."""
    with patch.object(client, "_trading_client", new_callable=MagicMock):
        with patch.object(client, "_init_trading_client"):
            await client.connect()
            assert client.connection_state == ConnectionState.CONNECTED


@pytest.mark.asyncio
async def test_client_submit_order(client):
    """Client submits an order and returns an AlpacaOrder."""
    mock_alpaca_order = MagicMock()
    mock_alpaca_order.id = "order-123"
    mock_alpaca_order.status = "accepted"
    mock_alpaca_order.filled_qty = "0"
    mock_alpaca_order.filled_avg_price = None
    mock_alpaca_order.submitted_at = datetime.now(timezone.utc)
    mock_alpaca_order.symbol = "AAPL"
    mock_alpaca_order.side = "buy"
    mock_alpaca_order.qty = "10"
    mock_alpaca_order.type = "market"

    client._trading_client = MagicMock()
    client._trading_client.submit_order = MagicMock(return_value=mock_alpaca_order)
    client._connection_state = ConnectionState.CONNECTED

    request = OrderRequest(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=10.0,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
    )

    order = await client.submit_order(request)
    assert order.order_id == "order-123"
    assert order.status == OrderStatus.ACCEPTED
    assert order.symbol == "AAPL"


@pytest.mark.asyncio
async def test_client_rejects_order_when_disconnected(client):
    """Client raises if not connected."""
    request = OrderRequest(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=10.0,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
    )
    with pytest.raises(ConnectionError, match="not connected"):
        await client.submit_order(request)


@pytest.mark.asyncio
async def test_client_get_positions(client):
    """Client fetches current positions."""
    mock_position = MagicMock()
    mock_position.symbol = "AAPL"
    mock_position.qty = "10"
    mock_position.avg_entry_price = "150.00"
    mock_position.current_price = "155.00"
    mock_position.unrealized_pl = "50.00"
    mock_position.market_value = "1550.00"
    mock_position.side = "long"

    client._trading_client = MagicMock()
    client._trading_client.get_all_positions = MagicMock(return_value=[mock_position])
    client._connection_state = ConnectionState.CONNECTED

    positions = await client.get_positions()
    assert "AAPL" in positions
    assert positions["AAPL"].qty == 10.0
    assert positions["AAPL"].unrealized_pl == 50.0


@pytest.mark.asyncio
async def test_client_get_account_equity(client):
    """Client fetches account equity."""
    mock_account = MagicMock()
    mock_account.equity = "105000.00"
    mock_account.cash = "50000.00"
    mock_account.buying_power = "100000.00"

    client._trading_client = MagicMock()
    client._trading_client.get_account = MagicMock(return_value=mock_account)
    client._connection_state = ConnectionState.CONNECTED

    equity = await client.get_account_equity()
    assert equity == 105000.0


def test_client_supports_fractional_shares(client):
    """Client configuration allows fractional shares."""
    assert client.supports_fractional is True


def test_client_supports_extended_hours(client):
    """Client configuration allows extended hours trading."""
    assert client.supports_extended_hours is True


# ─── OrderMapper ─────────────────────────────────────────────────────


def test_order_mapper_market_order():
    """Maps a simple market buy to OrderRequest."""
    mapper = OrderMapper()
    request = mapper.create_market_order("AAPL", OrderSide.BUY, 10.0)
    assert request.symbol == "AAPL"
    assert request.side == OrderSide.BUY
    assert request.qty == 10.0
    assert request.order_type == OrderType.MARKET
    assert request.time_in_force == TimeInForce.DAY


def test_order_mapper_limit_order():
    """Maps a limit order with price."""
    mapper = OrderMapper()
    request = mapper.create_limit_order("AAPL", OrderSide.BUY, 10.0, limit_price=150.0)
    assert request.order_type == OrderType.LIMIT
    assert request.limit_price == 150.0


def test_order_mapper_stop_loss_order():
    """Maps a stop-loss order."""
    mapper = OrderMapper()
    request = mapper.create_stop_order("AAPL", OrderSide.SELL, 10.0, stop_price=140.0)
    assert request.order_type == OrderType.STOP
    assert request.stop_price == 140.0


def test_order_mapper_fractional_shares():
    """Maps fractional share quantities."""
    mapper = OrderMapper()
    request = mapper.create_market_order("AAPL", OrderSide.BUY, 0.5)
    assert request.qty == 0.5
    assert request.is_fractional is True


def test_order_mapper_extended_hours():
    """Maps extended hours order."""
    mapper = OrderMapper()
    request = mapper.create_limit_order(
        "AAPL", OrderSide.BUY, 10.0, limit_price=150.0, extended_hours=True,
    )
    assert request.extended_hours is True
    assert request.time_in_force == TimeInForce.DAY


def test_order_mapper_stop_limit_order():
    """Maps a stop-limit order with both prices."""
    mapper = OrderMapper()
    request = mapper.create_stop_limit_order(
        "AAPL", OrderSide.SELL, 10.0, stop_price=140.0, limit_price=139.0,
    )
    assert request.order_type == OrderType.STOP_LIMIT
    assert request.stop_price == 140.0
    assert request.limit_price == 139.0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_alpaca_client.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.execution.alpaca_client'`

**Step 3: Implement the Alpaca client and order mapper**

```python
# src/evolve_trader/execution/order_mapper.py
"""Order mapping utilities for translating trade intents to broker orders."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class OrderType(str, Enum):
    """Supported order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(str, Enum):
    """Order direction."""
    BUY = "buy"
    SELL = "sell"


class TimeInForce(str, Enum):
    """Order time-in-force policy."""
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    OPG = "opg"
    CLS = "cls"


@dataclass(frozen=True)
class OrderRequest:
    """Broker-agnostic order request."""
    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: float | None = None
    stop_price: float | None = None
    trail_percent: float | None = None
    extended_hours: bool = False
    client_order_id: str | None = None

    @property
    def is_fractional(self) -> bool:
        """True if quantity is not a whole number."""
        return self.qty != int(self.qty)


class OrderMapper:
    """Maps trade parameters to OrderRequest objects."""

    def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        time_in_force: TimeInForce = TimeInForce.DAY,
        client_order_id: str | None = None,
    ) -> OrderRequest:
        """Create a market order request."""
        return OrderRequest(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=OrderType.MARKET,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        )

    def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        limit_price: float,
        time_in_force: TimeInForce = TimeInForce.DAY,
        extended_hours: bool = False,
        client_order_id: str | None = None,
    ) -> OrderRequest:
        """Create a limit order request."""
        return OrderRequest(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=OrderType.LIMIT,
            time_in_force=time_in_force,
            limit_price=limit_price,
            extended_hours=extended_hours,
            client_order_id=client_order_id,
        )

    def create_stop_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        stop_price: float,
        time_in_force: TimeInForce = TimeInForce.DAY,
        client_order_id: str | None = None,
    ) -> OrderRequest:
        """Create a stop order request."""
        return OrderRequest(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=OrderType.STOP,
            time_in_force=time_in_force,
            stop_price=stop_price,
            client_order_id=client_order_id,
        )

    def create_stop_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        stop_price: float,
        limit_price: float,
        time_in_force: TimeInForce = TimeInForce.DAY,
        client_order_id: str | None = None,
    ) -> OrderRequest:
        """Create a stop-limit order request."""
        return OrderRequest(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=OrderType.STOP_LIMIT,
            time_in_force=time_in_force,
            stop_price=stop_price,
            limit_price=limit_price,
            client_order_id=client_order_id,
        )
```

```python
# src/evolve_trader/execution/alpaca_client.py
"""Alpaca paper trading client with WebSocket position/P&L streaming."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from typing import Any

from evolve_trader.execution.order_mapper import OrderRequest, OrderSide, OrderType


PAPER_URL = "https://paper-api.alpaca.markets"


class ConnectionState(str, Enum):
    """Client connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class OrderStatus(str, Enum):
    """Order lifecycle status."""
    NEW = "new"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    PENDING_NEW = "pending_new"


@dataclass
class AlpacaConfig:
    """Configuration for Alpaca API connection."""
    api_key: str
    secret_key: str
    base_url: str = PAPER_URL
    is_paper: bool = True
    data_feed: str = "iex"  # "iex" (free) or "sip" (paid)
    max_retries: int = 3
    timeout_seconds: int = 30

    def __post_init__(self):
        if not self.is_paper or self.base_url != PAPER_URL:
            if "paper" not in self.base_url:
                raise ValueError(
                    "Only paper trading is supported. "
                    "base_url must contain 'paper' and is_paper must be True."
                )

    @classmethod
    def from_env(cls) -> AlpacaConfig:
        """Load configuration from environment variables."""
        return cls(
            api_key=os.environ["ALPACA_API_KEY"],
            secret_key=os.environ["ALPACA_SECRET_KEY"],
            base_url=os.environ.get("ALPACA_BASE_URL", PAPER_URL),
            is_paper=True,
        )


@dataclass
class AlpacaPosition:
    """Current position snapshot."""
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    unrealized_pl: float
    market_value: float
    side: str  # "long" or "short"

    @property
    def return_pct(self) -> float:
        if self.avg_entry_price == 0:
            return 0.0
        return (self.current_price - self.avg_entry_price) / self.avg_entry_price


@dataclass
class AlpacaOrder:
    """Order execution result."""
    order_id: str
    symbol: str
    side: str
    qty: float
    order_type: str
    status: OrderStatus
    filled_qty: float
    filled_avg_price: float | None
    submitted_at: datetime
    filled_at: datetime | None = None

    @property
    def is_complete(self) -> bool:
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )


class AlpacaPaperClient:
    """Paper trading client wrapping alpaca-py SDK.

    Provides:
    - Order submission (market, limit, stop, stop-limit)
    - Real-time position and P&L tracking via WebSocket
    - Fractional share support
    - Extended hours trading
    """

    def __init__(self, config: AlpacaConfig):
        self._config = config
        self._connection_state = ConnectionState.DISCONNECTED
        self._positions: dict[str, AlpacaPosition] = {}
        self._account_equity: float | None = None
        self._trading_client: Any = None
        self._stream_client: Any = None

    @property
    def connection_state(self) -> ConnectionState:
        return self._connection_state

    @property
    def positions(self) -> dict[str, AlpacaPosition]:
        return dict(self._positions)

    @property
    def account_equity(self) -> float | None:
        return self._account_equity

    @property
    def supports_fractional(self) -> bool:
        return True

    @property
    def supports_extended_hours(self) -> bool:
        return True

    def _init_trading_client(self):
        """Initialize the alpaca-py TradingClient."""
        from alpaca.trading.client import TradingClient

        self._trading_client = TradingClient(
            api_key=self._config.api_key,
            secret_key=self._config.secret_key,
            paper=self._config.is_paper,
        )

    async def connect(self) -> None:
        """Connect to Alpaca paper trading API."""
        self._connection_state = ConnectionState.CONNECTING
        try:
            self._init_trading_client()
            self._connection_state = ConnectionState.CONNECTED
        except Exception as e:
            self._connection_state = ConnectionState.ERROR
            raise ConnectionError(f"Failed to connect to Alpaca: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from Alpaca."""
        if self._stream_client:
            await self._stream_client.close()
        self._trading_client = None
        self._stream_client = None
        self._connection_state = ConnectionState.DISCONNECTED

    async def submit_order(self, request: OrderRequest) -> AlpacaOrder:
        """Submit an order to Alpaca paper trading.

        Args:
            request: Broker-agnostic order request.

        Returns:
            AlpacaOrder with execution details.

        Raises:
            ConnectionError: If client is not connected.
        """
        if self._connection_state != ConnectionState.CONNECTED:
            raise ConnectionError(
                "Client is not connected. Call connect() first."
            )

        from alpaca.trading.requests import (
            MarketOrderRequest,
            LimitOrderRequest,
            StopOrderRequest,
            StopLimitOrderRequest,
        )
        from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce as AlpacaTIF

        side = AlpacaSide.BUY if request.side.value == "buy" else AlpacaSide.SELL
        tif_map = {
            "day": AlpacaTIF.DAY,
            "gtc": AlpacaTIF.GTC,
            "ioc": AlpacaTIF.IOC,
            "fok": AlpacaTIF.FOK,
        }
        tif = tif_map.get(request.time_in_force.value, AlpacaTIF.DAY)

        if request.order_type.value == "market":
            alpaca_request = MarketOrderRequest(
                symbol=request.symbol, qty=request.qty, side=side, time_in_force=tif,
            )
        elif request.order_type.value == "limit":
            alpaca_request = LimitOrderRequest(
                symbol=request.symbol, qty=request.qty, side=side,
                time_in_force=tif, limit_price=request.limit_price,
                extended_hours=request.extended_hours,
            )
        elif request.order_type.value == "stop":
            alpaca_request = StopOrderRequest(
                symbol=request.symbol, qty=request.qty, side=side,
                time_in_force=tif, stop_price=request.stop_price,
            )
        elif request.order_type.value == "stop_limit":
            alpaca_request = StopLimitOrderRequest(
                symbol=request.symbol, qty=request.qty, side=side,
                time_in_force=tif, stop_price=request.stop_price,
                limit_price=request.limit_price,
            )
        else:
            raise ValueError(f"Unsupported order type: {request.order_type}")

        result = self._trading_client.submit_order(alpaca_request)
        return self._parse_order_result(result)

    async def get_positions(self) -> dict[str, AlpacaPosition]:
        """Fetch all current positions from Alpaca."""
        if self._connection_state != ConnectionState.CONNECTED:
            raise ConnectionError("Client is not connected.")

        raw_positions = self._trading_client.get_all_positions()
        self._positions = {}
        for p in raw_positions:
            pos = AlpacaPosition(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                unrealized_pl=float(p.unrealized_pl),
                market_value=float(p.market_value),
                side=str(p.side),
            )
            self._positions[pos.symbol] = pos
        return dict(self._positions)

    async def get_account_equity(self) -> float:
        """Fetch current account equity."""
        if self._connection_state != ConnectionState.CONNECTED:
            raise ConnectionError("Client is not connected.")

        account = self._trading_client.get_account()
        self._account_equity = float(account.equity)
        return self._account_equity

    async def get_order_status(self, order_id: str) -> AlpacaOrder:
        """Fetch the current status of an order."""
        if self._connection_state != ConnectionState.CONNECTED:
            raise ConnectionError("Client is not connected.")

        result = self._trading_client.get_order_by_id(order_id)
        return self._parse_order_result(result)

    async def cancel_order(self, order_id: str) -> None:
        """Cancel a pending order."""
        if self._connection_state != ConnectionState.CONNECTED:
            raise ConnectionError("Client is not connected.")

        self._trading_client.cancel_order_by_id(order_id)

    def _parse_order_result(self, result: Any) -> AlpacaOrder:
        """Parse alpaca-py order response into AlpacaOrder."""
        status_map = {
            "new": OrderStatus.NEW,
            "accepted": OrderStatus.ACCEPTED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED,
            "pending_new": OrderStatus.PENDING_NEW,
        }
        return AlpacaOrder(
            order_id=str(result.id),
            symbol=str(result.symbol),
            side=str(result.side),
            qty=float(result.qty),
            order_type=str(result.type),
            status=status_map.get(str(result.status), OrderStatus.NEW),
            filled_qty=float(result.filled_qty) if result.filled_qty else 0.0,
            filled_avg_price=(
                float(result.filled_avg_price) if result.filled_avg_price else None
            ),
            submitted_at=result.submitted_at,
        )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_alpaca_client.py -v
```

Expected: PASS (unit tests use mocks, no live Alpaca connection needed)

**Step 5: Commit**

```bash
git add src/evolve_trader/execution/alpaca_client.py src/evolve_trader/execution/order_mapper.py tests/unit/test_alpaca_client.py
git commit -m "feat: Alpaca paper trading client with order mapping and WebSocket support"
```

---

## Task 2: TradeIntent Object

**Files:**
- Create: `src/evolve_trader/execution/trade_intent.py`
- Create: `tests/unit/test_trade_intent.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_trade_intent.py
"""Tests for TradeIntent dataclass — full provenance capture."""
import pytest
from datetime import datetime, timezone
from evolve_trader.execution.trade_intent import (
    TradeIntent,
    TradeDirection,
    IntentStatus,
    GateResult,
    GateVerdict,
)


# ─── TradeIntent Construction ───────────────────────────────────────


def test_trade_intent_has_all_provenance_fields():
    """TradeIntent captures full lineage from signal to execution."""
    intent = TradeIntent(
        intent_id="intent-001",
        ticker="AAPL",
        direction=TradeDirection.BUY,
        quantity=10.0,
        order_type="market",
        strategy_skill="momentum-v3",
        strategy_lineage=["momentum-v1", "momentum-v2", "momentum-v3"],
        sizing_skill="kelly-fractional-v1",
        regime_label="risk-on",
        signal_sources=["edgar_13f", "form4_insider"],
        confidence=0.87,
        reasoning_chain=(
            "13F filing shows Buffett accumulating AAPL. "
            "Form 4 confirms insider purchases. "
            "Momentum indicators aligned with risk-on regime."
        ),
        position_impact={
            "current_weight": 0.0,
            "target_weight": 0.03,
            "sector": "Technology",
            "sector_exposure_after": 0.18,
        },
        paper_track_record={
            "paper_trades": 25,
            "paper_sharpe": 1.4,
            "paper_win_rate": 0.62,
            "paper_max_drawdown": 0.08,
        },
        created_at=datetime.now(timezone.utc),
    )
    assert intent.ticker == "AAPL"
    assert intent.direction == TradeDirection.BUY
    assert intent.strategy_skill == "momentum-v3"
    assert len(intent.strategy_lineage) == 3
    assert intent.confidence == 0.87
    assert intent.sizing_skill == "kelly-fractional-v1"
    assert "sector" in intent.position_impact


def test_trade_intent_default_status_is_pending():
    """New intents start as PENDING."""
    intent = TradeIntent(
        intent_id="intent-002",
        ticker="MSFT",
        direction=TradeDirection.BUY,
        quantity=5.0,
        order_type="limit",
        strategy_skill="mean-reversion-v1",
        strategy_lineage=["mean-reversion-v1"],
        sizing_skill="fixed-fractional-v1",
        regime_label="neutral",
        signal_sources=["congressional"],
        confidence=0.72,
        reasoning_chain="Congressional purchase signal with neutral regime.",
    )
    assert intent.status == IntentStatus.PENDING


def test_trade_intent_immutable_core_fields():
    """Core identification fields cannot be changed after creation."""
    intent = TradeIntent(
        intent_id="intent-003",
        ticker="GOOGL",
        direction=TradeDirection.SELL,
        quantity=3.0,
        order_type="market",
        strategy_skill="momentum-v1",
        strategy_lineage=["momentum-v1"],
        sizing_skill="equal-weight-v1",
        regime_label="risk-off",
        signal_sources=["edgar_13f"],
        confidence=0.65,
        reasoning_chain="Risk-off signal from 13F reduction.",
    )
    with pytest.raises(AttributeError):
        intent.intent_id = "changed"
    with pytest.raises(AttributeError):
        intent.ticker = "CHANGED"


# ─── GateResult ──────────────────────────────────────────────────────


def test_gate_result_pass():
    """GateResult records a passing gate check."""
    result = GateResult(
        gate_name="risk_constraints",
        verdict=GateVerdict.PASS,
        details={"position_pct": 0.03, "max_allowed": 0.05},
        checked_at=datetime.now(timezone.utc),
    )
    assert result.verdict == GateVerdict.PASS
    assert result.gate_name == "risk_constraints"


def test_gate_result_block():
    """GateResult records a blocked gate check with reason."""
    result = GateResult(
        gate_name="risk_constraints",
        verdict=GateVerdict.BLOCK,
        reason="Position would exceed 5% limit (requested 7.2%)",
        details={"position_pct": 0.072, "max_allowed": 0.05},
        checked_at=datetime.now(timezone.utc),
    )
    assert result.verdict == GateVerdict.BLOCK
    assert "5%" in result.reason


def test_gate_result_pending_approval():
    """GateResult records a pending approval state."""
    result = GateResult(
        gate_name="approval",
        verdict=GateVerdict.PENDING_APPROVAL,
        details={"approval_channel": "slack", "timeout_hours": 4},
        checked_at=datetime.now(timezone.utc),
    )
    assert result.verdict == GateVerdict.PENDING_APPROVAL


# ─── TradeIntent Gate Tracking ───────────────────────────────────────


def test_trade_intent_tracks_gate_results():
    """Intent accumulates gate results as it passes through pipeline."""
    intent = TradeIntent(
        intent_id="intent-004",
        ticker="NVDA",
        direction=TradeDirection.BUY,
        quantity=2.0,
        order_type="market",
        strategy_skill="momentum-v2",
        strategy_lineage=["momentum-v1", "momentum-v2"],
        sizing_skill="kelly-fractional-v1",
        regime_label="risk-on",
        signal_sources=["congressional"],
        confidence=0.91,
        reasoning_chain="Strong momentum in risk-on regime.",
    )

    gate1 = GateResult(
        gate_name="risk_constraints",
        verdict=GateVerdict.PASS,
        details={},
        checked_at=datetime.now(timezone.utc),
    )
    intent.add_gate_result(gate1)

    gate2 = GateResult(
        gate_name="paper_shadow",
        verdict=GateVerdict.PASS,
        details={"paper_order_id": "paper-123"},
        checked_at=datetime.now(timezone.utc),
    )
    intent.add_gate_result(gate2)

    assert len(intent.gate_results) == 2
    assert intent.gate_results[0].gate_name == "risk_constraints"
    assert intent.gate_results[1].gate_name == "paper_shadow"


def test_trade_intent_blocked_by_gate():
    """Intent status becomes BLOCKED when a gate blocks it."""
    intent = TradeIntent(
        intent_id="intent-005",
        ticker="TSLA",
        direction=TradeDirection.BUY,
        quantity=50.0,
        order_type="market",
        strategy_skill="momentum-v1",
        strategy_lineage=["momentum-v1"],
        sizing_skill="equal-weight-v1",
        regime_label="risk-on",
        signal_sources=["form4_insider"],
        confidence=0.78,
        reasoning_chain="Insider buying detected.",
    )

    gate1 = GateResult(
        gate_name="risk_constraints",
        verdict=GateVerdict.BLOCK,
        reason="Position exceeds 5% limit",
        details={},
        checked_at=datetime.now(timezone.utc),
    )
    intent.add_gate_result(gate1)

    assert intent.status == IntentStatus.BLOCKED
    assert intent.blocking_gate == "risk_constraints"


def test_trade_intent_to_dict():
    """Intent serializes to dict for notification formatting."""
    intent = TradeIntent(
        intent_id="intent-006",
        ticker="AMZN",
        direction=TradeDirection.BUY,
        quantity=5.0,
        order_type="market",
        strategy_skill="mean-reversion-v1",
        strategy_lineage=["mean-reversion-v1"],
        sizing_skill="fixed-fractional-v1",
        regime_label="neutral",
        signal_sources=["edgar_13f"],
        confidence=0.80,
        reasoning_chain="Mean reversion signal.",
    )
    d = intent.to_dict()
    assert d["intent_id"] == "intent-006"
    assert d["ticker"] == "AMZN"
    assert d["direction"] == "BUY"
    assert "gate_results" in d
    assert "created_at" in d
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_trade_intent.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.execution.trade_intent'`

**Step 3: Implement the TradeIntent object**

```python
# src/evolve_trader/execution/trade_intent.py
"""TradeIntent — full provenance dataclass for trade execution pipeline.

Every trade flowing through the system originates as a TradeIntent, capturing
the complete chain from signal sources through strategy selection, sizing,
regime context, and gate evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TradeDirection(str, Enum):
    """Trade direction."""
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"


class IntentStatus(str, Enum):
    """Lifecycle status of a TradeIntent."""
    PENDING = "PENDING"
    PASSED_GATES = "PASSED_GATES"
    BLOCKED = "BLOCKED"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTED_PAPER = "EXECUTED_PAPER"
    EXECUTED_LIVE = "EXECUTED_LIVE"
    EXPIRED = "EXPIRED"
    CANCELED = "CANCELED"


class GateVerdict(str, Enum):
    """Result of a single gate check."""
    PASS = "PASS"
    BLOCK = "BLOCK"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    SKIP = "SKIP"


@dataclass(frozen=True)
class GateResult:
    """Immutable record of a single gate evaluation."""
    gate_name: str
    verdict: GateVerdict
    checked_at: datetime
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeIntent:
    """Full-provenance trade intent flowing through execution gates.

    Core identification fields (intent_id, ticker, direction) are set at
    creation and must not change. Mutable state (status, gate_results) is
    updated as the intent moves through the gate pipeline.

    Attributes:
        intent_id: Unique identifier for this intent.
        ticker: Target security symbol.
        direction: BUY, SELL, SHORT, or COVER.
        quantity: Number of shares (supports fractional).
        order_type: market, limit, stop, stop_limit.
        strategy_skill: Name of the strategy skill that generated this intent.
        strategy_lineage: Full evolution chain of the strategy skill.
        sizing_skill: Name of the sizing skill used.
        regime_label: Current market regime label.
        signal_sources: List of signal source names that contributed.
        confidence: Composite confidence score [0, 1].
        reasoning_chain: Human-readable explanation of the trade thesis.
        position_impact: Expected impact on portfolio composition.
        paper_track_record: Historical paper trading stats for this strategy.
        created_at: Timestamp of intent creation.
        status: Current lifecycle status.
        gate_results: Ordered list of gate evaluation results.
    """

    # ── Immutable core fields (set __setattr__ guard below) ──
    intent_id: str
    ticker: str
    direction: TradeDirection
    quantity: float
    order_type: str

    # ── Strategy provenance ──
    strategy_skill: str
    strategy_lineage: list[str]
    sizing_skill: str
    regime_label: str
    signal_sources: list[str]
    confidence: float
    reasoning_chain: str

    # ── Optional enrichment ──
    position_impact: dict[str, Any] = field(default_factory=dict)
    paper_track_record: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Mutable pipeline state ──
    status: IntentStatus = field(default=IntentStatus.PENDING)
    gate_results: list[GateResult] = field(default_factory=list)
    blocking_gate: str | None = field(default=None)

    # Fields that must not be reassigned after __init__
    _IMMUTABLE_FIELDS: frozenset[str] = field(
        default=frozenset({"intent_id", "ticker", "direction", "quantity", "order_type"}),
        init=False,
        repr=False,
        compare=False,
    )

    def __setattr__(self, name: str, value: Any) -> None:
        # Allow setting during __init__ (gate_results won't exist yet)
        if (
            name != "_IMMUTABLE_FIELDS"
            and hasattr(self, "_IMMUTABLE_FIELDS")
            and name in self._IMMUTABLE_FIELDS
            and hasattr(self, name)
        ):
            raise AttributeError(
                f"Cannot modify immutable field '{name}' after creation."
            )
        super().__setattr__(name, value)

    def add_gate_result(self, result: GateResult) -> None:
        """Record a gate evaluation result and update status accordingly."""
        self.gate_results.append(result)

        if result.verdict == GateVerdict.BLOCK:
            self.status = IntentStatus.BLOCKED
            self.blocking_gate = result.gate_name
        elif result.verdict == GateVerdict.PENDING_APPROVAL:
            self.status = IntentStatus.PENDING_APPROVAL

    def all_gates_passed(self) -> bool:
        """True if all recorded gates passed."""
        return (
            len(self.gate_results) > 0
            and all(r.verdict == GateVerdict.PASS for r in self.gate_results)
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for notification formatting and logging."""
        return {
            "intent_id": self.intent_id,
            "ticker": self.ticker,
            "direction": self.direction.value,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "strategy_skill": self.strategy_skill,
            "strategy_lineage": list(self.strategy_lineage),
            "sizing_skill": self.sizing_skill,
            "regime_label": self.regime_label,
            "signal_sources": list(self.signal_sources),
            "confidence": self.confidence,
            "reasoning_chain": self.reasoning_chain,
            "position_impact": dict(self.position_impact),
            "paper_track_record": dict(self.paper_track_record),
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "gate_results": [
                {
                    "gate_name": g.gate_name,
                    "verdict": g.verdict.value,
                    "reason": g.reason,
                    "details": g.details,
                    "checked_at": g.checked_at.isoformat(),
                }
                for g in self.gate_results
            ],
            "blocking_gate": self.blocking_gate,
        }
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_trade_intent.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/execution/trade_intent.py tests/unit/test_trade_intent.py
git commit -m "feat: TradeIntent dataclass with full provenance and gate tracking"
```

---

## Task 3: Gate 1 — Immutable Risk Constraints

**Files:**
- Create: `src/evolve_trader/execution/gates/risk_gate.py`
- Create: `tests/unit/test_risk_gate.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_risk_gate.py
"""Tests for Gate 1: Immutable Risk Constraints.

These constraints are NEVER bypassed. Violation = blocked.
- Max 5% of portfolio in any single position
- Max 25% of portfolio in any single sector
- Max 20% portfolio drawdown triggers trading halt
"""
import pytest
from datetime import datetime, timezone
from evolve_trader.execution.gates.risk_gate import (
    RiskGate,
    RiskConstraints,
    PortfolioState,
)
from evolve_trader.execution.trade_intent import (
    TradeIntent,
    TradeDirection,
    IntentStatus,
    GateVerdict,
)


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def default_constraints():
    return RiskConstraints(
        max_position_pct=0.05,
        max_sector_pct=0.25,
        max_drawdown_pct=0.20,
    )


@pytest.fixture
def healthy_portfolio():
    """Portfolio with room for new positions."""
    return PortfolioState(
        total_equity=100_000.0,
        cash=40_000.0,
        positions={
            "AAPL": {"value": 3_000.0, "sector": "Technology"},
            "JPM": {"value": 4_000.0, "sector": "Financials"},
            "XOM": {"value": 3_000.0, "sector": "Energy"},
        },
        sector_exposure={
            "Technology": 0.03,
            "Financials": 0.04,
            "Energy": 0.03,
        },
        peak_equity=105_000.0,
        current_drawdown=0.048,  # 4.8% from peak
    )


@pytest.fixture
def stressed_portfolio():
    """Portfolio near drawdown limit."""
    return PortfolioState(
        total_equity=84_000.0,
        cash=5_000.0,
        positions={
            "AAPL": {"value": 4_500.0, "sector": "Technology"},
            "MSFT": {"value": 4_200.0, "sector": "Technology"},
            "GOOGL": {"value": 4_000.0, "sector": "Technology"},
            "NVDA": {"value": 3_800.0, "sector": "Technology"},
            "AMZN": {"value": 3_500.0, "sector": "Technology"},
        },
        sector_exposure={
            "Technology": 0.238,
        },
        peak_equity=105_000.0,
        current_drawdown=0.20,  # exactly at 20% drawdown
    )


def _make_intent(ticker="AAPL", direction=TradeDirection.BUY, quantity=10.0,
                  order_type="market", confidence=0.85, sector="Technology"):
    return TradeIntent(
        intent_id=f"intent-{ticker}",
        ticker=ticker,
        direction=direction,
        quantity=quantity,
        order_type=order_type,
        strategy_skill="momentum-v1",
        strategy_lineage=["momentum-v1"],
        sizing_skill="kelly-fractional-v1",
        regime_label="risk-on",
        signal_sources=["edgar_13f"],
        confidence=confidence,
        reasoning_chain="Test intent.",
        position_impact={"sector": sector},
    )


# ─── Position Size Constraint (5%) ──────────────────────────────────


def test_position_within_limit_passes(default_constraints, healthy_portfolio):
    """Trade keeping position under 5% passes."""
    gate = RiskGate(default_constraints)
    intent = _make_intent(ticker="META", quantity=2.0)  # small position
    # Assume META price ~$500 → $1,000 / $100,000 = 1%
    result = gate.check(intent, healthy_portfolio, estimated_cost=1_000.0)
    assert result.verdict == GateVerdict.PASS


def test_position_exceeds_limit_blocked(default_constraints, healthy_portfolio):
    """Trade pushing position over 5% is blocked."""
    gate = RiskGate(default_constraints)
    intent = _make_intent(ticker="AAPL", quantity=20.0)
    # AAPL existing $3,000 + new $3,000 = $6,000 / $100,000 = 6% > 5%
    result = gate.check(intent, healthy_portfolio, estimated_cost=3_000.0)
    assert result.verdict == GateVerdict.BLOCK
    assert "position" in result.reason.lower()
    assert "5%" in result.reason


def test_new_position_exactly_at_limit_passes(default_constraints, healthy_portfolio):
    """Position at exactly 5% is allowed (not exceeded)."""
    gate = RiskGate(default_constraints)
    intent = _make_intent(ticker="META", quantity=10.0)
    # New position exactly at $5,000 / $100,000 = 5.0%
    result = gate.check(intent, healthy_portfolio, estimated_cost=5_000.0)
    assert result.verdict == GateVerdict.PASS


# ─── Sector Exposure Constraint (25%) ───────────────────────────────


def test_sector_within_limit_passes(default_constraints, healthy_portfolio):
    """Trade keeping sector under 25% passes."""
    gate = RiskGate(default_constraints)
    intent = _make_intent(ticker="CRM", quantity=5.0, sector="Technology")
    # Tech at 3% + $2,000/$100,000 = 5% — well under 25%
    result = gate.check(intent, healthy_portfolio, estimated_cost=2_000.0)
    assert result.verdict == GateVerdict.PASS


def test_sector_exceeds_limit_blocked(default_constraints, stressed_portfolio):
    """Trade pushing sector over 25% is blocked."""
    gate = RiskGate(default_constraints)
    intent = _make_intent(ticker="INTC", quantity=10.0, sector="Technology")
    # Tech at 23.8% + $2,000/$84,000 = 26.2% > 25%
    result = gate.check(intent, stressed_portfolio, estimated_cost=2_000.0)
    assert result.verdict == GateVerdict.BLOCK
    assert "sector" in result.reason.lower()


# ─── Drawdown Halt Constraint (20%) ─────────────────────────────────


def test_drawdown_under_limit_passes(default_constraints, healthy_portfolio):
    """Trading allowed when drawdown is under 20%."""
    gate = RiskGate(default_constraints)
    intent = _make_intent(ticker="META", quantity=2.0)
    result = gate.check(intent, healthy_portfolio, estimated_cost=1_000.0)
    assert result.verdict == GateVerdict.PASS


def test_drawdown_at_limit_blocked(default_constraints, stressed_portfolio):
    """Trading halted when drawdown reaches 20%."""
    gate = RiskGate(default_constraints)
    intent = _make_intent(ticker="META", quantity=1.0, sector="Consumer")
    result = gate.check(intent, stressed_portfolio, estimated_cost=500.0)
    assert result.verdict == GateVerdict.BLOCK
    assert "drawdown" in result.reason.lower()


def test_drawdown_check_allows_sells(default_constraints, stressed_portfolio):
    """SELL orders pass even during drawdown halt (reducing exposure)."""
    gate = RiskGate(default_constraints)
    intent = _make_intent(
        ticker="AAPL", direction=TradeDirection.SELL, quantity=5.0, sector="Technology",
    )
    result = gate.check(intent, stressed_portfolio, estimated_cost=0.0)
    assert result.verdict == GateVerdict.PASS


# ─── Constraint Immutability ─────────────────────────────────────────


def test_constraints_are_frozen():
    """Risk constraints cannot be modified after creation."""
    constraints = RiskConstraints(
        max_position_pct=0.05,
        max_sector_pct=0.25,
        max_drawdown_pct=0.20,
    )
    with pytest.raises(AttributeError):
        constraints.max_position_pct = 0.10


def test_gate_cannot_be_bypassed(default_constraints, stressed_portfolio):
    """Gate has no bypass mechanism — blocked is blocked."""
    gate = RiskGate(default_constraints)
    intent = _make_intent(ticker="META", quantity=1.0, sector="Consumer")
    result = gate.check(intent, stressed_portfolio, estimated_cost=500.0)
    assert result.verdict == GateVerdict.BLOCK
    # Verify there's no bypass method
    assert not hasattr(gate, "bypass")
    assert not hasattr(gate, "override")
    assert not hasattr(gate, "force")


# ─── Multiple Constraint Violations ─────────────────────────────────


def test_multiple_violations_reports_first(default_constraints, stressed_portfolio):
    """When multiple constraints are violated, reports all violations."""
    gate = RiskGate(default_constraints)
    intent = _make_intent(ticker="NVDA", quantity=50.0, sector="Technology")
    # Violates: drawdown (20%), sector (>25%), position (>5%)
    result = gate.check(intent, stressed_portfolio, estimated_cost=5_000.0)
    assert result.verdict == GateVerdict.BLOCK
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_risk_gate.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.execution.gates.risk_gate'`

**Step 3: Implement the risk gate**

```python
# src/evolve_trader/execution/gates/__init__.py
"""Execution gates for the TradeIntent pipeline."""
```

```python
# src/evolve_trader/execution/gates/risk_gate.py
"""Gate 1: Immutable Risk Constraints.

This gate enforces hard limits that can NEVER be bypassed:
- Max 5% of portfolio in any single position
- Max 25% of portfolio in any single sector
- Max 20% portfolio drawdown triggers trading halt (sells still allowed)

Violation = BLOCK. No override. No bypass. No exceptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from evolve_trader.execution.trade_intent import (
    TradeIntent,
    TradeDirection,
    GateResult,
    GateVerdict,
)


@dataclass(frozen=True)
class RiskConstraints:
    """Immutable risk constraint thresholds.

    Frozen dataclass — cannot be modified after creation.
    """
    max_position_pct: float  # Max single position as fraction of portfolio
    max_sector_pct: float    # Max single sector as fraction of portfolio
    max_drawdown_pct: float  # Max drawdown before trading halt


@dataclass
class PortfolioState:
    """Current portfolio state snapshot for risk evaluation."""
    total_equity: float
    cash: float
    positions: dict[str, dict[str, Any]]  # ticker -> {value, sector, ...}
    sector_exposure: dict[str, float]     # sector -> fraction of portfolio
    peak_equity: float
    current_drawdown: float               # fraction (0.15 = 15%)


class RiskGate:
    """Gate 1: Immutable risk constraint checker.

    Evaluates a TradeIntent against hard portfolio limits.
    Returns GateResult with PASS or BLOCK verdict.
    There is no bypass, override, or force mechanism.
    """

    GATE_NAME = "risk_constraints"

    def __init__(self, constraints: RiskConstraints):
        self._constraints = constraints

    def check(
        self,
        intent: TradeIntent,
        portfolio: PortfolioState,
        estimated_cost: float,
    ) -> GateResult:
        """Evaluate intent against all risk constraints.

        Args:
            intent: The trade intent to evaluate.
            portfolio: Current portfolio state.
            estimated_cost: Estimated dollar cost of the trade.

        Returns:
            GateResult with PASS or BLOCK verdict.
        """
        violations: list[str] = []
        details: dict[str, Any] = {}

        # ── Check 1: Drawdown halt ──
        # Sells are always allowed during drawdown (reduce exposure)
        is_reducing = intent.direction in (TradeDirection.SELL, TradeDirection.COVER)
        if not is_reducing:
            if portfolio.current_drawdown >= self._constraints.max_drawdown_pct:
                pct = portfolio.current_drawdown * 100
                limit = self._constraints.max_drawdown_pct * 100
                violations.append(
                    f"Drawdown halt: portfolio drawdown {pct:.1f}% "
                    f">= {limit:.1f}% limit. New buys blocked."
                )
                details["current_drawdown"] = portfolio.current_drawdown
                details["max_drawdown_pct"] = self._constraints.max_drawdown_pct

        # ── Check 2: Position size ──
        if not is_reducing and portfolio.total_equity > 0:
            existing_value = portfolio.positions.get(
                intent.ticker, {}
            ).get("value", 0.0)
            new_total = existing_value + estimated_cost
            position_pct = new_total / portfolio.total_equity

            if position_pct > self._constraints.max_position_pct:
                violations.append(
                    f"Position limit: {intent.ticker} would be "
                    f"{position_pct * 100:.1f}% of portfolio, "
                    f"exceeding 5% limit."
                )
                details["position_pct"] = position_pct
                details["max_position_pct"] = self._constraints.max_position_pct

        # ── Check 3: Sector exposure ──
        if not is_reducing and portfolio.total_equity > 0:
            sector = intent.position_impact.get("sector", "Unknown")
            current_sector_pct = portfolio.sector_exposure.get(sector, 0.0)
            added_sector_pct = estimated_cost / portfolio.total_equity
            new_sector_pct = current_sector_pct + added_sector_pct

            if new_sector_pct > self._constraints.max_sector_pct:
                violations.append(
                    f"Sector limit: {sector} would be "
                    f"{new_sector_pct * 100:.1f}% of portfolio, "
                    f"exceeding 25% sector limit."
                )
                details["sector"] = sector
                details["sector_pct"] = new_sector_pct
                details["max_sector_pct"] = self._constraints.max_sector_pct

        # ── Verdict ──
        if violations:
            return GateResult(
                gate_name=self.GATE_NAME,
                verdict=GateVerdict.BLOCK,
                reason=" | ".join(violations),
                details=details,
                checked_at=datetime.now(timezone.utc),
            )

        return GateResult(
            gate_name=self.GATE_NAME,
            verdict=GateVerdict.PASS,
            details={
                "position_pct": (
                    (portfolio.positions.get(intent.ticker, {}).get("value", 0.0)
                     + estimated_cost) / portfolio.total_equity
                    if portfolio.total_equity > 0 else 0.0
                ),
                "max_position_pct": self._constraints.max_position_pct,
                "current_drawdown": portfolio.current_drawdown,
            },
            checked_at=datetime.now(timezone.utc),
        )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_risk_gate.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/execution/gates/ tests/unit/test_risk_gate.py
git commit -m "feat: Gate 1 immutable risk constraints — position, sector, drawdown limits"
```

---

## Task 4: Gate 2 — Paper Trading Shadow

**Files:**
- Create: `src/evolve_trader/execution/gates/paper_shadow.py`
- Create: `tests/unit/test_paper_shadow.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_paper_shadow.py
"""Tests for Gate 2: Paper Trading Shadow.

Every intent executes on paper. Includes vetoed trades for counterfactual
benchmarking. This gate NEVER stops — it always passes through while
recording the paper execution.
"""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from evolve_trader.execution.gates.paper_shadow import (
    PaperShadowGate,
    PaperTradeRecord,
    PaperTradeBook,
    CounterfactualResult,
)
from evolve_trader.execution.trade_intent import (
    TradeIntent,
    TradeDirection,
    IntentStatus,
    GateVerdict,
)


# ─── Fixtures ────────────────────────────────────────────────────────


def _make_intent(ticker="AAPL", direction=TradeDirection.BUY, quantity=10.0,
                  confidence=0.85, status=IntentStatus.PENDING):
    intent = TradeIntent(
        intent_id=f"intent-{ticker}-{id(ticker)}",
        ticker=ticker,
        direction=direction,
        quantity=quantity,
        order_type="market",
        strategy_skill="momentum-v1",
        strategy_lineage=["momentum-v1"],
        sizing_skill="kelly-fractional-v1",
        regime_label="risk-on",
        signal_sources=["edgar_13f"],
        confidence=confidence,
        reasoning_chain="Test intent.",
    )
    intent.status = status
    return intent


@pytest.fixture
def paper_book():
    return PaperTradeBook()


@pytest.fixture
def mock_alpaca_client():
    client = AsyncMock()
    client.submit_order = AsyncMock(return_value=MagicMock(
        order_id="paper-order-123",
        symbol="AAPL",
        status="filled",
        filled_qty=10.0,
        filled_avg_price=150.0,
    ))
    return client


# ─── Paper Shadow Gate Always Passes ─────────────────────────────────


@pytest.mark.asyncio
async def test_shadow_gate_always_passes(paper_book, mock_alpaca_client):
    """Gate 2 always returns PASS — it never blocks."""
    gate = PaperShadowGate(paper_book=paper_book, paper_client=mock_alpaca_client)
    intent = _make_intent()
    result = await gate.check(intent, estimated_price=150.0)
    assert result.verdict == GateVerdict.PASS


@pytest.mark.asyncio
async def test_shadow_records_paper_trade(paper_book, mock_alpaca_client):
    """Gate 2 records every intent as a paper trade."""
    gate = PaperShadowGate(paper_book=paper_book, paper_client=mock_alpaca_client)
    intent = _make_intent(ticker="AAPL")
    await gate.check(intent, estimated_price=150.0)

    records = paper_book.get_trades(strategy="momentum-v1")
    assert len(records) == 1
    assert records[0].ticker == "AAPL"
    assert records[0].paper_order_id == "paper-order-123"


@pytest.mark.asyncio
async def test_shadow_includes_vetoed_trades(paper_book, mock_alpaca_client):
    """Vetoed (blocked) trades are still recorded for counterfactual analysis."""
    gate = PaperShadowGate(paper_book=paper_book, paper_client=mock_alpaca_client)
    intent = _make_intent(ticker="TSLA", status=IntentStatus.BLOCKED)
    await gate.check(intent, estimated_price=250.0)

    records = paper_book.get_trades(include_vetoed=True)
    assert len(records) == 1
    assert records[0].was_vetoed is True
    assert records[0].ticker == "TSLA"


@pytest.mark.asyncio
async def test_shadow_never_stops(paper_book, mock_alpaca_client):
    """Shadow gate processes even when paper client fails — logs error, passes."""
    failing_client = AsyncMock()
    failing_client.submit_order = AsyncMock(side_effect=Exception("Connection lost"))

    gate = PaperShadowGate(paper_book=paper_book, paper_client=failing_client)
    intent = _make_intent()
    result = await gate.check(intent, estimated_price=150.0)

    # Still passes — paper shadow failure should not block live trading
    assert result.verdict == GateVerdict.PASS
    assert "error" in result.details


# ─── Paper Trade Book ────────────────────────────────────────────────


def test_paper_book_tracks_by_strategy(paper_book):
    """Book filters trades by strategy skill."""
    record1 = PaperTradeRecord(
        intent_id="i1", ticker="AAPL", direction="BUY", quantity=10.0,
        strategy_skill="momentum-v1", paper_order_id="p1",
        estimated_price=150.0, was_vetoed=False,
        recorded_at=datetime.now(timezone.utc),
    )
    record2 = PaperTradeRecord(
        intent_id="i2", ticker="MSFT", direction="BUY", quantity=5.0,
        strategy_skill="mean-reversion-v1", paper_order_id="p2",
        estimated_price=350.0, was_vetoed=False,
        recorded_at=datetime.now(timezone.utc),
    )
    paper_book.add(record1)
    paper_book.add(record2)

    momentum_trades = paper_book.get_trades(strategy="momentum-v1")
    assert len(momentum_trades) == 1
    assert momentum_trades[0].ticker == "AAPL"


def test_paper_book_excludes_vetoed_by_default(paper_book):
    """get_trades excludes vetoed trades unless explicitly requested."""
    record1 = PaperTradeRecord(
        intent_id="i1", ticker="AAPL", direction="BUY", quantity=10.0,
        strategy_skill="momentum-v1", paper_order_id="p1",
        estimated_price=150.0, was_vetoed=False,
        recorded_at=datetime.now(timezone.utc),
    )
    record2 = PaperTradeRecord(
        intent_id="i2", ticker="TSLA", direction="BUY", quantity=5.0,
        strategy_skill="momentum-v1", paper_order_id="p2",
        estimated_price=250.0, was_vetoed=True,
        recorded_at=datetime.now(timezone.utc),
    )
    paper_book.add(record1)
    paper_book.add(record2)

    default_trades = paper_book.get_trades()
    assert len(default_trades) == 1

    all_trades = paper_book.get_trades(include_vetoed=True)
    assert len(all_trades) == 2


# ─── Counterfactual Analysis ────────────────────────────────────────


def test_paper_book_counterfactual_analysis(paper_book):
    """Counterfactual compares vetoed vs executed outcomes."""
    now = datetime.now(timezone.utc)

    executed = PaperTradeRecord(
        intent_id="i1", ticker="AAPL", direction="BUY", quantity=10.0,
        strategy_skill="momentum-v1", paper_order_id="p1",
        estimated_price=150.0, was_vetoed=False,
        recorded_at=now - timedelta(days=10),
        exit_price=160.0,
        pnl=100.0,
    )
    vetoed = PaperTradeRecord(
        intent_id="i2", ticker="TSLA", direction="BUY", quantity=5.0,
        strategy_skill="momentum-v1", paper_order_id="p2",
        estimated_price=250.0, was_vetoed=True,
        recorded_at=now - timedelta(days=10),
        exit_price=280.0,
        pnl=150.0,  # Would have made money
    )
    paper_book.add(executed)
    paper_book.add(vetoed)

    analysis = paper_book.counterfactual_analysis(strategy="momentum-v1")
    assert isinstance(analysis, CounterfactualResult)
    assert analysis.executed_pnl == 100.0
    assert analysis.vetoed_pnl == 150.0
    assert analysis.opportunity_cost == 50.0  # missed profit
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_paper_shadow.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.execution.gates.paper_shadow'`

**Step 3: Implement the paper shadow gate**

```python
# src/evolve_trader/execution/gates/paper_shadow.py
"""Gate 2: Paper Trading Shadow.

Every TradeIntent executes on paper, regardless of whether it was blocked
by Gate 1 or will be rejected by Gate 3. This provides:
- Full paper trading track record for every strategy
- Counterfactual analysis of vetoed trades
- Benchmark data for promotion decisions

This gate NEVER blocks. It ALWAYS returns PASS.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from evolve_trader.execution.trade_intent import (
    TradeIntent,
    IntentStatus,
    GateResult,
    GateVerdict,
)

logger = logging.getLogger(__name__)


@dataclass
class PaperTradeRecord:
    """Record of a single paper trade execution."""
    intent_id: str
    ticker: str
    direction: str
    quantity: float
    strategy_skill: str
    paper_order_id: str
    estimated_price: float
    was_vetoed: bool
    recorded_at: datetime
    exit_price: float | None = None
    pnl: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CounterfactualResult:
    """Result of comparing executed vs vetoed trade outcomes."""
    executed_count: int = 0
    vetoed_count: int = 0
    executed_pnl: float = 0.0
    vetoed_pnl: float = 0.0

    @property
    def opportunity_cost(self) -> float:
        """Profit missed by vetoing trades (positive = missed profit)."""
        return self.vetoed_pnl - 0.0  # vetoed trades weren't executed live


class PaperTradeBook:
    """In-memory ledger of all paper trades for counterfactual tracking."""

    def __init__(self):
        self._records: list[PaperTradeRecord] = []

    def add(self, record: PaperTradeRecord) -> None:
        """Add a paper trade record."""
        self._records.append(record)

    def get_trades(
        self,
        strategy: str | None = None,
        include_vetoed: bool = False,
    ) -> list[PaperTradeRecord]:
        """Get paper trades, optionally filtered.

        Args:
            strategy: Filter by strategy skill name.
            include_vetoed: If False, excludes vetoed trades.

        Returns:
            List of matching PaperTradeRecords.
        """
        results = self._records
        if not include_vetoed:
            results = [r for r in results if not r.was_vetoed]
        if strategy:
            results = [r for r in results if r.strategy_skill == strategy]
        return results

    def counterfactual_analysis(
        self,
        strategy: str | None = None,
    ) -> CounterfactualResult:
        """Compare outcomes of executed vs vetoed trades.

        Args:
            strategy: Filter by strategy skill name.

        Returns:
            CounterfactualResult with aggregated P&L comparison.
        """
        all_trades = [r for r in self._records]
        if strategy:
            all_trades = [r for r in all_trades if r.strategy_skill == strategy]

        executed = [r for r in all_trades if not r.was_vetoed]
        vetoed = [r for r in all_trades if r.was_vetoed]

        executed_pnl = sum(r.pnl or 0.0 for r in executed)
        vetoed_pnl = sum(r.pnl or 0.0 for r in vetoed)

        return CounterfactualResult(
            executed_count=len(executed),
            vetoed_count=len(vetoed),
            executed_pnl=executed_pnl,
            vetoed_pnl=vetoed_pnl,
        )


class PaperShadowGate:
    """Gate 2: Paper trading shadow execution.

    This gate:
    1. Executes every intent on paper (via paper Alpaca client)
    2. Records the result in PaperTradeBook
    3. Always returns PASS — never blocks

    Even if the paper client fails, this gate passes and logs the error.
    """

    GATE_NAME = "paper_shadow"

    def __init__(self, paper_book: PaperTradeBook, paper_client: Any):
        self._paper_book = paper_book
        self._paper_client = paper_client

    async def check(
        self,
        intent: TradeIntent,
        estimated_price: float,
    ) -> GateResult:
        """Execute intent on paper and record result.

        Args:
            intent: The trade intent to shadow-execute.
            estimated_price: Current estimated price for the security.

        Returns:
            GateResult — always PASS.
        """
        was_vetoed = intent.status == IntentStatus.BLOCKED
        details: dict[str, Any] = {"was_vetoed": was_vetoed}

        paper_order_id = "none"
        try:
            from evolve_trader.execution.order_mapper import (
                OrderRequest, OrderSide, OrderType, TimeInForce,
            )
            side = OrderSide.BUY if intent.direction.value in ("BUY", "COVER") else OrderSide.SELL
            request = OrderRequest(
                symbol=intent.ticker,
                side=side,
                qty=intent.quantity,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
            )
            result = await self._paper_client.submit_order(request)
            paper_order_id = str(result.order_id)
            details["paper_order_id"] = paper_order_id
        except Exception as e:
            logger.warning(f"Paper shadow execution failed for {intent.intent_id}: {e}")
            details["error"] = str(e)
            paper_order_id = "error"

        record = PaperTradeRecord(
            intent_id=intent.intent_id,
            ticker=intent.ticker,
            direction=intent.direction.value,
            quantity=intent.quantity,
            strategy_skill=intent.strategy_skill,
            paper_order_id=paper_order_id,
            estimated_price=estimated_price,
            was_vetoed=was_vetoed,
            recorded_at=datetime.now(timezone.utc),
        )
        self._paper_book.add(record)

        return GateResult(
            gate_name=self.GATE_NAME,
            verdict=GateVerdict.PASS,
            details=details,
            checked_at=datetime.now(timezone.utc),
        )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_paper_shadow.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/execution/gates/paper_shadow.py tests/unit/test_paper_shadow.py
git commit -m "feat: Gate 2 paper shadow — counterfactual tracking of all trades including vetoed"
```

---

## Task 5: Gate 3 — Graduated Approval Gate

**Files:**
- Create: `src/evolve_trader/execution/gates/approval_gate.py`
- Create: `tests/unit/test_approval_gate.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_approval_gate.py
"""Tests for Gate 3: Graduated Approval Gate.

Starts fully manual. Graduates to auto-approval based on track record.
Auto-approval criteria: confidence > 0.85 from a validated strategy skill.
"""
import pytest
from datetime import datetime, timezone, timedelta
from evolve_trader.execution.gates.approval_gate import (
    ApprovalGate,
    ApprovalPolicy,
    ApprovalMode,
    SkillValidationRecord,
)
from evolve_trader.execution.trade_intent import (
    TradeIntent,
    TradeDirection,
    GateVerdict,
)


# ─── Fixtures ────────────────────────────────────────────────────────


def _make_intent(confidence=0.85, strategy="momentum-v1"):
    return TradeIntent(
        intent_id=f"intent-{id(confidence)}",
        ticker="AAPL",
        direction=TradeDirection.BUY,
        quantity=10.0,
        order_type="market",
        strategy_skill=strategy,
        strategy_lineage=[strategy],
        sizing_skill="kelly-fractional-v1",
        regime_label="risk-on",
        signal_sources=["edgar_13f"],
        confidence=confidence,
        reasoning_chain="Test intent.",
    )


@pytest.fixture
def manual_policy():
    """Policy that requires manual approval for everything."""
    return ApprovalPolicy(
        mode=ApprovalMode.MANUAL,
        auto_approve_min_confidence=0.85,
        auto_approve_min_paper_trades=50,
        auto_approve_min_sharpe=0.5,
    )


@pytest.fixture
def graduated_policy():
    """Policy that auto-approves validated skills above confidence threshold."""
    return ApprovalPolicy(
        mode=ApprovalMode.GRADUATED,
        auto_approve_min_confidence=0.85,
        auto_approve_min_paper_trades=50,
        auto_approve_min_sharpe=0.5,
    )


@pytest.fixture
def validated_skill():
    """A skill that has met validation criteria."""
    return SkillValidationRecord(
        skill_name="momentum-v1",
        paper_trades=75,
        paper_sharpe=1.2,
        paper_win_rate=0.58,
        paper_max_drawdown=0.08,
        validated_at=datetime.now(timezone.utc),
        is_validated=True,
    )


@pytest.fixture
def unvalidated_skill():
    """A skill that has NOT met validation criteria."""
    return SkillValidationRecord(
        skill_name="experimental-v1",
        paper_trades=15,
        paper_sharpe=0.3,
        paper_win_rate=0.45,
        paper_max_drawdown=0.15,
        validated_at=datetime.now(timezone.utc),
        is_validated=False,
    )


# ─── Manual Mode ─────────────────────────────────────────────────────


def test_manual_mode_always_requires_approval(manual_policy):
    """In MANUAL mode, every trade requires human approval."""
    gate = ApprovalGate(policy=manual_policy, skill_records={})
    intent = _make_intent(confidence=0.99)
    result = gate.check(intent)
    assert result.verdict == GateVerdict.PENDING_APPROVAL
    assert "manual" in result.reason.lower()


# ─── Graduated Mode ─────────────────────────────────────────────────


def test_graduated_auto_approves_validated_high_confidence(
    graduated_policy, validated_skill,
):
    """Auto-approves when skill is validated AND confidence > threshold."""
    gate = ApprovalGate(
        policy=graduated_policy,
        skill_records={"momentum-v1": validated_skill},
    )
    intent = _make_intent(confidence=0.90, strategy="momentum-v1")
    result = gate.check(intent)
    assert result.verdict == GateVerdict.PASS
    assert "auto" in result.details.get("approval_type", "").lower()


def test_graduated_requires_approval_low_confidence(
    graduated_policy, validated_skill,
):
    """Requires manual approval when confidence is below threshold."""
    gate = ApprovalGate(
        policy=graduated_policy,
        skill_records={"momentum-v1": validated_skill},
    )
    intent = _make_intent(confidence=0.60, strategy="momentum-v1")
    result = gate.check(intent)
    assert result.verdict == GateVerdict.PENDING_APPROVAL
    assert "confidence" in result.reason.lower()


def test_graduated_requires_approval_unvalidated_skill(
    graduated_policy, unvalidated_skill,
):
    """Requires manual approval when skill is not validated."""
    gate = ApprovalGate(
        policy=graduated_policy,
        skill_records={"experimental-v1": unvalidated_skill},
    )
    intent = _make_intent(confidence=0.95, strategy="experimental-v1")
    result = gate.check(intent)
    assert result.verdict == GateVerdict.PENDING_APPROVAL
    assert "not validated" in result.reason.lower()


def test_graduated_requires_approval_unknown_skill(graduated_policy):
    """Requires manual approval when skill has no validation record."""
    gate = ApprovalGate(policy=graduated_policy, skill_records={})
    intent = _make_intent(confidence=0.95, strategy="unknown-v1")
    result = gate.check(intent)
    assert result.verdict == GateVerdict.PENDING_APPROVAL
    assert "no validation record" in result.reason.lower()


def test_graduated_boundary_confidence(graduated_policy, validated_skill):
    """Confidence exactly at threshold still requires approval (must exceed)."""
    gate = ApprovalGate(
        policy=graduated_policy,
        skill_records={"momentum-v1": validated_skill},
    )
    intent = _make_intent(confidence=0.85, strategy="momentum-v1")
    result = gate.check(intent)
    assert result.verdict == GateVerdict.PENDING_APPROVAL


# ─── Auto Mode ───────────────────────────────────────────────────────


def test_auto_mode_approves_validated_skills(validated_skill):
    """AUTO mode approves all validated skills regardless of confidence."""
    policy = ApprovalPolicy(
        mode=ApprovalMode.AUTO,
        auto_approve_min_confidence=0.85,
        auto_approve_min_paper_trades=50,
        auto_approve_min_sharpe=0.5,
    )
    gate = ApprovalGate(
        policy=policy,
        skill_records={"momentum-v1": validated_skill},
    )
    intent = _make_intent(confidence=0.50, strategy="momentum-v1")
    result = gate.check(intent)
    assert result.verdict == GateVerdict.PASS


def test_auto_mode_blocks_unvalidated_skills(unvalidated_skill):
    """AUTO mode still requires approval for unvalidated skills."""
    policy = ApprovalPolicy(
        mode=ApprovalMode.AUTO,
        auto_approve_min_confidence=0.85,
        auto_approve_min_paper_trades=50,
        auto_approve_min_sharpe=0.5,
    )
    gate = ApprovalGate(
        policy=policy,
        skill_records={"experimental-v1": unvalidated_skill},
    )
    intent = _make_intent(confidence=0.95, strategy="experimental-v1")
    result = gate.check(intent)
    assert result.verdict == GateVerdict.PENDING_APPROVAL


# ─── Skill Validation Record ────────────────────────────────────────


def test_skill_validation_checks_all_criteria():
    """Validation requires meeting ALL thresholds."""
    record = SkillValidationRecord(
        skill_name="test-v1",
        paper_trades=75,
        paper_sharpe=1.2,
        paper_win_rate=0.58,
        paper_max_drawdown=0.08,
        validated_at=datetime.now(timezone.utc),
        is_validated=True,
    )
    assert record.is_validated is True
    assert record.paper_trades >= 50
    assert record.paper_sharpe >= 0.5


def test_skill_validation_fails_insufficient_trades():
    """Skill with too few trades is not validated."""
    record = SkillValidationRecord(
        skill_name="new-v1",
        paper_trades=10,
        paper_sharpe=2.0,
        paper_win_rate=0.70,
        paper_max_drawdown=0.05,
        validated_at=datetime.now(timezone.utc),
        is_validated=False,
    )
    assert record.is_validated is False
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_approval_gate.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.execution.gates.approval_gate'`

**Step 3: Implement the approval gate**

```python
# src/evolve_trader/execution/gates/approval_gate.py
"""Gate 3: Graduated Approval Gate.

Controls trade execution approval with three modes:
- MANUAL: Every trade requires human approval
- GRADUATED: Auto-approves validated skills above confidence threshold
- AUTO: Auto-approves all validated skills (unvalidated still need approval)

Graduation criteria for auto-approval:
- Confidence > 0.85 (configurable)
- Strategy skill must be validated (sufficient paper trades, Sharpe, etc.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from evolve_trader.execution.trade_intent import (
    TradeIntent,
    GateResult,
    GateVerdict,
)


class ApprovalMode(str, Enum):
    """Approval mode for the gate."""
    MANUAL = "manual"        # Every trade needs human approval
    GRADUATED = "graduated"  # Auto-approve validated + high-confidence
    AUTO = "auto"            # Auto-approve all validated skills


@dataclass(frozen=True)
class ApprovalPolicy:
    """Configuration for the approval gate."""
    mode: ApprovalMode
    auto_approve_min_confidence: float = 0.85
    auto_approve_min_paper_trades: int = 50
    auto_approve_min_sharpe: float = 0.5


@dataclass
class SkillValidationRecord:
    """Track record of a strategy skill's paper trading performance."""
    skill_name: str
    paper_trades: int
    paper_sharpe: float
    paper_win_rate: float
    paper_max_drawdown: float
    validated_at: datetime
    is_validated: bool


class ApprovalGate:
    """Gate 3: Graduated approval based on skill validation and confidence.

    Decision logic:
    - MANUAL mode: always PENDING_APPROVAL
    - GRADUATED mode: PASS if skill is validated AND confidence > threshold
    - AUTO mode: PASS if skill is validated (any confidence)

    Unvalidated or unknown skills always require approval in all modes.
    """

    GATE_NAME = "approval"

    def __init__(
        self,
        policy: ApprovalPolicy,
        skill_records: dict[str, SkillValidationRecord],
    ):
        self._policy = policy
        self._skill_records = skill_records

    def check(self, intent: TradeIntent) -> GateResult:
        """Evaluate whether the intent can be auto-approved.

        Args:
            intent: The trade intent to evaluate.

        Returns:
            GateResult with PASS or PENDING_APPROVAL verdict.
        """
        now = datetime.now(timezone.utc)

        # MANUAL mode — always require approval
        if self._policy.mode == ApprovalMode.MANUAL:
            return GateResult(
                gate_name=self.GATE_NAME,
                verdict=GateVerdict.PENDING_APPROVAL,
                reason="Manual approval required (policy mode: MANUAL).",
                details={"approval_type": "manual", "mode": "manual"},
                checked_at=now,
            )

        # Look up skill validation
        skill_record = self._skill_records.get(intent.strategy_skill)

        if skill_record is None:
            return GateResult(
                gate_name=self.GATE_NAME,
                verdict=GateVerdict.PENDING_APPROVAL,
                reason=(
                    f"No validation record for skill '{intent.strategy_skill}'. "
                    f"Manual approval required."
                ),
                details={"approval_type": "manual", "reason": "no_validation_record"},
                checked_at=now,
            )

        if not skill_record.is_validated:
            return GateResult(
                gate_name=self.GATE_NAME,
                verdict=GateVerdict.PENDING_APPROVAL,
                reason=(
                    f"Skill '{intent.strategy_skill}' is not validated "
                    f"(trades={skill_record.paper_trades}, "
                    f"sharpe={skill_record.paper_sharpe:.2f}). "
                    f"Manual approval required."
                ),
                details={
                    "approval_type": "manual",
                    "reason": "skill_not_validated",
                    "paper_trades": skill_record.paper_trades,
                    "paper_sharpe": skill_record.paper_sharpe,
                },
                checked_at=now,
            )

        # Skill is validated — check mode-specific criteria
        if self._policy.mode == ApprovalMode.AUTO:
            return GateResult(
                gate_name=self.GATE_NAME,
                verdict=GateVerdict.PASS,
                details={
                    "approval_type": "auto",
                    "mode": "auto",
                    "skill_validated": True,
                },
                checked_at=now,
            )

        # GRADUATED mode — need confidence > threshold
        if intent.confidence > self._policy.auto_approve_min_confidence:
            return GateResult(
                gate_name=self.GATE_NAME,
                verdict=GateVerdict.PASS,
                details={
                    "approval_type": "auto_graduated",
                    "mode": "graduated",
                    "confidence": intent.confidence,
                    "threshold": self._policy.auto_approve_min_confidence,
                    "skill_validated": True,
                },
                checked_at=now,
            )

        return GateResult(
            gate_name=self.GATE_NAME,
            verdict=GateVerdict.PENDING_APPROVAL,
            reason=(
                f"Confidence {intent.confidence:.2f} does not exceed "
                f"auto-approval threshold {self._policy.auto_approve_min_confidence}. "
                f"Manual approval required."
            ),
            details={
                "approval_type": "manual",
                "reason": "confidence_below_threshold",
                "confidence": intent.confidence,
                "threshold": self._policy.auto_approve_min_confidence,
            },
            checked_at=now,
        )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_approval_gate.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/execution/gates/approval_gate.py tests/unit/test_approval_gate.py
git commit -m "feat: Gate 3 graduated approval — manual to auto based on skill validation"
```

---

## Task 6: Paper-to-Live Promotion Protocol

**Files:**
- Create: `src/evolve_trader/execution/promotion.py`
- Create: `tests/unit/test_promotion_protocol.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_promotion_protocol.py
"""Tests for Paper-to-Live Promotion Protocol.

5 stages with bidirectional demotion:
1. Paper Training (90d)
2. Paper Validation (60d, Sharpe > 0.5, DD < 15%)
3. Micro-Live (30d, 5-10% allocation)
4. Partial-Live (60d, 25-50% allocation)
5. Full-Live
"""
import pytest
from datetime import datetime, timezone, timedelta
from evolve_trader.execution.promotion import (
    PromotionProtocol,
    PromotionStage,
    StageRequirements,
    PromotionRecord,
    PromotionDecision,
    DemotionTrigger,
)


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def protocol():
    return PromotionProtocol()


@pytest.fixture
def paper_training_record():
    """Record at Paper Training stage with 90+ days."""
    return PromotionRecord(
        skill_name="momentum-v1",
        current_stage=PromotionStage.PAPER_TRAINING,
        stage_entered_at=datetime.now(timezone.utc) - timedelta(days=95),
        paper_trades=120,
        sharpe_ratio=0.8,
        max_drawdown=0.10,
        win_rate=0.55,
        total_pnl=5_000.0,
        allocation_pct=0.0,
    )


@pytest.fixture
def paper_validation_record():
    """Record at Paper Validation stage meeting all criteria."""
    return PromotionRecord(
        skill_name="momentum-v1",
        current_stage=PromotionStage.PAPER_VALIDATION,
        stage_entered_at=datetime.now(timezone.utc) - timedelta(days=65),
        paper_trades=200,
        sharpe_ratio=0.7,
        max_drawdown=0.12,
        win_rate=0.58,
        total_pnl=10_000.0,
        allocation_pct=0.0,
    )


@pytest.fixture
def micro_live_record():
    """Record at Micro-Live stage meeting criteria."""
    return PromotionRecord(
        skill_name="momentum-v1",
        current_stage=PromotionStage.MICRO_LIVE,
        stage_entered_at=datetime.now(timezone.utc) - timedelta(days=35),
        paper_trades=250,
        sharpe_ratio=0.9,
        max_drawdown=0.08,
        win_rate=0.60,
        total_pnl=15_000.0,
        allocation_pct=0.08,
    )


@pytest.fixture
def partial_live_record():
    """Record at Partial-Live stage meeting criteria."""
    return PromotionRecord(
        skill_name="momentum-v1",
        current_stage=PromotionStage.PARTIAL_LIVE,
        stage_entered_at=datetime.now(timezone.utc) - timedelta(days=65),
        paper_trades=350,
        sharpe_ratio=1.1,
        max_drawdown=0.07,
        win_rate=0.62,
        total_pnl=25_000.0,
        allocation_pct=0.35,
    )


# ─── Stage Definitions ──────────────────────────────────────────────


def test_promotion_stages_are_ordered():
    """Stages have a clear progression order."""
    assert PromotionStage.PAPER_TRAINING.value < PromotionStage.PAPER_VALIDATION.value
    assert PromotionStage.PAPER_VALIDATION.value < PromotionStage.MICRO_LIVE.value
    assert PromotionStage.MICRO_LIVE.value < PromotionStage.PARTIAL_LIVE.value
    assert PromotionStage.PARTIAL_LIVE.value < PromotionStage.FULL_LIVE.value


def test_default_stage_requirements(protocol):
    """Protocol has sensible default requirements per stage."""
    reqs = protocol.get_requirements(PromotionStage.PAPER_TRAINING)
    assert reqs.min_days == 90

    reqs = protocol.get_requirements(PromotionStage.PAPER_VALIDATION)
    assert reqs.min_days == 60
    assert reqs.min_sharpe == 0.5
    assert reqs.max_drawdown == 0.15

    reqs = protocol.get_requirements(PromotionStage.MICRO_LIVE)
    assert reqs.min_days == 30
    assert reqs.min_allocation == 0.05
    assert reqs.max_allocation == 0.10

    reqs = protocol.get_requirements(PromotionStage.PARTIAL_LIVE)
    assert reqs.min_days == 60
    assert reqs.min_allocation == 0.25
    assert reqs.max_allocation == 0.50


# ─── Promotion Decisions ─────────────────────────────────────────────


def test_promote_from_paper_training(protocol, paper_training_record):
    """Promotes from Paper Training after 90 days with sufficient trades."""
    decision = protocol.evaluate(paper_training_record)
    assert decision == PromotionDecision.PROMOTE
    next_stage = protocol.next_stage(paper_training_record.current_stage)
    assert next_stage == PromotionStage.PAPER_VALIDATION


def test_promote_from_paper_validation(protocol, paper_validation_record):
    """Promotes from Paper Validation when Sharpe > 0.5, DD < 15%."""
    decision = protocol.evaluate(paper_validation_record)
    assert decision == PromotionDecision.PROMOTE


def test_promote_from_micro_live(protocol, micro_live_record):
    """Promotes from Micro-Live after 30 days with good performance."""
    decision = protocol.evaluate(micro_live_record)
    assert decision == PromotionDecision.PROMOTE


def test_promote_from_partial_live(protocol, partial_live_record):
    """Promotes from Partial-Live to Full-Live after 60 days."""
    decision = protocol.evaluate(partial_live_record)
    assert decision == PromotionDecision.PROMOTE


def test_hold_when_time_insufficient(protocol):
    """Holds when minimum days not met."""
    record = PromotionRecord(
        skill_name="momentum-v1",
        current_stage=PromotionStage.PAPER_TRAINING,
        stage_entered_at=datetime.now(timezone.utc) - timedelta(days=30),
        paper_trades=50,
        sharpe_ratio=1.0,
        max_drawdown=0.05,
        win_rate=0.60,
        total_pnl=3_000.0,
        allocation_pct=0.0,
    )
    decision = protocol.evaluate(record)
    assert decision == PromotionDecision.HOLD


def test_hold_when_sharpe_insufficient(protocol):
    """Holds when Sharpe is below threshold."""
    record = PromotionRecord(
        skill_name="momentum-v1",
        current_stage=PromotionStage.PAPER_VALIDATION,
        stage_entered_at=datetime.now(timezone.utc) - timedelta(days=65),
        paper_trades=200,
        sharpe_ratio=0.3,  # Below 0.5 threshold
        max_drawdown=0.10,
        win_rate=0.52,
        total_pnl=1_000.0,
        allocation_pct=0.0,
    )
    decision = protocol.evaluate(record)
    assert decision == PromotionDecision.HOLD


# ─── Demotion (Bidirectional) ────────────────────────────────────────


def test_demote_on_drawdown_breach(protocol):
    """Demotes when drawdown exceeds stage limit."""
    record = PromotionRecord(
        skill_name="momentum-v1",
        current_stage=PromotionStage.MICRO_LIVE,
        stage_entered_at=datetime.now(timezone.utc) - timedelta(days=20),
        paper_trades=180,
        sharpe_ratio=0.2,
        max_drawdown=0.18,  # Exceeds threshold
        win_rate=0.45,
        total_pnl=-2_000.0,
        allocation_pct=0.08,
    )
    decision = protocol.evaluate(record)
    assert decision == PromotionDecision.DEMOTE


def test_demote_on_negative_sharpe(protocol):
    """Demotes when Sharpe goes negative during live stages."""
    record = PromotionRecord(
        skill_name="momentum-v1",
        current_stage=PromotionStage.PARTIAL_LIVE,
        stage_entered_at=datetime.now(timezone.utc) - timedelta(days=40),
        paper_trades=300,
        sharpe_ratio=-0.5,  # Negative Sharpe
        max_drawdown=0.12,
        win_rate=0.40,
        total_pnl=-5_000.0,
        allocation_pct=0.30,
    )
    decision = protocol.evaluate(record)
    assert decision == PromotionDecision.DEMOTE


def test_previous_stage_for_demotion(protocol):
    """Previous stage returns the correct demotion target."""
    assert protocol.previous_stage(PromotionStage.FULL_LIVE) == PromotionStage.PARTIAL_LIVE
    assert protocol.previous_stage(PromotionStage.PARTIAL_LIVE) == PromotionStage.MICRO_LIVE
    assert protocol.previous_stage(PromotionStage.MICRO_LIVE) == PromotionStage.PAPER_VALIDATION
    assert protocol.previous_stage(PromotionStage.PAPER_VALIDATION) == PromotionStage.PAPER_TRAINING
    assert protocol.previous_stage(PromotionStage.PAPER_TRAINING) == PromotionStage.PAPER_TRAINING


def test_demotion_from_full_live(protocol):
    """Full-Live demotes to Partial-Live on poor performance."""
    record = PromotionRecord(
        skill_name="momentum-v1",
        current_stage=PromotionStage.FULL_LIVE,
        stage_entered_at=datetime.now(timezone.utc) - timedelta(days=30),
        paper_trades=400,
        sharpe_ratio=-0.3,
        max_drawdown=0.16,
        win_rate=0.42,
        total_pnl=-8_000.0,
        allocation_pct=1.0,
    )
    decision = protocol.evaluate(record)
    assert decision == PromotionDecision.DEMOTE
    prev = protocol.previous_stage(record.current_stage)
    assert prev == PromotionStage.PARTIAL_LIVE


# ─── Allocation Ranges ──────────────────────────────────────────────


def test_allocation_range_paper_stages(protocol):
    """Paper stages have 0% allocation."""
    alloc = protocol.get_allocation_range(PromotionStage.PAPER_TRAINING)
    assert alloc == (0.0, 0.0)

    alloc = protocol.get_allocation_range(PromotionStage.PAPER_VALIDATION)
    assert alloc == (0.0, 0.0)


def test_allocation_range_micro_live(protocol):
    """Micro-Live: 5-10% allocation."""
    alloc = protocol.get_allocation_range(PromotionStage.MICRO_LIVE)
    assert alloc == (0.05, 0.10)


def test_allocation_range_partial_live(protocol):
    """Partial-Live: 25-50% allocation."""
    alloc = protocol.get_allocation_range(PromotionStage.PARTIAL_LIVE)
    assert alloc == (0.25, 0.50)


def test_allocation_range_full_live(protocol):
    """Full-Live: 100% allocation."""
    alloc = protocol.get_allocation_range(PromotionStage.FULL_LIVE)
    assert alloc == (1.0, 1.0)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_promotion_protocol.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.execution.promotion'`

**Step 3: Implement the promotion protocol**

```python
# src/evolve_trader/execution/promotion.py
"""Paper-to-Live Promotion Protocol.

5-stage pipeline with bidirectional demotion:

1. Paper Training (90 days) — Accumulate paper trading data
2. Paper Validation (60 days) — Must achieve Sharpe > 0.5, DD < 15%
3. Micro-Live (30 days) — 5-10% allocation with real money
4. Partial-Live (60 days) — 25-50% allocation
5. Full-Live — Full allocation, ongoing monitoring

Demotion triggers: negative Sharpe, excessive drawdown, or sustained losses.
Demotion drops one stage (never jumps multiple stages).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import IntEnum, Enum
from typing import Any


class PromotionStage(IntEnum):
    """Ordered stages in the paper-to-live pipeline."""
    PAPER_TRAINING = 1
    PAPER_VALIDATION = 2
    MICRO_LIVE = 3
    PARTIAL_LIVE = 4
    FULL_LIVE = 5


class PromotionDecision(str, Enum):
    """Outcome of a promotion evaluation."""
    PROMOTE = "promote"
    HOLD = "hold"
    DEMOTE = "demote"


class DemotionTrigger(str, Enum):
    """Reason for demotion."""
    NEGATIVE_SHARPE = "negative_sharpe"
    EXCESSIVE_DRAWDOWN = "excessive_drawdown"
    SUSTAINED_LOSSES = "sustained_losses"
    MANUAL_OVERRIDE = "manual_override"


@dataclass(frozen=True)
class StageRequirements:
    """Requirements to remain at or be promoted from a stage."""
    min_days: int
    min_sharpe: float = 0.0
    max_drawdown: float = 1.0
    min_paper_trades: int = 0
    min_allocation: float = 0.0
    max_allocation: float = 0.0
    demote_sharpe_below: float = -999.0  # Threshold for demotion
    demote_drawdown_above: float = 1.0   # Threshold for demotion


@dataclass
class PromotionRecord:
    """Current state of a skill in the promotion pipeline."""
    skill_name: str
    current_stage: PromotionStage
    stage_entered_at: datetime
    paper_trades: int
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_pnl: float
    allocation_pct: float
    demotion_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def days_in_stage(self) -> int:
        """Number of days spent in current stage."""
        delta = datetime.now(timezone.utc) - self.stage_entered_at
        return delta.days


# Default requirements per stage
_DEFAULT_REQUIREMENTS: dict[PromotionStage, StageRequirements] = {
    PromotionStage.PAPER_TRAINING: StageRequirements(
        min_days=90,
        min_paper_trades=50,
        min_allocation=0.0,
        max_allocation=0.0,
    ),
    PromotionStage.PAPER_VALIDATION: StageRequirements(
        min_days=60,
        min_sharpe=0.5,
        max_drawdown=0.15,
        min_paper_trades=100,
        min_allocation=0.0,
        max_allocation=0.0,
        demote_sharpe_below=-0.2,
        demote_drawdown_above=0.20,
    ),
    PromotionStage.MICRO_LIVE: StageRequirements(
        min_days=30,
        min_sharpe=0.3,
        max_drawdown=0.15,
        min_allocation=0.05,
        max_allocation=0.10,
        demote_sharpe_below=-0.1,
        demote_drawdown_above=0.15,
    ),
    PromotionStage.PARTIAL_LIVE: StageRequirements(
        min_days=60,
        min_sharpe=0.5,
        max_drawdown=0.12,
        min_allocation=0.25,
        max_allocation=0.50,
        demote_sharpe_below=0.0,
        demote_drawdown_above=0.15,
    ),
    PromotionStage.FULL_LIVE: StageRequirements(
        min_days=0,  # No promotion from full live
        min_sharpe=0.3,
        max_drawdown=0.15,
        min_allocation=1.0,
        max_allocation=1.0,
        demote_sharpe_below=0.0,
        demote_drawdown_above=0.15,
    ),
}

# Allocation ranges per stage
_ALLOCATION_RANGES: dict[PromotionStage, tuple[float, float]] = {
    PromotionStage.PAPER_TRAINING: (0.0, 0.0),
    PromotionStage.PAPER_VALIDATION: (0.0, 0.0),
    PromotionStage.MICRO_LIVE: (0.05, 0.10),
    PromotionStage.PARTIAL_LIVE: (0.25, 0.50),
    PromotionStage.FULL_LIVE: (1.0, 1.0),
}


class PromotionProtocol:
    """Evaluates promotion/demotion decisions for strategy skills.

    Manages the 5-stage paper-to-live pipeline with bidirectional movement.
    """

    def __init__(
        self,
        requirements: dict[PromotionStage, StageRequirements] | None = None,
    ):
        self._requirements = requirements or dict(_DEFAULT_REQUIREMENTS)

    def get_requirements(self, stage: PromotionStage) -> StageRequirements:
        """Get requirements for a stage."""
        return self._requirements[stage]

    def get_allocation_range(
        self, stage: PromotionStage,
    ) -> tuple[float, float]:
        """Get the min/max allocation range for a stage."""
        return _ALLOCATION_RANGES[stage]

    def next_stage(self, current: PromotionStage) -> PromotionStage | None:
        """Get the next stage after promotion."""
        if current == PromotionStage.FULL_LIVE:
            return None
        return PromotionStage(current.value + 1)

    def previous_stage(self, current: PromotionStage) -> PromotionStage:
        """Get the previous stage for demotion."""
        if current == PromotionStage.PAPER_TRAINING:
            return PromotionStage.PAPER_TRAINING
        return PromotionStage(current.value - 1)

    def evaluate(self, record: PromotionRecord) -> PromotionDecision:
        """Evaluate whether a skill should be promoted, held, or demoted.

        Args:
            record: Current state of the skill in the pipeline.

        Returns:
            PromotionDecision: PROMOTE, HOLD, or DEMOTE.
        """
        reqs = self._requirements[record.current_stage]

        # ── Check demotion triggers first (most urgent) ──
        if record.current_stage.value >= PromotionStage.PAPER_VALIDATION.value:
            if record.sharpe_ratio < reqs.demote_sharpe_below:
                return PromotionDecision.DEMOTE
            if record.max_drawdown > reqs.demote_drawdown_above:
                return PromotionDecision.DEMOTE

        # ── Check if enough time has passed ──
        if record.days_in_stage < reqs.min_days:
            return PromotionDecision.HOLD

        # ── Already at FULL_LIVE — can only demote or hold ──
        if record.current_stage == PromotionStage.FULL_LIVE:
            return PromotionDecision.HOLD

        # ── Check promotion criteria ──
        if record.sharpe_ratio < reqs.min_sharpe:
            return PromotionDecision.HOLD

        if record.max_drawdown > reqs.max_drawdown:
            return PromotionDecision.HOLD

        if record.paper_trades < reqs.min_paper_trades:
            return PromotionDecision.HOLD

        return PromotionDecision.PROMOTE
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_promotion_protocol.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/execution/promotion.py tests/unit/test_promotion_protocol.py
git commit -m "feat: paper-to-live promotion protocol — 5 stages with bidirectional demotion"
```

---

## Task 7: Notification Dispatcher

**Files:**
- Create: `src/evolve_trader/notifications/dispatcher.py`
- Create: `src/evolve_trader/notifications/formatters.py`
- Create: `tests/unit/test_notification_dispatcher.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_notification_dispatcher.py
"""Tests for multi-channel notification dispatcher and formatters."""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from evolve_trader.notifications.dispatcher import (
    NotificationDispatcher,
    NotificationChannel,
    NotificationPriority,
    NotificationResult,
    ChannelConfig,
)
from evolve_trader.notifications.formatters import (
    TradeIntentFormatter,
    ApprovalRequestFormatter,
    FormattedMessage,
)
from evolve_trader.execution.trade_intent import (
    TradeIntent,
    TradeDirection,
    GateResult,
    GateVerdict,
)


# ─── Fixtures ────────────────────────────────────────────────────────


def _make_intent(ticker="AAPL", confidence=0.85):
    return TradeIntent(
        intent_id="intent-001",
        ticker=ticker,
        direction=TradeDirection.BUY,
        quantity=10.0,
        order_type="market",
        strategy_skill="momentum-v1",
        strategy_lineage=["momentum-v1"],
        sizing_skill="kelly-fractional-v1",
        regime_label="risk-on",
        signal_sources=["edgar_13f", "form4_insider"],
        confidence=confidence,
        reasoning_chain="Strong institutional buying in risk-on regime.",
    )


@pytest.fixture
def mock_slack_channel():
    channel = AsyncMock(spec=NotificationChannel)
    channel.name = "slack"
    channel.is_enabled = True
    channel.send = AsyncMock(return_value=NotificationResult(
        channel="slack", success=True, message_id="slack-msg-001",
    ))
    return channel


@pytest.fixture
def mock_telegram_channel():
    channel = AsyncMock(spec=NotificationChannel)
    channel.name = "telegram"
    channel.is_enabled = True
    channel.send = AsyncMock(return_value=NotificationResult(
        channel="telegram", success=True, message_id="tg-msg-001",
    ))
    return channel


@pytest.fixture
def mock_email_channel():
    channel = AsyncMock(spec=NotificationChannel)
    channel.name = "email"
    channel.is_enabled = True
    channel.send = AsyncMock(return_value=NotificationResult(
        channel="email", success=True, message_id="email-msg-001",
    ))
    return channel


# ─── Dispatcher ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatcher_sends_to_all_enabled_channels(
    mock_slack_channel, mock_telegram_channel, mock_email_channel,
):
    """Dispatcher sends to all registered enabled channels."""
    dispatcher = NotificationDispatcher()
    dispatcher.register_channel(mock_slack_channel)
    dispatcher.register_channel(mock_telegram_channel)
    dispatcher.register_channel(mock_email_channel)

    message = FormattedMessage(
        subject="Trade Alert: BUY AAPL",
        body="Momentum strategy buying 10 shares of AAPL.",
        priority=NotificationPriority.NORMAL,
        metadata={"intent_id": "intent-001"},
    )

    results = await dispatcher.send(message)
    assert len(results) == 3
    assert all(r.success for r in results)
    mock_slack_channel.send.assert_called_once()
    mock_telegram_channel.send.assert_called_once()
    mock_email_channel.send.assert_called_once()


@pytest.mark.asyncio
async def test_dispatcher_skips_disabled_channels(
    mock_slack_channel, mock_telegram_channel,
):
    """Disabled channels are skipped."""
    mock_telegram_channel.is_enabled = False

    dispatcher = NotificationDispatcher()
    dispatcher.register_channel(mock_slack_channel)
    dispatcher.register_channel(mock_telegram_channel)

    message = FormattedMessage(
        subject="Trade Alert",
        body="Test message.",
        priority=NotificationPriority.NORMAL,
    )

    results = await dispatcher.send(message)
    assert len(results) == 1
    assert results[0].channel == "slack"
    mock_telegram_channel.send.assert_not_called()


@pytest.mark.asyncio
async def test_dispatcher_handles_channel_failure_gracefully(
    mock_slack_channel, mock_telegram_channel,
):
    """Channel failure doesn't prevent other channels from receiving."""
    mock_slack_channel.send = AsyncMock(side_effect=Exception("Slack down"))

    dispatcher = NotificationDispatcher()
    dispatcher.register_channel(mock_slack_channel)
    dispatcher.register_channel(mock_telegram_channel)

    message = FormattedMessage(
        subject="Trade Alert",
        body="Test message.",
        priority=NotificationPriority.NORMAL,
    )

    results = await dispatcher.send(message)
    assert len(results) == 2
    slack_result = next(r for r in results if r.channel == "slack")
    assert slack_result.success is False
    tg_result = next(r for r in results if r.channel == "telegram")
    assert tg_result.success is True


@pytest.mark.asyncio
async def test_dispatcher_respects_priority_routing(mock_slack_channel):
    """HIGH priority messages route to high-priority channels."""
    mock_slack_channel.supports_priority = True

    dispatcher = NotificationDispatcher()
    dispatcher.register_channel(mock_slack_channel)

    message = FormattedMessage(
        subject="URGENT: Drawdown Alert",
        body="Portfolio drawdown at 18%.",
        priority=NotificationPriority.HIGH,
    )

    results = await dispatcher.send(message)
    assert len(results) == 1


def test_dispatcher_list_channels(
    mock_slack_channel, mock_telegram_channel,
):
    """Dispatcher lists all registered channels."""
    dispatcher = NotificationDispatcher()
    dispatcher.register_channel(mock_slack_channel)
    dispatcher.register_channel(mock_telegram_channel)

    channels = dispatcher.list_channels()
    assert len(channels) == 2
    assert "slack" in channels
    assert "telegram" in channels


# ─── TradeIntentFormatter ────────────────────────────────────────────


def test_format_trade_intent():
    """Formats TradeIntent into structured notification message."""
    formatter = TradeIntentFormatter()
    intent = _make_intent()

    message = formatter.format(intent)
    assert isinstance(message, FormattedMessage)
    assert "AAPL" in message.subject
    assert "BUY" in message.subject
    assert "momentum-v1" in message.body
    assert "0.85" in message.body or "85" in message.body
    assert "edgar_13f" in message.body


def test_format_trade_intent_with_gate_results():
    """Format includes gate results when present."""
    formatter = TradeIntentFormatter()
    intent = _make_intent()

    gate = GateResult(
        gate_name="risk_constraints",
        verdict=GateVerdict.PASS,
        details={"position_pct": 0.03},
        checked_at=datetime.now(timezone.utc),
    )
    intent.add_gate_result(gate)

    message = formatter.format(intent)
    assert "risk_constraints" in message.body
    assert "PASS" in message.body


# ─── ApprovalRequestFormatter ───────────────────────────────────────


def test_format_approval_request():
    """Formats an approval request with action buttons metadata."""
    formatter = ApprovalRequestFormatter()
    intent = _make_intent()

    message = formatter.format(intent, timeout_hours=4)
    assert isinstance(message, FormattedMessage)
    assert "APPROVE" in message.body or "approve" in message.body.lower()
    assert "REJECT" in message.body or "reject" in message.body.lower()
    assert message.priority == NotificationPriority.HIGH
    assert "timeout" in message.metadata
    assert message.metadata["timeout"] == 4
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_notification_dispatcher.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.notifications.dispatcher'`

**Step 3: Implement the notification dispatcher and formatters**

```python
# src/evolve_trader/notifications/__init__.py
"""Notification system for trade alerts and approval workflows."""
```

```python
# src/evolve_trader/notifications/formatters.py
"""Notification message formatters for trade intents and approval requests."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from evolve_trader.execution.trade_intent import TradeIntent


class NotificationPriority(str, Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FormattedMessage:
    """Structured notification message ready for dispatch."""
    subject: str
    body: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    metadata: dict[str, Any] = field(default_factory=dict)
    actions: list[dict[str, str]] = field(default_factory=list)


class TradeIntentFormatter:
    """Formats TradeIntent into human-readable notification messages."""

    def format(self, intent: TradeIntent) -> FormattedMessage:
        """Format a TradeIntent for notification.

        Args:
            intent: The trade intent to format.

        Returns:
            FormattedMessage with structured trade details.
        """
        subject = (
            f"Trade Alert: {intent.direction.value} {intent.ticker} "
            f"({intent.quantity} shares)"
        )

        lines = [
            f"Strategy: {intent.strategy_skill}",
            f"Direction: {intent.direction.value}",
            f"Ticker: {intent.ticker}",
            f"Quantity: {intent.quantity}",
            f"Order Type: {intent.order_type}",
            f"Confidence: {intent.confidence:.2f}",
            f"Regime: {intent.regime_label}",
            f"Signals: {', '.join(intent.signal_sources)}",
            f"Sizing: {intent.sizing_skill}",
            f"Lineage: {' -> '.join(intent.strategy_lineage)}",
            "",
            f"Reasoning: {intent.reasoning_chain}",
        ]

        if intent.gate_results:
            lines.append("")
            lines.append("Gate Results:")
            for gr in intent.gate_results:
                lines.append(f"  - {gr.gate_name}: {gr.verdict.value}")
                if gr.reason:
                    lines.append(f"    Reason: {gr.reason}")

        body = "\n".join(lines)

        return FormattedMessage(
            subject=subject,
            body=body,
            priority=NotificationPriority.NORMAL,
            metadata={"intent_id": intent.intent_id, "ticker": intent.ticker},
        )


class ApprovalRequestFormatter:
    """Formats TradeIntent as an approval request with action buttons."""

    def format(
        self,
        intent: TradeIntent,
        timeout_hours: int = 4,
    ) -> FormattedMessage:
        """Format an approval request notification.

        Args:
            intent: The trade intent requiring approval.
            timeout_hours: Hours before auto-reject.

        Returns:
            FormattedMessage with approval actions and timeout.
        """
        subject = (
            f"Approval Required: {intent.direction.value} {intent.ticker} "
            f"({intent.quantity} shares)"
        )

        lines = [
            f"Trade requires your approval.",
            "",
            f"Strategy: {intent.strategy_skill}",
            f"Direction: {intent.direction.value}",
            f"Ticker: {intent.ticker}",
            f"Quantity: {intent.quantity}",
            f"Confidence: {intent.confidence:.2f}",
            f"Regime: {intent.regime_label}",
            f"Signals: {', '.join(intent.signal_sources)}",
            "",
            f"Reasoning: {intent.reasoning_chain}",
            "",
            f"Actions: [APPROVE] [REJECT]",
            f"Timeout: {timeout_hours}h (auto-reject if no response)",
        ]

        body = "\n".join(lines)

        return FormattedMessage(
            subject=subject,
            body=body,
            priority=NotificationPriority.HIGH,
            metadata={
                "intent_id": intent.intent_id,
                "ticker": intent.ticker,
                "timeout": timeout_hours,
                "requires_approval": True,
            },
            actions=[
                {"action": "approve", "label": "APPROVE", "style": "primary"},
                {"action": "reject", "label": "REJECT", "style": "danger"},
            ],
        )
```

```python
# src/evolve_trader/notifications/dispatcher.py
"""Multi-channel notification dispatcher.

Routes formatted messages to all registered and enabled channels.
Handles individual channel failures gracefully — one channel failing
does not prevent others from receiving the notification.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from evolve_trader.notifications.formatters import (
    FormattedMessage,
    NotificationPriority,
)

logger = logging.getLogger(__name__)


# Re-export for convenience
__all__ = [
    "NotificationDispatcher",
    "NotificationChannel",
    "NotificationPriority",
    "NotificationResult",
    "ChannelConfig",
]


@dataclass
class NotificationResult:
    """Result of sending a notification to a single channel."""
    channel: str
    success: bool
    message_id: str | None = None
    error: str | None = None


@dataclass
class ChannelConfig:
    """Base configuration for a notification channel."""
    enabled: bool = True
    priority_filter: NotificationPriority | None = None  # Only send >= this priority


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Channel identifier."""
        ...

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        """Whether this channel is currently enabled."""
        ...

    @abstractmethod
    async def send(self, message: FormattedMessage) -> NotificationResult:
        """Send a formatted message through this channel."""
        ...


class NotificationDispatcher:
    """Routes notifications to all registered enabled channels.

    Features:
    - Multi-channel dispatch (Slack, Telegram, email, dashboard)
    - Graceful failure handling per channel
    - Priority-based routing
    - Channel registration and discovery
    """

    def __init__(self):
        self._channels: dict[str, NotificationChannel] = {}

    def register_channel(self, channel: NotificationChannel) -> None:
        """Register a notification channel."""
        self._channels[channel.name] = channel

    def unregister_channel(self, name: str) -> None:
        """Remove a notification channel."""
        self._channels.pop(name, None)

    def list_channels(self) -> list[str]:
        """List all registered channel names."""
        return list(self._channels.keys())

    async def send(
        self,
        message: FormattedMessage,
    ) -> list[NotificationResult]:
        """Send a message to all enabled channels.

        Args:
            message: The formatted message to dispatch.

        Returns:
            List of NotificationResults, one per channel attempted.
        """
        results: list[NotificationResult] = []

        tasks = []
        for name, channel in self._channels.items():
            if not channel.is_enabled:
                continue
            tasks.append(self._send_to_channel(channel, message))

        if tasks:
            results = await asyncio.gather(*tasks)

        return list(results)

    async def _send_to_channel(
        self,
        channel: NotificationChannel,
        message: FormattedMessage,
    ) -> NotificationResult:
        """Send to a single channel with error handling."""
        try:
            return await channel.send(message)
        except Exception as e:
            logger.error(f"Failed to send to {channel.name}: {e}")
            return NotificationResult(
                channel=channel.name,
                success=False,
                error=str(e),
            )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_notification_dispatcher.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/notifications/ tests/unit/test_notification_dispatcher.py
git commit -m "feat: multi-channel notification dispatcher with trade intent formatters"
```

---

## Task 8: Slack Integration

**Files:**
- Create: `src/evolve_trader/notifications/channels/slack.py`
- Create: `tests/unit/test_slack_notifications.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_slack_notifications.py
"""Tests for Slack notification channel with approval workflow."""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock

from evolve_trader.notifications.channels.slack import (
    SlackChannel,
    SlackConfig,
    SlackBlockBuilder,
)
from evolve_trader.notifications.formatters import (
    FormattedMessage,
    NotificationPriority,
)
from evolve_trader.notifications.dispatcher import NotificationResult


# ─── SlackConfig ─────────────────────────────────────────────────────


def test_slack_config_requires_webhook_url():
    """Config requires a webhook URL."""
    config = SlackConfig(webhook_url="https://hooks.slack.com/services/T00/B00/xxx")
    assert config.webhook_url.startswith("https://hooks.slack.com")


def test_slack_config_from_env(monkeypatch):
    """Config loads from environment."""
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/T00/B00/xxx")
    monkeypatch.setenv("SLACK_CHANNEL", "#trading-alerts")
    config = SlackConfig.from_env()
    assert config.webhook_url == "https://hooks.slack.com/services/T00/B00/xxx"
    assert config.channel == "#trading-alerts"


# ─── SlackChannel ────────────────────────────────────────────────────


@pytest.fixture
def slack_config():
    return SlackConfig(
        webhook_url="https://hooks.slack.com/services/T00/B00/xxx",
        channel="#trading-alerts",
    )


@pytest.fixture
def slack_channel(slack_config):
    return SlackChannel(config=slack_config)


def test_slack_channel_name(slack_channel):
    assert slack_channel.name == "slack"


def test_slack_channel_enabled(slack_channel):
    assert slack_channel.is_enabled is True


@pytest.mark.asyncio
async def test_slack_sends_trade_alert(slack_channel):
    """Sends a trade alert via webhook."""
    message = FormattedMessage(
        subject="Trade Alert: BUY AAPL",
        body="Momentum strategy buying 10 shares.",
        priority=NotificationPriority.NORMAL,
        metadata={"intent_id": "intent-001"},
    )

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "ok"
        mock_post.return_value = mock_response

        result = await slack_channel.send(message)
        assert result.success is True
        assert result.channel == "slack"
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_slack_handles_webhook_failure(slack_channel):
    """Returns failure result on webhook error."""
    message = FormattedMessage(
        subject="Trade Alert",
        body="Test.",
        priority=NotificationPriority.NORMAL,
    )

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "server_error"
        mock_post.return_value = mock_response

        result = await slack_channel.send(message)
        assert result.success is False
        assert "500" in result.error


# ─── SlackBlockBuilder ──────────────────────────────────────────────


def test_block_builder_trade_alert():
    """Builds Slack Block Kit payload for trade alert."""
    builder = SlackBlockBuilder()
    message = FormattedMessage(
        subject="Trade Alert: BUY AAPL",
        body="Momentum strategy buying 10 shares.",
        priority=NotificationPriority.NORMAL,
    )
    blocks = builder.build_trade_alert(message)
    assert isinstance(blocks, list)
    assert len(blocks) > 0
    # Should contain a header block
    assert any(b.get("type") == "header" for b in blocks)


def test_block_builder_approval_request():
    """Builds Slack Block Kit payload with APPROVE/REJECT buttons."""
    builder = SlackBlockBuilder()
    message = FormattedMessage(
        subject="Approval Required: BUY AAPL",
        body="Trade requires approval.",
        priority=NotificationPriority.HIGH,
        metadata={"intent_id": "intent-001", "requires_approval": True},
        actions=[
            {"action": "approve", "label": "APPROVE", "style": "primary"},
            {"action": "reject", "label": "REJECT", "style": "danger"},
        ],
    )
    blocks = builder.build_approval_request(message)
    assert isinstance(blocks, list)
    # Should contain an actions block with buttons
    action_blocks = [b for b in blocks if b.get("type") == "actions"]
    assert len(action_blocks) == 1
    buttons = action_blocks[0].get("elements", [])
    assert len(buttons) == 2
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_slack_notifications.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.notifications.channels.slack'`

**Step 3: Implement the Slack channel**

```python
# src/evolve_trader/notifications/channels/__init__.py
"""Notification channel implementations."""
```

```python
# src/evolve_trader/notifications/channels/slack.py
"""Slack notification channel with webhook delivery and Block Kit formatting.

Supports:
- Trade alert notifications via incoming webhooks
- Interactive approval workflow with APPROVE/REJECT buttons
- Block Kit rich formatting
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from evolve_trader.notifications.dispatcher import (
    NotificationChannel,
    NotificationResult,
)
from evolve_trader.notifications.formatters import (
    FormattedMessage,
    NotificationPriority,
)

logger = logging.getLogger(__name__)


@dataclass
class SlackConfig:
    """Configuration for Slack notification channel."""
    webhook_url: str
    channel: str = "#trading-alerts"
    username: str = "Evolve-Trader"
    icon_emoji: str = ":chart_with_upwards_trend:"
    enabled: bool = True

    @classmethod
    def from_env(cls) -> SlackConfig:
        """Load config from environment variables."""
        return cls(
            webhook_url=os.environ["SLACK_WEBHOOK_URL"],
            channel=os.environ.get("SLACK_CHANNEL", "#trading-alerts"),
        )


class SlackBlockBuilder:
    """Builds Slack Block Kit payloads for different message types."""

    def build_trade_alert(self, message: FormattedMessage) -> list[dict[str, Any]]:
        """Build Block Kit blocks for a trade alert.

        Args:
            message: The formatted message to render.

        Returns:
            List of Slack Block Kit block objects.
        """
        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": message.subject[:150],
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message.body[:3000],
                },
            },
            {"type": "divider"},
        ]

        if message.metadata:
            context_elements = []
            for key, value in message.metadata.items():
                context_elements.append({
                    "type": "mrkdwn",
                    "text": f"*{key}:* {value}",
                })
            if context_elements:
                blocks.append({
                    "type": "context",
                    "elements": context_elements[:10],
                })

        return blocks

    def build_approval_request(
        self, message: FormattedMessage,
    ) -> list[dict[str, Any]]:
        """Build Block Kit blocks with interactive approval buttons.

        Args:
            message: The formatted approval request message.

        Returns:
            List of Slack Block Kit block objects including action buttons.
        """
        blocks = self.build_trade_alert(message)

        # Add action buttons
        buttons = []
        for action in message.actions:
            button: dict[str, Any] = {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": action.get("label", action["action"].upper()),
                },
                "action_id": f"trade_{action['action']}",
                "value": json.dumps({
                    "intent_id": message.metadata.get("intent_id"),
                    "action": action["action"],
                }),
            }
            if action.get("style") == "primary":
                button["style"] = "primary"
            elif action.get("style") == "danger":
                button["style"] = "danger"
            buttons.append(button)

        if buttons:
            blocks.append({
                "type": "actions",
                "elements": buttons,
            })

        return blocks


class SlackChannel(NotificationChannel):
    """Slack notification channel using incoming webhooks.

    Sends formatted messages to Slack via webhook with Block Kit formatting.
    Supports interactive approval workflows via Slack's action buttons.
    """

    def __init__(self, config: SlackConfig):
        self._config = config
        self._block_builder = SlackBlockBuilder()

    @property
    def name(self) -> str:
        return "slack"

    @property
    def is_enabled(self) -> bool:
        return self._config.enabled

    async def send(self, message: FormattedMessage) -> NotificationResult:
        """Send a message to Slack via webhook.

        Args:
            message: The formatted message to send.

        Returns:
            NotificationResult indicating success or failure.
        """
        is_approval = message.metadata.get("requires_approval", False)

        if is_approval:
            blocks = self._block_builder.build_approval_request(message)
        else:
            blocks = self._block_builder.build_trade_alert(message)

        payload = {
            "channel": self._config.channel,
            "username": self._config.username,
            "icon_emoji": self._config.icon_emoji,
            "blocks": blocks,
            "text": message.subject,  # Fallback for notifications
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._config.webhook_url,
                    json=payload,
                    timeout=10.0,
                )

            if response.status_code == 200:
                return NotificationResult(
                    channel="slack",
                    success=True,
                    message_id=f"slack-{id(response)}",
                )
            else:
                return NotificationResult(
                    channel="slack",
                    success=False,
                    error=f"Slack webhook returned {response.status_code}: {response.text}",
                )

        except Exception as e:
            logger.error(f"Slack send failed: {e}")
            return NotificationResult(
                channel="slack",
                success=False,
                error=str(e),
            )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_slack_notifications.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/notifications/channels/ tests/unit/test_slack_notifications.py
git commit -m "feat: Slack notification channel with Block Kit formatting and approval buttons"
```

---

## Task 9: Telegram Integration

**Files:**
- Create: `src/evolve_trader/notifications/channels/telegram.py`
- Create: `tests/unit/test_telegram_notifications.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_telegram_notifications.py
"""Tests for Telegram notification channel with bot approval workflow."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from evolve_trader.notifications.channels.telegram import (
    TelegramChannel,
    TelegramConfig,
    TelegramMessageBuilder,
)
from evolve_trader.notifications.formatters import (
    FormattedMessage,
    NotificationPriority,
)
from evolve_trader.notifications.dispatcher import NotificationResult


# ─── TelegramConfig ─────────────────────────────────────────────────


def test_telegram_config_requires_bot_token_and_chat_id():
    """Config requires both bot token and chat ID."""
    config = TelegramConfig(bot_token="123456:ABC-DEF", chat_id="-1001234567890")
    assert config.bot_token == "123456:ABC-DEF"
    assert config.chat_id == "-1001234567890"


def test_telegram_config_from_env(monkeypatch):
    """Config loads from environment."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:ABC-DEF")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-1001234567890")
    config = TelegramConfig.from_env()
    assert config.bot_token == "123456:ABC-DEF"


# ─── TelegramChannel ────────────────────────────────────────────────


@pytest.fixture
def telegram_config():
    return TelegramConfig(bot_token="123456:ABC-DEF", chat_id="-1001234567890")


@pytest.fixture
def telegram_channel(telegram_config):
    return TelegramChannel(config=telegram_config)


def test_telegram_channel_name(telegram_channel):
    assert telegram_channel.name == "telegram"


def test_telegram_channel_enabled(telegram_channel):
    assert telegram_channel.is_enabled is True


@pytest.mark.asyncio
async def test_telegram_sends_message(telegram_channel):
    """Sends message via Telegram Bot API."""
    message = FormattedMessage(
        subject="Trade Alert: BUY AAPL",
        body="Momentum strategy buying 10 shares.",
        priority=NotificationPriority.NORMAL,
    )

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True, "result": {"message_id": 42}}
        mock_post.return_value = mock_response

        result = await telegram_channel.send(message)
        assert result.success is True
        assert result.channel == "telegram"


@pytest.mark.asyncio
async def test_telegram_handles_api_failure(telegram_channel):
    """Returns failure on API error."""
    message = FormattedMessage(
        subject="Trade Alert",
        body="Test.",
        priority=NotificationPriority.NORMAL,
    )

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.json.return_value = {"ok": False, "description": "Forbidden"}
        mock_post.return_value = mock_response

        result = await telegram_channel.send(message)
        assert result.success is False


@pytest.mark.asyncio
async def test_telegram_sends_approval_with_inline_keyboard(telegram_channel):
    """Approval requests include inline keyboard buttons."""
    message = FormattedMessage(
        subject="Approval Required: BUY AAPL",
        body="Trade requires approval.",
        priority=NotificationPriority.HIGH,
        metadata={"intent_id": "intent-001", "requires_approval": True},
        actions=[
            {"action": "approve", "label": "APPROVE", "style": "primary"},
            {"action": "reject", "label": "REJECT", "style": "danger"},
        ],
    )

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True, "result": {"message_id": 43}}
        mock_post.return_value = mock_response

        result = await telegram_channel.send(message)
        assert result.success is True

        # Verify inline keyboard was included in the payload
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1].get("json", call_kwargs[0][1] if len(call_kwargs[0]) > 1 else {})
        if isinstance(payload, dict):
            assert "reply_markup" in payload


# ─── TelegramMessageBuilder ─────────────────────────────────────────


def test_message_builder_formats_html():
    """Builder formats message as HTML for Telegram."""
    builder = TelegramMessageBuilder()
    message = FormattedMessage(
        subject="Trade Alert: BUY AAPL",
        body="Strategy: momentum-v1\nConfidence: 0.85",
        priority=NotificationPriority.NORMAL,
    )
    html = builder.build_html(message)
    assert "<b>" in html
    assert "AAPL" in html


def test_message_builder_inline_keyboard():
    """Builder creates inline keyboard for approval requests."""
    builder = TelegramMessageBuilder()
    actions = [
        {"action": "approve", "label": "APPROVE"},
        {"action": "reject", "label": "REJECT"},
    ]
    keyboard = builder.build_inline_keyboard(actions, intent_id="intent-001")
    assert "inline_keyboard" in keyboard
    assert len(keyboard["inline_keyboard"][0]) == 2
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_telegram_notifications.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.notifications.channels.telegram'`

**Step 3: Implement the Telegram channel**

```python
# src/evolve_trader/notifications/channels/telegram.py
"""Telegram notification channel with Bot API delivery and inline keyboards.

Supports:
- HTML-formatted trade alerts via Bot API sendMessage
- Interactive approval workflow with inline keyboard buttons
- Callback query handling for APPROVE/REJECT responses
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

from evolve_trader.notifications.dispatcher import (
    NotificationChannel,
    NotificationResult,
)
from evolve_trader.notifications.formatters import FormattedMessage

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"


@dataclass
class TelegramConfig:
    """Configuration for Telegram Bot notification channel."""
    bot_token: str
    chat_id: str
    enabled: bool = True
    parse_mode: str = "HTML"

    @classmethod
    def from_env(cls) -> TelegramConfig:
        """Load config from environment variables."""
        return cls(
            bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
            chat_id=os.environ["TELEGRAM_CHAT_ID"],
        )

    @property
    def api_url(self) -> str:
        return f"{TELEGRAM_API_BASE}/bot{self.bot_token}"


class TelegramMessageBuilder:
    """Builds Telegram-formatted messages and inline keyboards."""

    def build_html(self, message: FormattedMessage) -> str:
        """Format message as HTML for Telegram.

        Args:
            message: The formatted message to render.

        Returns:
            HTML string for Telegram's sendMessage API.
        """
        lines = [
            f"<b>{self._escape_html(message.subject)}</b>",
            "",
        ]

        for line in message.body.split("\n"):
            lines.append(self._escape_html(line))

        return "\n".join(lines)

    def build_inline_keyboard(
        self,
        actions: list[dict[str, str]],
        intent_id: str,
    ) -> dict[str, Any]:
        """Build an inline keyboard for approval buttons.

        Args:
            actions: List of action dicts with 'action' and 'label' keys.
            intent_id: The trade intent ID for callback data.

        Returns:
            Telegram InlineKeyboardMarkup dict.
        """
        buttons = []
        for action in actions:
            callback_data = json.dumps({
                "intent_id": intent_id,
                "action": action["action"],
            })
            buttons.append({
                "text": action.get("label", action["action"].upper()),
                "callback_data": callback_data[:64],  # Telegram limit
            })

        return {"inline_keyboard": [buttons]}

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters for Telegram."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )


class TelegramChannel(NotificationChannel):
    """Telegram notification channel using Bot API.

    Sends formatted messages via sendMessage with HTML formatting.
    Approval requests include inline keyboard buttons for APPROVE/REJECT.
    """

    def __init__(self, config: TelegramConfig):
        self._config = config
        self._builder = TelegramMessageBuilder()

    @property
    def name(self) -> str:
        return "telegram"

    @property
    def is_enabled(self) -> bool:
        return self._config.enabled

    async def send(self, message: FormattedMessage) -> NotificationResult:
        """Send a message via Telegram Bot API.

        Args:
            message: The formatted message to send.

        Returns:
            NotificationResult indicating success or failure.
        """
        html_text = self._builder.build_html(message)

        payload: dict[str, Any] = {
            "chat_id": self._config.chat_id,
            "text": html_text,
            "parse_mode": self._config.parse_mode,
        }

        # Add inline keyboard for approval requests
        is_approval = message.metadata.get("requires_approval", False)
        if is_approval and message.actions:
            intent_id = message.metadata.get("intent_id", "unknown")
            keyboard = self._builder.build_inline_keyboard(
                message.actions, intent_id=intent_id,
            )
            payload["reply_markup"] = keyboard

        try:
            url = f"{self._config.api_url}/sendMessage"
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=10.0)

            data = response.json()
            if response.status_code == 200 and data.get("ok"):
                msg_id = str(data.get("result", {}).get("message_id", ""))
                return NotificationResult(
                    channel="telegram",
                    success=True,
                    message_id=f"tg-{msg_id}",
                )
            else:
                desc = data.get("description", "Unknown error")
                return NotificationResult(
                    channel="telegram",
                    success=False,
                    error=f"Telegram API error {response.status_code}: {desc}",
                )

        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return NotificationResult(
                channel="telegram",
                success=False,
                error=str(e),
            )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_telegram_notifications.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/notifications/channels/telegram.py tests/unit/test_telegram_notifications.py
git commit -m "feat: Telegram notification channel with inline keyboard approval buttons"
```

---

## Task 10: Email Integration

**Files:**
- Create: `src/evolve_trader/notifications/channels/email.py`
- Create: `tests/unit/test_email_notifications.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_email_notifications.py
"""Tests for email notification channel via SMTP."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from evolve_trader.notifications.channels.email import (
    EmailChannel,
    EmailConfig,
    EmailBuilder,
)
from evolve_trader.notifications.formatters import (
    FormattedMessage,
    NotificationPriority,
)
from evolve_trader.notifications.dispatcher import NotificationResult


# ─── EmailConfig ─────────────────────────────────────────────────────


def test_email_config_defaults():
    """Config has sensible SMTP defaults."""
    config = EmailConfig(
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        username="trader@example.com",
        password="app-password",
        from_addr="trader@example.com",
        to_addrs=["alerts@example.com"],
    )
    assert config.smtp_port == 587
    assert config.use_tls is True


def test_email_config_from_env(monkeypatch):
    """Config loads from environment variables."""
    monkeypatch.setenv("SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USERNAME", "trader@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-password")
    monkeypatch.setenv("EMAIL_FROM", "trader@example.com")
    monkeypatch.setenv("EMAIL_TO", "alerts@example.com,backup@example.com")
    config = EmailConfig.from_env()
    assert config.smtp_host == "smtp.gmail.com"
    assert len(config.to_addrs) == 2


# ─── EmailChannel ────────────────────────────────────────────────────


@pytest.fixture
def email_config():
    return EmailConfig(
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        username="trader@example.com",
        password="app-password",
        from_addr="trader@example.com",
        to_addrs=["alerts@example.com"],
    )


@pytest.fixture
def email_channel(email_config):
    return EmailChannel(config=email_config)


def test_email_channel_name(email_channel):
    assert email_channel.name == "email"


def test_email_channel_enabled(email_channel):
    assert email_channel.is_enabled is True


@pytest.mark.asyncio
async def test_email_sends_message(email_channel):
    """Sends email via SMTP."""
    message = FormattedMessage(
        subject="Trade Alert: BUY AAPL",
        body="Momentum strategy buying 10 shares.",
        priority=NotificationPriority.NORMAL,
    )

    with patch("smtplib.SMTP") as mock_smtp_cls:
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = await email_channel.send(message)
        assert result.success is True
        assert result.channel == "email"


@pytest.mark.asyncio
async def test_email_handles_smtp_failure(email_channel):
    """Returns failure on SMTP error."""
    message = FormattedMessage(
        subject="Trade Alert",
        body="Test.",
        priority=NotificationPriority.NORMAL,
    )

    with patch("smtplib.SMTP") as mock_smtp_cls:
        mock_smtp_cls.side_effect = ConnectionRefusedError("SMTP refused")

        result = await email_channel.send(message)
        assert result.success is False
        assert "SMTP" in result.error or "refused" in result.error.lower()


# ─── EmailBuilder ────────────────────────────────────────────────────


def test_email_builder_creates_html_body():
    """Builder creates HTML email body."""
    builder = EmailBuilder()
    message = FormattedMessage(
        subject="Trade Alert: BUY AAPL",
        body="Strategy: momentum-v1\nConfidence: 0.85\nRegime: risk-on",
        priority=NotificationPriority.NORMAL,
    )
    html = builder.build_html(message)
    assert "<html>" in html
    assert "AAPL" in html
    assert "momentum-v1" in html


def test_email_builder_high_priority_subject():
    """High priority messages get subject prefix."""
    builder = EmailBuilder()
    message = FormattedMessage(
        subject="Drawdown Alert",
        body="Portfolio drawdown at 18%.",
        priority=NotificationPriority.HIGH,
    )
    subject = builder.build_subject(message)
    assert "URGENT" in subject or "HIGH" in subject


def test_email_builder_normal_priority_no_prefix():
    """Normal priority messages have no prefix."""
    builder = EmailBuilder()
    message = FormattedMessage(
        subject="Trade Alert: BUY AAPL",
        body="Test.",
        priority=NotificationPriority.NORMAL,
    )
    subject = builder.build_subject(message)
    assert subject == "Trade Alert: BUY AAPL"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_email_notifications.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.notifications.channels.email'`

**Step 3: Implement the email channel**

```python
# src/evolve_trader/notifications/channels/email.py
"""Email notification channel via SMTP.

Supports:
- HTML-formatted trade alert emails
- Priority-based subject prefixing
- TLS encryption
- Multiple recipient addresses
"""
from __future__ import annotations

import logging
import os
import smtplib
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from evolve_trader.notifications.dispatcher import (
    NotificationChannel,
    NotificationResult,
)
from evolve_trader.notifications.formatters import (
    FormattedMessage,
    NotificationPriority,
)

logger = logging.getLogger(__name__)


@dataclass
class EmailConfig:
    """Configuration for SMTP email notifications."""
    smtp_host: str
    smtp_port: int
    username: str
    password: str
    from_addr: str
    to_addrs: list[str]
    use_tls: bool = True
    enabled: bool = True

    @classmethod
    def from_env(cls) -> EmailConfig:
        """Load config from environment variables."""
        to_addrs = os.environ["EMAIL_TO"].split(",")
        return cls(
            smtp_host=os.environ["SMTP_HOST"],
            smtp_port=int(os.environ.get("SMTP_PORT", "587")),
            username=os.environ["SMTP_USERNAME"],
            password=os.environ["SMTP_PASSWORD"],
            from_addr=os.environ["EMAIL_FROM"],
            to_addrs=[addr.strip() for addr in to_addrs],
        )


class EmailBuilder:
    """Builds HTML email content from formatted messages."""

    _PRIORITY_PREFIXES = {
        NotificationPriority.HIGH: "[URGENT] ",
        NotificationPriority.CRITICAL: "[CRITICAL] ",
    }

    def build_subject(self, message: FormattedMessage) -> str:
        """Build email subject with optional priority prefix.

        Args:
            message: The formatted message.

        Returns:
            Subject line string.
        """
        prefix = self._PRIORITY_PREFIXES.get(message.priority, "")
        return f"{prefix}{message.subject}"

    def build_html(self, message: FormattedMessage) -> str:
        """Build HTML email body.

        Args:
            message: The formatted message.

        Returns:
            HTML string for email body.
        """
        body_html = message.body.replace("\n", "<br>\n")

        return f"""<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        .header {{ background-color: #1a1a2e; color: white; padding: 15px; border-radius: 5px; }}
        .content {{ padding: 15px; border: 1px solid #ddd; border-radius: 5px; margin-top: 10px; }}
        .footer {{ color: #666; font-size: 12px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>{message.subject}</h2>
    </div>
    <div class="content">
        {body_html}
    </div>
    <div class="footer">
        Evolve-Trader AI Notification System
    </div>
</body>
</html>"""


class EmailChannel(NotificationChannel):
    """Email notification channel via SMTP.

    Sends HTML-formatted trade alerts and notifications via SMTP.
    Uses TLS encryption by default.
    """

    def __init__(self, config: EmailConfig):
        self._config = config
        self._builder = EmailBuilder()

    @property
    def name(self) -> str:
        return "email"

    @property
    def is_enabled(self) -> bool:
        return self._config.enabled

    async def send(self, message: FormattedMessage) -> NotificationResult:
        """Send an email notification via SMTP.

        Args:
            message: The formatted message to send.

        Returns:
            NotificationResult indicating success or failure.
        """
        try:
            subject = self._builder.build_subject(message)
            html_body = self._builder.build_html(message)

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self._config.from_addr
            msg["To"] = ", ".join(self._config.to_addrs)

            # Plain text fallback
            msg.attach(MIMEText(message.body, "plain"))
            # HTML version
            msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(self._config.smtp_host, self._config.smtp_port) as server:
                if self._config.use_tls:
                    server.starttls()
                server.login(self._config.username, self._config.password)
                server.sendmail(
                    self._config.from_addr,
                    self._config.to_addrs,
                    msg.as_string(),
                )

            return NotificationResult(
                channel="email",
                success=True,
                message_id=f"email-{id(msg)}",
            )

        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return NotificationResult(
                channel="email",
                success=False,
                error=str(e),
            )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_email_notifications.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/notifications/channels/email.py tests/unit/test_email_notifications.py
git commit -m "feat: email notification channel via SMTP with HTML formatting"
```

---

## Task 11: Approval Workflow Engine

**Files:**
- Create: `src/evolve_trader/notifications/approval.py`
- Create: `tests/unit/test_approval_workflow.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_approval_workflow.py
"""Tests for Approval Workflow Engine.

Tracks pending approvals, enforces timeouts, and routes decisions.
- Configurable timeout (4h default)
- No response = auto-reject
- Per-user notification preferences
"""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock

from evolve_trader.notifications.approval import (
    ApprovalWorkflow,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
    ApprovalTimeout,
    NotificationPreferences,
)
from evolve_trader.execution.trade_intent import (
    TradeIntent,
    TradeDirection,
    IntentStatus,
)


# ─── Fixtures ────────────────────────────────────────────────────────


def _make_intent(intent_id="intent-001"):
    return TradeIntent(
        intent_id=intent_id,
        ticker="AAPL",
        direction=TradeDirection.BUY,
        quantity=10.0,
        order_type="market",
        strategy_skill="momentum-v1",
        strategy_lineage=["momentum-v1"],
        sizing_skill="kelly-fractional-v1",
        regime_label="risk-on",
        signal_sources=["edgar_13f"],
        confidence=0.75,
        reasoning_chain="Momentum signal.",
    )


@pytest.fixture
def workflow():
    return ApprovalWorkflow(default_timeout_hours=4)


@pytest.fixture
def mock_dispatcher():
    return AsyncMock()


# ─── Approval Request Lifecycle ──────────────────────────────────────


@pytest.mark.asyncio
async def test_submit_creates_pending_request(workflow):
    """Submitting an intent creates a pending approval request."""
    intent = _make_intent()
    request = await workflow.submit(intent)
    assert isinstance(request, ApprovalRequest)
    assert request.intent_id == "intent-001"
    assert request.status == ApprovalStatus.PENDING
    assert request.timeout_hours == 4


@pytest.mark.asyncio
async def test_submit_tracks_in_pending_list(workflow):
    """Pending requests are tracked and queryable."""
    intent = _make_intent("intent-001")
    await workflow.submit(intent)

    pending = workflow.get_pending()
    assert len(pending) == 1
    assert pending[0].intent_id == "intent-001"


@pytest.mark.asyncio
async def test_approve_request(workflow):
    """Approving a pending request updates status."""
    intent = _make_intent("intent-002")
    await workflow.submit(intent)

    response = await workflow.respond("intent-002", ApprovalResponse.APPROVE, approver="user1")
    assert response.status == ApprovalStatus.APPROVED
    assert response.approver == "user1"

    pending = workflow.get_pending()
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_reject_request(workflow):
    """Rejecting a pending request updates status."""
    intent = _make_intent("intent-003")
    await workflow.submit(intent)

    response = await workflow.respond(
        "intent-003", ApprovalResponse.REJECT,
        approver="user1", reason="Too risky",
    )
    assert response.status == ApprovalStatus.REJECTED
    assert response.reason == "Too risky"


@pytest.mark.asyncio
async def test_respond_to_nonexistent_request(workflow):
    """Responding to unknown intent raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        await workflow.respond("nonexistent", ApprovalResponse.APPROVE, approver="user1")


@pytest.mark.asyncio
async def test_duplicate_response_rejected(workflow):
    """Cannot respond to already-resolved request."""
    intent = _make_intent("intent-004")
    await workflow.submit(intent)
    await workflow.respond("intent-004", ApprovalResponse.APPROVE, approver="user1")

    with pytest.raises(ValueError, match="already resolved"):
        await workflow.respond("intent-004", ApprovalResponse.REJECT, approver="user2")


# ─── Timeout (Auto-Reject) ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_expired_requests_auto_rejected(workflow):
    """Requests past timeout are auto-rejected."""
    intent = _make_intent("intent-005")
    request = await workflow.submit(intent)

    # Manually backdate the request to simulate timeout
    request.submitted_at = datetime.now(timezone.utc) - timedelta(hours=5)

    expired = await workflow.process_timeouts()
    assert len(expired) == 1
    assert expired[0].status == ApprovalStatus.TIMED_OUT
    assert expired[0].intent_id == "intent-005"


@pytest.mark.asyncio
async def test_non_expired_requests_unaffected(workflow):
    """Recent requests are not affected by timeout processing."""
    intent = _make_intent("intent-006")
    await workflow.submit(intent)

    expired = await workflow.process_timeouts()
    assert len(expired) == 0

    pending = workflow.get_pending()
    assert len(pending) == 1


@pytest.mark.asyncio
async def test_custom_timeout(workflow):
    """Requests can have custom timeout."""
    intent = _make_intent("intent-007")
    request = await workflow.submit(intent, timeout_hours=1)
    assert request.timeout_hours == 1


# ─── Notification Preferences ───────────────────────────────────────


def test_notification_preferences_default():
    """Default preferences enable all channels."""
    prefs = NotificationPreferences()
    assert prefs.slack_enabled is True
    assert prefs.telegram_enabled is True
    assert prefs.email_enabled is True


def test_notification_preferences_custom():
    """Custom preferences allow disabling channels."""
    prefs = NotificationPreferences(
        slack_enabled=True,
        telegram_enabled=False,
        email_enabled=False,
    )
    assert prefs.slack_enabled is True
    assert prefs.telegram_enabled is False
    assert prefs.email_enabled is False

    enabled = prefs.get_enabled_channels()
    assert enabled == ["slack"]


# ─── Approval History ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_approval_history(workflow):
    """Workflow maintains history of all resolved requests."""
    intent1 = _make_intent("intent-h1")
    intent2 = _make_intent("intent-h2")

    await workflow.submit(intent1)
    await workflow.submit(intent2)

    await workflow.respond("intent-h1", ApprovalResponse.APPROVE, approver="user1")
    await workflow.respond("intent-h2", ApprovalResponse.REJECT, approver="user1")

    history = workflow.get_history(limit=10)
    assert len(history) == 2
    assert any(r.intent_id == "intent-h1" and r.status == ApprovalStatus.APPROVED for r in history)
    assert any(r.intent_id == "intent-h2" and r.status == ApprovalStatus.REJECTED for r in history)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_approval_workflow.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.notifications.approval'`

**Step 3: Implement the approval workflow engine**

```python
# src/evolve_trader/notifications/approval.py
"""Approval Workflow Engine.

Manages the lifecycle of trade approval requests:
- Submit: Create a pending approval from a TradeIntent
- Respond: APPROVE or REJECT with optional reason
- Timeout: Auto-reject after configurable timeout (default 4h)
- History: Full audit trail of all approval decisions

No response within timeout = auto-reject (fail-safe).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

from evolve_trader.execution.trade_intent import TradeIntent

logger = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"


class ApprovalResponse(str, Enum):
    """Human response to an approval request."""
    APPROVE = "approve"
    REJECT = "reject"


class ApprovalTimeout(str, Enum):
    """Timeout disposition."""
    AUTO_REJECT = "auto_reject"


@dataclass
class ApprovalRequest:
    """A pending or resolved approval request."""
    intent_id: str
    ticker: str
    direction: str
    strategy_skill: str
    confidence: float
    status: ApprovalStatus = ApprovalStatus.PENDING
    timeout_hours: int = 4
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: datetime | None = None
    approver: str | None = None
    reason: str | None = None

    @property
    def is_expired(self) -> bool:
        """Check if request has exceeded its timeout."""
        if self.status != ApprovalStatus.PENDING:
            return False
        deadline = self.submitted_at + timedelta(hours=self.timeout_hours)
        return datetime.now(timezone.utc) > deadline


@dataclass
class NotificationPreferences:
    """Per-user notification channel preferences."""
    slack_enabled: bool = True
    telegram_enabled: bool = True
    email_enabled: bool = True
    dashboard_enabled: bool = True

    def get_enabled_channels(self) -> list[str]:
        """Return list of enabled channel names."""
        channels = []
        if self.slack_enabled:
            channels.append("slack")
        if self.telegram_enabled:
            channels.append("telegram")
        if self.email_enabled:
            channels.append("email")
        if self.dashboard_enabled:
            channels.append("dashboard")
        return channels


class ApprovalWorkflow:
    """Engine for managing trade approval requests.

    Tracks pending approvals, processes timeouts, and maintains
    a complete audit trail of all approval decisions.
    """

    def __init__(self, default_timeout_hours: int = 4):
        self._default_timeout = default_timeout_hours
        self._pending: dict[str, ApprovalRequest] = {}
        self._history: list[ApprovalRequest] = []

    async def submit(
        self,
        intent: TradeIntent,
        timeout_hours: int | None = None,
    ) -> ApprovalRequest:
        """Submit a TradeIntent for approval.

        Args:
            intent: The trade intent requiring approval.
            timeout_hours: Custom timeout (default: instance default).

        Returns:
            The created ApprovalRequest in PENDING status.
        """
        request = ApprovalRequest(
            intent_id=intent.intent_id,
            ticker=intent.ticker,
            direction=intent.direction.value,
            strategy_skill=intent.strategy_skill,
            confidence=intent.confidence,
            timeout_hours=timeout_hours or self._default_timeout,
        )
        self._pending[intent.intent_id] = request
        logger.info(f"Approval request submitted: {intent.intent_id}")
        return request

    async def respond(
        self,
        intent_id: str,
        response: ApprovalResponse,
        approver: str,
        reason: str | None = None,
    ) -> ApprovalRequest:
        """Record an approval or rejection response.

        Args:
            intent_id: The intent ID to respond to.
            response: APPROVE or REJECT.
            approver: Identifier of the person approving.
            reason: Optional reason (especially for rejections).

        Returns:
            The updated ApprovalRequest.

        Raises:
            ValueError: If intent not found or already resolved.
        """
        request = self._pending.get(intent_id)
        if request is None:
            # Check history too
            if any(r.intent_id == intent_id for r in self._history):
                raise ValueError(
                    f"Approval request '{intent_id}' is already resolved."
                )
            raise ValueError(f"Approval request '{intent_id}' not found.")

        if request.status != ApprovalStatus.PENDING:
            raise ValueError(
                f"Approval request '{intent_id}' is already resolved "
                f"(status: {request.status.value})."
            )

        now = datetime.now(timezone.utc)
        if response == ApprovalResponse.APPROVE:
            request.status = ApprovalStatus.APPROVED
        else:
            request.status = ApprovalStatus.REJECTED

        request.resolved_at = now
        request.approver = approver
        request.reason = reason

        # Move from pending to history
        del self._pending[intent_id]
        self._history.append(request)

        logger.info(
            f"Approval {request.status.value}: {intent_id} by {approver}"
        )
        return request

    async def process_timeouts(self) -> list[ApprovalRequest]:
        """Process expired approval requests — auto-reject.

        Returns:
            List of requests that were timed out.
        """
        expired: list[ApprovalRequest] = []
        now = datetime.now(timezone.utc)

        timed_out_ids = []
        for intent_id, request in self._pending.items():
            if request.is_expired:
                request.status = ApprovalStatus.TIMED_OUT
                request.resolved_at = now
                request.reason = (
                    f"Auto-rejected: no response within {request.timeout_hours}h timeout."
                )
                expired.append(request)
                timed_out_ids.append(intent_id)

        for intent_id in timed_out_ids:
            self._history.append(self._pending.pop(intent_id))

        if expired:
            logger.warning(f"Auto-rejected {len(expired)} timed-out approval requests.")

        return expired

    def get_pending(self) -> list[ApprovalRequest]:
        """Get all currently pending approval requests."""
        return list(self._pending.values())

    def get_history(self, limit: int = 50) -> list[ApprovalRequest]:
        """Get resolved approval request history.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of resolved ApprovalRequests, most recent first.
        """
        return list(reversed(self._history[-limit:]))
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_approval_workflow.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/notifications/approval.py tests/unit/test_approval_workflow.py
git commit -m "feat: approval workflow engine with timeout auto-reject and audit trail"
```

---

## Task 12: Integration Testing & Final Verification

**Files:**
- Create: `tests/integration/test_execution_pipeline.py`

**Step 1: Write the integration test**

```python
# tests/integration/test_execution_pipeline.py
"""Integration test: TradeIntent → Gate 1 → Gate 2 → Gate 3 → Notification.

Tests the full execution pipeline from intent creation through all three
gates to notification dispatch.
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from evolve_trader.execution.trade_intent import (
    TradeIntent,
    TradeDirection,
    IntentStatus,
    GateVerdict,
)
from evolve_trader.execution.gates.risk_gate import (
    RiskGate,
    RiskConstraints,
    PortfolioState,
)
from evolve_trader.execution.gates.paper_shadow import (
    PaperShadowGate,
    PaperTradeBook,
)
from evolve_trader.execution.gates.approval_gate import (
    ApprovalGate,
    ApprovalPolicy,
    ApprovalMode,
    SkillValidationRecord,
)
from evolve_trader.execution.promotion import (
    PromotionProtocol,
    PromotionStage,
    PromotionRecord,
    PromotionDecision,
)
from evolve_trader.notifications.dispatcher import NotificationDispatcher, NotificationResult
from evolve_trader.notifications.formatters import (
    TradeIntentFormatter,
    ApprovalRequestFormatter,
    NotificationPriority,
)
from evolve_trader.notifications.approval import (
    ApprovalWorkflow,
    ApprovalResponse,
    ApprovalStatus,
)


# ─── Full Pipeline: Intent → Gates → Notification ───────────────────


@pytest.mark.asyncio
async def test_full_pipeline_approved_trade():
    """High-confidence trade from validated skill passes all gates."""
    # Setup
    intent = TradeIntent(
        intent_id="pipeline-001",
        ticker="AAPL",
        direction=TradeDirection.BUY,
        quantity=10.0,
        order_type="market",
        strategy_skill="momentum-v1",
        strategy_lineage=["momentum-v1"],
        sizing_skill="kelly-fractional-v1",
        regime_label="risk-on",
        signal_sources=["edgar_13f", "form4_insider"],
        confidence=0.92,
        reasoning_chain="Strong institutional buying in risk-on regime.",
        position_impact={"sector": "Technology"},
    )

    portfolio = PortfolioState(
        total_equity=100_000.0,
        cash=50_000.0,
        positions={},
        sector_exposure={"Technology": 0.05},
        peak_equity=102_000.0,
        current_drawdown=0.02,
    )

    constraints = RiskConstraints(
        max_position_pct=0.05,
        max_sector_pct=0.25,
        max_drawdown_pct=0.20,
    )

    validated_skill = SkillValidationRecord(
        skill_name="momentum-v1",
        paper_trades=100,
        paper_sharpe=1.5,
        paper_win_rate=0.60,
        paper_max_drawdown=0.08,
        validated_at=datetime.now(timezone.utc),
        is_validated=True,
    )

    # Gate 1: Risk Constraints
    risk_gate = RiskGate(constraints)
    gate1_result = risk_gate.check(intent, portfolio, estimated_cost=1_500.0)
    intent.add_gate_result(gate1_result)
    assert gate1_result.verdict == GateVerdict.PASS

    # Gate 2: Paper Shadow
    paper_book = PaperTradeBook()
    mock_paper_client = AsyncMock()
    mock_paper_client.submit_order = AsyncMock(return_value=MagicMock(
        order_id="paper-001", symbol="AAPL", status="filled",
        filled_qty=10.0, filled_avg_price=150.0,
    ))
    paper_gate = PaperShadowGate(paper_book=paper_book, paper_client=mock_paper_client)
    gate2_result = await paper_gate.check(intent, estimated_price=150.0)
    intent.add_gate_result(gate2_result)
    assert gate2_result.verdict == GateVerdict.PASS

    # Gate 3: Approval (should auto-approve)
    approval_policy = ApprovalPolicy(
        mode=ApprovalMode.GRADUATED,
        auto_approve_min_confidence=0.85,
        auto_approve_min_paper_trades=50,
        auto_approve_min_sharpe=0.5,
    )
    approval_gate = ApprovalGate(
        policy=approval_policy,
        skill_records={"momentum-v1": validated_skill},
    )
    gate3_result = approval_gate.check(intent)
    intent.add_gate_result(gate3_result)
    assert gate3_result.verdict == GateVerdict.PASS

    # All gates passed
    assert intent.all_gates_passed()
    assert intent.status == IntentStatus.PENDING  # Not blocked

    # Format and dispatch notification
    formatter = TradeIntentFormatter()
    message = formatter.format(intent)
    assert "AAPL" in message.subject
    assert "momentum-v1" in message.body

    # Paper book recorded the trade
    assert len(paper_book.get_trades()) == 1


@pytest.mark.asyncio
async def test_full_pipeline_blocked_by_risk():
    """Trade blocked by risk constraints still gets paper-shadowed."""
    intent = TradeIntent(
        intent_id="pipeline-002",
        ticker="AAPL",
        direction=TradeDirection.BUY,
        quantity=100.0,
        order_type="market",
        strategy_skill="momentum-v1",
        strategy_lineage=["momentum-v1"],
        sizing_skill="kelly-fractional-v1",
        regime_label="risk-on",
        signal_sources=["edgar_13f"],
        confidence=0.80,
        reasoning_chain="Oversized position attempt.",
        position_impact={"sector": "Technology"},
    )

    portfolio = PortfolioState(
        total_equity=100_000.0,
        cash=50_000.0,
        positions={"AAPL": {"value": 4_500.0, "sector": "Technology"}},
        sector_exposure={"Technology": 0.045},
        peak_equity=100_000.0,
        current_drawdown=0.0,
    )

    constraints = RiskConstraints(
        max_position_pct=0.05,
        max_sector_pct=0.25,
        max_drawdown_pct=0.20,
    )

    # Gate 1: Should block (position too large)
    risk_gate = RiskGate(constraints)
    gate1_result = risk_gate.check(intent, portfolio, estimated_cost=15_000.0)
    intent.add_gate_result(gate1_result)
    assert gate1_result.verdict == GateVerdict.BLOCK
    assert intent.status == IntentStatus.BLOCKED

    # Gate 2: Paper shadow still executes (records vetoed trade)
    paper_book = PaperTradeBook()
    mock_paper_client = AsyncMock()
    mock_paper_client.submit_order = AsyncMock(return_value=MagicMock(
        order_id="paper-002", symbol="AAPL", status="filled",
        filled_qty=100.0, filled_avg_price=150.0,
    ))
    paper_gate = PaperShadowGate(paper_book=paper_book, paper_client=mock_paper_client)
    gate2_result = await paper_gate.check(intent, estimated_price=150.0)
    assert gate2_result.verdict == GateVerdict.PASS  # Shadow always passes

    # Vetoed trade recorded for counterfactual
    vetoed_trades = paper_book.get_trades(include_vetoed=True)
    assert len(vetoed_trades) == 1
    assert vetoed_trades[0].was_vetoed is True


@pytest.mark.asyncio
async def test_full_pipeline_manual_approval_workflow():
    """Low-confidence trade routes through manual approval workflow."""
    intent = TradeIntent(
        intent_id="pipeline-003",
        ticker="TSLA",
        direction=TradeDirection.BUY,
        quantity=5.0,
        order_type="market",
        strategy_skill="experimental-v1",
        strategy_lineage=["experimental-v1"],
        sizing_skill="fixed-fractional-v1",
        regime_label="neutral",
        signal_sources=["congressional"],
        confidence=0.65,
        reasoning_chain="Experimental strategy with low confidence.",
        position_impact={"sector": "Consumer Discretionary"},
    )

    # Gate 3: Should require manual approval
    approval_policy = ApprovalPolicy(
        mode=ApprovalMode.GRADUATED,
        auto_approve_min_confidence=0.85,
    )
    approval_gate = ApprovalGate(policy=approval_policy, skill_records={})
    gate3_result = approval_gate.check(intent)
    assert gate3_result.verdict == GateVerdict.PENDING_APPROVAL

    # Submit to approval workflow
    workflow = ApprovalWorkflow(default_timeout_hours=4)
    request = await workflow.submit(intent)
    assert request.status == ApprovalStatus.PENDING

    # Format approval notification
    formatter = ApprovalRequestFormatter()
    message = formatter.format(intent, timeout_hours=4)
    assert message.priority == NotificationPriority.HIGH
    assert message.metadata.get("requires_approval") is True

    # Human approves
    result = await workflow.respond(
        "pipeline-003", ApprovalResponse.APPROVE, approver="trader1",
    )
    assert result.status == ApprovalStatus.APPROVED


@pytest.mark.asyncio
async def test_promotion_protocol_in_pipeline():
    """Promotion protocol evaluates skill readiness for live trading."""
    protocol = PromotionProtocol()

    # Skill in paper training
    from datetime import timedelta
    record = PromotionRecord(
        skill_name="momentum-v1",
        current_stage=PromotionStage.PAPER_TRAINING,
        stage_entered_at=datetime.now(timezone.utc) - timedelta(days=95),
        paper_trades=120,
        sharpe_ratio=0.8,
        max_drawdown=0.10,
        win_rate=0.55,
        total_pnl=5_000.0,
        allocation_pct=0.0,
    )

    decision = protocol.evaluate(record)
    assert decision == PromotionDecision.PROMOTE

    next_stage = protocol.next_stage(record.current_stage)
    assert next_stage == PromotionStage.PAPER_VALIDATION


@pytest.mark.asyncio
async def test_notification_dispatcher_integration():
    """Dispatcher sends to multiple mock channels without failure."""
    dispatcher = NotificationDispatcher()

    # Register mock channels
    for name in ["slack", "telegram", "email"]:
        channel = AsyncMock()
        channel.name = name
        channel.is_enabled = True
        channel.send = AsyncMock(return_value=NotificationResult(
            channel=name, success=True, message_id=f"{name}-001",
        ))
        dispatcher.register_channel(channel)

    formatter = TradeIntentFormatter()
    intent = TradeIntent(
        intent_id="pipeline-004",
        ticker="NVDA",
        direction=TradeDirection.BUY,
        quantity=3.0,
        order_type="market",
        strategy_skill="momentum-v1",
        strategy_lineage=["momentum-v1"],
        sizing_skill="kelly-fractional-v1",
        regime_label="risk-on",
        signal_sources=["edgar_13f"],
        confidence=0.88,
        reasoning_chain="Strong momentum.",
    )
    message = formatter.format(intent)

    results = await dispatcher.send(message)
    assert len(results) == 3
    assert all(r.success for r in results)
```

**Step 2: Run test**

```bash
pytest tests/integration/test_execution_pipeline.py -v
```

Expected: PASS (all unit components already implemented)

**Step 3: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: ALL PASS — both previous phases and Phase 6 tests

**Step 4: Run linting and type checking**

```bash
ruff check src/evolve_trader/execution/ src/evolve_trader/notifications/
mypy src/evolve_trader/execution/ src/evolve_trader/notifications/ --ignore-missing-imports
```

Expected: No errors

**Step 5: Commit**

```bash
git add tests/integration/test_execution_pipeline.py
git commit -m "test: integration tests for full execution pipeline — gates, promotion, notifications"
```

---

## Parallelization Notes

Tasks in this phase have the following dependency structure:

```
Task 1 (Alpaca Client) ─────────┐
Task 2 (TradeIntent) ───────────┤
                                 ├── Task 3 (Gate 1: Risk) ──────────┐
                                 ├── Task 4 (Gate 2: Paper Shadow) ──┤
                                 ├── Task 5 (Gate 3: Approval) ──────┤
                                 │                                    │
Task 6 (Promotion Protocol) ────┘                                    │
                                                                     │
Task 7 (Notification Dispatcher) ────────────────────────────────────┤
                                 ┌── Task 8 (Slack) ─────────────────┤
Task 7 ─────────────────────────┤── Task 9 (Telegram) ──────────────┤
                                 └── Task 10 (Email) ────────────────┤
                                                                     │
Task 11 (Approval Workflow) ─────────────────────────────────────────┤
                                                                     │
Task 12 (Integration Tests) ─────────────────────────────────────────┘
```

**Can run in parallel:**
- Tasks 1 (Alpaca Client) and 2 (TradeIntent) are independent — run simultaneously
- Tasks 3, 4, 5 (gates) depend on Task 2 but are independent of each other — run simultaneously after Task 2
- Task 6 (Promotion) depends only on Task 2
- Tasks 8, 9, 10 (channel implementations) depend on Task 7 but are independent of each other — run simultaneously
- Task 11 (Approval Workflow) depends on Tasks 2 and 7
- Task 12 (Integration) depends on everything — must be last
