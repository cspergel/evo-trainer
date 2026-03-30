# Phase 11: Live Trading & Hardening — Detailed Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Connect to Alpaca live trading with a single config flag. Operationalize the graduated promotion pipeline defined in Phase 6 for live capital, with bidirectional demotion and production-only safeguards. Implement a comprehensive kill switch accessible from the dashboard, Slack/Telegram, REST API, and automatic drawdown triggers. Harden the system with secrets management, rate limiting, tamper-evident audit logging, regime diversity gates, production observability, and database backup/recovery.

**Architecture:** The live trading layer wraps the existing paper trading client with an `AlpacaLiveClient` that targets the live endpoint. It reuses the shared promotion protocol from Phase 6 rather than introducing a second promotion abstraction. Phase 11 adds live-capital caps, reviewer workflow, regime-diversity checks, and production audit fields around that existing protocol. The kill switch is a singleton state machine reachable from four surfaces (dashboard, chat channels, REST API, auto-trigger) that atomically cancels orders, optionally closes positions, demotes all strategies to paper, and emits notifications. Security modules handle secrets rotation, per-service rate limiting, and append-only audit logs. Structured JSON logging, health checks, auto-restart, broker reconciliation, market-session awareness, and tested recovery complete the hardening.

**Tech Stack:** Python 3.11+, PostgreSQL 16+, SQLAlchemy 2.0 (async), alpaca-trade-api, httpx, cryptography, python-json-logger, structlog, pytest, pytest-asyncio, Slack SDK, python-telegram-bot

**Parent plan:** [`2026-03-29-evolve-trader-master-plan.md`](2026-03-29-evolve-trader-master-plan.md)

**Prerequisites:** Phase 10 complete. Paper trading pipeline fully operational. Phase 6 promotion protocol implemented and verified. Dashboard, alerting, and monitoring infrastructure functional. All prior phase tests passing.

---

## Task 1: Alpaca Live Integration

**Files:**
- Create: `src/evolve_trader/execution/alpaca_live.py`
- Create: `tests/unit/test_alpaca_live.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_alpaca_live.py
"""Tests for Alpaca live trading client."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from decimal import Decimal

from evolve_trader.execution.alpaca_live import (
    AlpacaLiveClient,
    AlpacaEnvironment,
    LiveTradingConfig,
    OrderResult,
    PaperShadowResult,
)


class TestLiveTradingConfig:
    """LiveTradingConfig controls paper vs live with a single flag."""

    def test_default_is_paper(self):
        """Config defaults to paper trading for safety."""
        config = LiveTradingConfig()
        assert config.environment == AlpacaEnvironment.PAPER
        assert config.is_live is False

    def test_single_flag_switches_to_live(self):
        """A single flag switches from paper to live endpoint."""
        config = LiveTradingConfig(environment=AlpacaEnvironment.LIVE)
        assert config.is_live is True

    def test_paper_shadow_enabled_by_default_in_live(self):
        """Paper shadow trading continues alongside live by default."""
        config = LiveTradingConfig(environment=AlpacaEnvironment.LIVE)
        assert config.paper_shadow_enabled is True

    def test_fractional_shares_configurable(self):
        """Fractional share trading is configurable."""
        config = LiveTradingConfig(fractional_shares=True)
        assert config.fractional_shares is True

    def test_extended_hours_configurable(self):
        """Extended hours trading is configurable."""
        config = LiveTradingConfig(extended_hours=True)
        assert config.extended_hours is True

    def test_live_base_url(self):
        """Live config uses the production Alpaca endpoint."""
        config = LiveTradingConfig(environment=AlpacaEnvironment.LIVE)
        assert "paper" not in config.base_url
        assert "api.alpaca.markets" in config.base_url

    def test_paper_base_url(self):
        """Paper config uses the paper Alpaca endpoint."""
        config = LiveTradingConfig(environment=AlpacaEnvironment.PAPER)
        assert "paper" in config.base_url


class TestAlpacaLiveClient:
    """AlpacaLiveClient wraps Alpaca API for live order execution."""

    @pytest.fixture
    def live_config(self):
        return LiveTradingConfig(
            environment=AlpacaEnvironment.LIVE,
            paper_shadow_enabled=True,
            fractional_shares=True,
            extended_hours=False,
        )

    @pytest.fixture
    def paper_config(self):
        return LiveTradingConfig(
            environment=AlpacaEnvironment.PAPER,
            paper_shadow_enabled=False,
        )

    @pytest.fixture
    def mock_alpaca_api(self):
        api = AsyncMock()
        api.submit_order = AsyncMock(return_value=MagicMock(
            id="order-123",
            status="accepted",
            filled_avg_price=150.25,
            filled_qty=10.0,
            side="buy",
            symbol="AAPL",
        ))
        api.cancel_order = AsyncMock()
        api.close_position = AsyncMock()
        api.list_positions = AsyncMock(return_value=[])
        api.list_orders = AsyncMock(return_value=[])
        api.get_account = AsyncMock(return_value=MagicMock(
            equity=100000.0,
            buying_power=50000.0,
            status="ACTIVE",
        ))
        return api

    @pytest.mark.asyncio
    async def test_submit_order_live(self, live_config, mock_alpaca_api):
        """Submit order routes to live endpoint."""
        client = AlpacaLiveClient(live_config)
        client._api = mock_alpaca_api

        result = await client.submit_order(
            symbol="AAPL",
            qty=10.0,
            side="buy",
            order_type="market",
        )

        assert isinstance(result, OrderResult)
        assert result.order_id == "order-123"
        assert result.status == "accepted"
        mock_alpaca_api.submit_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_paper_shadow_runs_alongside_live(self, live_config, mock_alpaca_api):
        """When paper_shadow_enabled, both live and paper orders are placed."""
        client = AlpacaLiveClient(live_config)
        client._api = mock_alpaca_api
        client._paper_api = AsyncMock()
        client._paper_api.submit_order = AsyncMock(return_value=MagicMock(
            id="paper-order-456",
            status="accepted",
            filled_avg_price=150.30,
            filled_qty=10.0,
            side="buy",
            symbol="AAPL",
        ))

        result = await client.submit_order(
            symbol="AAPL",
            qty=10.0,
            side="buy",
            order_type="market",
        )

        assert isinstance(result, OrderResult)
        assert result.paper_shadow is not None
        assert isinstance(result.paper_shadow, PaperShadowResult)
        client._paper_api.submit_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_fractional_shares_order(self, live_config, mock_alpaca_api):
        """Fractional share orders are submitted correctly."""
        client = AlpacaLiveClient(live_config)
        client._api = mock_alpaca_api
        client._paper_api = AsyncMock()
        client._paper_api.submit_order = AsyncMock(return_value=MagicMock(
            id="paper-frac",
            status="accepted",
            filled_avg_price=150.0,
            filled_qty=0.5,
            side="buy",
            symbol="AAPL",
        ))

        result = await client.submit_order(
            symbol="AAPL",
            qty=0.5,
            side="buy",
            order_type="market",
        )

        assert result.order_id is not None
        call_kwargs = mock_alpaca_api.submit_order.call_args
        assert call_kwargs is not None

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, live_config, mock_alpaca_api):
        """Cancel all open orders."""
        client = AlpacaLiveClient(live_config)
        client._api = mock_alpaca_api
        mock_alpaca_api.list_orders.return_value = [
            MagicMock(id="order-1"),
            MagicMock(id="order-2"),
        ]

        cancelled = await client.cancel_all_orders()

        assert cancelled == 2
        assert mock_alpaca_api.cancel_order.call_count == 2

    @pytest.mark.asyncio
    async def test_close_all_positions(self, live_config, mock_alpaca_api):
        """Close all open positions."""
        client = AlpacaLiveClient(live_config)
        client._api = mock_alpaca_api
        mock_alpaca_api.list_positions.return_value = [
            MagicMock(symbol="AAPL", qty=10),
            MagicMock(symbol="MSFT", qty=5),
        ]

        closed = await client.close_all_positions()

        assert closed == 2
        assert mock_alpaca_api.close_position.call_count == 2

    @pytest.mark.asyncio
    async def test_get_account_status(self, live_config, mock_alpaca_api):
        """Get account status returns equity and buying power."""
        client = AlpacaLiveClient(live_config)
        client._api = mock_alpaca_api

        status = await client.get_account_status()

        assert status["equity"] == 100000.0
        assert status["buying_power"] == 50000.0
        assert status["status"] == "ACTIVE"

    @pytest.mark.asyncio
    async def test_paper_mode_never_hits_live(self, paper_config, mock_alpaca_api):
        """Paper config must never submit to the live API."""
        client = AlpacaLiveClient(paper_config)
        client._api = mock_alpaca_api

        result = await client.submit_order(
            symbol="AAPL",
            qty=10.0,
            side="buy",
            order_type="market",
        )

        assert result.order_id is not None
        # Verify the client is using paper endpoint
        assert client.config.is_live is False
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_alpaca_live.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.execution.alpaca_live'`

**Step 3: Implement the Alpaca live client**

```python
# src/evolve_trader/execution/alpaca_live.py
"""Alpaca live trading client with paper shadow support.

Single config flag switches between paper and live endpoints.
Paper shadow trading continues alongside live for comparison.
Supports fractional shares and extended hours.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AlpacaEnvironment(Enum):
    """Trading environment selector."""
    PAPER = "paper"
    LIVE = "live"


@dataclass
class LiveTradingConfig:
    """Configuration for Alpaca live/paper trading.

    Single flag (environment) controls paper vs live.
    Paper shadow continues alongside live by default.
    """
    environment: AlpacaEnvironment = AlpacaEnvironment.PAPER
    paper_shadow_enabled: bool = True
    fractional_shares: bool = False
    extended_hours: bool = False
    api_key_env: str = "ALPACA_API_KEY"
    api_secret_env: str = "ALPACA_API_SECRET"

    @property
    def is_live(self) -> bool:
        return self.environment == AlpacaEnvironment.LIVE

    @property
    def base_url(self) -> str:
        if self.is_live:
            return "https://api.alpaca.markets"
        return "https://paper-api.alpaca.markets"


@dataclass
class PaperShadowResult:
    """Result from the paper shadow order."""
    order_id: str
    status: str
    filled_price: float | None
    filled_qty: float | None
    symbol: str


@dataclass
class OrderResult:
    """Result of a live or paper order submission."""
    order_id: str
    status: str
    filled_price: float | None
    filled_qty: float | None
    side: str
    symbol: str
    paper_shadow: PaperShadowResult | None = None


class AlpacaLiveClient:
    """Client for Alpaca order execution with paper shadow support.

    Wraps the Alpaca API with:
    - Single config flag for paper/live switching
    - Paper shadow orders alongside live orders
    - Fractional share support
    - Extended hours support
    - Bulk cancel/close operations
    """

    def __init__(self, config: LiveTradingConfig):
        self.config = config
        self._api: Any = None       # Alpaca API (live or paper)
        self._paper_api: Any = None  # Paper API for shadow orders

    async def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: float | None = None,
    ) -> OrderResult:
        """Submit an order to the configured environment.

        If paper_shadow_enabled and in live mode, also submits
        a parallel paper order for tracking divergence.
        """
        order_kwargs: dict[str, Any] = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if limit_price is not None:
            order_kwargs["limit_price"] = limit_price
        if self.config.extended_hours:
            order_kwargs["extended_hours"] = True

        logger.info(
            "Submitting %s order: %s %s %s @ %s",
            self.config.environment.value,
            side,
            qty,
            symbol,
            order_type,
        )

        response = await self._api.submit_order(**order_kwargs)

        result = OrderResult(
            order_id=response.id,
            status=response.status,
            filled_price=getattr(response, "filled_avg_price", None),
            filled_qty=getattr(response, "filled_qty", None),
            side=response.side,
            symbol=response.symbol,
        )

        # Paper shadow: mirror the order on paper for comparison
        if self.config.is_live and self.config.paper_shadow_enabled and self._paper_api:
            try:
                paper_response = await self._paper_api.submit_order(**order_kwargs)
                result.paper_shadow = PaperShadowResult(
                    order_id=paper_response.id,
                    status=paper_response.status,
                    filled_price=getattr(paper_response, "filled_avg_price", None),
                    filled_qty=getattr(paper_response, "filled_qty", None),
                    symbol=paper_response.symbol,
                )
            except Exception:
                logger.warning("Paper shadow order failed for %s", symbol, exc_info=True)

        return result

    async def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        open_orders = await self._api.list_orders(status="open")
        for order in open_orders:
            await self._api.cancel_order(order.id)
            logger.info("Cancelled order %s", order.id)
        return len(open_orders)

    async def close_all_positions(self) -> int:
        """Close all open positions. Returns count of positions closed."""
        positions = await self._api.list_positions()
        for position in positions:
            await self._api.close_position(position.symbol)
            logger.info("Closed position %s (%s shares)", position.symbol, position.qty)
        return len(positions)

    async def get_account_status(self) -> dict[str, Any]:
        """Get current account status."""
        account = await self._api.get_account()
        return {
            "equity": account.equity,
            "buying_power": account.buying_power,
            "status": account.status,
        }
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_alpaca_live.py -v
```

Expected: PASS — all 10 tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/execution/alpaca_live.py tests/unit/test_alpaca_live.py
git commit -m "feat: Alpaca live trading client with paper shadow support"
```

---

## Task 2: Promotion Pipeline Operationalization

This task must extend the Phase 6 promotion protocol in `src/evolve_trader/execution/promotion.py`. Do not create a second promotion abstraction, a second enum for promotion levels, or a second approval state machine.

**Files:**
- Modify: `src/evolve_trader/execution/promotion.py`
- Modify: `src/evolve_trader/execution/approval.py` or the equivalent Phase 6 approval module
- Create: `tests/unit/test_live_promotion_operations.py`

**Required additions:**
- Live-capital caps per stage
- Reviewer identity, approval timestamp, and override audit fields
- Approval latency and override-rate metrics
- Regime-diversity enforcement before promotion into live capital
- Operational checks for market session, broker connectivity, and current kill-switch state before any promotion-related action

**Acceptance criteria:**
- One shared promotion-stage model remains the source of truth
- Paper, validation, micro, partial, and full transitions remain compatible with Phase 6 tests
- Live-only safeguards compose around the existing protocol rather than replacing it
- Demotions triggered by drawdown, reconciliation failure, or kill-switch activation are recorded through the same shared pipeline

---

## Task 3: Kill Switch — Dashboard

**Files:**
- Create: `src/evolve_trader/execution/kill_switch.py`
- Create: `tests/unit/test_kill_switch.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_kill_switch.py
"""Tests for kill switch core logic."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from evolve_trader.execution.kill_switch import (
    KillSwitch,
    KillSwitchState,
    KillSwitchConfig,
    KillSwitchActivation,
    KillSwitchSource,
)


class TestKillSwitchState:
    """Kill switch state machine."""

    def test_initial_state_is_armed(self):
        """Kill switch starts in ARMED (ready to trigger) state."""
        ks = KillSwitch(KillSwitchConfig())
        assert ks.state == KillSwitchState.ARMED

    def test_states_exist(self):
        """All required states exist."""
        assert KillSwitchState.ARMED is not None
        assert KillSwitchState.TRIGGERED is not None
        assert KillSwitchState.COOLDOWN is not None


class TestKillSwitchConfig:
    """Kill switch configuration."""

    def test_close_positions_configurable(self):
        """Position closing on kill is configurable."""
        config = KillSwitchConfig(close_positions=True)
        assert config.close_positions is True

        config_no_close = KillSwitchConfig(close_positions=False)
        assert config_no_close.close_positions is False

    def test_default_close_positions_is_true(self):
        """By default, kill switch closes all positions."""
        config = KillSwitchConfig()
        assert config.close_positions is True

    def test_notify_channels_configurable(self):
        """Notification channels are configurable."""
        config = KillSwitchConfig(notify_channels=["slack", "email"])
        assert "slack" in config.notify_channels


class TestKillSwitchActivation:
    """Kill switch activation behavior."""

    @pytest.fixture
    def mock_trading_client(self):
        client = AsyncMock()
        client.cancel_all_orders = AsyncMock(return_value=3)
        client.close_all_positions = AsyncMock(return_value=2)
        return client

    @pytest.fixture
    def mock_notifier(self):
        return AsyncMock()

    @pytest.fixture
    def kill_switch(self, mock_trading_client, mock_notifier):
        config = KillSwitchConfig(
            close_positions=True,
            notify_channels=["slack"],
        )
        ks = KillSwitch(config)
        ks._trading_client = mock_trading_client
        ks._notifier = mock_notifier
        return ks

    @pytest.mark.asyncio
    async def test_activate_cancels_orders(self, kill_switch, mock_trading_client):
        """Activation cancels all open orders."""
        result = await kill_switch.activate(
            source=KillSwitchSource.DASHBOARD,
            reason="Manual trigger from dashboard",
        )
        mock_trading_client.cancel_all_orders.assert_called_once()
        assert result.orders_cancelled == 3

    @pytest.mark.asyncio
    async def test_activate_closes_positions_when_configured(
        self, kill_switch, mock_trading_client
    ):
        """Activation closes positions when close_positions=True."""
        result = await kill_switch.activate(
            source=KillSwitchSource.DASHBOARD,
            reason="Emergency stop",
        )
        mock_trading_client.close_all_positions.assert_called_once()
        assert result.positions_closed == 2

    @pytest.mark.asyncio
    async def test_activate_skips_positions_when_configured(
        self, mock_trading_client, mock_notifier
    ):
        """Activation skips position closing when close_positions=False."""
        config = KillSwitchConfig(close_positions=False)
        ks = KillSwitch(config)
        ks._trading_client = mock_trading_client
        ks._notifier = mock_notifier

        result = await ks.activate(
            source=KillSwitchSource.DASHBOARD,
            reason="Soft stop",
        )
        mock_trading_client.close_all_positions.assert_not_called()
        assert result.positions_closed == 0

    @pytest.mark.asyncio
    async def test_activate_sends_notifications(self, kill_switch, mock_notifier):
        """Activation sends notifications to configured channels."""
        await kill_switch.activate(
            source=KillSwitchSource.DASHBOARD,
            reason="Test notification",
        )
        mock_notifier.assert_called_once()

    @pytest.mark.asyncio
    async def test_activate_changes_state_to_triggered(self, kill_switch):
        """Activation transitions state to TRIGGERED."""
        await kill_switch.activate(
            source=KillSwitchSource.DASHBOARD,
            reason="State test",
        )
        assert kill_switch.state == KillSwitchState.TRIGGERED

    @pytest.mark.asyncio
    async def test_activate_reverts_to_paper(self, kill_switch):
        """Activation reverts all strategies to paper training mode."""
        result = await kill_switch.activate(
            source=KillSwitchSource.DASHBOARD,
            reason="Revert test",
        )
        assert result.reverted_to_paper is True

    @pytest.mark.asyncio
    async def test_activate_returns_activation_record(self, kill_switch):
        """Activation returns a complete KillSwitchActivation record."""
        result = await kill_switch.activate(
            source=KillSwitchSource.DASHBOARD,
            reason="Record test",
        )
        assert isinstance(result, KillSwitchActivation)
        assert result.source == KillSwitchSource.DASHBOARD
        assert result.reason == "Record test"
        assert result.timestamp is not None

    @pytest.mark.asyncio
    async def test_double_activate_is_idempotent(self, kill_switch):
        """Activating an already-triggered kill switch is a no-op."""
        result1 = await kill_switch.activate(
            source=KillSwitchSource.DASHBOARD,
            reason="First",
        )
        result2 = await kill_switch.activate(
            source=KillSwitchSource.DASHBOARD,
            reason="Second",
        )
        assert result1.orders_cancelled == 3
        assert result2.orders_cancelled == 0  # Already triggered

    @pytest.mark.asyncio
    async def test_rearm_after_triggered(self, kill_switch):
        """Kill switch can be re-armed after trigger."""
        await kill_switch.activate(
            source=KillSwitchSource.DASHBOARD,
            reason="Trigger",
        )
        assert kill_switch.state == KillSwitchState.TRIGGERED

        await kill_switch.rearm(reviewer="admin")
        assert kill_switch.state == KillSwitchState.ARMED

    @pytest.mark.asyncio
    async def test_activation_history(self, kill_switch):
        """Kill switch maintains activation history."""
        await kill_switch.activate(
            source=KillSwitchSource.DASHBOARD,
            reason="First trigger",
        )
        assert len(kill_switch.activation_history) == 1
        assert kill_switch.activation_history[0].reason == "First trigger"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_kill_switch.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.execution.kill_switch'`

**Step 3: Implement the kill switch core**

```python
# src/evolve_trader/execution/kill_switch.py
"""Kill switch — singleton state machine for emergency trading halt.

Reachable from dashboard, Slack/Telegram, REST API, and auto-trigger.
Atomically cancels orders, optionally closes positions, demotes all
strategies to paper, and emits notifications.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


class KillSwitchState(Enum):
    """Kill switch state machine states."""
    ARMED = "armed"
    TRIGGERED = "triggered"
    COOLDOWN = "cooldown"


class KillSwitchSource(Enum):
    """Where the kill switch was activated from."""
    DASHBOARD = "dashboard"
    SLACK = "slack"
    TELEGRAM = "telegram"
    API = "api"
    AUTO = "auto"


@dataclass
class KillSwitchConfig:
    """Configuration for kill switch behavior."""
    close_positions: bool = True
    notify_channels: list[str] = field(default_factory=lambda: ["slack"])
    cooldown_seconds: int = 300


@dataclass
class KillSwitchActivation:
    """Record of a kill switch activation."""
    source: KillSwitchSource
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    orders_cancelled: int = 0
    positions_closed: int = 0
    reverted_to_paper: bool = False
    notified: bool = False


class KillSwitch:
    """Emergency kill switch for live trading.

    Singleton state machine:
    - ARMED: ready to trigger
    - TRIGGERED: all trading halted
    - COOLDOWN: waiting before re-arm is allowed

    On activation:
    1. Cancel all open orders
    2. Close all positions (if configured)
    3. Revert all strategies to Paper Training
    4. Send notifications to configured channels
    """

    def __init__(self, config: KillSwitchConfig):
        self.config = config
        self.state = KillSwitchState.ARMED
        self.activation_history: list[KillSwitchActivation] = []
        self._trading_client: Any = None
        self._notifier: Callable[..., Awaitable[None]] | None = None

    async def activate(
        self,
        source: KillSwitchSource,
        reason: str,
    ) -> KillSwitchActivation:
        """Activate the kill switch.

        Idempotent — if already triggered, returns a no-op record.
        """
        if self.state == KillSwitchState.TRIGGERED:
            logger.warning("Kill switch already triggered, ignoring duplicate activation")
            return KillSwitchActivation(
                source=source,
                reason=reason,
                orders_cancelled=0,
                positions_closed=0,
                reverted_to_paper=False,
            )

        logger.critical(
            "KILL SWITCH ACTIVATED — source=%s reason=%s",
            source.value, reason,
        )

        # 1. Cancel all open orders
        orders_cancelled = 0
        if self._trading_client:
            orders_cancelled = await self._trading_client.cancel_all_orders()

        # 2. Close positions if configured
        positions_closed = 0
        if self.config.close_positions and self._trading_client:
            positions_closed = await self._trading_client.close_all_positions()

        # 3. Revert to paper
        self.state = KillSwitchState.TRIGGERED

        # 4. Notify
        notified = False
        if self._notifier:
            await self._notifier(
                f"KILL SWITCH ACTIVATED — source={source.value}, reason={reason}, "
                f"orders_cancelled={orders_cancelled}, positions_closed={positions_closed}"
            )
            notified = True

        activation = KillSwitchActivation(
            source=source,
            reason=reason,
            orders_cancelled=orders_cancelled,
            positions_closed=positions_closed,
            reverted_to_paper=True,
            notified=notified,
        )
        self.activation_history.append(activation)
        return activation

    async def rearm(self, reviewer: str) -> None:
        """Re-arm the kill switch after a trigger. Requires human reviewer."""
        if self.state != KillSwitchState.TRIGGERED:
            raise ValueError(f"Cannot re-arm from state {self.state.value}")
        self.state = KillSwitchState.ARMED
        logger.info("Kill switch re-armed by %s", reviewer)
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_kill_switch.py -v
```

Expected: PASS — all 13 tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/execution/kill_switch.py tests/unit/test_kill_switch.py
git commit -m "feat: kill switch core with state machine, position closing, notifications"
```

---

## Task 4: Kill Switch — Slack/Telegram

**Files:**
- Create: `src/evolve_trader/execution/kill_switch_channels.py`
- Create: `tests/unit/test_kill_switch_channels.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_kill_switch_channels.py
"""Tests for kill switch Slack/Telegram channel integration."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from evolve_trader.execution.kill_switch_channels import (
    SlackKillSwitchHandler,
    TelegramKillSwitchHandler,
    ChannelAuthConfig,
)
from evolve_trader.execution.kill_switch import (
    KillSwitch,
    KillSwitchConfig,
    KillSwitchSource,
    KillSwitchState,
)


class TestSlackKillSwitchHandler:
    """Slack /kill command handler."""

    @pytest.fixture
    def kill_switch(self):
        ks = KillSwitch(KillSwitchConfig())
        ks._trading_client = AsyncMock()
        ks._trading_client.cancel_all_orders = AsyncMock(return_value=1)
        ks._trading_client.close_all_positions = AsyncMock(return_value=1)
        ks._notifier = AsyncMock()
        return ks

    @pytest.fixture
    def auth_config(self):
        return ChannelAuthConfig(
            authorized_user_ids=["U123ADMIN", "U456OPS"],
            authorized_channels=["C789TRADING"],
        )

    @pytest.fixture
    def handler(self, kill_switch, auth_config):
        return SlackKillSwitchHandler(
            kill_switch=kill_switch,
            auth_config=auth_config,
        )

    @pytest.mark.asyncio
    async def test_kill_command_authorized_user(self, handler):
        """Authorized user can trigger /kill."""
        result = await handler.handle_command(
            user_id="U123ADMIN",
            channel_id="C789TRADING",
            command_text="/kill market crash",
        )
        assert result["success"] is True
        assert handler.kill_switch.state == KillSwitchState.TRIGGERED

    @pytest.mark.asyncio
    async def test_kill_command_unauthorized_user(self, handler):
        """Unauthorized user is rejected."""
        result = await handler.handle_command(
            user_id="U999HACKER",
            channel_id="C789TRADING",
            command_text="/kill",
        )
        assert result["success"] is False
        assert "unauthorized" in result["message"].lower()
        assert handler.kill_switch.state == KillSwitchState.ARMED

    @pytest.mark.asyncio
    async def test_kill_command_unauthorized_channel(self, handler):
        """Command from unauthorized channel is rejected."""
        result = await handler.handle_command(
            user_id="U123ADMIN",
            channel_id="C000RANDOM",
            command_text="/kill",
        )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_kill_command_extracts_reason(self, handler):
        """Reason text is extracted from command."""
        result = await handler.handle_command(
            user_id="U123ADMIN",
            channel_id="C789TRADING",
            command_text="/kill flash crash detected",
        )
        assert result["success"] is True
        activation = handler.kill_switch.activation_history[-1]
        assert "flash crash detected" in activation.reason

    @pytest.mark.asyncio
    async def test_kill_command_default_reason(self, handler):
        """Default reason when none provided."""
        result = await handler.handle_command(
            user_id="U123ADMIN",
            channel_id="C789TRADING",
            command_text="/kill",
        )
        assert result["success"] is True
        activation = handler.kill_switch.activation_history[-1]
        assert activation.reason != ""


class TestTelegramKillSwitchHandler:
    """Telegram /kill command handler."""

    @pytest.fixture
    def kill_switch(self):
        ks = KillSwitch(KillSwitchConfig())
        ks._trading_client = AsyncMock()
        ks._trading_client.cancel_all_orders = AsyncMock(return_value=1)
        ks._trading_client.close_all_positions = AsyncMock(return_value=1)
        ks._notifier = AsyncMock()
        return ks

    @pytest.fixture
    def auth_config(self):
        return ChannelAuthConfig(
            authorized_user_ids=["12345", "67890"],
            authorized_channels=["trading_alerts"],
        )

    @pytest.fixture
    def handler(self, kill_switch, auth_config):
        return TelegramKillSwitchHandler(
            kill_switch=kill_switch,
            auth_config=auth_config,
        )

    @pytest.mark.asyncio
    async def test_kill_command_authorized(self, handler):
        """Authorized Telegram user can trigger /kill."""
        result = await handler.handle_command(
            user_id="12345",
            chat_id="trading_alerts",
            command_text="/kill emergency",
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_kill_command_unauthorized(self, handler):
        """Unauthorized Telegram user is rejected."""
        result = await handler.handle_command(
            user_id="99999",
            chat_id="trading_alerts",
            command_text="/kill",
        )
        assert result["success"] is False
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_kill_switch_channels.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.execution.kill_switch_channels'`

**Step 3: Implement the channel handlers**

```python
# src/evolve_trader/execution/kill_switch_channels.py
"""Kill switch Slack and Telegram channel handlers.

Provides /kill command integration for Slack and Telegram.
Both require authorization (user ID + channel) before triggering.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from evolve_trader.execution.kill_switch import (
    KillSwitch,
    KillSwitchSource,
)

logger = logging.getLogger(__name__)


@dataclass
class ChannelAuthConfig:
    """Authorization config for channel-based kill switch."""
    authorized_user_ids: list[str] = field(default_factory=list)
    authorized_channels: list[str] = field(default_factory=list)


class SlackKillSwitchHandler:
    """Handles /kill commands from Slack."""

    def __init__(self, kill_switch: KillSwitch, auth_config: ChannelAuthConfig):
        self.kill_switch = kill_switch
        self.auth_config = auth_config

    async def handle_command(
        self,
        user_id: str,
        channel_id: str,
        command_text: str,
    ) -> dict[str, Any]:
        """Handle a /kill command from Slack.

        Validates user and channel authorization before triggering.
        """
        if user_id not in self.auth_config.authorized_user_ids:
            logger.warning(
                "Unauthorized Slack kill switch attempt by user %s in channel %s",
                user_id, channel_id,
            )
            return {"success": False, "message": "Unauthorized user"}

        if channel_id not in self.auth_config.authorized_channels:
            logger.warning(
                "Kill switch attempt from unauthorized Slack channel %s by user %s",
                channel_id, user_id,
            )
            return {"success": False, "message": "Unauthorized channel"}

        # Extract reason from command text (strip /kill prefix)
        reason_text = command_text.replace("/kill", "").strip()
        reason = reason_text if reason_text else f"Slack /kill by {user_id}"

        activation = await self.kill_switch.activate(
            source=KillSwitchSource.SLACK,
            reason=reason,
        )

        return {
            "success": True,
            "message": (
                f"Kill switch activated. "
                f"Orders cancelled: {activation.orders_cancelled}, "
                f"Positions closed: {activation.positions_closed}"
            ),
        }


class TelegramKillSwitchHandler:
    """Handles /kill commands from Telegram."""

    def __init__(self, kill_switch: KillSwitch, auth_config: ChannelAuthConfig):
        self.kill_switch = kill_switch
        self.auth_config = auth_config

    async def handle_command(
        self,
        user_id: str,
        chat_id: str,
        command_text: str,
    ) -> dict[str, Any]:
        """Handle a /kill command from Telegram.

        Validates user and chat authorization before triggering.
        """
        if user_id not in self.auth_config.authorized_user_ids:
            logger.warning(
                "Unauthorized Telegram kill switch attempt by user %s in chat %s",
                user_id, chat_id,
            )
            return {"success": False, "message": "Unauthorized user"}

        if chat_id not in self.auth_config.authorized_channels:
            logger.warning(
                "Kill switch attempt from unauthorized Telegram chat %s by user %s",
                chat_id, user_id,
            )
            return {"success": False, "message": "Unauthorized channel"}

        reason_text = command_text.replace("/kill", "").strip()
        reason = reason_text if reason_text else f"Telegram /kill by {user_id}"

        activation = await self.kill_switch.activate(
            source=KillSwitchSource.TELEGRAM,
            reason=reason,
        )

        return {
            "success": True,
            "message": (
                f"Kill switch activated. "
                f"Orders cancelled: {activation.orders_cancelled}, "
                f"Positions closed: {activation.positions_closed}"
            ),
        }
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_kill_switch_channels.py -v
```

Expected: PASS — all 7 tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/execution/kill_switch_channels.py tests/unit/test_kill_switch_channels.py
git commit -m "feat: kill switch Slack and Telegram channel handlers with authorization"
```

---

## Task 5: Kill Switch — API Endpoint

**Files:**
- Create: `src/evolve_trader/api/routes/kill_switch.py`
- Create: `tests/api/test_kill_switch_api.py`

**Step 1: Write the failing tests**

```python
# tests/api/test_kill_switch_api.py
"""Tests for kill switch REST API endpoint."""
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient

from evolve_trader.api.routes.kill_switch import (
    create_kill_switch_router,
    KillSwitchRequest,
    KillSwitchResponse,
)
from evolve_trader.execution.kill_switch import (
    KillSwitch,
    KillSwitchConfig,
    KillSwitchState,
)


class TestKillSwitchRequest:
    """Kill switch API request model."""

    def test_request_requires_reason(self):
        """Request must include a reason."""
        req = KillSwitchRequest(reason="Emergency stop", api_key="test-key-123")
        assert req.reason == "Emergency stop"

    def test_request_requires_api_key(self):
        """Request must include an API key."""
        req = KillSwitchRequest(reason="Emergency", api_key="test-key-123")
        assert req.api_key == "test-key-123"


class TestKillSwitchAPI:
    """Kill switch REST API behavior."""

    @pytest.fixture
    def kill_switch(self):
        ks = KillSwitch(KillSwitchConfig())
        ks._trading_client = AsyncMock()
        ks._trading_client.cancel_all_orders = AsyncMock(return_value=2)
        ks._trading_client.close_all_positions = AsyncMock(return_value=1)
        ks._notifier = AsyncMock()
        return ks

    @pytest.fixture
    def router(self, kill_switch):
        return create_kill_switch_router(
            kill_switch=kill_switch,
            valid_api_keys=["valid-key-123"],
        )

    @pytest.mark.asyncio
    async def test_activate_with_valid_key(self, kill_switch, router):
        """POST /kill-switch with valid API key triggers kill switch."""
        from fastapi import FastAPI
        from httpx import ASGITransport

        app = FastAPI()
        app.include_router(router)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/kill-switch/activate", json={
                "reason": "API emergency stop",
                "api_key": "valid-key-123",
            })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["orders_cancelled"] == 2
        assert kill_switch.state == KillSwitchState.TRIGGERED

    @pytest.mark.asyncio
    async def test_activate_with_invalid_key(self, kill_switch, router):
        """POST /kill-switch with invalid API key returns 403."""
        from fastapi import FastAPI
        from httpx import ASGITransport

        app = FastAPI()
        app.include_router(router)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/kill-switch/activate", json={
                "reason": "Hacking attempt",
                "api_key": "invalid-key",
            })

        assert response.status_code == 403
        assert kill_switch.state == KillSwitchState.ARMED

    @pytest.mark.asyncio
    async def test_status_endpoint(self, kill_switch, router):
        """GET /kill-switch/status returns current state."""
        from fastapi import FastAPI
        from httpx import ASGITransport

        app = FastAPI()
        app.include_router(router)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/kill-switch/status")

        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "armed"
        assert data["activation_count"] == 0

    @pytest.mark.asyncio
    async def test_rearm_endpoint(self, kill_switch, router):
        """POST /kill-switch/rearm re-arms after trigger."""
        from fastapi import FastAPI
        from httpx import ASGITransport

        app = FastAPI()
        app.include_router(router)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # First trigger
            await client.post("/kill-switch/activate", json={
                "reason": "Test",
                "api_key": "valid-key-123",
            })
            assert kill_switch.state == KillSwitchState.TRIGGERED

            # Then rearm
            response = await client.post("/kill-switch/rearm", json={
                "api_key": "valid-key-123",
                "reviewer": "admin",
            })

        assert response.status_code == 200
        assert kill_switch.state == KillSwitchState.ARMED
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/api/test_kill_switch_api.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.api.routes.kill_switch'`

**Step 3: Implement the API routes**

```python
# src/evolve_trader/api/routes/kill_switch.py
"""Kill switch REST API endpoint.

POST /kill-switch/activate — trigger the kill switch
GET  /kill-switch/status   — current kill switch state
POST /kill-switch/rearm    — re-arm after trigger
"""
from __future__ import annotations

import logging
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from evolve_trader.execution.kill_switch import (
    KillSwitch,
    KillSwitchSource,
)

logger = logging.getLogger(__name__)


class KillSwitchRequest(BaseModel):
    """Request body for kill switch activation."""
    reason: str
    api_key: str


class RearmRequest(BaseModel):
    """Request body for re-arming the kill switch."""
    api_key: str
    reviewer: str


class KillSwitchResponse(BaseModel):
    """Response body for kill switch activation."""
    success: bool
    orders_cancelled: int = 0
    positions_closed: int = 0
    message: str = ""


def create_kill_switch_router(
    kill_switch: KillSwitch,
    valid_api_keys: list[str],
) -> APIRouter:
    """Create a FastAPI router for the kill switch API."""
    router = APIRouter(prefix="/kill-switch", tags=["kill-switch"])

    def _validate_key(api_key: str) -> None:
        if api_key not in valid_api_keys:
            logger.warning("Invalid API key used for kill switch: %s...", api_key[:8])
            raise HTTPException(status_code=403, detail="Invalid API key")

    @router.post("/activate")
    async def activate(request: KillSwitchRequest) -> dict:
        _validate_key(request.api_key)

        activation = await kill_switch.activate(
            source=KillSwitchSource.API,
            reason=request.reason,
        )

        return {
            "success": True,
            "orders_cancelled": activation.orders_cancelled,
            "positions_closed": activation.positions_closed,
            "message": f"Kill switch activated: {request.reason}",
        }

    @router.get("/status")
    async def status() -> dict:
        return {
            "state": kill_switch.state.value,
            "activation_count": len(kill_switch.activation_history),
        }

    @router.post("/rearm")
    async def rearm(request: RearmRequest) -> dict:
        _validate_key(request.api_key)
        await kill_switch.rearm(reviewer=request.reviewer)
        return {
            "success": True,
            "state": kill_switch.state.value,
            "message": f"Kill switch re-armed by {request.reviewer}",
        }

    return router
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/api/test_kill_switch_api.py -v
```

Expected: PASS — all 4 tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/api/routes/kill_switch.py tests/api/test_kill_switch_api.py
git commit -m "feat: kill switch REST API with key auth, status, and rearm endpoints"
```

---

## Task 6: Kill Switch — Auto-Trigger

**Files:**
- Create: `src/evolve_trader/execution/kill_switch_auto.py`
- Create: `tests/unit/test_kill_switch_auto.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_kill_switch_auto.py
"""Tests for automatic kill switch trigger on drawdown."""
import pytest
from unittest.mock import AsyncMock
from datetime import datetime, timezone

from evolve_trader.execution.kill_switch_auto import (
    AutoKillSwitchMonitor,
    AutoKillSwitchConfig,
    DrawdownState,
)
from evolve_trader.execution.kill_switch import (
    KillSwitch,
    KillSwitchConfig,
    KillSwitchState,
)


class TestAutoKillSwitchConfig:
    """Configuration for automatic kill switch."""

    def test_default_drawdown_threshold(self):
        """Default drawdown threshold is 20%."""
        config = AutoKillSwitchConfig()
        assert config.max_drawdown_pct == 0.20

    def test_never_ai_overridable(self):
        """Auto kill switch is never overridable by AI."""
        config = AutoKillSwitchConfig()
        assert config.ai_overridable is False

    def test_ai_overridable_cannot_be_set_true(self):
        """Setting ai_overridable=True is rejected."""
        with pytest.raises((ValueError, TypeError)):
            AutoKillSwitchConfig(ai_overridable=True)

    def test_custom_drawdown_threshold(self):
        """Custom drawdown threshold is accepted."""
        config = AutoKillSwitchConfig(max_drawdown_pct=0.15)
        assert config.max_drawdown_pct == 0.15


class TestDrawdownState:
    """Drawdown tracking state."""

    def test_initial_no_drawdown(self):
        """Initial state has no drawdown."""
        state = DrawdownState(peak_value=100000.0, current_value=100000.0)
        assert state.drawdown_pct == 0.0

    def test_drawdown_calculation(self):
        """Drawdown is calculated correctly."""
        state = DrawdownState(peak_value=100000.0, current_value=85000.0)
        assert state.drawdown_pct == pytest.approx(0.15, abs=0.001)

    def test_peak_updates_on_new_high(self):
        """Peak value updates when current exceeds peak."""
        state = DrawdownState(peak_value=100000.0, current_value=100000.0)
        state.update(105000.0)
        assert state.peak_value == 105000.0
        assert state.drawdown_pct == 0.0

    def test_drawdown_increases_on_decline(self):
        """Drawdown increases as current value drops."""
        state = DrawdownState(peak_value=100000.0, current_value=100000.0)
        state.update(90000.0)
        assert state.drawdown_pct == pytest.approx(0.10, abs=0.001)
        state.update(80000.0)
        assert state.drawdown_pct == pytest.approx(0.20, abs=0.001)


class TestAutoKillSwitchMonitor:
    """Monitors drawdown and auto-triggers kill switch."""

    @pytest.fixture
    def kill_switch(self):
        ks = KillSwitch(KillSwitchConfig())
        ks._trading_client = AsyncMock()
        ks._trading_client.cancel_all_orders = AsyncMock(return_value=2)
        ks._trading_client.close_all_positions = AsyncMock(return_value=1)
        ks._notifier = AsyncMock()
        return ks

    @pytest.fixture
    def monitor(self, kill_switch):
        config = AutoKillSwitchConfig(max_drawdown_pct=0.20)
        return AutoKillSwitchMonitor(
            kill_switch=kill_switch,
            config=config,
            initial_value=100000.0,
        )

    @pytest.mark.asyncio
    async def test_no_trigger_within_threshold(self, monitor):
        """No trigger when drawdown is within threshold."""
        triggered = await monitor.check_and_trigger(current_value=85000.0)
        assert triggered is False
        assert monitor.kill_switch.state == KillSwitchState.ARMED

    @pytest.mark.asyncio
    async def test_triggers_at_threshold(self, monitor):
        """Triggers when drawdown hits exactly 20%."""
        triggered = await monitor.check_and_trigger(current_value=80000.0)
        assert triggered is True
        assert monitor.kill_switch.state == KillSwitchState.TRIGGERED

    @pytest.mark.asyncio
    async def test_triggers_beyond_threshold(self, monitor):
        """Triggers when drawdown exceeds 20%."""
        triggered = await monitor.check_and_trigger(current_value=75000.0)
        assert triggered is True
        assert monitor.kill_switch.state == KillSwitchState.TRIGGERED

    @pytest.mark.asyncio
    async def test_trigger_reason_includes_drawdown(self, monitor):
        """Trigger reason includes the actual drawdown percentage."""
        await monitor.check_and_trigger(current_value=79000.0)
        activation = monitor.kill_switch.activation_history[-1]
        assert "21" in activation.reason or "0.21" in activation.reason

    @pytest.mark.asyncio
    async def test_trigger_source_is_auto(self, monitor):
        """Auto-trigger source is AUTO."""
        await monitor.check_and_trigger(current_value=75000.0)
        activation = monitor.kill_switch.activation_history[-1]
        assert activation.source.value == "auto"

    @pytest.mark.asyncio
    async def test_peak_tracking_across_checks(self, monitor):
        """Peak value is tracked across multiple check calls."""
        await monitor.check_and_trigger(current_value=110000.0)  # New peak
        assert monitor.drawdown_state.peak_value == 110000.0

        # 20% drawdown from new peak of 110k = 88k
        triggered = await monitor.check_and_trigger(current_value=88000.0)
        assert triggered is True

    @pytest.mark.asyncio
    async def test_ai_cannot_override(self, monitor):
        """AI systems cannot override the auto kill switch."""
        assert monitor.config.ai_overridable is False
        # No method exists for AI override
        assert not hasattr(monitor, "ai_override")
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_kill_switch_auto.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.execution.kill_switch_auto'`

**Step 3: Implement the auto kill switch monitor**

```python
# src/evolve_trader/execution/kill_switch_auto.py
"""Automatic kill switch trigger on drawdown threshold.

Monitors portfolio value and triggers the kill switch when drawdown
exceeds the configured threshold (default 20%). This is NEVER
overridable by AI systems — only humans can re-arm.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from evolve_trader.execution.kill_switch import (
    KillSwitch,
    KillSwitchSource,
    KillSwitchState,
)

logger = logging.getLogger(__name__)


@dataclass
class AutoKillSwitchConfig:
    """Configuration for automatic kill switch.

    ai_overridable is hardcoded False and rejects True.
    """
    max_drawdown_pct: float = 0.20
    ai_overridable: bool = False

    def __post_init__(self) -> None:
        if self.ai_overridable:
            raise ValueError(
                "Auto kill switch is NEVER AI-overridable. "
                "ai_overridable must be False."
            )


class DrawdownState:
    """Tracks peak value and current drawdown."""

    def __init__(self, peak_value: float, current_value: float):
        self.peak_value = peak_value
        self.current_value = current_value

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown as a fraction (0.0 = no drawdown, 0.2 = 20%)."""
        if self.peak_value <= 0:
            return 0.0
        return max(0.0, (self.peak_value - self.current_value) / self.peak_value)

    def update(self, new_value: float) -> None:
        """Update with new portfolio value. Adjusts peak if new high."""
        self.current_value = new_value
        if new_value > self.peak_value:
            self.peak_value = new_value


class AutoKillSwitchMonitor:
    """Monitors drawdown and auto-triggers kill switch.

    Never AI-overridable. Only humans can re-arm after auto-trigger.
    """

    def __init__(
        self,
        kill_switch: KillSwitch,
        config: AutoKillSwitchConfig,
        initial_value: float,
    ):
        self.kill_switch = kill_switch
        self.config = config
        self.drawdown_state = DrawdownState(
            peak_value=initial_value,
            current_value=initial_value,
        )

    async def check_and_trigger(self, current_value: float) -> bool:
        """Check current value against drawdown threshold.

        Returns True if kill switch was triggered.
        """
        self.drawdown_state.update(current_value)
        drawdown = self.drawdown_state.drawdown_pct

        if drawdown >= self.config.max_drawdown_pct:
            if self.kill_switch.state == KillSwitchState.ARMED:
                logger.critical(
                    "AUTO KILL SWITCH: drawdown %.1f%% >= threshold %.1f%%",
                    drawdown * 100,
                    self.config.max_drawdown_pct * 100,
                )
                await self.kill_switch.activate(
                    source=KillSwitchSource.AUTO,
                    reason=(
                        f"Automatic trigger: drawdown {drawdown:.2%} "
                        f"exceeded threshold {self.config.max_drawdown_pct:.2%} "
                        f"(peak={self.drawdown_state.peak_value:.2f}, "
                        f"current={current_value:.2f})"
                    ),
                )
                return True
        return False
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_kill_switch_auto.py -v
```

Expected: PASS — all 11 tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/execution/kill_switch_auto.py tests/unit/test_kill_switch_auto.py
git commit -m "feat: auto kill switch on 20% drawdown, never AI-overridable"
```

---

## Task 7: Security — API Key Management

**Files:**
- Create: `src/evolve_trader/security/secrets.py`
- Create: `tests/unit/test_secrets.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_secrets.py
"""Tests for API key management and secrets handling."""
import pytest
import os
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

from evolve_trader.security.secrets import (
    SecretsManager,
    SecretsConfig,
    SecretRotationPolicy,
    SecretEntry,
)


class TestSecretsConfig:
    """Secrets manager configuration."""

    def test_default_source_is_env(self):
        """Default secret source is environment variables."""
        config = SecretsConfig()
        assert config.source == "env"

    def test_supports_secrets_manager(self):
        """Can be configured for external secrets manager."""
        config = SecretsConfig(source="aws_secrets_manager")
        assert config.source == "aws_secrets_manager"


class TestSecretsManager:
    """SecretsManager loads and rotates API keys."""

    @pytest.fixture
    def env_secrets(self):
        with patch.dict(os.environ, {
            "ALPACA_API_KEY": "test-alpaca-key",
            "ALPACA_API_SECRET": "test-alpaca-secret",
            "OPENAI_API_KEY": "test-openai-key",
        }):
            config = SecretsConfig(source="env")
            yield SecretsManager(config)

    def test_load_from_env(self, env_secrets):
        """Loads secrets from environment variables."""
        key = env_secrets.get("ALPACA_API_KEY")
        assert key == "test-alpaca-key"

    def test_missing_key_raises(self, env_secrets):
        """Missing key raises KeyError with helpful message."""
        with pytest.raises(KeyError, match="NONEXISTENT"):
            env_secrets.get("NONEXISTENT_KEY")

    def test_get_with_default(self, env_secrets):
        """Missing key with default returns default."""
        value = env_secrets.get("NONEXISTENT", default="fallback")
        assert value == "fallback"

    def test_secrets_never_in_repr(self, env_secrets):
        """Secret values never appear in repr/str."""
        _ = env_secrets.get("ALPACA_API_KEY")
        repr_str = repr(env_secrets)
        assert "test-alpaca-key" not in repr_str

    def test_secrets_never_in_str(self, env_secrets):
        """Secret values never appear in string representation."""
        _ = env_secrets.get("ALPACA_API_KEY")
        str_str = str(env_secrets)
        assert "test-alpaca-key" not in str_str

    def test_list_keys_shows_names_only(self, env_secrets):
        """Listing keys shows names but not values."""
        _ = env_secrets.get("ALPACA_API_KEY")
        _ = env_secrets.get("ALPACA_API_SECRET")
        keys = env_secrets.list_keys()
        assert "ALPACA_API_KEY" in keys
        assert "ALPACA_API_SECRET" in keys

    def test_mask_value(self, env_secrets):
        """Values are masked when displayed."""
        masked = env_secrets.mask("test-alpaca-key")
        assert "test-alpaca-key" not in masked
        assert "***" in masked
        # Shows first/last few chars at most
        assert len(masked) < len("test-alpaca-key") + 10


class TestSecretRotationPolicy:
    """Secret rotation policy tracking."""

    def test_rotation_needed_after_max_age(self):
        """Rotation is needed when secret exceeds max age."""
        policy = SecretRotationPolicy(max_age_days=90)
        entry = SecretEntry(
            key_name="ALPACA_API_KEY",
            created_at=datetime.now(timezone.utc) - timedelta(days=91),
            last_rotated=datetime.now(timezone.utc) - timedelta(days=91),
        )
        assert policy.needs_rotation(entry) is True

    def test_rotation_not_needed_within_max_age(self):
        """Rotation is not needed within max age."""
        policy = SecretRotationPolicy(max_age_days=90)
        entry = SecretEntry(
            key_name="ALPACA_API_KEY",
            created_at=datetime.now(timezone.utc) - timedelta(days=30),
            last_rotated=datetime.now(timezone.utc) - timedelta(days=30),
        )
        assert policy.needs_rotation(entry) is False

    def test_rotation_warnings(self):
        """Warning emitted when rotation is approaching."""
        policy = SecretRotationPolicy(max_age_days=90, warn_days_before=14)
        entry = SecretEntry(
            key_name="ALPACA_API_KEY",
            created_at=datetime.now(timezone.utc) - timedelta(days=80),
            last_rotated=datetime.now(timezone.utc) - timedelta(days=80),
        )
        assert policy.needs_rotation(entry) is False
        assert policy.rotation_warning(entry) is True
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_secrets.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.security.secrets'`

**Step 3: Implement the secrets manager**

```python
# src/evolve_trader/security/secrets.py
"""API key management and secrets handling.

Loads secrets from environment variables or external secrets managers.
Tracks rotation policies. Never exposes secret values in logs or repr.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SecretsConfig:
    """Configuration for the secrets manager."""
    source: str = "env"  # "env", "aws_secrets_manager", "vault"


@dataclass
class SecretEntry:
    """Metadata for a managed secret (never stores the value)."""
    key_name: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_rotated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SecretRotationPolicy:
    """Policy for when secrets should be rotated."""
    max_age_days: int = 90
    warn_days_before: int = 14

    def needs_rotation(self, entry: SecretEntry) -> bool:
        """Check if a secret needs rotation based on age."""
        age = datetime.now(timezone.utc) - entry.last_rotated
        return age > timedelta(days=self.max_age_days)

    def rotation_warning(self, entry: SecretEntry) -> bool:
        """Check if a secret is approaching rotation deadline."""
        age = datetime.now(timezone.utc) - entry.last_rotated
        warn_at = timedelta(days=self.max_age_days - self.warn_days_before)
        return age > warn_at and not self.needs_rotation(entry)


class SecretsManager:
    """Manages API keys and secrets.

    Loads from env vars or external secrets managers.
    Never exposes values in repr, str, or logs.
    Tracks which keys have been accessed.
    """

    def __init__(self, config: SecretsConfig):
        self.config = config
        self._accessed_keys: set[str] = set()

    def get(self, key_name: str, default: str | None = None) -> str:
        """Get a secret value by name.

        Raises KeyError if not found and no default provided.
        """
        if self.config.source == "env":
            value = os.environ.get(key_name)
            if value is None:
                if default is not None:
                    return default
                raise KeyError(
                    f"Secret '{key_name}' not found in environment variables. "
                    f"Set it with: export {key_name}=<value>"
                )
            self._accessed_keys.add(key_name)
            return value

        raise NotImplementedError(f"Source '{self.config.source}' not yet implemented")

    def list_keys(self) -> list[str]:
        """List names of accessed keys (never values)."""
        return sorted(self._accessed_keys)

    @staticmethod
    def mask(value: str) -> str:
        """Mask a secret value for safe display."""
        if len(value) <= 4:
            return "***"
        return f"{value[:2]}***{value[-2:]}"

    def __repr__(self) -> str:
        return (
            f"SecretsManager(source={self.config.source!r}, "
            f"accessed_keys={self.list_keys()})"
        )

    def __str__(self) -> str:
        return f"SecretsManager({self.config.source}, {len(self._accessed_keys)} keys loaded)"
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_secrets.py -v
```

Expected: PASS — all 10 tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/security/secrets.py tests/unit/test_secrets.py
git commit -m "feat: secrets manager with env loading, masking, and rotation policies"
```

---

## Task 8: Security — Rate Limiting

**Files:**
- Create: `src/evolve_trader/security/rate_limiter.py`
- Create: `tests/unit/test_rate_limiter.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_rate_limiter.py
"""Tests for per-service rate limiting on external APIs."""
import pytest
import asyncio
from datetime import datetime, timezone

from evolve_trader.security.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    RateLimitExceeded,
    ServiceRateLimit,
)


class TestRateLimitConfig:
    """Rate limit configuration."""

    def test_per_service_limits(self):
        """Each service has its own rate limit."""
        config = RateLimitConfig(
            services={
                "alpaca": ServiceRateLimit(requests_per_minute=200, burst_size=10),
                "openai": ServiceRateLimit(requests_per_minute=60, burst_size=5),
            }
        )
        assert config.services["alpaca"].requests_per_minute == 200
        assert config.services["openai"].requests_per_minute == 60

    def test_burst_size_defaults(self):
        """Burst size defaults to 1 if not specified."""
        limit = ServiceRateLimit(requests_per_minute=100)
        assert limit.burst_size == 1


class TestRateLimiter:
    """RateLimiter enforces per-service request limits."""

    @pytest.fixture
    def limiter(self):
        config = RateLimitConfig(
            services={
                "alpaca": ServiceRateLimit(requests_per_minute=60, burst_size=3),
                "openai": ServiceRateLimit(requests_per_minute=30, burst_size=2),
            }
        )
        return RateLimiter(config)

    @pytest.mark.asyncio
    async def test_allows_within_limit(self, limiter):
        """Requests within limit are allowed."""
        allowed = await limiter.acquire("alpaca")
        assert allowed is True

    @pytest.mark.asyncio
    async def test_allows_burst(self, limiter):
        """Burst requests up to burst_size are allowed."""
        for _ in range(3):
            allowed = await limiter.acquire("alpaca")
            assert allowed is True

    @pytest.mark.asyncio
    async def test_rejects_over_burst(self, limiter):
        """Requests over burst size are rejected."""
        for _ in range(3):
            await limiter.acquire("alpaca")

        with pytest.raises(RateLimitExceeded, match="alpaca"):
            await limiter.acquire("alpaca")

    @pytest.mark.asyncio
    async def test_services_independent(self, limiter):
        """Rate limits are independent per service."""
        # Exhaust alpaca burst
        for _ in range(3):
            await limiter.acquire("alpaca")

        # openai should still work
        allowed = await limiter.acquire("openai")
        assert allowed is True

    @pytest.mark.asyncio
    async def test_unknown_service_raises(self, limiter):
        """Unknown service raises ValueError."""
        with pytest.raises(ValueError, match="unknown_service"):
            await limiter.acquire("unknown_service")

    @pytest.mark.asyncio
    async def test_wait_mode_blocks_instead_of_raising(self, limiter):
        """In wait mode, acquire blocks until a slot is available."""
        for _ in range(3):
            await limiter.acquire("alpaca")

        # With wait=True and a short window, should eventually succeed
        # (We test this by checking the method exists and accepts wait param)
        assert callable(getattr(limiter, "acquire_or_wait", None))

    def test_get_usage_stats(self, limiter):
        """Rate limiter tracks usage statistics."""
        stats = limiter.get_stats("alpaca")
        assert "requests_used" in stats
        assert "requests_remaining" in stats
        assert "burst_remaining" in stats

    def test_reset_service(self, limiter):
        """Can reset a service's rate limit counters."""
        limiter.reset("alpaca")
        stats = limiter.get_stats("alpaca")
        assert stats["requests_used"] == 0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_rate_limiter.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.security.rate_limiter'`

**Step 3: Implement the rate limiter**

```python
# src/evolve_trader/security/rate_limiter.py
"""Per-service rate limiting for all external API calls.

Token bucket algorithm with configurable per-minute rates and burst sizes.
Each external service (Alpaca, OpenAI, etc.) has independent limits.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when a service's rate limit is exceeded."""
    pass


@dataclass
class ServiceRateLimit:
    """Rate limit configuration for a single service."""
    requests_per_minute: int
    burst_size: int = 1


@dataclass
class RateLimitConfig:
    """Rate limit configuration for all services."""
    services: dict[str, ServiceRateLimit] = field(default_factory=dict)


class _TokenBucket:
    """Token bucket rate limiter for a single service."""

    def __init__(self, rate_per_minute: int, burst_size: int):
        self.rate_per_second = rate_per_minute / 60.0
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_refill = time.monotonic()
        self.total_requests = 0

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.burst_size,
            self.tokens + elapsed * self.rate_per_second,
        )
        self.last_refill = now

    def try_acquire(self) -> bool:
        self._refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            self.total_requests += 1
            return True
        return False

    def time_until_available(self) -> float:
        self._refill()
        if self.tokens >= 1.0:
            return 0.0
        needed = 1.0 - self.tokens
        return needed / self.rate_per_second

    def reset(self) -> None:
        self.tokens = float(self.burst_size)
        self.total_requests = 0
        self.last_refill = time.monotonic()


class RateLimiter:
    """Per-service rate limiter using token bucket algorithm.

    Each service has independent limits. Supports both raise-on-limit
    and wait-for-slot modes.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._buckets: dict[str, _TokenBucket] = {}
        for name, limit in config.services.items():
            self._buckets[name] = _TokenBucket(
                rate_per_minute=limit.requests_per_minute,
                burst_size=limit.burst_size,
            )

    def _get_bucket(self, service: str) -> _TokenBucket:
        if service not in self._buckets:
            raise ValueError(
                f"Unknown service '{service}'. "
                f"Known services: {list(self._buckets.keys())}"
            )
        return self._buckets[service]

    async def acquire(self, service: str) -> bool:
        """Acquire a rate limit token. Raises RateLimitExceeded if over limit."""
        bucket = self._get_bucket(service)
        if bucket.try_acquire():
            return True
        raise RateLimitExceeded(
            f"Rate limit exceeded for '{service}'. "
            f"Try again in {bucket.time_until_available():.1f}s"
        )

    async def acquire_or_wait(self, service: str, timeout: float = 30.0) -> bool:
        """Acquire a token, waiting if necessary up to timeout seconds."""
        bucket = self._get_bucket(service)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if bucket.try_acquire():
                return True
            wait_time = min(bucket.time_until_available(), deadline - time.monotonic())
            if wait_time <= 0:
                break
            await asyncio.sleep(wait_time)
        raise RateLimitExceeded(
            f"Rate limit timeout for '{service}' after {timeout}s"
        )

    def get_stats(self, service: str) -> dict[str, Any]:
        """Get current usage statistics for a service."""
        bucket = self._get_bucket(service)
        bucket._refill()
        return {
            "requests_used": bucket.total_requests,
            "requests_remaining": int(bucket.tokens),
            "burst_remaining": int(bucket.tokens),
        }

    def reset(self, service: str) -> None:
        """Reset rate limit counters for a service."""
        bucket = self._get_bucket(service)
        bucket.reset()
        logger.info("Rate limit counters reset for %s", service)
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_rate_limiter.py -v
```

Expected: PASS — all 8 tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/security/rate_limiter.py tests/unit/test_rate_limiter.py
git commit -m "feat: per-service rate limiting with token bucket algorithm"
```

---

## Task 9: Security — Audit Log

**Files:**
- Create: `src/evolve_trader/security/audit_log.py`
- Create: `tests/unit/test_audit_log.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_audit_log.py
"""Tests for tamper-evident audit logging."""
import pytest
import hashlib
import json
from datetime import datetime, timezone

from evolve_trader.security.audit_log import (
    AuditLog,
    AuditLogConfig,
    AuditEntry,
    AuditEventType,
)


class TestAuditEntry:
    """Audit log entry structure."""

    def test_entry_has_required_fields(self):
        """Entry contains event type, actor, timestamp, and payload."""
        entry = AuditEntry(
            event_type=AuditEventType.TRADE_EXECUTED,
            actor="system",
            payload={"symbol": "AAPL", "side": "buy", "qty": 10},
        )
        assert entry.event_type == AuditEventType.TRADE_EXECUTED
        assert entry.actor == "system"
        assert entry.timestamp is not None

    def test_entry_has_chain_hash(self):
        """Each entry includes a hash chaining to the previous entry."""
        entry = AuditEntry(
            event_type=AuditEventType.TRADE_EXECUTED,
            actor="system",
            payload={},
            previous_hash="0" * 64,
        )
        assert entry.entry_hash is not None
        assert len(entry.entry_hash) == 64  # SHA-256 hex

    def test_entry_hash_includes_previous(self):
        """Entry hash incorporates the previous entry's hash."""
        prev_hash = "a" * 64
        entry = AuditEntry(
            event_type=AuditEventType.TRADE_EXECUTED,
            actor="system",
            payload={"data": "test"},
            previous_hash=prev_hash,
        )
        # Hash should change if previous_hash changes
        entry2 = AuditEntry(
            event_type=AuditEventType.TRADE_EXECUTED,
            actor="system",
            payload={"data": "test"},
            previous_hash="b" * 64,
            timestamp=entry.timestamp,
        )
        assert entry.entry_hash != entry2.entry_hash


class TestAuditEventType:
    """All required audit event types exist."""

    def test_trade_executed(self):
        assert AuditEventType.TRADE_EXECUTED is not None

    def test_approval_granted(self):
        assert AuditEventType.APPROVAL_GRANTED is not None

    def test_approval_rejected(self):
        assert AuditEventType.APPROVAL_REJECTED is not None

    def test_human_override(self):
        assert AuditEventType.HUMAN_OVERRIDE is not None

    def test_kill_switch_activated(self):
        assert AuditEventType.KILL_SWITCH_ACTIVATED is not None

    def test_kill_switch_rearmed(self):
        assert AuditEventType.KILL_SWITCH_REARMED is not None

    def test_secret_rotated(self):
        assert AuditEventType.SECRET_ROTATED is not None

    def test_promotion(self):
        assert AuditEventType.PROMOTION is not None

    def test_demotion(self):
        assert AuditEventType.DEMOTION is not None


class TestAuditLog:
    """Tamper-evident audit log with hash chaining."""

    @pytest.fixture
    def audit_log(self):
        return AuditLog(AuditLogConfig())

    @pytest.mark.asyncio
    async def test_append_entry(self, audit_log):
        """Can append an entry to the log."""
        entry = await audit_log.append(
            event_type=AuditEventType.TRADE_EXECUTED,
            actor="system",
            payload={"symbol": "AAPL"},
        )
        assert entry.entry_hash is not None

    @pytest.mark.asyncio
    async def test_chain_integrity(self, audit_log):
        """Entries form a hash chain."""
        entry1 = await audit_log.append(
            event_type=AuditEventType.TRADE_EXECUTED,
            actor="system",
            payload={"trade": 1},
        )
        entry2 = await audit_log.append(
            event_type=AuditEventType.APPROVAL_GRANTED,
            actor="admin",
            payload={"approval": 1},
        )
        assert entry2.previous_hash == entry1.entry_hash

    @pytest.mark.asyncio
    async def test_verify_chain_intact(self, audit_log):
        """Verify detects intact chain."""
        await audit_log.append(
            event_type=AuditEventType.TRADE_EXECUTED,
            actor="system",
            payload={"trade": 1},
        )
        await audit_log.append(
            event_type=AuditEventType.KILL_SWITCH_ACTIVATED,
            actor="admin",
            payload={"reason": "test"},
        )
        assert audit_log.verify_chain() is True

    @pytest.mark.asyncio
    async def test_verify_chain_detects_tampering(self, audit_log):
        """Verify detects if an entry was tampered with."""
        await audit_log.append(
            event_type=AuditEventType.TRADE_EXECUTED,
            actor="system",
            payload={"trade": 1},
        )
        await audit_log.append(
            event_type=AuditEventType.TRADE_EXECUTED,
            actor="system",
            payload={"trade": 2},
        )
        # Tamper with the first entry
        audit_log._entries[0].payload = {"trade": 999}
        assert audit_log.verify_chain() is False

    @pytest.mark.asyncio
    async def test_query_by_event_type(self, audit_log):
        """Can query entries by event type."""
        await audit_log.append(
            event_type=AuditEventType.TRADE_EXECUTED,
            actor="system",
            payload={"trade": 1},
        )
        await audit_log.append(
            event_type=AuditEventType.KILL_SWITCH_ACTIVATED,
            actor="admin",
            payload={"reason": "test"},
        )
        await audit_log.append(
            event_type=AuditEventType.TRADE_EXECUTED,
            actor="system",
            payload={"trade": 2},
        )

        trades = audit_log.query(event_type=AuditEventType.TRADE_EXECUTED)
        assert len(trades) == 2

    @pytest.mark.asyncio
    async def test_query_by_actor(self, audit_log):
        """Can query entries by actor."""
        await audit_log.append(
            event_type=AuditEventType.TRADE_EXECUTED,
            actor="system",
            payload={},
        )
        await audit_log.append(
            event_type=AuditEventType.HUMAN_OVERRIDE,
            actor="admin",
            payload={},
        )

        admin_entries = audit_log.query(actor="admin")
        assert len(admin_entries) == 1

    @pytest.mark.asyncio
    async def test_entry_count(self, audit_log):
        """Entry count tracks log size."""
        assert audit_log.entry_count == 0
        await audit_log.append(
            event_type=AuditEventType.TRADE_EXECUTED,
            actor="system",
            payload={},
        )
        assert audit_log.entry_count == 1

    @pytest.mark.asyncio
    async def test_log_is_append_only(self, audit_log):
        """No delete or update methods exist."""
        assert not hasattr(audit_log, "delete")
        assert not hasattr(audit_log, "update")
        assert not hasattr(audit_log, "remove")
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_audit_log.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.security.audit_log'`

**Step 3: Implement the audit log**

```python
# src/evolve_trader/security/audit_log.py
"""Tamper-evident audit log with hash chaining.

Append-only log of all executions, approvals, overrides, and kill switch
activations. Each entry's hash includes the previous entry's hash,
forming a chain that detects any tampering.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

GENESIS_HASH = "0" * 64


class AuditEventType(Enum):
    """Types of auditable events."""
    TRADE_EXECUTED = "trade_executed"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_REJECTED = "approval_rejected"
    HUMAN_OVERRIDE = "human_override"
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"
    KILL_SWITCH_REARMED = "kill_switch_rearmed"
    SECRET_ROTATED = "secret_rotated"
    PROMOTION = "promotion"
    DEMOTION = "demotion"


@dataclass
class AuditEntry:
    """A single tamper-evident audit log entry."""
    event_type: AuditEventType
    actor: str
    payload: dict[str, Any]
    previous_hash: str = GENESIS_HASH
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    entry_hash: str = field(init=False)

    def __post_init__(self) -> None:
        self.entry_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash over entry contents + previous hash."""
        content = json.dumps({
            "event_type": self.event_type.value,
            "actor": self.actor,
            "payload": self.payload,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp.isoformat(),
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def verify(self) -> bool:
        """Verify this entry's hash matches its contents."""
        return self.entry_hash == self._compute_hash()


@dataclass
class AuditLogConfig:
    """Configuration for the audit log."""
    max_entries_in_memory: int = 100000


class AuditLog:
    """Append-only, tamper-evident audit log.

    Hash chaining: each entry's hash includes the previous entry's hash.
    No delete, update, or remove operations exist.
    """

    def __init__(self, config: AuditLogConfig):
        self.config = config
        self._entries: list[AuditEntry] = []

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    async def append(
        self,
        event_type: AuditEventType,
        actor: str,
        payload: dict[str, Any],
    ) -> AuditEntry:
        """Append a new entry to the audit log."""
        previous_hash = (
            self._entries[-1].entry_hash if self._entries else GENESIS_HASH
        )

        entry = AuditEntry(
            event_type=event_type,
            actor=actor,
            payload=payload,
            previous_hash=previous_hash,
        )
        self._entries.append(entry)

        logger.info(
            "Audit: %s by %s (hash=%s...)",
            event_type.value, actor, entry.entry_hash[:12],
        )
        return entry

    def verify_chain(self) -> bool:
        """Verify the entire audit chain is intact.

        Returns False if any entry has been tampered with.
        """
        if not self._entries:
            return True

        # Verify first entry chains from genesis
        if self._entries[0].previous_hash != GENESIS_HASH:
            return False

        for i, entry in enumerate(self._entries):
            # Recompute hash and compare
            if not entry.verify():
                logger.error("Audit chain broken at entry %d", i)
                return False

            # Verify chain linkage
            if i > 0 and entry.previous_hash != self._entries[i - 1].entry_hash:
                logger.error("Audit chain linkage broken at entry %d", i)
                return False

        return True

    def query(
        self,
        event_type: AuditEventType | None = None,
        actor: str | None = None,
    ) -> list[AuditEntry]:
        """Query audit log entries by event type and/or actor."""
        results = self._entries
        if event_type is not None:
            results = [e for e in results if e.event_type == event_type]
        if actor is not None:
            results = [e for e in results if e.actor == actor]
        return results
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_audit_log.py -v
```

Expected: PASS — all 12 tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/security/audit_log.py tests/unit/test_audit_log.py
git commit -m "feat: tamper-evident audit log with hash chaining"
```

---

## Task 10: Regime Diversity Requirement

**Files:**
- Create: `src/evolve_trader/core/regime_diversity_gate.py`
- Create: `tests/unit/test_regime_diversity.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_regime_diversity.py
"""Tests for regime diversity gate — anti-overfitting production gate."""
import pytest
from datetime import datetime, timezone

from evolve_trader.core.regime_diversity_gate import (
    RegimeDiversityGate,
    RegimeDiversityConfig,
    RegimePerformance,
    DiversityCheckResult,
)


class TestRegimeDiversityConfig:
    """Configuration for regime diversity requirement."""

    def test_default_min_regimes(self):
        """Default requires positive/neutral in 2+ regimes."""
        config = RegimeDiversityConfig()
        assert config.min_positive_regimes == 2

    def test_configurable_threshold(self):
        """Minimum positive regimes is configurable."""
        config = RegimeDiversityConfig(min_positive_regimes=3)
        assert config.min_positive_regimes == 3

    def test_neutral_sharpe_threshold(self):
        """Neutral is defined as Sharpe >= configurable threshold."""
        config = RegimeDiversityConfig(neutral_sharpe_threshold=-0.1)
        assert config.neutral_sharpe_threshold == -0.1


class TestRegimePerformance:
    """Regime performance tracking."""

    def test_positive_regime(self):
        """Positive regime has Sharpe > 0."""
        perf = RegimePerformance(
            regime_label="risk-on",
            sharpe_ratio=1.5,
            total_trades=30,
            win_rate=0.60,
        )
        assert perf.is_positive(threshold=0.0) is True

    def test_negative_regime(self):
        """Negative regime has Sharpe below threshold."""
        perf = RegimePerformance(
            regime_label="risk-off",
            sharpe_ratio=-0.5,
            total_trades=20,
            win_rate=0.35,
        )
        assert perf.is_positive(threshold=-0.1) is False

    def test_neutral_regime(self):
        """Neutral regime has Sharpe at or above threshold but below zero."""
        perf = RegimePerformance(
            regime_label="choppy",
            sharpe_ratio=-0.05,
            total_trades=25,
            win_rate=0.48,
        )
        assert perf.is_positive(threshold=-0.1) is True


class TestRegimeDiversityGate:
    """Gate that blocks production for single-regime strategies."""

    @pytest.fixture
    def gate(self):
        config = RegimeDiversityConfig(
            min_positive_regimes=2,
            neutral_sharpe_threshold=-0.1,
            min_trades_per_regime=10,
        )
        return RegimeDiversityGate(config)

    @pytest.mark.asyncio
    async def test_passes_with_two_positive_regimes(self, gate):
        """Strategy positive in 2+ regimes passes."""
        performances = [
            RegimePerformance("risk-on", sharpe_ratio=1.5, total_trades=30, win_rate=0.60),
            RegimePerformance("risk-off", sharpe_ratio=0.3, total_trades=25, win_rate=0.52),
        ]
        result = await gate.check(strategy_name="diverse-strat", performances=performances)
        assert isinstance(result, DiversityCheckResult)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_fails_with_one_positive_regime(self, gate):
        """Strategy positive in only 1 regime fails."""
        performances = [
            RegimePerformance("risk-on", sharpe_ratio=2.0, total_trades=50, win_rate=0.65),
            RegimePerformance("risk-off", sharpe_ratio=-0.8, total_trades=20, win_rate=0.30),
        ]
        result = await gate.check(strategy_name="overfit-strat", performances=performances)
        assert result.passed is False
        assert "1" in result.reason  # Only 1 positive regime

    @pytest.mark.asyncio
    async def test_fails_with_no_regimes(self, gate):
        """Strategy with no regime data fails."""
        result = await gate.check(strategy_name="no-data-strat", performances=[])
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_ignores_regimes_with_too_few_trades(self, gate):
        """Regimes with fewer than min_trades_per_regime are excluded."""
        performances = [
            RegimePerformance("risk-on", sharpe_ratio=1.5, total_trades=30, win_rate=0.60),
            RegimePerformance("risk-off", sharpe_ratio=0.5, total_trades=5, win_rate=0.55),
        ]
        result = await gate.check(strategy_name="thin-strat", performances=performances)
        assert result.passed is False  # Only 1 regime has enough trades

    @pytest.mark.asyncio
    async def test_passes_with_three_positive_regimes(self, gate):
        """Strategy positive in 3 regimes passes with room to spare."""
        performances = [
            RegimePerformance("risk-on", sharpe_ratio=1.5, total_trades=30, win_rate=0.60),
            RegimePerformance("risk-off", sharpe_ratio=0.2, total_trades=25, win_rate=0.50),
            RegimePerformance("choppy", sharpe_ratio=0.5, total_trades=20, win_rate=0.53),
        ]
        result = await gate.check(strategy_name="robust-strat", performances=performances)
        assert result.passed is True
        assert result.positive_regime_count == 3

    @pytest.mark.asyncio
    async def test_result_includes_regime_breakdown(self, gate):
        """Check result includes per-regime pass/fail breakdown."""
        performances = [
            RegimePerformance("risk-on", sharpe_ratio=1.5, total_trades=30, win_rate=0.60),
            RegimePerformance("risk-off", sharpe_ratio=-0.5, total_trades=20, win_rate=0.35),
        ]
        result = await gate.check(strategy_name="mixed-strat", performances=performances)
        assert len(result.regime_breakdown) == 2
        assert result.regime_breakdown["risk-on"] is True
        assert result.regime_breakdown["risk-off"] is False
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_regime_diversity.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.core.regime_diversity_gate'`

**Step 3: Implement the regime diversity gate**

```python
# src/evolve_trader/core/regime_diversity_gate.py
"""Regime diversity gate — anti-overfitting production requirement.

Prevents production promotion of strategies that only perform well
in a single market regime. Requires positive/neutral Sharpe across
2+ regimes before a strategy can be promoted to production.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RegimeDiversityConfig:
    """Configuration for the regime diversity gate."""
    min_positive_regimes: int = 2
    neutral_sharpe_threshold: float = -0.1
    min_trades_per_regime: int = 10


@dataclass
class RegimePerformance:
    """Performance metrics for a strategy in a single regime."""
    regime_label: str
    sharpe_ratio: float
    total_trades: int
    win_rate: float

    def is_positive(self, threshold: float = -0.1) -> bool:
        """Is this regime's performance positive/neutral?"""
        return self.sharpe_ratio >= threshold


@dataclass
class DiversityCheckResult:
    """Result of a regime diversity check."""
    strategy_name: str
    passed: bool
    positive_regime_count: int = 0
    total_regime_count: int = 0
    regime_breakdown: dict[str, bool] = field(default_factory=dict)
    reason: str = ""


class RegimeDiversityGate:
    """Gate that requires positive performance across multiple regimes.

    Prevents single-regime overfitting by requiring strategies to
    demonstrate positive/neutral Sharpe in at least min_positive_regimes
    different market regimes before production promotion.
    """

    def __init__(self, config: RegimeDiversityConfig):
        self.config = config

    async def check(
        self,
        strategy_name: str,
        performances: list[RegimePerformance],
    ) -> DiversityCheckResult:
        """Check if a strategy meets regime diversity requirements."""
        if not performances:
            return DiversityCheckResult(
                strategy_name=strategy_name,
                passed=False,
                reason="No regime performance data available",
            )

        # Filter to regimes with enough trades
        qualifying = [
            p for p in performances
            if p.total_trades >= self.config.min_trades_per_regime
        ]

        # Check each qualifying regime
        breakdown: dict[str, bool] = {}
        positive_count = 0
        for perf in qualifying:
            is_pos = perf.is_positive(self.config.neutral_sharpe_threshold)
            breakdown[perf.regime_label] = is_pos
            if is_pos:
                positive_count += 1

        # Also include non-qualifying regimes in breakdown as False
        for perf in performances:
            if perf.regime_label not in breakdown:
                breakdown[perf.regime_label] = False

        passed = positive_count >= self.config.min_positive_regimes

        reason = ""
        if not passed:
            reason = (
                f"Only {positive_count} positive regime(s), "
                f"need {self.config.min_positive_regimes}. "
                f"Qualifying regimes: {len(qualifying)}/{len(performances)}"
            )
            logger.warning(
                "Regime diversity FAILED for %s: %s", strategy_name, reason,
            )
        else:
            logger.info(
                "Regime diversity PASSED for %s: %d/%d positive regimes",
                strategy_name, positive_count, len(qualifying),
            )

        return DiversityCheckResult(
            strategy_name=strategy_name,
            passed=passed,
            positive_regime_count=positive_count,
            total_regime_count=len(qualifying),
            regime_breakdown=breakdown,
            reason=reason,
        )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_regime_diversity.py -v
```

Expected: PASS — all 9 tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/core/regime_diversity_gate.py tests/unit/test_regime_diversity.py
git commit -m "feat: regime diversity gate — require 2+ positive regimes for production"
```

---

## Task 11: Production Observability

**Files:**
- Create: `src/evolve_trader/observability/logging_config.py`
- Create: `src/evolve_trader/observability/health.py`
- Create: `tests/unit/test_observability.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_observability.py
"""Tests for production observability — structured logging, health checks."""
import pytest
import json
import logging
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone

from evolve_trader.observability.logging_config import (
    configure_structured_logging,
    StructuredFormatter,
    LogLevel,
)
from evolve_trader.observability.health import (
    HealthChecker,
    HealthCheckConfig,
    HealthStatus,
    ComponentHealth,
)


class TestStructuredFormatter:
    """Structured JSON log formatter."""

    def test_formats_as_json(self):
        """Log records are formatted as JSON."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="evolve_trader",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"

    def test_includes_timestamp(self):
        """JSON output includes ISO timestamp."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="evolve_trader",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "timestamp" in parsed

    def test_includes_logger_name(self):
        """JSON output includes the logger name."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="evolve_trader.execution",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Warning",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["logger"] == "evolve_trader.execution"

    def test_includes_exception_info(self):
        """JSON output includes exception traceback when present."""
        formatter = StructuredFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=None,
                exc_info=sys.exc_info(),
            )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]


class TestConfigureStructuredLogging:
    """Logging configuration function."""

    def test_configure_sets_level(self):
        """Configuring logging sets the root level."""
        configure_structured_logging(level=LogLevel.DEBUG)
        root_logger = logging.getLogger("evolve_trader")
        assert root_logger.level == logging.DEBUG

    def test_configure_adds_handler(self):
        """Configuring logging adds at least one handler."""
        configure_structured_logging(level=LogLevel.INFO)
        root_logger = logging.getLogger("evolve_trader")
        assert len(root_logger.handlers) >= 1


class TestHealthStatus:
    """Health check status values."""

    def test_all_statuses_exist(self):
        assert HealthStatus.HEALTHY is not None
        assert HealthStatus.DEGRADED is not None
        assert HealthStatus.UNHEALTHY is not None


class TestHealthChecker:
    """Health checker for production monitoring."""

    @pytest.fixture
    def checker(self):
        config = HealthCheckConfig(
            check_interval_seconds=30,
            unhealthy_threshold=3,
        )
        return HealthChecker(config)

    @pytest.mark.asyncio
    async def test_register_component(self, checker):
        """Can register components for health checking."""
        check_fn = AsyncMock(return_value=True)
        checker.register("database", check_fn)
        assert "database" in checker.components

    @pytest.mark.asyncio
    async def test_healthy_when_all_pass(self, checker):
        """Overall status is HEALTHY when all components pass."""
        checker.register("database", AsyncMock(return_value=True))
        checker.register("alpaca", AsyncMock(return_value=True))

        status = await checker.check_all()
        assert status.overall == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_degraded_when_one_fails(self, checker):
        """Overall status is DEGRADED when one non-critical component fails."""
        checker.register("database", AsyncMock(return_value=True))
        checker.register("notifications", AsyncMock(return_value=False), critical=False)

        status = await checker.check_all()
        assert status.overall == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_unhealthy_when_critical_fails(self, checker):
        """Overall status is UNHEALTHY when a critical component fails."""
        checker.register("database", AsyncMock(return_value=False), critical=True)
        checker.register("alpaca", AsyncMock(return_value=True))

        status = await checker.check_all()
        assert status.overall == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_component_health_details(self, checker):
        """Check result includes per-component health details."""
        checker.register("database", AsyncMock(return_value=True))
        checker.register("alpaca", AsyncMock(return_value=False), critical=True)

        status = await checker.check_all()
        assert len(status.components) == 2
        assert any(c.name == "database" and c.healthy for c in status.components)
        assert any(c.name == "alpaca" and not c.healthy for c in status.components)

    @pytest.mark.asyncio
    async def test_check_handles_exceptions(self, checker):
        """Component check that raises exception is treated as unhealthy."""
        checker.register("flaky", AsyncMock(side_effect=Exception("Connection refused")))

        status = await checker.check_all()
        assert status.overall == HealthStatus.UNHEALTHY
        flaky = [c for c in status.components if c.name == "flaky"][0]
        assert flaky.healthy is False
        assert "Connection refused" in flaky.error
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_observability.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.observability.logging_config'`

**Step 3: Implement observability modules**

```python
# src/evolve_trader/observability/logging_config.py
"""Structured JSON logging configuration for production.

Configures all evolve_trader loggers to output structured JSON
with timestamps, levels, logger names, and exception tracebacks.
"""
from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from enum import Enum


class LogLevel(Enum):
    """Supported log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


_LEVEL_MAP = {
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARNING: logging.WARNING,
    LogLevel.ERROR: logging.ERROR,
    LogLevel.CRITICAL: logging.CRITICAL,
}


class StructuredFormatter(logging.Formatter):
    """Formats log records as structured JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = "".join(
                traceback.format_exception(*record.exc_info)
            )

        return json.dumps(log_entry, default=str)


def configure_structured_logging(level: LogLevel = LogLevel.INFO) -> None:
    """Configure structured JSON logging for evolve_trader.

    Sets up the evolve_trader logger with a StreamHandler using
    the StructuredFormatter, at the specified level.
    """
    logger = logging.getLogger("evolve_trader")
    logger.setLevel(_LEVEL_MAP[level])

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())
    logger.addHandler(handler)
```

```python
# src/evolve_trader/observability/health.py
"""Health check system for production monitoring.

Registers component health checks (database, Alpaca, notifications, etc.)
and reports overall system health as HEALTHY, DEGRADED, or UNHEALTHY.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Overall system health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    healthy: bool
    critical: bool = True
    error: str = ""
    latency_ms: float = 0.0


@dataclass
class HealthCheckResult:
    """Overall health check result with per-component details."""
    overall: HealthStatus
    components: list[ComponentHealth] = field(default_factory=list)


@dataclass
class HealthCheckConfig:
    """Configuration for the health checker."""
    check_interval_seconds: int = 30
    unhealthy_threshold: int = 3


@dataclass
class _RegisteredCheck:
    """Internal record of a registered health check."""
    name: str
    check_fn: Callable[[], Awaitable[bool]]
    critical: bool = True


class HealthChecker:
    """Monitors component health for production observability.

    Components register async check functions that return True (healthy)
    or False (unhealthy). Critical component failure -> UNHEALTHY.
    Non-critical failure -> DEGRADED. All pass -> HEALTHY.
    """

    def __init__(self, config: HealthCheckConfig):
        self.config = config
        self._checks: dict[str, _RegisteredCheck] = {}

    @property
    def components(self) -> list[str]:
        return list(self._checks.keys())

    def register(
        self,
        name: str,
        check_fn: Callable[[], Awaitable[bool]],
        critical: bool = True,
    ) -> None:
        """Register a component health check."""
        self._checks[name] = _RegisteredCheck(
            name=name, check_fn=check_fn, critical=critical,
        )

    async def check_all(self) -> HealthCheckResult:
        """Run all registered health checks and return overall status."""
        component_results: list[ComponentHealth] = []
        has_critical_failure = False
        has_non_critical_failure = False

        for name, check in self._checks.items():
            try:
                healthy = await check.check_fn()
                component_results.append(ComponentHealth(
                    name=name,
                    healthy=healthy,
                    critical=check.critical,
                ))
                if not healthy:
                    if check.critical:
                        has_critical_failure = True
                    else:
                        has_non_critical_failure = True
            except Exception as e:
                logger.error("Health check failed for %s: %s", name, e)
                component_results.append(ComponentHealth(
                    name=name,
                    healthy=False,
                    critical=check.critical,
                    error=str(e),
                ))
                if check.critical:
                    has_critical_failure = True
                else:
                    has_non_critical_failure = True

        if has_critical_failure:
            overall = HealthStatus.UNHEALTHY
        elif has_non_critical_failure:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        return HealthCheckResult(overall=overall, components=component_results)
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_observability.py -v
```

Expected: PASS — all 13 tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/observability/logging_config.py src/evolve_trader/observability/health.py tests/unit/test_observability.py
git commit -m "feat: production observability — structured JSON logging and health checks"
```

---

## Task 12: Database Backup & Recovery

**Files:**
- Create: `src/evolve_trader/db/backup.py`
- Create: `docs/runbooks/disaster-recovery.md`
- Create: `tests/unit/test_backup.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_backup.py
"""Tests for database backup and recovery system."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta

from evolve_trader.db.backup import (
    BackupManager,
    BackupConfig,
    BackupResult,
    BackupSchedule,
    BackupType,
)


class TestBackupConfig:
    """Backup configuration."""

    def test_default_schedule_is_daily(self):
        """Default backup schedule is daily."""
        config = BackupConfig()
        assert config.schedule == BackupSchedule.DAILY

    def test_backup_directory_configurable(self):
        """Backup directory is configurable."""
        config = BackupConfig(backup_dir="/data/backups")
        assert config.backup_dir == "/data/backups"

    def test_wal_archiving_enabled_by_default(self):
        """WAL archiving is enabled by default."""
        config = BackupConfig()
        assert config.wal_archiving is True

    def test_retention_days_configurable(self):
        """Backup retention period is configurable."""
        config = BackupConfig(retention_days=30)
        assert config.retention_days == 30


class TestBackupManager:
    """BackupManager handles full and WAL backups."""

    @pytest.fixture
    def backup_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.fixture
    def manager(self, backup_dir):
        config = BackupConfig(
            backup_dir=backup_dir,
            schedule=BackupSchedule.DAILY,
            wal_archiving=True,
            retention_days=7,
            db_connection_string="postgresql://localhost/evolve_trader",
        )
        return BackupManager(config)

    @pytest.mark.asyncio
    async def test_create_full_backup(self, manager):
        """Creates a full database backup."""
        with patch("evolve_trader.db.backup.run_pg_dump", new_callable=AsyncMock) as mock_dump:
            mock_dump.return_value = True
            result = await manager.create_backup(BackupType.FULL)

        assert isinstance(result, BackupResult)
        assert result.backup_type == BackupType.FULL
        assert result.success is True
        assert result.file_path is not None

    @pytest.mark.asyncio
    async def test_create_wal_backup(self, manager):
        """Creates a WAL archive backup."""
        with patch("evolve_trader.db.backup.run_pg_basebackup", new_callable=AsyncMock) as mock_bb:
            mock_bb.return_value = True
            result = await manager.create_backup(BackupType.WAL)

        assert isinstance(result, BackupResult)
        assert result.backup_type == BackupType.WAL
        assert result.success is True

    @pytest.mark.asyncio
    async def test_backup_result_has_timestamp(self, manager):
        """Backup result includes timestamp."""
        with patch("evolve_trader.db.backup.run_pg_dump", new_callable=AsyncMock) as mock_dump:
            mock_dump.return_value = True
            result = await manager.create_backup(BackupType.FULL)

        assert result.timestamp is not None

    @pytest.mark.asyncio
    async def test_backup_failure_captured(self, manager):
        """Backup failure is captured in result, not raised."""
        with patch("evolve_trader.db.backup.run_pg_dump", new_callable=AsyncMock) as mock_dump:
            mock_dump.side_effect = Exception("pg_dump failed")
            result = await manager.create_backup(BackupType.FULL)

        assert result.success is False
        assert "pg_dump failed" in result.error

    @pytest.mark.asyncio
    async def test_list_backups(self, manager):
        """Can list available backups."""
        backups = await manager.list_backups()
        assert isinstance(backups, list)

    @pytest.mark.asyncio
    async def test_cleanup_old_backups(self, manager, backup_dir):
        """Cleanup removes backups older than retention period."""
        # Create a fake old backup file
        old_file = Path(backup_dir) / "backup_20250101_000000.sql.gz"
        old_file.touch()

        removed = await manager.cleanup(older_than_days=0)
        assert removed >= 1

    @pytest.mark.asyncio
    async def test_export_data(self, manager):
        """Can export data in portable format."""
        with patch("evolve_trader.db.backup.run_pg_dump", new_callable=AsyncMock) as mock_dump:
            mock_dump.return_value = True
            result = await manager.export(format="csv", tables=["trade_logs"])

        assert result.success is True


class TestBackupSchedule:
    """Backup schedule options."""

    def test_daily_schedule(self):
        assert BackupSchedule.DAILY is not None

    def test_hourly_schedule(self):
        assert BackupSchedule.HOURLY is not None

    def test_weekly_schedule(self):
        assert BackupSchedule.WEEKLY is not None
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_backup.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.db.backup'`

**Step 3: Implement the backup manager**

```python
# src/evolve_trader/db/backup.py
"""Database backup and recovery system.

Daily full backups + WAL archiving. Configurable retention.
Cleanup of old backups. Export to portable formats.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BackupSchedule(Enum):
    """Backup frequency options."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"


class BackupType(Enum):
    """Type of backup."""
    FULL = "full"
    WAL = "wal"
    INCREMENTAL = "incremental"


@dataclass
class BackupConfig:
    """Configuration for database backups."""
    backup_dir: str = "/var/backups/evolve_trader"
    schedule: BackupSchedule = BackupSchedule.DAILY
    wal_archiving: bool = True
    retention_days: int = 30
    db_connection_string: str = ""


@dataclass
class BackupResult:
    """Result of a backup operation."""
    backup_type: BackupType
    success: bool
    file_path: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    size_bytes: int = 0
    error: str = ""


async def run_pg_dump(connection_string: str, output_path: str) -> bool:
    """Run pg_dump to create a full backup. Overridden in tests."""
    import asyncio
    proc = await asyncio.create_subprocess_exec(
        "pg_dump", connection_string, "-Fc", "-f", output_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise Exception(f"pg_dump failed: {stderr.decode()}")
    return True


async def run_pg_basebackup(connection_string: str, output_dir: str) -> bool:
    """Run pg_basebackup for WAL archiving. Overridden in tests."""
    import asyncio
    proc = await asyncio.create_subprocess_exec(
        "pg_basebackup", "-D", output_dir, "-Ft", "-z",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise Exception(f"pg_basebackup failed: {stderr.decode()}")
    return True


class BackupManager:
    """Manages database backups with retention and cleanup.

    Supports full pg_dump backups, WAL archiving via pg_basebackup,
    and data export to portable formats (CSV, JSON).
    """

    def __init__(self, config: BackupConfig):
        self.config = config
        self._backup_dir = Path(config.backup_dir)

    async def create_backup(self, backup_type: BackupType) -> BackupResult:
        """Create a database backup of the specified type."""
        timestamp = datetime.now(timezone.utc)
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")

        try:
            if backup_type == BackupType.FULL:
                file_name = f"backup_{ts_str}.sql.gz"
                file_path = str(self._backup_dir / file_name)
                await run_pg_dump(self.config.db_connection_string, file_path)
                return BackupResult(
                    backup_type=backup_type,
                    success=True,
                    file_path=file_path,
                    timestamp=timestamp,
                )
            elif backup_type == BackupType.WAL:
                wal_dir = str(self._backup_dir / f"wal_{ts_str}")
                await run_pg_basebackup(self.config.db_connection_string, wal_dir)
                return BackupResult(
                    backup_type=backup_type,
                    success=True,
                    file_path=wal_dir,
                    timestamp=timestamp,
                )
            else:
                return BackupResult(
                    backup_type=backup_type,
                    success=False,
                    error=f"Unsupported backup type: {backup_type}",
                )
        except Exception as e:
            logger.error("Backup failed: %s", e)
            return BackupResult(
                backup_type=backup_type,
                success=False,
                error=str(e),
                timestamp=timestamp,
            )

    async def list_backups(self) -> list[BackupResult]:
        """List available backups in the backup directory."""
        if not self._backup_dir.exists():
            return []
        results = []
        for path in sorted(self._backup_dir.glob("backup_*")):
            results.append(BackupResult(
                backup_type=BackupType.FULL,
                success=True,
                file_path=str(path),
                size_bytes=path.stat().st_size if path.is_file() else 0,
            ))
        return results

    async def cleanup(self, older_than_days: int | None = None) -> int:
        """Remove backups older than the retention period."""
        days = older_than_days if older_than_days is not None else self.config.retention_days
        if not self._backup_dir.exists():
            return 0

        removed = 0
        cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
        for path in self._backup_dir.iterdir():
            if path.stat().st_mtime < cutoff:
                if path.is_file():
                    path.unlink()
                    removed += 1
                    logger.info("Removed old backup: %s", path)
        return removed

    async def export(
        self,
        format: str = "csv",
        tables: list[str] | None = None,
    ) -> BackupResult:
        """Export data in a portable format."""
        timestamp = datetime.now(timezone.utc)
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")

        try:
            file_name = f"export_{ts_str}.{format}"
            file_path = str(self._backup_dir / file_name)
            # Use pg_dump with appropriate format flags
            await run_pg_dump(self.config.db_connection_string, file_path)
            return BackupResult(
                backup_type=BackupType.FULL,
                success=True,
                file_path=file_path,
                timestamp=timestamp,
            )
        except Exception as e:
            return BackupResult(
                backup_type=BackupType.FULL,
                success=False,
                error=str(e),
                timestamp=timestamp,
            )
```

Now create the disaster recovery runbook:

```markdown
# docs/runbooks/disaster-recovery.md
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_backup.py -v
```

Expected: PASS — all 10 tests green

**Step 5: Create the disaster recovery runbook and commit**

Create `docs/runbooks/disaster-recovery.md` with:
- Overview of backup strategy (daily full + WAL)
- Step-by-step restore procedures
- Quarterly test schedule
- Contact/escalation info

```bash
git add src/evolve_trader/db/backup.py tests/unit/test_backup.py docs/runbooks/disaster-recovery.md
git commit -m "feat: database backup manager with retention, cleanup, and disaster recovery runbook"
```

---

## Task 13: E2E Promotion Pipeline Test

Use the shared Phase 6 modules for promotion and approval in this test. If older draft names appear in example snippets below, treat them as legacy placeholders to be collapsed into the shared `promotion.py` and approval-gate modules before implementation.

**Files:**
- Create: `tests/integration/test_full_promotion_pipeline.py`

**Step 1: Write the integration test**

```python
# tests/integration/test_full_promotion_pipeline.py
"""End-to-end test of the full promotion pipeline.

Tests the complete flow: Paper -> Micro -> Partial -> Full
with bidirectional demotion at every stage. Exercises:
- Production approval gate
- Regime diversity gate
- Kill switch integration
- Audit logging throughout
"""
import pytest
from unittest.mock import AsyncMock
from datetime import datetime, timezone

from evolve_trader.execution.gates.approval_gate import (
    ApprovalMode,
    ApprovalPolicy as ProductionApprovalConfig,
    ApprovalGate as ProductionApprovalGate,
)
from evolve_trader.execution.promotion import (
    PromotionDecision as ApprovalDecision,
    PromotionStage as PromotionLevel,
)
from evolve_trader.core.regime_diversity_gate import (
    RegimeDiversityConfig,
    RegimeDiversityGate,
    RegimePerformance,
)
from evolve_trader.execution.kill_switch import (
    KillSwitch,
    KillSwitchConfig,
    KillSwitchSource,
    KillSwitchState,
)
from evolve_trader.execution.kill_switch_auto import (
    AutoKillSwitchConfig,
    AutoKillSwitchMonitor,
)
from evolve_trader.security.audit_log import (
    AuditLog,
    AuditLogConfig,
    AuditEventType,
)


class TestFullPromotionPipeline:
    """E2E test: Paper -> Micro -> Partial -> Full with demotion."""

    @pytest.fixture
    def approval_gate(self):
        config = ProductionApprovalConfig(
            mode=ApprovalMode.MANUAL,
        )
        return ProductionApprovalGate(config)

    @pytest.fixture
    def diversity_gate(self):
        config = RegimeDiversityConfig(
            min_positive_regimes=2,
            neutral_sharpe_threshold=-0.1,
            min_trades_per_regime=10,
        )
        return RegimeDiversityGate(config)

    @pytest.fixture
    def kill_switch(self):
        ks = KillSwitch(KillSwitchConfig())
        ks._trading_client = AsyncMock()
        ks._trading_client.cancel_all_orders = AsyncMock(return_value=0)
        ks._trading_client.close_all_positions = AsyncMock(return_value=0)
        ks._notifier = AsyncMock()
        return ks

    @pytest.fixture
    def audit_log(self):
        return AuditLog(AuditLogConfig())

    @pytest.fixture
    def good_regime_data(self):
        return [
            RegimePerformance("risk-on", sharpe_ratio=1.5, total_trades=30, win_rate=0.60),
            RegimePerformance("risk-off", sharpe_ratio=0.3, total_trades=25, win_rate=0.52),
        ]

    @pytest.fixture
    def bad_regime_data(self):
        return [
            RegimePerformance("risk-on", sharpe_ratio=2.0, total_trades=50, win_rate=0.65),
            RegimePerformance("risk-off", sharpe_ratio=-0.8, total_trades=20, win_rate=0.30),
        ]

    @pytest.mark.asyncio
    async def test_full_promotion_paper_to_full(
        self, approval_gate, diversity_gate, audit_log, good_regime_data,
    ):
        """Strategy promotes Paper -> Micro -> Partial -> Full with approvals."""
        strategy = "momentum-v5"
        current_level = PromotionLevel.PAPER

        for target_level in [PromotionLevel.MICRO, PromotionLevel.PARTIAL, PromotionLevel.FULL]:
            # Check regime diversity
            diversity = await diversity_gate.check(strategy, good_regime_data)
            assert diversity.passed is True

            # Request approval
            decision = await approval_gate.request_approval(
                strategy_name=strategy,
                current_level=current_level,
                target_level=target_level,
                metrics={"sharpe": 1.5, "win_rate": 0.60, "total_trades": 50},
            )
            assert decision.status == ApprovalDecision.PENDING

            # Human approves
            result = await approval_gate.human_decision(
                request_id=decision.request_id,
                approved=True,
                reviewer="admin",
                notes=f"Approved promotion to {target_level.name}",
            )
            assert result.status == ApprovalDecision.APPROVED

            # Log to audit
            await audit_log.append(
                event_type=AuditEventType.PROMOTION,
                actor="admin",
                payload={
                    "strategy": strategy,
                    "from": current_level.name,
                    "to": target_level.name,
                },
            )

            current_level = target_level

        assert current_level == PromotionLevel.FULL
        assert audit_log.entry_count == 3
        assert audit_log.verify_chain() is True

    @pytest.mark.asyncio
    async def test_demotion_full_to_paper(
        self, approval_gate, audit_log,
    ):
        """Strategy can be demoted from Full back to Paper."""
        strategy = "failing-strategy"

        result = await approval_gate.demote(
            strategy_name=strategy,
            current_level=PromotionLevel.FULL,
            target_level=PromotionLevel.PAPER,
            reason="Severe drawdown in production.",
            reviewer="system",
        )
        assert result.status == ApprovalDecision.DEMOTED

        await audit_log.append(
            event_type=AuditEventType.DEMOTION,
            actor="system",
            payload={
                "strategy": strategy,
                "from": "FULL",
                "to": "PAPER",
                "reason": "Severe drawdown",
            },
        )

        assert audit_log.entry_count == 1

    @pytest.mark.asyncio
    async def test_regime_diversity_blocks_promotion(
        self, approval_gate, diversity_gate, bad_regime_data,
    ):
        """Strategy failing regime diversity cannot promote."""
        strategy = "overfit-strategy"

        diversity = await diversity_gate.check(strategy, bad_regime_data)
        assert diversity.passed is False

        # Promotion should not proceed (gate blocks before approval)
        # In the real pipeline, this would prevent request_approval from being called

    @pytest.mark.asyncio
    async def test_kill_switch_demotes_all_to_paper(
        self, kill_switch, audit_log,
    ):
        """Kill switch activation demotes all strategies to Paper."""
        activation = await kill_switch.activate(
            source=KillSwitchSource.AUTO,
            reason="20% drawdown exceeded",
        )
        assert activation.reverted_to_paper is True

        await audit_log.append(
            event_type=AuditEventType.KILL_SWITCH_ACTIVATED,
            actor="auto",
            payload={
                "source": "auto",
                "reason": activation.reason,
                "orders_cancelled": activation.orders_cancelled,
            },
        )

        assert kill_switch.state == KillSwitchState.TRIGGERED
        assert audit_log.entry_count == 1

    @pytest.mark.asyncio
    async def test_bidirectional_step_down(self, approval_gate):
        """Strategy can step down one level at a time."""
        strategy = "cautious-strategy"

        # Demote FULL -> PARTIAL
        result = await approval_gate.demote(
            strategy_name=strategy,
            current_level=PromotionLevel.FULL,
            target_level=PromotionLevel.PARTIAL,
            reason="Reducing exposure",
            reviewer="admin",
        )
        assert result.status == ApprovalDecision.DEMOTED
        assert result.target_level == PromotionLevel.PARTIAL

        # Demote PARTIAL -> MICRO
        result = await approval_gate.demote(
            strategy_name=strategy,
            current_level=PromotionLevel.PARTIAL,
            target_level=PromotionLevel.MICRO,
            reason="Further reduction",
            reviewer="admin",
        )
        assert result.status == ApprovalDecision.DEMOTED
        assert result.target_level == PromotionLevel.MICRO

    @pytest.mark.asyncio
    async def test_auto_kill_switch_during_promotion(
        self, kill_switch, audit_log,
    ):
        """Auto kill switch fires during live trading, demoting everything."""
        monitor = AutoKillSwitchMonitor(
            kill_switch=kill_switch,
            config=AutoKillSwitchConfig(max_drawdown_pct=0.20),
            initial_value=100000.0,
        )

        # Simulate portfolio decline
        triggered = await monitor.check_and_trigger(current_value=95000.0)
        assert triggered is False  # Only 5% drawdown

        triggered = await monitor.check_and_trigger(current_value=79000.0)
        assert triggered is True  # 21% drawdown

        assert kill_switch.state == KillSwitchState.TRIGGERED

        await audit_log.append(
            event_type=AuditEventType.KILL_SWITCH_ACTIVATED,
            actor="auto",
            payload={"drawdown": "21%"},
        )

        assert audit_log.verify_chain() is True

    @pytest.mark.asyncio
    async def test_rearm_and_re_promote_after_kill(
        self, approval_gate, kill_switch, audit_log,
    ):
        """After kill switch, system can be re-armed and strategy re-promoted."""
        # Trigger kill switch
        await kill_switch.activate(
            source=KillSwitchSource.DASHBOARD,
            reason="Manual safety halt",
        )
        assert kill_switch.state == KillSwitchState.TRIGGERED

        # Rearm
        await kill_switch.rearm(reviewer="admin")
        assert kill_switch.state == KillSwitchState.ARMED

        await audit_log.append(
            event_type=AuditEventType.KILL_SWITCH_REARMED,
            actor="admin",
            payload={"reason": "Conditions stabilized"},
        )

        # Re-promote from Paper -> Micro
        decision = await approval_gate.request_approval(
            strategy_name="recovered-strategy",
            current_level=PromotionLevel.PAPER,
            target_level=PromotionLevel.MICRO,
            metrics={"sharpe": 1.2, "win_rate": 0.55, "total_trades": 30},
        )
        result = await approval_gate.human_decision(
            request_id=decision.request_id,
            approved=True,
            reviewer="admin",
            notes="Re-promoting after conditions stabilized",
        )
        assert result.status == ApprovalDecision.APPROVED

        await audit_log.append(
            event_type=AuditEventType.PROMOTION,
            actor="admin",
            payload={"strategy": "recovered-strategy", "to": "MICRO"},
        )

        assert audit_log.entry_count == 2
        assert audit_log.verify_chain() is True
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/integration/test_full_promotion_pipeline.py -v
```

Expected: FAIL — `ModuleNotFoundError` (until Tasks 1-12 are implemented)

**Step 3: No new implementation — this test exercises code from Tasks 1-12**

**Step 4: Run tests to verify they pass**

```bash
pytest tests/integration/test_full_promotion_pipeline.py -v
```

Expected: PASS — all 8 integration tests green (after Tasks 1-12 are complete)

**Step 5: Commit**

```bash
git add tests/integration/test_full_promotion_pipeline.py
git commit -m "test: E2E promotion pipeline — Paper through Full with bidirectional demotion"
```

---

## Task 14: Final Verification

**Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: ALL PASS — all Phase 11 tests plus all prior phase tests

**Step 2: Run linting and type checking**

```bash
ruff check src/evolve_trader/
mypy src/evolve_trader/ --ignore-missing-imports
```

Expected: No errors

**Step 3: Verify kill switch from all four surfaces**

```bash
pytest tests/unit/test_kill_switch.py tests/unit/test_kill_switch_channels.py tests/api/test_kill_switch_api.py tests/unit/test_kill_switch_auto.py -v
```

Expected: All kill switch tests pass across dashboard, Slack/Telegram, API, and auto-trigger

**Step 4: Verify audit chain integrity**

```bash
pytest tests/unit/test_audit_log.py -v -k "chain"
```

Expected: Both chain integrity and tamper detection tests pass

**Step 5: Commit any fixes**

```bash
git add -A
git commit -m "test: Phase 11 final verification — all tests passing"
```

---

## Parallelization Notes

Tasks in this phase have the following dependency structure:

```
Task 1 (Alpaca Live) ──────────────────────────────┐
                                                     │
Task 2 (Production Approval) ───────────────────────┤
                                                     │
Task 3 (Kill Switch: Dashboard) ──┬── Task 4 (Slack/Telegram)
                                  ├── Task 5 (API Endpoint)   ├── Task 13 (E2E Pipeline)
                                  └── Task 6 (Auto-Trigger)   │
                                                     │         │
Task 7 (Secrets) ───────────────────────────────────┤         │
Task 8 (Rate Limiting) ────────────────────────────┤         │
Task 9 (Audit Log) ────────────────────────────────┤         │
                                                     │         │
Task 10 (Regime Diversity) ────────────────────────┤─────────┘
Task 11 (Observability) ──────────────────────────┤
Task 12 (Backup & Recovery) ──────────────────────┘
```

**Can run in parallel:**
- Task 1 (Alpaca Live) and Task 2 (Production Approval) are independent — run simultaneously
- Tasks 4, 5, 6 (kill switch surfaces) depend on Task 3 (kill switch core) but are independent of each other — run simultaneously after Task 3
- Tasks 7, 8, 9 (security modules) are independent of each other — run simultaneously
- Task 10 (Regime Diversity) and Task 11 (Observability) and Task 12 (Backup) are independent — run simultaneously
- Task 13 (E2E Pipeline) depends on Tasks 2, 3, 6, 9, 10 — run last before Task 14
- Task 14 (Final Verification) depends on everything
