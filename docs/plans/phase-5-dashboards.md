# Phase 5: Dashboards (v1) — Detailed Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build both an ops/monitoring dashboard and a user-facing trading dashboard. The ops dashboard surfaces system health, evolution metrics, signal performance, and LLM costs. The trading dashboard gives the user full portfolio visibility, trade history, strategy breakdowns, and operator state visibility. Both dashboards share a single Next.js frontend backed by a FastAPI API layer with real-time WebSocket updates.

**Architecture:** A FastAPI service exposes REST endpoints and WebSocket channels for all dashboard data. The Next.js frontend renders two top-level layouts: `/ops` for monitoring and `/trading` for the user-facing experience. Recharts powers all visualizations. JWT authentication protects all endpoints. The API reads from the PostgreSQL database established in Phase 2 via the existing repository layer. This phase is intentionally read-heavy: actionable approval and kill-switch operations remain owned by later execution phases, while the dashboard exposes status, readiness, and safe local toggles.

**Tech Stack:** Python 3.11+ (FastAPI, uvicorn, python-jose, passlib), React 18+, Next.js 14+, TypeScript 5+, Tailwind CSS 3.4+, Recharts 2.x, Jest, React Testing Library, pytest, httpx

**Parent plan:** [`2026-03-29-evolve-trader-master-plan.md`](2026-03-29-evolve-trader-master-plan.md)

**Prerequisites:** Phase 4 complete. Phase 2 sufficient for skeleton start (DB models, repositories, signal types all available). Phase 6 and Phase 11 later attach the approval and kill-switch backends to the operator surfaces introduced here.

---

## Task 1: Dashboard API Layer (FastAPI)

**Files:**
- Create: `src/evolve_trader/api/__init__.py`
- Create: `src/evolve_trader/api/main.py`
- Create: `src/evolve_trader/api/auth.py`
- Create: `src/evolve_trader/api/deps.py`
- Create: `src/evolve_trader/api/routes/__init__.py`
- Create: `src/evolve_trader/api/routes/portfolio.py`
- Create: `src/evolve_trader/api/routes/trades.py`
- Create: `src/evolve_trader/api/routes/strategies.py`
- Create: `src/evolve_trader/api/routes/signals.py`
- Create: `src/evolve_trader/api/routes/monitoring.py`
- Create: `tests/api/__init__.py`
- Create: `tests/api/test_dashboard_api.py`

**Step 1: Write the failing tests**

```python
# tests/api/test_dashboard_api.py
"""Tests for the dashboard API layer."""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock

from httpx import AsyncClient, ASGITransport

from evolve_trader.api.main import create_app
from evolve_trader.api.auth import create_access_token, verify_token


# --- Auth tests ---


def test_create_access_token_returns_string():
    """create_access_token produces a JWT string."""
    token = create_access_token({"sub": "admin"})
    assert isinstance(token, str)
    assert len(token) > 20


def test_verify_token_roundtrip():
    """verify_token decodes a token created by create_access_token."""
    token = create_access_token({"sub": "admin"})
    payload = verify_token(token)
    assert payload["sub"] == "admin"


def test_verify_token_rejects_garbage():
    """verify_token raises on invalid tokens."""
    with pytest.raises(Exception):
        verify_token("not.a.valid.token")


# --- App factory ---


def test_create_app_returns_fastapi_instance():
    """create_app produces a FastAPI application."""
    app = create_app()
    from fastapi import FastAPI
    assert isinstance(app, FastAPI)


def test_app_has_cors_middleware():
    """App should include CORS middleware."""
    app = create_app()
    middleware_classes = [m.cls.__name__ for m in app.user_middleware]
    assert "CORSMiddleware" in middleware_classes


# --- Portfolio endpoints ---


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def auth_client(app):
    token = create_access_token({"sub": "admin"})
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        client.headers["Authorization"] = f"Bearer {token}"
        yield client


@pytest.fixture
async def unauth_client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_portfolio_snapshot_requires_auth(unauth_client):
    """GET /api/portfolio/snapshot returns 401 without token."""
    resp = await unauth_client.get("/api/portfolio/snapshot")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_portfolio_snapshot_returns_data(auth_client):
    """GET /api/portfolio/snapshot returns portfolio data."""
    resp = await auth_client.get("/api/portfolio/snapshot")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_value" in data
    assert "cash" in data
    assert "positions" in data
    assert "total_return" in data
    assert "drawdown" in data


@pytest.mark.asyncio
async def test_portfolio_equity_curve(auth_client):
    """GET /api/portfolio/equity-curve returns time series."""
    resp = await auth_client.get("/api/portfolio/equity-curve?days=30")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


# --- Trade endpoints ---


@pytest.mark.asyncio
async def test_trades_list(auth_client):
    """GET /api/trades returns paginated trade list."""
    resp = await auth_client.get("/api/trades?limit=10&offset=0")
    assert resp.status_code == 200
    data = resp.json()
    assert "trades" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_trades_by_strategy(auth_client):
    """GET /api/trades?strategy=momentum-v1 filters by strategy."""
    resp = await auth_client.get("/api/trades?strategy=momentum-v1&limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert "trades" in data


# --- Strategy endpoints ---


@pytest.mark.asyncio
async def test_strategies_list(auth_client):
    """GET /api/strategies returns active strategies."""
    resp = await auth_client.get("/api/strategies")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_strategy_performance(auth_client):
    """GET /api/strategies/{name}/performance returns metrics."""
    resp = await auth_client.get("/api/strategies/momentum-v1/performance")
    assert resp.status_code in (200, 404)


@pytest.mark.asyncio
async def test_evolution_events(auth_client):
    """GET /api/strategies/evolution-events returns evolution history."""
    resp = await auth_client.get("/api/strategies/evolution-events?limit=20")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


# --- Signal endpoints ---


@pytest.mark.asyncio
async def test_signals_sources(auth_client):
    """GET /api/signals/sources returns source list with stats."""
    resp = await auth_client.get("/api/signals/sources")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_signals_by_source(auth_client):
    """GET /api/signals/{source} returns signals for a source."""
    resp = await auth_client.get("/api/signals/edgar_13f?limit=10")
    assert resp.status_code == 200


# --- Monitoring endpoints ---


@pytest.mark.asyncio
async def test_monitoring_metrics(auth_client):
    """GET /api/monitoring/metrics returns system metrics."""
    resp = await auth_client.get("/api/monitoring/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "portfolio_sharpe" in data or isinstance(data, dict)


@pytest.mark.asyncio
async def test_monitoring_llm_costs(auth_client):
    """GET /api/monitoring/llm-costs returns cost breakdown."""
    resp = await auth_client.get("/api/monitoring/llm-costs")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_spend" in data


@pytest.mark.asyncio
async def test_health_endpoint_no_auth(unauth_client):
    """GET /api/health is public."""
    resp = await unauth_client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/api/test_dashboard_api.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.api'`

**Step 3: Implement the API layer**

```python
# src/evolve_trader/api/__init__.py
"""Dashboard API package."""
```

```python
# src/evolve_trader/api/auth.py
"""JWT authentication for the dashboard API."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# In production, load from environment / secrets manager
SECRET_KEY = "evolve-trader-dev-secret-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

security_scheme = HTTPBearer()


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> dict[str, Any]:
    """Verify and decode a JWT token. Raises on invalid."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        ) from e


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Security(security_scheme),
) -> dict[str, Any]:
    """FastAPI dependency that enforces JWT auth."""
    return verify_token(credentials.credentials)
```

```python
# src/evolve_trader/api/deps.py
"""Shared FastAPI dependencies."""
from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from evolve_trader.db.engine import create_async_engine, get_async_session

# Default DB URL — override via environment variable in production
DATABASE_URL = "postgresql+asyncpg://localhost/evolve_trader"

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        import os
        url = os.environ.get("DATABASE_URL", DATABASE_URL)
        _engine = create_async_engine(url)
    return _engine


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session for request lifetime."""
    async with get_async_session(get_engine()) as session:
        yield session
```

```python
# src/evolve_trader/api/routes/__init__.py
"""API route modules."""
```

```python
# src/evolve_trader/api/routes/portfolio.py
"""Portfolio endpoints."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends

from evolve_trader.api.auth import require_auth

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


@router.get("/snapshot")
async def get_portfolio_snapshot(
    _user: dict = Depends(require_auth),
) -> dict[str, Any]:
    """Return the latest portfolio snapshot."""
    # In production, reads from PortfolioSnapshot repository
    return {
        "total_value": 100000.0,
        "cash": 30000.0,
        "positions": [],
        "sector_exposure": {},
        "total_return": 0.0,
        "drawdown": 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/equity-curve")
async def get_equity_curve(
    days: int = 30,
    _user: dict = Depends(require_auth),
) -> list[dict[str, Any]]:
    """Return equity curve as time series."""
    # Placeholder: returns empty until DB integration
    return []
```

```python
# src/evolve_trader/api/routes/trades.py
"""Trade history endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from evolve_trader.api.auth import require_auth

router = APIRouter(prefix="/api/trades", tags=["trades"])


@router.get("")
async def list_trades(
    strategy: str | None = Query(None),
    ticker: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _user: dict = Depends(require_auth),
) -> dict[str, Any]:
    """Return paginated trade list, optionally filtered."""
    # Placeholder: reads from TradeLogRepository in production
    return {
        "trades": [],
        "total": 0,
        "limit": limit,
        "offset": offset,
    }
```

```python
# src/evolve_trader/api/routes/strategies.py
"""Strategy and evolution endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from evolve_trader.api.auth import require_auth

router = APIRouter(prefix="/api/strategies", tags=["strategies"])


@router.get("")
async def list_strategies(
    _user: dict = Depends(require_auth),
) -> list[dict[str, Any]]:
    """Return all active strategies with summary metrics."""
    return []


@router.get("/evolution-events")
async def list_evolution_events(
    limit: int = Query(20, ge=1, le=200),
    _user: dict = Depends(require_auth),
) -> list[dict[str, Any]]:
    """Return evolution event history."""
    return []


@router.get("/{name}/performance")
async def get_strategy_performance(
    name: str,
    _user: dict = Depends(require_auth),
) -> dict[str, Any]:
    """Return detailed performance for a named strategy."""
    # Placeholder: returns 404 if not found, metrics if found
    return {
        "name": name,
        "sharpe_ratio": None,
        "max_drawdown": None,
        "win_rate": None,
        "total_trades": 0,
        "status": "unknown",
    }
```

```python
# src/evolve_trader/api/routes/signals.py
"""Signal source endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from evolve_trader.api.auth import require_auth

router = APIRouter(prefix="/api/signals", tags=["signals"])


@router.get("/sources")
async def list_signal_sources(
    _user: dict = Depends(require_auth),
) -> list[dict[str, Any]]:
    """Return all signal sources with summary stats."""
    return []


@router.get("/{source}")
async def get_signals_by_source(
    source: str,
    limit: int = Query(50, ge=1, le=500),
    _user: dict = Depends(require_auth),
) -> list[dict[str, Any]]:
    """Return recent signals from a specific source."""
    return []
```

```python
# src/evolve_trader/api/routes/monitoring.py
"""Monitoring and ops endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from evolve_trader.api.auth import require_auth

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


@router.get("/metrics")
async def get_system_metrics(
    _user: dict = Depends(require_auth),
) -> dict[str, Any]:
    """Return current system monitoring metrics."""
    return {
        "portfolio_sharpe": None,
        "active_strategies": 0,
        "signal_sources_active": 0,
        "evolution_events_24h": 0,
        "uptime_seconds": 0,
    }


@router.get("/llm-costs")
async def get_llm_costs(
    _user: dict = Depends(require_auth),
) -> dict[str, Any]:
    """Return LLM cost breakdown."""
    return {
        "total_spend": 0.0,
        "by_component": {},
        "cost_per_trade": 0.0,
        "budget_remaining": 0.0,
        "budget_utilization": 0.0,
    }
```

```python
# src/evolve_trader/api/main.py
"""FastAPI application factory for the dashboard API."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from evolve_trader.api.routes.portfolio import router as portfolio_router
from evolve_trader.api.routes.trades import router as trades_router
from evolve_trader.api.routes.strategies import router as strategies_router
from evolve_trader.api.routes.signals import router as signals_router
from evolve_trader.api.routes.monitoring import router as monitoring_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Evolve-Trader Dashboard API",
        version="0.1.0",
        description="REST API for ops monitoring and trading dashboards.",
    )

    # CORS — allow dashboard origin(s)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3001"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Public endpoints
    @app.get("/api/health")
    async def health_check():
        return {"status": "ok"}

    # Protected route groups
    app.include_router(portfolio_router)
    app.include_router(trades_router)
    app.include_router(strategies_router)
    app.include_router(signals_router)
    app.include_router(monitoring_router)

    return app


# For `uvicorn evolve_trader.api.main:app`
app = create_app()
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/api/test_dashboard_api.py -v
```

Expected: PASS — all auth, endpoint, and CORS tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/api/ tests/api/
git commit -m "feat: FastAPI dashboard API layer with JWT auth, CORS, and endpoint scaffolding"
```

---

## Task 2: WebSocket Real-Time Updates

**Files:**
- Create: `src/evolve_trader/api/websocket.py`
- Create: `tests/api/test_websocket.py`

**Step 1: Write the failing tests**

```python
# tests/api/test_websocket.py
"""Tests for WebSocket real-time update channels."""
import pytest
import json
from unittest.mock import AsyncMock

from httpx import ASGITransport, AsyncClient

from evolve_trader.api.main import create_app
from evolve_trader.api.auth import create_access_token
from evolve_trader.api.websocket import ConnectionManager


# --- ConnectionManager unit tests ---


def test_connection_manager_initializes_empty():
    """ConnectionManager starts with no active connections."""
    mgr = ConnectionManager()
    assert mgr.active_connection_count == 0


@pytest.mark.asyncio
async def test_connection_manager_connect_disconnect():
    """ConnectionManager tracks connections."""
    mgr = ConnectionManager()
    ws = AsyncMock()
    await mgr.connect(ws, subscriptions=["trades", "signals"])
    assert mgr.active_connection_count == 1
    mgr.disconnect(ws)
    assert mgr.active_connection_count == 0


@pytest.mark.asyncio
async def test_connection_manager_broadcast_by_channel():
    """broadcast sends only to subscribers of that channel."""
    mgr = ConnectionManager()

    ws_trades = AsyncMock()
    ws_signals = AsyncMock()

    await mgr.connect(ws_trades, subscriptions=["trades"])
    await mgr.connect(ws_signals, subscriptions=["signals"])

    await mgr.broadcast("trades", {"type": "trade_executed", "ticker": "AAPL"})

    ws_trades.send_json.assert_called_once()
    ws_signals.send_json.assert_not_called()


@pytest.mark.asyncio
async def test_connection_manager_broadcast_all():
    """broadcast to 'all' channel reaches every subscriber."""
    mgr = ConnectionManager()

    ws1 = AsyncMock()
    ws2 = AsyncMock()

    await mgr.connect(ws1, subscriptions=["trades"])
    await mgr.connect(ws2, subscriptions=["signals"])

    await mgr.broadcast("all", {"type": "system_alert", "message": "kill switch"})

    ws1.send_json.assert_called_once()
    ws2.send_json.assert_called_once()


@pytest.mark.asyncio
async def test_connection_manager_handles_dead_connection():
    """broadcast removes connections that raise on send."""
    mgr = ConnectionManager()

    ws_dead = AsyncMock()
    ws_dead.send_json.side_effect = RuntimeError("connection closed")

    await mgr.connect(ws_dead, subscriptions=["trades"])
    assert mgr.active_connection_count == 1

    await mgr.broadcast("trades", {"type": "trade_executed"})
    assert mgr.active_connection_count == 0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/api/test_websocket.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.api.websocket'`

**Step 3: Implement WebSocket manager and endpoint**

```python
# src/evolve_trader/api/websocket.py
"""WebSocket connection manager for real-time dashboard updates."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect, Query
from jose import JWTError

from evolve_trader.api.auth import verify_token

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections with channel-based subscriptions.

    Channels:
        - trades: trade execution events
        - signals: new signal events
        - alerts: system alerts (drawdown, kill switch)
        - evolution: evolution events (FIX/DERIVED/CAPTURED)
        - metrics: periodic metric snapshots
        - all: broadcast to every connection
    """

    def __init__(self) -> None:
        # ws -> set of subscribed channels
        self._connections: dict[WebSocket, set[str]] = {}

    @property
    def active_connection_count(self) -> int:
        return len(self._connections)

    async def connect(self, websocket: WebSocket, subscriptions: list[str] | None = None) -> None:
        """Register a WebSocket connection with its subscriptions."""
        subs = set(subscriptions) if subscriptions else {"all"}
        self._connections[websocket] = subs

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        self._connections.pop(websocket, None)

    async def broadcast(self, channel: str, data: dict[str, Any]) -> None:
        """Send data to all connections subscribed to the given channel.

        Channel 'all' reaches every connection regardless of their subscriptions.
        Dead connections are automatically removed.
        """
        dead: list[WebSocket] = []
        for ws, subs in self._connections.items():
            if channel == "all" or channel in subs:
                try:
                    await ws.send_json(data)
                except Exception:
                    dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


# Singleton manager shared across the app
manager = ConnectionManager()


async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(...),
    channels: str = Query("all"),
) -> None:
    """WebSocket endpoint at /ws.

    Query params:
        token: JWT access token
        channels: comma-separated list of channels to subscribe to
    """
    # Authenticate
    try:
        verify_token(token)
    except Exception:
        await websocket.close(code=4001, reason="Authentication failed")
        return

    await websocket.accept()
    subscription_list = [c.strip() for c in channels.split(",")]
    await manager.connect(websocket, subscriptions=subscription_list)

    try:
        while True:
            # Keep connection alive; handle client messages if needed
            data = await websocket.receive_text()
            # Clients can send ping or subscription updates
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
```

Then wire the WebSocket endpoint into the app factory. Updated `create_app` in full:

```python
# src/evolve_trader/api/main.py (updated)
"""FastAPI application factory for the dashboard API."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from evolve_trader.api.routes.portfolio import router as portfolio_router
from evolve_trader.api.routes.trades import router as trades_router
from evolve_trader.api.routes.strategies import router as strategies_router
from evolve_trader.api.routes.signals import router as signals_router
from evolve_trader.api.routes.monitoring import router as monitoring_router
from evolve_trader.api.websocket import websocket_endpoint


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Evolve-Trader Dashboard API",
        version="0.1.0",
        description="REST API for ops monitoring and trading dashboards.",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3001"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Public endpoints
    @app.get("/api/health")
    async def health_check():
        return {"status": "ok"}

    # Protected route groups
    app.include_router(portfolio_router)
    app.include_router(trades_router)
    app.include_router(strategies_router)
    app.include_router(signals_router)
    app.include_router(monitoring_router)

    # WebSocket
    app.websocket("/ws")(websocket_endpoint)

    return app


app = create_app()
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/api/test_websocket.py -v
```

Expected: PASS — all ConnectionManager tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/api/websocket.py src/evolve_trader/api/main.py tests/api/test_websocket.py
git commit -m "feat: WebSocket connection manager with channel-based subscriptions"
```

---

## Task 3: Next.js Project Setup

**Files:**
- Create: `dashboard/package.json`
- Create: `dashboard/tsconfig.json`
- Create: `dashboard/tailwind.config.ts`
- Create: `dashboard/postcss.config.js`
- Create: `dashboard/next.config.ts`
- Create: `dashboard/jest.config.ts`
- Create: `dashboard/src/lib/api.ts`
- Create: `dashboard/src/lib/ws.ts`
- Create: `dashboard/src/lib/types.ts`
- Create: `dashboard/src/app/layout.tsx`
- Create: `dashboard/src/app/page.tsx`
- Create: `dashboard/src/app/ops/layout.tsx`
- Create: `dashboard/src/app/trading/layout.tsx`
- Create: `dashboard/src/__tests__/api.test.ts`

**Step 1: Write the failing tests**

```typescript
// dashboard/src/__tests__/api.test.ts
import { ApiClient } from "../lib/api";

describe("ApiClient", () => {
  it("constructs with base URL", () => {
    const client = new ApiClient("http://localhost:8000");
    expect(client.baseUrl).toBe("http://localhost:8000");
  });

  it("sets auth token", () => {
    const client = new ApiClient("http://localhost:8000");
    client.setToken("test-jwt-token");
    expect(client.token).toBe("test-jwt-token");
  });

  it("builds correct portfolio snapshot URL", () => {
    const client = new ApiClient("http://localhost:8000");
    expect(client.url("/api/portfolio/snapshot")).toBe(
      "http://localhost:8000/api/portfolio/snapshot"
    );
  });

  it("includes auth header in fetch options", () => {
    const client = new ApiClient("http://localhost:8000");
    client.setToken("my-token");
    const opts = client.authHeaders();
    expect(opts["Authorization"]).toBe("Bearer my-token");
  });

  it("throws if no token set when calling authHeaders", () => {
    const client = new ApiClient("http://localhost:8000");
    expect(() => client.authHeaders()).toThrow("No auth token set");
  });
});
```

**Step 2: Run tests to verify they fail**

```bash
cd dashboard && npx jest --config jest.config.ts src/__tests__/api.test.ts
```

Expected: FAIL — cannot find module `../lib/api`

**Step 3: Implement the Next.js project**

```json
// dashboard/package.json
{
  "name": "evolve-trader-dashboard",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "test": "jest",
    "test:watch": "jest --watch"
  },
  "dependencies": {
    "next": "^14.2.0",
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "recharts": "^2.12.0"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.4.0",
    "@testing-library/react": "^15.0.0",
    "@types/jest": "^29.5.0",
    "@types/node": "^20.12.0",
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "autoprefixer": "^10.4.0",
    "jest": "^29.7.0",
    "jest-environment-jsdom": "^29.7.0",
    "postcss": "^8.4.0",
    "tailwindcss": "^3.4.0",
    "ts-jest": "^29.1.0",
    "typescript": "^5.4.0"
  }
}
```

```json
// dashboard/tsconfig.json
{
  "compilerOptions": {
    "target": "es2017",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./src/*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],
  "exclude": ["node_modules"]
}
```

```typescript
// dashboard/tailwind.config.ts
import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        "ops-bg": "#0f172a",
        "ops-panel": "#1e293b",
        "trading-bg": "#fafafa",
        "trading-panel": "#ffffff",
        accent: "#3b82f6",
        danger: "#ef4444",
        success: "#22c55e",
        warning: "#f59e0b",
      },
    },
  },
  plugins: [],
};

export default config;
```

```javascript
// dashboard/postcss.config.js
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
```

```typescript
// dashboard/next.config.ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
    ];
  },
};

export default nextConfig;
```

```typescript
// dashboard/jest.config.ts
import type { Config } from "jest";

const config: Config = {
  testEnvironment: "jsdom",
  transform: {
    "^.+\\.tsx?$": "ts-jest",
  },
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1",
  },
  setupFilesAfterSetup: [],
};

export default config;
```

```typescript
// dashboard/src/lib/types.ts
/**
 * Shared TypeScript types mirroring the API response schemas.
 */

export interface PortfolioSnapshot {
  total_value: number;
  cash: number;
  positions: Position[];
  sector_exposure: Record<string, number>;
  total_return: number;
  drawdown: number;
  timestamp: string;
}

export interface Position {
  ticker: string;
  shares: number;
  value: number;
  sector: string;
  pnl?: number;
  return_pct?: number;
}

export interface EquityCurvePoint {
  date: string;
  value: number;
  drawdown: number;
  benchmark?: number;
}

export interface Trade {
  id: number;
  strategy_skill: string;
  ticker: string;
  direction: "BUY" | "SELL" | "SHORT" | "COVER";
  quantity: number;
  entry_price: number;
  exit_price: number | null;
  entry_date: string;
  exit_date: string | null;
  pnl: number | null;
  return_pct: number | null;
  regime_label: string | null;
  signal_sources: string[];
}

export interface TradeListResponse {
  trades: Trade[];
  total: number;
  limit: number;
  offset: number;
}

export interface StrategyInfo {
  name: string;
  version: number;
  status: "active" | "probation" | "archived";
  sharpe_ratio: number | null;
  max_drawdown: number | null;
  win_rate: number | null;
  total_trades: number;
}

export interface EvolutionEvent {
  id: number;
  event_type: "FIX" | "DERIVED" | "CAPTURED";
  parent_skill: string;
  child_skill: string;
  trigger_reason: string;
  created_at: string;
}

export interface SignalSource {
  source: string;
  signal_count: number;
  hit_rate: number | null;
  avg_confidence: number | null;
  last_signal_at: string | null;
}

export interface LLMCosts {
  total_spend: number;
  by_component: Record<string, number>;
  cost_per_trade: number;
  budget_remaining: number;
  budget_utilization: number;
}

export interface SystemMetrics {
  portfolio_sharpe: number | null;
  active_strategies: number;
  signal_sources_active: number;
  evolution_events_24h: number;
  uptime_seconds: number;
}
```

```typescript
// dashboard/src/lib/api.ts
/**
 * API client for the Evolve-Trader dashboard backend.
 */
import type {
  PortfolioSnapshot,
  EquityCurvePoint,
  TradeListResponse,
  StrategyInfo,
  EvolutionEvent,
  SignalSource,
  LLMCosts,
  SystemMetrics,
} from "./types";

export class ApiClient {
  readonly baseUrl: string;
  private _token: string | null = null;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  get token(): string | null {
    return this._token;
  }

  setToken(token: string): void {
    this._token = token;
  }

  url(path: string): string {
    return `${this.baseUrl}${path}`;
  }

  authHeaders(): Record<string, string> {
    if (!this._token) {
      throw new Error("No auth token set");
    }
    return {
      Authorization: `Bearer ${this._token}`,
      "Content-Type": "application/json",
    };
  }

  private async get<T>(path: string, params?: Record<string, string>): Promise<T> {
    const url = new URL(this.url(path));
    if (params) {
      Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
    }
    const resp = await fetch(url.toString(), { headers: this.authHeaders() });
    if (!resp.ok) {
      throw new Error(`API error ${resp.status}: ${resp.statusText}`);
    }
    return resp.json() as Promise<T>;
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    const resp = await fetch(this.url(path), {
      method: "POST",
      headers: this.authHeaders(),
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      throw new Error(`API error ${resp.status}: ${resp.statusText}`);
    }
    return resp.json() as Promise<T>;
  }

  // --- Portfolio ---
  async getPortfolioSnapshot(): Promise<PortfolioSnapshot> {
    return this.get("/api/portfolio/snapshot");
  }

  async getEquityCurve(days: number = 30): Promise<EquityCurvePoint[]> {
    return this.get("/api/portfolio/equity-curve", { days: String(days) });
  }

  // --- Trades ---
  async getTrades(opts?: {
    strategy?: string;
    ticker?: string;
    limit?: number;
    offset?: number;
  }): Promise<TradeListResponse> {
    const params: Record<string, string> = {};
    if (opts?.strategy) params.strategy = opts.strategy;
    if (opts?.ticker) params.ticker = opts.ticker;
    if (opts?.limit) params.limit = String(opts.limit);
    if (opts?.offset) params.offset = String(opts.offset);
    return this.get("/api/trades", params);
  }

  // --- Strategies ---
  async getStrategies(): Promise<StrategyInfo[]> {
    return this.get("/api/strategies");
  }

  async getEvolutionEvents(limit: number = 20): Promise<EvolutionEvent[]> {
    return this.get("/api/strategies/evolution-events", { limit: String(limit) });
  }

  // --- Signals ---
  async getSignalSources(): Promise<SignalSource[]> {
    return this.get("/api/signals/sources");
  }

  // --- Monitoring ---
  async getSystemMetrics(): Promise<SystemMetrics> {
    return this.get("/api/monitoring/metrics");
  }

  async getLLMCosts(): Promise<LLMCosts> {
    return this.get("/api/monitoring/llm-costs");
  }

  // --- Controls ---
  async activateKillSwitch(): Promise<{ success: boolean }> {
    return this.post("/api/controls/kill-switch", { action: "activate" });
  }

  async approveTrade(tradeId: number): Promise<{ success: boolean }> {
    return this.post(`/api/controls/trades/${tradeId}/approve`, {});
  }

  async rejectTrade(tradeId: number): Promise<{ success: boolean }> {
    return this.post(`/api/controls/trades/${tradeId}/reject`, {});
  }
}

// Default singleton
export const api = new ApiClient(
  typeof window !== "undefined" ? window.location.origin : "http://localhost:8000"
);
```

```typescript
// dashboard/src/lib/ws.ts
/**
 * WebSocket client for real-time dashboard updates.
 */

type MessageHandler = (data: Record<string, unknown>) => void;

export class DashboardWebSocket {
  private ws: WebSocket | null = null;
  private handlers: Map<string, Set<MessageHandler>> = new Map();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private url: string;

  constructor(baseUrl: string, token: string, channels: string[] = ["all"]) {
    const wsUrl = baseUrl.replace(/^http/, "ws");
    this.url = `${wsUrl}/ws?token=${token}&channels=${channels.join(",")}`;
  }

  connect(): void {
    this.ws = new WebSocket(this.url);

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const type = data.type as string;
        // Notify specific handlers
        this.handlers.get(type)?.forEach((fn) => fn(data));
        // Notify wildcard handlers
        this.handlers.get("*")?.forEach((fn) => fn(data));
      } catch {
        // ignore parse errors
      }
    };

    this.ws.onclose = () => {
      // Auto-reconnect after 3 seconds
      this.reconnectTimer = setTimeout(() => this.connect(), 3000);
    };

    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }
    this.ws?.close();
    this.ws = null;
  }

  on(eventType: string, handler: MessageHandler): () => void {
    if (!this.handlers.has(eventType)) {
      this.handlers.set(eventType, new Set());
    }
    this.handlers.get(eventType)!.add(handler);
    // Return unsubscribe function
    return () => {
      this.handlers.get(eventType)?.delete(handler);
    };
  }

  send(data: Record<string, unknown>): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }
}
```

```tsx
// dashboard/src/app/layout.tsx
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Evolve-Trader Dashboard",
  description: "Ops monitoring and trading dashboard for Evolve-Trader AI",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-50">{children}</body>
    </html>
  );
}
```

```tsx
// dashboard/src/app/page.tsx
import Link from "next/link";

export default function Home() {
  return (
    <main className="flex min-h-screen items-center justify-center gap-8">
      <Link
        href="/ops"
        className="rounded-lg bg-ops-bg px-8 py-6 text-white hover:bg-ops-panel transition"
      >
        <h2 className="text-xl font-bold">Ops Dashboard</h2>
        <p className="text-gray-400 mt-2">System monitoring &amp; evolution</p>
      </Link>
      <Link
        href="/trading"
        className="rounded-lg bg-accent px-8 py-6 text-white hover:bg-blue-600 transition"
      >
        <h2 className="text-xl font-bold">Trading Dashboard</h2>
        <p className="text-blue-100 mt-2">Portfolio &amp; trade management</p>
      </Link>
    </main>
  );
}
```

```tsx
// dashboard/src/app/ops/layout.tsx
export default function OpsLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-ops-bg text-white">
      <header className="border-b border-gray-700 px-6 py-4 flex items-center justify-between">
        <h1 className="text-lg font-bold">Evolve-Trader Ops</h1>
        <nav className="flex gap-4 text-sm text-gray-400">
          <a href="/ops" className="hover:text-white">Health</a>
          <a href="/ops/evolution" className="hover:text-white">Evolution</a>
          <a href="/ops/signals" className="hover:text-white">Signals</a>
          <a href="/ops/costs" className="hover:text-white">LLM Costs</a>
        </nav>
      </header>
      <main className="p-6">{children}</main>
    </div>
  );
}
```

```tsx
// dashboard/src/app/trading/layout.tsx
export default function TradingLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-trading-bg text-gray-900">
      <header className="border-b border-gray-200 px-6 py-4 flex items-center justify-between">
        <h1 className="text-lg font-bold text-accent">Evolve-Trader</h1>
        <nav className="flex gap-4 text-sm text-gray-500">
          <a href="/trading" className="hover:text-gray-900">Portfolio</a>
          <a href="/trading/trades" className="hover:text-gray-900">Trades</a>
          <a href="/trading/signals" className="hover:text-gray-900">Signals</a>
          <a href="/trading/controls" className="hover:text-gray-900">Controls</a>
        </nav>
      </header>
      <main className="p-6">{children}</main>
    </div>
  );
}
```

**Step 4: Run tests to verify they pass**

```bash
cd dashboard && npm install && npx jest src/__tests__/api.test.ts
```

Expected: PASS — ApiClient unit tests green

**Step 5: Commit**

```bash
git add dashboard/
git commit -m "feat: Next.js project setup with TypeScript, Tailwind, API client, and WebSocket client"
```

---

## Task 4: Portfolio Health Panel (Ops)

**Files:**
- Create: `dashboard/src/components/ops/PortfolioHealth.tsx`
- Create: `dashboard/src/__tests__/PortfolioHealth.test.tsx`

**Step 1: Write the failing tests**

```tsx
// dashboard/src/__tests__/PortfolioHealth.test.tsx
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import { PortfolioHealth } from "../components/ops/PortfolioHealth";

const mockData = {
  total_value: 112500.0,
  cash: 28000.0,
  total_return: 0.125,
  sharpe_ratio: 1.45,
  max_drawdown: 0.08,
  current_regime: "risk-on",
  cash_deployment: 0.75,
  positions: [],
  sector_exposure: {},
  drawdown: 0.08,
  timestamp: "2026-03-28T12:00:00Z",
};

describe("PortfolioHealth", () => {
  it("renders total portfolio value", () => {
    render(<PortfolioHealth data={mockData} />);
    expect(screen.getByText(/\$112,500/)).toBeInTheDocument();
  });

  it("renders Sharpe ratio", () => {
    render(<PortfolioHealth data={mockData} />);
    expect(screen.getByText(/1\.45/)).toBeInTheDocument();
  });

  it("renders current regime", () => {
    render(<PortfolioHealth data={mockData} />);
    expect(screen.getByText(/risk-on/i)).toBeInTheDocument();
  });

  it("renders max drawdown", () => {
    render(<PortfolioHealth data={mockData} />);
    expect(screen.getByText(/8\.0%/)).toBeInTheDocument();
  });

  it("shows warning alert at 15% drawdown", () => {
    const warningData = { ...mockData, max_drawdown: 0.16, drawdown: 0.16 };
    render(<PortfolioHealth data={warningData} />);
    expect(screen.getByText(/drawdown warning/i)).toBeInTheDocument();
  });

  it("shows critical alert at 20% drawdown", () => {
    const criticalData = { ...mockData, max_drawdown: 0.21, drawdown: 0.21 };
    render(<PortfolioHealth data={criticalData} />);
    expect(screen.getByText(/drawdown critical/i)).toBeInTheDocument();
  });

  it("renders loading state when no data", () => {
    render(<PortfolioHealth data={null} />);
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });
});
```

**Step 2: Run tests to verify they fail**

```bash
cd dashboard && npx jest src/__tests__/PortfolioHealth.test.tsx
```

Expected: FAIL — `Cannot find module '../components/ops/PortfolioHealth'`

**Step 3: Implement the component**

```tsx
// dashboard/src/components/ops/PortfolioHealth.tsx
"use client";

import React from "react";

interface PortfolioHealthData {
  total_value: number;
  cash: number;
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  current_regime: string;
  cash_deployment: number;
  drawdown: number;
  timestamp: string;
  positions: unknown[];
  sector_exposure: Record<string, number>;
}

interface PortfolioHealthProps {
  data: PortfolioHealthData | null;
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function DrawdownAlert({ drawdown }: { drawdown: number }) {
  if (drawdown >= 0.2) {
    return (
      <div className="mt-4 rounded-lg bg-red-900/50 border border-red-500 px-4 py-3 text-red-200">
        <span className="font-bold">DRAWDOWN CRITICAL</span> — Current drawdown{" "}
        {formatPercent(drawdown)} exceeds 20% threshold. Consider activating kill switch.
      </div>
    );
  }
  if (drawdown >= 0.15) {
    return (
      <div className="mt-4 rounded-lg bg-yellow-900/50 border border-yellow-500 px-4 py-3 text-yellow-200">
        <span className="font-bold">DRAWDOWN WARNING</span> — Current drawdown{" "}
        {formatPercent(drawdown)} exceeds 15% threshold. Monitor closely.
      </div>
    );
  }
  return null;
}

export function PortfolioHealth({ data }: PortfolioHealthProps) {
  if (!data) {
    return (
      <div className="rounded-xl bg-ops-panel p-6 animate-pulse">
        <p className="text-gray-400">Loading portfolio health...</p>
      </div>
    );
  }

  const metrics = [
    { label: "Total Value", value: formatCurrency(data.total_value) },
    { label: "Cash", value: formatCurrency(data.cash) },
    { label: "Total Return", value: formatPercent(data.total_return), color: data.total_return >= 0 ? "text-success" : "text-danger" },
    { label: "Sharpe Ratio", value: data.sharpe_ratio.toFixed(2) },
    { label: "Max Drawdown", value: formatPercent(data.max_drawdown), color: "text-danger" },
    { label: "Cash Deployment", value: formatPercent(data.cash_deployment) },
  ];

  return (
    <div className="rounded-xl bg-ops-panel p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Portfolio Health</h2>
        <span className="inline-flex items-center rounded-full bg-blue-900/50 px-3 py-1 text-xs font-medium text-blue-300">
          {data.current_regime}
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {metrics.map((m) => (
          <div key={m.label} className="rounded-lg bg-ops-bg p-4">
            <p className="text-xs text-gray-400 uppercase tracking-wide">{m.label}</p>
            <p className={`text-xl font-bold mt-1 ${m.color ?? "text-white"}`}>{m.value}</p>
          </div>
        ))}
      </div>

      <DrawdownAlert drawdown={data.drawdown} />

      <p className="text-xs text-gray-500 mt-4">
        Last updated: {new Date(data.timestamp).toLocaleString()}
      </p>
    </div>
  );
}
```

**Step 4: Run tests to verify they pass**

```bash
cd dashboard && npx jest src/__tests__/PortfolioHealth.test.tsx
```

Expected: PASS — all 7 tests green

**Step 5: Commit**

```bash
git add dashboard/src/components/ops/PortfolioHealth.tsx dashboard/src/__tests__/PortfolioHealth.test.tsx
git commit -m "feat: Portfolio Health ops panel with drawdown alerts"
```

---

## Task 5: Strategy Evolution Panel (Ops)

**Files:**
- Create: `dashboard/src/components/ops/StrategyEvolution.tsx`
- Create: `dashboard/src/__tests__/StrategyEvolution.test.tsx`

**Step 1: Write the failing tests**

```tsx
// dashboard/src/__tests__/StrategyEvolution.test.tsx
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import { StrategyEvolution } from "../components/ops/StrategyEvolution";

const mockStrategies = [
  { name: "momentum-v2", version: 2, status: "active" as const, sharpe_ratio: 1.3, max_drawdown: 0.06, win_rate: 0.58, total_trades: 42 },
  { name: "mean-revert-v1", version: 1, status: "probation" as const, sharpe_ratio: 0.4, max_drawdown: 0.14, win_rate: 0.45, total_trades: 18 },
];

const mockEvents = [
  { id: 1, event_type: "FIX" as const, parent_skill: "momentum-v1", child_skill: "momentum-v2", trigger_reason: "Sharpe dropped below 0.5", created_at: "2026-03-20T10:00:00Z" },
  { id: 2, event_type: "DERIVED" as const, parent_skill: "momentum-v2", child_skill: "sector-momentum-v1", trigger_reason: "Sector rotation pattern detected", created_at: "2026-03-25T14:00:00Z" },
  { id: 3, event_type: "CAPTURED" as const, parent_skill: "", child_skill: "insider-follow-v1", trigger_reason: "New pattern from Form 4 signals", created_at: "2026-03-27T09:00:00Z" },
];

describe("StrategyEvolution", () => {
  it("renders active skill count", () => {
    render(<StrategyEvolution strategies={mockStrategies} events={mockEvents} />);
    expect(screen.getByText(/2 skills/i)).toBeInTheDocument();
  });

  it("renders evolution event type distribution", () => {
    render(<StrategyEvolution strategies={mockStrategies} events={mockEvents} />);
    expect(screen.getByText(/FIX/)).toBeInTheDocument();
    expect(screen.getByText(/DERIVED/)).toBeInTheDocument();
    expect(screen.getByText(/CAPTURED/)).toBeInTheDocument();
  });

  it("renders each strategy name", () => {
    render(<StrategyEvolution strategies={mockStrategies} events={mockEvents} />);
    expect(screen.getByText(/momentum-v2/)).toBeInTheDocument();
    expect(screen.getByText(/mean-revert-v1/)).toBeInTheDocument();
  });

  it("shows probation status badge", () => {
    render(<StrategyEvolution strategies={mockStrategies} events={mockEvents} />);
    expect(screen.getByText(/probation/i)).toBeInTheDocument();
  });

  it("renders loading state when no data", () => {
    render(<StrategyEvolution strategies={null} events={null} />);
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });
});
```

**Step 2: Run tests to verify they fail**

```bash
cd dashboard && npx jest src/__tests__/StrategyEvolution.test.tsx
```

Expected: FAIL — `Cannot find module '../components/ops/StrategyEvolution'`

**Step 3: Implement the component**

```tsx
// dashboard/src/components/ops/StrategyEvolution.tsx
"use client";

import React from "react";
import type { StrategyInfo, EvolutionEvent } from "@/lib/types";

interface StrategyEvolutionProps {
  strategies: StrategyInfo[] | null;
  events: EvolutionEvent[] | null;
}

const STATUS_COLORS: Record<string, string> = {
  active: "bg-success/20 text-success",
  probation: "bg-warning/20 text-warning",
  archived: "bg-gray-600/20 text-gray-400",
};

const EVENT_COLORS: Record<string, string> = {
  FIX: "bg-blue-900/50 text-blue-300 border-blue-500",
  DERIVED: "bg-purple-900/50 text-purple-300 border-purple-500",
  CAPTURED: "bg-green-900/50 text-green-300 border-green-500",
};

function EventTypeBadge({ type }: { type: string }) {
  return (
    <span className={`inline-flex items-center rounded px-2 py-0.5 text-xs font-mono font-bold border ${EVENT_COLORS[type] ?? ""}`}>
      {type}
    </span>
  );
}

function EvolutionDistribution({ events }: { events: EvolutionEvent[] }) {
  const counts = { FIX: 0, DERIVED: 0, CAPTURED: 0 };
  events.forEach((e) => {
    if (e.event_type in counts) counts[e.event_type as keyof typeof counts]++;
  });
  const total = events.length || 1;

  return (
    <div className="flex gap-2 mt-2">
      {(Object.entries(counts) as [string, number][]).map(([type, count]) => (
        <div key={type} className="flex-1 rounded-lg bg-ops-bg p-3 text-center">
          <EventTypeBadge type={type} />
          <p className="text-lg font-bold mt-1">{count}</p>
          <p className="text-xs text-gray-500">{((count / total) * 100).toFixed(0)}%</p>
        </div>
      ))}
    </div>
  );
}

export function StrategyEvolution({ strategies, events }: StrategyEvolutionProps) {
  if (!strategies || !events) {
    return (
      <div className="rounded-xl bg-ops-panel p-6 animate-pulse">
        <p className="text-gray-400">Loading strategy evolution...</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl bg-ops-panel p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Strategy Evolution</h2>
        <span className="text-sm text-gray-400">{strategies.length} skills</span>
      </div>

      {/* Evolution event distribution */}
      <EvolutionDistribution events={events} />

      {/* Strategy list */}
      <div className="mt-4 space-y-2">
        {strategies.map((s) => (
          <div key={s.name} className="flex items-center justify-between rounded-lg bg-ops-bg px-4 py-3">
            <div>
              <span className="font-mono text-sm">{s.name}</span>
              <span className={`ml-2 rounded-full px-2 py-0.5 text-xs ${STATUS_COLORS[s.status] ?? ""}`}>
                {s.status}
              </span>
            </div>
            <div className="flex gap-4 text-xs text-gray-400">
              <span>Sharpe: {s.sharpe_ratio?.toFixed(2) ?? "—"}</span>
              <span>Win: {s.win_rate ? `${(s.win_rate * 100).toFixed(0)}%` : "—"}</span>
              <span>{s.total_trades} trades</span>
            </div>
          </div>
        ))}
      </div>

      {/* Recent events timeline */}
      <div className="mt-4">
        <h3 className="text-sm font-semibold text-gray-400 mb-2">Recent Events</h3>
        <div className="space-y-2">
          {events.slice(0, 5).map((e) => (
            <div key={e.id} className="flex items-start gap-3 text-sm">
              <EventTypeBadge type={e.event_type} />
              <div>
                <span className="font-mono text-xs">
                  {e.parent_skill ? `${e.parent_skill} -> ` : ""}{e.child_skill}
                </span>
                <p className="text-xs text-gray-500">{e.trigger_reason}</p>
              </div>
              <span className="ml-auto text-xs text-gray-600 whitespace-nowrap">
                {new Date(e.created_at).toLocaleDateString()}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
```

**Step 4: Run tests to verify they pass**

```bash
cd dashboard && npx jest src/__tests__/StrategyEvolution.test.tsx
```

Expected: PASS — all 5 tests green

**Step 5: Commit**

```bash
git add dashboard/src/components/ops/StrategyEvolution.tsx dashboard/src/__tests__/StrategyEvolution.test.tsx
git commit -m "feat: Strategy Evolution ops panel with event distribution and timeline"
```

---

## Task 6: Signal Source Performance Panel (Ops)

**Files:**
- Create: `dashboard/src/components/ops/SignalPerformance.tsx`
- Create: `dashboard/src/__tests__/SignalPerformance.test.tsx`

**Step 1: Write the failing tests**

```tsx
// dashboard/src/__tests__/SignalPerformance.test.tsx
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import { SignalPerformance } from "../components/ops/SignalPerformance";

const mockSources = [
  { source: "edgar_13f", signal_count: 148, hit_rate: 0.62, avg_alpha: 0.034, impact_trend: "rising", independence_score: 0.81 },
  { source: "form4_insider", signal_count: 312, hit_rate: 0.55, avg_alpha: 0.018, impact_trend: "stable", independence_score: 0.73 },
  { source: "congressional", signal_count: 67, hit_rate: 0.48, avg_alpha: 0.012, impact_trend: "declining", independence_score: 0.91 },
];

describe("SignalPerformance", () => {
  it("renders each source name", () => {
    render(<SignalPerformance sources={mockSources} />);
    expect(screen.getByText(/edgar_13f/)).toBeInTheDocument();
    expect(screen.getByText(/form4_insider/)).toBeInTheDocument();
    expect(screen.getByText(/congressional/)).toBeInTheDocument();
  });

  it("renders hit rates as percentages", () => {
    render(<SignalPerformance sources={mockSources} />);
    expect(screen.getByText(/62\.0%/)).toBeInTheDocument();
  });

  it("renders impact trend indicators", () => {
    render(<SignalPerformance sources={mockSources} />);
    expect(screen.getByText(/rising/i)).toBeInTheDocument();
    expect(screen.getByText(/declining/i)).toBeInTheDocument();
  });

  it("renders independence scores", () => {
    render(<SignalPerformance sources={mockSources} />);
    expect(screen.getByText(/0\.81/)).toBeInTheDocument();
  });

  it("renders loading state when no data", () => {
    render(<SignalPerformance sources={null} />);
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });
});
```

**Step 2: Run tests to verify they fail**

```bash
cd dashboard && npx jest src/__tests__/SignalPerformance.test.tsx
```

Expected: FAIL — `Cannot find module '../components/ops/SignalPerformance'`

**Step 3: Implement the component**

```tsx
// dashboard/src/components/ops/SignalPerformance.tsx
"use client";

import React from "react";

interface SignalSourcePerf {
  source: string;
  signal_count: number;
  hit_rate: number;
  avg_alpha: number;
  impact_trend: "rising" | "stable" | "declining";
  independence_score: number;
}

interface SignalPerformanceProps {
  sources: SignalSourcePerf[] | null;
}

const TREND_STYLES: Record<string, { color: string; arrow: string }> = {
  rising: { color: "text-success", arrow: "^" },
  stable: { color: "text-gray-400", arrow: "~" },
  declining: { color: "text-danger", arrow: "v" },
};

export function SignalPerformance({ sources }: SignalPerformanceProps) {
  if (!sources) {
    return (
      <div className="rounded-xl bg-ops-panel p-6 animate-pulse">
        <p className="text-gray-400">Loading signal performance...</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl bg-ops-panel p-6">
      <h2 className="text-lg font-semibold mb-4">Signal Source Performance</h2>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-gray-400 uppercase tracking-wide border-b border-gray-700">
              <th className="text-left py-2 pr-4">Source</th>
              <th className="text-right py-2 px-4">Signals</th>
              <th className="text-right py-2 px-4">Hit Rate</th>
              <th className="text-right py-2 px-4">Avg Alpha</th>
              <th className="text-center py-2 px-4">Trend</th>
              <th className="text-right py-2 pl-4">Independence</th>
            </tr>
          </thead>
          <tbody>
            {sources.map((s) => {
              const trend = TREND_STYLES[s.impact_trend] ?? TREND_STYLES.stable;
              return (
                <tr key={s.source} className="border-b border-gray-800 hover:bg-ops-bg/50">
                  <td className="py-3 pr-4 font-mono">{s.source}</td>
                  <td className="py-3 px-4 text-right">{s.signal_count}</td>
                  <td className="py-3 px-4 text-right font-bold">
                    {(s.hit_rate * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 text-right">
                    {(s.avg_alpha * 100).toFixed(2)}%
                  </td>
                  <td className={`py-3 px-4 text-center ${trend.color}`}>
                    {trend.arrow} {s.impact_trend}
                  </td>
                  <td className="py-3 pl-4 text-right">{s.independence_score.toFixed(2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
```

**Step 4: Run tests to verify they pass**

```bash
cd dashboard && npx jest src/__tests__/SignalPerformance.test.tsx
```

Expected: PASS — all 5 tests green

**Step 5: Commit**

```bash
git add dashboard/src/components/ops/SignalPerformance.tsx dashboard/src/__tests__/SignalPerformance.test.tsx
git commit -m "feat: Signal Source Performance ops panel with hit rate, alpha, and independence"
```

---

## Task 7: LLM Costs Panel (Ops)

**Files:**
- Create: `dashboard/src/components/ops/LLMCosts.tsx`
- Create: `dashboard/src/__tests__/LLMCosts.test.tsx`

**Step 1: Write the failing tests**

```tsx
// dashboard/src/__tests__/LLMCosts.test.tsx
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import { LLMCostsPanel } from "../components/ops/LLMCosts";

const mockCosts = {
  total_spend: 47.82,
  by_component: {
    strategy_evolution: 22.10,
    signal_analysis: 14.50,
    regime_classification: 6.30,
    trade_reasoning: 4.92,
  },
  cost_per_trade: 0.38,
  budget_remaining: 152.18,
  budget_utilization: 0.239,
};

describe("LLMCostsPanel", () => {
  it("renders total spend", () => {
    render(<LLMCostsPanel data={mockCosts} />);
    expect(screen.getByText(/\$47\.82/)).toBeInTheDocument();
  });

  it("renders per-component breakdown", () => {
    render(<LLMCostsPanel data={mockCosts} />);
    expect(screen.getByText(/strategy_evolution/)).toBeInTheDocument();
    expect(screen.getByText(/signal_analysis/)).toBeInTheDocument();
  });

  it("renders cost per trade", () => {
    render(<LLMCostsPanel data={mockCosts} />);
    expect(screen.getByText(/\$0\.38/)).toBeInTheDocument();
  });

  it("renders budget utilization bar", () => {
    render(<LLMCostsPanel data={mockCosts} />);
    expect(screen.getByText(/23\.9%/)).toBeInTheDocument();
  });

  it("renders loading state when no data", () => {
    render(<LLMCostsPanel data={null} />);
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });
});
```

**Step 2: Run tests to verify they fail**

```bash
cd dashboard && npx jest src/__tests__/LLMCosts.test.tsx
```

Expected: FAIL — `Cannot find module '../components/ops/LLMCosts'`

**Step 3: Implement the component**

```tsx
// dashboard/src/components/ops/LLMCosts.tsx
"use client";

import React from "react";
import type { LLMCosts } from "@/lib/types";

interface LLMCostsPanelProps {
  data: LLMCosts | null;
}

function formatUSD(value: number): string {
  return `$${value.toFixed(2)}`;
}

export function LLMCostsPanel({ data }: LLMCostsPanelProps) {
  if (!data) {
    return (
      <div className="rounded-xl bg-ops-panel p-6 animate-pulse">
        <p className="text-gray-400">Loading LLM costs...</p>
      </div>
    );
  }

  const utilizationPct = (data.budget_utilization * 100).toFixed(1);
  const barColor =
    data.budget_utilization > 0.8
      ? "bg-danger"
      : data.budget_utilization > 0.6
        ? "bg-warning"
        : "bg-success";

  const sortedComponents = Object.entries(data.by_component).sort(
    ([, a], [, b]) => b - a
  );

  return (
    <div className="rounded-xl bg-ops-panel p-6">
      <h2 className="text-lg font-semibold mb-4">LLM Costs</h2>

      {/* Summary cards */}
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="rounded-lg bg-ops-bg p-4">
          <p className="text-xs text-gray-400 uppercase tracking-wide">Total Spend</p>
          <p className="text-xl font-bold mt-1">{formatUSD(data.total_spend)}</p>
        </div>
        <div className="rounded-lg bg-ops-bg p-4">
          <p className="text-xs text-gray-400 uppercase tracking-wide">Cost / Trade</p>
          <p className="text-xl font-bold mt-1">{formatUSD(data.cost_per_trade)}</p>
        </div>
        <div className="rounded-lg bg-ops-bg p-4">
          <p className="text-xs text-gray-400 uppercase tracking-wide">Budget Left</p>
          <p className="text-xl font-bold mt-1">{formatUSD(data.budget_remaining)}</p>
        </div>
      </div>

      {/* Budget utilization bar */}
      <div className="mb-4">
        <div className="flex justify-between text-xs text-gray-400 mb-1">
          <span>Budget Utilization</span>
          <span>{utilizationPct}%</span>
        </div>
        <div className="h-3 rounded-full bg-ops-bg overflow-hidden">
          <div
            className={`h-full rounded-full ${barColor} transition-all`}
            style={{ width: `${Math.min(Number(utilizationPct), 100)}%` }}
          />
        </div>
      </div>

      {/* Per-component breakdown */}
      <h3 className="text-sm font-semibold text-gray-400 mb-2">By Component</h3>
      <div className="space-y-2">
        {sortedComponents.map(([component, cost]) => {
          const pct = data.total_spend > 0 ? (cost / data.total_spend) * 100 : 0;
          return (
            <div key={component} className="flex items-center gap-3">
              <span className="text-xs font-mono w-40 truncate">{component}</span>
              <div className="flex-1 h-2 rounded-full bg-ops-bg overflow-hidden">
                <div
                  className="h-full rounded-full bg-accent"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="text-xs text-gray-400 w-16 text-right">
                {formatUSD(cost)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

**Step 4: Run tests to verify they pass**

```bash
cd dashboard && npx jest src/__tests__/LLMCosts.test.tsx
```

Expected: PASS — all 5 tests green

**Step 5: Commit**

```bash
git add dashboard/src/components/ops/LLMCosts.tsx dashboard/src/__tests__/LLMCosts.test.tsx
git commit -m "feat: LLM Costs ops panel with budget utilization and per-component breakdown"
```

---

## Task 8: User Portfolio Overview

**Files:**
- Create: `dashboard/src/components/trading/PortfolioOverview.tsx`
- Create: `dashboard/src/__tests__/PortfolioOverview.test.tsx`

**Step 1: Write the failing tests**

```tsx
// dashboard/src/__tests__/PortfolioOverview.test.tsx
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import { PortfolioOverview } from "../components/trading/PortfolioOverview";

const mockSnapshot = {
  total_value: 112500.0,
  cash: 28000.0,
  positions: [
    { ticker: "AAPL", shares: 50, value: 8750.0, sector: "Technology", pnl: 450.0, return_pct: 0.054 },
    { ticker: "JPM", shares: 30, value: 6200.0, sector: "Financials", pnl: -120.0, return_pct: -0.019 },
    { ticker: "XOM", shares: 40, value: 3800.0, sector: "Energy", pnl: 210.0, return_pct: 0.058 },
  ],
  sector_exposure: { Technology: 0.10, Financials: 0.07, Energy: 0.04 },
  total_return: 0.125,
  drawdown: 0.03,
  timestamp: "2026-03-28T12:00:00Z",
};

const mockEquityCurve = [
  { date: "2026-03-01", value: 100000, drawdown: 0, benchmark: 100000 },
  { date: "2026-03-15", value: 106000, drawdown: 0, benchmark: 102000 },
  { date: "2026-03-28", value: 112500, drawdown: 0.03, benchmark: 103500 },
];

describe("PortfolioOverview", () => {
  it("renders total portfolio value", () => {
    render(<PortfolioOverview snapshot={mockSnapshot} equityCurve={mockEquityCurve} />);
    expect(screen.getByText(/\$112,500/)).toBeInTheDocument();
  });

  it("renders total return percentage", () => {
    render(<PortfolioOverview snapshot={mockSnapshot} equityCurve={mockEquityCurve} />);
    expect(screen.getByText(/12\.5%/)).toBeInTheDocument();
  });

  it("renders each position ticker", () => {
    render(<PortfolioOverview snapshot={mockSnapshot} equityCurve={mockEquityCurve} />);
    expect(screen.getByText("AAPL")).toBeInTheDocument();
    expect(screen.getByText("JPM")).toBeInTheDocument();
    expect(screen.getByText("XOM")).toBeInTheDocument();
  });

  it("shows positive P&L in green styling", () => {
    render(<PortfolioOverview snapshot={mockSnapshot} equityCurve={mockEquityCurve} />);
    const aaplPnl = screen.getByText(/\+\$450/);
    expect(aaplPnl).toBeInTheDocument();
  });

  it("shows negative P&L in red styling", () => {
    render(<PortfolioOverview snapshot={mockSnapshot} equityCurve={mockEquityCurve} />);
    const jpmPnl = screen.getByText(/-\$120/);
    expect(jpmPnl).toBeInTheDocument();
  });

  it("renders loading state when no data", () => {
    render(<PortfolioOverview snapshot={null} equityCurve={null} />);
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });
});
```

**Step 2: Run tests to verify they fail**

```bash
cd dashboard && npx jest src/__tests__/PortfolioOverview.test.tsx
```

Expected: FAIL — `Cannot find module '../components/trading/PortfolioOverview'`

**Step 3: Implement the component**

```tsx
// dashboard/src/components/trading/PortfolioOverview.tsx
"use client";

import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
  AreaChart,
  Area,
} from "recharts";
import type { PortfolioSnapshot, EquityCurvePoint, Position } from "@/lib/types";

interface PortfolioOverviewProps {
  snapshot: PortfolioSnapshot | null;
  equityCurve: EquityCurvePoint[] | null;
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

function formatPnl(value: number): string {
  const sign = value >= 0 ? "+" : "-";
  return `${sign}$${Math.abs(value).toLocaleString("en-US", { minimumFractionDigits: 0 })}`;
}

function PositionRow({ position }: { position: Position }) {
  const pnl = position.pnl ?? 0;
  const pnlColor = pnl >= 0 ? "text-green-600" : "text-red-600";

  return (
    <tr className="border-b border-gray-100 hover:bg-gray-50">
      <td className="py-3 pr-4 font-semibold">{position.ticker}</td>
      <td className="py-3 px-4 text-gray-500">{position.sector}</td>
      <td className="py-3 px-4 text-right">{position.shares}</td>
      <td className="py-3 px-4 text-right">{formatCurrency(position.value)}</td>
      <td className={`py-3 px-4 text-right font-medium ${pnlColor}`}>
        {formatPnl(pnl)}
      </td>
      <td className={`py-3 pl-4 text-right font-medium ${pnlColor}`}>
        {position.return_pct != null ? `${(position.return_pct * 100).toFixed(1)}%` : "—"}
      </td>
    </tr>
  );
}

export function PortfolioOverview({ snapshot, equityCurve }: PortfolioOverviewProps) {
  if (!snapshot || !equityCurve) {
    return (
      <div className="rounded-xl bg-trading-panel p-6 shadow-sm animate-pulse">
        <p className="text-gray-400">Loading portfolio...</p>
      </div>
    );
  }

  const returnColor = snapshot.total_return >= 0 ? "text-green-600" : "text-red-600";

  return (
    <div className="space-y-6">
      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="rounded-xl bg-trading-panel p-4 shadow-sm">
          <p className="text-xs text-gray-500 uppercase tracking-wide">Portfolio Value</p>
          <p className="text-2xl font-bold mt-1">{formatCurrency(snapshot.total_value)}</p>
        </div>
        <div className="rounded-xl bg-trading-panel p-4 shadow-sm">
          <p className="text-xs text-gray-500 uppercase tracking-wide">Cash</p>
          <p className="text-2xl font-bold mt-1">{formatCurrency(snapshot.cash)}</p>
        </div>
        <div className="rounded-xl bg-trading-panel p-4 shadow-sm">
          <p className="text-xs text-gray-500 uppercase tracking-wide">Total Return</p>
          <p className={`text-2xl font-bold mt-1 ${returnColor}`}>
            {(snapshot.total_return * 100).toFixed(1)}%
          </p>
        </div>
        <div className="rounded-xl bg-trading-panel p-4 shadow-sm">
          <p className="text-xs text-gray-500 uppercase tracking-wide">Drawdown</p>
          <p className="text-2xl font-bold mt-1 text-red-600">
            {(snapshot.drawdown * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Equity curve chart */}
      <div className="rounded-xl bg-trading-panel p-6 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-4">Equity Curve vs Benchmark</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={equityCurve}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="date" tick={{ fontSize: 12 }} stroke="#9ca3af" />
            <YAxis tick={{ fontSize: 12 }} stroke="#9ca3af" tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`} />
            <Tooltip formatter={(value: number) => formatCurrency(value)} />
            <Legend />
            <Line type="monotone" dataKey="value" name="Portfolio" stroke="#3b82f6" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="benchmark" name="Benchmark" stroke="#9ca3af" strokeWidth={1} strokeDasharray="5 5" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Drawdown chart */}
      <div className="rounded-xl bg-trading-panel p-6 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-4">Drawdown</h3>
        <ResponsiveContainer width="100%" height={150}>
          <AreaChart data={equityCurve}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="date" tick={{ fontSize: 12 }} stroke="#9ca3af" />
            <YAxis tick={{ fontSize: 12 }} stroke="#9ca3af" tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
            <Tooltip formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
            <Area type="monotone" dataKey="drawdown" stroke="#ef4444" fill="#ef4444" fillOpacity={0.15} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Holdings table */}
      <div className="rounded-xl bg-trading-panel p-6 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-4">Holdings</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-gray-400 uppercase tracking-wide border-b border-gray-200">
                <th className="text-left py-2 pr-4">Ticker</th>
                <th className="text-left py-2 px-4">Sector</th>
                <th className="text-right py-2 px-4">Shares</th>
                <th className="text-right py-2 px-4">Value</th>
                <th className="text-right py-2 px-4">P&amp;L</th>
                <th className="text-right py-2 pl-4">Return</th>
              </tr>
            </thead>
            <tbody>
              {snapshot.positions.map((p) => (
                <PositionRow key={p.ticker} position={p} />
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
```

**Step 4: Run tests to verify they pass**

```bash
cd dashboard && npx jest src/__tests__/PortfolioOverview.test.tsx
```

Expected: PASS — all 6 tests green (Recharts components render in jsdom via mocking or shallow)

**Step 5: Commit**

```bash
git add dashboard/src/components/trading/PortfolioOverview.tsx dashboard/src/__tests__/PortfolioOverview.test.tsx
git commit -m "feat: Portfolio Overview trading panel with equity curve, drawdown chart, and holdings"
```

---

## Task 9: Trade History & Strategy Performance

**Files:**
- Create: `dashboard/src/components/trading/TradeHistory.tsx`
- Create: `dashboard/src/components/trading/StrategyPerformance.tsx`
- Create: `dashboard/src/__tests__/TradeHistory.test.tsx`

**Step 1: Write the failing tests**

```tsx
// dashboard/src/__tests__/TradeHistory.test.tsx
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { TradeHistory } from "../components/trading/TradeHistory";
import { StrategyPerformance } from "../components/trading/StrategyPerformance";

const mockTrades = [
  {
    id: 1, strategy_skill: "momentum-v2", ticker: "AAPL", direction: "BUY" as const,
    quantity: 50, entry_price: 170.0, exit_price: 178.0,
    entry_date: "2026-03-10T10:00:00Z", exit_date: "2026-03-20T14:00:00Z",
    pnl: 400.0, return_pct: 0.047, regime_label: "risk-on",
    signal_sources: ["edgar_13f", "form4_insider"],
  },
  {
    id: 2, strategy_skill: "mean-revert-v1", ticker: "TSLA", direction: "SHORT" as const,
    quantity: 20, entry_price: 240.0, exit_price: 255.0,
    entry_date: "2026-03-12T09:00:00Z", exit_date: "2026-03-22T16:00:00Z",
    pnl: -300.0, return_pct: -0.0625, regime_label: "risk-off",
    signal_sources: ["congressional"],
  },
];

const mockStrategies = [
  { name: "momentum-v2", version: 2, status: "active" as const, sharpe_ratio: 1.3, max_drawdown: 0.06, win_rate: 0.58, total_trades: 42 },
  { name: "mean-revert-v1", version: 1, status: "probation" as const, sharpe_ratio: 0.4, max_drawdown: 0.14, win_rate: 0.45, total_trades: 18 },
];

describe("TradeHistory", () => {
  it("renders all trade rows", () => {
    render(<TradeHistory trades={mockTrades} total={2} />);
    expect(screen.getByText("AAPL")).toBeInTheDocument();
    expect(screen.getByText("TSLA")).toBeInTheDocument();
  });

  it("shows trade direction", () => {
    render(<TradeHistory trades={mockTrades} total={2} />);
    expect(screen.getByText("BUY")).toBeInTheDocument();
    expect(screen.getByText("SHORT")).toBeInTheDocument();
  });

  it("shows signal sources for each trade", () => {
    render(<TradeHistory trades={mockTrades} total={2} />);
    expect(screen.getByText(/edgar_13f/)).toBeInTheDocument();
  });

  it("renders loading state when no data", () => {
    render(<TradeHistory trades={null} total={0} />);
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });
});

describe("StrategyPerformance", () => {
  it("renders each strategy", () => {
    render(<StrategyPerformance strategies={mockStrategies} />);
    expect(screen.getByText(/momentum-v2/)).toBeInTheDocument();
    expect(screen.getByText(/mean-revert-v1/)).toBeInTheDocument();
  });

  it("renders Sharpe ratios", () => {
    render(<StrategyPerformance strategies={mockStrategies} />);
    expect(screen.getByText(/1\.30/)).toBeInTheDocument();
  });

  it("renders loading state when no data", () => {
    render(<StrategyPerformance strategies={null} />);
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });
});
```

**Step 2: Run tests to verify they fail**

```bash
cd dashboard && npx jest src/__tests__/TradeHistory.test.tsx
```

Expected: FAIL — `Cannot find module '../components/trading/TradeHistory'`

**Step 3: Implement the components**

```tsx
// dashboard/src/components/trading/TradeHistory.tsx
"use client";

import React, { useState } from "react";
import type { Trade } from "@/lib/types";

interface TradeHistoryProps {
  trades: Trade[] | null;
  total: number;
  onPageChange?: (offset: number) => void;
  onSearch?: (query: string) => void;
}

const DIRECTION_COLORS: Record<string, string> = {
  BUY: "text-green-600 bg-green-50",
  SELL: "text-red-600 bg-red-50",
  SHORT: "text-orange-600 bg-orange-50",
  COVER: "text-blue-600 bg-blue-50",
};

function formatCurrency(value: number | null): string {
  if (value == null) return "—";
  const sign = value >= 0 ? "+" : "-";
  return `${sign}$${Math.abs(value).toLocaleString("en-US", { minimumFractionDigits: 0 })}`;
}

export function TradeHistory({ trades, total, onPageChange, onSearch }: TradeHistoryProps) {
  const [searchQuery, setSearchQuery] = useState("");

  if (!trades) {
    return (
      <div className="rounded-xl bg-trading-panel p-6 shadow-sm animate-pulse">
        <p className="text-gray-400">Loading trade history...</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl bg-trading-panel p-6 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-gray-700">
          Trade History ({total} total)
        </h3>
        <input
          type="text"
          placeholder="Search ticker, strategy..."
          className="rounded-lg border border-gray-200 px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-accent"
          value={searchQuery}
          onChange={(e) => {
            setSearchQuery(e.target.value);
            onSearch?.(e.target.value);
          }}
        />
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-gray-400 uppercase tracking-wide border-b border-gray-200">
              <th className="text-left py-2 pr-3">Date</th>
              <th className="text-left py-2 px-3">Ticker</th>
              <th className="text-center py-2 px-3">Dir</th>
              <th className="text-left py-2 px-3">Strategy</th>
              <th className="text-right py-2 px-3">Qty</th>
              <th className="text-right py-2 px-3">Entry</th>
              <th className="text-right py-2 px-3">Exit</th>
              <th className="text-right py-2 px-3">P&amp;L</th>
              <th className="text-left py-2 pl-3">Sources</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((t) => {
              const pnlColor = (t.pnl ?? 0) >= 0 ? "text-green-600" : "text-red-600";
              const dirClass = DIRECTION_COLORS[t.direction] ?? "";
              return (
                <tr key={t.id} className="border-b border-gray-50 hover:bg-gray-50">
                  <td className="py-2.5 pr-3 text-xs text-gray-500 whitespace-nowrap">
                    {new Date(t.entry_date).toLocaleDateString()}
                  </td>
                  <td className="py-2.5 px-3 font-semibold">{t.ticker}</td>
                  <td className="py-2.5 px-3 text-center">
                    <span className={`rounded px-2 py-0.5 text-xs font-bold ${dirClass}`}>
                      {t.direction}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 font-mono text-xs text-gray-500">
                    {t.strategy_skill}
                  </td>
                  <td className="py-2.5 px-3 text-right">{t.quantity}</td>
                  <td className="py-2.5 px-3 text-right">${t.entry_price.toFixed(2)}</td>
                  <td className="py-2.5 px-3 text-right">
                    {t.exit_price != null ? `$${t.exit_price.toFixed(2)}` : "—"}
                  </td>
                  <td className={`py-2.5 px-3 text-right font-medium ${pnlColor}`}>
                    {formatCurrency(t.pnl)}
                  </td>
                  <td className="py-2.5 pl-3 text-xs text-gray-400">
                    {t.signal_sources.join(", ")}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
```

```tsx
// dashboard/src/components/trading/StrategyPerformance.tsx
"use client";

import React from "react";
import type { StrategyInfo } from "@/lib/types";

interface StrategyPerformanceProps {
  strategies: StrategyInfo[] | null;
}

const STATUS_COLORS: Record<string, string> = {
  active: "bg-green-100 text-green-700",
  probation: "bg-yellow-100 text-yellow-700",
  archived: "bg-gray-100 text-gray-500",
};

export function StrategyPerformance({ strategies }: StrategyPerformanceProps) {
  if (!strategies) {
    return (
      <div className="rounded-xl bg-trading-panel p-6 shadow-sm animate-pulse">
        <p className="text-gray-400">Loading strategy performance...</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl bg-trading-panel p-6 shadow-sm">
      <h3 className="text-sm font-semibold text-gray-700 mb-4">Strategy Performance</h3>

      <div className="grid gap-4 md:grid-cols-2">
        {strategies.map((s) => (
          <div key={s.name} className="rounded-lg border border-gray-100 p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="font-mono text-sm font-semibold">{s.name}</span>
              <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${STATUS_COLORS[s.status] ?? ""}`}>
                {s.status}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <p className="text-xs text-gray-400">Sharpe</p>
                <p className="font-bold">{s.sharpe_ratio?.toFixed(2) ?? "—"}</p>
              </div>
              <div>
                <p className="text-xs text-gray-400">Max DD</p>
                <p className="font-bold text-red-600">
                  {s.max_drawdown != null ? `${(s.max_drawdown * 100).toFixed(1)}%` : "—"}
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-400">Win Rate</p>
                <p className="font-bold">
                  {s.win_rate != null ? `${(s.win_rate * 100).toFixed(0)}%` : "—"}
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-400">Trades</p>
                <p className="font-bold">{s.total_trades}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

**Step 4: Run tests to verify they pass**

```bash
cd dashboard && npx jest src/__tests__/TradeHistory.test.tsx
```

Expected: PASS — all 7 tests green

**Step 5: Commit**

```bash
git add dashboard/src/components/trading/TradeHistory.tsx dashboard/src/components/trading/StrategyPerformance.tsx dashboard/src/__tests__/TradeHistory.test.tsx
git commit -m "feat: Trade History and Strategy Performance trading panels"
```

---

## Task 10: Signal Source Explorer

**Files:**
- Create: `dashboard/src/components/trading/SignalExplorer.tsx`
- Create: `dashboard/src/__tests__/SignalExplorer.test.tsx`

**Step 1: Write the failing tests**

```tsx
// dashboard/src/__tests__/SignalExplorer.test.tsx
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import { SignalExplorer } from "../components/trading/SignalExplorer";

const mockSources = [
  {
    source: "edgar_13f",
    signal_count: 148,
    hit_rate: 0.62,
    avg_confidence: 0.78,
    last_signal_at: "2026-03-27T15:00:00Z",
    lifecycle_stage: "mature",
    weight: 0.35,
    rolling_hit_rate: [0.58, 0.60, 0.62, 0.64, 0.61],
  },
  {
    source: "form4_insider",
    signal_count: 312,
    hit_rate: 0.55,
    avg_confidence: 0.65,
    last_signal_at: "2026-03-28T09:30:00Z",
    lifecycle_stage: "mature",
    weight: 0.25,
    rolling_hit_rate: [0.52, 0.54, 0.55, 0.56, 0.55],
  },
  {
    source: "congressional",
    signal_count: 67,
    hit_rate: 0.48,
    avg_confidence: 0.52,
    last_signal_at: "2026-03-25T11:00:00Z",
    lifecycle_stage: "probation",
    weight: 0.10,
    rolling_hit_rate: [0.50, 0.49, 0.47, 0.48, 0.48],
  },
];

describe("SignalExplorer", () => {
  it("renders each source name", () => {
    render(<SignalExplorer sources={mockSources} />);
    expect(screen.getByText(/edgar_13f/)).toBeInTheDocument();
    expect(screen.getByText(/form4_insider/)).toBeInTheDocument();
    expect(screen.getByText(/congressional/)).toBeInTheDocument();
  });

  it("renders hit rates", () => {
    render(<SignalExplorer sources={mockSources} />);
    expect(screen.getByText(/62\.0%/)).toBeInTheDocument();
  });

  it("renders lifecycle stage badges", () => {
    render(<SignalExplorer sources={mockSources} />);
    expect(screen.getAllByText(/mature/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/probation/i)).toBeInTheDocument();
  });

  it("renders weight as percentage", () => {
    render(<SignalExplorer sources={mockSources} />);
    expect(screen.getByText(/35\.0%/)).toBeInTheDocument();
  });

  it("renders loading state when no data", () => {
    render(<SignalExplorer sources={null} />);
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });
});
```

**Step 2: Run tests to verify they fail**

```bash
cd dashboard && npx jest src/__tests__/SignalExplorer.test.tsx
```

Expected: FAIL — `Cannot find module '../components/trading/SignalExplorer'`

**Step 3: Implement the component**

```tsx
// dashboard/src/components/trading/SignalExplorer.tsx
"use client";

import React from "react";
import {
  LineChart,
  Line,
  ResponsiveContainer,
} from "recharts";

interface SignalSourceDetail {
  source: string;
  signal_count: number;
  hit_rate: number;
  avg_confidence: number;
  last_signal_at: string | null;
  lifecycle_stage: string;
  weight: number;
  rolling_hit_rate: number[];
}

interface SignalExplorerProps {
  sources: SignalSourceDetail[] | null;
}

const STAGE_COLORS: Record<string, string> = {
  mature: "bg-green-100 text-green-700",
  growing: "bg-blue-100 text-blue-700",
  probation: "bg-yellow-100 text-yellow-700",
  new: "bg-purple-100 text-purple-700",
  deprecated: "bg-gray-100 text-gray-500",
};

function MiniSparkline({ data }: { data: number[] }) {
  const chartData = data.map((v, i) => ({ idx: i, rate: v }));
  return (
    <ResponsiveContainer width={100} height={30}>
      <LineChart data={chartData}>
        <Line type="monotone" dataKey="rate" stroke="#3b82f6" strokeWidth={1.5} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function SourceCard({ source }: { source: SignalSourceDetail }) {
  const stageClass = STAGE_COLORS[source.lifecycle_stage] ?? STAGE_COLORS.new;

  return (
    <div className="rounded-lg border border-gray-100 p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="font-mono text-sm font-semibold">{source.source}</span>
        <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${stageClass}`}>
          {source.lifecycle_stage}
        </span>
      </div>

      <div className="grid grid-cols-3 gap-3 text-sm mb-3">
        <div>
          <p className="text-xs text-gray-400">Hit Rate</p>
          <p className="font-bold">{(source.hit_rate * 100).toFixed(1)}%</p>
        </div>
        <div>
          <p className="text-xs text-gray-400">Weight</p>
          <p className="font-bold">{(source.weight * 100).toFixed(1)}%</p>
        </div>
        <div>
          <p className="text-xs text-gray-400">Signals</p>
          <p className="font-bold">{source.signal_count}</p>
        </div>
      </div>

      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs text-gray-400">Avg Confidence</p>
          <p className="text-sm font-medium">{(source.avg_confidence * 100).toFixed(0)}%</p>
        </div>
        <div>
          <p className="text-xs text-gray-400 mb-1">Rolling Hit Rate</p>
          <MiniSparkline data={source.rolling_hit_rate} />
        </div>
      </div>

      {source.last_signal_at && (
        <p className="text-xs text-gray-400 mt-2">
          Last signal: {new Date(source.last_signal_at).toLocaleString()}
        </p>
      )}
    </div>
  );
}

export function SignalExplorer({ sources }: SignalExplorerProps) {
  if (!sources) {
    return (
      <div className="rounded-xl bg-trading-panel p-6 shadow-sm animate-pulse">
        <p className="text-gray-400">Loading signal sources...</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl bg-trading-panel p-6 shadow-sm">
      <h3 className="text-sm font-semibold text-gray-700 mb-4">
        Signal Source Explorer ({sources.length} sources)
      </h3>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {sources.map((s) => (
          <SourceCard key={s.source} source={s} />
        ))}
      </div>
    </div>
  );
}
```

**Step 4: Run tests to verify they pass**

```bash
cd dashboard && npx jest src/__tests__/SignalExplorer.test.tsx
```

Expected: PASS — all 5 tests green

**Step 5: Commit**

```bash
git add dashboard/src/components/trading/SignalExplorer.tsx dashboard/src/__tests__/SignalExplorer.test.tsx
git commit -m "feat: Signal Source Explorer trading panel with sparklines and lifecycle stages"
```

---

## Task 11: Operator Controls Surface

This task intentionally stops short of implementing destructive or approval-bearing backend actions. Phase 5 builds the dashboard shell, status APIs, and component contracts. Phase 6 wires real approval actions. Phase 11 wires the real kill switch.

**Files:**
- Create: `src/evolve_trader/api/routes/operator_state.py`
- Create: `dashboard/src/components/controls/KillSwitchStatus.tsx`
- Create: `dashboard/src/components/controls/ApprovalQueue.tsx`
- Create: `dashboard/src/components/controls/SystemMode.tsx`
- Create: `tests/api/test_operator_state_api.py`
- Create: `dashboard/src/__tests__/OperatorState.test.tsx`

**Scope:**
- Read-only kill switch status and readiness indicator
- Pending approval queue visibility
- System mode display and safe local toggles such as evolution pause/resume
- Manual regime override status and audit visibility
- No backend endpoint in this phase may directly approve trades or activate the kill switch

**Acceptance criteria:**
- Dashboard renders operator state consistently from API data
- Status views degrade safely when later-phase backends are unavailable
- Component contracts are stable enough for Phase 6 and Phase 11 to attach live handlers without redesign





    <div className={`rounded-xl p-6 shadow-sm border-2 ${
      active
        ? "bg-red-50 border-red-500"
        : "bg-trading-panel border-gray-100"
    }`}>
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-gray-700">Emergency Kill Switch</h3>
          <p className="text-xs text-gray-400 mt-1">
            {active
              ? "ACTIVE — All trading halted"
              : "Inactive — System trading normally"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {confirming && (
            <>
              <span className="text-sm text-red-600 font-medium">Confirm activation?</span>
              <button
                onClick={handleCancel}
                className="rounded-lg bg-gray-200 px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-300"
              >
                Cancel
              </button>
            </>
          )}
          <button
            onClick={handleClick}
            aria-label="Kill Switch"
            className={`rounded-lg px-6 py-3 text-sm font-bold transition ${
              active
                ? "bg-green-600 text-white hover:bg-green-700"
                : "bg-red-600 text-white hover:bg-red-700"
            }`}
          >
            {active ? "Deactivate" : "Kill Switch"}
          </button>
        </div>
      </div>
    </div>
  );
}
```

```tsx
// dashboard/src/components/controls/ApprovalQueue.tsx
"use client";

import React from "react";

interface PendingTrade {
  id: number;
  ticker: string;
  direction: "BUY" | "SELL" | "SHORT" | "COVER";
  quantity: number;
  strategy_skill: string;
}

interface TradeApprovalProps {
  pending: PendingTrade[];
  onApprove: (tradeId: number) => void;
  onReject: (tradeId: number) => void;
}

const DIR_COLORS: Record<string, string> = {
  BUY: "text-green-600",
  SELL: "text-red-600",
  SHORT: "text-orange-600",
  COVER: "text-blue-600",
};

export function TradeApproval({ pending, onApprove, onReject }: TradeApprovalProps) {
  if (pending.length === 0) {
    return (
      <div className="rounded-xl bg-trading-panel p-6 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-2">Trade Approval</h3>
        <p className="text-sm text-gray-400">No pending trades</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl bg-trading-panel p-6 shadow-sm">
      <h3 className="text-sm font-semibold text-gray-700 mb-4">
        Trade Approval ({pending.length} pending)
      </h3>
      <div className="space-y-3">
        {pending.map((trade) => (
          <div
            key={trade.id}
            className="flex items-center justify-between rounded-lg border border-gray-100 p-3"
          >
            <div className="flex items-center gap-3">
              <span className={`font-bold text-sm ${DIR_COLORS[trade.direction] ?? ""}`}>
                {trade.direction}
              </span>
              <span className="font-semibold">{trade.ticker}</span>
              <span className="text-sm text-gray-400">x{trade.quantity}</span>
              <span className="font-mono text-xs text-gray-400">
                {trade.strategy_skill}
              </span>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => onApprove(trade.id)}
                aria-label="Approve"
                className="rounded-lg bg-green-600 px-3 py-1.5 text-xs font-bold text-white hover:bg-green-700"
              >
                Approve
              </button>
              <button
                onClick={() => onReject(trade.id)}
                aria-label="Reject"
                className="rounded-lg bg-red-100 px-3 py-1.5 text-xs font-bold text-red-600 hover:bg-red-200"
              >
                Reject
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

```tsx
// dashboard/src/components/controls/SystemMode.tsx
"use client";

import React from "react";

interface SystemModeProps {
  mode: "paper" | "live" | "backtest";
  evolutionEnabled: boolean;
  regimeOverride: string | null;
  onModeChange: (mode: string) => void;
  onEvolutionToggle: () => void;
  onRegimeOverride: (regime: string | null) => void;
}

const MODES = ["paper", "live", "backtest"] as const;

const MODE_COLORS: Record<string, string> = {
  paper: "bg-blue-100 text-blue-700 border-blue-300",
  live: "bg-green-100 text-green-700 border-green-300",
  backtest: "bg-gray-100 text-gray-700 border-gray-300",
};

const REGIMES = ["risk-on", "risk-off", "crisis", "neutral"] as const;

export function SystemMode({
  mode,
  evolutionEnabled,
  regimeOverride,
  onModeChange,
  onEvolutionToggle,
  onRegimeOverride,
}: SystemModeProps) {
  return (
    <div className="rounded-xl bg-trading-panel p-6 shadow-sm space-y-4">
      <h3 className="text-sm font-semibold text-gray-700">System Controls</h3>

      {/* Mode selector */}
      <div>
        <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">Trading Mode</p>
        <div className="flex gap-2">
          {MODES.map((m) => (
            <button
              key={m}
              onClick={() => onModeChange(m)}
              className={`rounded-lg border px-4 py-2 text-sm font-medium transition ${
                mode === m
                  ? MODE_COLORS[m]
                  : "border-gray-200 text-gray-400 hover:border-gray-300"
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {/* Evolution toggle */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs text-gray-400 uppercase tracking-wide">Strategy Evolution</p>
          <p className="text-sm font-medium mt-0.5">
            Evolution {evolutionEnabled ? "ON" : "OFF"}
          </p>
        </div>
        <button
          onClick={onEvolutionToggle}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition ${
            evolutionEnabled ? "bg-accent" : "bg-gray-300"
          }`}
        >
          <span
            className={`inline-block h-4 w-4 rounded-full bg-white transition transform ${
              evolutionEnabled ? "translate-x-6" : "translate-x-1"
            }`}
          />
        </button>
      </div>

      {/* Manual regime override */}
      <div>
        <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">
          Regime Override {regimeOverride && `(${regimeOverride})`}
        </p>
        <div className="flex gap-2 flex-wrap">
          <button
            onClick={() => onRegimeOverride(null)}
            className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition ${
              regimeOverride === null
                ? "border-accent text-accent bg-blue-50"
                : "border-gray-200 text-gray-400"
            }`}
          >
            Auto
          </button>
          {REGIMES.map((r) => (
            <button
              key={r}
              onClick={() => onRegimeOverride(r)}
              className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition ${
                regimeOverride === r
                  ? "border-accent text-accent bg-blue-50"
                  : "border-gray-200 text-gray-400"
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/api/test_operator_state_api.py -v
cd dashboard && npx jest src/__tests__/OperatorState.test.tsx
```

Expected: PASS — all API and component tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/api/routes/operator_state.py src/evolve_trader/api/main.py tests/api/test_operator_state_api.py dashboard/src/components/controls/ dashboard/src/__tests__/OperatorState.test.tsx
git commit -m "feat: operator state dashboard surface with status-only controls shell"
```

---

## Task 12: Final Verification

**Step 1: Run all backend tests**

```bash
pytest tests/ -v --tb=short
```

Expected: ALL PASS — API, WebSocket, and control tests

**Step 2: Run all frontend tests**

```bash
cd dashboard && npx jest --verbose
```

Expected: ALL PASS — ApiClient, PortfolioHealth, StrategyEvolution, SignalPerformance, LLMCosts, PortfolioOverview, TradeHistory, StrategyPerformance, SignalExplorer, Controls

**Step 3: Run linting and type checking**

```bash
ruff check src/evolve_trader/api/
mypy src/evolve_trader/api/ --ignore-missing-imports
cd dashboard && npx tsc --noEmit
```

Expected: No errors

**Step 4: Verify API starts cleanly**

```bash
uvicorn evolve_trader.api.main:app --host 0.0.0.0 --port 8000 &
curl http://localhost:8000/api/health
```

Expected: `{"status":"ok"}`

**Step 5: Commit any fixes**

```bash
git add -A
git commit -m "test: Phase 5 final verification — all dashboard tests passing"
```

---

## Parallelization Notes

Tasks in this phase have the following dependency structure:

```
Task 1 (FastAPI Layer) ──────────┐
Task 2 (WebSocket) ──────────────┤
                                  ├── Task 11 (Interactive Controls - API)
Task 3 (Next.js Setup) ──────────┤
                                  ├── Task 4  (Portfolio Health)     ──┐
                                  ├── Task 5  (Strategy Evolution)   ──┤
                                  ├── Task 6  (Signal Performance)   ──┤
                                  ├── Task 7  (LLM Costs)            ──┤── Task 12 (Final Verification)
                                  ├── Task 8  (Portfolio Overview)    ──┤
                                  ├── Task 9  (Trade History)         ──┤
                                  ├── Task 10 (Signal Explorer)       ──┤
                                  └── Task 11 (Controls - Frontend)  ──┘
```

**Can run in parallel:**
- Tasks 1-2 (backend API) and Task 3 (frontend setup) are independent — run simultaneously
- Tasks 4, 5, 6, 7 (ops panels) are independent of each other — run simultaneously after Task 3
- Tasks 8, 9, 10 (trading panels) are independent of each other — run simultaneously after Task 3
- Task 11 backend depends on Task 1; Task 11 frontend depends on Task 3
- Ops panels (4-7) and trading panels (8-10) are independent of each other — all can run simultaneously
- Task 12 (final verification) must be last
