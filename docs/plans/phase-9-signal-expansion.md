# Phase 9: Signal Expansion — Detailed Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand the signal layer with additional sources: ARK daily trades, NANC/GOP congressional ETFs, prediction markets (Polymarket + Kalshi), options unusual activity, on-chain whale tracking, institutional investor letters, and news/macro feeds. Build the automated signal source discovery engine with a two-stage coarse+fine filter pipeline.

**Architecture:** Each new signal source implements the existing `SignalSource` ABC and produces typed `SignalEvent` objects through the existing framework. Prediction markets introduce a new continuous-repricing paradigm (no traditional decay — always current). The discovery engine runs a two-stage pipeline: fast coarse filter (weekly, structured data) feeding a slower fine filter (LLM analysis) to maintain 50-100 active sources from a universe of 10,000+ candidates. Cross-platform prediction market consensus treats Polymarket-Kalshi divergence as an independent signal.

**Tech Stack:** Python 3.12+, polymarket-apis, httpx, ccxt, LiteLLM, PostgreSQL, pytest

**Parent plan:** [`2026-03-29-evolve-trader-master-plan.md`](2026-03-29-evolve-trader-master-plan.md)

**Prerequisites:** Phase 6 complete (paper trading provides live data to test against). Can run in parallel with Phase 7-8. Individual signal sources and discovery channels are independent of each other and can be built in parallel.

---

## Task 1: ARK Invest Daily Trades

**Files:**
- Create: `src/evolve_trader/signals/sources/ark_trades.py`
- Create: `tests/unit/test_ark_trades.py`
- Create: `tests/fixtures/ark/daily_trades_sample.csv`

**Step 1: Write the failing tests**

```python
# tests/unit/test_ark_trades.py
"""Tests for ARK Invest daily trade signal source."""
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from evolve_trader.signals.sources.ark_trades import (
    ArkTradesSource,
    parse_ark_daily_csv,
    ArkTrade,
    ARK_TRADE_DECAY,
)
from evolve_trader.signals.types import SignalType, DecayType


FIXTURES = Path(__file__).parent.parent / "fixtures" / "ark"


def test_ark_trade_decay_profile():
    """ARK decay: ~10-day half-life, fast exponential."""
    assert ARK_TRADE_DECAY.initial_confidence == pytest.approx(0.75, abs=0.05)
    assert ARK_TRADE_DECAY.half_life_days == 10
    assert ARK_TRADE_DECAY.decay_type == DecayType.EXPONENTIAL


def test_parse_ark_daily_csv():
    """Parser extracts trades from ARK daily CSV format."""
    csv_content = (FIXTURES / "daily_trades_sample.csv").read_text()
    trades = parse_ark_daily_csv(csv_content)
    assert len(trades) >= 1
    trade = trades[0]
    assert isinstance(trade, ArkTrade)
    assert trade.fund in ("ARKK", "ARKW", "ARKG", "ARKF", "ARKQ", "ARKX", "IZRL", "PRNT")
    assert trade.direction in ("Buy", "Sell")
    assert trade.ticker is not None
    assert trade.shares > 0
    assert trade.weight > 0


def test_ark_trade_has_company_and_cusip():
    """Each parsed trade includes company name and CUSIP."""
    csv_content = (FIXTURES / "daily_trades_sample.csv").read_text()
    trades = parse_ark_daily_csv(csv_content)
    for trade in trades:
        assert trade.company is not None
        assert len(trade.cusip) >= 6


def test_ark_source_produces_signal_events():
    """ArkTradesSource converts trades into SignalEvents."""
    source = ArkTradesSource()
    trade = ArkTrade(
        date=datetime(2026, 3, 15, tzinfo=timezone.utc),
        fund="ARKK",
        company="Tesla Inc",
        ticker="TSLA",
        cusip="88160R101",
        direction="Buy",
        shares=50000,
        weight=0.045,
    )
    events = source.trade_to_signals([trade])
    assert len(events) >= 1
    event = events[0]
    assert event.source == "ark_trades"
    assert event.source_entity == "ARK Invest"
    assert event.signal_type == SignalType.CONVICTION
    assert event.payload["ticker"] == "TSLA"
    assert event.payload["fund"] == "ARKK"
    assert event.payload["direction"] == "Buy"
    assert event.payload["shares"] == 50000


def test_ark_source_same_day_latency():
    """ARK trades have same-day latency — filing_date == trade_date."""
    source = ArkTradesSource()
    trade_date = datetime(2026, 3, 15, tzinfo=timezone.utc)
    trade = ArkTrade(
        date=trade_date,
        fund="ARKK",
        company="Tesla Inc",
        ticker="TSLA",
        cusip="88160R101",
        direction="Buy",
        shares=50000,
        weight=0.045,
    )
    events = source.trade_to_signals([trade])
    assert events[0].trade_date == trade_date
    assert events[0].filing_date == trade_date


def test_ark_multi_fund_same_ticker_amplifies():
    """Same ticker bought across multiple ARK funds → higher confidence."""
    source = ArkTradesSource()
    base_date = datetime(2026, 3, 15, tzinfo=timezone.utc)
    trades = [
        ArkTrade(date=base_date, fund="ARKK", company="Tesla Inc",
                 ticker="TSLA", cusip="88160R101", direction="Buy",
                 shares=50000, weight=0.045),
        ArkTrade(date=base_date, fund="ARKW", company="Tesla Inc",
                 ticker="TSLA", cusip="88160R101", direction="Buy",
                 shares=20000, weight=0.03),
    ]
    events = source.trade_to_signals(trades)
    tsla_events = [e for e in events if e.payload["ticker"] == "TSLA"]
    # Multi-fund conviction should be reflected
    assert any(e.confidence > ARK_TRADE_DECAY.initial_confidence for e in tsla_events) or \
        any("multi_fund" in e.metadata for e in tsla_events)


def test_ark_source_interface():
    """ArkTradesSource implements SignalSource ABC."""
    source = ArkTradesSource()
    assert source.source_name == "ark_trades"
    assert source.rate_limit_per_second > 0


def test_ark_expired_signal():
    """ARK signal expires after sufficient time."""
    source = ArkTradesSource()
    old_date = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trade = ArkTrade(
        date=old_date,
        fund="ARKK",
        company="Tesla Inc",
        ticker="TSLA",
        cusip="88160R101",
        direction="Buy",
        shares=50000,
        weight=0.045,
    )
    events = source.trade_to_signals([trade])
    # After 10 half-lives (~100 days), signal should be effectively expired
    assert events[0].is_expired(min_confidence=0.01)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_ark_trades.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.signals.sources.ark_trades'`

**Step 3: Create test fixture**

```csv
# tests/fixtures/ark/daily_trades_sample.csv
Date,Fund,Company,Ticker,CUSIP,Shares,Market Value ($),Weight (%)
03/15/2026,ARKK,TESLA INC,TSLA,88160R101,50000,8750000,4.50
03/15/2026,ARKK,ROKU INC,ROKU,77543R102,30000,2400000,1.23
03/15/2026,ARKW,TESLA INC,TSLA,88160R101,20000,3500000,3.00
```

**Step 4: Implement the ARK trades source**

```python
# src/evolve_trader/signals/sources/ark_trades.py
"""ARK Invest daily trade signal source.

Parses ARK's daily trade disclosure emails/API for same-day latency signals.
Decay: ~10 days, fast exponential — ARK trades are widely followed and
quickly priced in by the market.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from evolve_trader.signals.base import SignalSource
from evolve_trader.signals.events import SignalEvent
from evolve_trader.signals.types import DecayProfile, DecayType, SignalType


ARK_TRADE_DECAY = DecayProfile(
    initial_confidence=0.75,
    half_life_days=10,
    decay_type=DecayType.EXPONENTIAL,
)

ARK_FUNDS = {"ARKK", "ARKW", "ARKG", "ARKF", "ARKQ", "ARKX", "IZRL", "PRNT"}


@dataclass
class ArkTrade:
    """A single ARK fund trade from daily disclosure."""
    date: datetime
    fund: str
    company: str
    ticker: str
    cusip: str
    direction: str  # "Buy" or "Sell"
    shares: int
    weight: float  # portfolio weight percentage


def parse_ark_daily_csv(csv_content: str) -> list[ArkTrade]:
    """Parse ARK daily trades CSV into structured trades."""
    trades = []
    reader = csv.DictReader(io.StringIO(csv_content))
    for row in reader:
        trades.append(ArkTrade(
            date=datetime.strptime(row["Date"].strip(), "%m/%d/%Y").replace(
                tzinfo=timezone.utc
            ),
            fund=row["Fund"].strip(),
            company=row["Company"].strip(),
            ticker=row["Ticker"].strip(),
            cusip=row["CUSIP"].strip(),
            direction="Buy" if float(row["Shares"]) > 0 else "Sell",
            shares=abs(int(float(row["Shares"]))),
            weight=float(row["Weight (%)"].strip()),
        ))
    return trades


class ArkTradesSource(SignalSource):
    """Signal source for ARK Invest daily trade disclosures."""

    @property
    def source_name(self) -> str:
        return "ark_trades"

    @property
    def rate_limit_per_second(self) -> float:
        return 2.0

    def trade_to_signals(self, trades: list[ArkTrade]) -> list[SignalEvent]:
        """Convert parsed ARK trades into SignalEvents.

        Multi-fund buys of the same ticker on the same day are flagged
        as higher-conviction signals.
        """
        # Group by (date, ticker) to detect multi-fund conviction
        from collections import defaultdict
        ticker_groups: dict[tuple[str, str], list[ArkTrade]] = defaultdict(list)
        for trade in trades:
            key = (trade.date.isoformat(), trade.ticker)
            ticker_groups[key].append(trade)

        events = []
        for (_date_str, ticker), group in ticker_groups.items():
            multi_fund = len(set(t.fund for t in group)) > 1
            confidence = ARK_TRADE_DECAY.initial_confidence
            if multi_fund:
                confidence = min(confidence * 1.15, 0.95)

            for trade in group:
                events.append(SignalEvent(
                    source="ark_trades",
                    source_entity="ARK Invest",
                    timestamp=trade.date,
                    trade_date=trade.date,
                    filing_date=trade.date,  # same-day disclosure
                    confidence=confidence,
                    decay_profile=ARK_TRADE_DECAY,
                    signal_type=SignalType.CONVICTION,
                    payload={
                        "ticker": trade.ticker,
                        "fund": trade.fund,
                        "company": trade.company,
                        "cusip": trade.cusip,
                        "direction": trade.direction,
                        "shares": trade.shares,
                        "portfolio_weight": trade.weight,
                    },
                    metadata={
                        "multi_fund": multi_fund,
                        "funds_trading": [t.fund for t in group],
                    },
                ))
        return events

    async def fetch_signals(self) -> list[SignalEvent]:
        """Fetch latest ARK daily trades."""
        # TODO: Implement ARK API / email polling
        return []

    async def validate_schema(self, response: dict) -> bool:
        """Validate ARK API response schema."""
        return "trades" in response or "date" in response
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/test_ark_trades.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/evolve_trader/signals/sources/ark_trades.py tests/unit/test_ark_trades.py tests/fixtures/ark/
git commit -m "feat: ARK Invest daily trade signal source with multi-fund conviction detection"
```

---

## Task 2: NANC/GOP Congressional ETFs

**Files:**
- Create: `src/evolve_trader/signals/sources/congressional_etfs.py`
- Create: `tests/unit/test_congressional_etfs.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_congressional_etfs.py
"""Tests for NANC/GOP congressional ETF signal source."""
import pytest
from datetime import datetime, timedelta, timezone
from evolve_trader.signals.sources.congressional_etfs import (
    CongressionalEtfSource,
    EtfSnapshot,
    PartisanDifferential,
    CONGRESSIONAL_ETF_DECAY,
    compute_partisan_differential,
)
from evolve_trader.signals.types import SignalType, DecayType


def test_congressional_etf_decay_profile():
    """Congressional ETFs: ~15-day half-life, moderate exponential."""
    assert CONGRESSIONAL_ETF_DECAY.half_life_days == 15
    assert CONGRESSIONAL_ETF_DECAY.decay_type == DecayType.EXPONENTIAL


def test_etf_snapshot_has_required_fields():
    """EtfSnapshot captures daily NAV data for NANC and GOP."""
    snap = EtfSnapshot(
        date=datetime(2026, 3, 15, tzinfo=timezone.utc),
        ticker="NANC",
        nav=32.50,
        close=32.45,
        volume=150000,
        total_return_ytd=0.208,
    )
    assert snap.ticker == "NANC"
    assert snap.nav == 32.50


def test_compute_partisan_differential():
    """Partisan differential = NANC return - GOP return."""
    nanc = EtfSnapshot(
        date=datetime(2026, 3, 15, tzinfo=timezone.utc),
        ticker="NANC", nav=32.50, close=32.45, volume=150000,
        total_return_ytd=0.208,
    )
    gop = EtfSnapshot(
        date=datetime(2026, 3, 15, tzinfo=timezone.utc),
        ticker="KRUZ", nav=28.10, close=28.05, volume=120000,
        total_return_ytd=0.188,
    )
    diff = compute_partisan_differential(nanc, gop)
    assert isinstance(diff, PartisanDifferential)
    assert diff.spread == pytest.approx(0.02, abs=0.001)
    assert diff.leading_party in ("D", "R")


def test_source_produces_benchmark_signal():
    """CongressionalEtfSource produces benchmark comparison signals."""
    source = CongressionalEtfSource()
    nanc = EtfSnapshot(
        date=datetime(2026, 3, 15, tzinfo=timezone.utc),
        ticker="NANC", nav=32.50, close=32.45, volume=150000,
        total_return_ytd=0.208,
    )
    gop = EtfSnapshot(
        date=datetime(2026, 3, 15, tzinfo=timezone.utc),
        ticker="KRUZ", nav=28.10, close=28.05, volume=120000,
        total_return_ytd=0.188,
    )
    events = source.snapshots_to_signals(nanc, gop)
    assert len(events) >= 1
    assert events[0].source == "congressional_etfs"
    assert events[0].signal_type == SignalType.REGIME_READ
    assert "partisan_spread" in events[0].payload


def test_rebalancing_date_detection():
    """Detects rebalancing dates from volume spikes."""
    source = CongressionalEtfSource()
    snapshots = [
        EtfSnapshot(date=datetime(2026, 3, i, tzinfo=timezone.utc),
                     ticker="NANC", nav=32.0 + i * 0.1, close=32.0 + i * 0.1,
                     volume=100000 if i != 15 else 500000,
                     total_return_ytd=0.10)
        for i in range(1, 21)
    ]
    rebalance_dates = source.detect_rebalancing(snapshots)
    assert datetime(2026, 3, 15, tzinfo=timezone.utc) in rebalance_dates


def test_source_interface():
    """CongressionalEtfSource implements SignalSource ABC."""
    source = CongressionalEtfSource()
    assert source.source_name == "congressional_etfs"
    assert source.rate_limit_per_second > 0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_congressional_etfs.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.signals.sources.congressional_etfs'`

**Step 3: Implement the congressional ETF source**

```python
# src/evolve_trader/signals/sources/congressional_etfs.py
"""NANC/GOP (KRUZ) congressional ETF signal source.

Daily NAV tracking of Democratic (NANC) and Republican (KRUZ/GOP) congressional
portfolio ETFs from Unusual Whales. Provides:
  (a) benchmark for our own congressional signal performance
  (b) partisan differential as a sentiment signal
  (c) rebalancing date patterns exploitable by the incubator
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean, stdev

from evolve_trader.signals.base import SignalSource
from evolve_trader.signals.events import SignalEvent
from evolve_trader.signals.types import DecayProfile, DecayType, SignalType


CONGRESSIONAL_ETF_DECAY = DecayProfile(
    initial_confidence=0.60,
    half_life_days=15,
    decay_type=DecayType.EXPONENTIAL,
)

VOLUME_SPIKE_THRESHOLD = 3.0  # standard deviations above mean


@dataclass
class EtfSnapshot:
    """Daily snapshot for a congressional ETF."""
    date: datetime
    ticker: str  # NANC or KRUZ
    nav: float
    close: float
    volume: int
    total_return_ytd: float


@dataclass
class PartisanDifferential:
    """Difference in performance between NANC and GOP ETFs."""
    date: datetime
    nanc_return: float
    gop_return: float
    spread: float  # nanc - gop
    leading_party: str  # "D" or "R"


def compute_partisan_differential(
    nanc: EtfSnapshot, gop: EtfSnapshot
) -> PartisanDifferential:
    """Compute the partisan performance differential."""
    spread = nanc.total_return_ytd - gop.total_return_ytd
    return PartisanDifferential(
        date=nanc.date,
        nanc_return=nanc.total_return_ytd,
        gop_return=gop.total_return_ytd,
        spread=spread,
        leading_party="D" if spread > 0 else "R",
    )


class CongressionalEtfSource(SignalSource):
    """Signal source for NANC/GOP congressional ETFs."""

    @property
    def source_name(self) -> str:
        return "congressional_etfs"

    @property
    def rate_limit_per_second(self) -> float:
        return 5.0  # standard market data rate

    def snapshots_to_signals(
        self, nanc: EtfSnapshot, gop: EtfSnapshot
    ) -> list[SignalEvent]:
        """Convert daily ETF snapshots into SignalEvents."""
        diff = compute_partisan_differential(nanc, gop)
        return [
            SignalEvent(
                source="congressional_etfs",
                source_entity="Unusual Whales ETFs",
                timestamp=nanc.date,
                trade_date=nanc.date,
                filing_date=nanc.date,
                confidence=CONGRESSIONAL_ETF_DECAY.initial_confidence,
                decay_profile=CONGRESSIONAL_ETF_DECAY,
                signal_type=SignalType.REGIME_READ,
                payload={
                    "nanc_return_ytd": diff.nanc_return,
                    "gop_return_ytd": diff.gop_return,
                    "partisan_spread": diff.spread,
                    "leading_party": diff.leading_party,
                    "nanc_nav": nanc.nav,
                    "gop_nav": gop.nav,
                },
                metadata={
                    "nanc_volume": nanc.volume,
                    "gop_volume": gop.volume,
                },
            )
        ]

    def detect_rebalancing(
        self, snapshots: list[EtfSnapshot]
    ) -> list[datetime]:
        """Detect rebalancing dates from volume spikes."""
        if len(snapshots) < 5:
            return []
        volumes = [s.volume for s in snapshots]
        avg = mean(volumes)
        sd = stdev(volumes) if len(volumes) > 1 else 1.0
        return [
            s.date for s in snapshots
            if (s.volume - avg) / sd > VOLUME_SPIKE_THRESHOLD
        ]

    async def fetch_signals(self) -> list[SignalEvent]:
        """Fetch latest NANC/GOP ETF data."""
        # TODO: Implement market data feed polling
        return []

    async def validate_schema(self, response: dict) -> bool:
        """Validate market data response schema."""
        return "nav" in response or "close" in response
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_congressional_etfs.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/signals/sources/congressional_etfs.py tests/unit/test_congressional_etfs.py
git commit -m "feat: NANC/GOP congressional ETF signal source with partisan differential"
```

---

## Task 3: Polymarket Integration

**Files:**
- Create: `src/evolve_trader/signals/sources/polymarket.py`
- Create: `tests/unit/test_polymarket.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_polymarket.py
"""Tests for Polymarket prediction market signal source."""
import pytest
from datetime import datetime, timezone
from evolve_trader.signals.sources.polymarket import (
    PolymarketSource,
    PolymarketMarket,
    PolymarketCategory,
    whale_tracker,
)
from evolve_trader.signals.types import SignalType


def test_polymarket_categories():
    """All required prediction market categories are defined."""
    assert PolymarketCategory.MONETARY_POLICY.value == "monetary_policy"
    assert PolymarketCategory.RECESSION.value == "recession"
    assert PolymarketCategory.TARIFFS.value == "tariffs"
    assert PolymarketCategory.GEOPOLITICAL.value == "geopolitical"
    assert PolymarketCategory.ELECTIONS.value == "elections"
    assert PolymarketCategory.CRYPTO.value == "crypto"


def test_polymarket_market_has_required_fields():
    """PolymarketMarket captures CLOB market state."""
    market = PolymarketMarket(
        condition_id="0xabc123",
        question="Will the Fed cut rates in June 2026?",
        category=PolymarketCategory.MONETARY_POLICY,
        outcome_yes_price=0.72,
        outcome_no_price=0.28,
        volume_24h=1_500_000.0,
        liquidity=3_200_000.0,
        end_date=datetime(2026, 6, 30, tzinfo=timezone.utc),
        last_updated=datetime(2026, 3, 15, tzinfo=timezone.utc),
    )
    assert market.outcome_yes_price == 0.72
    assert market.category == PolymarketCategory.MONETARY_POLICY


def test_polymarket_source_produces_signals():
    """PolymarketSource converts markets into SignalEvents."""
    source = PolymarketSource()
    market = PolymarketMarket(
        condition_id="0xabc123",
        question="Will the Fed cut rates in June 2026?",
        category=PolymarketCategory.MONETARY_POLICY,
        outcome_yes_price=0.72,
        outcome_no_price=0.28,
        volume_24h=1_500_000.0,
        liquidity=3_200_000.0,
        end_date=datetime(2026, 6, 30, tzinfo=timezone.utc),
        last_updated=datetime(2026, 3, 15, tzinfo=timezone.utc),
    )
    events = source.market_to_signals([market])
    assert len(events) >= 1
    event = events[0]
    assert event.source == "polymarket"
    assert event.signal_type == SignalType.REGIME_READ
    assert event.payload["yes_price"] == 0.72
    assert event.payload["category"] == "monetary_policy"


def test_polymarket_no_traditional_decay():
    """Prediction market signals are always current — no traditional decay.

    Confidence is the market price itself, not a decaying initial value.
    """
    source = PolymarketSource()
    market = PolymarketMarket(
        condition_id="0xabc123",
        question="Recession in 2026?",
        category=PolymarketCategory.RECESSION,
        outcome_yes_price=0.35,
        outcome_no_price=0.65,
        volume_24h=2_000_000.0,
        liquidity=5_000_000.0,
        end_date=datetime(2026, 12, 31, tzinfo=timezone.utc),
        last_updated=datetime(2026, 3, 15, tzinfo=timezone.utc),
    )
    events = source.market_to_signals([market])
    # Confidence should reflect market price, not a static initial value
    assert events[0].confidence == pytest.approx(0.35, abs=0.05)


def test_polymarket_high_volume_filter():
    """Low-volume markets are filtered or flagged as low-confidence."""
    source = PolymarketSource()
    low_vol = PolymarketMarket(
        condition_id="0xlow",
        question="Will X happen?",
        category=PolymarketCategory.GEOPOLITICAL,
        outcome_yes_price=0.50,
        outcome_no_price=0.50,
        volume_24h=500.0,  # very low
        liquidity=1000.0,
        end_date=datetime(2026, 12, 31, tzinfo=timezone.utc),
        last_updated=datetime(2026, 3, 15, tzinfo=timezone.utc),
    )
    events = source.market_to_signals([low_vol])
    if events:
        assert events[0].confidence < 0.3  # very low confidence for thin markets


def test_whale_tracker_detects_large_positions():
    """Whale tracker identifies large position changes."""
    positions = [
        {"address": "0xwhale1", "market_id": "0xabc", "size": 500_000, "side": "YES"},
        {"address": "0xwhale2", "market_id": "0xabc", "size": 1_000, "side": "NO"},
    ]
    whales = whale_tracker(positions, threshold=100_000)
    assert len(whales) == 1
    assert whales[0]["address"] == "0xwhale1"


def test_polymarket_source_interface():
    """PolymarketSource implements SignalSource ABC."""
    source = PolymarketSource()
    assert source.source_name == "polymarket"
    assert source.rate_limit_per_second > 0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_polymarket.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.signals.sources.polymarket'`

**Step 3: Implement the Polymarket source**

```python
# src/evolve_trader/signals/sources/polymarket.py
"""Polymarket prediction market signal source.

Fastest signal layer — reprices within minutes of breaking news.
No traditional decay; confidence IS the market price.
Categories: monetary policy, recession, tariffs, geopolitical, elections, crypto.
Includes whale tracking via position size analysis.

API: polymarket-apis PyPI package (REST, WebSocket, CLOB, Pydantic validation).
On-chain data via Bitquery GraphQL on Polygon for ultra-low-latency.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from evolve_trader.signals.base import SignalSource
from evolve_trader.signals.events import SignalEvent
from evolve_trader.signals.types import DecayProfile, DecayType, SignalType


# Prediction markets use a very long half-life since they self-update;
# the "decay" is irrelevant because each poll replaces the previous signal.
POLYMARKET_DECAY = DecayProfile(
    initial_confidence=0.50,  # overridden per-market by actual price
    half_life_days=365,  # effectively no decay — always replaced by fresh data
    decay_type=DecayType.LINEAR,
)

MIN_VOLUME_24H = 10_000.0  # minimum 24h volume for signal inclusion
MIN_LIQUIDITY = 50_000.0


class PolymarketCategory(str, Enum):
    """Categories of prediction markets tracked."""
    MONETARY_POLICY = "monetary_policy"
    RECESSION = "recession"
    TARIFFS = "tariffs"
    GEOPOLITICAL = "geopolitical"
    ELECTIONS = "elections"
    CRYPTO = "crypto"


@dataclass
class PolymarketMarket:
    """State of a single Polymarket CLOB market."""
    condition_id: str
    question: str
    category: PolymarketCategory
    outcome_yes_price: float  # 0.0 - 1.0
    outcome_no_price: float
    volume_24h: float
    liquidity: float
    end_date: datetime
    last_updated: datetime


def whale_tracker(
    positions: list[dict[str, Any]], threshold: float = 100_000
) -> list[dict[str, Any]]:
    """Identify whale-sized positions exceeding threshold."""
    return [p for p in positions if p.get("size", 0) >= threshold]


class PolymarketSource(SignalSource):
    """Signal source for Polymarket prediction markets."""

    @property
    def source_name(self) -> str:
        return "polymarket"

    @property
    def rate_limit_per_second(self) -> float:
        return 5.0

    def market_to_signals(
        self, markets: list[PolymarketMarket]
    ) -> list[SignalEvent]:
        """Convert prediction market states into SignalEvents.

        Confidence is the market price itself — no static decay.
        Low-volume markets are filtered or assigned very low confidence.
        """
        events = []
        for market in markets:
            # Filter thin markets
            if market.volume_24h < MIN_VOLUME_24H or market.liquidity < MIN_LIQUIDITY:
                confidence = market.outcome_yes_price * 0.3  # heavy discount
            else:
                confidence = market.outcome_yes_price

            events.append(SignalEvent(
                source="polymarket",
                source_entity="Polymarket",
                timestamp=market.last_updated,
                trade_date=market.last_updated,
                filing_date=market.last_updated,
                confidence=confidence,
                decay_profile=POLYMARKET_DECAY,
                signal_type=SignalType.REGIME_READ,
                payload={
                    "condition_id": market.condition_id,
                    "question": market.question,
                    "category": market.category.value,
                    "yes_price": market.outcome_yes_price,
                    "no_price": market.outcome_no_price,
                    "volume_24h": market.volume_24h,
                    "liquidity": market.liquidity,
                },
                metadata={
                    "end_date": market.end_date.isoformat(),
                },
            ))
        return events

    async def fetch_signals(self) -> list[SignalEvent]:
        """Fetch latest Polymarket data via polymarket-apis."""
        # TODO: Implement polymarket-apis polling + WebSocket
        return []

    async def validate_schema(self, response: dict) -> bool:
        """Validate Polymarket API response schema."""
        return "markets" in response or "condition_id" in response
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_polymarket.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/signals/sources/polymarket.py tests/unit/test_polymarket.py
git commit -m "feat: Polymarket prediction market signal source with whale tracking"
```

---

## Task 4: Kalshi Integration

**Files:**
- Create: `src/evolve_trader/signals/sources/kalshi.py`
- Create: `tests/unit/test_kalshi.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_kalshi.py
"""Tests for Kalshi prediction market signal source."""
import pytest
from datetime import datetime, timezone
from evolve_trader.signals.sources.kalshi import (
    KalshiSource,
    KalshiMarket,
    KalshiCategory,
    KALSHI_DECAY,
)
from evolve_trader.signals.types import SignalType


def test_kalshi_categories_match_polymarket():
    """Kalshi categories align with Polymarket for cross-platform comparison."""
    assert KalshiCategory.MONETARY_POLICY.value == "monetary_policy"
    assert KalshiCategory.RECESSION.value == "recession"
    assert KalshiCategory.TARIFFS.value == "tariffs"
    assert KalshiCategory.GEOPOLITICAL.value == "geopolitical"
    assert KalshiCategory.ELECTIONS.value == "elections"


def test_kalshi_market_has_required_fields():
    """KalshiMarket captures CFTC-regulated market state."""
    market = KalshiMarket(
        ticker="FED-26JUN-T4.75",
        title="Fed funds rate above 4.75% on June 2026 meeting",
        category=KalshiCategory.MONETARY_POLICY,
        yes_price=0.68,
        no_price=0.32,
        volume=85000,
        open_interest=120000,
        expiration=datetime(2026, 6, 30, tzinfo=timezone.utc),
        last_updated=datetime(2026, 3, 15, tzinfo=timezone.utc),
    )
    assert market.yes_price == 0.68
    assert market.open_interest == 120000


def test_kalshi_source_produces_signals():
    """KalshiSource converts markets into SignalEvents."""
    source = KalshiSource()
    market = KalshiMarket(
        ticker="FED-26JUN-T4.75",
        title="Fed funds rate above 4.75% on June 2026 meeting",
        category=KalshiCategory.MONETARY_POLICY,
        yes_price=0.68,
        no_price=0.32,
        volume=85000,
        open_interest=120000,
        expiration=datetime(2026, 6, 30, tzinfo=timezone.utc),
        last_updated=datetime(2026, 3, 15, tzinfo=timezone.utc),
    )
    events = source.market_to_signals([market])
    assert len(events) >= 1
    event = events[0]
    assert event.source == "kalshi"
    assert event.signal_type == SignalType.REGIME_READ
    assert event.payload["yes_price"] == 0.68
    assert "open_interest" in event.payload


def test_kalshi_cftc_regulated_flag():
    """Kalshi signals are flagged as CFTC-regulated."""
    source = KalshiSource()
    market = KalshiMarket(
        ticker="FED-26JUN-T4.75",
        title="Test market",
        category=KalshiCategory.MONETARY_POLICY,
        yes_price=0.50,
        no_price=0.50,
        volume=50000,
        open_interest=80000,
        expiration=datetime(2026, 12, 31, tzinfo=timezone.utc),
        last_updated=datetime(2026, 3, 15, tzinfo=timezone.utc),
    )
    events = source.market_to_signals([market])
    assert events[0].metadata.get("regulated") is True


def test_kalshi_source_interface():
    """KalshiSource implements SignalSource ABC."""
    source = KalshiSource()
    assert source.source_name == "kalshi"
    assert source.rate_limit_per_second > 0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_kalshi.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.signals.sources.kalshi'`

**Step 3: Implement the Kalshi source**

```python
# src/evolve_trader/signals/sources/kalshi.py
"""Kalshi prediction market signal source.

CFTC-regulated prediction market. Cross-platform consensus with Polymarket:
divergence between platforms is itself informative (regulatory risk,
information asymmetry, liquidity differences).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from evolve_trader.signals.base import SignalSource
from evolve_trader.signals.events import SignalEvent
from evolve_trader.signals.types import DecayProfile, DecayType, SignalType


KALSHI_DECAY = DecayProfile(
    initial_confidence=0.50,
    half_life_days=365,
    decay_type=DecayType.LINEAR,
)


class KalshiCategory(str, Enum):
    """Kalshi market categories — aligned with Polymarket for cross-platform comparison."""
    MONETARY_POLICY = "monetary_policy"
    RECESSION = "recession"
    TARIFFS = "tariffs"
    GEOPOLITICAL = "geopolitical"
    ELECTIONS = "elections"


@dataclass
class KalshiMarket:
    """State of a single Kalshi event contract."""
    ticker: str
    title: str
    category: KalshiCategory
    yes_price: float
    no_price: float
    volume: int
    open_interest: int
    expiration: datetime
    last_updated: datetime


class KalshiSource(SignalSource):
    """Signal source for Kalshi CFTC-regulated prediction markets."""

    @property
    def source_name(self) -> str:
        return "kalshi"

    @property
    def rate_limit_per_second(self) -> float:
        return 5.0

    def market_to_signals(
        self, markets: list[KalshiMarket]
    ) -> list[SignalEvent]:
        """Convert Kalshi market states into SignalEvents."""
        events = []
        for market in markets:
            events.append(SignalEvent(
                source="kalshi",
                source_entity="Kalshi",
                timestamp=market.last_updated,
                trade_date=market.last_updated,
                filing_date=market.last_updated,
                confidence=market.yes_price,
                decay_profile=KALSHI_DECAY,
                signal_type=SignalType.REGIME_READ,
                payload={
                    "ticker": market.ticker,
                    "title": market.title,
                    "category": market.category.value,
                    "yes_price": market.yes_price,
                    "no_price": market.no_price,
                    "volume": market.volume,
                    "open_interest": market.open_interest,
                },
                metadata={
                    "regulated": True,
                    "regulator": "CFTC",
                    "expiration": market.expiration.isoformat(),
                },
            ))
        return events

    async def fetch_signals(self) -> list[SignalEvent]:
        """Fetch latest Kalshi market data."""
        # TODO: Implement Kalshi REST API polling
        return []

    async def validate_schema(self, response: dict) -> bool:
        """Validate Kalshi API response schema."""
        return "markets" in response or "ticker" in response
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_kalshi.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/signals/sources/kalshi.py tests/unit/test_kalshi.py
git commit -m "feat: Kalshi CFTC-regulated prediction market signal source"
```

---

## Task 5: Cross-Platform Prediction Consensus

**Files:**
- Create: `src/evolve_trader/signals/sources/prediction_consensus.py`
- Create: `tests/unit/test_prediction_consensus.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_prediction_consensus.py
"""Tests for cross-platform prediction market consensus engine."""
import pytest
from datetime import datetime, timezone
from evolve_trader.signals.sources.prediction_consensus import (
    PredictionConsensus,
    PlatformPrice,
    ConsensusResult,
    compute_consensus,
    detect_divergence,
    DIVERGENCE_THRESHOLD,
)
from evolve_trader.signals.types import SignalType


def test_consensus_aligned_platforms():
    """When Polymarket and Kalshi agree, consensus is high-confidence."""
    poly = PlatformPrice(platform="polymarket", price=0.72, volume=1_500_000)
    kalshi = PlatformPrice(platform="kalshi", price=0.70, volume=85_000)
    result = compute_consensus([poly, kalshi])
    assert isinstance(result, ConsensusResult)
    assert result.consensus_price == pytest.approx(0.71, abs=0.02)
    assert result.divergence < DIVERGENCE_THRESHOLD
    assert result.aligned is True


def test_consensus_divergent_platforms():
    """When platforms diverge beyond threshold, divergence signal fires."""
    poly = PlatformPrice(platform="polymarket", price=0.80, volume=1_500_000)
    kalshi = PlatformPrice(platform="kalshi", price=0.55, volume=85_000)
    result = compute_consensus([poly, kalshi])
    assert result.divergence >= DIVERGENCE_THRESHOLD
    assert result.aligned is False


def test_detect_divergence_returns_events():
    """Divergence detection produces EVENT_DRIVEN signals."""
    poly = PlatformPrice(platform="polymarket", price=0.80, volume=1_500_000)
    kalshi = PlatformPrice(platform="kalshi", price=0.55, volume=85_000)
    events = detect_divergence(
        topic="Fed rate cut June 2026",
        category="monetary_policy",
        prices=[poly, kalshi],
        timestamp=datetime(2026, 3, 15, tzinfo=timezone.utc),
    )
    assert len(events) >= 1
    assert events[0].source == "prediction_consensus"
    assert events[0].signal_type == SignalType.EVENT_DRIVEN
    assert "divergence" in events[0].payload
    assert events[0].payload["divergence"] >= DIVERGENCE_THRESHOLD


def test_no_divergence_signal_when_aligned():
    """No divergence signal when platforms agree."""
    poly = PlatformPrice(platform="polymarket", price=0.72, volume=1_500_000)
    kalshi = PlatformPrice(platform="kalshi", price=0.70, volume=85_000)
    events = detect_divergence(
        topic="Fed rate cut June 2026",
        category="monetary_policy",
        prices=[poly, kalshi],
        timestamp=datetime(2026, 3, 15, tzinfo=timezone.utc),
    )
    assert len(events) == 0


def test_volume_weighted_consensus():
    """Consensus price is volume-weighted across platforms."""
    poly = PlatformPrice(platform="polymarket", price=0.80, volume=10_000_000)
    kalshi = PlatformPrice(platform="kalshi", price=0.60, volume=100_000)
    result = compute_consensus([poly, kalshi])
    # Should be much closer to polymarket price (higher volume)
    assert result.consensus_price > 0.75


def test_prediction_consensus_source_interface():
    """PredictionConsensus implements SignalSource ABC."""
    source = PredictionConsensus()
    assert source.source_name == "prediction_consensus"
    assert source.rate_limit_per_second > 0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_prediction_consensus.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.signals.sources.prediction_consensus'`

**Step 3: Implement the consensus engine**

```python
# src/evolve_trader/signals/sources/prediction_consensus.py
"""Cross-platform prediction market consensus engine.

Polymarket (crypto-native, global) vs Kalshi (CFTC-regulated, US).
Divergence between platforms is itself informative:
  - Regulatory risk differential
  - Information asymmetry
  - Liquidity differences
  - Geographic/demographic skew
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from evolve_trader.signals.base import SignalSource
from evolve_trader.signals.events import SignalEvent
from evolve_trader.signals.types import DecayProfile, DecayType, SignalType


DIVERGENCE_THRESHOLD = 0.10  # 10 percentage points = meaningful divergence

CONSENSUS_DECAY = DecayProfile(
    initial_confidence=0.80,
    half_life_days=3,
    decay_type=DecayType.EXPONENTIAL,
)


@dataclass
class PlatformPrice:
    """Price from a single prediction market platform."""
    platform: str  # "polymarket" or "kalshi"
    price: float  # 0.0 - 1.0
    volume: float


@dataclass
class ConsensusResult:
    """Result of cross-platform consensus computation."""
    consensus_price: float
    divergence: float
    aligned: bool
    platforms: list[PlatformPrice]


def compute_consensus(prices: list[PlatformPrice]) -> ConsensusResult:
    """Compute volume-weighted consensus across platforms."""
    total_volume = sum(p.volume for p in prices)
    if total_volume == 0:
        avg = sum(p.price for p in prices) / len(prices)
        return ConsensusResult(
            consensus_price=avg,
            divergence=max(p.price for p in prices) - min(p.price for p in prices),
            aligned=True,
            platforms=prices,
        )

    weighted_price = sum(p.price * p.volume for p in prices) / total_volume
    divergence = max(p.price for p in prices) - min(p.price for p in prices)

    return ConsensusResult(
        consensus_price=weighted_price,
        divergence=divergence,
        aligned=divergence < DIVERGENCE_THRESHOLD,
        platforms=prices,
    )


def detect_divergence(
    topic: str,
    category: str,
    prices: list[PlatformPrice],
    timestamp: datetime,
) -> list[SignalEvent]:
    """Detect cross-platform divergence and produce EVENT_DRIVEN signals."""
    result = compute_consensus(prices)
    if result.aligned:
        return []

    return [
        SignalEvent(
            source="prediction_consensus",
            source_entity="Cross-Platform Consensus",
            timestamp=timestamp,
            trade_date=timestamp,
            filing_date=timestamp,
            confidence=CONSENSUS_DECAY.initial_confidence,
            decay_profile=CONSENSUS_DECAY,
            signal_type=SignalType.EVENT_DRIVEN,
            payload={
                "topic": topic,
                "category": category,
                "consensus_price": result.consensus_price,
                "divergence": result.divergence,
                "platforms": {
                    p.platform: {"price": p.price, "volume": p.volume}
                    for p in prices
                },
            },
            metadata={
                "threshold": DIVERGENCE_THRESHOLD,
                "aligned": result.aligned,
            },
        )
    ]


class PredictionConsensus(SignalSource):
    """Signal source for cross-platform prediction market consensus."""

    @property
    def source_name(self) -> str:
        return "prediction_consensus"

    @property
    def rate_limit_per_second(self) -> float:
        return 10.0

    async def fetch_signals(self) -> list[SignalEvent]:
        """Fetch and compare Polymarket vs Kalshi for tracked topics."""
        # TODO: Poll both platforms, match topics, compute divergence
        return []

    async def validate_schema(self, response: dict) -> bool:
        """Validate cross-platform response."""
        return "topic" in response or "prices" in response
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_prediction_consensus.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/signals/sources/prediction_consensus.py tests/unit/test_prediction_consensus.py
git commit -m "feat: cross-platform prediction market consensus with divergence detection"
```

---

## Task 6: Options Unusual Activity

**Files:**
- Create: `src/evolve_trader/signals/sources/options_activity.py`
- Create: `tests/unit/test_options_activity.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_options_activity.py
"""Tests for options unusual activity signal source."""
import pytest
from datetime import datetime, timezone
from evolve_trader.signals.sources.options_activity import (
    OptionsActivitySource,
    OptionsAlert,
    OptionsAlertType,
    OPTIONS_ACTIVITY_DECAY,
    detect_strike_clustering,
)
from evolve_trader.signals.types import SignalType, DecayType


def test_options_decay_profile():
    """Options activity: ~2-day half-life, very fast exponential."""
    assert OPTIONS_ACTIVITY_DECAY.half_life_days == 2
    assert OPTIONS_ACTIVITY_DECAY.decay_type == DecayType.EXPONENTIAL
    assert OPTIONS_ACTIVITY_DECAY.initial_confidence >= 0.80


def test_options_alert_types():
    """Alert types cover unusual volume, put/call shifts, strike clustering."""
    assert OptionsAlertType.UNUSUAL_VOLUME.value == "unusual_volume"
    assert OptionsAlertType.PUT_CALL_SHIFT.value == "put_call_shift"
    assert OptionsAlertType.STRIKE_CLUSTER.value == "strike_cluster"
    assert OptionsAlertType.LARGE_BLOCK.value == "large_block"


def test_options_alert_has_required_fields():
    """OptionsAlert captures pre-event positioning data."""
    alert = OptionsAlert(
        timestamp=datetime(2026, 3, 15, tzinfo=timezone.utc),
        ticker="AAPL",
        alert_type=OptionsAlertType.UNUSUAL_VOLUME,
        put_call_ratio=0.3,
        volume=50000,
        avg_volume=10000,
        volume_ratio=5.0,
        strike=200.0,
        expiry=datetime(2026, 4, 17, tzinfo=timezone.utc),
        premium=2_500_000.0,
    )
    assert alert.volume_ratio == 5.0
    assert alert.premium == 2_500_000.0


def test_source_produces_conviction_signals():
    """OptionsActivitySource converts alerts into CONVICTION SignalEvents."""
    source = OptionsActivitySource()
    alert = OptionsAlert(
        timestamp=datetime(2026, 3, 15, tzinfo=timezone.utc),
        ticker="AAPL",
        alert_type=OptionsAlertType.UNUSUAL_VOLUME,
        put_call_ratio=0.3,
        volume=50000,
        avg_volume=10000,
        volume_ratio=5.0,
        strike=200.0,
        expiry=datetime(2026, 4, 17, tzinfo=timezone.utc),
        premium=2_500_000.0,
    )
    events = source.alert_to_signals([alert])
    assert len(events) >= 1
    event = events[0]
    assert event.source == "options_activity"
    assert event.signal_type == SignalType.CONVICTION
    assert event.payload["ticker"] == "AAPL"
    assert event.payload["volume_ratio"] == 5.0


def test_detect_strike_clustering():
    """Strike clustering detects concentrated positioning at similar strikes."""
    alerts = [
        OptionsAlert(
            timestamp=datetime(2026, 3, 15, tzinfo=timezone.utc),
            ticker="AAPL", alert_type=OptionsAlertType.UNUSUAL_VOLUME,
            put_call_ratio=0.3, volume=20000, avg_volume=5000,
            volume_ratio=4.0, strike=200.0,
            expiry=datetime(2026, 4, 17, tzinfo=timezone.utc), premium=500_000.0,
        ),
        OptionsAlert(
            timestamp=datetime(2026, 3, 15, tzinfo=timezone.utc),
            ticker="AAPL", alert_type=OptionsAlertType.UNUSUAL_VOLUME,
            put_call_ratio=0.3, volume=15000, avg_volume=5000,
            volume_ratio=3.0, strike=202.5,
            expiry=datetime(2026, 4, 17, tzinfo=timezone.utc), premium=400_000.0,
        ),
        OptionsAlert(
            timestamp=datetime(2026, 3, 15, tzinfo=timezone.utc),
            ticker="AAPL", alert_type=OptionsAlertType.UNUSUAL_VOLUME,
            put_call_ratio=0.3, volume=18000, avg_volume=5000,
            volume_ratio=3.6, strike=205.0,
            expiry=datetime(2026, 4, 17, tzinfo=timezone.utc), premium=450_000.0,
        ),
    ]
    clusters = detect_strike_clustering(alerts, strike_range_pct=0.05)
    assert len(clusters) >= 1
    assert clusters[0]["ticker"] == "AAPL"
    assert clusters[0]["center_strike"] == pytest.approx(202.5, abs=5.0)


def test_options_source_interface():
    """OptionsActivitySource implements SignalSource ABC."""
    source = OptionsActivitySource()
    assert source.source_name == "options_activity"
    assert source.rate_limit_per_second > 0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_options_activity.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.signals.sources.options_activity'`

**Step 3: Implement the options activity source**

```python
# src/evolve_trader/signals/sources/options_activity.py
"""Options unusual activity signal source.

Unusual volume, put/call ratio shifts, strike clustering, large block trades.
Pre-event positioning signals. Decay: ~2 days, very fast exponential.
Sources: CBOE, Unusual Whales, Market Chameleon.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from statistics import mean
from typing import Any

from evolve_trader.signals.base import SignalSource
from evolve_trader.signals.events import SignalEvent
from evolve_trader.signals.types import DecayProfile, DecayType, SignalType


OPTIONS_ACTIVITY_DECAY = DecayProfile(
    initial_confidence=0.85,
    half_life_days=2,
    decay_type=DecayType.EXPONENTIAL,
)


class OptionsAlertType(str, Enum):
    """Types of unusual options activity."""
    UNUSUAL_VOLUME = "unusual_volume"
    PUT_CALL_SHIFT = "put_call_shift"
    STRIKE_CLUSTER = "strike_cluster"
    LARGE_BLOCK = "large_block"


@dataclass
class OptionsAlert:
    """A single unusual options activity alert."""
    timestamp: datetime
    ticker: str
    alert_type: OptionsAlertType
    put_call_ratio: float
    volume: int
    avg_volume: int
    volume_ratio: float
    strike: float
    expiry: datetime
    premium: float  # total premium in dollars


def detect_strike_clustering(
    alerts: list[OptionsAlert],
    strike_range_pct: float = 0.05,
) -> list[dict[str, Any]]:
    """Detect concentrated positioning at similar strikes.

    Groups alerts by ticker where strikes are within strike_range_pct
    of each other, indicating coordinated positioning.
    """
    from collections import defaultdict

    by_ticker: dict[str, list[OptionsAlert]] = defaultdict(list)
    for alert in alerts:
        by_ticker[alert.ticker].append(alert)

    clusters = []
    for ticker, ticker_alerts in by_ticker.items():
        if len(ticker_alerts) < 2:
            continue
        strikes = sorted(a.strike for a in ticker_alerts)
        center = mean(strikes)
        spread = (max(strikes) - min(strikes)) / center if center > 0 else 0
        if spread <= strike_range_pct:
            clusters.append({
                "ticker": ticker,
                "center_strike": center,
                "strike_spread_pct": spread,
                "alert_count": len(ticker_alerts),
                "total_volume": sum(a.volume for a in ticker_alerts),
                "total_premium": sum(a.premium for a in ticker_alerts),
            })
    return clusters


class OptionsActivitySource(SignalSource):
    """Signal source for unusual options activity."""

    @property
    def source_name(self) -> str:
        return "options_activity"

    @property
    def rate_limit_per_second(self) -> float:
        return 5.0

    def alert_to_signals(
        self, alerts: list[OptionsAlert]
    ) -> list[SignalEvent]:
        """Convert options alerts into SignalEvents."""
        events = []
        for alert in alerts:
            events.append(SignalEvent(
                source="options_activity",
                source_entity=f"Options:{alert.ticker}",
                timestamp=alert.timestamp,
                trade_date=alert.timestamp,
                filing_date=alert.timestamp,
                confidence=OPTIONS_ACTIVITY_DECAY.initial_confidence,
                decay_profile=OPTIONS_ACTIVITY_DECAY,
                signal_type=SignalType.CONVICTION,
                payload={
                    "ticker": alert.ticker,
                    "alert_type": alert.alert_type.value,
                    "put_call_ratio": alert.put_call_ratio,
                    "volume": alert.volume,
                    "volume_ratio": alert.volume_ratio,
                    "strike": alert.strike,
                    "premium": alert.premium,
                },
                metadata={
                    "avg_volume": alert.avg_volume,
                    "expiry": alert.expiry.isoformat(),
                },
            ))
        return events

    async def fetch_signals(self) -> list[SignalEvent]:
        """Fetch latest unusual options activity."""
        # TODO: Implement CBOE / Unusual Whales polling
        return []

    async def validate_schema(self, response: dict) -> bool:
        """Validate options data response schema."""
        return "alerts" in response or "volume" in response
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_options_activity.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/signals/sources/options_activity.py tests/unit/test_options_activity.py
git commit -m "feat: options unusual activity signal source with strike clustering"
```

---

## Task 7: On-Chain Whale Movements

**Files:**
- Create: `src/evolve_trader/signals/sources/onchain_whales.py`
- Create: `tests/unit/test_onchain_whales.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_onchain_whales.py
"""Tests for on-chain whale movement signal source."""
import pytest
from datetime import datetime, timezone
from evolve_trader.signals.sources.onchain_whales import (
    OnChainWhaleSource,
    WhaleMovement,
    MovementType,
    WHALE_MOVEMENT_DECAY,
)
from evolve_trader.signals.types import SignalType, DecayType


def test_whale_decay_profile():
    """On-chain whales: ~3-day half-life, fast exponential."""
    assert WHALE_MOVEMENT_DECAY.half_life_days == 3
    assert WHALE_MOVEMENT_DECAY.decay_type == DecayType.EXPONENTIAL
    assert WHALE_MOVEMENT_DECAY.initial_confidence >= 0.70


def test_movement_types():
    """Movement types cover accumulation and distribution."""
    assert MovementType.ACCUMULATION.value == "accumulation"
    assert MovementType.DISTRIBUTION.value == "distribution"
    assert MovementType.EXCHANGE_INFLOW.value == "exchange_inflow"
    assert MovementType.EXCHANGE_OUTFLOW.value == "exchange_outflow"


def test_whale_movement_has_required_fields():
    """WhaleMovement captures large wallet transfer data."""
    movement = WhaleMovement(
        timestamp=datetime(2026, 3, 15, tzinfo=timezone.utc),
        wallet_address="0xabc123def456",
        asset="BTC",
        amount=500.0,
        usd_value=22_500_000.0,
        movement_type=MovementType.ACCUMULATION,
        from_address="0xexchange",
        to_address="0xabc123def456",
        chain="bitcoin",
    )
    assert movement.usd_value == 22_500_000.0
    assert movement.movement_type == MovementType.ACCUMULATION


def test_source_produces_signals():
    """OnChainWhaleSource converts movements into SignalEvents."""
    source = OnChainWhaleSource()
    movement = WhaleMovement(
        timestamp=datetime(2026, 3, 15, tzinfo=timezone.utc),
        wallet_address="0xabc123def456",
        asset="BTC",
        amount=500.0,
        usd_value=22_500_000.0,
        movement_type=MovementType.ACCUMULATION,
        from_address="0xexchange",
        to_address="0xabc123def456",
        chain="bitcoin",
    )
    events = source.movement_to_signals([movement])
    assert len(events) >= 1
    event = events[0]
    assert event.source == "onchain_whales"
    assert event.signal_type == SignalType.CONVICTION
    assert event.payload["asset"] == "BTC"
    assert event.payload["movement_type"] == "accumulation"
    assert event.payload["usd_value"] == 22_500_000.0


def test_exchange_inflow_is_bearish():
    """Large exchange inflows are distribution/sell signals."""
    source = OnChainWhaleSource()
    movement = WhaleMovement(
        timestamp=datetime(2026, 3, 15, tzinfo=timezone.utc),
        wallet_address="0xabc123def456",
        asset="ETH",
        amount=10000.0,
        usd_value=25_000_000.0,
        movement_type=MovementType.EXCHANGE_INFLOW,
        from_address="0xabc123def456",
        to_address="0xbinance",
        chain="ethereum",
    )
    events = source.movement_to_signals([movement])
    assert events[0].payload["direction"] == "bearish"


def test_exchange_outflow_is_bullish():
    """Large exchange outflows are accumulation/buy signals."""
    source = OnChainWhaleSource()
    movement = WhaleMovement(
        timestamp=datetime(2026, 3, 15, tzinfo=timezone.utc),
        wallet_address="0xabc123def456",
        asset="ETH",
        amount=10000.0,
        usd_value=25_000_000.0,
        movement_type=MovementType.EXCHANGE_OUTFLOW,
        from_address="0xbinance",
        to_address="0xabc123def456",
        chain="ethereum",
    )
    events = source.movement_to_signals([movement])
    assert events[0].payload["direction"] == "bullish"


def test_small_movements_filtered():
    """Movements below threshold are not converted to signals."""
    source = OnChainWhaleSource(min_usd_value=1_000_000)
    movement = WhaleMovement(
        timestamp=datetime(2026, 3, 15, tzinfo=timezone.utc),
        wallet_address="0xsmall",
        asset="ETH",
        amount=1.0,
        usd_value=2500.0,
        movement_type=MovementType.ACCUMULATION,
        from_address="0xexchange",
        to_address="0xsmall",
        chain="ethereum",
    )
    events = source.movement_to_signals([movement])
    assert len(events) == 0


def test_whale_source_interface():
    """OnChainWhaleSource implements SignalSource ABC."""
    source = OnChainWhaleSource()
    assert source.source_name == "onchain_whales"
    assert source.rate_limit_per_second > 0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_onchain_whales.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.signals.sources.onchain_whales'`

**Step 3: Implement the on-chain whale source**

```python
# src/evolve_trader/signals/sources/onchain_whales.py
"""On-chain whale movement signal source.

Large wallet accumulation/distribution for crypto assets. Real-time.
Decay: ~3 days, fast exponential.
Sources: Arkham Intelligence, Whale Alert, Etherscan.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from evolve_trader.signals.base import SignalSource
from evolve_trader.signals.events import SignalEvent
from evolve_trader.signals.types import DecayProfile, DecayType, SignalType


WHALE_MOVEMENT_DECAY = DecayProfile(
    initial_confidence=0.75,
    half_life_days=3,
    decay_type=DecayType.EXPONENTIAL,
)

DEFAULT_MIN_USD = 1_000_000  # $1M minimum for whale classification


class MovementType(str, Enum):
    """Types of on-chain whale movement."""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    EXCHANGE_INFLOW = "exchange_inflow"
    EXCHANGE_OUTFLOW = "exchange_outflow"


BEARISH_MOVEMENTS = {MovementType.DISTRIBUTION, MovementType.EXCHANGE_INFLOW}
BULLISH_MOVEMENTS = {MovementType.ACCUMULATION, MovementType.EXCHANGE_OUTFLOW}


@dataclass
class WhaleMovement:
    """A single large on-chain transfer."""
    timestamp: datetime
    wallet_address: str
    asset: str
    amount: float
    usd_value: float
    movement_type: MovementType
    from_address: str
    to_address: str
    chain: str  # bitcoin, ethereum, polygon, etc.


class OnChainWhaleSource(SignalSource):
    """Signal source for on-chain whale movements."""

    def __init__(self, min_usd_value: float = DEFAULT_MIN_USD):
        self._min_usd_value = min_usd_value

    @property
    def source_name(self) -> str:
        return "onchain_whales"

    @property
    def rate_limit_per_second(self) -> float:
        return 10.0

    def movement_to_signals(
        self, movements: list[WhaleMovement]
    ) -> list[SignalEvent]:
        """Convert whale movements into SignalEvents."""
        events = []
        for movement in movements:
            if movement.usd_value < self._min_usd_value:
                continue

            direction = (
                "bearish" if movement.movement_type in BEARISH_MOVEMENTS
                else "bullish"
            )

            events.append(SignalEvent(
                source="onchain_whales",
                source_entity=f"Whale:{movement.wallet_address[:10]}",
                timestamp=movement.timestamp,
                trade_date=movement.timestamp,
                filing_date=movement.timestamp,
                confidence=WHALE_MOVEMENT_DECAY.initial_confidence,
                decay_profile=WHALE_MOVEMENT_DECAY,
                signal_type=SignalType.CONVICTION,
                payload={
                    "asset": movement.asset,
                    "amount": movement.amount,
                    "usd_value": movement.usd_value,
                    "movement_type": movement.movement_type.value,
                    "direction": direction,
                    "chain": movement.chain,
                },
                metadata={
                    "wallet_address": movement.wallet_address,
                    "from_address": movement.from_address,
                    "to_address": movement.to_address,
                },
            ))
        return events

    async def fetch_signals(self) -> list[SignalEvent]:
        """Fetch latest whale movements."""
        # TODO: Implement Arkham Intelligence / Whale Alert polling
        return []

    async def validate_schema(self, response: dict) -> bool:
        """Validate on-chain data response schema."""
        return "transfers" in response or "movements" in response
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_onchain_whales.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/signals/sources/onchain_whales.py tests/unit/test_onchain_whales.py
git commit -m "feat: on-chain whale movement signal source with direction inference"
```

---

## Task 8: Investor Letter Parser

**Files:**
- Create: `src/evolve_trader/signals/sources/investor_letters.py`
- Create: `tests/unit/test_investor_letters.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_investor_letters.py
"""Tests for institutional investor letter parser signal source."""
import pytest
from datetime import datetime, timezone
from evolve_trader.signals.sources.investor_letters import (
    InvestorLetterSource,
    InvestorLetter,
    LetterAuthor,
    INVESTOR_LETTER_DECAY,
    TRACKED_AUTHORS,
)
from evolve_trader.signals.types import SignalType, DecayType


def test_letter_decay_profile():
    """Investor letters: ~45-day half-life, slow linear decay (quarterly)."""
    assert INVESTOR_LETTER_DECAY.half_life_days == 45
    assert INVESTOR_LETTER_DECAY.decay_type == DecayType.LINEAR


def test_tracked_authors():
    """Tracked author list includes key institutional voices."""
    author_names = {a.name for a in TRACKED_AUTHORS}
    assert "Ray Dalio" in author_names
    assert "Howard Marks" in author_names
    assert "Seth Klarman" in author_names
    assert "Jeremy Grantham" in author_names


def test_letter_author_has_specialization():
    """Each tracked author has a known specialization."""
    for author in TRACKED_AUTHORS:
        assert author.specialization is not None
        assert len(author.specialization) > 0


def test_investor_letter_has_required_fields():
    """InvestorLetter captures parsed letter content."""
    letter = InvestorLetter(
        author=LetterAuthor(
            name="Ray Dalio",
            firm="Bridgewater Associates",
            specialization="debt_cycles",
        ),
        date=datetime(2026, 3, 1, tzinfo=timezone.utc),
        title="Principles for Navigating the Current Debt Cycle",
        url="https://example.com/dalio-q1-2026",
        raw_text="The current debt cycle is in its late stages...",
        llm_summary="Dalio warns of late-stage debt cycle dynamics...",
        regime_implications=["risk-off", "defensive_positioning"],
        sector_tilts={"Financials": -0.3, "Commodities": 0.4},
        confidence_in_thesis=0.75,
    )
    assert letter.author.name == "Ray Dalio"
    assert len(letter.regime_implications) == 2
    assert letter.sector_tilts["Financials"] == -0.3


def test_source_produces_thesis_signals():
    """InvestorLetterSource produces THESIS-type SignalEvents."""
    source = InvestorLetterSource()
    letter = InvestorLetter(
        author=LetterAuthor(
            name="Howard Marks",
            firm="Oaktree Capital",
            specialization="credit_cycles",
        ),
        date=datetime(2026, 3, 1, tzinfo=timezone.utc),
        title="The Importance of Being Cautious",
        url="https://example.com/marks-q1-2026",
        raw_text="Credit spreads are at historically tight levels...",
        llm_summary="Marks advocates caution in current tight-spread environment...",
        regime_implications=["risk-off"],
        sector_tilts={"High Yield": -0.5},
        confidence_in_thesis=0.80,
    )
    events = source.letter_to_signals(letter)
    assert len(events) >= 1
    event = events[0]
    assert event.source == "investor_letters"
    assert event.source_entity == "Howard Marks"
    assert event.signal_type == SignalType.THESIS
    assert "regime_implications" in event.payload
    assert "sector_tilts" in event.payload


def test_letter_requires_llm_summary():
    """Letters without LLM summary are rejected."""
    source = InvestorLetterSource()
    letter = InvestorLetter(
        author=LetterAuthor(
            name="Ray Dalio",
            firm="Bridgewater Associates",
            specialization="debt_cycles",
        ),
        date=datetime(2026, 3, 1, tzinfo=timezone.utc),
        title="Test Letter",
        url="https://example.com/test",
        raw_text="Some raw text...",
        llm_summary="",  # empty
        regime_implications=[],
        sector_tilts={},
        confidence_in_thesis=0.0,
    )
    events = source.letter_to_signals(letter)
    assert len(events) == 0


def test_letter_source_interface():
    """InvestorLetterSource implements SignalSource ABC."""
    source = InvestorLetterSource()
    assert source.source_name == "investor_letters"
    assert source.rate_limit_per_second > 0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_investor_letters.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.signals.sources.investor_letters'`

**Step 3: Implement the investor letter source**

```python
# src/evolve_trader/signals/sources/investor_letters.py
"""Institutional investor letter parser signal source.

Parses letters from Dalio (debt cycles), Marks (credit cycles),
Klarman, Grantham (bubble calls), and others via LLM.
Thesis-level macro regime frameworks. Quarterly/annual cadence.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from evolve_trader.signals.base import SignalSource
from evolve_trader.signals.events import SignalEvent
from evolve_trader.signals.types import DecayProfile, DecayType, SignalType


INVESTOR_LETTER_DECAY = DecayProfile(
    initial_confidence=0.70,
    half_life_days=45,
    decay_type=DecayType.LINEAR,
)


@dataclass
class LetterAuthor:
    """A tracked institutional investor."""
    name: str
    firm: str
    specialization: str  # debt_cycles, credit_cycles, bubbles, macro, etc.


TRACKED_AUTHORS = [
    LetterAuthor("Ray Dalio", "Bridgewater Associates", "debt_cycles"),
    LetterAuthor("Howard Marks", "Oaktree Capital", "credit_cycles"),
    LetterAuthor("Seth Klarman", "Baupost Group", "value_contrarian"),
    LetterAuthor("Jeremy Grantham", "GMO", "bubble_identification"),
    LetterAuthor("Jamie Dimon", "JPMorgan Chase", "banking_macro"),
    LetterAuthor("Larry Fink", "BlackRock", "institutional_flows"),
]


@dataclass
class InvestorLetter:
    """A parsed institutional investor letter."""
    author: LetterAuthor
    date: datetime
    title: str
    url: str
    raw_text: str
    llm_summary: str
    regime_implications: list[str]  # e.g. ["risk-off", "defensive_positioning"]
    sector_tilts: dict[str, float]  # sector -> tilt (-1.0 to 1.0)
    confidence_in_thesis: float  # LLM's assessed confidence in the thesis


class InvestorLetterSource(SignalSource):
    """Signal source for institutional investor letters."""

    @property
    def source_name(self) -> str:
        return "investor_letters"

    @property
    def rate_limit_per_second(self) -> float:
        return 1.0  # LLM processing rate

    def letter_to_signals(
        self, letter: InvestorLetter
    ) -> list[SignalEvent]:
        """Convert a parsed investor letter into SignalEvents.

        Requires LLM summary to produce signals. Letters without
        summaries are rejected — raw text alone is insufficient.
        """
        if not letter.llm_summary or letter.confidence_in_thesis <= 0:
            return []

        return [
            SignalEvent(
                source="investor_letters",
                source_entity=letter.author.name,
                timestamp=letter.date,
                trade_date=letter.date,
                filing_date=letter.date,
                confidence=min(
                    INVESTOR_LETTER_DECAY.initial_confidence,
                    letter.confidence_in_thesis,
                ),
                decay_profile=INVESTOR_LETTER_DECAY,
                signal_type=SignalType.THESIS,
                payload={
                    "title": letter.title,
                    "llm_summary": letter.llm_summary,
                    "regime_implications": letter.regime_implications,
                    "sector_tilts": letter.sector_tilts,
                    "confidence_in_thesis": letter.confidence_in_thesis,
                },
                metadata={
                    "firm": letter.author.firm,
                    "specialization": letter.author.specialization,
                    "url": letter.url,
                },
            )
        ]

    async def fetch_signals(self) -> list[SignalEvent]:
        """Fetch and parse latest investor letters via LLM."""
        # TODO: Implement letter scraping + LiteLLM parsing
        return []

    async def validate_schema(self, response: dict) -> bool:
        """Validate letter parsing response."""
        return "summary" in response or "text" in response
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_investor_letters.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/signals/sources/investor_letters.py tests/unit/test_investor_letters.py
git commit -m "feat: investor letter parser signal source with LLM-extracted thesis signals"
```

---

## Task 9: News & Macro Feeds

**Files:**
- Create: `src/evolve_trader/signals/sources/news_macro.py`
- Create: `tests/unit/test_news_macro.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_news_macro.py
"""Tests for news and macro feed signal source."""
import pytest
from datetime import datetime, timezone
from evolve_trader.signals.sources.news_macro import (
    NewsMacroSource,
    NewsEvent,
    MacroIndicator,
    NewsCategory,
    NEWS_MACRO_DECAY,
    FED_LANGUAGE_DECAY,
)
from evolve_trader.signals.types import SignalType, DecayType


def test_news_decay_profile():
    """News: ~5-day half-life, exponential decay."""
    assert NEWS_MACRO_DECAY.half_life_days == 5
    assert NEWS_MACRO_DECAY.decay_type == DecayType.EXPONENTIAL


def test_fed_language_decay_profile():
    """Fed language: ~45-day half-life, slow step-function-like."""
    assert FED_LANGUAGE_DECAY.half_life_days == 45
    assert FED_LANGUAGE_DECAY.decay_type == DecayType.LINEAR


def test_news_categories():
    """News categories cover all required types."""
    assert NewsCategory.BREAKING.value == "breaking"
    assert NewsCategory.FED_LANGUAGE.value == "fed_language"
    assert NewsCategory.EARNINGS.value == "earnings"
    assert NewsCategory.GEOPOLITICAL.value == "geopolitical"
    assert NewsCategory.MACRO_DATA.value == "macro_data"


def test_news_event_has_required_fields():
    """NewsEvent captures breaking event data."""
    event = NewsEvent(
        timestamp=datetime(2026, 3, 15, tzinfo=timezone.utc),
        headline="Fed signals potential rate cut at June meeting",
        source_feed="jina_ai",
        category=NewsCategory.FED_LANGUAGE,
        url="https://example.com/fed-signal",
        sentiment_score=0.6,
        relevance_score=0.9,
        tickers_mentioned=["SPY", "TLT"],
    )
    assert event.category == NewsCategory.FED_LANGUAGE
    assert len(event.tickers_mentioned) == 2


def test_macro_indicator_has_required_fields():
    """MacroIndicator captures FRED-sourced economic data."""
    indicator = MacroIndicator(
        timestamp=datetime(2026, 3, 15, tzinfo=timezone.utc),
        series_id="UNRATE",
        name="Unemployment Rate",
        value=3.8,
        previous_value=3.9,
        expected_value=3.8,
        surprise=0.0,
    )
    assert indicator.series_id == "UNRATE"
    assert indicator.surprise == 0.0


def test_source_produces_event_driven_signals_for_news():
    """NewsMacroSource converts news into EVENT_DRIVEN SignalEvents."""
    source = NewsMacroSource()
    event = NewsEvent(
        timestamp=datetime(2026, 3, 15, tzinfo=timezone.utc),
        headline="Surprise tariff announcement on tech imports",
        source_feed="newsapi",
        category=NewsCategory.GEOPOLITICAL,
        url="https://example.com/tariffs",
        sentiment_score=-0.8,
        relevance_score=0.95,
        tickers_mentioned=["AAPL", "NVDA", "MSFT"],
    )
    signals = source.news_to_signals([event])
    assert len(signals) >= 1
    signal = signals[0]
    assert signal.source == "news_macro"
    assert signal.signal_type == SignalType.EVENT_DRIVEN
    assert signal.payload["headline"] == "Surprise tariff announcement on tech imports"
    assert signal.payload["sentiment"] == -0.8


def test_source_produces_regime_signals_for_macro():
    """NewsMacroSource converts macro data into REGIME_READ SignalEvents."""
    source = NewsMacroSource()
    indicator = MacroIndicator(
        timestamp=datetime(2026, 3, 15, tzinfo=timezone.utc),
        series_id="UNRATE",
        name="Unemployment Rate",
        value=4.5,
        previous_value=3.8,
        expected_value=3.9,
        surprise=0.6,  # big upside miss
    )
    signals = source.macro_to_signals([indicator])
    assert len(signals) >= 1
    signal = signals[0]
    assert signal.source == "news_macro"
    assert signal.signal_type == SignalType.REGIME_READ
    assert signal.payload["series_id"] == "UNRATE"
    assert signal.payload["surprise"] == 0.6


def test_fed_language_uses_longer_decay():
    """Fed language events use the slower FED_LANGUAGE_DECAY profile."""
    source = NewsMacroSource()
    event = NewsEvent(
        timestamp=datetime(2026, 3, 15, tzinfo=timezone.utc),
        headline="Fed Chair: 'We remain data dependent'",
        source_feed="jina_ai",
        category=NewsCategory.FED_LANGUAGE,
        url="https://example.com/fed-statement",
        sentiment_score=0.1,
        relevance_score=0.95,
        tickers_mentioned=[],
    )
    signals = source.news_to_signals([event])
    assert signals[0].decay_profile.half_life_days == FED_LANGUAGE_DECAY.half_life_days


def test_jina_ai_wrapped_as_source_feed():
    """Jina AI (existing AI-Trader integration) is a recognized source feed."""
    source = NewsMacroSource()
    event = NewsEvent(
        timestamp=datetime(2026, 3, 15, tzinfo=timezone.utc),
        headline="Market analysis from Jina AI",
        source_feed="jina_ai",
        category=NewsCategory.BREAKING,
        url="https://example.com/jina",
        sentiment_score=0.5,
        relevance_score=0.7,
        tickers_mentioned=["AAPL"],
    )
    signals = source.news_to_signals([event])
    assert signals[0].metadata["source_feed"] == "jina_ai"


def test_news_source_interface():
    """NewsMacroSource implements SignalSource ABC."""
    source = NewsMacroSource()
    assert source.source_name == "news_macro"
    assert source.rate_limit_per_second > 0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_news_macro.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.signals.sources.news_macro'`

**Step 3: Implement the news & macro source**

```python
# src/evolve_trader/signals/sources/news_macro.py
"""News and macro feed signal source.

Breaking events, Fed language, earnings surprises, geopolitical triggers.
Sources: Jina AI (existing AI-Trader integration — wrapped into SignalEvent
format), NewsAPI (new), FRED (new — macro economic data).

Fed language uses a slower ~45-day half-life (step-function-like).
General news uses a ~5-day half-life (fast exponential).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from evolve_trader.signals.base import SignalSource
from evolve_trader.signals.events import SignalEvent
from evolve_trader.signals.types import DecayProfile, DecayType, SignalType


NEWS_MACRO_DECAY = DecayProfile(
    initial_confidence=0.70,
    half_life_days=5,
    decay_type=DecayType.EXPONENTIAL,
)

FED_LANGUAGE_DECAY = DecayProfile(
    initial_confidence=0.75,
    half_life_days=45,
    decay_type=DecayType.LINEAR,
)


class NewsCategory(str, Enum):
    """Categories of news events."""
    BREAKING = "breaking"
    FED_LANGUAGE = "fed_language"
    EARNINGS = "earnings"
    GEOPOLITICAL = "geopolitical"
    MACRO_DATA = "macro_data"


@dataclass
class NewsEvent:
    """A news event from any feed source."""
    timestamp: datetime
    headline: str
    source_feed: str  # jina_ai, newsapi, etc.
    category: NewsCategory
    url: str
    sentiment_score: float  # -1.0 (bearish) to 1.0 (bullish)
    relevance_score: float  # 0.0 to 1.0
    tickers_mentioned: list[str]


@dataclass
class MacroIndicator:
    """A macro economic indicator from FRED."""
    timestamp: datetime
    series_id: str  # FRED series ID (UNRATE, GDP, CPI, etc.)
    name: str
    value: float
    previous_value: float
    expected_value: float
    surprise: float  # actual - expected (positive = above expectations)


class NewsMacroSource(SignalSource):
    """Signal source for news and macro economic data."""

    @property
    def source_name(self) -> str:
        return "news_macro"

    @property
    def rate_limit_per_second(self) -> float:
        return 10.0

    def news_to_signals(
        self, events: list[NewsEvent]
    ) -> list[SignalEvent]:
        """Convert news events into SignalEvents.

        Fed language uses the slower FED_LANGUAGE_DECAY profile.
        """
        signals = []
        for event in events:
            decay = (
                FED_LANGUAGE_DECAY
                if event.category == NewsCategory.FED_LANGUAGE
                else NEWS_MACRO_DECAY
            )
            signals.append(SignalEvent(
                source="news_macro",
                source_entity=event.source_feed,
                timestamp=event.timestamp,
                trade_date=event.timestamp,
                filing_date=event.timestamp,
                confidence=decay.initial_confidence * event.relevance_score,
                decay_profile=decay,
                signal_type=SignalType.EVENT_DRIVEN,
                payload={
                    "headline": event.headline,
                    "category": event.category.value,
                    "sentiment": event.sentiment_score,
                    "relevance": event.relevance_score,
                    "tickers_mentioned": event.tickers_mentioned,
                },
                metadata={
                    "source_feed": event.source_feed,
                    "url": event.url,
                },
            ))
        return signals

    def macro_to_signals(
        self, indicators: list[MacroIndicator]
    ) -> list[SignalEvent]:
        """Convert macro economic indicators into REGIME_READ SignalEvents."""
        signals = []
        for indicator in indicators:
            signals.append(SignalEvent(
                source="news_macro",
                source_entity="FRED",
                timestamp=indicator.timestamp,
                trade_date=indicator.timestamp,
                filing_date=indicator.timestamp,
                confidence=FED_LANGUAGE_DECAY.initial_confidence,
                decay_profile=FED_LANGUAGE_DECAY,
                signal_type=SignalType.REGIME_READ,
                payload={
                    "series_id": indicator.series_id,
                    "name": indicator.name,
                    "value": indicator.value,
                    "previous_value": indicator.previous_value,
                    "expected_value": indicator.expected_value,
                    "surprise": indicator.surprise,
                },
                metadata={
                    "source_feed": "fred",
                },
            ))
        return signals

    async def fetch_signals(self) -> list[SignalEvent]:
        """Fetch latest news and macro data."""
        # TODO: Implement Jina AI wrapper, NewsAPI polling, FRED data pull
        return []

    async def validate_schema(self, response: dict) -> bool:
        """Validate news/macro response schema."""
        return "articles" in response or "observations" in response or "headline" in response
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_news_macro.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/signals/sources/news_macro.py tests/unit/test_news_macro.py
git commit -m "feat: news and macro feed signal source with Jina AI, NewsAPI, FRED integration"
```

---

## Task 10: Signal Latency Hierarchy Verification

**Files:**
- Create: `src/evolve_trader/monitoring/latency_hierarchy.py`
- Create: `tests/unit/test_latency_hierarchy.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_latency_hierarchy.py
"""Tests for signal latency hierarchy verification in monitoring."""
import pytest
from datetime import timedelta
from evolve_trader.monitoring.latency_hierarchy import (
    SignalLatencyTier,
    LATENCY_HIERARCHY,
    verify_latency_ordering,
    get_expected_latency,
)


def test_latency_hierarchy_has_all_sources():
    """Hierarchy includes all signal sources in correct order."""
    source_names = [tier.source_name for tier in LATENCY_HIERARCHY]
    assert "polymarket" in source_names
    assert "kalshi" in source_names
    assert "options_activity" in source_names
    assert "news_macro" in source_names
    assert "onchain_whales" in source_names
    assert "ark_trades" in source_names
    assert "edgar_form4" in source_names
    assert "congressional" in source_names
    assert "edgar_13f" in source_names


def test_hierarchy_is_ordered_fastest_to_slowest():
    """Sources are ordered from fastest (prediction markets) to slowest (13F)."""
    for i in range(len(LATENCY_HIERARCHY) - 1):
        assert LATENCY_HIERARCHY[i].typical_latency <= LATENCY_HIERARCHY[i + 1].typical_latency, (
            f"{LATENCY_HIERARCHY[i].source_name} should be faster than "
            f"{LATENCY_HIERARCHY[i + 1].source_name}"
        )


def test_prediction_markets_fastest():
    """Prediction markets (Polymarket/Kalshi) are the fastest tier."""
    assert LATENCY_HIERARCHY[0].source_name in ("polymarket", "kalshi")
    assert LATENCY_HIERARCHY[0].typical_latency <= timedelta(minutes=5)


def test_13f_filings_slowest():
    """13F institutional filings are the slowest tier."""
    assert LATENCY_HIERARCHY[-1].source_name == "edgar_13f"
    assert LATENCY_HIERARCHY[-1].typical_latency >= timedelta(days=30)


def test_verify_latency_ordering_passes_correct_data():
    """Verification passes when observed latencies match hierarchy."""
    observed = {
        "polymarket": timedelta(minutes=2),
        "options_activity": timedelta(hours=3),
        "edgar_13f": timedelta(days=60),
    }
    result = verify_latency_ordering(observed)
    assert result.passed is True
    assert len(result.violations) == 0


def test_verify_latency_ordering_detects_violations():
    """Verification catches when a fast source is slower than expected."""
    observed = {
        "polymarket": timedelta(days=5),  # way too slow for prediction market
        "edgar_13f": timedelta(days=60),
    }
    result = verify_latency_ordering(observed)
    assert result.passed is False
    assert len(result.violations) >= 1


def test_get_expected_latency():
    """Can look up expected latency for any registered source."""
    latency = get_expected_latency("polymarket")
    assert latency <= timedelta(minutes=5)
    latency = get_expected_latency("edgar_13f")
    assert latency >= timedelta(days=30)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_latency_hierarchy.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.monitoring.latency_hierarchy'`

**Step 3: Implement the latency hierarchy**

```python
# src/evolve_trader/monitoring/latency_hierarchy.py
"""Signal latency hierarchy verification for monitoring.

Verified ordering (fastest → slowest):
  Prediction markets (minutes) → options unusual activity (hours) →
  news feeds (hours) → on-chain whale moves (hours-days) →
  ARK daily trades (same day) → Form 4 insider filings (days) →
  congressional STOCK Act (weeks) → 13F institutional filings (months)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any


@dataclass
class SignalLatencyTier:
    """A signal source with its expected latency characteristics."""
    source_name: str
    description: str
    typical_latency: timedelta
    max_acceptable_latency: timedelta


@dataclass
class LatencyVerificationResult:
    """Result of verifying observed latencies against the hierarchy."""
    passed: bool
    violations: list[dict[str, Any]]


LATENCY_HIERARCHY: list[SignalLatencyTier] = [
    SignalLatencyTier(
        source_name="polymarket",
        description="Polymarket prediction markets",
        typical_latency=timedelta(minutes=2),
        max_acceptable_latency=timedelta(minutes=15),
    ),
    SignalLatencyTier(
        source_name="kalshi",
        description="Kalshi prediction markets (CFTC-regulated)",
        typical_latency=timedelta(minutes=3),
        max_acceptable_latency=timedelta(minutes=15),
    ),
    SignalLatencyTier(
        source_name="options_activity",
        description="Options unusual activity",
        typical_latency=timedelta(hours=2),
        max_acceptable_latency=timedelta(hours=8),
    ),
    SignalLatencyTier(
        source_name="news_macro",
        description="News feeds and macro data",
        typical_latency=timedelta(hours=3),
        max_acceptable_latency=timedelta(hours=12),
    ),
    SignalLatencyTier(
        source_name="onchain_whales",
        description="On-chain whale movements",
        typical_latency=timedelta(hours=6),
        max_acceptable_latency=timedelta(days=1),
    ),
    SignalLatencyTier(
        source_name="ark_trades",
        description="ARK Invest daily trade disclosures",
        typical_latency=timedelta(hours=18),
        max_acceptable_latency=timedelta(days=1),
    ),
    SignalLatencyTier(
        source_name="edgar_form4",
        description="SEC EDGAR Form 4 insider filings",
        typical_latency=timedelta(days=2),
        max_acceptable_latency=timedelta(days=5),
    ),
    SignalLatencyTier(
        source_name="congressional",
        description="Congressional STOCK Act disclosures",
        typical_latency=timedelta(days=30),
        max_acceptable_latency=timedelta(days=45),
    ),
    SignalLatencyTier(
        source_name="edgar_13f",
        description="SEC EDGAR 13F institutional filings",
        typical_latency=timedelta(days=45),
        max_acceptable_latency=timedelta(days=90),
    ),
]

_LATENCY_MAP = {tier.source_name: tier for tier in LATENCY_HIERARCHY}


def get_expected_latency(source_name: str) -> timedelta:
    """Look up the expected typical latency for a signal source."""
    tier = _LATENCY_MAP.get(source_name)
    if tier is None:
        raise KeyError(f"Unknown signal source: {source_name}")
    return tier.typical_latency


def verify_latency_ordering(
    observed: dict[str, timedelta],
) -> LatencyVerificationResult:
    """Verify observed latencies against the expected hierarchy.

    Flags violations where a source's observed latency exceeds its
    max acceptable latency.
    """
    violations = []
    for source_name, observed_latency in observed.items():
        tier = _LATENCY_MAP.get(source_name)
        if tier is None:
            continue
        if observed_latency > tier.max_acceptable_latency:
            violations.append({
                "source": source_name,
                "observed": observed_latency,
                "max_acceptable": tier.max_acceptable_latency,
                "typical": tier.typical_latency,
                "severity": "critical" if observed_latency > tier.max_acceptable_latency * 3 else "warning",
            })

    return LatencyVerificationResult(
        passed=len(violations) == 0,
        violations=violations,
    )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_latency_hierarchy.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/monitoring/latency_hierarchy.py tests/unit/test_latency_hierarchy.py
git commit -m "feat: signal latency hierarchy verification for monitoring"
```

---

## Task 11: Discovery — WhaleWisdom Fund Scan

**Files:**
- Create: `src/evolve_trader/discovery/whalewisdom.py`
- Create: `tests/unit/test_whalewisdom.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_whalewisdom.py
"""Tests for WhaleWisdom fund performance scan discovery channel."""
import pytest
from datetime import datetime, timezone
from evolve_trader.discovery.whalewisdom import (
    WhaleWisdomScanner,
    FundProfile,
    ScanResult,
    QUALIFICATION_CRITERIA,
)


def test_qualification_criteria():
    """Qualification criteria match master plan specification."""
    assert QUALIFICATION_CRITERIA["whale_score_percentile"] >= 90
    assert QUALIFICATION_CRITERIA["top_10_concentration_min"] >= 0.60
    assert QUALIFICATION_CRITERIA["aum_min"] >= 50_000_000
    assert QUALIFICATION_CRITERIA["aum_max"] <= 2_000_000_000
    assert QUALIFICATION_CRITERIA["min_quarters_held"] >= 2


def test_fund_profile_has_required_fields():
    """FundProfile captures 13F filer characteristics."""
    fund = FundProfile(
        cik="0001234567",
        name="Concentrated Alpha Fund",
        aum=250_000_000,
        whale_score=92,
        top_10_concentration=0.72,
        avg_quarters_held=4.5,
        turnover_rate=0.15,
        trailing_alpha=0.18,
        num_holdings=25,
    )
    assert fund.whale_score == 92
    assert fund.top_10_concentration == 0.72


def test_scanner_qualifies_matching_fund():
    """Scanner qualifies funds meeting all criteria."""
    scanner = WhaleWisdomScanner()
    fund = FundProfile(
        cik="0001234567",
        name="Concentrated Alpha Fund",
        aum=250_000_000,
        whale_score=92,
        top_10_concentration=0.72,
        avg_quarters_held=4.5,
        turnover_rate=0.15,
        trailing_alpha=0.18,
        num_holdings=25,
    )
    assert scanner.qualifies(fund) is True


def test_scanner_rejects_low_whale_score():
    """Scanner rejects funds with low WhaleScore."""
    scanner = WhaleWisdomScanner()
    fund = FundProfile(
        cik="0001234567",
        name="Mediocre Fund",
        aum=250_000_000,
        whale_score=50,  # below top decile
        top_10_concentration=0.72,
        avg_quarters_held=4.5,
        turnover_rate=0.15,
        trailing_alpha=0.18,
        num_holdings=25,
    )
    assert scanner.qualifies(fund) is False


def test_scanner_rejects_too_large_aum():
    """Scanner rejects funds with AUM above $2B (too widely followed)."""
    scanner = WhaleWisdomScanner()
    fund = FundProfile(
        cik="0001234567",
        name="Too Big Fund",
        aum=5_000_000_000,  # $5B
        whale_score=95,
        top_10_concentration=0.72,
        avg_quarters_held=4.5,
        turnover_rate=0.15,
        trailing_alpha=0.18,
        num_holdings=25,
    )
    assert scanner.qualifies(fund) is False


def test_scanner_rejects_diversified_fund():
    """Scanner rejects funds with low concentration (too diversified)."""
    scanner = WhaleWisdomScanner()
    fund = FundProfile(
        cik="0001234567",
        name="Diversified Fund",
        aum=250_000_000,
        whale_score=95,
        top_10_concentration=0.30,  # too diversified
        avg_quarters_held=4.5,
        turnover_rate=0.15,
        trailing_alpha=0.18,
        num_holdings=200,
    )
    assert scanner.qualifies(fund) is False


def test_scan_result_has_ranked_candidates():
    """Scan results include ranked candidate list."""
    result = ScanResult(
        scan_date=datetime(2026, 3, 15, tzinfo=timezone.utc),
        total_filers_scanned=10000,
        candidates_found=15,
        candidates=[],
    )
    assert result.total_filers_scanned == 10000


def test_scanner_interface():
    """WhaleWisdomScanner has the expected public interface."""
    scanner = WhaleWisdomScanner()
    assert hasattr(scanner, "qualifies")
    assert hasattr(scanner, "scan")
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_whalewisdom.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.discovery.whalewisdom'`

**Step 3: Implement the WhaleWisdom scanner**

```python
# src/evolve_trader/discovery/whalewisdom.py
"""WhaleWisdom fund performance scan discovery channel.

Quarterly scan of 10,000+ institutional 13F filers. Filters:
  - WhaleScore top decile
  - Concentrated portfolios (top 10 holdings >60% AUM)
  - Small-to-mid AUM ($50M-$2B)
  - Low turnover (positions held 2+ quarters)
API + Apify scrapers.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


QUALIFICATION_CRITERIA = {
    "whale_score_percentile": 90,  # top decile
    "top_10_concentration_min": 0.60,  # >60% in top 10
    "aum_min": 50_000_000,  # $50M
    "aum_max": 2_000_000_000,  # $2B
    "min_quarters_held": 2,  # low turnover
}


@dataclass
class FundProfile:
    """Profile of an institutional 13F filer."""
    cik: str
    name: str
    aum: float
    whale_score: int  # 0-100
    top_10_concentration: float  # 0.0-1.0
    avg_quarters_held: float
    turnover_rate: float
    trailing_alpha: float  # trailing-twelve-month alpha
    num_holdings: int


@dataclass
class ScanResult:
    """Result of a quarterly WhaleWisdom scan."""
    scan_date: datetime
    total_filers_scanned: int
    candidates_found: int
    candidates: list[FundProfile]


class WhaleWisdomScanner:
    """Discovery channel: quarterly scan of 13F filers via WhaleWisdom."""

    def qualifies(self, fund: FundProfile) -> bool:
        """Check if a fund meets all qualification criteria."""
        return (
            fund.whale_score >= QUALIFICATION_CRITERIA["whale_score_percentile"]
            and fund.top_10_concentration >= QUALIFICATION_CRITERIA["top_10_concentration_min"]
            and fund.aum >= QUALIFICATION_CRITERIA["aum_min"]
            and fund.aum <= QUALIFICATION_CRITERIA["aum_max"]
            and fund.avg_quarters_held >= QUALIFICATION_CRITERIA["min_quarters_held"]
        )

    async def scan(self, filers: list[FundProfile]) -> ScanResult:
        """Run the quarterly scan across all filers."""
        candidates = [f for f in filers if self.qualifies(f)]
        # Sort by trailing alpha descending
        candidates.sort(key=lambda f: f.trailing_alpha, reverse=True)
        return ScanResult(
            scan_date=datetime.now(),
            total_filers_scanned=len(filers),
            candidates_found=len(candidates),
            candidates=candidates,
        )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_whalewisdom.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/discovery/whalewisdom.py tests/unit/test_whalewisdom.py
git commit -m "feat: WhaleWisdom quarterly fund scan discovery channel"
```

---

## Task 12: Discovery — EDGAR Real-Time 13F Stream

**Files:**
- Create: `src/evolve_trader/discovery/edgar_stream.py`
- Create: `tests/unit/test_edgar_stream.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_edgar_stream.py
"""Tests for EDGAR real-time 13F stream discovery channel."""
import pytest
from datetime import datetime, timezone
from evolve_trader.discovery.edgar_stream import (
    EdgarStreamMonitor,
    NewFiling,
    FilingCandidate,
    ALPHA_THRESHOLD,
)


def test_alpha_threshold():
    """Alpha threshold is >30% TTM as specified in master plan."""
    assert ALPHA_THRESHOLD >= 0.30


def test_new_filing_has_required_fields():
    """NewFiling captures real-time 13F-HR filing data."""
    filing = NewFiling(
        cik="0001234567",
        filer_name="Unknown Alpha Fund LLC",
        filing_date=datetime(2026, 3, 15, tzinfo=timezone.utc),
        report_period=datetime(2025, 12, 31, tzinfo=timezone.utc),
        accession_number="0001234567-26-000123",
        filing_url="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001234567",
        total_value=150_000_000,
        num_holdings=18,
    )
    assert filing.filer_name == "Unknown Alpha Fund LLC"
    assert filing.total_value == 150_000_000


def test_monitor_flags_high_alpha_filer():
    """Monitor flags untracked filers showing >30% TTM alpha."""
    monitor = EdgarStreamMonitor()
    filing = NewFiling(
        cik="0001234567",
        filer_name="Unknown Alpha Fund LLC",
        filing_date=datetime(2026, 3, 15, tzinfo=timezone.utc),
        report_period=datetime(2025, 12, 31, tzinfo=timezone.utc),
        accession_number="0001234567-26-000123",
        filing_url="https://sec.gov/test",
        total_value=150_000_000,
        num_holdings=18,
    )
    candidate = monitor.evaluate_filing(filing, ttm_alpha=0.45)
    assert isinstance(candidate, FilingCandidate)
    assert candidate.qualifies is True
    assert candidate.ttm_alpha == 0.45


def test_monitor_skips_low_alpha_filer():
    """Monitor ignores filers below alpha threshold."""
    monitor = EdgarStreamMonitor()
    filing = NewFiling(
        cik="0009999999",
        filer_name="Mediocre Fund",
        filing_date=datetime(2026, 3, 15, tzinfo=timezone.utc),
        report_period=datetime(2025, 12, 31, tzinfo=timezone.utc),
        accession_number="0009999999-26-000456",
        filing_url="https://sec.gov/test",
        total_value=100_000_000,
        num_holdings=50,
    )
    candidate = monitor.evaluate_filing(filing, ttm_alpha=0.10)
    assert candidate.qualifies is False


def test_monitor_skips_already_tracked():
    """Monitor ignores filers already in the tracked set."""
    monitor = EdgarStreamMonitor(tracked_ciks={"0001234567"})
    filing = NewFiling(
        cik="0001234567",
        filer_name="Already Tracked Fund",
        filing_date=datetime(2026, 3, 15, tzinfo=timezone.utc),
        report_period=datetime(2025, 12, 31, tzinfo=timezone.utc),
        accession_number="0001234567-26-000789",
        filing_url="https://sec.gov/test",
        total_value=200_000_000,
        num_holdings=20,
    )
    candidate = monitor.evaluate_filing(filing, ttm_alpha=0.50)
    assert candidate.qualifies is False


def test_monitor_interface():
    """EdgarStreamMonitor has the expected public interface."""
    monitor = EdgarStreamMonitor()
    assert hasattr(monitor, "evaluate_filing")
    assert hasattr(monitor, "stream")
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_edgar_stream.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.discovery.edgar_stream'`

**Step 3: Implement the EDGAR stream monitor**

```python
# src/evolve_trader/discovery/edgar_stream.py
"""EDGAR real-time 13F stream discovery channel.

Uses sec-api Python package with sub-200ms filing indexing.
Monitors new 13F-HR filings each quarter. Flags untracked filers
showing >30% trailing-twelve-month alpha.
Kaleidoscope API (api.kscope.io) for push notifications.
Backfill via EDGAR bulk data sets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


ALPHA_THRESHOLD = 0.30  # >30% TTM alpha required


@dataclass
class NewFiling:
    """A new 13F-HR filing detected in real-time."""
    cik: str
    filer_name: str
    filing_date: datetime
    report_period: datetime
    accession_number: str
    filing_url: str
    total_value: float
    num_holdings: int


@dataclass
class FilingCandidate:
    """Evaluation result for a new 13F filing."""
    filing: NewFiling
    ttm_alpha: float
    qualifies: bool
    reason: str


class EdgarStreamMonitor:
    """Discovery channel: real-time EDGAR 13F-HR stream monitoring."""

    def __init__(self, tracked_ciks: set[str] | None = None):
        self._tracked_ciks = tracked_ciks or set()

    def evaluate_filing(
        self, filing: NewFiling, ttm_alpha: float
    ) -> FilingCandidate:
        """Evaluate a new 13F filing for discovery potential."""
        if filing.cik in self._tracked_ciks:
            return FilingCandidate(
                filing=filing,
                ttm_alpha=ttm_alpha,
                qualifies=False,
                reason="Already tracked",
            )

        if ttm_alpha < ALPHA_THRESHOLD:
            return FilingCandidate(
                filing=filing,
                ttm_alpha=ttm_alpha,
                qualifies=False,
                reason=f"TTM alpha {ttm_alpha:.1%} below threshold {ALPHA_THRESHOLD:.0%}",
            )

        return FilingCandidate(
            filing=filing,
            ttm_alpha=ttm_alpha,
            qualifies=True,
            reason=f"New high-alpha filer: {ttm_alpha:.1%} TTM alpha",
        )

    async def stream(self) -> None:
        """Stream new 13F-HR filings in real-time via sec-api."""
        # TODO: Implement sec-api streaming + Kaleidoscope push notifications
        pass
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_edgar_stream.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/discovery/edgar_stream.py tests/unit/test_edgar_stream.py
git commit -m "feat: EDGAR real-time 13F stream discovery channel"
```

---

## Task 13: Discovery — Congressional Emergence

**Files:**
- Create: `src/evolve_trader/discovery/congressional_emergence.py`
- Create: `tests/unit/test_congressional_emergence.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_congressional_emergence.py
"""Tests for congressional trader emergence detection discovery channel."""
import pytest
from datetime import datetime, timezone
from evolve_trader.discovery.congressional_emergence import (
    CongressionalEmergenceDetector,
    MemberProfile,
    EmergenceEvent,
    EmergenceType,
)


def test_emergence_types():
    """All emergence trigger types are defined."""
    assert EmergenceType.NEW_ACTIVE_TRADER.value == "new_active_trader"
    assert EmergenceType.COMMITTEE_REASSIGNMENT.value == "committee_reassignment"
    assert EmergenceType.LEADERSHIP_PROMOTION.value == "leadership_promotion"


def test_member_profile_has_required_fields():
    """MemberProfile captures member trading characteristics."""
    member = MemberProfile(
        name="Jane Smith",
        party="D",
        state="CA",
        chamber="House",
        committees=["Financial Services", "Technology"],
        trade_count_6mo=25,
        tenure_months=4,
        leadership_role=None,
    )
    assert member.trade_count_6mo == 25
    assert member.tenure_months == 4


def test_detects_new_active_trader():
    """Detects newly seated members trading actively within first 6 months."""
    detector = CongressionalEmergenceDetector()
    member = MemberProfile(
        name="New Member",
        party="R",
        state="TX",
        chamber="Senate",
        committees=["Armed Services"],
        trade_count_6mo=15,
        tenure_months=3,
        leadership_role=None,
    )
    events = detector.evaluate(member)
    assert any(e.emergence_type == EmergenceType.NEW_ACTIVE_TRADER for e in events)


def test_detects_committee_reassignment():
    """Detects committee reassignment to trading-relevant sector."""
    detector = CongressionalEmergenceDetector()
    member = MemberProfile(
        name="Reassigned Member",
        party="D",
        state="NY",
        chamber="House",
        committees=["Financial Services"],  # newly assigned
        trade_count_6mo=8,
        tenure_months=24,
        leadership_role=None,
    )
    events = detector.evaluate(member, previous_committees=["Agriculture"])
    assert any(e.emergence_type == EmergenceType.COMMITTEE_REASSIGNMENT for e in events)


def test_detects_leadership_promotion():
    """Detects leadership promotion — highest priority signal."""
    detector = CongressionalEmergenceDetector()
    member = MemberProfile(
        name="Promoted Leader",
        party="R",
        state="KY",
        chamber="Senate",
        committees=["Finance"],
        trade_count_6mo=10,
        tenure_months=48,
        leadership_role="Majority Whip",
    )
    events = detector.evaluate(member, previous_leadership_role=None)
    assert any(e.emergence_type == EmergenceType.LEADERSHIP_PROMOTION for e in events)
    # Leadership promotion is highest priority
    leadership_events = [e for e in events if e.emergence_type == EmergenceType.LEADERSHIP_PROMOTION]
    assert leadership_events[0].priority == "high"


def test_ignores_inactive_new_member():
    """Does not flag new members who are not actively trading."""
    detector = CongressionalEmergenceDetector()
    member = MemberProfile(
        name="Quiet Member",
        party="D",
        state="VT",
        chamber="Senate",
        committees=["Environment"],
        trade_count_6mo=0,
        tenure_months=3,
        leadership_role=None,
    )
    events = detector.evaluate(member)
    assert len(events) == 0


def test_detector_interface():
    """CongressionalEmergenceDetector has the expected public interface."""
    detector = CongressionalEmergenceDetector()
    assert hasattr(detector, "evaluate")
    assert hasattr(detector, "scan_all")
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_congressional_emergence.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.discovery.congressional_emergence'`

**Step 3: Implement the congressional emergence detector**

```python
# src/evolve_trader/discovery/congressional_emergence.py
"""Congressional trader emergence detection discovery channel.

Monitors:
  - Newly seated members trading actively within first 6 months (Quiver)
  - Committee reassignment to trading-relevant sectors
  - Leadership promotions (highest priority — the Wei & Zhou inflection point)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


TRADING_RELEVANT_COMMITTEES = {
    "Financial Services",
    "Finance",
    "Banking",
    "Energy and Commerce",
    "Ways and Means",
    "Appropriations",
    "Armed Services",
    "Intelligence",
    "Technology",
    "Commerce",
}

NEW_MEMBER_TENURE_THRESHOLD = 6  # months
NEW_MEMBER_TRADE_THRESHOLD = 5  # minimum trades in 6 months


class EmergenceType(str, Enum):
    """Types of congressional emergence triggers."""
    NEW_ACTIVE_TRADER = "new_active_trader"
    COMMITTEE_REASSIGNMENT = "committee_reassignment"
    LEADERSHIP_PROMOTION = "leadership_promotion"


@dataclass
class MemberProfile:
    """Profile of a congressional member for emergence detection."""
    name: str
    party: str  # "D" or "R"
    state: str
    chamber: str  # "House" or "Senate"
    committees: list[str]
    trade_count_6mo: int
    tenure_months: int
    leadership_role: str | None


@dataclass
class EmergenceEvent:
    """A detected emergence event for a congressional member."""
    member: MemberProfile
    emergence_type: EmergenceType
    priority: str  # "high", "medium", "low"
    reason: str
    detected_at: datetime


class CongressionalEmergenceDetector:
    """Discovery channel: congressional trader emergence detection."""

    def evaluate(
        self,
        member: MemberProfile,
        previous_committees: list[str] | None = None,
        previous_leadership_role: str | None = None,
    ) -> list[EmergenceEvent]:
        """Evaluate a member for emergence signals."""
        events = []
        now = datetime.now(timezone.utc)

        # Check for leadership promotion (highest priority)
        if (
            member.leadership_role is not None
            and previous_leadership_role is None
            and previous_leadership_role != member.leadership_role
        ):
            events.append(EmergenceEvent(
                member=member,
                emergence_type=EmergenceType.LEADERSHIP_PROMOTION,
                priority="high",
                reason=f"Promoted to {member.leadership_role}",
                detected_at=now,
            ))

        # Check for committee reassignment to trading-relevant committee
        if previous_committees is not None:
            new_committees = set(member.committees) - set(previous_committees)
            relevant_new = new_committees & TRADING_RELEVANT_COMMITTEES
            if relevant_new:
                events.append(EmergenceEvent(
                    member=member,
                    emergence_type=EmergenceType.COMMITTEE_REASSIGNMENT,
                    priority="medium",
                    reason=f"Reassigned to: {', '.join(relevant_new)}",
                    detected_at=now,
                ))

        # Check for new active trader
        if (
            member.tenure_months <= NEW_MEMBER_TENURE_THRESHOLD
            and member.trade_count_6mo >= NEW_MEMBER_TRADE_THRESHOLD
        ):
            events.append(EmergenceEvent(
                member=member,
                emergence_type=EmergenceType.NEW_ACTIVE_TRADER,
                priority="medium",
                reason=f"New member with {member.trade_count_6mo} trades in {member.tenure_months} months",
                detected_at=now,
            ))

        return events

    async def scan_all(self, members: list[MemberProfile]) -> list[EmergenceEvent]:
        """Scan all members for emergence events."""
        # TODO: Implement Quiver API polling
        all_events = []
        for member in members:
            all_events.extend(self.evaluate(member))
        return all_events
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_congressional_emergence.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/discovery/congressional_emergence.py tests/unit/test_congressional_emergence.py
git commit -m "feat: congressional trader emergence detection with leadership priority"
```

---

## Task 14: Discovery — Insider Cluster Emergence

**Files:**
- Create: `src/evolve_trader/discovery/insider_clusters.py`
- Create: `tests/unit/test_insider_clusters.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_insider_clusters.py
"""Tests for insider cluster emergence detection discovery channel."""
import pytest
from datetime import datetime, timedelta, timezone
from evolve_trader.discovery.insider_clusters import (
    InsiderClusterDetector,
    InsiderFiling,
    ClusterEvent,
    CLUSTER_WINDOW_DAYS,
    CLUSTER_MIN_FILINGS,
)


def test_cluster_parameters():
    """Cluster window is 2 weeks, minimum 3 filings."""
    assert CLUSTER_WINDOW_DAYS == 14
    assert CLUSTER_MIN_FILINGS == 3


def test_insider_filing_has_required_fields():
    """InsiderFiling captures individual Form 4 data."""
    filing = InsiderFiling(
        filer_name="John Doe",
        filer_title="CEO",
        company="Acme Corp",
        ticker="ACME",
        sector="Technology",
        transaction_date=datetime(2026, 3, 15, tzinfo=timezone.utc),
        transaction_type="P",  # Purchase
        shares=10000,
        price=50.0,
        value=500_000.0,
    )
    assert filing.sector == "Technology"
    assert filing.transaction_type == "P"


def test_detects_cluster_in_same_sector():
    """Detects 3+ executives at different companies in same sector within 2 weeks."""
    detector = InsiderClusterDetector()
    base_date = datetime(2026, 3, 10, tzinfo=timezone.utc)
    filings = [
        InsiderFiling(
            filer_name="CEO A", filer_title="CEO", company="Tech Corp A",
            ticker="TECA", sector="Technology",
            transaction_date=base_date,
            transaction_type="P", shares=5000, price=100.0, value=500_000.0,
        ),
        InsiderFiling(
            filer_name="CFO B", filer_title="CFO", company="Tech Corp B",
            ticker="TECB", sector="Technology",
            transaction_date=base_date + timedelta(days=3),
            transaction_type="P", shares=3000, price=80.0, value=240_000.0,
        ),
        InsiderFiling(
            filer_name="CTO C", filer_title="CTO", company="Tech Corp C",
            ticker="TECC", sector="Technology",
            transaction_date=base_date + timedelta(days=7),
            transaction_type="P", shares=4000, price=60.0, value=240_000.0,
        ),
    ]
    clusters = detector.detect_clusters(filings)
    assert len(clusters) >= 1
    cluster = clusters[0]
    assert isinstance(cluster, ClusterEvent)
    assert cluster.sector == "Technology"
    assert cluster.num_filings >= 3
    assert len(cluster.companies) >= 3


def test_no_cluster_insufficient_filings():
    """No cluster when fewer than 3 filings in window."""
    detector = InsiderClusterDetector()
    base_date = datetime(2026, 3, 10, tzinfo=timezone.utc)
    filings = [
        InsiderFiling(
            filer_name="CEO A", filer_title="CEO", company="Tech Corp A",
            ticker="TECA", sector="Technology",
            transaction_date=base_date,
            transaction_type="P", shares=5000, price=100.0, value=500_000.0,
        ),
        InsiderFiling(
            filer_name="CFO B", filer_title="CFO", company="Tech Corp B",
            ticker="TECB", sector="Technology",
            transaction_date=base_date + timedelta(days=3),
            transaction_type="P", shares=3000, price=80.0, value=240_000.0,
        ),
    ]
    clusters = detector.detect_clusters(filings)
    assert len(clusters) == 0


def test_no_cluster_outside_window():
    """No cluster when filings are spread beyond 2-week window."""
    detector = InsiderClusterDetector()
    base_date = datetime(2026, 3, 1, tzinfo=timezone.utc)
    filings = [
        InsiderFiling(
            filer_name="CEO A", filer_title="CEO", company="Tech Corp A",
            ticker="TECA", sector="Technology",
            transaction_date=base_date,
            transaction_type="P", shares=5000, price=100.0, value=500_000.0,
        ),
        InsiderFiling(
            filer_name="CFO B", filer_title="CFO", company="Tech Corp B",
            ticker="TECB", sector="Technology",
            transaction_date=base_date + timedelta(days=5),
            transaction_type="P", shares=3000, price=80.0, value=240_000.0,
        ),
        InsiderFiling(
            filer_name="CTO C", filer_title="CTO", company="Tech Corp C",
            ticker="TECC", sector="Technology",
            transaction_date=base_date + timedelta(days=30),  # outside window
            transaction_type="P", shares=4000, price=60.0, value=240_000.0,
        ),
    ]
    clusters = detector.detect_clusters(filings)
    assert len(clusters) == 0


def test_no_cluster_same_company():
    """Multiple insiders at the SAME company is not a sector cluster."""
    detector = InsiderClusterDetector()
    base_date = datetime(2026, 3, 10, tzinfo=timezone.utc)
    filings = [
        InsiderFiling(
            filer_name="CEO", filer_title="CEO", company="Same Corp",
            ticker="SAME", sector="Technology",
            transaction_date=base_date,
            transaction_type="P", shares=5000, price=100.0, value=500_000.0,
        ),
        InsiderFiling(
            filer_name="CFO", filer_title="CFO", company="Same Corp",
            ticker="SAME", sector="Technology",
            transaction_date=base_date + timedelta(days=1),
            transaction_type="P", shares=3000, price=100.0, value=300_000.0,
        ),
        InsiderFiling(
            filer_name="CTO", filer_title="CTO", company="Same Corp",
            ticker="SAME", sector="Technology",
            transaction_date=base_date + timedelta(days=2),
            transaction_type="P", shares=4000, price=100.0, value=400_000.0,
        ),
    ]
    clusters = detector.detect_clusters(filings)
    assert len(clusters) == 0  # same company, not a sector-level signal


def test_detector_interface():
    """InsiderClusterDetector has the expected public interface."""
    detector = InsiderClusterDetector()
    assert hasattr(detector, "detect_clusters")
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_insider_clusters.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.discovery.insider_clusters'`

**Step 3: Implement the insider cluster detector**

```python
# src/evolve_trader/discovery/insider_clusters.py
"""Insider cluster emergence detection discovery channel.

Detects 3+ executives at different companies in the same sector filing
Form 4 purchases within a 2-week window. Sector-level signal is more
robust than individual insider trades.
Sources: HedgeFollow, Fintel for aggregation.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


CLUSTER_WINDOW_DAYS = 14  # 2-week window
CLUSTER_MIN_FILINGS = 3  # minimum 3 executives


@dataclass
class InsiderFiling:
    """A single Form 4 insider filing."""
    filer_name: str
    filer_title: str
    company: str
    ticker: str
    sector: str
    transaction_date: datetime
    transaction_type: str  # P=Purchase, S=Sale
    shares: int
    price: float
    value: float


@dataclass
class ClusterEvent:
    """A detected insider cluster event."""
    sector: str
    num_filings: int
    companies: list[str]
    tickers: list[str]
    filers: list[str]
    window_start: datetime
    window_end: datetime
    total_value: float
    transaction_type: str  # "P" or "S"


class InsiderClusterDetector:
    """Discovery channel: insider cluster emergence detection."""

    def detect_clusters(
        self, filings: list[InsiderFiling]
    ) -> list[ClusterEvent]:
        """Detect sector-level insider clusters.

        Groups filings by sector and transaction type, then applies
        sliding window to find 3+ different companies within 2 weeks.
        """
        # Group by (sector, transaction_type)
        groups: dict[tuple[str, str], list[InsiderFiling]] = defaultdict(list)
        for filing in filings:
            groups[(filing.sector, filing.transaction_type)].append(filing)

        clusters = []
        for (sector, tx_type), sector_filings in groups.items():
            # Sort by date
            sector_filings.sort(key=lambda f: f.transaction_date)

            # Sliding window approach
            for i, anchor in enumerate(sector_filings):
                window_end = anchor.transaction_date + timedelta(days=CLUSTER_WINDOW_DAYS)
                window_filings = [
                    f for f in sector_filings
                    if anchor.transaction_date <= f.transaction_date <= window_end
                ]

                # Must be from different companies
                unique_companies = set(f.company for f in window_filings)
                if len(unique_companies) >= CLUSTER_MIN_FILINGS:
                    clusters.append(ClusterEvent(
                        sector=sector,
                        num_filings=len(window_filings),
                        companies=sorted(unique_companies),
                        tickers=sorted(set(f.ticker for f in window_filings)),
                        filers=[f.filer_name for f in window_filings],
                        window_start=anchor.transaction_date,
                        window_end=max(f.transaction_date for f in window_filings),
                        total_value=sum(f.value for f in window_filings),
                        transaction_type=tx_type,
                    ))
                    break  # one cluster per sector per scan

        return clusters
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_insider_clusters.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/discovery/insider_clusters.py tests/unit/test_insider_clusters.py
git commit -m "feat: insider cluster emergence detection with sector-level signals"
```

---

## Task 15: Two-Stage Discovery Filter Pipeline

**Files:**
- Create: `src/evolve_trader/discovery/filter_pipeline.py`
- Create: `tests/unit/test_filter_pipeline.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_filter_pipeline.py
"""Tests for the two-stage discovery filter pipeline."""
import pytest
from datetime import datetime, timezone
from evolve_trader.discovery.filter_pipeline import (
    FilterPipeline,
    CoarseFilterResult,
    FineFilterResult,
    DiscoveryCandidate,
    PipelineConfig,
    STEADY_STATE_MIN,
    STEADY_STATE_MAX,
)


def test_steady_state_bounds():
    """Steady state: 50-100 actively tracked sources."""
    assert STEADY_STATE_MIN == 50
    assert STEADY_STATE_MAX == 100


def test_pipeline_config_defaults():
    """Pipeline config has sensible defaults."""
    config = PipelineConfig()
    assert config.coarse_cadence_days == 7  # weekly
    assert config.coarse_return_threshold >= 0.10  # 10% minimum
    assert config.fine_filter_batch_size > 0


def test_discovery_candidate_has_required_fields():
    """DiscoveryCandidate captures a potential new source."""
    candidate = DiscoveryCandidate(
        source_type="13f_filer",
        identifier="CIK-0001234567",
        name="Unknown Alpha Fund",
        discovery_channel="whalewisdom",
        coarse_score=0.85,
        fine_score=None,
        status="coarse_passed",
        discovered_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
    )
    assert candidate.status == "coarse_passed"
    assert candidate.fine_score is None


def test_coarse_filter_passes_high_return():
    """Coarse filter passes candidates above return threshold."""
    pipeline = FilterPipeline()
    result = pipeline.coarse_filter(
        identifier="CIK-0001234567",
        source_type="13f_filer",
        name="High Alpha Fund",
        ttm_return=0.35,
        additional_metrics={"whale_score": 95},
    )
    assert isinstance(result, CoarseFilterResult)
    assert result.passed is True


def test_coarse_filter_rejects_low_return():
    """Coarse filter rejects candidates below return threshold."""
    pipeline = FilterPipeline()
    result = pipeline.coarse_filter(
        identifier="CIK-0009999999",
        source_type="13f_filer",
        name="Low Alpha Fund",
        ttm_return=0.02,
        additional_metrics={},
    )
    assert result.passed is False


def test_fine_filter_uses_llm_analysis():
    """Fine filter performs LLM analysis on candidates passing coarse filter."""
    pipeline = FilterPipeline()
    candidate = DiscoveryCandidate(
        source_type="13f_filer",
        identifier="CIK-0001234567",
        name="Unknown Alpha Fund",
        discovery_channel="edgar_stream",
        coarse_score=0.85,
        fine_score=None,
        status="coarse_passed",
        discovered_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
    )
    # Simulate LLM analysis result
    result = pipeline.fine_filter(
        candidate=candidate,
        trading_pattern_analysis="Concentrated value strategy with low turnover...",
        llm_confidence=0.78,
    )
    assert isinstance(result, FineFilterResult)
    assert result.candidate.fine_score == 0.78
    assert result.candidate.status in ("fine_passed", "fine_rejected")


def test_pipeline_tracks_active_sources():
    """Pipeline maintains count of actively tracked sources."""
    pipeline = FilterPipeline()
    assert pipeline.active_source_count >= 0


def test_pipeline_respects_steady_state_max():
    """Pipeline does not exceed STEADY_STATE_MAX active sources."""
    config = PipelineConfig(coarse_cadence_days=7, coarse_return_threshold=0.10,
                            fine_filter_batch_size=10)
    pipeline = FilterPipeline(config=config)
    # Attempting to add beyond max should be flagged
    assert pipeline.can_add_source(current_count=50) is True
    assert pipeline.can_add_source(current_count=100) is False


def test_pipeline_interface():
    """FilterPipeline has the expected public interface."""
    pipeline = FilterPipeline()
    assert hasattr(pipeline, "coarse_filter")
    assert hasattr(pipeline, "fine_filter")
    assert hasattr(pipeline, "can_add_source")
    assert hasattr(pipeline, "run_coarse_scan")
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_filter_pipeline.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.discovery.filter_pipeline'`

**Step 3: Implement the filter pipeline**

```python
# src/evolve_trader/discovery/filter_pipeline.py
"""Two-stage discovery filter pipeline.

Stage 1 (Coarse): Fast filter on simple return thresholds and structured data.
  Runs weekly across the full universe of 10,000+ candidates.
Stage 2 (Fine): LLM analysis of trading patterns. Only runs on candidates
  passing the coarse filter.

Steady state: actively tracking 50-100 signal sources while continuously
scanning the full universe. Mirrors a quantitative fund's factor zoo approach.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


STEADY_STATE_MIN = 50
STEADY_STATE_MAX = 100


@dataclass
class PipelineConfig:
    """Configuration for the discovery filter pipeline."""
    coarse_cadence_days: int = 7  # run weekly
    coarse_return_threshold: float = 0.10  # 10% TTM minimum
    fine_filter_batch_size: int = 20


@dataclass
class DiscoveryCandidate:
    """A candidate signal source discovered by the pipeline."""
    source_type: str  # 13f_filer, congressional, insider_cluster, etc.
    identifier: str
    name: str
    discovery_channel: str
    coarse_score: float
    fine_score: float | None
    status: str  # coarse_passed, fine_passed, fine_rejected, promoted, demoted
    discovered_at: datetime


@dataclass
class CoarseFilterResult:
    """Result of the coarse (stage 1) filter."""
    identifier: str
    passed: bool
    score: float
    reason: str


@dataclass
class FineFilterResult:
    """Result of the fine (stage 2) LLM-powered filter."""
    candidate: DiscoveryCandidate
    llm_analysis: str
    passed: bool


class FilterPipeline:
    """Two-stage discovery filter pipeline."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        active_sources: list[str] | None = None,
    ):
        self._config = config or PipelineConfig()
        self._active_sources = active_sources or []

    @property
    def active_source_count(self) -> int:
        return len(self._active_sources)

    def can_add_source(self, current_count: int | None = None) -> bool:
        """Check if pipeline can accept new sources within steady state bounds."""
        count = current_count if current_count is not None else self.active_source_count
        return count < STEADY_STATE_MAX

    def coarse_filter(
        self,
        identifier: str,
        source_type: str,
        name: str,
        ttm_return: float,
        additional_metrics: dict[str, Any],
    ) -> CoarseFilterResult:
        """Stage 1: Fast coarse filter on structured data.

        Simple return threshold plus any additional structured metrics.
        """
        if ttm_return < self._config.coarse_return_threshold:
            return CoarseFilterResult(
                identifier=identifier,
                passed=False,
                score=ttm_return,
                reason=f"TTM return {ttm_return:.1%} below threshold {self._config.coarse_return_threshold:.0%}",
            )

        score = ttm_return
        return CoarseFilterResult(
            identifier=identifier,
            passed=True,
            score=score,
            reason=f"Passed coarse filter: {ttm_return:.1%} TTM return",
        )

    def fine_filter(
        self,
        candidate: DiscoveryCandidate,
        trading_pattern_analysis: str,
        llm_confidence: float,
    ) -> FineFilterResult:
        """Stage 2: LLM-powered fine filter on trading patterns.

        Analyzes trading patterns, concentration, conviction, consistency.
        """
        candidate.fine_score = llm_confidence
        passed = llm_confidence >= 0.60
        candidate.status = "fine_passed" if passed else "fine_rejected"

        return FineFilterResult(
            candidate=candidate,
            llm_analysis=trading_pattern_analysis,
            passed=passed,
        )

    async def run_coarse_scan(self, universe: list[dict[str, Any]]) -> list[CoarseFilterResult]:
        """Run the weekly coarse scan across the full universe."""
        # TODO: Implement bulk scanning with structured data queries
        results = []
        for entry in universe:
            result = self.coarse_filter(
                identifier=entry.get("identifier", ""),
                source_type=entry.get("source_type", ""),
                name=entry.get("name", ""),
                ttm_return=entry.get("ttm_return", 0.0),
                additional_metrics=entry.get("metrics", {}),
            )
            results.append(result)
        return results
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_filter_pipeline.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/evolve_trader/discovery/filter_pipeline.py tests/unit/test_filter_pipeline.py
git commit -m "feat: two-stage discovery filter pipeline with coarse and LLM-fine stages"
```

---

## Task 16: Integration Testing & Final Verification

**Files:**
- Create: `tests/integration/test_signal_expansion_integration.py`

**Step 1: Write the integration tests**

```python
# tests/integration/test_signal_expansion_integration.py
"""Integration tests for Phase 9 signal expansion.

Verifies all new signal sources feed into the existing framework:
  - Each source produces valid SignalEvents
  - SignalEvents flow through the signal registry
  - Latency hierarchy is verified
  - Discovery pipeline connects to signal ingestion
"""
import pytest
from datetime import datetime, timedelta, timezone

from evolve_trader.signals.registry import SignalSourceRegistry

# Signal sources
from evolve_trader.signals.sources.ark_trades import ArkTradesSource
from evolve_trader.signals.sources.congressional_etfs import CongressionalEtfSource
from evolve_trader.signals.sources.polymarket import PolymarketSource
from evolve_trader.signals.sources.kalshi import KalshiSource
from evolve_trader.signals.sources.prediction_consensus import PredictionConsensus
from evolve_trader.signals.sources.options_activity import OptionsActivitySource
from evolve_trader.signals.sources.onchain_whales import OnChainWhaleSource
from evolve_trader.signals.sources.investor_letters import InvestorLetterSource
from evolve_trader.signals.sources.news_macro import NewsMacroSource

# Monitoring
from evolve_trader.monitoring.latency_hierarchy import (
    LATENCY_HIERARCHY,
    verify_latency_ordering,
)

# Discovery
from evolve_trader.discovery.whalewisdom import WhaleWisdomScanner
from evolve_trader.discovery.edgar_stream import EdgarStreamMonitor
from evolve_trader.discovery.congressional_emergence import CongressionalEmergenceDetector
from evolve_trader.discovery.insider_clusters import InsiderClusterDetector
from evolve_trader.discovery.filter_pipeline import FilterPipeline


def test_all_new_sources_register_in_registry():
    """All Phase 9 signal sources can be registered in the SignalSourceRegistry."""
    registry = SignalSourceRegistry()
    sources = [
        ArkTradesSource(),
        CongressionalEtfSource(),
        PolymarketSource(),
        KalshiSource(),
        PredictionConsensus(),
        OptionsActivitySource(),
        OnChainWhaleSource(),
        InvestorLetterSource(),
        NewsMacroSource(),
    ]
    for source in sources:
        registry.register(source)

    assert len(registry.get_all_sources()) >= 9
    source_names = {s.source_name for s in registry.get_all_sources()}
    assert "ark_trades" in source_names
    assert "polymarket" in source_names
    assert "kalshi" in source_names
    assert "prediction_consensus" in source_names
    assert "options_activity" in source_names
    assert "onchain_whales" in source_names
    assert "investor_letters" in source_names
    assert "news_macro" in source_names
    assert "congressional_etfs" in source_names


def test_latency_hierarchy_covers_all_sources():
    """Latency hierarchy includes all signal sources including Phase 2 originals."""
    hierarchy_sources = {tier.source_name for tier in LATENCY_HIERARCHY}
    # Phase 9 sources
    assert "polymarket" in hierarchy_sources
    assert "kalshi" in hierarchy_sources
    assert "options_activity" in hierarchy_sources
    assert "onchain_whales" in hierarchy_sources
    assert "ark_trades" in hierarchy_sources
    assert "news_macro" in hierarchy_sources
    # Phase 2 originals
    assert "edgar_13f" in hierarchy_sources
    assert "edgar_form4" in hierarchy_sources
    assert "congressional" in hierarchy_sources


def test_latency_hierarchy_ordering_valid():
    """Hierarchy is properly ordered from fastest to slowest."""
    result = verify_latency_ordering({
        "polymarket": timedelta(minutes=2),
        "kalshi": timedelta(minutes=3),
        "options_activity": timedelta(hours=2),
        "news_macro": timedelta(hours=4),
        "onchain_whales": timedelta(hours=8),
        "ark_trades": timedelta(hours=18),
        "edgar_form4": timedelta(days=2),
        "congressional": timedelta(days=25),
        "edgar_13f": timedelta(days=50),
    })
    assert result.passed is True


def test_discovery_channels_all_instantiate():
    """All discovery channels instantiate without errors."""
    ww = WhaleWisdomScanner()
    edgar = EdgarStreamMonitor()
    congress = CongressionalEmergenceDetector()
    insider = InsiderClusterDetector()
    pipeline = FilterPipeline()

    assert ww is not None
    assert edgar is not None
    assert congress is not None
    assert insider is not None
    assert pipeline is not None


def test_discovery_pipeline_end_to_end():
    """Discovery pipeline: coarse → fine → candidate promotion flow."""
    pipeline = FilterPipeline()

    # Coarse filter
    coarse_result = pipeline.coarse_filter(
        identifier="CIK-0001234567",
        source_type="13f_filer",
        name="Alpha Fund",
        ttm_return=0.42,
        additional_metrics={"whale_score": 95},
    )
    assert coarse_result.passed is True

    # Fine filter (LLM analysis)
    from evolve_trader.discovery.filter_pipeline import DiscoveryCandidate
    candidate = DiscoveryCandidate(
        source_type="13f_filer",
        identifier="CIK-0001234567",
        name="Alpha Fund",
        discovery_channel="edgar_stream",
        coarse_score=coarse_result.score,
        fine_score=None,
        status="coarse_passed",
        discovered_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
    )
    fine_result = pipeline.fine_filter(
        candidate=candidate,
        trading_pattern_analysis="Concentrated value with sector rotation...",
        llm_confidence=0.82,
    )
    assert fine_result.passed is True
    assert fine_result.candidate.status == "fine_passed"
    assert fine_result.candidate.fine_score == 0.82


def test_10k_filers_through_coarse_filter_performance():
    """Scale test: 10,000 synthetic filers through coarse filter."""
    pipeline = FilterPipeline()
    import random
    random.seed(42)

    universe = [
        {
            "identifier": f"CIK-{i:010d}",
            "source_type": "13f_filer",
            "name": f"Fund {i}",
            "ttm_return": random.gauss(0.08, 0.15),
            "metrics": {},
        }
        for i in range(10_000)
    ]

    results = []
    for entry in universe:
        result = pipeline.coarse_filter(
            identifier=entry["identifier"],
            source_type=entry["source_type"],
            name=entry["name"],
            ttm_return=entry["ttm_return"],
            additional_metrics=entry["metrics"],
        )
        results.append(result)

    passed = [r for r in results if r.passed]
    # With mean 8% and threshold 10%, roughly 40-50% should pass
    assert 1000 < len(passed) < 8000  # reasonable range
    assert len(results) == 10_000
```

**Step 2: Run integration tests**

```bash
pytest tests/integration/test_signal_expansion_integration.py -v
```

Expected: PASS — all signal sources register, latency verified, discovery pipeline flows end-to-end, scale test passes.

**Step 3: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: ALL PASS

**Step 4: Commit**

```bash
git add tests/integration/test_signal_expansion_integration.py
git commit -m "test: Phase 9 integration tests — signal expansion and discovery pipeline"
```

---

## Parallelization & Dependency Notes

```
Independent (build in any order or parallel):
├── Task 1: ARK Trades
├── Task 2: NANC/GOP ETFs
├── Task 3: Polymarket
├── Task 4: Kalshi
├── Task 6: Options Activity
├── Task 7: On-Chain Whales
├── Task 8: Investor Letters
├── Task 9: News & Macro
├── Task 11: WhaleWisdom Scan
├── Task 12: EDGAR Stream
├── Task 13: Congressional Emergence
└── Task 14: Insider Clusters

Depends on Tasks 3+4:
└── Task 5: Prediction Consensus (needs Polymarket + Kalshi types)

Depends on Tasks 1-9:
└── Task 10: Latency Hierarchy (needs all source names)

Depends on Tasks 11-14:
└── Task 15: Filter Pipeline (needs discovery channel types)

Depends on everything:
└── Task 16: Integration Testing
```

- Individual signal sources (Tasks 1-4, 6-9) are INDEPENDENT — can all be built in parallel
- Discovery channels (Tasks 11-14) are also independent of each other
- Task 5 depends on Tasks 3+4 (Polymarket + Kalshi types)
- Task 10 depends on all signal source names being finalized
- Task 15 depends on discovery channel types being defined
- Task 16 (integration) depends on everything and must be last
- Target: ~1500-2000 lines of production code + tests
