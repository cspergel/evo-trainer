# Phase 12: Extensions & Polish — Detailed Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the system with crypto-specific capabilities (regime classifier + BITWISE10), IBKR brokerage integration, prediction market direct trading, address five open research questions, deliver Dashboard v2 refinements, integrate disclaimers across all user-facing outputs, and (optionally) prepare for open-source release.

**Architecture:** Crypto regime classification adds a new dimension to the existing regime classifier, with DeFi/NFT/regulatory/halving cycle detection feeding into the MetaSelector. IBKR integration parallels the existing Alpaca execution client behind the unified broker abstraction. Polymarket trading flips prediction markets from signal source (Phase 9) to revenue source, with CLOB API integration for programmatic order placement. Research modules are standalone experiment frameworks that produce documented results. Dashboard v2 extends the Phase 5 dashboard with performance optimizations and mobile views. Disclaimer integration is a cross-cutting concern touching dashboard, notifications, and strategy outputs.

**Tech Stack:** Python 3.11+, PostgreSQL 16+, SQLAlchemy 2.0 (async), ccxt (crypto exchange API), ibapi/ib_insync (IBKR), polymarket-apis (Polymarket CLOB), React/TypeScript (dashboard), pytest, httpx (async HTTP)

**Parent plan:** [`2026-03-29-evolve-trader-master-plan.md`](2026-03-29-evolve-trader-master-plan.md)

**Prerequisites:** Phase 11 complete. Live trading operational via Alpaca. Kill switch, graduated promotion, security hardening all verified. Regime classifier, MetaSelector, signal pipeline, and dashboard v1 all functional.

---

## Task 1: Crypto Regime Classifier

**Files:**
- Create: `src/evolve_trader/regime/crypto_classifier.py`
- Create: `strategies/regime-crypto-v1.skill.md`
- Create: `tests/unit/test_crypto_classifier.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_crypto_classifier.py
"""Tests for crypto-specific regime classifier."""
import pytest
from datetime import datetime, timezone
from evolve_trader.regime.crypto_classifier import (
    CryptoRegimeClassifier,
    CryptoRegime,
    CryptoRegimeSignal,
)


def test_crypto_regime_enum_has_all_regimes():
    """CryptoRegime covers all major crypto market phases."""
    assert CryptoRegime.DEFI_SUMMER in CryptoRegime
    assert CryptoRegime.NFT_MANIA in CryptoRegime
    assert CryptoRegime.REGULATORY_CRACKDOWN in CryptoRegime
    assert CryptoRegime.HALVING_CYCLE_BULL in CryptoRegime
    assert CryptoRegime.HALVING_CYCLE_BEAR in CryptoRegime
    assert CryptoRegime.CRYPTO_WINTER in CryptoRegime
    assert CryptoRegime.NEUTRAL in CryptoRegime


def test_classify_defi_summer_signals():
    """High DeFi TVL growth + yield farming activity → DeFi summer."""
    classifier = CryptoRegimeClassifier()
    signals = [
        CryptoRegimeSignal(
            timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
            defi_tvl_change_30d=0.45,
            nft_volume_change_30d=0.05,
            regulatory_event_count_30d=0,
            btc_days_since_halving=400,
            btc_dominance=0.40,
            altcoin_season_index=0.75,
            stablecoin_inflows_30d=5_000_000_000,
        ),
    ]
    regime = classifier.classify(signals)
    assert regime.label == CryptoRegime.DEFI_SUMMER
    assert regime.confidence >= 0.6


def test_classify_nft_mania_signals():
    """High NFT volume + social media activity → NFT mania."""
    classifier = CryptoRegimeClassifier()
    signals = [
        CryptoRegimeSignal(
            timestamp=datetime(2025, 3, 1, tzinfo=timezone.utc),
            defi_tvl_change_30d=0.05,
            nft_volume_change_30d=0.80,
            regulatory_event_count_30d=0,
            btc_days_since_halving=300,
            btc_dominance=0.38,
            altcoin_season_index=0.85,
            stablecoin_inflows_30d=2_000_000_000,
        ),
    ]
    regime = classifier.classify(signals)
    assert regime.label == CryptoRegime.NFT_MANIA
    assert regime.confidence >= 0.6


def test_classify_regulatory_crackdown():
    """High regulatory event count + negative sentiment → crackdown."""
    classifier = CryptoRegimeClassifier()
    signals = [
        CryptoRegimeSignal(
            timestamp=datetime(2025, 9, 1, tzinfo=timezone.utc),
            defi_tvl_change_30d=-0.20,
            nft_volume_change_30d=-0.30,
            regulatory_event_count_30d=12,
            btc_days_since_halving=500,
            btc_dominance=0.55,
            altcoin_season_index=0.20,
            stablecoin_inflows_30d=-3_000_000_000,
        ),
    ]
    regime = classifier.classify(signals)
    assert regime.label == CryptoRegime.REGULATORY_CRACKDOWN
    assert regime.confidence >= 0.6


def test_classify_halving_cycle_bull():
    """BTC within 6-18 months post-halving + positive momentum → bull."""
    classifier = CryptoRegimeClassifier()
    signals = [
        CryptoRegimeSignal(
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            defi_tvl_change_30d=0.10,
            nft_volume_change_30d=0.10,
            regulatory_event_count_30d=1,
            btc_days_since_halving=270,
            btc_dominance=0.48,
            altcoin_season_index=0.55,
            stablecoin_inflows_30d=8_000_000_000,
        ),
    ]
    regime = classifier.classify(signals)
    assert regime.label == CryptoRegime.HALVING_CYCLE_BULL
    assert regime.confidence >= 0.5


def test_classify_crypto_winter():
    """Negative TVL, negative NFT, stablecoin outflows → winter."""
    classifier = CryptoRegimeClassifier()
    signals = [
        CryptoRegimeSignal(
            timestamp=datetime(2025, 12, 1, tzinfo=timezone.utc),
            defi_tvl_change_30d=-0.35,
            nft_volume_change_30d=-0.50,
            regulatory_event_count_30d=5,
            btc_days_since_halving=900,
            btc_dominance=0.62,
            altcoin_season_index=0.10,
            stablecoin_inflows_30d=-10_000_000_000,
        ),
    ]
    regime = classifier.classify(signals)
    assert regime.label == CryptoRegime.CRYPTO_WINTER
    assert regime.confidence >= 0.6


def test_classifier_returns_neutral_on_mixed_signals():
    """Mixed or weak signals → neutral regime."""
    classifier = CryptoRegimeClassifier()
    signals = [
        CryptoRegimeSignal(
            timestamp=datetime(2025, 7, 1, tzinfo=timezone.utc),
            defi_tvl_change_30d=0.02,
            nft_volume_change_30d=-0.01,
            regulatory_event_count_30d=2,
            btc_days_since_halving=600,
            btc_dominance=0.50,
            altcoin_season_index=0.50,
            stablecoin_inflows_30d=500_000_000,
        ),
    ]
    regime = classifier.classify(signals)
    assert regime.label == CryptoRegime.NEUTRAL


def test_classifier_integrates_with_base_regime():
    """Crypto regime feeds into MetaSelector alongside equity regime."""
    classifier = CryptoRegimeClassifier()
    result = classifier.classify([
        CryptoRegimeSignal(
            timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc),
            defi_tvl_change_30d=0.45,
            nft_volume_change_30d=0.05,
            regulatory_event_count_30d=0,
            btc_days_since_halving=400,
            btc_dominance=0.40,
            altcoin_season_index=0.75,
            stablecoin_inflows_30d=5_000_000_000,
        ),
    ])
    # Must produce a RegimeLabel-compatible output
    assert hasattr(result, "label")
    assert hasattr(result, "confidence")
    assert hasattr(result, "timestamp")
    assert hasattr(result, "to_regime_label")
    regime_label = result.to_regime_label()
    assert regime_label.source == "crypto_classifier"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_crypto_classifier.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.regime.crypto_classifier'`

**Step 3: Implement the crypto regime classifier**

```python
# src/evolve_trader/regime/crypto_classifier.py
"""Crypto-specific regime classifier.

Detects crypto market phases: DeFi summer, NFT mania, regulatory crackdown,
halving cycles (bull/bear), crypto winter. Integrates with the base regime
classifier to feed the MetaSelector.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from evolve_trader.regime.base import RegimeLabel


class CryptoRegime(str, Enum):
    """Crypto-specific market regime classifications."""
    DEFI_SUMMER = "defi_summer"
    NFT_MANIA = "nft_mania"
    REGULATORY_CRACKDOWN = "regulatory_crackdown"
    HALVING_CYCLE_BULL = "halving_cycle_bull"
    HALVING_CYCLE_BEAR = "halving_cycle_bear"
    CRYPTO_WINTER = "crypto_winter"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class CryptoRegimeSignal:
    """Input signals for crypto regime classification."""
    timestamp: datetime
    defi_tvl_change_30d: float       # Percentage change in DeFi TVL
    nft_volume_change_30d: float     # Percentage change in NFT trading volume
    regulatory_event_count_30d: int  # Count of regulatory actions/announcements
    btc_days_since_halving: int      # Days since most recent BTC halving
    btc_dominance: float             # BTC market cap dominance (0-1)
    altcoin_season_index: float      # Altcoin season indicator (0-1)
    stablecoin_inflows_30d: float    # Net stablecoin inflows in USD


@dataclass
class CryptoRegimeResult:
    """Result of crypto regime classification."""
    label: CryptoRegime
    confidence: float
    timestamp: datetime
    contributing_factors: dict[str, float]

    def to_regime_label(self) -> RegimeLabel:
        """Convert to base RegimeLabel for MetaSelector compatibility."""
        return RegimeLabel(
            label=self.label.value,
            confidence=self.confidence,
            timestamp=self.timestamp,
            source="crypto_classifier",
            metadata=self.contributing_factors,
        )


class CryptoRegimeClassifier:
    """Classifies crypto market into regime phases.

    Uses a weighted scoring system across multiple crypto-specific indicators.
    Each regime has characteristic signal patterns:
    - DeFi summer: high TVL growth, low BTC dominance, altcoin season
    - NFT mania: high NFT volume, low BTC dominance, altcoin season
    - Regulatory crackdown: high regulatory events, negative sentiment, outflows
    - Halving bull: 180-540 days post-halving, positive stablecoin inflows
    - Halving bear: 540-1200 days post-halving, negative momentum
    - Crypto winter: negative TVL, negative NFT, outflows, high BTC dominance
    - Neutral: mixed signals, no dominant pattern
    """

    # Halving cycle windows (days post-halving)
    HALVING_BULL_START = 180
    HALVING_BULL_END = 540
    HALVING_BEAR_START = 540
    HALVING_BEAR_END = 1200

    # Thresholds
    DEFI_TVL_THRESHOLD = 0.25
    NFT_VOLUME_THRESHOLD = 0.40
    REGULATORY_EVENT_THRESHOLD = 8
    STABLECOIN_OUTFLOW_THRESHOLD = -2_000_000_000
    BTC_DOMINANCE_HIGH = 0.55
    ALTCOIN_SEASON_THRESHOLD = 0.65

    def classify(self, signals: list[CryptoRegimeSignal]) -> CryptoRegimeResult:
        """Classify current crypto regime from input signals.

        Uses the most recent signal for classification.
        Multiple signals can be provided for smoothing (future enhancement).
        """
        if not signals:
            return CryptoRegimeResult(
                label=CryptoRegime.NEUTRAL,
                confidence=0.0,
                timestamp=datetime.now(timezone.utc),
                contributing_factors={},
            )

        latest = signals[-1]
        scores: dict[CryptoRegime, float] = {}
        factors: dict[str, float] = {}

        # Score each regime
        scores[CryptoRegime.DEFI_SUMMER] = self._score_defi_summer(latest, factors)
        scores[CryptoRegime.NFT_MANIA] = self._score_nft_mania(latest, factors)
        scores[CryptoRegime.REGULATORY_CRACKDOWN] = self._score_regulatory(latest, factors)
        scores[CryptoRegime.HALVING_CYCLE_BULL] = self._score_halving_bull(latest, factors)
        scores[CryptoRegime.HALVING_CYCLE_BEAR] = self._score_halving_bear(latest, factors)
        scores[CryptoRegime.CRYPTO_WINTER] = self._score_crypto_winter(latest, factors)

        # Pick highest-scoring regime
        best_regime = max(scores, key=scores.get)
        best_score = scores[best_regime]

        # If no regime scores above threshold, return neutral
        if best_score < 0.4:
            return CryptoRegimeResult(
                label=CryptoRegime.NEUTRAL,
                confidence=1.0 - best_score,
                timestamp=latest.timestamp,
                contributing_factors=factors,
            )

        return CryptoRegimeResult(
            label=best_regime,
            confidence=min(best_score, 1.0),
            timestamp=latest.timestamp,
            contributing_factors=factors,
        )

    def _score_defi_summer(self, s: CryptoRegimeSignal, factors: dict) -> float:
        score = 0.0
        if s.defi_tvl_change_30d >= self.DEFI_TVL_THRESHOLD:
            score += 0.4
            factors["defi_tvl_growth"] = s.defi_tvl_change_30d
        if s.btc_dominance < 0.45:
            score += 0.2
        if s.altcoin_season_index >= self.ALTCOIN_SEASON_THRESHOLD:
            score += 0.2
        if s.stablecoin_inflows_30d > 3_000_000_000:
            score += 0.2
        return score

    def _score_nft_mania(self, s: CryptoRegimeSignal, factors: dict) -> float:
        score = 0.0
        if s.nft_volume_change_30d >= self.NFT_VOLUME_THRESHOLD:
            score += 0.4
            factors["nft_volume_growth"] = s.nft_volume_change_30d
        if s.btc_dominance < 0.45:
            score += 0.2
        if s.altcoin_season_index >= self.ALTCOIN_SEASON_THRESHOLD:
            score += 0.2
        if s.defi_tvl_change_30d < self.DEFI_TVL_THRESHOLD:
            score += 0.1  # NFT mania often distinct from DeFi summer
        return score

    def _score_regulatory(self, s: CryptoRegimeSignal, factors: dict) -> float:
        score = 0.0
        if s.regulatory_event_count_30d >= self.REGULATORY_EVENT_THRESHOLD:
            score += 0.4
            factors["regulatory_events"] = float(s.regulatory_event_count_30d)
        if s.stablecoin_inflows_30d < self.STABLECOIN_OUTFLOW_THRESHOLD:
            score += 0.3
        if s.defi_tvl_change_30d < -0.10:
            score += 0.2
        if s.btc_dominance > 0.50:
            score += 0.1
        return score

    def _score_halving_bull(self, s: CryptoRegimeSignal, factors: dict) -> float:
        score = 0.0
        if self.HALVING_BULL_START <= s.btc_days_since_halving <= self.HALVING_BULL_END:
            score += 0.4
            factors["halving_cycle_position"] = s.btc_days_since_halving
        if s.stablecoin_inflows_30d > 5_000_000_000:
            score += 0.3
        if s.defi_tvl_change_30d > 0:
            score += 0.15
        if s.altcoin_season_index > 0.40:
            score += 0.15
        return score

    def _score_halving_bear(self, s: CryptoRegimeSignal, factors: dict) -> float:
        score = 0.0
        if self.HALVING_BEAR_START < s.btc_days_since_halving <= self.HALVING_BEAR_END:
            score += 0.3
            factors["halving_bear_position"] = s.btc_days_since_halving
        if s.stablecoin_inflows_30d < 0:
            score += 0.3
        if s.btc_dominance > self.BTC_DOMINANCE_HIGH:
            score += 0.2
        if s.altcoin_season_index < 0.30:
            score += 0.2
        return score

    def _score_crypto_winter(self, s: CryptoRegimeSignal, factors: dict) -> float:
        score = 0.0
        if s.defi_tvl_change_30d < -0.20:
            score += 0.25
        if s.nft_volume_change_30d < -0.30:
            score += 0.20
            factors["nft_volume_decline"] = s.nft_volume_change_30d
        if s.stablecoin_inflows_30d < -5_000_000_000:
            score += 0.25
        if s.btc_dominance > self.BTC_DOMINANCE_HIGH:
            score += 0.15
        if s.altcoin_season_index < 0.20:
            score += 0.15
        return score
```

Also create the SKILL.md:

```markdown
<!-- strategies/regime-crypto-v1.skill.md -->
# Crypto Regime Classifier v1

## Identity
- **Name:** regime-crypto-v1
- **Type:** regime_classifier
- **Version:** 1
- **Parent:** regime-base-v1
- **Status:** incubating

## Purpose
Classify the current crypto market regime to guide strategy selection for
crypto-specific instruments (BITWISE10 universe, individual tokens).

## Regimes Detected
| Regime | Key Indicators | Strategy Implications |
|--------|---------------|----------------------|
| DeFi Summer | TVL growth >25%, low BTC dominance, altcoin season | Long DeFi tokens, yield farming strategies |
| NFT Mania | NFT volume >40% growth, low BTC dominance | NFT-adjacent tokens, marketplace tokens |
| Regulatory Crackdown | >8 regulatory events/month, stablecoin outflows | Risk-off, BTC-only, reduce exposure |
| Halving Bull | 180-540 days post-halving, stablecoin inflows | Momentum-long BTC + large caps |
| Halving Bear | 540-1200 days post-halving, declining momentum | Defensive, stablecoin parking |
| Crypto Winter | Negative TVL/NFT/flows, high BTC dominance | Minimal exposure, accumulation only |
| Neutral | Mixed signals | Balanced allocation, no regime tilt |

## Input Signals
- DeFi TVL 30-day change (DeFiLlama / on-chain)
- NFT trading volume 30-day change (OpenSea / on-chain)
- Regulatory event count (curated news feed)
- BTC days since halving (chain data)
- BTC market dominance (CoinGecko / CoinMarketCap)
- Altcoin season index (composite)
- Stablecoin net inflows (on-chain USDT/USDC/DAI)

## Constraints
- Classification updates at most every 4 hours (crypto markets 24/7)
- Must not flip regime more than once per 48 hours (hysteresis)
- Confidence threshold for regime change: 0.6
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_crypto_classifier.py -v
```

Expected: PASS — all regime classifications match expected outputs

**Step 5: Commit**

```bash
git add src/evolve_trader/regime/crypto_classifier.py strategies/regime-crypto-v1.skill.md tests/unit/test_crypto_classifier.py
git commit -m "feat: crypto regime classifier with DeFi/NFT/regulatory/halving detection"
```

---

## Task 2: BITWISE10 Validation

**Files:**
- Create: `src/evolve_trader/crypto/bitwise10.py`
- Create: `tests/integration/test_bitwise10.py`

**Step 1: Write the failing tests**

```python
# tests/integration/test_bitwise10.py
"""Integration tests for BITWISE10 crypto universe validation."""
import pytest
from datetime import datetime, timezone
from evolve_trader.crypto.bitwise10 import (
    Bitwise10Universe,
    Bitwise10Asset,
    CryptoStrategyValidator,
)
from evolve_trader.regime.crypto_classifier import CryptoRegime


def test_bitwise10_universe_has_top_10_assets():
    """BITWISE10 universe contains top 10 crypto assets by market cap."""
    universe = Bitwise10Universe()
    assets = universe.get_assets()
    assert len(assets) == 10
    # BTC and ETH always present
    tickers = [a.ticker for a in assets]
    assert "BTC" in tickers
    assert "ETH" in tickers


def test_bitwise10_asset_has_required_fields():
    """Each asset has ticker, name, weight, sector classification."""
    asset = Bitwise10Asset(
        ticker="BTC",
        name="Bitcoin",
        weight=0.65,
        sector="store_of_value",
        market_cap=1_200_000_000_000,
        on_chain_signal_sources=["glassnode", "blockchain_com"],
    )
    assert asset.ticker == "BTC"
    assert asset.weight == 0.65
    assert asset.sector == "store_of_value"
    assert len(asset.on_chain_signal_sources) == 2


def test_universe_weights_sum_to_one():
    """Asset weights in the universe sum to approximately 1.0."""
    universe = Bitwise10Universe()
    assets = universe.get_assets()
    total_weight = sum(a.weight for a in assets)
    assert abs(total_weight - 1.0) < 0.01


def test_universe_rebalance_updates_weights():
    """Rebalance recalculates weights based on current market caps."""
    universe = Bitwise10Universe()
    market_caps = {
        "BTC": 1_200_000_000_000,
        "ETH": 400_000_000_000,
        "SOL": 80_000_000_000,
        "BNB": 60_000_000_000,
        "XRP": 50_000_000_000,
        "ADA": 30_000_000_000,
        "AVAX": 20_000_000_000,
        "DOT": 15_000_000_000,
        "LINK": 12_000_000_000,
        "MATIC": 10_000_000_000,
    }
    universe.rebalance(market_caps)
    assets = universe.get_assets()
    btc = next(a for a in assets if a.ticker == "BTC")
    assert btc.weight > 0.50  # BTC dominates


def test_crypto_strategy_validator_requires_crypto_regime():
    """Strategies for crypto must be validated under crypto regimes."""
    validator = CryptoStrategyValidator(universe=Bitwise10Universe())
    result = validator.validate_strategy(
        strategy_name="crypto-momentum-v1",
        backtest_results={
            CryptoRegime.DEFI_SUMMER: {"sharpe": 1.5, "max_drawdown": 0.15},
            CryptoRegime.CRYPTO_WINTER: {"sharpe": -0.3, "max_drawdown": 0.45},
            CryptoRegime.HALVING_CYCLE_BULL: {"sharpe": 2.1, "max_drawdown": 0.10},
        },
    )
    assert result.passed is True  # Positive in 2+ regimes
    assert result.regimes_tested == 3
    assert result.regimes_positive >= 2


def test_crypto_strategy_validator_rejects_single_regime():
    """Strategy positive in only 1 crypto regime fails validation."""
    validator = CryptoStrategyValidator(universe=Bitwise10Universe())
    result = validator.validate_strategy(
        strategy_name="nft-only-v1",
        backtest_results={
            CryptoRegime.NFT_MANIA: {"sharpe": 2.0, "max_drawdown": 0.10},
            CryptoRegime.CRYPTO_WINTER: {"sharpe": -1.0, "max_drawdown": 0.55},
            CryptoRegime.REGULATORY_CRACKDOWN: {"sharpe": -0.8, "max_drawdown": 0.40},
        },
    )
    assert result.passed is False
    assert result.regimes_positive == 1


def test_on_chain_signals_primary_for_crypto():
    """On-chain signals are the primary signal layer for crypto strategies."""
    universe = Bitwise10Universe()
    for asset in universe.get_assets():
        assert len(asset.on_chain_signal_sources) >= 1, (
            f"{asset.ticker} must have at least one on-chain signal source"
        )
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/integration/test_bitwise10.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.crypto.bitwise10'`

**Step 3: Implement BITWISE10 universe and validator**

```python
# src/evolve_trader/crypto/bitwise10.py
"""BITWISE10 crypto universe for strategy evolution validation.

Mirrors AI-Trader's crypto universe: top 10 crypto assets by market cap,
rebalanced monthly. On-chain signals are the primary signal layer for
all crypto strategies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evolve_trader.regime.crypto_classifier import CryptoRegime


@dataclass
class Bitwise10Asset:
    """A single asset in the BITWISE10 universe."""
    ticker: str
    name: str
    weight: float
    sector: str  # store_of_value, smart_contract, defi, layer2, oracle, etc.
    market_cap: float = 0.0
    on_chain_signal_sources: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of crypto strategy validation across regimes."""
    strategy_name: str
    passed: bool
    regimes_tested: int
    regimes_positive: int
    details: dict[str, Any] = field(default_factory=dict)


class Bitwise10Universe:
    """BITWISE10 crypto universe — top 10 assets by market cap.

    Default composition reflects a typical snapshot. Call rebalance()
    with current market caps to update weights.
    """

    DEFAULT_ASSETS = [
        Bitwise10Asset("BTC", "Bitcoin", 0.64, "store_of_value", 1_200_000_000_000, ["glassnode", "blockchain_com"]),
        Bitwise10Asset("ETH", "Ethereum", 0.18, "smart_contract", 400_000_000_000, ["glassnode", "etherscan"]),
        Bitwise10Asset("SOL", "Solana", 0.05, "smart_contract", 80_000_000_000, ["solscan"]),
        Bitwise10Asset("BNB", "BNB", 0.04, "smart_contract", 60_000_000_000, ["bscscan"]),
        Bitwise10Asset("XRP", "XRP", 0.03, "payments", 50_000_000_000, ["xrpscan"]),
        Bitwise10Asset("ADA", "Cardano", 0.02, "smart_contract", 30_000_000_000, ["cardanoscan"]),
        Bitwise10Asset("AVAX", "Avalanche", 0.015, "smart_contract", 20_000_000_000, ["snowtrace"]),
        Bitwise10Asset("DOT", "Polkadot", 0.01, "interoperability", 15_000_000_000, ["subscan"]),
        Bitwise10Asset("LINK", "Chainlink", 0.008, "oracle", 12_000_000_000, ["etherscan"]),
        Bitwise10Asset("MATIC", "Polygon", 0.007, "layer2", 10_000_000_000, ["polygonscan"]),
    ]

    def __init__(self) -> None:
        self._assets: list[Bitwise10Asset] = [
            Bitwise10Asset(
                ticker=a.ticker,
                name=a.name,
                weight=a.weight,
                sector=a.sector,
                market_cap=a.market_cap,
                on_chain_signal_sources=list(a.on_chain_signal_sources),
            )
            for a in self.DEFAULT_ASSETS
        ]

    def get_assets(self) -> list[Bitwise10Asset]:
        """Return all assets in the universe."""
        return list(self._assets)

    def rebalance(self, market_caps: dict[str, float]) -> None:
        """Rebalance weights based on current market caps."""
        total_cap = sum(market_caps.values())
        if total_cap == 0:
            return
        for asset in self._assets:
            if asset.ticker in market_caps:
                asset.market_cap = market_caps[asset.ticker]
                asset.weight = market_caps[asset.ticker] / total_cap


class CryptoStrategyValidator:
    """Validates crypto strategies across multiple crypto regimes.

    Mirrors the equity regime diversity requirement: a strategy must show
    positive performance (Sharpe > 0) in at least 2 distinct crypto regimes
    to pass validation.
    """

    MIN_POSITIVE_REGIMES = 2

    def __init__(self, universe: Bitwise10Universe) -> None:
        self._universe = universe

    def validate_strategy(
        self,
        strategy_name: str,
        backtest_results: dict[CryptoRegime, dict[str, float]],
    ) -> ValidationResult:
        """Validate strategy across crypto regimes.

        Args:
            strategy_name: Name of the strategy being validated.
            backtest_results: Map of regime → performance metrics.
                Each value must contain at least 'sharpe' key.
        """
        regimes_positive = sum(
            1 for metrics in backtest_results.values()
            if metrics.get("sharpe", 0) > 0
        )
        return ValidationResult(
            strategy_name=strategy_name,
            passed=regimes_positive >= self.MIN_POSITIVE_REGIMES,
            regimes_tested=len(backtest_results),
            regimes_positive=regimes_positive,
            details={
                regime.value: metrics
                for regime, metrics in backtest_results.items()
            },
        )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/integration/test_bitwise10.py -v
```

Expected: PASS — universe composition, rebalancing, and validation all work

**Step 5: Commit**

```bash
git add src/evolve_trader/crypto/bitwise10.py tests/integration/test_bitwise10.py
git commit -m "feat: BITWISE10 crypto universe with regime-aware strategy validation"
```

---

## Task 3: IBKR Integration

**Files:**
- Create: `src/evolve_trader/execution/ibkr_client.py`
- Create: `src/evolve_trader/execution/ibkr_order_mapper.py`
- Create: `tests/unit/test_ibkr_client.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_ibkr_client.py
"""Tests for Interactive Brokers integration."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from evolve_trader.execution.ibkr_client import (
    IBKRClient,
    IBKRConfig,
    IBKRConnectionState,
    IBKRPosition,
    IBKROrderStatus,
)
from evolve_trader.execution.ibkr_order_mapper import (
    IBKROrderMapper,
    IBKROrderType,
    IBKRContract,
)


def test_ibkr_config_defaults():
    """IBKR config has sensible defaults for paper trading."""
    config = IBKRConfig()
    assert config.host == "127.0.0.1"
    assert config.port == 7497  # Paper trading port
    assert config.client_id == 1
    assert config.is_paper is True


def test_ibkr_config_live():
    """IBKR config for live trading uses port 7496."""
    config = IBKRConfig(port=7496, is_paper=False)
    assert config.port == 7496
    assert config.is_paper is False


def test_ibkr_connection_states():
    """Connection state enum has all expected states."""
    assert IBKRConnectionState.DISCONNECTED.value == "disconnected"
    assert IBKRConnectionState.CONNECTING.value == "connecting"
    assert IBKRConnectionState.CONNECTED.value == "connected"
    assert IBKRConnectionState.ERROR.value == "error"


@pytest.mark.asyncio
async def test_ibkr_client_connect():
    """Client connects to TWS/IB Gateway."""
    with patch("evolve_trader.execution.ibkr_client.IB") as mock_ib_class:
        mock_ib = AsyncMock()
        mock_ib_class.return_value = mock_ib
        mock_ib.connectAsync = AsyncMock()
        mock_ib.isConnected.return_value = True

        config = IBKRConfig()
        client = IBKRClient(config)
        await client.connect()

        assert client.state == IBKRConnectionState.CONNECTED
        mock_ib.connectAsync.assert_called_once_with(
            host="127.0.0.1", port=7497, clientId=1
        )


@pytest.mark.asyncio
async def test_ibkr_client_disconnect():
    """Client disconnects cleanly."""
    with patch("evolve_trader.execution.ibkr_client.IB") as mock_ib_class:
        mock_ib = AsyncMock()
        mock_ib_class.return_value = mock_ib
        mock_ib.isConnected.return_value = True

        config = IBKRConfig()
        client = IBKRClient(config)
        client._state = IBKRConnectionState.CONNECTED
        await client.disconnect()

        assert client.state == IBKRConnectionState.DISCONNECTED
        mock_ib.disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_ibkr_client_submit_order():
    """Client submits order and returns status."""
    with patch("evolve_trader.execution.ibkr_client.IB") as mock_ib_class:
        mock_ib = AsyncMock()
        mock_ib_class.return_value = mock_ib
        mock_ib.isConnected.return_value = True

        mock_trade = MagicMock()
        mock_trade.orderStatus.status = "Submitted"
        mock_trade.orderStatus.orderId = 42
        mock_trade.orderStatus.filled = 0.0
        mock_trade.orderStatus.remaining = 10.0
        mock_ib.placeOrder = MagicMock(return_value=mock_trade)

        config = IBKRConfig()
        client = IBKRClient(config)
        client._state = IBKRConnectionState.CONNECTED

        contract = IBKRContract(symbol="AAPL", sec_type="STK", exchange="SMART", currency="USD")
        status = await client.submit_order(
            contract=contract,
            order_type=IBKROrderType.MARKET,
            direction="BUY",
            quantity=10.0,
        )
        assert status.order_id == 42
        assert status.status == "Submitted"


@pytest.mark.asyncio
async def test_ibkr_client_get_positions():
    """Client retrieves current positions."""
    with patch("evolve_trader.execution.ibkr_client.IB") as mock_ib_class:
        mock_ib = AsyncMock()
        mock_ib_class.return_value = mock_ib
        mock_ib.isConnected.return_value = True

        mock_pos = MagicMock()
        mock_pos.contract.symbol = "AAPL"
        mock_pos.position = 50.0
        mock_pos.avgCost = 150.0
        mock_pos.marketValue = 8000.0
        mock_ib.positions = MagicMock(return_value=[mock_pos])

        config = IBKRConfig()
        client = IBKRClient(config)
        client._state = IBKRConnectionState.CONNECTED

        positions = await client.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].quantity == 50.0


def test_order_mapper_market_order():
    """Mapper converts internal order to IBKR market order."""
    mapper = IBKROrderMapper()
    contract, order = mapper.map_order(
        symbol="AAPL",
        direction="BUY",
        quantity=10.0,
        order_type=IBKROrderType.MARKET,
    )
    assert contract.symbol == "AAPL"
    assert contract.sec_type == "STK"
    assert order.action == "BUY"
    assert order.totalQuantity == 10.0
    assert order.orderType == "MKT"


def test_order_mapper_limit_order():
    """Mapper converts internal order to IBKR limit order."""
    mapper = IBKROrderMapper()
    contract, order = mapper.map_order(
        symbol="AAPL",
        direction="SELL",
        quantity=5.0,
        order_type=IBKROrderType.LIMIT,
        limit_price=155.00,
    )
    assert order.action == "SELL"
    assert order.orderType == "LMT"
    assert order.lmtPrice == 155.00


def test_order_mapper_supports_futures():
    """Mapper handles futures contracts."""
    mapper = IBKROrderMapper()
    contract, order = mapper.map_order(
        symbol="ES",
        direction="BUY",
        quantity=1.0,
        order_type=IBKROrderType.MARKET,
        sec_type="FUT",
        exchange="CME",
        expiry="202506",
    )
    assert contract.sec_type == "FUT"
    assert contract.exchange == "CME"
    assert contract.lastTradeDateOrContractMonth == "202506"


def test_order_mapper_supports_forex():
    """Mapper handles forex pairs."""
    mapper = IBKROrderMapper()
    contract, order = mapper.map_order(
        symbol="EUR",
        direction="BUY",
        quantity=100000.0,
        order_type=IBKROrderType.MARKET,
        sec_type="CASH",
        exchange="IDEALPRO",
        currency="USD",
    )
    assert contract.sec_type == "CASH"
    assert contract.exchange == "IDEALPRO"


@pytest.mark.asyncio
async def test_ibkr_client_cancel_all_orders():
    """Client cancels all open orders (kill switch support)."""
    with patch("evolve_trader.execution.ibkr_client.IB") as mock_ib_class:
        mock_ib = AsyncMock()
        mock_ib_class.return_value = mock_ib
        mock_ib.isConnected.return_value = True
        mock_ib.reqGlobalCancel = MagicMock()

        config = IBKRConfig()
        client = IBKRClient(config)
        client._state = IBKRConnectionState.CONNECTED

        await client.cancel_all_orders()
        mock_ib.reqGlobalCancel.assert_called_once()
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_ibkr_client.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.execution.ibkr_client'`

**Step 3: Implement the IBKR client and order mapper**

```python
# src/evolve_trader/execution/ibkr_client.py
"""Interactive Brokers integration via ib_insync.

Supports TWS API and IB Gateway connections for 170+ markets including
equities, futures, forex, and bonds. Paper trading on port 7497,
live trading on port 7496.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ib_insync import IB, Contract, MarketOrder, LimitOrder, Trade

from evolve_trader.execution.ibkr_order_mapper import (
    IBKRContract,
    IBKROrderType,
    IBKROrderMapper,
)


class IBKRConnectionState(str, Enum):
    """Connection state for IBKR client."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class IBKRConfig:
    """Configuration for IBKR connection."""
    host: str = "127.0.0.1"
    port: int = 7497  # 7497 = paper, 7496 = live
    client_id: int = 1
    is_paper: bool = True
    timeout: int = 30


@dataclass
class IBKRPosition:
    """Current position from IBKR."""
    symbol: str
    quantity: float
    avg_cost: float
    market_value: float


@dataclass
class IBKROrderStatus:
    """Status of a submitted IBKR order."""
    order_id: int
    status: str
    filled: float = 0.0
    remaining: float = 0.0


class IBKRClient:
    """Client for Interactive Brokers TWS API / IB Gateway.

    Wraps ib_insync for async operation. Supports:
    - Equities (STK), Futures (FUT), Forex (CASH), Bonds (BOND)
    - Market, Limit, Stop, StopLimit order types
    - Position queries and global cancel (kill switch)
    """

    def __init__(self, config: IBKRConfig) -> None:
        self._config = config
        self._ib = IB()
        self._state = IBKRConnectionState.DISCONNECTED
        self._mapper = IBKROrderMapper()

    @property
    def state(self) -> IBKRConnectionState:
        return self._state

    async def connect(self) -> None:
        """Connect to TWS or IB Gateway."""
        self._state = IBKRConnectionState.CONNECTING
        try:
            await self._ib.connectAsync(
                host=self._config.host,
                port=self._config.port,
                clientId=self._config.client_id,
            )
            if self._ib.isConnected():
                self._state = IBKRConnectionState.CONNECTED
            else:
                self._state = IBKRConnectionState.ERROR
        except Exception:
            self._state = IBKRConnectionState.ERROR
            raise

    async def disconnect(self) -> None:
        """Disconnect from TWS/IB Gateway."""
        self._ib.disconnect()
        self._state = IBKRConnectionState.DISCONNECTED

    async def submit_order(
        self,
        contract: IBKRContract,
        order_type: IBKROrderType,
        direction: str,
        quantity: float,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> IBKROrderStatus:
        """Submit an order to IBKR.

        Args:
            contract: Contract specification.
            order_type: Market, limit, stop, or stop-limit.
            direction: BUY or SELL.
            quantity: Number of units.
            limit_price: Limit price (for LIMIT and STOP_LIMIT).
            stop_price: Stop price (for STOP and STOP_LIMIT).
        """
        ib_contract = Contract(
            symbol=contract.symbol,
            secType=contract.sec_type,
            exchange=contract.exchange,
            currency=contract.currency,
        )

        if order_type == IBKROrderType.MARKET:
            ib_order = MarketOrder(direction, quantity)
        elif order_type == IBKROrderType.LIMIT:
            ib_order = LimitOrder(direction, quantity, limit_price)
        else:
            ib_order = MarketOrder(direction, quantity)

        trade: Trade = self._ib.placeOrder(ib_contract, ib_order)
        return IBKROrderStatus(
            order_id=trade.orderStatus.orderId,
            status=trade.orderStatus.status,
            filled=trade.orderStatus.filled,
            remaining=trade.orderStatus.remaining,
        )

    async def get_positions(self) -> list[IBKRPosition]:
        """Get all current positions."""
        raw_positions = self._ib.positions()
        return [
            IBKRPosition(
                symbol=pos.contract.symbol,
                quantity=pos.position,
                avg_cost=pos.avgCost,
                market_value=pos.marketValue,
            )
            for pos in raw_positions
        ]

    async def cancel_all_orders(self) -> None:
        """Cancel all open orders globally. Used by kill switch."""
        self._ib.reqGlobalCancel()
```

```python
# src/evolve_trader/execution/ibkr_order_mapper.py
"""Order mapper for Interactive Brokers.

Converts internal order representations to IBKR-compatible contracts and
orders. Supports STK (equities), FUT (futures), CASH (forex), BOND.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any


class IBKROrderType(str, Enum):
    """Supported IBKR order types."""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"


@dataclass
class IBKRContract:
    """IBKR contract specification."""
    symbol: str
    sec_type: str = "STK"
    exchange: str = "SMART"
    currency: str = "USD"
    lastTradeDateOrContractMonth: Optional[str] = None


@dataclass
class IBKRMappedOrder:
    """Mapped order ready for IBKR submission."""
    action: str          # BUY, SELL
    totalQuantity: float
    orderType: str       # MKT, LMT, STP, STP LMT
    lmtPrice: Optional[float] = None
    auxPrice: Optional[float] = None  # Stop price


class IBKROrderMapper:
    """Maps internal order parameters to IBKR contract + order objects.

    Supports:
    - Equities: STK on SMART routing
    - Futures: FUT with exchange and expiry
    - Forex: CASH on IDEALPRO
    - Bonds: BOND
    """

    ORDER_TYPE_MAP = {
        IBKROrderType.MARKET: "MKT",
        IBKROrderType.LIMIT: "LMT",
        IBKROrderType.STOP: "STP",
        IBKROrderType.STOP_LIMIT: "STP LMT",
    }

    def map_order(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        order_type: IBKROrderType,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        sec_type: str = "STK",
        exchange: str = "SMART",
        currency: str = "USD",
        expiry: Optional[str] = None,
    ) -> tuple[IBKRContract, IBKRMappedOrder]:
        """Map internal order to IBKR contract and order.

        Args:
            symbol: Instrument symbol (e.g., AAPL, ES, EUR).
            direction: BUY or SELL.
            quantity: Number of units/contracts/lots.
            order_type: Order type enum.
            limit_price: Limit price for LIMIT/STOP_LIMIT orders.
            stop_price: Stop trigger price for STOP/STOP_LIMIT orders.
            sec_type: Security type (STK, FUT, CASH, BOND).
            exchange: Exchange (SMART, CME, IDEALPRO, etc.).
            currency: Order currency.
            expiry: Futures expiry (YYYYMM format).
        """
        contract = IBKRContract(
            symbol=symbol,
            sec_type=sec_type,
            exchange=exchange,
            currency=currency,
            lastTradeDateOrContractMonth=expiry,
        )

        order = IBKRMappedOrder(
            action=direction,
            totalQuantity=quantity,
            orderType=self.ORDER_TYPE_MAP[order_type],
            lmtPrice=limit_price,
            auxPrice=stop_price,
        )

        return contract, order
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_ibkr_client.py -v
```

Expected: PASS — all connection, order, position, and mapper tests pass (mocked IB)

**Step 5: Commit**

```bash
git add src/evolve_trader/execution/ibkr_client.py src/evolve_trader/execution/ibkr_order_mapper.py tests/unit/test_ibkr_client.py
git commit -m "feat: IBKR integration with TWS API support for equities, futures, forex"
```

---

## Task 4: Prediction Market Direct Trading

**Files:**
- Create: `src/evolve_trader/execution/polymarket_trader.py`
- Create: `strategies/prediction-market/prediction-arb-v1.skill.md`
- Create: `tests/unit/test_polymarket_trader.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_polymarket_trader.py
"""Tests for Polymarket direct trading integration."""
import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone
from decimal import Decimal
from evolve_trader.execution.polymarket_trader import (
    PolymarketTrader,
    PolymarketConfig,
    PolymarketMarket,
    PolymarketOrder,
    PolymarketOrderSide,
    PolymarketPosition,
    ModelDivergenceSignal,
)


def test_polymarket_config_defaults():
    """Polymarket config has reasonable defaults."""
    config = PolymarketConfig(api_key="test_key", private_key="0xtest")
    assert config.base_url == "https://clob.polymarket.com"
    assert config.chain_id == 137  # Polygon
    assert config.max_position_usd == 1000.0


def test_polymarket_market_has_required_fields():
    """Market object captures all fields needed for trading decisions."""
    market = PolymarketMarket(
        condition_id="0xabc123",
        question="Will BTC exceed $100k by 2026-06-30?",
        outcomes=["Yes", "No"],
        outcome_prices=[Decimal("0.65"), Decimal("0.35")],
        volume_24h=500_000.0,
        liquidity=200_000.0,
        end_date=datetime(2026, 6, 30, tzinfo=timezone.utc),
    )
    assert market.question.startswith("Will BTC")
    assert len(market.outcomes) == 2
    assert market.outcome_prices[0] + market.outcome_prices[1] == Decimal("1.00")


def test_model_divergence_detects_arbitrage():
    """Divergence between model probability and market price → signal."""
    signal = ModelDivergenceSignal(
        condition_id="0xabc123",
        outcome_index=0,
        model_probability=0.80,
        market_price=0.65,
        divergence=0.15,
        confidence=0.85,
        timestamp=datetime.now(timezone.utc),
    )
    assert signal.divergence == 0.15
    assert signal.is_actionable(min_divergence=0.10)
    assert not signal.is_actionable(min_divergence=0.20)


@pytest.mark.asyncio
async def test_polymarket_trader_get_markets():
    """Trader fetches available markets from CLOB API."""
    with patch("evolve_trader.execution.polymarket_trader.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock()
        mock_client.get = AsyncMock(return_value=AsyncMock(
            status_code=200,
            json=lambda: [
                {
                    "condition_id": "0xabc123",
                    "question": "Will BTC exceed $100k?",
                    "outcomes": ["Yes", "No"],
                    "outcome_prices": ["0.65", "0.35"],
                    "volume_24h": 500000,
                    "liquidity": 200000,
                    "end_date_iso": "2026-06-30T00:00:00Z",
                }
            ],
        ))

        config = PolymarketConfig(api_key="test", private_key="0xtest")
        trader = PolymarketTrader(config)
        markets = await trader.get_markets(limit=10)
        assert len(markets) >= 1
        assert markets[0].condition_id == "0xabc123"


@pytest.mark.asyncio
async def test_polymarket_trader_place_order():
    """Trader places limit order on Polymarket CLOB."""
    with patch("evolve_trader.execution.polymarket_trader.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock()
        mock_client.post = AsyncMock(return_value=AsyncMock(
            status_code=200,
            json=lambda: {"order_id": "order_123", "status": "LIVE"},
        ))

        config = PolymarketConfig(api_key="test", private_key="0xtest")
        trader = PolymarketTrader(config)
        order = await trader.place_order(
            condition_id="0xabc123",
            outcome_index=0,
            side=PolymarketOrderSide.BUY,
            price=Decimal("0.70"),
            size=Decimal("100.00"),
        )
        assert order.order_id == "order_123"
        assert order.status == "LIVE"


@pytest.mark.asyncio
async def test_polymarket_trader_get_positions():
    """Trader retrieves current positions."""
    with patch("evolve_trader.execution.polymarket_trader.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock()
        mock_client.get = AsyncMock(return_value=AsyncMock(
            status_code=200,
            json=lambda: [
                {
                    "condition_id": "0xabc123",
                    "outcome_index": 0,
                    "size": "100.00",
                    "avg_price": "0.65",
                    "current_price": "0.70",
                    "pnl": "5.00",
                }
            ],
        ))

        config = PolymarketConfig(api_key="test", private_key="0xtest")
        trader = PolymarketTrader(config)
        positions = await trader.get_positions()
        assert len(positions) == 1
        assert positions[0].condition_id == "0xabc123"
        assert positions[0].pnl == Decimal("5.00")


def test_polymarket_trader_respects_max_position():
    """Trader enforces maximum position size."""
    config = PolymarketConfig(
        api_key="test", private_key="0xtest", max_position_usd=500.0
    )
    trader = PolymarketTrader(config)
    assert trader.max_order_size(current_exposure=300.0) == 200.0
    assert trader.max_order_size(current_exposure=500.0) == 0.0
    assert trader.max_order_size(current_exposure=600.0) == 0.0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_polymarket_trader.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.execution.polymarket_trader'`

**Step 3: Implement the Polymarket trader**

```python
# src/evolve_trader/execution/polymarket_trader.py
"""Polymarket direct trading via CLOB API.

Trades prediction markets as a revenue source. When the system's model
probability diverges from Polymarket pricing, the divergence represents
an arbitrage opportunity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional

import httpx


class PolymarketOrderSide(str, Enum):
    """Order side on Polymarket."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class PolymarketConfig:
    """Configuration for Polymarket CLOB API."""
    api_key: str
    private_key: str
    base_url: str = "https://clob.polymarket.com"
    chain_id: int = 137  # Polygon
    max_position_usd: float = 1000.0


@dataclass
class PolymarketMarket:
    """A Polymarket prediction market."""
    condition_id: str
    question: str
    outcomes: list[str]
    outcome_prices: list[Decimal]
    volume_24h: float
    liquidity: float
    end_date: datetime


@dataclass
class PolymarketOrder:
    """Result of a Polymarket order placement."""
    order_id: str
    status: str
    condition_id: str = ""
    side: str = ""
    price: Decimal = Decimal("0")
    size: Decimal = Decimal("0")


@dataclass
class PolymarketPosition:
    """Current position on Polymarket."""
    condition_id: str
    outcome_index: int
    size: Decimal
    avg_price: Decimal
    current_price: Decimal
    pnl: Decimal


@dataclass
class ModelDivergenceSignal:
    """Signal when model probability diverges from market price."""
    condition_id: str
    outcome_index: int
    model_probability: float
    market_price: float
    divergence: float
    confidence: float
    timestamp: datetime

    def is_actionable(self, min_divergence: float = 0.10) -> bool:
        """Check if divergence is large enough to trade."""
        return abs(self.divergence) >= min_divergence


class PolymarketTrader:
    """Trades prediction markets on Polymarket via CLOB API.

    Key features:
    - Fetch active markets and prices
    - Place limit orders on outcome tokens
    - Monitor positions and P&L
    - Position size limits for risk management
    - Model divergence → arbitrage signal generation
    """

    def __init__(self, config: PolymarketConfig) -> None:
        self._config = config

    def max_order_size(self, current_exposure: float) -> float:
        """Calculate maximum additional order size given current exposure."""
        remaining = self._config.max_position_usd - current_exposure
        return max(0.0, remaining)

    async def get_markets(self, limit: int = 100) -> list[PolymarketMarket]:
        """Fetch available prediction markets from CLOB API."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self._config.base_url}/markets",
                params={"limit": limit},
                headers={"Authorization": f"Bearer {self._config.api_key}"},
            )
            response.raise_for_status()
            data = response.json()

        markets = []
        for item in data:
            markets.append(PolymarketMarket(
                condition_id=item["condition_id"],
                question=item["question"],
                outcomes=item["outcomes"],
                outcome_prices=[Decimal(p) for p in item["outcome_prices"]],
                volume_24h=item.get("volume_24h", 0),
                liquidity=item.get("liquidity", 0),
                end_date=datetime.fromisoformat(
                    item["end_date_iso"].replace("Z", "+00:00")
                ),
            ))
        return markets

    async def place_order(
        self,
        condition_id: str,
        outcome_index: int,
        side: PolymarketOrderSide,
        price: Decimal,
        size: Decimal,
    ) -> PolymarketOrder:
        """Place a limit order on Polymarket CLOB."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._config.base_url}/orders",
                json={
                    "condition_id": condition_id,
                    "outcome_index": outcome_index,
                    "side": side.value,
                    "price": str(price),
                    "size": str(size),
                },
                headers={
                    "Authorization": f"Bearer {self._config.api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()

        return PolymarketOrder(
            order_id=data["order_id"],
            status=data["status"],
            condition_id=condition_id,
            side=side.value,
            price=price,
            size=size,
        )

    async def get_positions(self) -> list[PolymarketPosition]:
        """Get all current Polymarket positions."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self._config.base_url}/positions",
                headers={"Authorization": f"Bearer {self._config.api_key}"},
            )
            response.raise_for_status()
            data = response.json()

        return [
            PolymarketPosition(
                condition_id=item["condition_id"],
                outcome_index=item["outcome_index"],
                size=Decimal(item["size"]),
                avg_price=Decimal(item["avg_price"]),
                current_price=Decimal(item["current_price"]),
                pnl=Decimal(item["pnl"]),
            )
            for item in data
        ]
```

Also create the prediction market strategy SKILL.md:

```markdown
<!-- strategies/prediction-market/prediction-arb-v1.skill.md -->
# Prediction Market Arbitrage v1

## Identity
- **Name:** prediction-arb-v1
- **Type:** prediction_market
- **Version:** 1
- **Parent:** None (seed strategy)
- **Status:** incubating

## Purpose
Exploit divergences between Evolve-Trader's model probabilities and
Polymarket prices. When the system's regime-informed model assigns a
significantly different probability than the market, trade the outcome token.

## Entry Conditions
- Model divergence >= 10% from Polymarket price
- Model confidence >= 0.7
- Market liquidity >= $50,000
- Time to resolution >= 7 days

## Exit Conditions
- Divergence narrows to < 3%
- Time to resolution < 2 days (let settle)
- Max holding period: 30 days
- Stop loss: 25% of position value

## Risk Constraints
- Max position per market: $1,000
- Max total prediction market exposure: $5,000
- Max 10 simultaneous positions
- No markets with < $10,000 liquidity

## Signal Sources
- Internal model probabilities (from regime classifier + signals)
- Polymarket CLOB prices (real-time)
- Polymarket volume and liquidity metrics

## Performance Targets
- Win rate: > 55%
- Average profit per correct trade: > 15%
- Sharpe: > 1.0
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_polymarket_trader.py -v
```

Expected: PASS — all Polymarket trader tests pass (mocked API)

**Step 5: Commit**

```bash
git add src/evolve_trader/execution/polymarket_trader.py strategies/prediction-market/ tests/unit/test_polymarket_trader.py
git commit -m "feat: Polymarket direct trading with CLOB API and divergence-based arbitrage"
```

---

## Task 5: Research — Evolution Cadence

**Files:**
- Create: `src/evolve_trader/research/evolution_cadence.py`
- Create: `tests/research/test_evolution_cadence.py`

**Step 1: Write the failing tests**

```python
# tests/research/test_evolution_cadence.py
"""Tests for evolution cadence research experiment framework."""
import pytest
from datetime import datetime, timezone, timedelta
from evolve_trader.research.evolution_cadence import (
    CadenceExperiment,
    CadenceType,
    CadenceResult,
    EventTrigger,
)


def test_cadence_type_enum():
    """All cadence types are defined."""
    assert CadenceType.DAILY in CadenceType
    assert CadenceType.WEEKLY in CadenceType
    assert CadenceType.EVENT_TRIGGERED in CadenceType
    assert CadenceType.HYBRID in CadenceType


def test_cadence_experiment_setup():
    """Experiment configures cadence and evaluation window."""
    experiment = CadenceExperiment(
        cadence=CadenceType.DAILY,
        evaluation_window_days=90,
        num_trials=10,
    )
    assert experiment.cadence == CadenceType.DAILY
    assert experiment.evaluation_window_days == 90
    assert experiment.num_trials == 10


def test_cadence_experiment_run_returns_result():
    """Running an experiment produces structured results."""
    experiment = CadenceExperiment(
        cadence=CadenceType.WEEKLY,
        evaluation_window_days=90,
        num_trials=5,
    )
    result = experiment.run_mock()
    assert isinstance(result, CadenceResult)
    assert result.cadence == CadenceType.WEEKLY
    assert result.avg_sharpe is not None
    assert result.avg_evolution_count >= 0
    assert result.avg_strategy_churn >= 0.0


def test_event_trigger_fires_on_regime_change():
    """Event trigger detects regime changes as evolution triggers."""
    trigger = EventTrigger(trigger_type="regime_change")
    assert trigger.should_fire(
        prev_regime="risk-on",
        current_regime="risk-off",
    )
    assert not trigger.should_fire(
        prev_regime="risk-on",
        current_regime="risk-on",
    )


def test_event_trigger_fires_on_drawdown():
    """Event trigger detects drawdown threshold breach."""
    trigger = EventTrigger(trigger_type="drawdown", threshold=0.10)
    assert trigger.should_fire(current_drawdown=0.12)
    assert not trigger.should_fire(current_drawdown=0.05)


def test_cadence_comparison():
    """Compare multiple cadences and rank by Sharpe."""
    results = []
    for cadence in [CadenceType.DAILY, CadenceType.WEEKLY, CadenceType.EVENT_TRIGGERED]:
        experiment = CadenceExperiment(
            cadence=cadence,
            evaluation_window_days=90,
            num_trials=3,
        )
        results.append(experiment.run_mock())

    # All cadences produce results
    assert len(results) == 3

    # Results are comparable
    ranked = sorted(results, key=lambda r: r.avg_sharpe, reverse=True)
    assert ranked[0].avg_sharpe >= ranked[-1].avg_sharpe
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/research/test_evolution_cadence.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.research.evolution_cadence'`

**Step 3: Implement the evolution cadence experiment framework**

```python
# src/evolve_trader/research/evolution_cadence.py
"""Research: Optimal evolution cadence.

Experiment framework to empirically test daily vs weekly vs event-triggered
strategy evolution. Measures impact on Sharpe ratio, strategy churn,
and adaptation speed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CadenceType(str, Enum):
    """Evolution cadence options."""
    DAILY = "daily"
    WEEKLY = "weekly"
    EVENT_TRIGGERED = "event_triggered"
    HYBRID = "hybrid"  # Weekly + event-triggered


@dataclass
class CadenceResult:
    """Result of a cadence experiment."""
    cadence: CadenceType
    avg_sharpe: float
    avg_evolution_count: int
    avg_strategy_churn: float  # Fraction of strategies replaced per period
    avg_adaptation_lag_days: float  # Days to adapt after regime change
    trial_results: list[dict] = field(default_factory=list)


@dataclass
class EventTrigger:
    """Event-based evolution trigger."""
    trigger_type: str  # regime_change, drawdown, signal_spike
    threshold: Optional[float] = None

    def should_fire(self, **kwargs) -> bool:
        """Check if trigger condition is met."""
        if self.trigger_type == "regime_change":
            return kwargs.get("prev_regime") != kwargs.get("current_regime")
        elif self.trigger_type == "drawdown":
            return kwargs.get("current_drawdown", 0) >= (self.threshold or 0.10)
        elif self.trigger_type == "signal_spike":
            return kwargs.get("signal_magnitude", 0) >= (self.threshold or 2.0)
        return False


class CadenceExperiment:
    """Experiment framework for testing evolution cadence.

    Runs simulated evolution cycles at different cadences and measures:
    - Sharpe ratio of the strategy portfolio
    - Number of evolution events triggered
    - Strategy churn (how often strategies are replaced)
    - Adaptation lag (how quickly portfolio adapts to regime changes)
    """

    def __init__(
        self,
        cadence: CadenceType,
        evaluation_window_days: int = 90,
        num_trials: int = 10,
        seed: Optional[int] = None,
    ) -> None:
        self.cadence = cadence
        self.evaluation_window_days = evaluation_window_days
        self.num_trials = num_trials
        self._rng = random.Random(seed)

    def run_mock(self) -> CadenceResult:
        """Run a mock experiment producing synthetic results.

        In production, this calls the actual evolution loop with the
        configured cadence. The mock version uses calibrated distributions
        to produce plausible results for framework validation.
        """
        trial_results = []
        for _ in range(self.num_trials):
            trial = self._run_single_mock_trial()
            trial_results.append(trial)

        avg_sharpe = sum(t["sharpe"] for t in trial_results) / self.num_trials
        avg_evo_count = int(
            sum(t["evolution_count"] for t in trial_results) / self.num_trials
        )
        avg_churn = sum(t["churn"] for t in trial_results) / self.num_trials
        avg_lag = sum(t["adaptation_lag"] for t in trial_results) / self.num_trials

        return CadenceResult(
            cadence=self.cadence,
            avg_sharpe=avg_sharpe,
            avg_evolution_count=avg_evo_count,
            avg_strategy_churn=avg_churn,
            avg_adaptation_lag_days=avg_lag,
            trial_results=trial_results,
        )

    def _run_single_mock_trial(self) -> dict:
        """Produce a single mock trial result with cadence-dependent bias."""
        # Calibrated distributions per cadence type
        if self.cadence == CadenceType.DAILY:
            sharpe = self._rng.gauss(1.0, 0.3)
            evo_count = self._rng.randint(60, 90)
            churn = self._rng.uniform(0.15, 0.30)
            lag = self._rng.uniform(1.0, 2.0)
        elif self.cadence == CadenceType.WEEKLY:
            sharpe = self._rng.gauss(1.1, 0.25)
            evo_count = self._rng.randint(10, 15)
            churn = self._rng.uniform(0.05, 0.12)
            lag = self._rng.uniform(3.0, 7.0)
        elif self.cadence == CadenceType.EVENT_TRIGGERED:
            sharpe = self._rng.gauss(1.2, 0.35)
            evo_count = self._rng.randint(15, 40)
            churn = self._rng.uniform(0.08, 0.18)
            lag = self._rng.uniform(0.5, 3.0)
        else:  # HYBRID
            sharpe = self._rng.gauss(1.15, 0.28)
            evo_count = self._rng.randint(20, 45)
            churn = self._rng.uniform(0.08, 0.15)
            lag = self._rng.uniform(1.0, 4.0)

        return {
            "sharpe": sharpe,
            "evolution_count": evo_count,
            "churn": churn,
            "adaptation_lag": lag,
        }
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/research/test_evolution_cadence.py -v
```

Expected: PASS — all cadence experiment tests pass

**Step 5: Commit**

```bash
git add src/evolve_trader/research/evolution_cadence.py tests/research/test_evolution_cadence.py
git commit -m "feat: research framework for evolution cadence experiments (daily/weekly/event)"
```

---

## Task 6: Research — Carrying Capacity

**Files:**
- Create: `src/evolve_trader/research/carrying_capacity.py`
- Create: `tests/research/test_carrying_capacity.py`

**Step 1: Write the failing tests**

```python
# tests/research/test_carrying_capacity.py
"""Tests for carrying capacity research experiment."""
import pytest
from evolve_trader.research.carrying_capacity import (
    CarryingCapacityExperiment,
    CapacityResult,
    PortfolioDiversificationMetrics,
)


def test_carrying_capacity_experiment_setup():
    """Experiment configures range of active strategy counts."""
    experiment = CarryingCapacityExperiment(
        min_strategies=3,
        max_strategies=30,
        step=3,
        evaluation_window_days=180,
    )
    assert experiment.min_strategies == 3
    assert experiment.max_strategies == 30
    assert experiment.strategy_counts == [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]


def test_capacity_result_has_required_fields():
    """Result captures Sharpe, drawdown, and diversification per count."""
    result = CapacityResult(
        strategy_count=10,
        avg_sharpe=1.3,
        avg_max_drawdown=0.12,
        avg_correlation=0.25,
        diversification_benefit=0.85,
    )
    assert result.strategy_count == 10
    assert result.avg_sharpe == 1.3
    assert result.diversification_benefit == 0.85


def test_experiment_run_produces_results():
    """Running experiment produces one result per strategy count."""
    experiment = CarryingCapacityExperiment(
        min_strategies=5,
        max_strategies=15,
        step=5,
    )
    results = experiment.run_mock()
    assert len(results) == 3  # 5, 10, 15
    for result in results:
        assert result.strategy_count in [5, 10, 15]
        assert result.avg_sharpe is not None


def test_find_optimal_count():
    """Experiment identifies the optimal strategy count."""
    experiment = CarryingCapacityExperiment(
        min_strategies=5,
        max_strategies=25,
        step=5,
    )
    results = experiment.run_mock()
    optimal = experiment.find_optimal(results)
    assert optimal.strategy_count >= 5
    assert optimal.strategy_count <= 25
    assert optimal.avg_sharpe == max(r.avg_sharpe for r in results)


def test_diversification_metrics():
    """Diversification metrics detect when adding strategies stops helping."""
    metrics = PortfolioDiversificationMetrics()
    # Marginal benefit decreases with more strategies
    benefit_5 = metrics.marginal_diversification_benefit(n_strategies=5)
    benefit_20 = metrics.marginal_diversification_benefit(n_strategies=20)
    assert benefit_5 > benefit_20  # Diminishing returns
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/research/test_carrying_capacity.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.research.carrying_capacity'`

**Step 3: Implement the carrying capacity experiment**

```python
# src/evolve_trader/research/carrying_capacity.py
"""Research: Optimal active strategy count (carrying capacity).

Determines the ideal number of concurrently active strategy skills.
Too few = underdiversified. Too many = diluted alpha + management overhead.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CapacityResult:
    """Result for a single strategy count experiment."""
    strategy_count: int
    avg_sharpe: float
    avg_max_drawdown: float
    avg_correlation: float
    diversification_benefit: float


class PortfolioDiversificationMetrics:
    """Measures diversification benefit of adding strategies."""

    def marginal_diversification_benefit(self, n_strategies: int) -> float:
        """Approximate marginal diversification benefit.

        Models the diminishing returns of diversification:
        benefit ≈ 1 / sqrt(n) — adding the nth strategy contributes less.
        """
        if n_strategies <= 0:
            return 0.0
        return 1.0 / math.sqrt(n_strategies)


class CarryingCapacityExperiment:
    """Experiment to find optimal number of active strategies.

    Tests portfolios with different numbers of active strategies
    across the same evaluation window and measures:
    - Sharpe ratio (risk-adjusted return)
    - Maximum drawdown
    - Average pairwise correlation
    - Diversification benefit vs. single best strategy
    """

    def __init__(
        self,
        min_strategies: int = 3,
        max_strategies: int = 30,
        step: int = 3,
        evaluation_window_days: int = 180,
        seed: Optional[int] = None,
    ) -> None:
        self.min_strategies = min_strategies
        self.max_strategies = max_strategies
        self.step = step
        self.evaluation_window_days = evaluation_window_days
        self._rng = random.Random(seed)

    @property
    def strategy_counts(self) -> list[int]:
        """List of strategy counts to test."""
        return list(range(self.min_strategies, self.max_strategies + 1, self.step))

    def run_mock(self) -> list[CapacityResult]:
        """Run mock experiment for each strategy count.

        Mock calibration:
        - Sharpe peaks around 10-15 strategies, then declines slightly
        - Drawdown improves with more strategies (diversification)
        - Correlation increases with more strategies (alpha overlap)
        """
        results = []
        for count in self.strategy_counts:
            # Sharpe peaks around 12 strategies
            sharpe_base = 1.5 - 0.005 * (count - 12) ** 2
            sharpe = max(0.3, sharpe_base + self._rng.gauss(0, 0.1))

            # Drawdown improves with diversification, then plateaus
            drawdown = 0.20 / math.sqrt(count) + self._rng.gauss(0, 0.01)
            drawdown = max(0.03, min(0.30, drawdown))

            # Correlation increases with more strategies
            correlation = 0.10 + 0.015 * count + self._rng.gauss(0, 0.02)
            correlation = max(0.0, min(1.0, correlation))

            # Diversification benefit
            div_benefit = 1.0 / math.sqrt(count)

            results.append(CapacityResult(
                strategy_count=count,
                avg_sharpe=sharpe,
                avg_max_drawdown=drawdown,
                avg_correlation=correlation,
                diversification_benefit=div_benefit,
            ))
        return results

    def find_optimal(self, results: list[CapacityResult]) -> CapacityResult:
        """Find the strategy count with the highest Sharpe ratio."""
        return max(results, key=lambda r: r.avg_sharpe)
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/research/test_carrying_capacity.py -v
```

Expected: PASS — all carrying capacity tests pass

**Step 5: Commit**

```bash
git add src/evolve_trader/research/carrying_capacity.py tests/research/test_carrying_capacity.py
git commit -m "feat: research framework for optimal active strategy count (carrying capacity)"
```

---

## Task 7: Research — Cross-Market Transfer

**Files:**
- Create: `src/evolve_trader/research/cross_market.py`
- Create: `tests/research/test_cross_market.py`

**Step 1: Write the failing tests**

```python
# tests/research/test_cross_market.py
"""Tests for cross-market skill transfer research."""
import pytest
from evolve_trader.research.cross_market import (
    CrossMarketExperiment,
    TransferResult,
    MarketDomain,
)


def test_market_domain_enum():
    """All market domains are defined."""
    assert MarketDomain.NASDAQ in MarketDomain
    assert MarketDomain.SP500 in MarketDomain
    assert MarketDomain.CRYPTO_BITWISE10 in MarketDomain
    assert MarketDomain.FOREX_MAJORS in MarketDomain
    assert MarketDomain.COMMODITIES in MarketDomain


def test_cross_market_experiment_setup():
    """Experiment configures source and target domains."""
    experiment = CrossMarketExperiment(
        source_domain=MarketDomain.NASDAQ,
        target_domain=MarketDomain.CRYPTO_BITWISE10,
        strategy_type="momentum",
        evaluation_window_days=180,
    )
    assert experiment.source_domain == MarketDomain.NASDAQ
    assert experiment.target_domain == MarketDomain.CRYPTO_BITWISE10


def test_transfer_result_has_required_fields():
    """Transfer result captures source/target performance and delta."""
    result = TransferResult(
        source_domain=MarketDomain.NASDAQ,
        target_domain=MarketDomain.CRYPTO_BITWISE10,
        strategy_type="momentum",
        source_sharpe=1.5,
        target_sharpe=0.8,
        transfer_efficiency=0.53,
        significant_decay_factors=["volatility_mismatch", "24_7_trading"],
    )
    assert result.transfer_efficiency == 0.53
    assert result.performance_delta == -0.7
    assert len(result.significant_decay_factors) == 2


def test_experiment_run_mock():
    """Running experiment produces transfer result."""
    experiment = CrossMarketExperiment(
        source_domain=MarketDomain.NASDAQ,
        target_domain=MarketDomain.CRYPTO_BITWISE10,
        strategy_type="momentum",
    )
    result = experiment.run_mock()
    assert isinstance(result, TransferResult)
    assert result.source_sharpe > 0
    assert result.transfer_efficiency >= 0.0
    assert result.transfer_efficiency <= 1.0


def test_experiment_matrix():
    """Run all source → target combinations."""
    experiment = CrossMarketExperiment(
        source_domain=MarketDomain.NASDAQ,
        target_domain=MarketDomain.CRYPTO_BITWISE10,
        strategy_type="momentum",
    )
    matrix = experiment.run_transfer_matrix(
        domains=[MarketDomain.NASDAQ, MarketDomain.CRYPTO_BITWISE10, MarketDomain.FOREX_MAJORS],
        strategy_types=["momentum", "mean_reversion"],
    )
    # 3 domains × 2 strategies × (3-1) targets = 12 results
    assert len(matrix) == 12
    for result in matrix:
        assert result.source_domain != result.target_domain
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/research/test_cross_market.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.research.cross_market'`

**Step 3: Implement the cross-market transfer experiment**

```python
# src/evolve_trader/research/cross_market.py
"""Research: Cross-market skill transfer.

Tests whether strategies evolved in one market domain (e.g., NASDAQ momentum)
transfer effectively to another domain (e.g., crypto). Measures transfer
efficiency and identifies factors that cause performance decay.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MarketDomain(str, Enum):
    """Market domains for cross-market transfer testing."""
    NASDAQ = "nasdaq"
    SP500 = "sp500"
    CRYPTO_BITWISE10 = "crypto_bitwise10"
    FOREX_MAJORS = "forex_majors"
    COMMODITIES = "commodities"


# Transfer efficiency priors (source, target) → base efficiency
_TRANSFER_PRIORS: dict[tuple[str, str], float] = {
    ("nasdaq", "sp500"): 0.85,
    ("nasdaq", "crypto_bitwise10"): 0.40,
    ("nasdaq", "forex_majors"): 0.30,
    ("nasdaq", "commodities"): 0.25,
    ("sp500", "nasdaq"): 0.85,
    ("sp500", "crypto_bitwise10"): 0.35,
    ("sp500", "forex_majors"): 0.30,
    ("sp500", "commodities"): 0.25,
    ("crypto_bitwise10", "nasdaq"): 0.35,
    ("crypto_bitwise10", "sp500"): 0.30,
    ("crypto_bitwise10", "forex_majors"): 0.20,
    ("crypto_bitwise10", "commodities"): 0.15,
    ("forex_majors", "nasdaq"): 0.30,
    ("forex_majors", "sp500"): 0.30,
    ("forex_majors", "crypto_bitwise10"): 0.25,
    ("forex_majors", "commodities"): 0.40,
    ("commodities", "nasdaq"): 0.25,
    ("commodities", "sp500"): 0.25,
    ("commodities", "crypto_bitwise10"): 0.15,
    ("commodities", "forex_majors"): 0.40,
}

_DECAY_FACTORS: dict[str, list[str]] = {
    "crypto_bitwise10": ["volatility_mismatch", "24_7_trading", "on_chain_dynamics"],
    "forex_majors": ["macro_sensitivity", "carry_trade_dynamics", "central_bank_policy"],
    "commodities": ["supply_demand_fundamentals", "seasonality", "contango_backwardation"],
    "nasdaq": ["sector_concentration", "earnings_sensitivity"],
    "sp500": ["broad_market_beta"],
}


@dataclass
class TransferResult:
    """Result of a cross-market transfer experiment."""
    source_domain: MarketDomain
    target_domain: MarketDomain
    strategy_type: str
    source_sharpe: float
    target_sharpe: float
    transfer_efficiency: float  # target_sharpe / source_sharpe (clamped 0-1)
    significant_decay_factors: list[str] = field(default_factory=list)

    @property
    def performance_delta(self) -> float:
        """Absolute Sharpe delta (target - source)."""
        return self.target_sharpe - self.source_sharpe


class CrossMarketExperiment:
    """Experiment framework for cross-market skill transfer.

    Tests a strategy type evolved in the source domain and measures
    its performance when applied to the target domain. Transfer
    efficiency = target_sharpe / source_sharpe.
    """

    def __init__(
        self,
        source_domain: MarketDomain,
        target_domain: MarketDomain,
        strategy_type: str,
        evaluation_window_days: int = 180,
        seed: Optional[int] = None,
    ) -> None:
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.strategy_type = strategy_type
        self.evaluation_window_days = evaluation_window_days
        self._rng = random.Random(seed)

    def run_mock(self) -> TransferResult:
        """Run a mock transfer experiment.

        Uses calibrated priors for transfer efficiency between domain pairs.
        """
        source_sharpe = self._rng.gauss(1.3, 0.2)
        source_sharpe = max(0.3, source_sharpe)

        base_efficiency = _TRANSFER_PRIORS.get(
            (self.source_domain.value, self.target_domain.value), 0.30
        )
        efficiency = base_efficiency + self._rng.gauss(0, 0.05)
        efficiency = max(0.0, min(1.0, efficiency))

        target_sharpe = source_sharpe * efficiency

        decay_factors = _DECAY_FACTORS.get(self.target_domain.value, [])

        return TransferResult(
            source_domain=self.source_domain,
            target_domain=self.target_domain,
            strategy_type=self.strategy_type,
            source_sharpe=source_sharpe,
            target_sharpe=target_sharpe,
            transfer_efficiency=efficiency,
            significant_decay_factors=decay_factors,
        )

    def run_transfer_matrix(
        self,
        domains: list[MarketDomain],
        strategy_types: list[str],
    ) -> list[TransferResult]:
        """Run all source → target × strategy combinations.

        Skips source == target pairs.
        """
        results = []
        for strategy in strategy_types:
            for source in domains:
                for target in domains:
                    if source == target:
                        continue
                    exp = CrossMarketExperiment(
                        source_domain=source,
                        target_domain=target,
                        strategy_type=strategy,
                        evaluation_window_days=self.evaluation_window_days,
                        seed=self._rng.randint(0, 2**32),
                    )
                    results.append(exp.run_mock())
        return results
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/research/test_cross_market.py -v
```

Expected: PASS — all cross-market transfer tests pass

**Step 5: Commit**

```bash
git add src/evolve_trader/research/cross_market.py tests/research/test_cross_market.py
git commit -m "feat: research framework for cross-market skill transfer experiments"
```

---

## Task 8: Research — LLM Model Sensitivity

**Files:**
- Create: `src/evolve_trader/research/model_sensitivity.py`
- Create: `tests/research/test_model_sensitivity.py`

**Step 1: Write the failing tests**

```python
# tests/research/test_model_sensitivity.py
"""Tests for LLM model sensitivity research."""
import pytest
from evolve_trader.research.model_sensitivity import (
    ModelSensitivityExperiment,
    ModelConfig,
    SensitivityResult,
    PortabilityScore,
)


def test_model_config():
    """Model config captures provider, model name, and parameters."""
    config = ModelConfig(
        provider="anthropic",
        model_name="claude-sonnet-4-20250514",
        temperature=0.7,
        max_tokens=4096,
    )
    assert config.provider == "anthropic"
    assert config.model_name == "claude-sonnet-4-20250514"


def test_experiment_setup():
    """Experiment configures models to compare and skills to test."""
    models = [
        ModelConfig("anthropic", "claude-sonnet-4-20250514"),
        ModelConfig("openai", "gpt-4o"),
        ModelConfig("alibaba", "qwen-2.5-72b"),
    ]
    experiment = ModelSensitivityExperiment(
        models=models,
        skill_names=["momentum-v3", "mean-reversion-v2"],
        num_trials=5,
    )
    assert len(experiment.models) == 3
    assert len(experiment.skill_names) == 2


def test_sensitivity_result_has_required_fields():
    """Result captures per-model performance and cross-model variance."""
    result = SensitivityResult(
        skill_name="momentum-v3",
        model_results={
            "claude-sonnet-4-20250514": {"sharpe": 1.4, "win_rate": 0.58},
            "gpt-4o": {"sharpe": 1.2, "win_rate": 0.55},
            "qwen-2.5-72b": {"sharpe": 0.9, "win_rate": 0.51},
        },
        cross_model_sharpe_std=0.25,
        trade_decision_agreement=0.72,
    )
    assert result.skill_name == "momentum-v3"
    assert result.cross_model_sharpe_std == 0.25
    assert result.trade_decision_agreement == 0.72


def test_portability_score_high_for_similar_results():
    """Portability is high when models produce similar results."""
    score = PortabilityScore.from_results(
        sharpe_values=[1.4, 1.3, 1.35],
        decision_agreement=0.90,
    )
    assert score.value >= 0.8
    assert score.is_portable


def test_portability_score_low_for_divergent_results():
    """Portability is low when models produce very different results."""
    score = PortabilityScore.from_results(
        sharpe_values=[1.5, 0.3, -0.2],
        decision_agreement=0.40,
    )
    assert score.value < 0.5
    assert not score.is_portable


def test_experiment_run_mock():
    """Running experiment produces sensitivity results per skill."""
    models = [
        ModelConfig("anthropic", "claude-sonnet-4-20250514"),
        ModelConfig("openai", "gpt-4o"),
    ]
    experiment = ModelSensitivityExperiment(
        models=models,
        skill_names=["momentum-v3"],
        num_trials=3,
    )
    results = experiment.run_mock()
    assert len(results) == 1  # One per skill
    assert results[0].skill_name == "momentum-v3"
    assert len(results[0].model_results) == 2
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/research/test_model_sensitivity.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.research.model_sensitivity'`

**Step 3: Implement the model sensitivity experiment**

```python
# src/evolve_trader/research/model_sensitivity.py
"""Research: LLM model sensitivity.

Tests whether evolved strategy skills are model-specific or portable
across LLM providers. Compares trade decisions and performance when
the same skill is executed by different models (Claude, GPT-4o, Qwen).
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for an LLM model."""
    provider: str  # anthropic, openai, alibaba, etc.
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class SensitivityResult:
    """Result of model sensitivity experiment for a single skill."""
    skill_name: str
    model_results: dict[str, dict[str, float]]  # model_name → metrics
    cross_model_sharpe_std: float
    trade_decision_agreement: float  # Fraction of identical trade decisions


@dataclass
class PortabilityScore:
    """Portability score for a strategy skill across models."""
    value: float  # 0-1, higher = more portable
    sharpe_std: float
    decision_agreement: float

    @property
    def is_portable(self) -> bool:
        """Skill is portable if score >= 0.7."""
        return self.value >= 0.7

    @classmethod
    def from_results(
        cls,
        sharpe_values: list[float],
        decision_agreement: float,
    ) -> PortabilityScore:
        """Calculate portability from Sharpe values and agreement rate."""
        if len(sharpe_values) < 2:
            return cls(value=1.0, sharpe_std=0.0, decision_agreement=decision_agreement)

        sharpe_std = statistics.stdev(sharpe_values)
        # Normalize: low std → high portability
        # std of 0 → score 1.0, std of 1.0+ → score ~0.2
        std_component = max(0.0, 1.0 - sharpe_std)
        # Weighted: 40% consistency, 60% decision agreement
        value = 0.4 * std_component + 0.6 * decision_agreement
        return cls(
            value=value,
            sharpe_std=sharpe_std,
            decision_agreement=decision_agreement,
        )


class ModelSensitivityExperiment:
    """Experiment framework for testing LLM model sensitivity.

    For each skill × model combination, runs the skill with the given
    model and records trade decisions and performance metrics. Compares
    across models to determine if skills are portable.
    """

    def __init__(
        self,
        models: list[ModelConfig],
        skill_names: list[str],
        num_trials: int = 10,
        seed: Optional[int] = None,
    ) -> None:
        self.models = models
        self.skill_names = skill_names
        self.num_trials = num_trials
        self._rng = random.Random(seed)

    def run_mock(self) -> list[SensitivityResult]:
        """Run mock sensitivity experiment for each skill."""
        results = []
        for skill_name in self.skill_names:
            model_results = {}
            sharpe_values = []

            for model in self.models:
                # Each model gets a slightly different base performance
                base_sharpe = self._rng.gauss(1.2, 0.3)
                win_rate = 0.50 + self._rng.gauss(0.05, 0.03)
                win_rate = max(0.35, min(0.70, win_rate))

                model_results[model.model_name] = {
                    "sharpe": base_sharpe,
                    "win_rate": win_rate,
                    "total_trades": self._rng.randint(30, 60),
                }
                sharpe_values.append(base_sharpe)

            sharpe_std = statistics.stdev(sharpe_values) if len(sharpe_values) > 1 else 0.0
            agreement = self._rng.uniform(0.55, 0.85)

            results.append(SensitivityResult(
                skill_name=skill_name,
                model_results=model_results,
                cross_model_sharpe_std=sharpe_std,
                trade_decision_agreement=agreement,
            ))

        return results
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/research/test_model_sensitivity.py -v
```

Expected: PASS — all model sensitivity tests pass

**Step 5: Commit**

```bash
git add src/evolve_trader/research/model_sensitivity.py tests/research/test_model_sensitivity.py
git commit -m "feat: research framework for LLM model sensitivity and skill portability"
```

---

## Task 9: Research — Adversarial Robustness

**Files:**
- Create: `src/evolve_trader/research/adversarial.py`
- Create: `tests/research/test_adversarial.py`

**Step 1: Write the failing tests**

```python
# tests/research/test_adversarial.py
"""Tests for adversarial robustness research (TradeTrap framework)."""
import pytest
from evolve_trader.research.adversarial import (
    TradeTrap,
    AdversarialScenario,
    AdversarialResult,
    AttackType,
)


def test_attack_type_enum():
    """All attack types are defined."""
    assert AttackType.SIGNAL_POISONING in AttackType
    assert AttackType.WASH_TRADING in AttackType
    assert AttackType.FRONT_RUNNING in AttackType
    assert AttackType.SPOOFING in AttackType
    assert AttackType.COORDINATED_MANIPULATION in AttackType


def test_adversarial_scenario_setup():
    """Scenario configures attack type, intensity, and duration."""
    scenario = AdversarialScenario(
        attack_type=AttackType.SIGNAL_POISONING,
        intensity=0.5,  # 50% of signals poisoned
        duration_days=30,
        target_signal_source="edgar_13f",
    )
    assert scenario.attack_type == AttackType.SIGNAL_POISONING
    assert scenario.intensity == 0.5
    assert scenario.target_signal_source == "edgar_13f"


def test_trade_trap_run_scenario():
    """TradeTrap runs an adversarial scenario and returns result."""
    trap = TradeTrap()
    scenario = AdversarialScenario(
        attack_type=AttackType.SIGNAL_POISONING,
        intensity=0.3,
        duration_days=30,
        target_signal_source="congressional",
    )
    result = trap.run_mock(scenario)
    assert isinstance(result, AdversarialResult)
    assert result.attack_type == AttackType.SIGNAL_POISONING
    assert result.baseline_sharpe > 0
    assert result.attacked_sharpe is not None
    assert result.degradation_pct is not None


def test_trade_trap_signal_poisoning_degrades_performance():
    """Signal poisoning should degrade strategy performance."""
    trap = TradeTrap(seed=42)
    scenario = AdversarialScenario(
        attack_type=AttackType.SIGNAL_POISONING,
        intensity=0.8,  # 80% of signals poisoned
        duration_days=60,
        target_signal_source="edgar_13f",
    )
    result = trap.run_mock(scenario)
    assert result.attacked_sharpe < result.baseline_sharpe
    assert result.degradation_pct > 0


def test_trade_trap_low_intensity_minimal_impact():
    """Low-intensity attacks should have minimal performance impact."""
    trap = TradeTrap(seed=42)
    scenario = AdversarialScenario(
        attack_type=AttackType.WASH_TRADING,
        intensity=0.05,
        duration_days=10,
    )
    result = trap.run_mock(scenario)
    # Degradation should be small
    assert result.degradation_pct < 20.0


def test_trade_trap_detection_rate():
    """TradeTrap measures how often the system detects the attack."""
    trap = TradeTrap(seed=42)
    scenario = AdversarialScenario(
        attack_type=AttackType.SPOOFING,
        intensity=0.5,
        duration_days=30,
    )
    result = trap.run_mock(scenario)
    assert 0.0 <= result.detection_rate <= 1.0


def test_trade_trap_run_all_attacks():
    """Run all attack types and produce a vulnerability report."""
    trap = TradeTrap(seed=42)
    results = trap.run_all_attacks(intensity=0.5, duration_days=30)
    assert len(results) == len(AttackType)
    for result in results:
        assert result.baseline_sharpe > 0
        assert result.attacked_sharpe is not None
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/research/test_adversarial.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.research.adversarial'`

**Step 3: Implement the TradeTrap adversarial framework**

```python
# src/evolve_trader/research/adversarial.py
"""Research: Adversarial robustness (TradeTrap framework).

Tests whether the system's signal sources can be gamed. Simulates
various attack types (signal poisoning, wash trading, front-running,
spoofing, coordinated manipulation) and measures performance degradation
and detection rate.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AttackType(str, Enum):
    """Types of adversarial attacks on trading signals."""
    SIGNAL_POISONING = "signal_poisoning"
    WASH_TRADING = "wash_trading"
    FRONT_RUNNING = "front_running"
    SPOOFING = "spoofing"
    COORDINATED_MANIPULATION = "coordinated_manipulation"


# How much each attack type degrades performance per unit intensity
_ATTACK_IMPACT: dict[AttackType, float] = {
    AttackType.SIGNAL_POISONING: 0.60,       # High impact — corrupts signal quality
    AttackType.WASH_TRADING: 0.25,           # Moderate — inflates volume signals
    AttackType.FRONT_RUNNING: 0.40,          # High — steals alpha
    AttackType.SPOOFING: 0.30,               # Moderate — distorts order book
    AttackType.COORDINATED_MANIPULATION: 0.50,  # High — multi-vector attack
}

# Detection difficulty per attack type (higher = harder to detect)
_DETECTION_DIFFICULTY: dict[AttackType, float] = {
    AttackType.SIGNAL_POISONING: 0.40,
    AttackType.WASH_TRADING: 0.30,
    AttackType.FRONT_RUNNING: 0.60,
    AttackType.SPOOFING: 0.25,
    AttackType.COORDINATED_MANIPULATION: 0.70,
}


@dataclass
class AdversarialScenario:
    """Configuration for an adversarial attack scenario."""
    attack_type: AttackType
    intensity: float  # 0-1, fraction of signals/trades affected
    duration_days: int
    target_signal_source: Optional[str] = None


@dataclass
class AdversarialResult:
    """Result of an adversarial attack simulation."""
    attack_type: AttackType
    intensity: float
    duration_days: int
    baseline_sharpe: float
    attacked_sharpe: float
    degradation_pct: float  # Percentage degradation in Sharpe
    detection_rate: float   # Fraction of attacks detected (0-1)
    false_positive_rate: float = 0.0
    recovery_time_days: Optional[float] = None


class TradeTrap:
    """Adversarial robustness testing framework.

    Simulates attacks on the trading system's signal pipeline and
    measures how well the system detects and resists manipulation.
    Named after the concept of a "trade trap" — adversarial conditions
    designed to exploit algorithmic trading systems.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def run_mock(self, scenario: AdversarialScenario) -> AdversarialResult:
        """Run a mock adversarial scenario.

        Uses calibrated impact and detection models per attack type.
        """
        baseline_sharpe = self._rng.gauss(1.3, 0.15)
        baseline_sharpe = max(0.5, baseline_sharpe)

        # Calculate degradation based on attack type and intensity
        impact_factor = _ATTACK_IMPACT.get(scenario.attack_type, 0.30)
        degradation = impact_factor * scenario.intensity
        # Duration amplifies degradation (diminishing returns)
        duration_factor = min(1.5, 1.0 + scenario.duration_days / 60.0 * 0.5)
        degradation *= duration_factor
        degradation += self._rng.gauss(0, 0.03)
        degradation = max(0.0, min(0.95, degradation))

        attacked_sharpe = baseline_sharpe * (1.0 - degradation)
        degradation_pct = degradation * 100.0

        # Detection rate inversely related to difficulty
        difficulty = _DETECTION_DIFFICULTY.get(scenario.attack_type, 0.50)
        detection_base = 1.0 - difficulty
        # Higher intensity is easier to detect
        intensity_bonus = scenario.intensity * 0.3
        detection_rate = min(1.0, detection_base + intensity_bonus + self._rng.gauss(0, 0.05))
        detection_rate = max(0.0, detection_rate)

        # False positive rate
        false_positive_rate = self._rng.uniform(0.01, 0.08)

        # Recovery time
        recovery_days = scenario.duration_days * 0.5 + self._rng.gauss(5, 2)
        recovery_days = max(1.0, recovery_days)

        return AdversarialResult(
            attack_type=scenario.attack_type,
            intensity=scenario.intensity,
            duration_days=scenario.duration_days,
            baseline_sharpe=baseline_sharpe,
            attacked_sharpe=attacked_sharpe,
            degradation_pct=degradation_pct,
            detection_rate=detection_rate,
            false_positive_rate=false_positive_rate,
            recovery_time_days=recovery_days,
        )

    def run_all_attacks(
        self,
        intensity: float = 0.5,
        duration_days: int = 30,
        target_signal_source: Optional[str] = None,
    ) -> list[AdversarialResult]:
        """Run all attack types and produce a vulnerability report."""
        results = []
        for attack_type in AttackType:
            scenario = AdversarialScenario(
                attack_type=attack_type,
                intensity=intensity,
                duration_days=duration_days,
                target_signal_source=target_signal_source,
            )
            results.append(self.run_mock(scenario))
        return results
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/research/test_adversarial.py -v
```

Expected: PASS — all adversarial robustness tests pass

**Step 5: Commit**

```bash
git add src/evolve_trader/research/adversarial.py tests/research/test_adversarial.py
git commit -m "feat: TradeTrap adversarial robustness research framework"
```

---

## Task 10: Dashboard v2

**Files:**
- Modify: `dashboard/src/components/performance/PerformanceChart.tsx`
- Create: `dashboard/src/components/mobile/MobileLayout.tsx`
- Create: `dashboard/src/components/alerts/CustomAlertConfig.tsx`
- Modify: `dashboard/src/components/costs/LLMCostPanel.tsx`
- Create: `dashboard/src/hooks/useVirtualScroll.ts`

**Step 1: Write the failing tests**

```typescript
// dashboard/src/__tests__/MobileLayout.test.tsx
import { render, screen } from "@testing-library/react";
import { MobileLayout } from "../components/mobile/MobileLayout";

describe("MobileLayout", () => {
  it("renders mobile-optimized navigation", () => {
    render(<MobileLayout />);
    expect(screen.getByTestId("mobile-nav")).toBeInTheDocument();
  });

  it("collapses sidebar on mobile viewport", () => {
    // Mock window.innerWidth = 375
    Object.defineProperty(window, "innerWidth", { value: 375, writable: true });
    window.dispatchEvent(new Event("resize"));
    render(<MobileLayout />);
    expect(screen.queryByTestId("desktop-sidebar")).not.toBeInTheDocument();
    expect(screen.getByTestId("mobile-hamburger")).toBeInTheDocument();
  });

  it("supports swipe gestures for tab navigation", () => {
    render(<MobileLayout />);
    expect(screen.getByTestId("swipeable-tabs")).toBeInTheDocument();
  });
});

// dashboard/src/__tests__/CustomAlertConfig.test.tsx
import { render, screen, fireEvent } from "@testing-library/react";
import { CustomAlertConfig } from "../components/alerts/CustomAlertConfig";

describe("CustomAlertConfig", () => {
  it("renders alert configuration form", () => {
    render(<CustomAlertConfig onSave={jest.fn()} />);
    expect(screen.getByLabelText("Alert Name")).toBeInTheDocument();
    expect(screen.getByLabelText("Metric")).toBeInTheDocument();
    expect(screen.getByLabelText("Threshold")).toBeInTheDocument();
    expect(screen.getByLabelText("Channel")).toBeInTheDocument();
  });

  it("calls onSave with alert configuration", () => {
    const onSave = jest.fn();
    render(<CustomAlertConfig onSave={onSave} />);
    fireEvent.change(screen.getByLabelText("Alert Name"), {
      target: { value: "High Drawdown" },
    });
    fireEvent.change(screen.getByLabelText("Metric"), {
      target: { value: "portfolio_drawdown" },
    });
    fireEvent.change(screen.getByLabelText("Threshold"), {
      target: { value: "0.15" },
    });
    fireEvent.click(screen.getByText("Save Alert"));
    expect(onSave).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "High Drawdown",
        metric: "portfolio_drawdown",
        threshold: 0.15,
      })
    );
  });

  it("validates required fields", () => {
    render(<CustomAlertConfig onSave={jest.fn()} />);
    fireEvent.click(screen.getByText("Save Alert"));
    expect(screen.getByText("Alert name is required")).toBeInTheDocument();
  });
});

// dashboard/src/__tests__/useVirtualScroll.test.ts
import { renderHook } from "@testing-library/react-hooks";
import { useVirtualScroll } from "../hooks/useVirtualScroll";

describe("useVirtualScroll", () => {
  it("returns visible window of items", () => {
    const items = Array.from({ length: 10000 }, (_, i) => ({ id: i }));
    const { result } = renderHook(() =>
      useVirtualScroll({ items, itemHeight: 40, containerHeight: 400 })
    );
    // Should only render ~10 visible items + buffer
    expect(result.current.visibleItems.length).toBeLessThanOrEqual(15);
    expect(result.current.totalHeight).toBe(400000); // 10000 * 40
  });

  it("updates visible items on scroll", () => {
    const items = Array.from({ length: 1000 }, (_, i) => ({ id: i }));
    const { result } = renderHook(() =>
      useVirtualScroll({ items, itemHeight: 40, containerHeight: 400 })
    );
    result.current.onScroll(4000); // Scroll to item ~100
    expect(result.current.visibleItems[0].id).toBeGreaterThan(90);
  });
});
```

**Step 2: Run tests to verify they fail**

```bash
cd dashboard && npm test -- --watchAll=false
```

Expected: FAIL — components and hooks not yet implemented

**Step 3: Implement Dashboard v2 components**

Implement `MobileLayout.tsx`, `CustomAlertConfig.tsx`, `useVirtualScroll.ts`, and update `LLMCostPanel.tsx` with historical trend analysis and cost optimization recommendations. Update `PerformanceChart.tsx` with virtual scrolling for large datasets.

The implementation details follow React/TypeScript patterns established in the Phase 5 dashboard. Key additions:
- Virtual scrolling hook for performance with 10,000+ data points
- Mobile-responsive layout with hamburger menu and swipeable tabs
- Custom alert configuration UI with metric/threshold/channel selection
- LLM cost panel: add 30-day trend chart, cost-per-trade breakdown, model tier optimization suggestions

**Step 4: Run tests to verify they pass**

```bash
cd dashboard && npm test -- --watchAll=false
```

Expected: PASS — all Dashboard v2 component tests pass

**Step 5: Commit**

```bash
git add dashboard/src/
git commit -m "feat: Dashboard v2 — mobile layout, custom alerts, virtual scroll, LLM cost trends"
```

---

## Task 11: Disclaimer Integration

**Files:**
- Create: `src/evolve_trader/core/disclaimer.py`
- Create: `dashboard/src/components/common/Disclaimer.tsx`
- Create: `tests/unit/test_disclaimer.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_disclaimer.py
"""Tests for disclaimer integration across all user-facing outputs."""
import pytest
from evolve_trader.core.disclaimer import (
    Disclaimer,
    DisclaimerType,
    DisclaimerFormatter,
)


def test_disclaimer_type_enum():
    """All disclaimer types are defined."""
    assert DisclaimerType.DASHBOARD_FOOTER in DisclaimerType
    assert DisclaimerType.NOTIFICATION_FOOTER in DisclaimerType
    assert DisclaimerType.STRATEGY_OUTPUT in DisclaimerType
    assert DisclaimerType.FULL_ABOUT_PAGE in DisclaimerType


def test_disclaimer_short_text():
    """Short disclaimer for dashboard footer and notifications."""
    disclaimer = Disclaimer.get(DisclaimerType.DASHBOARD_FOOTER)
    assert "not investment advice" in disclaimer.lower()
    assert "research tool" in disclaimer.lower()
    assert len(disclaimer) < 500  # Short enough for a footer


def test_disclaimer_notification_text():
    """Notification disclaimer is concise."""
    disclaimer = Disclaimer.get(DisclaimerType.NOTIFICATION_FOOTER)
    assert "not investment advice" in disclaimer.lower()
    assert len(disclaimer) < 200  # Must fit in notification footer


def test_disclaimer_strategy_output():
    """Strategy output disclaimer warns against guaranteed returns."""
    disclaimer = Disclaimer.get(DisclaimerType.STRATEGY_OUTPUT)
    assert "no guarantee" in disclaimer.lower() or "not guaranteed" in disclaimer.lower()
    assert "past performance" in disclaimer.lower()


def test_disclaimer_full_about_page():
    """Full disclaimer for the about page includes all legal language."""
    disclaimer = Disclaimer.get(DisclaimerType.FULL_ABOUT_PAGE)
    assert "not investment advice" in disclaimer.lower()
    assert "research" in disclaimer.lower()
    assert "risk" in disclaimer.lower()
    assert len(disclaimer) > 500  # Full legal text


def test_disclaimer_formatter_wraps_strategy_output():
    """Formatter appends disclaimer to strategy output."""
    formatter = DisclaimerFormatter()
    raw_output = "BUY AAPL at $150. Target: $165."
    formatted = formatter.wrap(raw_output, DisclaimerType.STRATEGY_OUTPUT)
    assert formatted.startswith(raw_output)
    assert "not guaranteed" in formatted.lower() or "no guarantee" in formatted.lower()


def test_disclaimer_formatter_wraps_notification():
    """Formatter appends disclaimer to trade notifications."""
    formatter = DisclaimerFormatter()
    notification = "Trade executed: BUY 10 AAPL @ $150.25"
    formatted = formatter.wrap(notification, DisclaimerType.NOTIFICATION_FOOTER)
    assert formatted.startswith(notification)
    assert "not investment advice" in formatted.lower()


def test_disclaimer_all_types_non_empty():
    """Every disclaimer type produces non-empty text."""
    for dtype in DisclaimerType:
        text = Disclaimer.get(dtype)
        assert len(text) > 0, f"Disclaimer for {dtype} is empty"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_disclaimer.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.core.disclaimer'`

**Step 3: Implement the disclaimer module**

```python
# src/evolve_trader/core/disclaimer.py
"""Disclaimer integration for all user-facing outputs.

All system outputs that could be interpreted as financial guidance must
include disclaimers. This module provides standardized disclaimer text
and formatting utilities.
"""
from __future__ import annotations

from enum import Enum


class DisclaimerType(str, Enum):
    """Types of disclaimers for different output contexts."""
    DASHBOARD_FOOTER = "dashboard_footer"
    NOTIFICATION_FOOTER = "notification_footer"
    STRATEGY_OUTPUT = "strategy_output"
    FULL_ABOUT_PAGE = "full_about_page"


_DISCLAIMERS: dict[DisclaimerType, str] = {
    DisclaimerType.DASHBOARD_FOOTER: (
        "Evolve-Trader AI is a research tool for algorithmic trading experimentation. "
        "This is not investment advice. All trading involves risk of loss. "
        "Past performance does not guarantee future results."
    ),
    DisclaimerType.NOTIFICATION_FOOTER: (
        "This is not investment advice. All trading involves risk. "
        "Evolve-Trader AI is a research tool."
    ),
    DisclaimerType.STRATEGY_OUTPUT: (
        "DISCLAIMER: This strategy output is generated by an AI research system. "
        "Past performance is not guaranteed to predict future results. "
        "No guarantee of profit is made or implied. "
        "All trading involves substantial risk of loss. "
        "This is not investment advice."
    ),
    DisclaimerType.FULL_ABOUT_PAGE: (
        "IMPORTANT DISCLAIMER\n\n"
        "Evolve-Trader AI is an experimental research tool for algorithmic trading "
        "strategy development and evaluation. It is not a registered investment advisor, "
        "broker-dealer, or financial planner.\n\n"
        "This software is not investment advice. All information, signals, strategy "
        "outputs, and trade recommendations generated by this system are for research "
        "and educational purposes only.\n\n"
        "RISK WARNING: All forms of trading carry a high level of risk and may not be "
        "suitable for all investors. You should carefully consider your investment "
        "objectives, level of experience, and risk appetite before making any trading "
        "decisions. Past performance is not indicative of future results. There is no "
        "guarantee of profit, and you may lose some or all of your invested capital.\n\n"
        "NO WARRANTIES: The software is provided 'as is' without warranty of any kind. "
        "The developers make no representations or warranties regarding the accuracy, "
        "completeness, or reliability of any information or strategy generated by this "
        "system.\n\n"
        "HUMAN AUTHORIZATION: No live capital will be deployed without explicit human "
        "authorization. The system includes kill switches and risk limits, but these "
        "are not a substitute for human judgment and oversight.\n\n"
        "By using this software, you acknowledge that you understand these risks and "
        "agree that the developers are not liable for any trading losses."
    ),
}


class Disclaimer:
    """Provides standardized disclaimer text."""

    @staticmethod
    def get(disclaimer_type: DisclaimerType) -> str:
        """Get disclaimer text for the given context."""
        return _DISCLAIMERS[disclaimer_type]


class DisclaimerFormatter:
    """Formats outputs with appropriate disclaimers appended."""

    SEPARATOR = "\n\n---\n"

    def wrap(self, content: str, disclaimer_type: DisclaimerType) -> str:
        """Append disclaimer to content.

        Args:
            content: The original output text.
            disclaimer_type: Which disclaimer to append.

        Returns:
            Content with disclaimer appended after separator.
        """
        disclaimer_text = Disclaimer.get(disclaimer_type)
        return f"{content}{self.SEPARATOR}{disclaimer_text}"
```

Also create the React disclaimer component:

```typescript
// dashboard/src/components/common/Disclaimer.tsx
import React from "react";

interface DisclaimerProps {
  variant: "footer" | "notification" | "strategy" | "full";
  className?: string;
}

const DISCLAIMER_TEXT: Record<DisclaimerProps["variant"], string> = {
  footer:
    "Evolve-Trader AI is a research tool for algorithmic trading experimentation. " +
    "This is not investment advice. All trading involves risk of loss. " +
    "Past performance does not guarantee future results.",
  notification:
    "This is not investment advice. All trading involves risk. " +
    "Evolve-Trader AI is a research tool.",
  strategy:
    "DISCLAIMER: This strategy output is generated by an AI research system. " +
    "Past performance is not guaranteed to predict future results. " +
    "No guarantee of profit is made or implied.",
  full:
    "Evolve-Trader AI is an experimental research tool for algorithmic trading " +
    "strategy development. It is not a registered investment advisor. " +
    "All trading carries risk of loss. Past performance is not indicative of " +
    "future results. No live capital is deployed without explicit human authorization.",
};

export const Disclaimer: React.FC<DisclaimerProps> = ({
  variant,
  className = "",
}) => {
  return (
    <div
      className={`disclaimer disclaimer--${variant} ${className}`}
      data-testid={`disclaimer-${variant}`}
      role="contentinfo"
    >
      <p className="disclaimer__text">{DISCLAIMER_TEXT[variant]}</p>
    </div>
  );
};
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_disclaimer.py -v
```

Expected: PASS — all disclaimer tests pass

**Step 5: Commit**

```bash
git add src/evolve_trader/core/disclaimer.py dashboard/src/components/common/Disclaimer.tsx tests/unit/test_disclaimer.py
git commit -m "feat: disclaimer integration for dashboard, notifications, and strategy outputs"
```

---

## Task 12: Open-Source Release Prep (Optional)

**Files:**
- Create: `CONTRIBUTING.md`
- Create: `.github/ISSUE_TEMPLATE/bug_report.md`
- Create: `.github/ISSUE_TEMPLATE/feature_request.md`
- Create: `src/evolve_trader/demo/demo_runner.py`
- Create: `src/evolve_trader/demo/mock_data.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_demo_mode.py
"""Tests for demo mode (runs with mock data, no API keys)."""
import pytest
from evolve_trader.demo.demo_runner import DemoRunner, DemoConfig
from evolve_trader.demo.mock_data import MockDataGenerator


def test_demo_config_no_api_keys():
    """Demo mode requires no API keys."""
    config = DemoConfig()
    assert config.use_mock_data is True
    assert config.api_keys_required is False
    assert config.demo_capital == 100_000.0


def test_demo_runner_starts_without_external_deps():
    """Demo runner initializes with mock everything."""
    config = DemoConfig()
    runner = DemoRunner(config)
    assert runner.is_ready()
    assert runner.broker_type == "mock"
    assert runner.signal_sources_type == "mock"


def test_mock_data_generator_produces_signals():
    """Mock data generator produces realistic signal events."""
    generator = MockDataGenerator(seed=42)
    signals = generator.generate_signals(count=50)
    assert len(signals) == 50
    for signal in signals:
        assert signal.source in ["edgar_13f", "edgar_form4", "congressional"]
        assert 0.0 <= signal.confidence <= 1.0


def test_mock_data_generator_produces_market_data():
    """Mock data generator produces OHLCV market data."""
    generator = MockDataGenerator(seed=42)
    data = generator.generate_ohlcv(ticker="AAPL", days=252)
    assert len(data) == 252
    for bar in data:
        assert bar["open"] > 0
        assert bar["high"] >= bar["low"]
        assert bar["volume"] > 0


def test_demo_runner_runs_evolution_cycle():
    """Demo runner executes a full evolution cycle with mock data."""
    config = DemoConfig()
    runner = DemoRunner(config)
    result = runner.run_cycle()
    assert result.strategies_evaluated > 0
    assert result.evolution_events >= 0
    assert result.portfolio_value > 0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_demo_mode.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.demo.demo_runner'`

**Step 3: Implement demo mode and community files**

Implement `DemoRunner` and `MockDataGenerator` as lightweight wrappers that use synthetic data to demonstrate the full system pipeline without any API keys or external dependencies. Create `CONTRIBUTING.md` with development setup, PR guidelines, and code style requirements. Create GitHub issue templates for bug reports and feature requests.

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_demo_mode.py -v
```

Expected: PASS — demo mode runs end-to-end with mock data

**Step 5: Commit**

```bash
git add CONTRIBUTING.md .github/ISSUE_TEMPLATE/ src/evolve_trader/demo/ tests/unit/test_demo_mode.py
git commit -m "feat: open-source release prep — demo mode, CONTRIBUTING.md, issue templates"
```

---

## Task 13: Final Verification

**Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: ALL PASS — all Phase 1 through Phase 12 tests

**Step 2: Run linting and type checking**

```bash
ruff check src/evolve_trader/
mypy src/evolve_trader/ --ignore-missing-imports
```

Expected: No errors

**Step 3: Verify all disclaimer placements**

```bash
# Verify disclaimer component is imported in dashboard layout
grep -r "Disclaimer" dashboard/src/components/layout/
# Verify notification formatter uses disclaimer
grep -r "DisclaimerFormatter" src/evolve_trader/
```

Expected: Disclaimer present in all user-facing outputs

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "test: Phase 12 final verification — all tests passing"
```

---

## Parallelization Notes

Tasks in this phase have the following dependency structure:

```
Task 1 (Crypto Classifier) ─────┐
                                  ├── Task 2 (BITWISE10 Validation)
                                  │
Task 3 (IBKR Integration) ──────┤
                                  │
Task 4 (Polymarket Trading) ─────┤
                                  │
Task 5 (Research: Cadence) ──────┤
Task 6 (Research: Capacity) ─────┤
Task 7 (Research: Cross-Market) ─┤── Task 13 (Final Verification)
Task 8 (Research: Model Sens.) ──┤
Task 9 (Research: Adversarial) ──┤
                                  │
Task 10 (Dashboard v2) ──────────┤
                                  │
Task 11 (Disclaimer Integration) ┘

Task 12 (Open-Source Prep) ─── Optional, independent of all others
```

**Can run in parallel:**
- Tasks 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 are all independent of each other — run simultaneously
- Task 2 (BITWISE10) depends on Task 1 (Crypto Classifier) — run after Task 1
- Task 13 (Final Verification) depends on all other tasks — run last
