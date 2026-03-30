"""Tests for position sizing skills and composition interface."""

import pytest

from evolve_trader.core.risk_constraints import PortfolioState, RiskConstraints
from evolve_trader.sizing.composition import TradeProposal, compose_trade
from evolve_trader.sizing.models import (
    SizingContext,
    SizingMethod,
    SizingResult,
    SizingSkill,
    compute_kelly_size,
    compute_regime_adjusted_size,
    compute_sizing,
    compute_volatility_target_size,
)

# --- Kelly Criterion tests ---


class TestKelly:
    def test_positive_edge_produces_position(self) -> None:
        """Positive expected value → non-zero position."""
        size = compute_kelly_size(win_rate=0.6, avg_win=0.10, avg_loss=0.05)
        assert size > 0

    def test_negative_edge_produces_zero(self) -> None:
        """Negative expected value → zero position."""
        size = compute_kelly_size(win_rate=0.3, avg_win=0.05, avg_loss=0.10)
        assert size == 0.0

    def test_half_kelly_is_conservative(self) -> None:
        """Half Kelly < full Kelly."""
        half = compute_kelly_size(0.6, 0.10, 0.05, kelly_fraction=0.5)
        full = compute_kelly_size(0.6, 0.10, 0.05, kelly_fraction=1.0)
        assert half < full

    def test_zero_loss_returns_zero(self) -> None:
        """Zero average loss → zero (avoid division by zero)."""
        assert compute_kelly_size(0.6, 0.10, 0.0) == 0.0

    def test_capped_at_one(self) -> None:
        """Position size never exceeds 100%."""
        size = compute_kelly_size(0.95, 0.50, 0.01, kelly_fraction=1.0)
        assert size <= 1.0


# --- Volatility Targeting tests ---


class TestVolatilityTargeting:
    def test_high_vol_asset_gets_smaller_position(self) -> None:
        """High-volatility asset → smaller position."""
        high_vol = compute_volatility_target_size(0.15, 0.40)
        low_vol = compute_volatility_target_size(0.15, 0.10)
        assert high_vol < low_vol

    def test_zero_volatility_returns_zero(self) -> None:
        """Zero asset volatility → zero (avoid division by zero)."""
        assert compute_volatility_target_size(0.15, 0.0) == 0.0

    def test_capped_at_one(self) -> None:
        """Never exceeds 100% of portfolio."""
        size = compute_volatility_target_size(0.50, 0.01)
        assert size <= 1.0


# --- Regime Adjusted tests ---


class TestRegimeAdjusted:
    def test_risk_off_reduces_size(self) -> None:
        """Risk-off regime → position reduced."""
        risk_on = compute_regime_adjusted_size(0.05, "risk-on")
        risk_off = compute_regime_adjusted_size(0.05, "risk-off")
        assert risk_off < risk_on

    def test_unknown_regime_uses_default(self) -> None:
        """Unknown regime uses 0.7x multiplier."""
        size = compute_regime_adjusted_size(0.10, "unknown-regime")
        assert size == pytest.approx(0.07)


# --- Compute Sizing dispatch ---


class TestComputeSizing:
    def _make_context(self) -> SizingContext:
        return SizingContext(
            portfolio_value=100_000,
            current_price=150.0,
            win_rate=0.55,
            avg_win=0.08,
            avg_loss=0.05,
            recent_volatility=0.25,
            regime="risk-on",
            existing_exposure=0.3,
        )

    def test_kelly_sizing(self) -> None:
        """Kelly method produces valid SizingResult."""
        skill = SizingSkill("kelly-v1", SizingMethod.KELLY, "Half Kelly", {"kelly_fraction": 0.5})
        result = compute_sizing(self._make_context(), skill)
        assert isinstance(result, SizingResult)
        assert result.position_size_pct > 0
        assert result.shares > 0
        assert "Kelly" in result.rationale

    def test_volatility_sizing(self) -> None:
        """Volatility method produces valid SizingResult."""
        skill = SizingSkill(
            "vol-target-v1",
            SizingMethod.VOLATILITY_TARGET,
            "Vol target",
            {"target_volatility": 0.15},
        )
        result = compute_sizing(self._make_context(), skill)
        assert result.position_size_pct > 0
        assert "olatilit" in result.rationale

    def test_regime_sizing(self) -> None:
        """Regime method adjusts based on current regime."""
        skill = SizingSkill(
            "regime-v1",
            SizingMethod.REGIME_ADJUSTED,
            "Regime adjusted",
            {"base_size": 0.04},
        )
        result = compute_sizing(self._make_context(), skill)
        assert result.position_size_pct > 0

    def test_fixed_fractional_fallback(self) -> None:
        """Unknown method falls back to fixed fractional."""
        skill = SizingSkill(
            "fixed-v1",
            SizingMethod.FIXED_FRACTIONAL,
            "Fixed",
            {"fixed_fraction": 0.02},
        )
        result = compute_sizing(self._make_context(), skill)
        assert result.position_size_pct == 0.02

    def test_existing_exposure_caps_sizing(self) -> None:
        """Position capped by remaining portfolio capacity."""
        ctx = self._make_context()
        ctx.existing_exposure = 0.95  # already 95% exposed
        skill = SizingSkill(
            "fixed-v1",
            SizingMethod.FIXED_FRACTIONAL,
            "Fixed",
            {"fixed_fraction": 0.10},
        )
        result = compute_sizing(ctx, skill)
        assert result.position_size_pct <= 0.05  # capped at remaining 5%


# --- Composition interface tests ---


class TestComposition:
    def test_approved_trade(self) -> None:
        """Small trade within constraints is approved."""
        proposal = TradeProposal(
            strategy_skill="momentum-v1",
            ticker="AAPL",
            direction="BUY",
            sector="Technology",
            confidence=0.8,
            regime="risk-on",
            rationale="Momentum confirmed",
        )
        sizing = SizingResult(
            position_size_pct=0.03,
            position_value=3000,
            shares=20,
            method_used="kelly",
            rationale="Half Kelly",
        )
        portfolio = PortfolioState(
            total_value=100_000,
            positions={},
            sector_exposure={},
            current_drawdown=0.0,
        )
        trade = compose_trade(proposal, sizing, RiskConstraints(), portfolio)
        assert trade.is_approved

    def test_trade_exceeding_position_limit_vetoed(self) -> None:
        """Trade exceeding 5% position limit is vetoed."""
        proposal = TradeProposal(
            strategy_skill="momentum-v1",
            ticker="AAPL",
            direction="BUY",
            sector="Technology",
            confidence=0.8,
            regime="risk-on",
            rationale="Momentum confirmed",
        )
        sizing = SizingResult(
            position_size_pct=0.06,
            position_value=6000,
            shares=40,
            method_used="kelly",
            rationale="Full Kelly (aggressive)",
        )
        portfolio = PortfolioState(
            total_value=100_000,
            positions={},
            sector_exposure={},
            current_drawdown=0.0,
        )
        trade = compose_trade(proposal, sizing, RiskConstraints(), portfolio)
        assert not trade.is_approved

    def test_sell_always_passes_constraints(self) -> None:
        """Sells reduce risk and always pass constraints."""
        proposal = TradeProposal(
            strategy_skill="momentum-v1",
            ticker="AAPL",
            direction="SELL",
            sector="Technology",
            confidence=0.8,
            regime="risk-off",
            rationale="Exit position",
        )
        sizing = SizingResult(
            position_size_pct=0.05,
            position_value=5000,
            shares=33,
            method_used="kelly",
            rationale="Full exit",
        )
        portfolio = PortfolioState(
            total_value=100_000,
            positions={"AAPL": 5000},
            sector_exposure={"Technology": 0.05},
            current_drawdown=0.25,  # even during drawdown
        )
        trade = compose_trade(proposal, sizing, RiskConstraints(), portfolio)
        assert trade.is_approved

    def test_drawdown_vetoes_buy(self) -> None:
        """Buy during 20%+ drawdown is vetoed."""
        proposal = TradeProposal(
            strategy_skill="momentum-v1",
            ticker="AAPL",
            direction="BUY",
            sector="Technology",
            confidence=0.9,
            regime="risk-on",
            rationale="Strong signal",
        )
        sizing = SizingResult(
            position_size_pct=0.02,
            position_value=2000,
            shares=13,
            method_used="kelly",
            rationale="Conservative",
        )
        portfolio = PortfolioState(
            total_value=80_000,
            positions={},
            sector_exposure={},
            current_drawdown=0.21,
        )
        trade = compose_trade(proposal, sizing, RiskConstraints(), portfolio)
        assert not trade.is_approved
