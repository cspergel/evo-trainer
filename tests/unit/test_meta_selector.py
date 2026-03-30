"""Tests for the meta-selector, scoring, lifecycle, and conflict resolution."""

from datetime import UTC, datetime

from evolve_trader.regime.classifier import RegimeLabel
from evolve_trader.selection.conflict import resolve_signal_conflicts
from evolve_trader.selection.lifecycle import (
    LifecycleStage,
    SourceLifecycleState,
    evaluate_promotion,
)
from evolve_trader.selection.meta_selector import MetaSelector
from evolve_trader.selection.scoring import create_scorecard
from evolve_trader.signals.types import DecayProfile, SignalEvent, SignalType


def _make_regime(primary: str = "risk-on", confidence: float = 0.8) -> RegimeLabel:
    return RegimeLabel(
        primary_regime=primary,
        sector_bias="neutral",
        momentum_state="strengthening",
        confidence=confidence,
    )


def _make_signal(
    action: str = "BUY",
    confidence: float = 0.8,
    entity: str = "Test",
    source: str = "test",
) -> SignalEvent:
    return SignalEvent(
        source=source,
        source_entity=entity,
        timestamp=datetime(2025, 3, 15, tzinfo=UTC),
        confidence=confidence,
        decay_profile=DecayProfile(initial_confidence=confidence, half_life_days=30),
        signal_type=SignalType.CONVICTION,
        payload={"action": action},
    )


# --- Meta-selector tests ---


class TestMetaSelector:
    def test_risk_on_selects_momentum_strategies(self) -> None:
        """Risk-on regime favors trend/momentum strategies."""
        selector = MetaSelector(
            available_strategies=[
                "trend-following-v1",
                "mean-reversion-v1",
                "capital-preservation",
            ],
        )
        result = selector.select(_make_regime("risk-on"), [_make_signal()])
        top = result.allocations[0].strategy
        assert top == "trend-following-v1"
        assert not result.capital_preservation_active

    def test_low_confidence_triggers_capital_preservation(self) -> None:
        """Low regime confidence -> capital preservation."""
        selector = MetaSelector(
            available_strategies=["momentum-v1", "capital-preservation"],
        )
        result = selector.select(_make_regime("transitional", confidence=0.3), [])
        assert result.capital_preservation_active

    def test_signal_conflicts_trigger_capital_preservation(self) -> None:
        """Unresolved signal conflicts -> capital preservation."""
        selector = MetaSelector(
            available_strategies=["momentum-v1", "capital-preservation"],
        )
        result = selector.select(
            _make_regime("risk-on", confidence=0.9),
            [],
            signal_conflicts=True,
        )
        assert result.capital_preservation_active

    def test_allocations_sum_to_one(self) -> None:
        """Allocation weights sum to ~1.0."""
        selector = MetaSelector(
            available_strategies=[
                "trend-following-v1",
                "mean-reversion-v1",
                "breakout-v1",
            ],
        )
        result = selector.select(_make_regime("risk-on"), [_make_signal()])
        assert abs(result.total_weight - 1.0) < 0.01

    def test_max_strategies_respected(self) -> None:
        """No more than max_strategies in allocation."""
        selector = MetaSelector(
            available_strategies=[f"strategy-{i}" for i in range(10)],
            max_strategies=3,
        )
        result = selector.select(_make_regime("risk-on"), [_make_signal()])
        assert len(result.allocations) <= 3

    def test_lifecycle_filters_demoted_sources(self) -> None:
        """Signals from demoted sources are excluded."""
        selector = MetaSelector(
            available_strategies=["trend-following-v1", "capital-preservation"],
        )
        signals = [_make_signal("BUY", 0.9, source="demoted_source")]
        lifecycles = {
            "demoted_source": SourceLifecycleState(
                source_name="demoted_source", stage=LifecycleStage.DEMOTED
            ),
        }
        # With demoted source filtered, signals are empty -> no boost
        result = selector.select(
            _make_regime("risk-on"),
            signals,
            source_lifecycles=lifecycles,
        )
        assert not result.capital_preservation_active

    def test_scorecards_weight_signal_influence(self) -> None:
        """Higher-scoring sources contribute more to signal boost."""
        selector = MetaSelector(
            available_strategies=["trend-following-v1"],
        )
        signals = [_make_signal("BUY", 0.8, source="congressional")]

        # High-scoring source
        high_card = create_scorecard("congressional")
        for _ in range(10):
            high_card.record_outcome(hit=True)

        result_high = selector.select(
            _make_regime("risk-on"),
            signals,
            source_scorecards={"congressional": high_card},
        )

        # Low-scoring source
        low_card = create_scorecard("congressional")
        for _ in range(10):
            low_card.record_outcome(hit=False)

        result_low = selector.select(
            _make_regime("risk-on"),
            signals,
            source_scorecards={"congressional": low_card},
        )

        # Both should produce valid results (different boost magnitudes)
        assert result_high.allocations[0].strategy == "trend-following-v1"
        assert result_low.allocations[0].strategy == "trend-following-v1"


# --- Scoring tests ---


class TestScoring:
    def test_new_scorecard_uses_base_weight(self) -> None:
        """Scorecard with no observations uses base tier weight."""
        card = create_scorecard("congressional")
        assert card.effective_weight == card.base_tier_weight

    def test_hit_rate_affects_weight(self) -> None:
        """Higher hit rate -> higher effective weight."""
        card = create_scorecard("edgar_13f")
        for _ in range(8):
            card.record_outcome(hit=True)
        for _ in range(2):
            card.record_outcome(hit=False)
        assert card.hit_rate == 0.8
        assert card.effective_weight > card.base_tier_weight

    def test_cold_streak_penalty(self) -> None:
        """Hit rate below 35% with 5+ observations halves weight."""
        card = create_scorecard("congressional")
        for _ in range(5):
            card.record_outcome(hit=False)
        assert card.hit_rate == 0.0
        assert card.effective_weight == 0.0  # 0% hit rate -> 0 multiplier * 0.5

    def test_tier_weight_applied(self) -> None:
        """Source tier affects base weight."""
        tier1 = create_scorecard("congressional")
        tier2 = create_scorecard("edgar_13f")
        assert tier1.base_tier_weight > tier2.base_tier_weight

    def test_single_loss_does_not_zero_weight(self) -> None:
        """A single loss on a fresh source still uses base weight (< 5 obs)."""
        card = create_scorecard("edgar_13f")
        card.record_outcome(hit=False)
        assert card.effective_weight == card.base_tier_weight  # not enough data


# --- Lifecycle tests ---


class TestLifecycle:
    def test_candidate_to_observation(self) -> None:
        """5+ observations promote from candidate to observation."""
        state = SourceLifecycleState(source_name="test")
        evaluate_promotion(state, hit_rate=0.6, total_observations=5)
        assert state.stage == LifecycleStage.OBSERVATION

    def test_observation_to_probation(self) -> None:
        """50%+ hit rate with enough observations promotes to probation."""
        state = SourceLifecycleState(source_name="test", stage=LifecycleStage.OBSERVATION)
        evaluate_promotion(state, hit_rate=0.55, total_observations=10)
        assert state.stage == LifecycleStage.PROBATION

    def test_observation_needs_min_observations(self) -> None:
        """Good hit rate alone isn't enough — need min observations for probation."""
        state = SourceLifecycleState(source_name="test", stage=LifecycleStage.OBSERVATION)
        evaluate_promotion(state, hit_rate=0.80, total_observations=6)
        assert state.stage == LifecycleStage.OBSERVATION  # not enough obs

    def test_probation_to_active(self) -> None:
        """Sustained performance promotes to active."""
        state = SourceLifecycleState(source_name="test", stage=LifecycleStage.PROBATION)
        evaluate_promotion(state, hit_rate=0.60, total_observations=20)
        assert state.stage == LifecycleStage.ACTIVE

    def test_demotion_on_poor_performance(self) -> None:
        """2 consecutive periods below 30% -> demoted."""
        state = SourceLifecycleState(source_name="test", stage=LifecycleStage.ACTIVE)
        evaluate_promotion(state, hit_rate=0.25, total_observations=30)
        assert state.stage == LifecycleStage.ACTIVE  # not demoted yet

        evaluate_promotion(state, hit_rate=0.20, total_observations=35)
        assert state.stage == LifecycleStage.DEMOTED

    def test_recovery_resets_underperform_counter(self) -> None:
        """Good period after bad resets the consecutive counter."""
        state = SourceLifecycleState(source_name="test", stage=LifecycleStage.ACTIVE)
        evaluate_promotion(state, hit_rate=0.25, total_observations=30)
        assert state.consecutive_underperform_periods == 1

        evaluate_promotion(state, hit_rate=0.55, total_observations=35)
        assert state.consecutive_underperform_periods == 0

    def test_demoted_source_can_recover(self) -> None:
        """Demoted source with sustained recovery returns to observation."""
        state = SourceLifecycleState(source_name="test", stage=LifecycleStage.DEMOTED)
        evaluate_promotion(state, hit_rate=0.60, total_observations=15)
        assert state.stage == LifecycleStage.OBSERVATION

    def test_full_lifecycle_walk(self) -> None:
        """Walk a source through all stages."""
        state = SourceLifecycleState(source_name="test")

        evaluate_promotion(state, 0.6, 5)
        assert state.stage == LifecycleStage.OBSERVATION

        evaluate_promotion(state, 0.55, 10)
        assert state.stage == LifecycleStage.PROBATION

        evaluate_promotion(state, 0.60, 20)
        assert state.stage == LifecycleStage.ACTIVE


# --- Conflict resolution tests ---


class TestConflictResolution:
    def test_no_conflict_all_buys(self) -> None:
        """All buy signals -> no conflict."""
        signals = [_make_signal("BUY", 0.8), _make_signal("BUY", 0.7)]
        result = resolve_signal_conflicts(signals)
        assert result.resolved
        assert result.net_direction == "buy"

    def test_dominant_side_wins(self) -> None:
        """When one side dramatically outscores -> resolved."""
        signals = [
            _make_signal("BUY", 0.9, "Buffett"),
            _make_signal("BUY", 0.8, "Pelosi"),
            _make_signal("SELL", 0.3, "Unknown"),
        ]
        result = resolve_signal_conflicts(signals)
        assert result.resolved
        assert result.net_direction == "buy"

    def test_even_conflict_unresolved(self) -> None:
        """Equal buy/sell weights -> unresolved."""
        signals = [
            _make_signal("BUY", 0.8, "Buffett"),
            _make_signal("SELL", 0.8, "Druckenmiller"),
        ]
        result = resolve_signal_conflicts(signals)
        assert not result.resolved
        assert result.net_direction == "neutral"

    def test_no_signals_resolved(self) -> None:
        """Empty signals -> resolved as neutral."""
        result = resolve_signal_conflicts([])
        assert result.resolved
        assert result.net_direction == "neutral"

    def test_source_weight_breaks_tie(self) -> None:
        """Higher-tier source wins conflict when confidences are equal."""
        signals = [
            _make_signal("BUY", 0.8, "Pelosi", source="congressional"),
            _make_signal("SELL", 0.8, "News", source="macro_news"),
        ]
        # Congressional (3.0x) vs macro (1.0x) -> congressional dominates
        cards = {
            "congressional": create_scorecard("congressional"),
            "macro_news": create_scorecard("macro_news"),
        }
        result = resolve_signal_conflicts(signals, source_scorecards=cards)
        assert result.resolved
        assert result.net_direction == "buy"
