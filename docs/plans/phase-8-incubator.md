# Phase 8: Strategy Incubator — Detailed Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a concurrent tournament of candidate strategies. Maintain a small population (5-10), implement 3 initial generation methods (mutation, crossover, regime-conditioned search), a multi-component fitness function, and a fossil record of eliminated strategies. Add remaining generation methods (academic mining, anomaly detection, counter-strategy, meta-strategy) incrementally.

**Architecture:** The Strategy Incubator runs a parallel population of candidate strategies isolated from production. Each candidate executes its SKILL.md against live market data via the shared production signal layer (read-only). Candidates operate on isolated paper portfolios. A tournament loop cycles through phases: Seeding → Incubation → Evaluation → Selection → Reproduction → Promotion. Generation methods produce new candidates via LLM-driven mutation, crossover, and regime-conditioned search. A fossil record archives eliminated strategies with failure conditions for potential resurrection.

**Tech Stack:** Python 3.12+, LiteLLM, PostgreSQL, numpy, pytest

**Parent plan:** [`2026-03-29-evolve-trader-master-plan.md`](2026-03-29-evolve-trader-master-plan.md)

**Prerequisites:** Phase 7 complete. Production strategy pipeline, signal layer, meta-selector, risk management, orchestrator, and monitoring all verified and operational.

---

## Task 1: Tournament Architecture

**Files:**
- Create: `src/evolve_trader/incubator/__init__.py`
- Create: `src/evolve_trader/incubator/tournament.py`
- Create: `src/evolve_trader/incubator/candidate.py`
- Create: `tests/unit/test_tournament.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_tournament.py
"""Tests for the Strategy Incubator tournament architecture."""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from evolve_trader.incubator.candidate import (
    Candidate,
    CandidateStatus,
    CandidateOrigin,
    PaperPortfolio,
)
from evolve_trader.incubator.tournament import Tournament, TournamentConfig


def test_candidate_status_enum():
    """CandidateStatus has all required lifecycle states."""
    assert CandidateStatus.SEEDED.value == "SEEDED"
    assert CandidateStatus.INCUBATING.value == "INCUBATING"
    assert CandidateStatus.EVALUATING.value == "EVALUATING"
    assert CandidateStatus.PROMOTED.value == "PROMOTED"
    assert CandidateStatus.ELIMINATED.value == "ELIMINATED"


def test_candidate_origin_enum():
    """CandidateOrigin tracks how a candidate was generated."""
    assert CandidateOrigin.MUTATION.value == "MUTATION"
    assert CandidateOrigin.CROSSOVER.value == "CROSSOVER"
    assert CandidateOrigin.REGIME_SEARCH.value == "REGIME_SEARCH"
    assert CandidateOrigin.ACADEMIC.value == "ACADEMIC"
    assert CandidateOrigin.ANOMALY.value == "ANOMALY"
    assert CandidateOrigin.COUNTER_STRATEGY.value == "COUNTER_STRATEGY"
    assert CandidateOrigin.META_STRATEGY.value == "META_STRATEGY"
    assert CandidateOrigin.SEED.value == "SEED"


def test_candidate_has_required_fields():
    """Candidate model carries all metadata for tournament participation."""
    candidate = Candidate(
        id="cand-001",
        name="momentum-variant-1",
        skill_md_path="incubator/momentum-variant-1.skill.md",
        origin=CandidateOrigin.MUTATION,
        parent_ids=["prod-momentum-v3"],
        hypothesis="Adding RSI filter improves risk-off performance",
        status=CandidateStatus.SEEDED,
        generation=1,
        created_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        portfolio=PaperPortfolio(initial_capital=100_000.0),
    )
    assert candidate.name == "momentum-variant-1"
    assert candidate.origin == CandidateOrigin.MUTATION
    assert candidate.status == CandidateStatus.SEEDED
    assert len(candidate.parent_ids) == 1


def test_paper_portfolio_isolation():
    """PaperPortfolio tracks positions independently from production."""
    portfolio = PaperPortfolio(initial_capital=100_000.0)
    assert portfolio.cash == 100_000.0
    assert portfolio.positions == {}
    assert portfolio.total_value == 100_000.0

    portfolio.execute_trade("AAPL", direction="BUY", quantity=10, price=150.0)
    assert portfolio.cash == 98_500.0
    assert portfolio.positions["AAPL"]["quantity"] == 10
    assert portfolio.positions["AAPL"]["avg_price"] == 150.0

    portfolio.execute_trade("AAPL", direction="SELL", quantity=5, price=160.0)
    assert portfolio.cash == 99_300.0
    assert portfolio.positions["AAPL"]["quantity"] == 5


def test_paper_portfolio_rejects_oversell():
    """PaperPortfolio raises on selling more than held."""
    portfolio = PaperPortfolio(initial_capital=100_000.0)
    portfolio.execute_trade("AAPL", direction="BUY", quantity=10, price=150.0)
    with pytest.raises(ValueError, match="Insufficient"):
        portfolio.execute_trade("AAPL", direction="SELL", quantity=20, price=160.0)


def test_paper_portfolio_rejects_insufficient_cash():
    """PaperPortfolio raises on buying without enough cash."""
    portfolio = PaperPortfolio(initial_capital=1_000.0)
    with pytest.raises(ValueError, match="Insufficient cash"):
        portfolio.execute_trade("AAPL", direction="BUY", quantity=100, price=150.0)


def test_tournament_config_defaults():
    """TournamentConfig has sensible defaults."""
    config = TournamentConfig()
    assert config.max_population == 10
    assert config.incubation_days == 30
    assert config.elimination_pct == 0.30
    assert config.reproduction_pct == 0.10
    assert config.promotion_min_days == 60
    assert config.promotion_min_sharpe == 0.5


def test_tournament_initializes_empty():
    """Tournament starts with no candidates."""
    config = TournamentConfig(max_population=10)
    tournament = Tournament(config=config)
    assert tournament.population == []
    assert tournament.generation == 0


def test_tournament_add_candidate():
    """Tournament accepts candidates up to max population."""
    config = TournamentConfig(max_population=3)
    tournament = Tournament(config=config)

    for i in range(3):
        candidate = Candidate(
            id=f"cand-{i:03d}",
            name=f"strategy-{i}",
            skill_md_path=f"incubator/strategy-{i}.skill.md",
            origin=CandidateOrigin.SEED,
            parent_ids=[],
            hypothesis="Seed strategy",
            status=CandidateStatus.SEEDED,
            generation=0,
            created_at=datetime.now(timezone.utc),
            portfolio=PaperPortfolio(initial_capital=100_000.0),
        )
        tournament.add_candidate(candidate)

    assert len(tournament.population) == 3


def test_tournament_rejects_over_max_population():
    """Tournament raises when population limit is reached."""
    config = TournamentConfig(max_population=2)
    tournament = Tournament(config=config)

    for i in range(2):
        candidate = Candidate(
            id=f"cand-{i:03d}",
            name=f"strategy-{i}",
            skill_md_path=f"incubator/strategy-{i}.skill.md",
            origin=CandidateOrigin.SEED,
            parent_ids=[],
            hypothesis="Seed strategy",
            status=CandidateStatus.SEEDED,
            generation=0,
            created_at=datetime.now(timezone.utc),
            portfolio=PaperPortfolio(initial_capital=100_000.0),
        )
        tournament.add_candidate(candidate)

    overflow = Candidate(
        id="cand-overflow",
        name="strategy-overflow",
        skill_md_path="incubator/strategy-overflow.skill.md",
        origin=CandidateOrigin.SEED,
        parent_ids=[],
        hypothesis="Overflow",
        status=CandidateStatus.SEEDED,
        generation=0,
        created_at=datetime.now(timezone.utc),
        portfolio=PaperPortfolio(initial_capital=100_000.0),
    )
    with pytest.raises(ValueError, match="Population limit"):
        tournament.add_candidate(overflow)


def test_tournament_get_active_candidates():
    """Tournament filters to only non-eliminated, non-promoted candidates."""
    config = TournamentConfig(max_population=5)
    tournament = Tournament(config=config)

    statuses = [
        CandidateStatus.INCUBATING,
        CandidateStatus.EVALUATING,
        CandidateStatus.ELIMINATED,
        CandidateStatus.PROMOTED,
        CandidateStatus.SEEDED,
    ]
    for i, status in enumerate(statuses):
        candidate = Candidate(
            id=f"cand-{i:03d}",
            name=f"strategy-{i}",
            skill_md_path=f"incubator/strategy-{i}.skill.md",
            origin=CandidateOrigin.SEED,
            parent_ids=[],
            hypothesis="Test",
            status=status,
            generation=0,
            created_at=datetime.now(timezone.utc),
            portfolio=PaperPortfolio(initial_capital=100_000.0),
        )
        tournament.add_candidate(candidate)

    active = tournament.get_active_candidates()
    assert len(active) == 3  # INCUBATING, EVALUATING, SEEDED
    assert all(
        c.status not in (CandidateStatus.ELIMINATED, CandidateStatus.PROMOTED)
        for c in active
    )


def test_candidate_reads_production_signals_readonly():
    """Candidate accesses production signal layer without write capability."""
    signal_layer = MagicMock()
    signal_layer.get_active_signals.return_value = [
        {"source": "edgar_13f", "ticker": "AAPL", "confidence": 0.8}
    ]
    signal_layer.write_signal = MagicMock(side_effect=PermissionError)

    candidate = Candidate(
        id="cand-001",
        name="test-strategy",
        skill_md_path="incubator/test-strategy.skill.md",
        origin=CandidateOrigin.SEED,
        parent_ids=[],
        hypothesis="Test readonly",
        status=CandidateStatus.INCUBATING,
        generation=0,
        created_at=datetime.now(timezone.utc),
        portfolio=PaperPortfolio(initial_capital=100_000.0),
    )

    signals = signal_layer.get_active_signals()
    assert len(signals) == 1
    with pytest.raises(PermissionError):
        signal_layer.write_signal({"source": "fake"})
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_tournament.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.incubator'`

**Step 3: Implement the tournament architecture**

```python
# src/evolve_trader/incubator/__init__.py
"""Strategy Incubator — concurrent tournament of candidate strategies."""
```

```python
# src/evolve_trader/incubator/candidate.py
"""Candidate strategy model and paper portfolio for tournament participation."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class CandidateStatus(Enum):
    """Lifecycle states for a tournament candidate."""
    SEEDED = "SEEDED"
    INCUBATING = "INCUBATING"
    EVALUATING = "EVALUATING"
    PROMOTED = "PROMOTED"
    ELIMINATED = "ELIMINATED"


class CandidateOrigin(Enum):
    """How a candidate strategy was generated."""
    SEED = "SEED"
    MUTATION = "MUTATION"
    CROSSOVER = "CROSSOVER"
    REGIME_SEARCH = "REGIME_SEARCH"
    ACADEMIC = "ACADEMIC"
    ANOMALY = "ANOMALY"
    COUNTER_STRATEGY = "COUNTER_STRATEGY"
    META_STRATEGY = "META_STRATEGY"


@dataclass
class PaperPortfolio:
    """Isolated paper trading portfolio for a candidate strategy.

    Tracks positions, cash, and P&L independently from production.
    """
    initial_capital: float
    cash: float = field(init=False)
    positions: dict[str, dict[str, float]] = field(default_factory=dict)
    trade_history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.cash = self.initial_capital

    @property
    def total_value(self) -> float:
        """Total portfolio value = cash + sum of position values."""
        position_value = sum(
            pos["quantity"] * pos["avg_price"]
            for pos in self.positions.values()
        )
        return self.cash + position_value

    def execute_trade(
        self, ticker: str, direction: str, quantity: int, price: float
    ) -> None:
        """Execute a paper trade, updating cash and positions.

        Args:
            ticker: Stock symbol.
            direction: "BUY" or "SELL".
            quantity: Number of shares.
            price: Execution price per share.

        Raises:
            ValueError: If insufficient cash for BUY or insufficient
                        shares for SELL.
        """
        cost = quantity * price

        if direction == "BUY":
            if cost > self.cash:
                raise ValueError(
                    f"Insufficient cash: need {cost:.2f}, have {self.cash:.2f}"
                )
            self.cash -= cost
            if ticker in self.positions:
                existing = self.positions[ticker]
                total_qty = existing["quantity"] + quantity
                existing["avg_price"] = (
                    (existing["quantity"] * existing["avg_price"] + cost) / total_qty
                )
                existing["quantity"] = total_qty
            else:
                self.positions[ticker] = {
                    "quantity": quantity,
                    "avg_price": price,
                }
        elif direction == "SELL":
            if ticker not in self.positions:
                raise ValueError(
                    f"Insufficient shares: no position in {ticker}"
                )
            existing = self.positions[ticker]
            if quantity > existing["quantity"]:
                raise ValueError(
                    f"Insufficient shares: need {quantity}, have {existing['quantity']}"
                )
            self.cash += cost
            existing["quantity"] -= quantity
            if existing["quantity"] == 0:
                del self.positions[ticker]
        else:
            raise ValueError(f"Unknown direction: {direction}")

        self.trade_history.append({
            "ticker": ticker,
            "direction": direction,
            "quantity": quantity,
            "price": price,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


@dataclass
class Candidate:
    """A candidate strategy participating in the incubator tournament."""
    id: str
    name: str
    skill_md_path: str
    origin: CandidateOrigin
    parent_ids: list[str]
    hypothesis: str
    status: CandidateStatus
    generation: int
    created_at: datetime
    portfolio: PaperPortfolio
    metadata: dict[str, Any] = field(default_factory=dict)
    fitness_scores: dict[str, float] = field(default_factory=dict)
    incubation_start: datetime | None = None
    evaluation_results: dict[str, Any] = field(default_factory=dict)
```

```python
# src/evolve_trader/incubator/tournament.py
"""Tournament manager for the Strategy Incubator."""
from __future__ import annotations

from dataclasses import dataclass, field

from evolve_trader.incubator.candidate import Candidate, CandidateStatus


@dataclass
class TournamentConfig:
    """Configuration for tournament parameters."""
    max_population: int = 10
    incubation_days: int = 30
    elimination_pct: float = 0.30
    reproduction_pct: float = 0.10
    promotion_min_days: int = 60
    promotion_min_sharpe: float = 0.5
    initial_capital: float = 100_000.0


class Tournament:
    """Manages the population of candidate strategies.

    Provides add/remove/query operations on the candidate pool.
    Tournament phases (seeding, evaluation, selection, etc.) are
    handled by the phases module.
    """

    def __init__(self, config: TournamentConfig):
        self._config = config
        self._population: list[Candidate] = []
        self._generation: int = 0

    @property
    def config(self) -> TournamentConfig:
        return self._config

    @property
    def population(self) -> list[Candidate]:
        return list(self._population)

    @property
    def generation(self) -> int:
        return self._generation

    @generation.setter
    def generation(self, value: int) -> None:
        self._generation = value

    def add_candidate(self, candidate: Candidate) -> None:
        """Add a candidate to the tournament.

        Raises:
            ValueError: If the population limit has been reached.
        """
        if len(self._population) >= self._config.max_population:
            raise ValueError(
                f"Population limit reached: {self._config.max_population}"
            )
        self._population.append(candidate)

    def remove_candidate(self, candidate_id: str) -> Candidate | None:
        """Remove and return a candidate by ID, or None if not found."""
        for i, c in enumerate(self._population):
            if c.id == candidate_id:
                return self._population.pop(i)
        return None

    def get_candidate(self, candidate_id: str) -> Candidate | None:
        """Look up a candidate by ID."""
        for c in self._population:
            if c.id == candidate_id:
                return c
        return None

    def get_active_candidates(self) -> list[Candidate]:
        """Return candidates that are not eliminated or promoted."""
        return [
            c for c in self._population
            if c.status not in (CandidateStatus.ELIMINATED, CandidateStatus.PROMOTED)
        ]
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_tournament.py -v
```

Expected: PASS — all tournament architecture tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/incubator/__init__.py src/evolve_trader/incubator/candidate.py src/evolve_trader/incubator/tournament.py tests/unit/test_tournament.py
git commit -m "feat: tournament architecture with candidate model and paper portfolio"
```

---

## Task 2: Tournament Phases

**Files:**
- Create: `src/evolve_trader/incubator/phases.py`
- Create: `tests/unit/test_tournament_phases.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_tournament_phases.py
"""Tests for tournament phase transitions."""
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock, patch

from evolve_trader.incubator.candidate import (
    Candidate,
    CandidateOrigin,
    CandidateStatus,
    PaperPortfolio,
)
from evolve_trader.incubator.tournament import Tournament, TournamentConfig
from evolve_trader.incubator.phases import (
    TournamentPhaseManager,
    seed_candidates,
    transition_to_incubation,
    evaluate_candidates,
    select_candidates,
    reproduce_candidates,
    check_promotions,
)


def _make_candidate(
    id: str,
    status: CandidateStatus = CandidateStatus.SEEDED,
    generation: int = 0,
    created_at: datetime | None = None,
    sharpe: float = 0.0,
    max_drawdown: float = 0.0,
) -> Candidate:
    """Helper to create test candidates."""
    c = Candidate(
        id=id,
        name=f"strategy-{id}",
        skill_md_path=f"incubator/strategy-{id}.skill.md",
        origin=CandidateOrigin.SEED,
        parent_ids=[],
        hypothesis="Test candidate",
        status=status,
        generation=generation,
        created_at=created_at or datetime.now(timezone.utc),
        portfolio=PaperPortfolio(initial_capital=100_000.0),
    )
    c.fitness_scores = {"sharpe": sharpe, "max_drawdown": max_drawdown}
    return c


def test_seed_candidates_respects_max_per_generation():
    """Seeding phase adds at most max_population candidates."""
    config = TournamentConfig(max_population=10)
    tournament = Tournament(config=config)

    candidates = [_make_candidate(f"seed-{i}") for i in range(15)]
    added = seed_candidates(tournament, candidates)

    assert added == 10
    assert len(tournament.population) == 10


def test_seed_candidates_skips_when_full():
    """Seeding does nothing when population is already at max."""
    config = TournamentConfig(max_population=3)
    tournament = Tournament(config=config)
    for i in range(3):
        tournament.add_candidate(_make_candidate(f"existing-{i}"))

    added = seed_candidates(tournament, [_make_candidate("new-0")])
    assert added == 0


def test_transition_to_incubation():
    """SEEDED candidates transition to INCUBATING with start timestamp."""
    config = TournamentConfig(incubation_days=30)
    tournament = Tournament(config=config)
    c = _make_candidate("inc-1", status=CandidateStatus.SEEDED)
    tournament.add_candidate(c)

    transitioned = transition_to_incubation(tournament)
    assert transitioned == 1
    assert c.status == CandidateStatus.INCUBATING
    assert c.incubation_start is not None


def test_evaluate_candidates_after_incubation_period():
    """Candidates move to EVALUATING after incubation_days elapse."""
    config = TournamentConfig(incubation_days=30)
    tournament = Tournament(config=config)

    old_candidate = _make_candidate("eval-1", status=CandidateStatus.INCUBATING)
    old_candidate.incubation_start = datetime.now(timezone.utc) - timedelta(days=31)
    tournament.add_candidate(old_candidate)

    fresh_candidate = _make_candidate("eval-2", status=CandidateStatus.INCUBATING)
    fresh_candidate.incubation_start = datetime.now(timezone.utc) - timedelta(days=5)
    tournament.add_candidate(fresh_candidate)

    evaluated = evaluate_candidates(tournament)
    assert evaluated == 1
    assert old_candidate.status == CandidateStatus.EVALUATING
    assert fresh_candidate.status == CandidateStatus.INCUBATING


def test_select_candidates_eliminates_bottom_30_pct():
    """Selection eliminates bottom 30% by fitness."""
    config = TournamentConfig(max_population=10, elimination_pct=0.30)
    tournament = Tournament(config=config)

    # 10 candidates with fitness scores 0.1 to 1.0
    for i in range(10):
        c = _make_candidate(
            f"sel-{i}",
            status=CandidateStatus.EVALUATING,
            sharpe=(i + 1) * 0.1,
        )
        c.fitness_scores["composite"] = (i + 1) * 0.1
        tournament.add_candidate(c)

    eliminated = select_candidates(tournament)
    assert eliminated == 3  # bottom 30%

    eliminated_candidates = [
        c for c in tournament.population
        if c.status == CandidateStatus.ELIMINATED
    ]
    assert len(eliminated_candidates) == 3
    # The 3 with lowest composite fitness should be eliminated
    eliminated_scores = sorted(
        c.fitness_scores["composite"] for c in eliminated_candidates
    )
    assert eliminated_scores == [0.1, 0.2, 0.3]


def test_reproduce_candidates_from_top_10_pct():
    """Reproduction creates offspring from top 10% of candidates."""
    config = TournamentConfig(
        max_population=15,
        reproduction_pct=0.10,
    )
    tournament = Tournament(config=config)

    for i in range(10):
        c = _make_candidate(
            f"rep-{i}",
            status=CandidateStatus.EVALUATING,
            sharpe=(i + 1) * 0.15,
        )
        c.fitness_scores["composite"] = (i + 1) * 0.15
        tournament.add_candidate(c)

    offspring = reproduce_candidates(tournament)
    assert offspring >= 1  # At least the top candidate spawns offspring
    assert all(
        c.status == CandidateStatus.SEEDED
        for c in tournament.population
        if c.generation == tournament.generation + 1
    )


def test_check_promotions_sharpe_and_duration():
    """Promotion requires both minimum Sharpe and minimum incubation days."""
    config = TournamentConfig(
        promotion_min_days=60,
        promotion_min_sharpe=0.5,
    )
    tournament = Tournament(config=config)

    # Good Sharpe, enough days -> promotes
    good = _make_candidate("promo-1", status=CandidateStatus.EVALUATING, sharpe=0.8)
    good.incubation_start = datetime.now(timezone.utc) - timedelta(days=65)
    good.fitness_scores["composite"] = 0.8
    tournament.add_candidate(good)

    # Good Sharpe, too few days -> stays
    young = _make_candidate("promo-2", status=CandidateStatus.EVALUATING, sharpe=0.7)
    young.incubation_start = datetime.now(timezone.utc) - timedelta(days=30)
    young.fitness_scores["composite"] = 0.7
    tournament.add_candidate(young)

    # Enough days, bad Sharpe -> stays
    weak = _make_candidate("promo-3", status=CandidateStatus.EVALUATING, sharpe=0.2)
    weak.incubation_start = datetime.now(timezone.utc) - timedelta(days=90)
    weak.fitness_scores["composite"] = 0.2
    tournament.add_candidate(weak)

    promoted = check_promotions(tournament)
    assert promoted == 1
    assert good.status == CandidateStatus.PROMOTED
    assert young.status == CandidateStatus.EVALUATING
    assert weak.status == CandidateStatus.EVALUATING


def test_phase_manager_runs_full_cycle():
    """TournamentPhaseManager orchestrates all phases in sequence."""
    config = TournamentConfig(max_population=10, incubation_days=0)
    manager = TournamentPhaseManager(config=config)

    seeds = [_make_candidate(f"cycle-{i}") for i in range(5)]
    result = manager.run_cycle(seeds)

    assert "seeded" in result
    assert "incubated" in result
    assert "evaluated" in result
    assert "eliminated" in result
    assert "reproduced" in result
    assert "promoted" in result
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_tournament_phases.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.incubator.phases'`

**Step 3: Implement tournament phases**

```python
# src/evolve_trader/incubator/phases.py
"""Tournament phase transitions for the Strategy Incubator.

Phases:
  1. Seeding     — add new candidates (max 10 per generation)
  2. Incubation  — candidates trade on paper for incubation_days
  3. Evaluation  — compute fitness scores after incubation
  4. Selection   — eliminate bottom elimination_pct
  5. Reproduction — top reproduction_pct spawn offspring
  6. Promotion   — candidates meeting Sharpe + duration thresholds graduate
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

from evolve_trader.incubator.candidate import (
    Candidate,
    CandidateStatus,
    PaperPortfolio,
)
from evolve_trader.incubator.tournament import Tournament, TournamentConfig


def seed_candidates(
    tournament: Tournament, candidates: list[Candidate]
) -> int:
    """Add candidates to tournament up to max population.

    Returns:
        Number of candidates actually added.
    """
    added = 0
    for c in candidates:
        if len(tournament.population) >= tournament.config.max_population:
            break
        tournament.add_candidate(c)
        added += 1
    return added


def transition_to_incubation(tournament: Tournament) -> int:
    """Move SEEDED candidates to INCUBATING, setting start timestamp.

    Returns:
        Number of candidates transitioned.
    """
    now = datetime.now(timezone.utc)
    count = 0
    for c in tournament.population:
        if c.status == CandidateStatus.SEEDED:
            c.status = CandidateStatus.INCUBATING
            c.incubation_start = now
            count += 1
    return count


def evaluate_candidates(tournament: Tournament) -> int:
    """Move INCUBATING candidates to EVALUATING after incubation period.

    Returns:
        Number of candidates moved to evaluation.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=tournament.config.incubation_days)
    count = 0
    for c in tournament.population:
        if (
            c.status == CandidateStatus.INCUBATING
            and c.incubation_start is not None
            and c.incubation_start <= cutoff
        ):
            c.status = CandidateStatus.EVALUATING
            count += 1
    return count


def select_candidates(tournament: Tournament) -> int:
    """Eliminate bottom elimination_pct of EVALUATING candidates by fitness.

    Candidates are ranked by composite fitness score. The bottom
    fraction is marked ELIMINATED.

    Returns:
        Number of candidates eliminated.
    """
    evaluating = [
        c for c in tournament.population
        if c.status == CandidateStatus.EVALUATING
    ]
    if not evaluating:
        return 0

    evaluating.sort(key=lambda c: c.fitness_scores.get("composite", 0.0))
    n_eliminate = max(1, math.floor(len(evaluating) * tournament.config.elimination_pct))

    for c in evaluating[:n_eliminate]:
        c.status = CandidateStatus.ELIMINATED
    return n_eliminate


def reproduce_candidates(tournament: Tournament) -> int:
    """Spawn offspring from top reproduction_pct of EVALUATING candidates.

    Offspring are SEEDED into the next generation. Each parent produces
    one offspring (a placeholder — actual generation method is applied
    by the generators).

    Returns:
        Number of offspring created.
    """
    evaluating = [
        c for c in tournament.population
        if c.status == CandidateStatus.EVALUATING
    ]
    if not evaluating:
        return 0

    evaluating.sort(
        key=lambda c: c.fitness_scores.get("composite", 0.0), reverse=True
    )
    n_parents = max(1, math.ceil(len(evaluating) * tournament.config.reproduction_pct))
    parents = evaluating[:n_parents]

    offspring_count = 0
    for parent in parents:
        if len(tournament.population) >= tournament.config.max_population:
            break
        child = Candidate(
            id=f"{parent.id}-child-{tournament.generation + 1}",
            name=f"{parent.name}-offspring",
            skill_md_path=f"incubator/{parent.name}-offspring.skill.md",
            origin=parent.origin,
            parent_ids=[parent.id],
            hypothesis=f"Offspring of {parent.name}",
            status=CandidateStatus.SEEDED,
            generation=tournament.generation + 1,
            created_at=datetime.now(timezone.utc),
            portfolio=PaperPortfolio(
                initial_capital=tournament.config.initial_capital
            ),
        )
        tournament.add_candidate(child)
        offspring_count += 1
    return offspring_count


def check_promotions(tournament: Tournament) -> int:
    """Promote EVALUATING candidates that meet Sharpe + duration thresholds.

    Requirements:
        - Incubation duration >= promotion_min_days
        - Composite fitness (Sharpe proxy) >= promotion_min_sharpe

    Returns:
        Number of candidates promoted.
    """
    now = datetime.now(timezone.utc)
    count = 0
    for c in tournament.population:
        if c.status != CandidateStatus.EVALUATING:
            continue
        if c.incubation_start is None:
            continue

        days_incubated = (now - c.incubation_start).days
        composite = c.fitness_scores.get("composite", 0.0)

        if (
            days_incubated >= tournament.config.promotion_min_days
            and composite >= tournament.config.promotion_min_sharpe
        ):
            c.status = CandidateStatus.PROMOTED
            count += 1
    return count


@dataclass
class CycleResult:
    """Result of a full tournament cycle."""
    seeded: int
    incubated: int
    evaluated: int
    eliminated: int
    reproduced: int
    promoted: int


class TournamentPhaseManager:
    """Orchestrates a full tournament cycle through all phases."""

    def __init__(self, config: TournamentConfig):
        self._tournament = Tournament(config=config)

    @property
    def tournament(self) -> Tournament:
        return self._tournament

    def run_cycle(self, new_candidates: list[Candidate] | None = None) -> dict[str, int]:
        """Run one full tournament cycle.

        Returns:
            Dict with counts for each phase.
        """
        seeded = seed_candidates(
            self._tournament, new_candidates or []
        )
        incubated = transition_to_incubation(self._tournament)
        evaluated = evaluate_candidates(self._tournament)
        eliminated = select_candidates(self._tournament)
        reproduced = reproduce_candidates(self._tournament)
        promoted = check_promotions(self._tournament)

        self._tournament.generation += 1

        return {
            "seeded": seeded,
            "incubated": incubated,
            "evaluated": evaluated,
            "eliminated": eliminated,
            "reproduced": reproduced,
            "promoted": promoted,
        }
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_tournament_phases.py -v
```

Expected: PASS — all tournament phase tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/incubator/phases.py tests/unit/test_tournament_phases.py
git commit -m "feat: tournament phases — seeding, incubation, evaluation, selection, reproduction, promotion"
```

---

## Task 3: Fitness Function

**Files:**
- Create: `src/evolve_trader/incubator/fitness.py`
- Create: `tests/unit/test_fitness.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_fitness.py
"""Tests for the multi-component fitness function."""
import pytest
import numpy as np
from datetime import datetime, timedelta, timezone

from evolve_trader.incubator.fitness import (
    FitnessConfig,
    FitnessEvaluator,
    compute_sharpe_ratio,
    compute_max_drawdown,
    compute_complementarity,
    compute_regime_specificity,
)
from evolve_trader.incubator.candidate import (
    Candidate,
    CandidateOrigin,
    CandidateStatus,
    PaperPortfolio,
)


def _make_returns(mean: float, std: float, n: int = 252) -> np.ndarray:
    """Generate synthetic daily returns."""
    rng = np.random.default_rng(42)
    return rng.normal(mean, std, n)


def test_fitness_config_weights_sum_to_one():
    """Fitness weights must sum to 1.0."""
    config = FitnessConfig()
    total = (
        config.weight_risk_adjusted
        + config.weight_drawdown
        + config.weight_complementarity
        + config.weight_regime_specificity
    )
    assert abs(total - 1.0) < 1e-9


def test_fitness_config_default_weights():
    """Default weights match spec: 40/25/20/15."""
    config = FitnessConfig()
    assert config.weight_risk_adjusted == 0.40
    assert config.weight_drawdown == 0.25
    assert config.weight_complementarity == 0.20
    assert config.weight_regime_specificity == 0.15


def test_fitness_config_rejects_bad_weights():
    """FitnessConfig raises if weights don't sum to 1.0."""
    with pytest.raises(ValueError, match="sum to 1.0"):
        FitnessConfig(
            weight_risk_adjusted=0.5,
            weight_drawdown=0.5,
            weight_complementarity=0.5,
            weight_regime_specificity=0.5,
        )


def test_compute_sharpe_ratio_positive():
    """Sharpe ratio is positive for positive-mean returns."""
    returns = _make_returns(mean=0.001, std=0.01)
    sharpe = compute_sharpe_ratio(returns, annualization_factor=252)
    assert sharpe > 0.0


def test_compute_sharpe_ratio_zero_std():
    """Sharpe ratio is 0 when returns have zero variance."""
    returns = np.zeros(100)
    sharpe = compute_sharpe_ratio(returns, annualization_factor=252)
    assert sharpe == 0.0


def test_compute_sharpe_ratio_negative():
    """Sharpe ratio is negative for negative-mean returns."""
    returns = _make_returns(mean=-0.002, std=0.01)
    sharpe = compute_sharpe_ratio(returns, annualization_factor=252)
    assert sharpe < 0.0


def test_compute_max_drawdown():
    """Max drawdown computes peak-to-trough correctly."""
    # Simple sequence: goes up to 110, drops to 90, recovers
    prices = [100, 105, 110, 100, 90, 95, 100]
    dd = compute_max_drawdown(np.array(prices))
    expected = (110 - 90) / 110  # ~0.1818
    assert abs(dd - expected) < 0.01


def test_compute_max_drawdown_no_drawdown():
    """Max drawdown is 0 for monotonically increasing prices."""
    prices = [100, 101, 102, 103, 104]
    dd = compute_max_drawdown(np.array(prices))
    assert dd == 0.0


def test_compute_complementarity_uncorrelated():
    """Complementarity is high when candidate returns are uncorrelated with production."""
    rng = np.random.default_rng(42)
    prod_returns = rng.normal(0, 0.01, 252)
    cand_returns = rng.normal(0, 0.01, 252)  # independent draws
    comp = compute_complementarity(cand_returns, prod_returns)
    assert comp > 0.5  # Should be high for uncorrelated


def test_compute_complementarity_perfectly_correlated():
    """Complementarity is low when candidate mirrors production."""
    returns = _make_returns(mean=0.001, std=0.01)
    comp = compute_complementarity(returns, returns)
    assert comp < 0.2  # Should be low for identical


def test_compute_regime_specificity():
    """Regime specificity is high when strategy fills production gaps."""
    # Production weak in regimes [2, 3], candidate strong there
    production_regime_sharpes = {0: 1.0, 1: 0.8, 2: -0.2, 3: -0.5}
    candidate_regime_sharpes = {0: 0.1, 1: 0.1, 2: 0.9, 3: 0.7}
    spec = compute_regime_specificity(
        candidate_regime_sharpes, production_regime_sharpes
    )
    assert spec > 0.5


def test_compute_regime_specificity_redundant():
    """Regime specificity is low when candidate mirrors production strengths."""
    regime_sharpes = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.5}
    spec = compute_regime_specificity(regime_sharpes, regime_sharpes)
    assert spec < 0.3


def test_fitness_evaluator_composite_score():
    """FitnessEvaluator produces a composite score in [0, 1]."""
    config = FitnessConfig()
    evaluator = FitnessEvaluator(config)

    rng = np.random.default_rng(42)
    candidate_returns = rng.normal(0.001, 0.01, 252)
    production_returns = rng.normal(0.0005, 0.012, 252)
    candidate_prices = np.cumprod(1 + candidate_returns) * 100
    candidate_regime_sharpes = {0: 0.5, 1: 0.3, 2: 0.8, 3: -0.1}
    production_regime_sharpes = {0: 1.0, 1: 0.9, 2: -0.3, 3: -0.5}

    result = evaluator.evaluate(
        candidate_returns=candidate_returns,
        candidate_prices=candidate_prices,
        production_returns=production_returns,
        candidate_regime_sharpes=candidate_regime_sharpes,
        production_regime_sharpes=production_regime_sharpes,
    )

    assert "composite" in result
    assert "sharpe" in result
    assert "drawdown_discipline" in result
    assert "complementarity" in result
    assert "regime_specificity" in result
    assert 0.0 <= result["composite"] <= 1.0


def test_fitness_evaluator_respects_weights():
    """FitnessEvaluator composite changes when weights change."""
    config1 = FitnessConfig(
        weight_risk_adjusted=0.90,
        weight_drawdown=0.05,
        weight_complementarity=0.03,
        weight_regime_specificity=0.02,
    )
    config2 = FitnessConfig(
        weight_risk_adjusted=0.05,
        weight_drawdown=0.05,
        weight_complementarity=0.85,
        weight_regime_specificity=0.05,
    )

    rng = np.random.default_rng(42)
    cand_ret = rng.normal(0.002, 0.01, 252)
    prod_ret = rng.normal(0.0, 0.01, 252)
    cand_prices = np.cumprod(1 + cand_ret) * 100

    eval1 = FitnessEvaluator(config1)
    eval2 = FitnessEvaluator(config2)

    kwargs = dict(
        candidate_returns=cand_ret,
        candidate_prices=cand_prices,
        production_returns=prod_ret,
        candidate_regime_sharpes={0: 0.5},
        production_regime_sharpes={0: -0.2},
    )

    r1 = eval1.evaluate(**kwargs)
    r2 = eval2.evaluate(**kwargs)

    # Different weights should produce different composites
    assert r1["composite"] != r2["composite"]
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_fitness.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.incubator.fitness'`

**Step 3: Implement the fitness function**

```python
# src/evolve_trader/incubator/fitness.py
"""Multi-component fitness function for strategy evaluation.

Components:
  - Risk-Adjusted Returns (40%) — annualized Sharpe ratio
  - Drawdown Discipline (25%)   — inverse of max drawdown
  - Complementarity (20%)       — uncorrelation with production
  - Regime Specificity (15%)    — fills production weak spots
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FitnessConfig:
    """Weights for the composite fitness function.

    Raises:
        ValueError: If weights don't sum to 1.0 (within tolerance).
    """
    weight_risk_adjusted: float = 0.40
    weight_drawdown: float = 0.25
    weight_complementarity: float = 0.20
    weight_regime_specificity: float = 0.15

    def __post_init__(self):
        total = (
            self.weight_risk_adjusted
            + self.weight_drawdown
            + self.weight_complementarity
            + self.weight_regime_specificity
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Fitness weights must sum to 1.0, got {total:.6f}"
            )


def compute_sharpe_ratio(
    returns: np.ndarray, annualization_factor: int = 252
) -> float:
    """Compute annualized Sharpe ratio from daily returns.

    Args:
        returns: Array of daily returns.
        annualization_factor: Trading days per year.

    Returns:
        Annualized Sharpe ratio. Returns 0.0 if std is zero.
    """
    if len(returns) == 0:
        return 0.0
    std = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
    if std == 0.0:
        return 0.0
    mean = np.mean(returns)
    return float(mean / std * np.sqrt(annualization_factor))


def compute_max_drawdown(prices: np.ndarray) -> float:
    """Compute maximum drawdown from a price series.

    Args:
        prices: Array of portfolio values / prices.

    Returns:
        Max drawdown as a positive fraction (0.0 = no drawdown).
    """
    if len(prices) < 2:
        return 0.0
    peak = prices[0]
    max_dd = 0.0
    for price in prices:
        if price > peak:
            peak = price
        dd = (peak - price) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def compute_complementarity(
    candidate_returns: np.ndarray, production_returns: np.ndarray
) -> float:
    """Compute complementarity score based on return correlation.

    Complementarity = (1 - abs(correlation)) / 1, normalized to [0, 1].
    Perfectly uncorrelated strategies score 1.0; perfectly correlated
    (or anti-correlated) score 0.0.

    Args:
        candidate_returns: Candidate daily returns.
        production_returns: Production portfolio daily returns.

    Returns:
        Complementarity score in [0, 1].
    """
    if len(candidate_returns) < 2 or len(production_returns) < 2:
        return 0.5  # neutral default

    min_len = min(len(candidate_returns), len(production_returns))
    corr = np.corrcoef(
        candidate_returns[:min_len], production_returns[:min_len]
    )[0, 1]

    if np.isnan(corr):
        return 0.5
    return float(1.0 - abs(corr))


def compute_regime_specificity(
    candidate_regime_sharpes: dict[int, float],
    production_regime_sharpes: dict[int, float],
) -> float:
    """Compute regime specificity — how well the candidate fills production gaps.

    For each regime where production is weak (Sharpe < 0), the candidate
    gets credit for positive Sharpe. Normalized to [0, 1].

    Args:
        candidate_regime_sharpes: {regime_id: sharpe} for candidate.
        production_regime_sharpes: {regime_id: sharpe} for production.

    Returns:
        Regime specificity score in [0, 1].
    """
    all_regimes = set(candidate_regime_sharpes) | set(production_regime_sharpes)
    if not all_regimes:
        return 0.0

    gap_score = 0.0
    max_possible = 0.0

    for regime in all_regimes:
        prod_sharpe = production_regime_sharpes.get(regime, 0.0)
        cand_sharpe = candidate_regime_sharpes.get(regime, 0.0)

        # Weight by how weak production is in this regime
        weakness = max(0.0, -prod_sharpe + 0.5)  # 0.5 threshold
        max_possible += weakness

        if weakness > 0 and cand_sharpe > 0:
            gap_score += min(cand_sharpe, 1.0) * weakness

    if max_possible == 0.0:
        return 0.0
    return float(min(1.0, gap_score / max_possible))


def _normalize_sharpe(sharpe: float) -> float:
    """Normalize Sharpe to [0, 1] range using sigmoid-like mapping."""
    # Map Sharpe: -2..+3 -> 0..1
    return float(max(0.0, min(1.0, (sharpe + 2.0) / 5.0)))


def _normalize_drawdown(max_dd: float) -> float:
    """Convert max drawdown to a discipline score in [0, 1].

    Lower drawdown = higher discipline score.
    """
    return float(max(0.0, 1.0 - max_dd * 2.0))  # 50% DD -> 0.0


class FitnessEvaluator:
    """Evaluates candidate strategies using the multi-component fitness function."""

    def __init__(self, config: FitnessConfig | None = None):
        self._config = config or FitnessConfig()

    def evaluate(
        self,
        candidate_returns: np.ndarray,
        candidate_prices: np.ndarray,
        production_returns: np.ndarray,
        candidate_regime_sharpes: dict[int, float],
        production_regime_sharpes: dict[int, float],
    ) -> dict[str, float]:
        """Compute all fitness components and composite score.

        Returns:
            Dict with keys: sharpe, drawdown_discipline, complementarity,
            regime_specificity, composite.
        """
        sharpe = compute_sharpe_ratio(candidate_returns)
        max_dd = compute_max_drawdown(candidate_prices)
        comp = compute_complementarity(candidate_returns, production_returns)
        regime = compute_regime_specificity(
            candidate_regime_sharpes, production_regime_sharpes
        )

        # Normalize all components to [0, 1]
        norm_sharpe = _normalize_sharpe(sharpe)
        norm_dd = _normalize_drawdown(max_dd)
        # comp and regime are already in [0, 1]

        composite = (
            self._config.weight_risk_adjusted * norm_sharpe
            + self._config.weight_drawdown * norm_dd
            + self._config.weight_complementarity * comp
            + self._config.weight_regime_specificity * regime
        )

        return {
            "sharpe": sharpe,
            "drawdown_discipline": norm_dd,
            "complementarity": comp,
            "regime_specificity": regime,
            "composite": float(max(0.0, min(1.0, composite))),
        }
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_fitness.py -v
```

Expected: PASS — all fitness function tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/incubator/fitness.py tests/unit/test_fitness.py
git commit -m "feat: multi-component fitness function — Sharpe, drawdown, complementarity, regime specificity"
```

---

## Task 4: Generation Method — Mutation

**Files:**
- Create: `src/evolve_trader/incubator/generators/__init__.py`
- Create: `src/evolve_trader/incubator/generators/base.py`
- Create: `src/evolve_trader/incubator/generators/mutation.py`
- Create: `tests/unit/test_mutation.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_mutation.py
"""Tests for the mutation generation method."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from evolve_trader.incubator.generators.base import GeneratorBase, GenerationResult
from evolve_trader.incubator.generators.mutation import MutationGenerator
from evolve_trader.incubator.candidate import (
    Candidate,
    CandidateOrigin,
    CandidateStatus,
    PaperPortfolio,
)


def _make_candidate(id: str = "parent-1", name: str = "momentum-v3") -> Candidate:
    return Candidate(
        id=id,
        name=name,
        skill_md_path=f"strategies/{name}.skill.md",
        origin=CandidateOrigin.SEED,
        parent_ids=[],
        hypothesis="Original strategy",
        status=CandidateStatus.EVALUATING,
        generation=0,
        created_at=datetime.now(timezone.utc),
        portfolio=PaperPortfolio(initial_capital=100_000.0),
    )


def test_generator_base_is_abstract():
    """GeneratorBase cannot be instantiated directly."""
    with pytest.raises(TypeError):
        GeneratorBase()


def test_mutation_generator_is_generator():
    """MutationGenerator inherits from GeneratorBase."""
    gen = MutationGenerator(llm_client=MagicMock())
    assert isinstance(gen, GeneratorBase)


def test_generation_result_has_required_fields():
    """GenerationResult carries candidate + metadata."""
    result = GenerationResult(
        candidate=_make_candidate(),
        hypothesis="Added RSI filter for volatility gating",
        mutation_description="Changed entry signal from SMA crossover to RSI < 30",
        parent_ids=["parent-1"],
    )
    assert result.hypothesis is not None
    assert len(result.parent_ids) == 1


@pytest.mark.asyncio
async def test_mutation_modifies_one_component():
    """Mutation changes exactly one component of the parent strategy."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = {
        "mutated_component": "entry_signal",
        "original_value": "SMA(20) crossover SMA(50)",
        "new_value": "RSI(14) < 30 AND SMA(20) > SMA(50)",
        "hypothesis": "RSI filter reduces false entries in choppy markets",
        "skill_md_content": "# Mutated Strategy\n...",
    }

    gen = MutationGenerator(llm_client=mock_llm)
    parent = _make_candidate()

    result = await gen.generate(parent=parent, generation=1)

    assert result is not None
    assert result.candidate.origin == CandidateOrigin.MUTATION
    assert result.candidate.parent_ids == ["parent-1"]
    assert result.candidate.generation == 1
    assert result.hypothesis is not None
    assert len(result.mutation_description) > 0


@pytest.mark.asyncio
async def test_mutation_preserves_parent_unchanged():
    """Mutation does not modify the parent candidate."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = {
        "mutated_component": "position_sizing",
        "original_value": "Fixed 5% allocation",
        "new_value": "Kelly criterion with half-Kelly cap",
        "hypothesis": "Dynamic sizing improves risk-adjusted returns",
        "skill_md_content": "# Mutated Strategy\n...",
    }

    gen = MutationGenerator(llm_client=mock_llm)
    parent = _make_candidate()
    original_name = parent.name
    original_hypothesis = parent.hypothesis

    await gen.generate(parent=parent, generation=1)

    assert parent.name == original_name
    assert parent.hypothesis == original_hypothesis


@pytest.mark.asyncio
async def test_mutation_handles_llm_failure():
    """Mutation returns None when LLM call fails."""
    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = RuntimeError("LLM unavailable")

    gen = MutationGenerator(llm_client=mock_llm)
    parent = _make_candidate()

    result = await gen.generate(parent=parent, generation=1)
    assert result is None
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_mutation.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.incubator.generators'`

**Step 3: Implement the mutation generator**

```python
# src/evolve_trader/incubator/generators/__init__.py
"""Strategy generation methods for the incubator."""
```

```python
# src/evolve_trader/incubator/generators/base.py
"""Base class for strategy generation methods."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from evolve_trader.incubator.candidate import Candidate


@dataclass
class GenerationResult:
    """Result of a strategy generation attempt."""
    candidate: Candidate
    hypothesis: str
    mutation_description: str = ""
    parent_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class GeneratorBase(ABC):
    """Abstract base for all strategy generators."""

    @abstractmethod
    async def generate(self, **kwargs) -> GenerationResult | None:
        """Generate a new candidate strategy.

        Returns:
            GenerationResult on success, None on failure.
        """
        ...
```

```python
# src/evolve_trader/incubator/generators/mutation.py
"""Mutation generator — modifies one component of an existing strategy.

The LLM proposes a targeted change to a single component (entry signal,
exit signal, position sizing, risk management, or universe filter) along
with a hypothesis for why the change should improve performance.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from evolve_trader.incubator.candidate import (
    Candidate,
    CandidateOrigin,
    CandidateStatus,
    PaperPortfolio,
)
from evolve_trader.incubator.generators.base import GeneratorBase, GenerationResult

logger = logging.getLogger(__name__)

MUTATION_PROMPT = """You are mutating an existing trading strategy. Given the parent strategy's SKILL.md,
propose a change to exactly ONE component. Components are:
- entry_signal: When to enter a position
- exit_signal: When to exit a position
- position_sizing: How much capital to allocate
- risk_management: Stop-loss, take-profit, hedging rules
- universe_filter: Which instruments to trade

Parent strategy SKILL.md:
{skill_md_content}

Respond with JSON:
{{
    "mutated_component": "<component_name>",
    "original_value": "<current description>",
    "new_value": "<proposed change>",
    "hypothesis": "<why this should improve performance>",
    "skill_md_content": "<full mutated SKILL.md>"
}}
"""


class MutationGenerator(GeneratorBase):
    """Generates new candidates by mutating one component of a parent strategy."""

    def __init__(self, llm_client: Any, initial_capital: float = 100_000.0):
        self._llm = llm_client
        self._initial_capital = initial_capital

    async def generate(
        self,
        parent: Candidate,
        generation: int,
        **kwargs,
    ) -> GenerationResult | None:
        """Mutate a parent strategy via LLM-proposed single-component change.

        Args:
            parent: The parent candidate to mutate.
            generation: Current tournament generation.

        Returns:
            GenerationResult with the mutated candidate, or None on failure.
        """
        try:
            skill_md_content = kwargs.get("skill_md_content", "")
            response = await self._llm.generate(
                prompt=MUTATION_PROMPT.format(skill_md_content=skill_md_content)
            )

            child_id = f"mut-{parent.id}-{uuid.uuid4().hex[:8]}"
            child_name = f"{parent.name}-mut-{response['mutated_component']}"

            child = Candidate(
                id=child_id,
                name=child_name,
                skill_md_path=f"incubator/{child_name}.skill.md",
                origin=CandidateOrigin.MUTATION,
                parent_ids=[parent.id],
                hypothesis=response["hypothesis"],
                status=CandidateStatus.SEEDED,
                generation=generation,
                created_at=datetime.now(timezone.utc),
                portfolio=PaperPortfolio(initial_capital=self._initial_capital),
                metadata={
                    "mutated_component": response["mutated_component"],
                    "original_value": response["original_value"],
                    "new_value": response["new_value"],
                },
            )

            return GenerationResult(
                candidate=child,
                hypothesis=response["hypothesis"],
                mutation_description=(
                    f"Changed {response['mutated_component']}: "
                    f"'{response['original_value']}' -> '{response['new_value']}'"
                ),
                parent_ids=[parent.id],
            )
        except Exception:
            logger.exception("Mutation generation failed for parent %s", parent.id)
            return None
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_mutation.py -v
```

Expected: PASS — all mutation tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/incubator/generators/ tests/unit/test_mutation.py
git commit -m "feat: mutation generator — LLM-driven single-component strategy mutation"
```

---

## Task 5: Generation Method — Crossover

**Files:**
- Create: `src/evolve_trader/incubator/generators/crossover.py`
- Create: `tests/unit/test_crossover.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_crossover.py
"""Tests for the crossover generation method."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from evolve_trader.incubator.generators.crossover import CrossoverGenerator
from evolve_trader.incubator.generators.base import GeneratorBase, GenerationResult
from evolve_trader.incubator.candidate import (
    Candidate,
    CandidateOrigin,
    CandidateStatus,
    PaperPortfolio,
)


def _make_candidate(id: str, name: str) -> Candidate:
    return Candidate(
        id=id,
        name=name,
        skill_md_path=f"strategies/{name}.skill.md",
        origin=CandidateOrigin.SEED,
        parent_ids=[],
        hypothesis="Original strategy",
        status=CandidateStatus.EVALUATING,
        generation=0,
        created_at=datetime.now(timezone.utc),
        portfolio=PaperPortfolio(initial_capital=100_000.0),
    )


def test_crossover_generator_is_generator():
    """CrossoverGenerator inherits from GeneratorBase."""
    gen = CrossoverGenerator(llm_client=MagicMock())
    assert isinstance(gen, GeneratorBase)


@pytest.mark.asyncio
async def test_crossover_combines_two_parents():
    """Crossover combines elements from two parent strategies."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = {
        "combined_components": {
            "entry_signal": "From parent A: RSI divergence",
            "position_sizing": "From parent B: volatility-scaled",
        },
        "coherence_score": 0.85,
        "hypothesis": "RSI divergence with vol-scaling reduces whipsaw risk",
        "skill_md_content": "# Crossover Strategy\n...",
    }

    gen = CrossoverGenerator(llm_client=mock_llm)
    parent_a = _make_candidate("parent-a", "rsi-divergence")
    parent_b = _make_candidate("parent-b", "vol-scaled-momentum")

    result = await gen.generate(
        parents=[parent_a, parent_b], generation=1
    )

    assert result is not None
    assert result.candidate.origin == CandidateOrigin.CROSSOVER
    assert set(result.candidate.parent_ids) == {"parent-a", "parent-b"}
    assert result.candidate.generation == 1


@pytest.mark.asyncio
async def test_crossover_rejects_incoherent_hybrid():
    """Crossover returns None when LLM coherence filter rejects the hybrid."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = {
        "combined_components": {},
        "coherence_score": 0.15,
        "rejection_reason": "Mean-reversion entry contradicts momentum exit",
        "hypothesis": None,
        "skill_md_content": None,
    }

    gen = CrossoverGenerator(llm_client=mock_llm, coherence_threshold=0.5)
    parent_a = _make_candidate("parent-a", "mean-reversion")
    parent_b = _make_candidate("parent-b", "momentum-trend")

    result = await gen.generate(
        parents=[parent_a, parent_b], generation=1
    )

    assert result is None


@pytest.mark.asyncio
async def test_crossover_requires_at_least_two_parents():
    """Crossover raises ValueError with fewer than 2 parents."""
    gen = CrossoverGenerator(llm_client=MagicMock())

    with pytest.raises(ValueError, match="at least 2"):
        await gen.generate(parents=[_make_candidate("solo", "solo")], generation=1)


@pytest.mark.asyncio
async def test_crossover_handles_llm_failure():
    """Crossover returns None on LLM failure."""
    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = RuntimeError("LLM timeout")

    gen = CrossoverGenerator(llm_client=mock_llm)
    parents = [
        _make_candidate("p1", "strat-1"),
        _make_candidate("p2", "strat-2"),
    ]

    result = await gen.generate(parents=parents, generation=1)
    assert result is None


@pytest.mark.asyncio
async def test_crossover_supports_three_parents():
    """Crossover can combine elements from 3+ parents."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = {
        "combined_components": {
            "entry_signal": "From parent A",
            "exit_signal": "From parent B",
            "risk_management": "From parent C",
        },
        "coherence_score": 0.72,
        "hypothesis": "Triple combination for diversified signals",
        "skill_md_content": "# Triple Crossover\n...",
    }

    gen = CrossoverGenerator(llm_client=mock_llm)
    parents = [
        _make_candidate("p1", "strat-1"),
        _make_candidate("p2", "strat-2"),
        _make_candidate("p3", "strat-3"),
    ]

    result = await gen.generate(parents=parents, generation=1)

    assert result is not None
    assert len(result.candidate.parent_ids) == 3
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_crossover.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.incubator.generators.crossover'`

**Step 3: Implement the crossover generator**

```python
# src/evolve_trader/incubator/generators/crossover.py
"""Crossover generator — combines elements from 2+ parent strategies.

The LLM selects the best component from each parent and checks that
the combination is coherent (e.g., no contradictory entry/exit logic).
Hybrids below the coherence threshold are rejected.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from evolve_trader.incubator.candidate import (
    Candidate,
    CandidateOrigin,
    CandidateStatus,
    PaperPortfolio,
)
from evolve_trader.incubator.generators.base import GeneratorBase, GenerationResult

logger = logging.getLogger(__name__)

CROSSOVER_PROMPT = """You are combining elements from multiple trading strategies into one hybrid.
For each parent strategy, select the best component (entry_signal, exit_signal,
position_sizing, risk_management, universe_filter).

Verify that the combination is coherent — contradictory components should be flagged.

Parent strategies:
{parents_description}

Respond with JSON:
{{
    "combined_components": {{"<component>": "From parent X: <description>", ...}},
    "coherence_score": <0.0 to 1.0>,
    "rejection_reason": "<if coherence < threshold, explain why>",
    "hypothesis": "<why this combination should work>",
    "skill_md_content": "<full hybrid SKILL.md or null if rejected>"
}}
"""


class CrossoverGenerator(GeneratorBase):
    """Generates new candidates by combining elements from multiple parents."""

    def __init__(
        self,
        llm_client: Any,
        coherence_threshold: float = 0.5,
        initial_capital: float = 100_000.0,
    ):
        self._llm = llm_client
        self._coherence_threshold = coherence_threshold
        self._initial_capital = initial_capital

    async def generate(
        self,
        parents: list[Candidate] | None = None,
        generation: int = 0,
        **kwargs,
    ) -> GenerationResult | None:
        """Combine elements from parents via LLM with coherence filtering.

        Args:
            parents: List of 2+ parent candidates.
            generation: Current tournament generation.

        Returns:
            GenerationResult with the hybrid, or None if rejected/failed.

        Raises:
            ValueError: If fewer than 2 parents provided.
        """
        if parents is None or len(parents) < 2:
            raise ValueError("Crossover requires at least 2 parents")

        try:
            parents_desc = "\n".join(
                f"Parent {chr(65 + i)} ({p.name}): {p.skill_md_path}"
                for i, p in enumerate(parents)
            )
            response = await self._llm.generate(
                prompt=CROSSOVER_PROMPT.format(parents_description=parents_desc)
            )

            coherence = response.get("coherence_score", 0.0)
            if coherence < self._coherence_threshold:
                logger.info(
                    "Crossover rejected (coherence %.2f < %.2f): %s",
                    coherence,
                    self._coherence_threshold,
                    response.get("rejection_reason", "unknown"),
                )
                return None

            parent_ids = [p.id for p in parents]
            parent_names = "-x-".join(p.name[:10] for p in parents)
            child_id = f"cross-{uuid.uuid4().hex[:8]}"
            child_name = f"crossover-{parent_names}"

            child = Candidate(
                id=child_id,
                name=child_name,
                skill_md_path=f"incubator/{child_name}.skill.md",
                origin=CandidateOrigin.CROSSOVER,
                parent_ids=parent_ids,
                hypothesis=response.get("hypothesis", ""),
                status=CandidateStatus.SEEDED,
                generation=generation,
                created_at=datetime.now(timezone.utc),
                portfolio=PaperPortfolio(initial_capital=self._initial_capital),
                metadata={
                    "combined_components": response.get("combined_components", {}),
                    "coherence_score": coherence,
                },
            )

            return GenerationResult(
                candidate=child,
                hypothesis=response.get("hypothesis", ""),
                mutation_description=f"Crossover of {len(parents)} parents",
                parent_ids=parent_ids,
                metadata={"coherence_score": coherence},
            )
        except ValueError:
            raise
        except Exception:
            logger.exception("Crossover generation failed")
            return None
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_crossover.py -v
```

Expected: PASS — all crossover tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/incubator/generators/crossover.py tests/unit/test_crossover.py
git commit -m "feat: crossover generator — LLM-driven multi-parent strategy combination with coherence filter"
```

---

## Task 6: Generation Method — Regime-Conditioned Search

**Files:**
- Create: `src/evolve_trader/incubator/generators/regime_search.py`
- Create: `tests/unit/test_regime_search.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_regime_search.py
"""Tests for the regime-conditioned search generation method."""
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from evolve_trader.incubator.generators.regime_search import (
    RegimeSearchGenerator,
    RegimeGap,
)
from evolve_trader.incubator.generators.base import GeneratorBase, GenerationResult
from evolve_trader.incubator.candidate import (
    CandidateOrigin,
    CandidateStatus,
)


def test_regime_search_is_generator():
    """RegimeSearchGenerator inherits from GeneratorBase."""
    gen = RegimeSearchGenerator(llm_client=MagicMock(), replay_engine=MagicMock())
    assert isinstance(gen, GeneratorBase)


def test_regime_gap_identifies_weak_regimes():
    """RegimeGap correctly identifies regimes where production is weak."""
    production_regime_sharpes = {
        0: 1.2,   # strong
        1: 0.3,   # moderate
        2: -0.5,  # weak
        3: -1.1,  # very weak
    }
    gaps = RegimeGap.find_gaps(production_regime_sharpes, threshold=0.5)

    assert len(gaps) == 2
    regime_ids = [g.regime_id for g in gaps]
    assert 2 in regime_ids
    assert 3 in regime_ids
    # Gaps should be sorted by severity (worst first)
    assert gaps[0].regime_id == 3
    assert gaps[0].production_sharpe == -1.1


def test_regime_gap_no_gaps_when_all_strong():
    """No gaps returned when production performs well in all regimes."""
    sharpes = {0: 1.0, 1: 0.8, 2: 1.5}
    gaps = RegimeGap.find_gaps(sharpes, threshold=0.5)
    assert len(gaps) == 0


@pytest.mark.asyncio
async def test_regime_search_targets_weakest_regime():
    """Regime search generates a strategy for the weakest production regime."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = {
        "target_regime": 3,
        "regime_characteristics": "High volatility, declining trend",
        "proposed_approach": "Short volatility with put spreads",
        "hypothesis": "Put spreads profit from vol mean-reversion in regime 3",
        "skill_md_content": "# Regime 3 Specialist\n...",
    }

    mock_replay = MagicMock()
    mock_replay.get_regime_data.return_value = {
        "regime_id": 3,
        "historical_periods": [
            {"start": "2024-01-15", "end": "2024-03-01"},
        ],
        "avg_vix": 32.5,
    }

    gen = RegimeSearchGenerator(llm_client=mock_llm, replay_engine=mock_replay)

    production_regime_sharpes = {0: 1.0, 1: 0.5, 2: -0.3, 3: -1.1}
    result = await gen.generate(
        production_regime_sharpes=production_regime_sharpes,
        generation=2,
    )

    assert result is not None
    assert result.candidate.origin == CandidateOrigin.REGIME_SEARCH
    assert result.candidate.generation == 2
    assert result.candidate.metadata.get("target_regime") == 3


@pytest.mark.asyncio
async def test_regime_search_returns_none_when_no_gaps():
    """Regime search returns None when production has no weak regimes."""
    gen = RegimeSearchGenerator(llm_client=MagicMock(), replay_engine=MagicMock())

    production_regime_sharpes = {0: 1.0, 1: 0.8, 2: 1.2}
    result = await gen.generate(
        production_regime_sharpes=production_regime_sharpes,
        generation=1,
    )

    assert result is None


@pytest.mark.asyncio
async def test_regime_search_handles_llm_failure():
    """Regime search returns None on LLM failure."""
    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = RuntimeError("LLM error")

    mock_replay = MagicMock()
    mock_replay.get_regime_data.return_value = {"regime_id": 2}

    gen = RegimeSearchGenerator(llm_client=mock_llm, replay_engine=mock_replay)

    result = await gen.generate(
        production_regime_sharpes={0: 1.0, 2: -0.5},
        generation=1,
    )

    assert result is None
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_regime_search.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.incubator.generators.regime_search'`

**Step 3: Implement the regime-conditioned search generator**

```python
# src/evolve_trader/incubator/generators/regime_search.py
"""Regime-conditioned search generator.

Identifies regimes where production is weakest, retrieves historical
replay data for those regimes, and uses the LLM to propose strategies
specifically designed for those market conditions.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from evolve_trader.incubator.candidate import (
    Candidate,
    CandidateOrigin,
    CandidateStatus,
    PaperPortfolio,
)
from evolve_trader.incubator.generators.base import GeneratorBase, GenerationResult

logger = logging.getLogger(__name__)

REGIME_SEARCH_PROMPT = """You are designing a trading strategy for a specific market regime
where the production portfolio performs poorly.

Target regime: {regime_id}
Regime characteristics: {regime_data}
Production Sharpe in this regime: {production_sharpe}

Historical periods when this regime occurred:
{historical_periods}

Design a strategy that specifically profits in this regime. Respond with JSON:
{{
    "target_regime": {regime_id},
    "regime_characteristics": "<description of regime>",
    "proposed_approach": "<strategy approach>",
    "hypothesis": "<why this should work in this regime>",
    "skill_md_content": "<full SKILL.md for the regime specialist>"
}}
"""


@dataclass
class RegimeGap:
    """Represents a regime where production underperforms."""
    regime_id: int
    production_sharpe: float
    severity: float  # how far below threshold

    @staticmethod
    def find_gaps(
        production_regime_sharpes: dict[int, float],
        threshold: float = 0.5,
    ) -> list[RegimeGap]:
        """Find regimes where production Sharpe is below threshold.

        Returns:
            List of RegimeGaps sorted by severity (worst first).
        """
        gaps = []
        for regime_id, sharpe in production_regime_sharpes.items():
            if sharpe < threshold:
                gaps.append(RegimeGap(
                    regime_id=regime_id,
                    production_sharpe=sharpe,
                    severity=threshold - sharpe,
                ))
        gaps.sort(key=lambda g: g.severity, reverse=True)
        return gaps


class RegimeSearchGenerator(GeneratorBase):
    """Generates strategies targeting specific weak regimes."""

    def __init__(
        self,
        llm_client: Any,
        replay_engine: Any,
        initial_capital: float = 100_000.0,
        gap_threshold: float = 0.5,
    ):
        self._llm = llm_client
        self._replay = replay_engine
        self._initial_capital = initial_capital
        self._gap_threshold = gap_threshold

    async def generate(
        self,
        production_regime_sharpes: dict[int, float] | None = None,
        generation: int = 0,
        **kwargs,
    ) -> GenerationResult | None:
        """Search for a strategy targeting the weakest production regime.

        Args:
            production_regime_sharpes: {regime_id: sharpe} for production.
            generation: Current tournament generation.

        Returns:
            GenerationResult for a regime-specialist candidate, or None.
        """
        if production_regime_sharpes is None:
            return None

        gaps = RegimeGap.find_gaps(
            production_regime_sharpes, threshold=self._gap_threshold
        )
        if not gaps:
            logger.info("No regime gaps found — production covers all regimes")
            return None

        target = gaps[0]  # worst gap

        try:
            regime_data = self._replay.get_regime_data(target.regime_id)

            response = await self._llm.generate(
                prompt=REGIME_SEARCH_PROMPT.format(
                    regime_id=target.regime_id,
                    regime_data=regime_data,
                    production_sharpe=target.production_sharpe,
                    historical_periods=regime_data.get("historical_periods", []),
                )
            )

            child_id = f"regime-{target.regime_id}-{uuid.uuid4().hex[:8]}"
            child_name = f"regime-{target.regime_id}-specialist"

            child = Candidate(
                id=child_id,
                name=child_name,
                skill_md_path=f"incubator/{child_name}.skill.md",
                origin=CandidateOrigin.REGIME_SEARCH,
                parent_ids=[],
                hypothesis=response.get("hypothesis", ""),
                status=CandidateStatus.SEEDED,
                generation=generation,
                created_at=datetime.now(timezone.utc),
                portfolio=PaperPortfolio(initial_capital=self._initial_capital),
                metadata={
                    "target_regime": target.regime_id,
                    "production_sharpe": target.production_sharpe,
                    "regime_characteristics": response.get(
                        "regime_characteristics", ""
                    ),
                },
            )

            return GenerationResult(
                candidate=child,
                hypothesis=response.get("hypothesis", ""),
                mutation_description=(
                    f"Regime-conditioned search targeting regime {target.regime_id} "
                    f"(production Sharpe: {target.production_sharpe:.2f})"
                ),
                parent_ids=[],
                metadata={"target_regime": target.regime_id},
            )
        except Exception:
            logger.exception(
                "Regime search failed for regime %d", target.regime_id
            )
            return None
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_regime_search.py -v
```

Expected: PASS — all regime search tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/incubator/generators/regime_search.py tests/unit/test_regime_search.py
git commit -m "feat: regime-conditioned search — targets production weak spots with specialist strategies"
```

---

## Task 7: Fossil Record

**Files:**
- Create: `src/evolve_trader/incubator/fossil_record.py`
- Create: `tests/unit/test_fossil_record.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_fossil_record.py
"""Tests for the fossil record — archive of eliminated strategies."""
import pytest
from datetime import datetime, timedelta, timezone

from evolve_trader.incubator.fossil_record import (
    FossilRecord,
    Fossil,
    FailureCondition,
)
from evolve_trader.incubator.candidate import (
    Candidate,
    CandidateOrigin,
    CandidateStatus,
    PaperPortfolio,
)


def _make_candidate(id: str, sharpe: float = -0.5, regime: int = 2) -> Candidate:
    c = Candidate(
        id=id,
        name=f"strategy-{id}",
        skill_md_path=f"incubator/strategy-{id}.skill.md",
        origin=CandidateOrigin.MUTATION,
        parent_ids=["parent-1"],
        hypothesis="Test strategy",
        status=CandidateStatus.ELIMINATED,
        generation=1,
        created_at=datetime.now(timezone.utc),
        portfolio=PaperPortfolio(initial_capital=100_000.0),
    )
    c.fitness_scores = {"sharpe": sharpe, "composite": max(0, sharpe)}
    c.metadata["dominant_regime"] = regime
    return c


def test_fossil_record_archives_candidate():
    """Fossil record stores an eliminated candidate with failure conditions."""
    record = FossilRecord()
    candidate = _make_candidate("fossil-1")

    failure = FailureCondition(
        reason="Sharpe below elimination threshold",
        regime_at_failure=2,
        market_conditions={"vix": 28.0, "trend": "bearish"},
        fitness_at_failure={"sharpe": -0.5, "composite": 0.0},
    )

    record.archive(candidate, failure)
    assert len(record) == 1


def test_fossil_record_retrieves_by_id():
    """Fossil record supports lookup by candidate ID."""
    record = FossilRecord()
    candidate = _make_candidate("fossil-2")
    failure = FailureCondition(
        reason="Max drawdown exceeded",
        regime_at_failure=3,
        market_conditions={"vix": 35.0},
        fitness_at_failure={"sharpe": -0.3, "composite": 0.05},
    )

    record.archive(candidate, failure)
    fossil = record.get("fossil-2")

    assert fossil is not None
    assert fossil.candidate_id == "fossil-2"
    assert fossil.failure.reason == "Max drawdown exceeded"


def test_fossil_record_returns_none_for_missing():
    """Fossil record returns None for unknown IDs."""
    record = FossilRecord()
    assert record.get("nonexistent") is None


def test_fossil_record_search_by_regime():
    """Fossil record supports searching for strategies that failed in a specific regime."""
    record = FossilRecord()

    for i, regime in enumerate([1, 2, 2, 3, 2]):
        candidate = _make_candidate(f"fossil-{i}", regime=regime)
        failure = FailureCondition(
            reason="Low fitness",
            regime_at_failure=regime,
            market_conditions={},
            fitness_at_failure={},
        )
        record.archive(candidate, failure)

    regime_2_fossils = record.search_by_regime(2)
    assert len(regime_2_fossils) == 3


def test_fossil_record_search_by_origin():
    """Fossil record supports filtering by generation method."""
    record = FossilRecord()

    origins = [
        CandidateOrigin.MUTATION,
        CandidateOrigin.CROSSOVER,
        CandidateOrigin.MUTATION,
        CandidateOrigin.REGIME_SEARCH,
    ]
    for i, origin in enumerate(origins):
        candidate = _make_candidate(f"fossil-{i}")
        candidate.origin = origin
        failure = FailureCondition(
            reason="Test", regime_at_failure=0,
            market_conditions={}, fitness_at_failure={},
        )
        record.archive(candidate, failure)

    mutations = record.search_by_origin(CandidateOrigin.MUTATION)
    assert len(mutations) == 2


def test_fossil_record_resurrection_candidates():
    """Fossil record identifies candidates for resurrection when conditions change."""
    record = FossilRecord()

    # Strategy that failed in regime 3 with high VIX
    candidate = _make_candidate("resurrect-1", sharpe=-0.3, regime=3)
    failure = FailureCondition(
        reason="Sharpe below threshold",
        regime_at_failure=3,
        market_conditions={"vix": 35.0, "trend": "bearish"},
        fitness_at_failure={"sharpe": -0.3, "composite": 0.1},
    )
    record.archive(candidate, failure)

    # Current conditions are different: regime shifted
    resurrection_candidates = record.find_resurrection_candidates(
        current_regime=1,
        current_conditions={"vix": 15.0, "trend": "bullish"},
    )

    # Strategy failed in regime 3 but we're now in regime 1 — it's a candidate
    assert len(resurrection_candidates) >= 1
    assert resurrection_candidates[0].candidate_id == "resurrect-1"


def test_fossil_record_no_resurrection_same_conditions():
    """No resurrection when current conditions match failure conditions."""
    record = FossilRecord()

    candidate = _make_candidate("no-res-1", regime=2)
    failure = FailureCondition(
        reason="Drawdown exceeded",
        regime_at_failure=2,
        market_conditions={"vix": 25.0},
        fitness_at_failure={"sharpe": -0.8},
    )
    record.archive(candidate, failure)

    # Same regime — don't resurrect
    candidates = record.find_resurrection_candidates(
        current_regime=2,
        current_conditions={"vix": 25.0},
    )
    assert len(candidates) == 0


def test_fossil_has_timestamp():
    """Fossil records when the candidate was archived."""
    record = FossilRecord()
    candidate = _make_candidate("ts-1")
    failure = FailureCondition(
        reason="Test", regime_at_failure=0,
        market_conditions={}, fitness_at_failure={},
    )

    record.archive(candidate, failure)
    fossil = record.get("ts-1")

    assert fossil is not None
    assert fossil.archived_at is not None
    assert isinstance(fossil.archived_at, datetime)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_fossil_record.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.incubator.fossil_record'`

**Step 3: Implement the fossil record**

```python
# src/evolve_trader/incubator/fossil_record.py
"""Fossil record — archive of eliminated strategies with failure conditions.

Maintains a searchable archive so the orchestrator can resurrect
strategies when market conditions change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from evolve_trader.incubator.candidate import Candidate, CandidateOrigin


@dataclass(frozen=True)
class FailureCondition:
    """Records why and when a candidate was eliminated."""
    reason: str
    regime_at_failure: int
    market_conditions: dict[str, Any]
    fitness_at_failure: dict[str, Any]


@dataclass
class Fossil:
    """An archived candidate with its failure context."""
    candidate_id: str
    candidate_name: str
    origin: CandidateOrigin
    parent_ids: list[str]
    hypothesis: str
    skill_md_path: str
    generation: int
    fitness_scores: dict[str, float]
    metadata: dict[str, Any]
    failure: FailureCondition
    archived_at: datetime


class FossilRecord:
    """Searchable archive of eliminated strategies.

    Supports:
    - Archive with failure conditions
    - Lookup by ID
    - Search by regime, origin
    - Resurrection candidate identification
    """

    def __init__(self):
        self._fossils: dict[str, Fossil] = {}

    def __len__(self) -> int:
        return len(self._fossils)

    def archive(self, candidate: Candidate, failure: FailureCondition) -> Fossil:
        """Archive an eliminated candidate with its failure context.

        Args:
            candidate: The eliminated candidate.
            failure: Why and when the candidate failed.

        Returns:
            The created Fossil record.
        """
        fossil = Fossil(
            candidate_id=candidate.id,
            candidate_name=candidate.name,
            origin=candidate.origin,
            parent_ids=list(candidate.parent_ids),
            hypothesis=candidate.hypothesis,
            skill_md_path=candidate.skill_md_path,
            generation=candidate.generation,
            fitness_scores=dict(candidate.fitness_scores),
            metadata=dict(candidate.metadata),
            failure=failure,
            archived_at=datetime.now(timezone.utc),
        )
        self._fossils[candidate.id] = fossil
        return fossil

    def get(self, candidate_id: str) -> Fossil | None:
        """Retrieve a fossil by candidate ID."""
        return self._fossils.get(candidate_id)

    def search_by_regime(self, regime_id: int) -> list[Fossil]:
        """Find all fossils that failed in a specific regime.

        Args:
            regime_id: The regime to search for.

        Returns:
            List of fossils that failed in the given regime.
        """
        return [
            f for f in self._fossils.values()
            if f.failure.regime_at_failure == regime_id
        ]

    def search_by_origin(self, origin: CandidateOrigin) -> list[Fossil]:
        """Find all fossils generated by a specific method.

        Args:
            origin: The generation method to filter by.

        Returns:
            List of fossils with the given origin.
        """
        return [
            f for f in self._fossils.values()
            if f.origin == origin
        ]

    def find_resurrection_candidates(
        self,
        current_regime: int,
        current_conditions: dict[str, Any],
    ) -> list[Fossil]:
        """Identify fossils that might perform well under changed conditions.

        A fossil is a resurrection candidate if:
        - It failed in a DIFFERENT regime than the current one
        - Current conditions differ meaningfully from failure conditions

        Args:
            current_regime: The current market regime ID.
            current_conditions: Current market condition snapshot.

        Returns:
            List of fossils suitable for resurrection, sorted by
            most recently archived first.
        """
        candidates = []
        for fossil in self._fossils.values():
            if fossil.failure.regime_at_failure == current_regime:
                continue  # Same regime — skip
            candidates.append(fossil)

        # Sort by most recently archived (newest first)
        candidates.sort(key=lambda f: f.archived_at, reverse=True)
        return candidates

    def get_all(self) -> list[Fossil]:
        """Return all fossils in the archive."""
        return list(self._fossils.values())
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_fossil_record.py -v
```

Expected: PASS — all fossil record tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/incubator/fossil_record.py tests/unit/test_fossil_record.py
git commit -m "feat: fossil record — archive eliminated strategies with failure conditions for resurrection"
```

---

## Task 8: Advanced Generation Methods

**Files:**
- Create: `src/evolve_trader/incubator/generators/academic.py`
- Create: `src/evolve_trader/incubator/generators/anomaly.py`
- Create: `src/evolve_trader/incubator/generators/counter_strategy.py`
- Create: `src/evolve_trader/incubator/generators/meta_strategy.py`
- Create: `tests/unit/test_academic.py`
- Create: `tests/unit/test_anomaly.py`
- Create: `tests/unit/test_counter_strategy.py`
- Create: `tests/unit/test_meta_strategy.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_academic.py
"""Tests for the academic paper mining generation method."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from evolve_trader.incubator.generators.academic import AcademicMiningGenerator
from evolve_trader.incubator.generators.base import GeneratorBase, GenerationResult
from evolve_trader.incubator.candidate import CandidateOrigin, CandidateStatus


def test_academic_generator_is_generator():
    """AcademicMiningGenerator inherits from GeneratorBase."""
    gen = AcademicMiningGenerator(llm_client=MagicMock())
    assert isinstance(gen, GeneratorBase)


@pytest.mark.asyncio
async def test_academic_mines_paper_and_generates_strategy():
    """Academic generator translates a paper abstract into a trading strategy."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = {
        "paper_title": "Momentum Crashes and Recovery",
        "key_insight": "Momentum strategies crash after market reversals but recover within 6 months",
        "proposed_strategy": "Conditional momentum with crash detection gate",
        "hypothesis": "Gating momentum on crash indicators avoids 80% of drawdown",
        "skill_md_content": "# Academic: Conditional Momentum\n...",
    }

    gen = AcademicMiningGenerator(llm_client=mock_llm)
    result = await gen.generate(
        paper_abstract="We study momentum strategy crashes...",
        generation=3,
    )

    assert result is not None
    assert result.candidate.origin == CandidateOrigin.ACADEMIC
    assert "momentum" in result.hypothesis.lower() or len(result.hypothesis) > 0


@pytest.mark.asyncio
async def test_academic_handles_irrelevant_paper():
    """Academic generator returns None for papers without actionable insight."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = {
        "actionable": False,
        "rejection_reason": "Paper is theoretical with no tradeable signal",
    }

    gen = AcademicMiningGenerator(llm_client=mock_llm)
    result = await gen.generate(
        paper_abstract="We prove the existence of...",
        generation=1,
    )

    assert result is None


@pytest.mark.asyncio
async def test_academic_handles_llm_failure():
    """Academic generator returns None on LLM failure."""
    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = RuntimeError("API error")

    gen = AcademicMiningGenerator(llm_client=mock_llm)
    result = await gen.generate(paper_abstract="...", generation=1)

    assert result is None
```

```python
# tests/unit/test_anomaly.py
"""Tests for the anomaly detection generation method."""
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from evolve_trader.incubator.generators.anomaly import (
    AnomalyDetectionGenerator,
    MarketAnomaly,
)
from evolve_trader.incubator.generators.base import GeneratorBase
from evolve_trader.incubator.candidate import CandidateOrigin


def test_anomaly_generator_is_generator():
    """AnomalyDetectionGenerator inherits from GeneratorBase."""
    gen = AnomalyDetectionGenerator(llm_client=MagicMock())
    assert isinstance(gen, GeneratorBase)


def test_market_anomaly_detection():
    """MarketAnomaly detects statistical anomalies in return data."""
    rng = np.random.default_rng(42)
    normal_returns = rng.normal(0, 0.01, 250)
    # Inject anomaly: 5 consecutive days of >3 std returns
    anomalous_returns = np.concatenate([
        normal_returns,
        np.array([0.05, 0.06, -0.07, 0.08, -0.05]),
    ])

    anomalies = MarketAnomaly.detect(anomalous_returns, z_threshold=3.0)
    assert len(anomalies) >= 1
    assert all(a.z_score >= 3.0 or a.z_score <= -3.0 for a in anomalies)


def test_market_anomaly_none_in_normal_data():
    """No anomalies detected in normally distributed data (high threshold)."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 100)
    anomalies = MarketAnomaly.detect(returns, z_threshold=10.0)
    assert len(anomalies) == 0


@pytest.mark.asyncio
async def test_anomaly_generates_strategy_for_detected_pattern():
    """Anomaly generator creates a strategy targeting a detected anomaly."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = {
        "anomaly_type": "volatility_clustering",
        "exploitation_approach": "Mean-reversion after extreme vol spike",
        "hypothesis": "Post-spike vol reverts within 5 days 78% of the time",
        "skill_md_content": "# Anomaly: Vol Spike Reversion\n...",
    }

    gen = AnomalyDetectionGenerator(llm_client=mock_llm)

    rng = np.random.default_rng(42)
    returns = np.concatenate([
        rng.normal(0, 0.01, 250),
        np.array([0.05, 0.06, -0.07]),
    ])

    result = await gen.generate(market_returns=returns, generation=2)

    assert result is not None
    assert result.candidate.origin == CandidateOrigin.ANOMALY


@pytest.mark.asyncio
async def test_anomaly_returns_none_when_no_anomalies():
    """Anomaly generator returns None when no anomalies detected."""
    gen = AnomalyDetectionGenerator(llm_client=MagicMock())

    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.001, 100)  # very low vol, no outliers

    result = await gen.generate(market_returns=returns, generation=1, z_threshold=10.0)

    assert result is None
```

```python
# tests/unit/test_counter_strategy.py
"""Tests for the counter-strategy generation method."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from evolve_trader.incubator.generators.counter_strategy import CounterStrategyGenerator
from evolve_trader.incubator.generators.base import GeneratorBase
from evolve_trader.incubator.candidate import (
    Candidate,
    CandidateOrigin,
    CandidateStatus,
    PaperPortfolio,
)


def _make_candidate(id: str, name: str) -> Candidate:
    return Candidate(
        id=id,
        name=name,
        skill_md_path=f"strategies/{name}.skill.md",
        origin=CandidateOrigin.SEED,
        parent_ids=[],
        hypothesis="Original",
        status=CandidateStatus.EVALUATING,
        generation=0,
        created_at=datetime.now(timezone.utc),
        portfolio=PaperPortfolio(initial_capital=100_000.0),
    )


def test_counter_strategy_is_generator():
    """CounterStrategyGenerator inherits from GeneratorBase."""
    gen = CounterStrategyGenerator(llm_client=MagicMock())
    assert isinstance(gen, GeneratorBase)


@pytest.mark.asyncio
async def test_counter_strategy_inverts_logic():
    """Counter-strategy generates the inverse of a target strategy."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = {
        "target_weakness": "Momentum strategy loses during mean-reversion regimes",
        "counter_approach": "Fade momentum signals when RSI is overbought",
        "hypothesis": "Counter-momentum captures reversals the target misses",
        "skill_md_content": "# Counter: Anti-Momentum\n...",
    }

    gen = CounterStrategyGenerator(llm_client=mock_llm)
    target = _make_candidate("target-1", "momentum-v3")

    result = await gen.generate(target_strategy=target, generation=2)

    assert result is not None
    assert result.candidate.origin == CandidateOrigin.COUNTER_STRATEGY
    assert "target-1" in result.candidate.parent_ids


@pytest.mark.asyncio
async def test_counter_strategy_handles_llm_failure():
    """Counter-strategy returns None on LLM failure."""
    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = RuntimeError("LLM error")

    gen = CounterStrategyGenerator(llm_client=mock_llm)
    target = _make_candidate("target-1", "momentum-v3")

    result = await gen.generate(target_strategy=target, generation=1)
    assert result is None
```

```python
# tests/unit/test_meta_strategy.py
"""Tests for the meta-strategy discovery generation method."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from evolve_trader.incubator.generators.meta_strategy import MetaStrategyGenerator
from evolve_trader.incubator.generators.base import GeneratorBase
from evolve_trader.incubator.candidate import (
    Candidate,
    CandidateOrigin,
    CandidateStatus,
    PaperPortfolio,
)


def _make_candidate(id: str, name: str, sharpe: float) -> Candidate:
    c = Candidate(
        id=id,
        name=name,
        skill_md_path=f"strategies/{name}.skill.md",
        origin=CandidateOrigin.SEED,
        parent_ids=[],
        hypothesis="Original",
        status=CandidateStatus.EVALUATING,
        generation=0,
        created_at=datetime.now(timezone.utc),
        portfolio=PaperPortfolio(initial_capital=100_000.0),
    )
    c.fitness_scores = {"sharpe": sharpe, "composite": sharpe}
    return c


def test_meta_strategy_is_generator():
    """MetaStrategyGenerator inherits from GeneratorBase."""
    gen = MetaStrategyGenerator(llm_client=MagicMock())
    assert isinstance(gen, GeneratorBase)


@pytest.mark.asyncio
async def test_meta_strategy_discovers_pattern_across_strategies():
    """Meta-strategy finds a common pattern across successful strategies."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = {
        "discovered_pattern": "All top strategies use volume confirmation",
        "meta_approach": "Ensemble of volume-confirmed signals across timeframes",
        "hypothesis": "Volume confirmation is the meta-alpha — pure volume strategy should work",
        "skill_md_content": "# Meta: Volume Confirmation Ensemble\n...",
    }

    gen = MetaStrategyGenerator(llm_client=mock_llm)
    population = [
        _make_candidate("s1", "momentum-v3", sharpe=1.2),
        _make_candidate("s2", "mean-rev-v2", sharpe=0.9),
        _make_candidate("s3", "breakout-v1", sharpe=0.7),
        _make_candidate("s4", "poor-strat", sharpe=-0.3),
    ]

    result = await gen.generate(population=population, generation=3)

    assert result is not None
    assert result.candidate.origin == CandidateOrigin.META_STRATEGY
    assert result.hypothesis is not None


@pytest.mark.asyncio
async def test_meta_strategy_requires_minimum_population():
    """Meta-strategy returns None with too few strategies to analyze."""
    gen = MetaStrategyGenerator(llm_client=MagicMock(), min_population=5)
    population = [_make_candidate("s1", "strat-1", sharpe=1.0)]

    result = await gen.generate(population=population, generation=1)
    assert result is None


@pytest.mark.asyncio
async def test_meta_strategy_handles_llm_failure():
    """Meta-strategy returns None on LLM failure."""
    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = RuntimeError("LLM error")

    gen = MetaStrategyGenerator(llm_client=mock_llm)
    population = [
        _make_candidate(f"s{i}", f"strat-{i}", sharpe=0.5 + i * 0.1)
        for i in range(5)
    ]

    result = await gen.generate(population=population, generation=1)
    assert result is None
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_academic.py tests/unit/test_anomaly.py tests/unit/test_counter_strategy.py tests/unit/test_meta_strategy.py -v
```

Expected: FAIL — `ModuleNotFoundError` for each generator module

**Step 3: Implement the advanced generators**

```python
# src/evolve_trader/incubator/generators/academic.py
"""Academic paper mining generator.

Translates insights from academic finance papers into
actionable trading strategies via LLM interpretation.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from evolve_trader.incubator.candidate import (
    Candidate,
    CandidateOrigin,
    CandidateStatus,
    PaperPortfolio,
)
from evolve_trader.incubator.generators.base import GeneratorBase, GenerationResult

logger = logging.getLogger(__name__)

ACADEMIC_PROMPT = """You are translating an academic finance paper into a trading strategy.

Paper abstract:
{paper_abstract}

If this paper contains an actionable trading insight, respond with JSON:
{{
    "actionable": true,
    "paper_title": "<inferred or provided title>",
    "key_insight": "<the core tradeable insight>",
    "proposed_strategy": "<how to implement it>",
    "hypothesis": "<expected edge and why>",
    "skill_md_content": "<full SKILL.md>"
}}

If the paper has no actionable trading signal, respond with:
{{
    "actionable": false,
    "rejection_reason": "<why it's not actionable>"
}}
"""


class AcademicMiningGenerator(GeneratorBase):
    """Generates strategies by mining academic paper insights."""

    def __init__(self, llm_client: Any, initial_capital: float = 100_000.0):
        self._llm = llm_client
        self._initial_capital = initial_capital

    async def generate(
        self,
        paper_abstract: str = "",
        generation: int = 0,
        **kwargs,
    ) -> GenerationResult | None:
        """Translate an academic paper into a candidate strategy.

        Args:
            paper_abstract: The paper's abstract text.
            generation: Current tournament generation.

        Returns:
            GenerationResult or None if not actionable / on failure.
        """
        try:
            response = await self._llm.generate(
                prompt=ACADEMIC_PROMPT.format(paper_abstract=paper_abstract)
            )

            if not response.get("actionable", False):
                logger.info(
                    "Paper not actionable: %s",
                    response.get("rejection_reason", "unknown"),
                )
                return None

            child_id = f"acad-{uuid.uuid4().hex[:8]}"
            paper_title = response.get("paper_title", "unknown-paper")
            child_name = f"academic-{paper_title[:30].replace(' ', '-').lower()}"

            child = Candidate(
                id=child_id,
                name=child_name,
                skill_md_path=f"incubator/{child_name}.skill.md",
                origin=CandidateOrigin.ACADEMIC,
                parent_ids=[],
                hypothesis=response.get("hypothesis", ""),
                status=CandidateStatus.SEEDED,
                generation=generation,
                created_at=datetime.now(timezone.utc),
                portfolio=PaperPortfolio(initial_capital=self._initial_capital),
                metadata={
                    "paper_title": paper_title,
                    "key_insight": response.get("key_insight", ""),
                },
            )

            return GenerationResult(
                candidate=child,
                hypothesis=response.get("hypothesis", ""),
                mutation_description=f"Academic mining: {paper_title}",
                parent_ids=[],
                metadata={"paper_title": paper_title},
            )
        except Exception:
            logger.exception("Academic mining generation failed")
            return None
```

```python
# src/evolve_trader/incubator/generators/anomaly.py
"""Anomaly detection generator.

Scans market data for statistical anomalies and generates strategies
that exploit the detected patterns.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from evolve_trader.incubator.candidate import (
    Candidate,
    CandidateOrigin,
    CandidateStatus,
    PaperPortfolio,
)
from evolve_trader.incubator.generators.base import GeneratorBase, GenerationResult

logger = logging.getLogger(__name__)

ANOMALY_PROMPT = """You are designing a trading strategy to exploit a detected market anomaly.

Anomaly details:
- Number of anomalous observations: {n_anomalies}
- Average z-score: {avg_z_score:.2f}
- Anomaly indices: {anomaly_indices}

Design a strategy that exploits this statistical pattern. Respond with JSON:
{{
    "anomaly_type": "<classification of the anomaly>",
    "exploitation_approach": "<how to trade it>",
    "hypothesis": "<expected edge>",
    "skill_md_content": "<full SKILL.md>"
}}
"""


@dataclass
class MarketAnomaly:
    """A detected statistical anomaly in market data."""
    index: int
    value: float
    z_score: float

    @staticmethod
    def detect(
        returns: np.ndarray, z_threshold: float = 3.0
    ) -> list[MarketAnomaly]:
        """Detect statistical anomalies in a return series.

        Args:
            returns: Array of returns.
            z_threshold: Minimum absolute z-score for anomaly.

        Returns:
            List of detected anomalies.
        """
        if len(returns) < 2:
            return []

        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        if std == 0:
            return []

        z_scores = (returns - mean) / std
        anomalies = []

        for i, (val, z) in enumerate(zip(returns, z_scores)):
            if abs(z) >= z_threshold:
                anomalies.append(MarketAnomaly(
                    index=i,
                    value=float(val),
                    z_score=float(z),
                ))

        return anomalies


class AnomalyDetectionGenerator(GeneratorBase):
    """Generates strategies by detecting and exploiting market anomalies."""

    def __init__(self, llm_client: Any, initial_capital: float = 100_000.0):
        self._llm = llm_client
        self._initial_capital = initial_capital

    async def generate(
        self,
        market_returns: np.ndarray | None = None,
        generation: int = 0,
        z_threshold: float = 3.0,
        **kwargs,
    ) -> GenerationResult | None:
        """Detect anomalies and generate a strategy to exploit them.

        Args:
            market_returns: Array of market returns.
            generation: Current tournament generation.
            z_threshold: Minimum z-score for anomaly detection.

        Returns:
            GenerationResult or None if no anomalies / on failure.
        """
        if market_returns is None:
            return None

        anomalies = MarketAnomaly.detect(market_returns, z_threshold=z_threshold)
        if not anomalies:
            logger.info("No market anomalies detected at z_threshold=%.1f", z_threshold)
            return None

        try:
            avg_z = np.mean([abs(a.z_score) for a in anomalies])
            indices = [a.index for a in anomalies]

            response = await self._llm.generate(
                prompt=ANOMALY_PROMPT.format(
                    n_anomalies=len(anomalies),
                    avg_z_score=avg_z,
                    anomaly_indices=indices[:20],  # limit for prompt
                )
            )

            child_id = f"anom-{uuid.uuid4().hex[:8]}"
            anomaly_type = response.get("anomaly_type", "unknown")
            child_name = f"anomaly-{anomaly_type[:25].replace(' ', '-').lower()}"

            child = Candidate(
                id=child_id,
                name=child_name,
                skill_md_path=f"incubator/{child_name}.skill.md",
                origin=CandidateOrigin.ANOMALY,
                parent_ids=[],
                hypothesis=response.get("hypothesis", ""),
                status=CandidateStatus.SEEDED,
                generation=generation,
                created_at=datetime.now(timezone.utc),
                portfolio=PaperPortfolio(initial_capital=self._initial_capital),
                metadata={
                    "anomaly_type": anomaly_type,
                    "n_anomalies": len(anomalies),
                    "avg_z_score": float(avg_z),
                },
            )

            return GenerationResult(
                candidate=child,
                hypothesis=response.get("hypothesis", ""),
                mutation_description=f"Anomaly detection: {anomaly_type}",
                parent_ids=[],
                metadata={"anomaly_type": anomaly_type},
            )
        except Exception:
            logger.exception("Anomaly detection generation failed")
            return None
```

```python
# src/evolve_trader/incubator/generators/counter_strategy.py
"""Counter-strategy generator.

Generates strategies that are the inverse of a target strategy,
designed to profit when the target fails.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from evolve_trader.incubator.candidate import (
    Candidate,
    CandidateOrigin,
    CandidateStatus,
    PaperPortfolio,
)
from evolve_trader.incubator.generators.base import GeneratorBase, GenerationResult

logger = logging.getLogger(__name__)

COUNTER_STRATEGY_PROMPT = """You are designing a counter-strategy — the inverse of a target trading strategy.
The counter-strategy should profit when the target strategy loses.

Target strategy:
- Name: {target_name}
- Hypothesis: {target_hypothesis}
- SKILL.md path: {target_skill_path}

Analyze the target's weaknesses and design a strategy that exploits them.
Respond with JSON:
{{
    "target_weakness": "<identified weakness>",
    "counter_approach": "<how to exploit the weakness>",
    "hypothesis": "<expected edge of the counter-strategy>",
    "skill_md_content": "<full SKILL.md>"
}}
"""


class CounterStrategyGenerator(GeneratorBase):
    """Generates inverse strategies targeting another strategy's weaknesses."""

    def __init__(self, llm_client: Any, initial_capital: float = 100_000.0):
        self._llm = llm_client
        self._initial_capital = initial_capital

    async def generate(
        self,
        target_strategy: Candidate | None = None,
        generation: int = 0,
        **kwargs,
    ) -> GenerationResult | None:
        """Generate a counter-strategy for a target.

        Args:
            target_strategy: The strategy to counter.
            generation: Current tournament generation.

        Returns:
            GenerationResult or None on failure.
        """
        if target_strategy is None:
            return None

        try:
            response = await self._llm.generate(
                prompt=COUNTER_STRATEGY_PROMPT.format(
                    target_name=target_strategy.name,
                    target_hypothesis=target_strategy.hypothesis,
                    target_skill_path=target_strategy.skill_md_path,
                )
            )

            child_id = f"counter-{target_strategy.id}-{uuid.uuid4().hex[:8]}"
            child_name = f"counter-{target_strategy.name[:25]}"

            child = Candidate(
                id=child_id,
                name=child_name,
                skill_md_path=f"incubator/{child_name}.skill.md",
                origin=CandidateOrigin.COUNTER_STRATEGY,
                parent_ids=[target_strategy.id],
                hypothesis=response.get("hypothesis", ""),
                status=CandidateStatus.SEEDED,
                generation=generation,
                created_at=datetime.now(timezone.utc),
                portfolio=PaperPortfolio(initial_capital=self._initial_capital),
                metadata={
                    "target_id": target_strategy.id,
                    "target_weakness": response.get("target_weakness", ""),
                },
            )

            return GenerationResult(
                candidate=child,
                hypothesis=response.get("hypothesis", ""),
                mutation_description=f"Counter-strategy to {target_strategy.name}",
                parent_ids=[target_strategy.id],
            )
        except Exception:
            logger.exception(
                "Counter-strategy generation failed for target %s",
                target_strategy.id,
            )
            return None
```

```python
# src/evolve_trader/incubator/generators/meta_strategy.py
"""Meta-strategy discovery generator.

Analyzes patterns across the population of successful strategies
to discover emergent meta-level insights (e.g., "all top strategies
use volume confirmation").
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from evolve_trader.incubator.candidate import (
    Candidate,
    CandidateOrigin,
    CandidateStatus,
    PaperPortfolio,
)
from evolve_trader.incubator.generators.base import GeneratorBase, GenerationResult

logger = logging.getLogger(__name__)

META_STRATEGY_PROMPT = """You are analyzing a population of trading strategies to discover meta-level patterns.

Top-performing strategies:
{top_strategies}

Underperforming strategies:
{bottom_strategies}

Find common patterns among the top performers that differ from the bottom performers.
Design a new strategy that distills the meta-pattern into a pure strategy.

Respond with JSON:
{{
    "discovered_pattern": "<pattern found across top strategies>",
    "meta_approach": "<strategy that embodies the meta-pattern>",
    "hypothesis": "<why this meta-pattern is the real alpha>",
    "skill_md_content": "<full SKILL.md>"
}}
"""


class MetaStrategyGenerator(GeneratorBase):
    """Discovers meta-strategies from population-level patterns."""

    def __init__(
        self,
        llm_client: Any,
        min_population: int = 3,
        initial_capital: float = 100_000.0,
    ):
        self._llm = llm_client
        self._min_population = min_population
        self._initial_capital = initial_capital

    async def generate(
        self,
        population: list[Candidate] | None = None,
        generation: int = 0,
        **kwargs,
    ) -> GenerationResult | None:
        """Discover meta-strategies from population patterns.

        Args:
            population: Current tournament population.
            generation: Current tournament generation.

        Returns:
            GenerationResult or None if insufficient population / on failure.
        """
        if population is None or len(population) < self._min_population:
            logger.info(
                "Insufficient population for meta-strategy: %d < %d",
                len(population) if population else 0,
                self._min_population,
            )
            return None

        try:
            # Sort by composite fitness
            sorted_pop = sorted(
                population,
                key=lambda c: c.fitness_scores.get("composite", 0.0),
                reverse=True,
            )

            top_n = max(1, len(sorted_pop) // 3)
            bottom_n = max(1, len(sorted_pop) // 3)

            top_desc = "\n".join(
                f"- {c.name}: Sharpe={c.fitness_scores.get('sharpe', 0):.2f}, "
                f"hypothesis='{c.hypothesis}'"
                for c in sorted_pop[:top_n]
            )
            bottom_desc = "\n".join(
                f"- {c.name}: Sharpe={c.fitness_scores.get('sharpe', 0):.2f}, "
                f"hypothesis='{c.hypothesis}'"
                for c in sorted_pop[-bottom_n:]
            )

            response = await self._llm.generate(
                prompt=META_STRATEGY_PROMPT.format(
                    top_strategies=top_desc,
                    bottom_strategies=bottom_desc,
                )
            )

            child_id = f"meta-{uuid.uuid4().hex[:8]}"
            pattern = response.get("discovered_pattern", "unknown")
            child_name = f"meta-{pattern[:25].replace(' ', '-').lower()}"

            child = Candidate(
                id=child_id,
                name=child_name,
                skill_md_path=f"incubator/{child_name}.skill.md",
                origin=CandidateOrigin.META_STRATEGY,
                parent_ids=[c.id for c in sorted_pop[:top_n]],
                hypothesis=response.get("hypothesis", ""),
                status=CandidateStatus.SEEDED,
                generation=generation,
                created_at=datetime.now(timezone.utc),
                portfolio=PaperPortfolio(initial_capital=self._initial_capital),
                metadata={
                    "discovered_pattern": pattern,
                    "population_size_analyzed": len(population),
                },
            )

            return GenerationResult(
                candidate=child,
                hypothesis=response.get("hypothesis", ""),
                mutation_description=f"Meta-strategy: {pattern}",
                parent_ids=[c.id for c in sorted_pop[:top_n]],
                metadata={"discovered_pattern": pattern},
            )
        except Exception:
            logger.exception("Meta-strategy discovery failed")
            return None
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_academic.py tests/unit/test_anomaly.py tests/unit/test_counter_strategy.py tests/unit/test_meta_strategy.py -v
```

Expected: PASS — all advanced generator tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/incubator/generators/academic.py src/evolve_trader/incubator/generators/anomaly.py src/evolve_trader/incubator/generators/counter_strategy.py src/evolve_trader/incubator/generators/meta_strategy.py tests/unit/test_academic.py tests/unit/test_anomaly.py tests/unit/test_counter_strategy.py tests/unit/test_meta_strategy.py
git commit -m "feat: advanced generators — academic mining, anomaly detection, counter-strategy, meta-strategy"
```

---

## Task 9: Population Dynamics

**Files:**
- Create: `src/evolve_trader/incubator/population.py`
- Create: `tests/unit/test_population_dynamics.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_population_dynamics.py
"""Tests for population dynamics — adaptive exploration/exploitation."""
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from evolve_trader.incubator.population import (
    PopulationManager,
    PopulationConfig,
    ExplorationExploitationBalance,
)
from evolve_trader.incubator.candidate import (
    Candidate,
    CandidateOrigin,
    CandidateStatus,
    PaperPortfolio,
)
from evolve_trader.incubator.tournament import Tournament, TournamentConfig


def _make_candidate(
    id: str,
    origin: CandidateOrigin = CandidateOrigin.SEED,
    sharpe: float = 0.5,
) -> Candidate:
    c = Candidate(
        id=id,
        name=f"strategy-{id}",
        skill_md_path=f"incubator/strategy-{id}.skill.md",
        origin=origin,
        parent_ids=[],
        hypothesis="Test",
        status=CandidateStatus.EVALUATING,
        generation=0,
        created_at=datetime.now(timezone.utc),
        portfolio=PaperPortfolio(initial_capital=100_000.0),
    )
    c.fitness_scores = {"sharpe": sharpe, "composite": sharpe}
    return c


def test_population_config_defaults():
    """PopulationConfig has sensible defaults."""
    config = PopulationConfig()
    assert config.target_population == 8
    assert config.min_population == 3
    assert config.max_population == 12
    assert 0.0 <= config.initial_exploration_rate <= 1.0


def test_population_config_target_in_range():
    """Target population must be between min and max."""
    with pytest.raises(ValueError, match="target"):
        PopulationConfig(target_population=15, min_population=3, max_population=10)


def test_exploration_exploitation_balance():
    """Balance correctly shifts between exploration and exploitation."""
    balance = ExplorationExploitationBalance(initial_rate=0.7)

    # Early on, exploration should be high
    assert balance.exploration_rate == 0.7

    # After many successful generations, exploitation increases
    for _ in range(10):
        balance.record_generation_result(
            avg_fitness_improvement=0.05,
            n_promotions=1,
        )
    assert balance.exploration_rate < 0.7  # should decrease

    # After stagnation, exploration increases
    for _ in range(10):
        balance.record_generation_result(
            avg_fitness_improvement=-0.01,
            n_promotions=0,
        )
    assert balance.exploration_rate > 0.3  # should bounce back


def test_exploration_exploitation_clamped():
    """Exploration rate stays in [0.1, 0.9] bounds."""
    balance = ExplorationExploitationBalance(initial_rate=0.5)

    # Massive exploitation pressure
    for _ in range(100):
        balance.record_generation_result(
            avg_fitness_improvement=0.5,
            n_promotions=5,
        )
    assert balance.exploration_rate >= 0.1

    # Massive exploration pressure
    balance = ExplorationExploitationBalance(initial_rate=0.5)
    for _ in range(100):
        balance.record_generation_result(
            avg_fitness_improvement=-0.5,
            n_promotions=0,
        )
    assert balance.exploration_rate <= 0.9


def test_population_manager_suggests_generation_methods():
    """PopulationManager suggests which generators to use based on balance."""
    config = PopulationConfig(target_population=8)
    manager = PopulationManager(config=config)

    # Exploration-heavy: should favor diverse methods
    manager._balance = ExplorationExploitationBalance(initial_rate=0.8)
    suggestions = manager.suggest_generation_methods(n_slots=3)
    # Higher exploration -> more diverse origins
    assert len(suggestions) >= 1
    assert any(
        s in (CandidateOrigin.REGIME_SEARCH, CandidateOrigin.ACADEMIC,
              CandidateOrigin.ANOMALY)
        for s in suggestions
    )

    # Exploitation-heavy: should favor mutation/crossover
    manager._balance = ExplorationExploitationBalance(initial_rate=0.2)
    suggestions = manager.suggest_generation_methods(n_slots=3)
    assert any(
        s in (CandidateOrigin.MUTATION, CandidateOrigin.CROSSOVER)
        for s in suggestions
    )


def test_population_manager_computes_slots_available():
    """PopulationManager correctly calculates available population slots."""
    config = PopulationConfig(target_population=8, max_population=12)
    manager = PopulationManager(config=config)

    tournament = Tournament(config=TournamentConfig(max_population=12))
    for i in range(5):
        tournament.add_candidate(_make_candidate(f"pop-{i}"))

    slots = manager.compute_available_slots(tournament)
    assert slots == 3  # target(8) - current(5)


def test_population_manager_respects_max():
    """PopulationManager caps slots at max_population."""
    config = PopulationConfig(target_population=8, max_population=10)
    manager = PopulationManager(config=config)

    tournament = Tournament(config=TournamentConfig(max_population=10))
    for i in range(9):
        tournament.add_candidate(_make_candidate(f"pop-{i}"))

    slots = manager.compute_available_slots(tournament)
    assert slots == 1  # max(10) - current(9), but target would say -1


def test_population_manager_diversity_score():
    """PopulationManager computes origin diversity score."""
    config = PopulationConfig()
    manager = PopulationManager(config=config)

    # Homogeneous population (all mutations)
    homogeneous = [
        _make_candidate(f"h-{i}", origin=CandidateOrigin.MUTATION)
        for i in range(5)
    ]
    score_homo = manager.compute_diversity_score(homogeneous)

    # Diverse population
    diverse = [
        _make_candidate("d-0", origin=CandidateOrigin.MUTATION),
        _make_candidate("d-1", origin=CandidateOrigin.CROSSOVER),
        _make_candidate("d-2", origin=CandidateOrigin.REGIME_SEARCH),
        _make_candidate("d-3", origin=CandidateOrigin.ACADEMIC),
        _make_candidate("d-4", origin=CandidateOrigin.ANOMALY),
    ]
    score_diverse = manager.compute_diversity_score(diverse)

    assert score_diverse > score_homo
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_population_dynamics.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evolve_trader.incubator.population'`

**Step 3: Implement population dynamics**

```python
# src/evolve_trader/incubator/population.py
"""Population dynamics — adaptive exploration vs exploitation.

The PopulationManager adjusts the target population size and generation
method mix based on tournament performance history. When fitness improves
steadily, it favors exploitation (mutation, crossover). When stagnating,
it shifts toward exploration (regime search, academic, anomaly, etc.).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from evolve_trader.incubator.candidate import Candidate, CandidateOrigin
from evolve_trader.incubator.tournament import Tournament


# Methods classified by exploration vs exploitation
EXPLOITATION_METHODS = {CandidateOrigin.MUTATION, CandidateOrigin.CROSSOVER}
EXPLORATION_METHODS = {
    CandidateOrigin.REGIME_SEARCH,
    CandidateOrigin.ACADEMIC,
    CandidateOrigin.ANOMALY,
    CandidateOrigin.COUNTER_STRATEGY,
    CandidateOrigin.META_STRATEGY,
}


@dataclass
class PopulationConfig:
    """Configuration for population dynamics.

    Raises:
        ValueError: If target is outside [min, max] range.
    """
    target_population: int = 8
    min_population: int = 3
    max_population: int = 12
    initial_exploration_rate: float = 0.6

    def __post_init__(self):
        if not (self.min_population <= self.target_population <= self.max_population):
            raise ValueError(
                f"target_population ({self.target_population}) must be between "
                f"min ({self.min_population}) and max ({self.max_population})"
            )


class ExplorationExploitationBalance:
    """Tracks and adapts the exploration vs exploitation balance.

    Exploration rate increases on stagnation, decreases on steady improvement.
    Clamped to [0.1, 0.9].
    """

    def __init__(self, initial_rate: float = 0.6):
        self._rate = initial_rate
        self._history: list[dict[str, Any]] = []

    @property
    def exploration_rate(self) -> float:
        return self._rate

    @exploration_rate.setter
    def exploration_rate(self, value: float) -> None:
        self._rate = max(0.1, min(0.9, value))

    def record_generation_result(
        self,
        avg_fitness_improvement: float,
        n_promotions: int,
    ) -> None:
        """Update balance based on generation outcome.

        Args:
            avg_fitness_improvement: Mean fitness change (positive = improving).
            n_promotions: Number of strategies promoted this generation.
        """
        self._history.append({
            "improvement": avg_fitness_improvement,
            "promotions": n_promotions,
        })

        # Positive improvement + promotions -> more exploitation
        if avg_fitness_improvement > 0 and n_promotions > 0:
            self._rate = max(0.1, self._rate - 0.05)
        elif avg_fitness_improvement > 0:
            self._rate = max(0.1, self._rate - 0.02)
        # Stagnation or decline -> more exploration
        elif avg_fitness_improvement <= 0 and n_promotions == 0:
            self._rate = min(0.9, self._rate + 0.05)
        else:
            self._rate = min(0.9, self._rate + 0.01)


class PopulationManager:
    """Manages population size and generation method selection."""

    def __init__(self, config: PopulationConfig | None = None):
        self._config = config or PopulationConfig()
        self._balance = ExplorationExploitationBalance(
            initial_rate=self._config.initial_exploration_rate
        )

    @property
    def balance(self) -> ExplorationExploitationBalance:
        return self._balance

    def compute_available_slots(self, tournament: Tournament) -> int:
        """Compute how many new candidates can be added.

        Uses target_population as the goal but caps at max_population.

        Args:
            tournament: Current tournament state.

        Returns:
            Number of available slots (>= 0).
        """
        current = len(tournament.get_active_candidates())
        target_slots = self._config.target_population - current
        max_slots = self._config.max_population - len(tournament.population)
        return max(0, min(target_slots, max_slots))

    def suggest_generation_methods(
        self, n_slots: int
    ) -> list[CandidateOrigin]:
        """Suggest generation methods for available slots.

        Mixes exploration and exploitation methods based on the
        current balance.

        Args:
            n_slots: Number of candidates to generate.

        Returns:
            List of suggested CandidateOrigin values.
        """
        if n_slots <= 0:
            return []

        exploration_methods = list(EXPLORATION_METHODS)
        exploitation_methods = list(EXPLOITATION_METHODS)

        n_explore = max(1, round(n_slots * self._balance.exploration_rate))
        n_exploit = n_slots - n_explore

        suggestions: list[CandidateOrigin] = []

        for i in range(n_explore):
            suggestions.append(
                exploration_methods[i % len(exploration_methods)]
            )

        for i in range(n_exploit):
            suggestions.append(
                exploitation_methods[i % len(exploitation_methods)]
            )

        return suggestions[:n_slots]

    def compute_diversity_score(
        self, population: list[Candidate]
    ) -> float:
        """Compute origin diversity using normalized entropy.

        Returns:
            Diversity score in [0, 1]. 1.0 = maximally diverse.
        """
        if not population:
            return 0.0

        counts = Counter(c.origin for c in population)
        n = len(population)

        if n <= 1:
            return 0.0

        # Shannon entropy
        entropy = 0.0
        for count in counts.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize by max possible entropy
        n_origins = len(CandidateOrigin)
        max_entropy = math.log2(min(n_origins, n))

        if max_entropy == 0:
            return 0.0
        return entropy / max_entropy
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_population_dynamics.py -v
```

Expected: PASS — all population dynamics tests green

**Step 5: Commit**

```bash
git add src/evolve_trader/incubator/population.py tests/unit/test_population_dynamics.py
git commit -m "feat: population dynamics — adaptive exploration/exploitation with diversity tracking"
```

---

## Task 10: Integration Testing & Final Verification

**Files:**
- Create: `tests/integration/test_incubator_pipeline.py`

**Step 1: Write the integration tests**

```python
# tests/integration/test_incubator_pipeline.py
"""Integration tests for the full incubator pipeline.

Verifies that all components work together: tournament, phases,
fitness, generators, fossil record, and population dynamics.
"""
import pytest
import numpy as np
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from evolve_trader.incubator.tournament import Tournament, TournamentConfig
from evolve_trader.incubator.candidate import (
    Candidate,
    CandidateOrigin,
    CandidateStatus,
    PaperPortfolio,
)
from evolve_trader.incubator.phases import (
    TournamentPhaseManager,
    seed_candidates,
    transition_to_incubation,
    evaluate_candidates,
    select_candidates,
    reproduce_candidates,
    check_promotions,
)
from evolve_trader.incubator.fitness import FitnessConfig, FitnessEvaluator
from evolve_trader.incubator.fossil_record import FossilRecord, FailureCondition
from evolve_trader.incubator.population import (
    PopulationManager,
    PopulationConfig,
)
from evolve_trader.incubator.generators.mutation import MutationGenerator
from evolve_trader.incubator.generators.crossover import CrossoverGenerator
from evolve_trader.incubator.generators.regime_search import RegimeSearchGenerator


def _make_candidate(
    id: str,
    status: CandidateStatus = CandidateStatus.SEEDED,
    sharpe: float = 0.0,
    generation: int = 0,
) -> Candidate:
    c = Candidate(
        id=id,
        name=f"strategy-{id}",
        skill_md_path=f"incubator/strategy-{id}.skill.md",
        origin=CandidateOrigin.SEED,
        parent_ids=[],
        hypothesis="Test candidate",
        status=status,
        generation=generation,
        created_at=datetime.now(timezone.utc),
        portfolio=PaperPortfolio(initial_capital=100_000.0),
    )
    c.fitness_scores = {"sharpe": sharpe, "composite": max(0, sharpe)}
    return c


def test_full_tournament_lifecycle():
    """Run a complete tournament lifecycle: seed -> incubate -> evaluate -> select -> reproduce."""
    config = TournamentConfig(
        max_population=10,
        incubation_days=0,  # immediate for testing
        elimination_pct=0.30,
        reproduction_pct=0.10,
        promotion_min_days=0,
        promotion_min_sharpe=0.8,
    )
    tournament = Tournament(config=config)

    # 1. Seed
    seeds = [_make_candidate(f"seed-{i}") for i in range(8)]
    added = seed_candidates(tournament, seeds)
    assert added == 8

    # 2. Incubate
    transitioned = transition_to_incubation(tournament)
    assert transitioned == 8

    # 3. Simulate some trading (set incubation_start in the past)
    for c in tournament.population:
        c.incubation_start = datetime.now(timezone.utc) - timedelta(days=1)

    # 4. Evaluate
    evaluated = evaluate_candidates(tournament)
    assert evaluated == 8

    # 5. Assign fitness scores
    for i, c in enumerate(tournament.population):
        c.fitness_scores["composite"] = (i + 1) * 0.1

    # 6. Select (eliminate bottom 30%)
    eliminated = select_candidates(tournament)
    assert eliminated >= 2  # 30% of 8 ~= 2.4 -> 2

    # 7. Reproduce (from top 10%)
    reproduced = reproduce_candidates(tournament)
    assert reproduced >= 1

    # 8. Check promotions
    top_candidate = max(
        tournament.population,
        key=lambda c: c.fitness_scores.get("composite", 0),
    )
    top_candidate.incubation_start = datetime.now(timezone.utc) - timedelta(days=1)
    top_candidate.status = CandidateStatus.EVALUATING
    promoted = check_promotions(tournament)
    # Only promotes if composite >= 0.8
    if top_candidate.fitness_scores["composite"] >= 0.8:
        assert promoted >= 1


def test_fitness_and_selection_integration():
    """Fitness evaluator scores feed into selection decisions."""
    config = FitnessConfig()
    evaluator = FitnessEvaluator(config)

    rng = np.random.default_rng(42)

    # Evaluate two candidates with different performance
    good_returns = rng.normal(0.002, 0.01, 252)
    bad_returns = rng.normal(-0.001, 0.02, 252)
    prod_returns = rng.normal(0.0005, 0.012, 252)

    good_prices = np.cumprod(1 + good_returns) * 100
    bad_prices = np.cumprod(1 + bad_returns) * 100

    regime_sharpes = {0: 0.5, 1: -0.3}
    prod_regime_sharpes = {0: 1.0, 1: -0.5}

    good_fitness = evaluator.evaluate(
        candidate_returns=good_returns,
        candidate_prices=good_prices,
        production_returns=prod_returns,
        candidate_regime_sharpes=regime_sharpes,
        production_regime_sharpes=prod_regime_sharpes,
    )
    bad_fitness = evaluator.evaluate(
        candidate_returns=bad_returns,
        candidate_prices=bad_prices,
        production_returns=prod_returns,
        candidate_regime_sharpes=regime_sharpes,
        production_regime_sharpes=prod_regime_sharpes,
    )

    assert good_fitness["composite"] > bad_fitness["composite"]
    assert good_fitness["sharpe"] > bad_fitness["sharpe"]


def test_fossil_record_and_resurrection_integration():
    """Eliminated strategies are archived and can be found for resurrection."""
    tournament_config = TournamentConfig(
        max_population=10,
        elimination_pct=0.50,
    )
    tournament = Tournament(config=tournament_config)
    fossil_record = FossilRecord()

    # Add candidates, assign fitness, select
    for i in range(6):
        c = _make_candidate(
            f"fossil-int-{i}",
            status=CandidateStatus.EVALUATING,
            sharpe=(i + 1) * 0.1,
        )
        c.fitness_scores["composite"] = (i + 1) * 0.1
        c.metadata["dominant_regime"] = i % 3
        tournament.add_candidate(c)

    eliminated_count = select_candidates(tournament)
    assert eliminated_count >= 2

    # Archive eliminated
    for c in tournament.population:
        if c.status == CandidateStatus.ELIMINATED:
            failure = FailureCondition(
                reason="Low composite fitness",
                regime_at_failure=c.metadata.get("dominant_regime", 0),
                market_conditions={"vix": 20.0},
                fitness_at_failure=dict(c.fitness_scores),
            )
            fossil_record.archive(c, failure)

    assert len(fossil_record) == eliminated_count

    # Check resurrection when regime changes
    resurrection = fossil_record.find_resurrection_candidates(
        current_regime=99,  # a regime none failed in
        current_conditions={"vix": 12.0},
    )
    assert len(resurrection) == eliminated_count


def test_population_manager_guides_generation():
    """PopulationManager's suggestions integrate with tournament."""
    pop_config = PopulationConfig(
        target_population=8,
        max_population=12,
    )
    manager = PopulationManager(config=pop_config)

    tournament = Tournament(config=TournamentConfig(max_population=12))
    for i in range(5):
        tournament.add_candidate(
            _make_candidate(f"pm-{i}", status=CandidateStatus.INCUBATING)
        )

    slots = manager.compute_available_slots(tournament)
    assert slots == 3  # target(8) - active(5)

    suggestions = manager.suggest_generation_methods(slots)
    assert len(suggestions) == 3
    assert all(isinstance(s, CandidateOrigin) for s in suggestions)


@pytest.mark.asyncio
async def test_generators_produce_valid_candidates():
    """All generators produce candidates with valid structure."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = {
        "mutated_component": "entry_signal",
        "original_value": "SMA crossover",
        "new_value": "RSI + SMA",
        "hypothesis": "Better entries",
        "skill_md_content": "# Strategy\n...",
        "combined_components": {"entry": "from A"},
        "coherence_score": 0.9,
    }

    parent = _make_candidate("gen-parent", status=CandidateStatus.EVALUATING)

    # Test mutation
    mut_gen = MutationGenerator(llm_client=mock_llm)
    mut_result = await mut_gen.generate(parent=parent, generation=1)
    assert mut_result is not None
    assert mut_result.candidate.status == CandidateStatus.SEEDED

    # Test crossover
    cross_gen = CrossoverGenerator(llm_client=mock_llm)
    parent2 = _make_candidate("gen-parent-2", status=CandidateStatus.EVALUATING)
    cross_result = await cross_gen.generate(
        parents=[parent, parent2], generation=1
    )
    assert cross_result is not None
    assert len(cross_result.candidate.parent_ids) == 2
```

**Step 2: Run all tests**

```bash
pytest tests/unit/test_tournament.py tests/unit/test_tournament_phases.py tests/unit/test_fitness.py tests/unit/test_mutation.py tests/unit/test_crossover.py tests/unit/test_regime_search.py tests/unit/test_fossil_record.py tests/unit/test_academic.py tests/unit/test_anomaly.py tests/unit/test_counter_strategy.py tests/unit/test_meta_strategy.py tests/unit/test_population_dynamics.py tests/integration/test_incubator_pipeline.py -v
```

Expected: ALL PASS

**Step 3: Run linting and type checking**

```bash
ruff check src/evolve_trader/incubator/
mypy src/evolve_trader/incubator/ --ignore-missing-imports
```

Expected: No errors

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "test: Phase 8 final verification — all incubator tests passing"
```

---

## Parallelization Notes

Tasks in this phase have the following dependency structure:

```
Task 1 (Tournament Architecture) ──────────┐
                                             ├── Task 2 (Tournament Phases)
                                             ├── Task 3 (Fitness Function)
                                             │
                                             ├── Task 4 (Mutation) ──────────┐
Task 4 depends on generators/base.py ───────┤── Task 5 (Crossover) ─────────┤
                                             ├── Task 6 (Regime Search) ─────┤
                                             │                                │
                                             ├── Task 7 (Fossil Record)       ├── Task 8 (Advanced Generators)
                                             │                                │
                                             └── Task 9 (Population) ─────────┘
                                                                              │
                                                          Task 10 (Integration Tests) ← depends on ALL
```

**Can run in parallel:**
- Tasks 2, 3 are independent of each other — run simultaneously after Task 1
- Tasks 4, 5, 6 are independent of each other — run simultaneously after Task 1 (Task 4 creates `generators/base.py` which 5 and 6 need, so start with 4)
- Task 7 (fossil record) is independent of Tasks 2-6 — run anytime after Task 1
- Task 8 (advanced generators) depends on `generators/base.py` from Task 4 but nothing else
- Task 9 (population dynamics) depends on Task 1 only
- Task 10 (integration) depends on everything — must be last
