"""Tests for the LLM usage logger."""

import json
import tempfile
from pathlib import Path

import pytest

from evolve_trader.core.llm_logger import LLMUsageLogger, LLMUsageRecord


def test_record_creation():
    """LLMUsageRecord captures all required fields."""
    record = LLMUsageRecord(
        model="claude-sonnet-4",
        component="strategy_execution",
        input_tokens=1500,
        output_tokens=500,
        cost_usd=0.012,
    )
    assert record.model == "claude-sonnet-4"
    assert record.total_tokens == 2000
    assert record.cost_usd == 0.012


def test_logger_writes_jsonl():
    """Logger persists records as JSONL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "llm_usage.jsonl"
        logger = LLMUsageLogger(log_path=log_path)

        logger.log(
            model="claude-sonnet-4",
            component="strategy_execution",
            input_tokens=1000,
            output_tokens=300,
            cost_usd=0.008,
        )

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["model"] == "claude-sonnet-4"
        assert data["component"] == "strategy_execution"


def test_logger_aggregates_by_component():
    """Logger can aggregate costs by component."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "llm_usage.jsonl"
        logger = LLMUsageLogger(log_path=log_path)

        logger.log("claude-sonnet-4", "strategy_execution", 1000, 300, 0.008)
        logger.log("claude-sonnet-4", "evolution", 2000, 500, 0.015)
        logger.log("claude-haiku-3.5", "strategy_execution", 500, 100, 0.001)

        totals = logger.cost_by_component()
        assert totals["strategy_execution"] == pytest.approx(0.009)
        assert totals["evolution"] == pytest.approx(0.015)


def test_logger_total_cost():
    """Logger tracks total cost."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "llm_usage.jsonl"
        logger = LLMUsageLogger(log_path=log_path)

        logger.log("claude-sonnet-4", "strategy_execution", 1000, 300, 0.008)
        logger.log("claude-sonnet-4", "evolution", 2000, 500, 0.015)

        assert logger.total_cost() == 0.023


def test_logger_budget_warning():
    """Logger reports budget status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "llm_usage.jsonl"
        logger = LLMUsageLogger(log_path=log_path, monthly_budget_usd=0.05)

        logger.log("claude-sonnet-4", "strategy_execution", 1000, 300, 0.045)

        assert logger.budget_utilization() >= 0.80
        assert logger.is_budget_warning()
        assert not logger.is_budget_exceeded()


def test_logger_budget_exceeded():
    """Logger enforces hard budget stop."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "llm_usage.jsonl"
        logger = LLMUsageLogger(log_path=log_path, monthly_budget_usd=0.01)

        logger.log("claude-sonnet-4", "strategy_execution", 1000, 300, 0.015)

        assert logger.is_budget_exceeded()
