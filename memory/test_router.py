from __future__ import annotations

import os

import pytest

from memory.episodic_store import EpisodicMemory
from memory.router import PromoteDropRouter
from memory.short_term import Turn


@pytest.fixture
def episodic_memory(tmp_path):
    db_path = os.path.join(tmp_path, "test_memory.db")
    return EpisodicMemory(db_path=db_path)


def test_significant_event_is_promoted(episodic_memory):
    router = PromoteDropRouter(episodic_memory)

    turn = Turn(
        role="tool",
        customer_id=12,
        session_id="sess-1",
        content={
            "event_type": "credit_hold_placed",
            "severity": "severe",
            "note": "90+ days overdue",
        },
    )

    decision = router.route(turn)

    assert decision.action == "episodic"
    assert "90+ days overdue" in decision.reason

    stored = episodic_memory.get_by_customer(12)
    assert len(stored) == 1
    assert stored[0].event_type == "credit_hold_placed"


def test_routine_turn_is_forgotten(episodic_memory):
    router = PromoteDropRouter(episodic_memory)

    turn = Turn(role="employee", customer_id=12, content="good morning")

    decision = router.route(turn)

    assert decision.action == "forget"
    assert episodic_memory.get_by_customer(12) == []


def test_every_decision_is_logged_with_reasoning(episodic_memory):
    router = PromoteDropRouter(episodic_memory)

    router.route(Turn(role="employee", customer_id=1, content="hi"))
    router.route(
        Turn(
            role="tool",
            customer_id=1,
            content={"event_type": "rate_exception_approved", "discount_pct": 20},
        )
    )

    assert len(router.decision_log) == 2
    actions = {d.action for d in router.decision_log}
    assert actions == {"forget", "episodic"}
    # Reasoning must be non-empty for every decision, not just promotions.
    assert all(d.reason for d in router.decision_log)


def test_router_never_touches_semantic_memory():
    """Static guardrail: the router module must not import
    memory.semantic_store, so writing to semantic memory from here is
    impossible at the code level, not just by convention."""

    import ast

    import memory.router as router_module

    with open(router_module.__file__) as f:
        tree = ast.parse(f.read())

    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)

    assert "memory.semantic_store" not in imported_modules
    assert "memory.consolidation" not in imported_modules
