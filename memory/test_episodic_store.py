from __future__ import annotations

import os

import pytest

from memory.episodic_store import EpisodicMemory


@pytest.fixture
def store(tmp_path):
    db_path = os.path.join(tmp_path, "test_memory.db")
    return EpisodicMemory(db_path=db_path)


def test_add_and_get_by_customer(store):
    store.add_episode(
        event_type="credit_hold_placed",
        content={"severity": "severe"},
        reason="90+ days overdue",
        customer_id=12,
        source_session_id="sess-1",
    )
    store.add_episode(
        event_type="rate_exception_approved",
        content={"discount_pct": 15},
        reason="within standard authority",
        customer_id=99,
    )

    customer_12 = store.get_by_customer(12)
    assert len(customer_12) == 1
    assert customer_12[0].event_type == "credit_hold_placed"

    assert store.get_by_customer(99)[0].content["discount_pct"] == 15


def test_unconsolidated_and_mark_consolidated(store):
    e1 = store.add_episode("credit_hold_placed", {}, "reason", customer_id=1)
    e2 = store.add_episode("credit_hold_released", {}, "reason", customer_id=1)

    unconsolidated = store.get_unconsolidated()
    assert {e.id for e in unconsolidated} == {e1.id, e2.id}

    store.mark_consolidated([e1.id])

    remaining = store.get_unconsolidated()
    assert [e.id for e in remaining] == [e2.id]
