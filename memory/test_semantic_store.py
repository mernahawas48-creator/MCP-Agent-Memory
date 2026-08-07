from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from memory.semantic_store import SemanticMemory


@pytest.fixture
def store(tmp_path):
    db_path = os.path.join(tmp_path, "test_memory.db")
    return SemanticMemory(db_path=db_path)


def test_first_fact_is_version_1(store):
    fact = store.upsert_fact(12, "customer_risk_level", "watch", source_episode_id=1)

    assert fact.version == 1
    assert fact.status == "active"
    assert store.get_active_fact(12, "customer_risk_level").fact_value == "watch"


def test_reaffirming_same_value_does_not_bump_version(store):
    first = store.upsert_fact(12, "customer_risk_level", "watch", source_episode_id=1)
    second = store.upsert_fact(12, "customer_risk_level", "watch", source_episode_id=2)

    assert second.version == first.version == 1
    assert second.id == first.id


def test_conflicting_value_creates_new_version_and_supersedes_old(store):
    """This is the real conflict the consolidation layer must resolve:
    one episode implies a customer is a good-standing risk, a later
    episode (a severe credit hold) implies the opposite."""

    old = store.upsert_fact(
        12, "customer_risk_level", "good_standing", source_episode_id=1
    )
    new = store.upsert_fact(
        12, "customer_risk_level", "high_risk", source_episode_id=7
    )

    assert new.version == old.version + 1
    assert new.status == "active"
    assert new.conflict_reason is not None
    assert "good_standing" in new.conflict_reason
    assert "high_risk" in new.conflict_reason

    # Old fact is never deleted -- it's superseded, with a pointer
    # forward to what replaced it.
    history = store.fact_history(12, "customer_risk_level")
    assert len(history) == 2
    assert history[0].status == "superseded"
    assert history[0].superseded_by == new.id
    assert history[1].status == "active"

    # Only the new value is returned as "the" active fact.
    assert store.get_active_fact(12, "customer_risk_level").fact_value == "high_risk"


def test_expire_stale_facts(store):
    fact = store.upsert_fact(
        12, "discount_authority", "approved_up_to_20pct",
        source_episode_id=1, ttl_days=1,
    )

    future = datetime.now(timezone.utc) + timedelta(days=2)
    expired = store.expire_stale_facts(as_of=future)

    assert len(expired) == 1
    assert expired[0].id == fact.id
    assert store.get_active_fact(12, "discount_authority") is None
