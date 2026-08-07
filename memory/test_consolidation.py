from __future__ import annotations

import os

import pytest

from memory.consolidation import ConsolidationLayer
from memory.episodic_store import EpisodicMemory
from memory.semantic_store import SemanticMemory


@pytest.fixture
def stores(tmp_path):
    db_path = os.path.join(tmp_path, "test_memory.db")
    return EpisodicMemory(db_path=db_path), SemanticMemory(db_path=db_path)


def test_consolidation_derives_facts_and_marks_episodes_done(stores):
    episodic, semantic = stores
    consolidation = ConsolidationLayer(episodic, semantic)

    episodic.add_episode(
        event_type="credit_hold_released",
        content={"event_type": "credit_hold_released"},
        reason="Customer settled overdue balance",
        customer_id=12,
    )

    result = consolidation.run()

    assert result.episodes_processed == 1
    assert len(result.facts_written) == 1
    assert result.facts_written[0].fact_value == "good_standing"
    assert episodic.get_unconsolidated() == []


def test_consolidation_resolves_a_real_conflict_across_two_runs(stores):
    """Reproduces the Swiftrail scenario from the README: a customer
    looked fine, then a severe credit hold landed. Two separate
    consolidation passes must not silently overwrite the earlier fact."""

    episodic, semantic = stores
    consolidation = ConsolidationLayer(episodic, semantic)

    episodic.add_episode(
        event_type="credit_hold_released",
        content={"event_type": "credit_hold_released"},
        reason="Customer cleared prior balance",
        customer_id=12,
    )
    first_run = consolidation.run()
    assert first_run.conflicts_resolved == []
    assert semantic.get_active_fact(12, "customer_risk_level").fact_value == (
        "good_standing"
    )

    # A new episode lands later (different session) implying the
    # opposite risk level.
    episodic.add_episode(
        event_type="credit_hold_placed",
        content={"event_type": "credit_hold_placed", "severity": "severe"},
        reason="90+ days overdue on shipment 512",
        customer_id=12,
    )
    second_run = consolidation.run()

    assert len(second_run.conflicts_resolved) == 1
    resolved = second_run.conflicts_resolved[0]
    assert resolved.fact_value == "high_risk"
    assert "good_standing" in resolved.conflict_reason
    assert "high_risk" in resolved.conflict_reason

    # Full history is preserved, not overwritten.
    history = semantic.fact_history(12, "customer_risk_level")
    assert [f.fact_value for f in history] == ["good_standing", "high_risk"]
    assert history[0].status == "superseded"
    assert history[1].status == "active"
