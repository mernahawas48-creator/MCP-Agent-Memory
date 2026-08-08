from __future__ import annotations

from types import SimpleNamespace

from memory.verified_recall import SAFE_MEMORY_ANSWER, VerifiedMemoryRecall


class StubGenerator:
    model_name = "stub"

    def __init__(self, answer: str):
        self.answer_text = answer
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return self.answer_text


class StubEpisodic:
    def __init__(self, episodes):
        self.episodes = episodes

    def get_by_customer(self, customer_id):
        return [item for item in self.episodes if item.customer_id == customer_id]


class StubSemantic:
    def __init__(self, facts):
        self.facts = facts

    def get_active_facts(self, customer_id):
        return [item for item in self.facts if item.customer_id == customer_id]


def test_memory_recall_runs_relevance_and_support_checks():
    fact = SimpleNamespace(
        id=1,
        customer_id=12,
        fact_key="customer_risk_level",
        fact_value="high_risk",
    )
    episode = SimpleNamespace(
        id=2,
        customer_id=12,
        event_type="credit_hold_placed",
        content={"severity": "severe", "note": "90 days overdue"},
        reason="Severe hold changes financial state.",
    )
    generator = StubGenerator(
        "The customer risk level is high_risk [1]."
    )

    result = VerifiedMemoryRecall(
        episodic_memory=StubEpisodic([episode]),
        semantic_memory=StubSemantic([fact]),
        generator=generator,
    ).answer(
        "What is the customer risk level?",
        customer_id=12,
    )

    assert result.verification.retrieval_relevant is True
    assert result.verification.answer_supported is True
    assert result.answer != SAFE_MEMORY_ANSWER
    assert generator.calls == 1


def test_irrelevant_memory_is_not_sent_to_generator():
    fact = SimpleNamespace(
        id=1,
        customer_id=12,
        fact_key="customer_risk_level",
        fact_value="high_risk",
    )
    generator = StubGenerator("Invented storage fee is 40 [1].")

    result = VerifiedMemoryRecall(
        episodic_memory=StubEpisodic([]),
        semantic_memory=StubSemantic([fact]),
        generator=generator,
    ).answer(
        "What was the warehouse pallet fee?",
        customer_id=12,
    )

    assert result.verification.retrieval_relevant is False
    assert result.answer == SAFE_MEMORY_ANSWER
    assert generator.calls == 0


def test_unsupported_memory_answer_is_replaced_by_safe_answer():
    fact = SimpleNamespace(
        id=1,
        customer_id=12,
        fact_key="customer_risk_level",
        fact_value="high_risk",
    )
    generator = StubGenerator(
        "The customer risk score is 40 [1]."
    )

    result = VerifiedMemoryRecall(
        episodic_memory=StubEpisodic([]),
        semantic_memory=StubSemantic([fact]),
        generator=generator,
    ).answer(
        "What is the customer risk level?",
        customer_id=12,
    )

    assert result.verification.retrieval_relevant is True
    assert result.verification.answer_supported is False
    assert result.answer == SAFE_MEMORY_ANSWER
