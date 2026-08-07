from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag.naive_rag.pipeline import (
    NO_CONTEXT_ANSWER,
    NaiveRAG,
)


class StubEmbedder:
    def __init__(self):
        self.last_query = None

    def embed_query(self, query: str):
        self.last_query = query
        return (1.0, 0.0, 0.0)


class StubVectorStore:
    def __init__(self, results):
        self.results = results
        self.last_vector = None
        self.last_filters = None
        self.last_top_k = None

    def search(
        self,
        query_vector,
        filters,
        *,
        top_k,
    ):
        self.last_vector = query_vector
        self.last_filters = filters
        self.last_top_k = top_k
        return self.results


class StubGenerator:
    model_name = "stub-generator"

    def __init__(self, answer: str):
        self.answer_text = answer
        self.last_prompt = None
        self.calls = 0

    def generate(self, prompt: str):
        self.last_prompt = prompt
        self.calls += 1
        return self.answer_text


def _result(
    *,
    section_id: str = "CH-3",
    section_title: str = "Severe Release",
    score: float = 0.91,
):
    return SimpleNamespace(
        chunk_id="8fe0dfbf-6e58-5817-a3e9-8fe48cd1b383",
        score=score,
        text=(
            "Only an authenticated finance manager may "
            "release an active severe credit hold."
        ),
        metadata={
            "doc_id": "credit_hold_policy",
            "title": (
                "Credit Hold Classification and "
                "Release Policy"
            ),
            "section_id": section_id,
            "section_title": section_title,
            "access_roles": [
                "finance_manager",
            ],
            "status": "active",
        },
    )


def test_naive_rag_retrieves_once_and_generates_once():
    embedder = StubEmbedder()
    store = StubVectorStore([_result()])
    generator = StubGenerator(
        "A finance manager may release it [1]."
    )

    response = NaiveRAG(
        embedder=embedder,
        vector_store=store,
        generator=generator,
    ).answer(
        "Who can release a severe credit hold?",
        role="finance_manager",
        top_k=3,
    )

    assert embedder.last_query == (
        "Who can release a severe credit hold?"
    )
    assert store.last_top_k == 3
    assert store.last_filters.role == "finance_manager"
    assert store.last_filters.statuses == ("active",)
    assert generator.calls == 1

    assert response.answer == (
        "A finance manager may release it [1]."
    )
    assert response.retrieved_count == 1
    assert response.sources[0].doc_id == (
        "credit_hold_policy"
    )
    assert response.sources[0].section_id == "CH-3"


def test_prompt_contains_grounding_rules_and_context():
    generator = StubGenerator("Grounded answer [1].")

    NaiveRAG(
        embedder=StubEmbedder(),
        vector_store=StubVectorStore([_result()]),
        generator=generator,
    ).answer(
        "Who can release the hold?",
        role="finance_manager",
    )

    prompt = generator.last_prompt

    assert "using ONLY the authorized context" in prompt
    assert "Do not use outside knowledge" in prompt
    assert "[1]" in prompt
    assert "CH-3" in prompt
    assert "Only an authenticated finance manager" in prompt
    assert "Who can release the hold?" in prompt


def test_no_results_returns_safe_answer_without_llm_call():
    generator = StubGenerator("This must not be used.")

    response = NaiveRAG(
        embedder=StubEmbedder(),
        vector_store=StubVectorStore([]),
        generator=generator,
    ).answer(
        "What is the private password?",
        role="sales_rep",
    )

    assert response.answer == NO_CONTEXT_ANSWER
    assert response.sources == ()
    assert response.retrieved_count == 0
    assert generator.calls == 0


def test_empty_query_is_rejected():
    rag = NaiveRAG(
        embedder=StubEmbedder(),
        vector_store=StubVectorStore([]),
        generator=StubGenerator("answer"),
    )

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        rag.answer(
            "   ",
            role="sales_rep",
        )


def test_invalid_role_is_rejected_by_metadata_schema():
    rag = NaiveRAG(
        embedder=StubEmbedder(),
        vector_store=StubVectorStore([]),
        generator=StubGenerator("answer"),
    )

    with pytest.raises(ValueError):
        rag.answer(
            "What is the policy?",
            role="administrator",
        )

def test_unsupported_numeric_claim_is_replaced_by_safe_answer():
    generator = StubGenerator(
        "A severe hold starts at 120 days overdue [1]."
    )

    response = NaiveRAG(
        embedder=StubEmbedder(),
        vector_store=StubVectorStore([_result()]),
        generator=generator,
    ).answer(
        "Who can release a severe credit hold?",
        role="finance_manager",
    )

    assert response.answer == NO_CONTEXT_ANSWER
    assert response.verification is not None
    assert not response.verification.passed

