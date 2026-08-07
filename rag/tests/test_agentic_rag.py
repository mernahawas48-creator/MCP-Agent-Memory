from __future__ import annotations

from types import SimpleNamespace

from rag.agentic_rag.controller import (
    AgenticRAG,
    EvidenceAssessment,
    SAFE_NO_EVIDENCE_ANSWER,
)


class StubRetriever:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append(
            {
                "query": query,
                **kwargs,
            }
        )

        if not self.responses:
            return []

        return self.responses.pop(0)


class StubGenerator:
    model_name = "stub-generator"

    def __init__(self, answer):
        self.answer_text = answer
        self.calls = 0
        self.prompt = ""

    def generate(self, prompt):
        self.calls += 1
        self.prompt = prompt
        return self.answer_text


class SequenceGrader:
    def __init__(self, assessments):
        self.assessments = list(assessments)
        self.calls = 0

    def grade(self, original_query, results):
        self.calls += 1
        return self.assessments.pop(0)


class StubRewriter:
    def __init__(self, rewritten):
        self.rewritten = rewritten
        self.calls = 0

    def rewrite(self, original_query, results):
        self.calls += 1
        return self.rewritten


def _result(
    *,
    chunk_id="chunk-ch3",
    doc_id="credit_hold_policy",
    section_id="CH-3",
    section_title="Severe Release",
):
    return SimpleNamespace(
        chunk_id=chunk_id,
        fused_score=0.032,
        dense_rank=1,
        sparse_rank=1,
        text=(
            "Only an authenticated finance manager may "
            "release an active severe credit hold."
        ),
        metadata={
            "doc_id": doc_id,
            "title": "Credit Hold Policy",
            "section_id": section_id,
            "section_title": section_title,
            "access_roles": [
                "finance_manager",
            ],
            "status": "active",
        },
    )


def test_agent_answers_after_one_sufficient_retrieval():
    retriever = StubRetriever(
        [[_result()]]
    )
    generator = StubGenerator(
        "A finance manager may release it [1]."
    )

    response = AgenticRAG(
        retriever=retriever,
        generator=generator,
        grader=SequenceGrader(
            [
                EvidenceAssessment(
                    sufficient=True,
                    reason="Evidence is sufficient.",
                    matched_terms=(
                        "credit",
                        "hold",
                    ),
                )
            ]
        ),
    ).answer(
        "Who can release a severe credit hold?",
        role="finance_manager",
        top_k=3,
    )

    assert response.attempts == 1
    assert response.sources[0].section_id == "CH-3"
    assert generator.calls == 1
    assert [
        step.action
        for step in response.trace
    ] == [
        "plan",
        "retrieve",
        "grade_evidence",
        "generate_answer",
    ]


def test_agent_rewrites_and_retries_weak_evidence():
    first_result = _result(
        chunk_id="weak",
        doc_id="shipment_pricing_reference",
        section_id="SP-2",
        section_title="Rate Exceptions",
    )
    final_result = _result(
        chunk_id="re2",
        doc_id="rate_exception_policy",
        section_id="RE-2",
        section_title="Above Authority Discount",
    )

    retriever = StubRetriever(
        [
            [first_result],
            [final_result],
        ]
    )
    rewriter = StubRewriter(
        "discount delegated authority approval RE-2"
    )

    response = AgenticRAG(
        retriever=retriever,
        generator=StubGenerator(
            "Finance approval is required [1]."
        ),
        grader=SequenceGrader(
            [
                EvidenceAssessment(
                    sufficient=False,
                    reason="Weak evidence.",
                ),
                EvidenceAssessment(
                    sufficient=True,
                    reason="Strong evidence.",
                    matched_terms=("discount",),
                ),
            ]
        ),
        rewriter=rewriter,
        max_attempts=2,
    ).answer(
        "What discount needs finance approval?",
        role="finance_manager",
        top_k=3,
    )

    assert response.attempts == 2
    assert len(retriever.calls) == 2
    assert retriever.calls[1]["query"] == (
        "discount delegated authority approval RE-2"
    )
    assert rewriter.calls == 1
    assert "rewrite_query" in {
        step.action
        for step in response.trace
    }


def test_exact_section_id_uses_exact_filter_and_sparse_priority():
    retriever = StubRetriever(
        [[
            _result(
                chunk_id="re2",
                doc_id="rate_exception_policy",
                section_id="RE-2",
                section_title=(
                    "Above Authority Discount"
                ),
            )
        ]]
    )

    response = AgenticRAG(
        retriever=retriever,
        generator=StubGenerator(
            "RE-2 requires finance approval [1]."
        ),
        grader=SequenceGrader(
            [
                EvidenceAssessment(
                    sufficient=True,
                    reason="Exact section matched.",
                    matched_terms=("RE-2",),
                )
            ]
        ),
    ).answer(
        "re-2",
        role="finance_manager",
        top_k=2,
    )

    call = retriever.calls[0]

    assert call["query"] == "RE-2"
    assert call["section_ids"] == ("RE-2",)
    assert call["dense_weight"] == 0.5
    assert call["sparse_weight"] == 1.5
    assert response.sources[0].section_id == "RE-2"


def test_agent_stops_safely_after_max_attempts():
    generator = StubGenerator(
        "This answer must not be generated."
    )

    response = AgenticRAG(
        retriever=StubRetriever(
            [[], []]
        ),
        generator=generator,
        grader=SequenceGrader(
            [
                EvidenceAssessment(
                    sufficient=False,
                    reason="No results.",
                ),
                EvidenceAssessment(
                    sufficient=False,
                    reason="Still no results.",
                ),
            ]
        ),
        rewriter=StubRewriter(
            "expanded unavailable query"
        ),
        max_attempts=2,
    ).answer(
        "What is the private password?",
        role="sales_rep",
    )

    assert response.answer == (
        SAFE_NO_EVIDENCE_ANSWER
    )
    assert response.attempts == 2
    assert generator.calls == 0
    assert response.trace[-1].action == "stop"


def test_role_and_metadata_filters_are_forwarded():
    retriever = StubRetriever(
        [[_result()]]
    )

    AgenticRAG(
        retriever=retriever,
        generator=StubGenerator(
            "Answer [1]."
        ),
        grader=SequenceGrader(
            [
                EvidenceAssessment(
                    sufficient=True,
                    reason="Enough.",
                )
            ]
        ),
    ).answer(
        "What is the finance policy?",
        role="finance_manager",
        departments=("finance",),
        document_types=("policy",),
        doc_ids=("credit_hold_policy",),
    )

    call = retriever.calls[0]

    assert call["role"] == "finance_manager"
    assert call["statuses"] == ("active",)
    assert call["departments"] == ("finance",)
    assert call["document_types"] == ("policy",)
    assert call["doc_ids"] == (
        "credit_hold_policy",
    )
