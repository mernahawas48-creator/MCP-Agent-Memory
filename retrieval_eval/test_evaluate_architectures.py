from __future__ import annotations

from types import SimpleNamespace

from retrieval_eval.evaluate_architectures import (
    EvaluationCase,
    score_answer,
)


def _case(**overrides):
    values = {
        "case_id": "case",
        "category": "test",
        "query": "Who can release it?",
        "role": "finance_manager",
        "top_k": 3,
        "expected_section_ids": ("CH-3",),
        "required_term_groups": (
            ("finance manager",),
        ),
        "expected_abstain": False,
        "forbidden_section_ids": (),
    }
    values.update(overrides)
    return EvaluationCase(**values)


def test_score_answer_requires_expected_source_and_terms():
    correct, reason = score_answer(
        _case(),
        "Only a finance manager may release it [1].",
        ("CH-3",),
        True,
    )

    assert correct
    assert "Passed" in reason


def test_score_answer_rejects_missing_expected_source():
    correct, _ = score_answer(
        _case(),
        "Only a finance manager may release it [1].",
        ("CH-2",),
        True,
    )

    assert not correct


def test_expected_abstention_is_scored_as_correct():
    correct, _ = score_answer(
        _case(
            expected_section_ids=(),
            required_term_groups=(),
            expected_abstain=True,
        ),
        (
            "I could not find enough authorized information "
            "in the Swiftrail knowledge base to answer this question."
        ),
        (),
        True,
    )

    assert correct
