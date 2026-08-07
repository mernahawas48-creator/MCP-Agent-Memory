from __future__ import annotations

from types import SimpleNamespace

import pytest

from retrieval_eval.evaluate_architectures import (
    EvaluationCase,
    _answer_with_transient_retry,
    _is_transient_503,
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

class _Transient503(Exception):
    status_code = 503


class _RetryPipeline:
    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def answer(self, query, *, role, top_k):
        self.calls += 1

        if self.calls <= self.failures:
            raise _Transient503(
                "503 UNAVAILABLE"
            )

        return SimpleNamespace(
            answer="grounded answer",
            sources=(),
        )


class _UsageGenerator:
    def reset_usage(self):
        return None


def test_transient_503_is_detected():
    assert _is_transient_503(
        _Transient503("503 UNAVAILABLE")
    )
    assert not _is_transient_503(
        RuntimeError("400 BAD REQUEST")
    )


def test_case_retries_transient_503_and_then_succeeds(
    monkeypatch,
):
    monkeypatch.setattr(
        "retrieval_eval.evaluate_architectures.time.sleep",
        lambda _: None,
    )

    pipeline = _RetryPipeline(failures=2)

    response, latency, retries, wait = (
        _answer_with_transient_retry(
            pipeline=pipeline,
            generator=_UsageGenerator(),
            case=_case(),
            max_api_retries=3,
            initial_retry_delay=1.0,
        )
    )

    assert response.answer == "grounded answer"
    assert pipeline.calls == 3
    assert retries == 2
    assert wait == 3.0
    assert latency >= 0.0


def test_non_503_error_is_not_hidden():
    class BrokenPipeline:
        def answer(self, query, *, role, top_k):
            raise ValueError("broken")

    with pytest.raises(
        ValueError,
        match="broken",
    ):
        _answer_with_transient_retry(
            pipeline=BrokenPipeline(),
            generator=_UsageGenerator(),
            case=_case(),
            max_api_retries=3,
            initial_retry_delay=0.0,
        )

