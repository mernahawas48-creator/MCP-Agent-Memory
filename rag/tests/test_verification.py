from __future__ import annotations

from types import SimpleNamespace

from rag.verification.verifier import (
    SelfRAGVerifier,
)


def _result(
    text: str,
    section_id: str = "CH-3",
):
    return SimpleNamespace(
        text=text,
        metadata={
            "title": "Credit Hold Policy",
            "section_id": section_id,
            "section_title": "Severe Release",
            "keywords": [],
        },
    )


def test_relevance_passes_for_matching_policy_evidence():
    check = SelfRAGVerifier().check_relevance(
        "Who can release a severe credit hold?",
        [
            _result(
                "Only an authenticated finance manager may "
                "release an active severe credit hold."
            )
        ],
    )

    assert check.passed


def test_exact_id_relevance_requires_exact_section():
    check = SelfRAGVerifier().check_relevance(
        "RE-2",
        [
            _result(
                "Another section mentions RE-2.",
                section_id="SP-2",
            )
        ],
    )

    assert not check.passed


def test_support_passes_for_cited_grounded_claim():
    check = SelfRAGVerifier().check_support(
        (
            "Only an authenticated finance manager may release "
            "an active severe credit hold [1]."
        ),
        [
            _result(
                "Only an authenticated finance manager may "
                "release an active severe credit hold."
            )
        ],
    )

    assert check.passed


def test_support_rejects_uncited_claim():
    check = SelfRAGVerifier().check_support(
        "A finance manager may release the hold.",
        [
            _result(
                "A finance manager may release the hold."
            )
        ],
    )

    assert not check.passed


def test_support_rejects_fabricated_numeric_threshold():
    check = SelfRAGVerifier().check_support(
        "A discount above 20 percent needs finance approval [1].",
        [
            _result(
                "A discount above 15 percent requires "
                "an authenticated finance manager.",
                section_id="RE-2",
            )
        ],
    )

    assert not check.passed
