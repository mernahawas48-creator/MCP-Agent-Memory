"""Run real hybrid searches against the populated Qdrant collection."""

from __future__ import annotations

from rag.hybrid_search.search import HybridSearch


def main() -> None:
    searcher = HybridSearch()

    semantic_results = searcher.search(
        "Who can release a severe credit hold?",
        role="finance_manager",
        top_k=3,
    )

    identifier_results = searcher.search(
        "RE-2",
        role="finance_manager",
        top_k=3,
        dense_weight=0.5,
        sparse_weight=1.5,
    )

    print("Semantic query:")
    for result in semantic_results:
        print(
            f"- {result.metadata['doc_id']} | "
            f"{result.metadata['section_id']} | "
            f"fusion={result.fused_score:.6f}"
        )

    print("\nExact identifier query:")
    for result in identifier_results:
        print(
            f"- {result.metadata['doc_id']} | "
            f"{result.metadata['section_id']} | "
            f"fusion={result.fused_score:.6f}"
        )

    assert semantic_results
    assert semantic_results[0].metadata[
        "section_id"
    ] == "CH-3"

    assert identifier_results
    assert identifier_results[0].metadata[
        "section_id"
    ] == "RE-2"

    print("\nHybrid search smoke test passed.")


if __name__ == "__main__":
    main()
