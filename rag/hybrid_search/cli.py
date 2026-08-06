"""Command-line interface for Swiftrail hybrid retrieval."""

from __future__ import annotations

import argparse

from rag.hybrid_search.search import HybridSearch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search Swiftrail using dense and BM25 retrieval."
        )
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Question or keyword query.",
    )
    parser.add_argument(
        "--role",
        required=True,
        choices=[
            "sales_rep",
            "finance_manager",
        ],
        help="Authenticated role used for filtering.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--sparse-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--section",
        action="append",
        dest="section_ids",
        help=(
            "Optional section ID filter. Repeat for "
            "multiple section IDs."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    results = HybridSearch().search(
        args.query,
        role=args.role,
        top_k=args.top_k,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
        section_ids=args.section_ids,
    )

    print(f"Query: {args.query}")
    print(f"Results: {len(results)}")

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n[{index}] "
            f"{result.metadata['doc_id']} | "
            f"{result.metadata['section_id']}"
        )
        print(
            f"fusion={result.fused_score:.6f} | "
            f"dense_rank={result.dense_rank} | "
            f"sparse_rank={result.sparse_rank}"
        )
        print(result.text)


if __name__ == "__main__":
    main()
