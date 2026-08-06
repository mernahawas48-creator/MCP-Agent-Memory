"""Command-line interface for asking the Naive RAG system."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from rag.naive_rag.pipeline import NaiveRAG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ask the Swiftrail Naive RAG knowledge base."
        )
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Question to answer.",
    )
    parser.add_argument(
        "--role",
        required=True,
        choices=[
            "sales_rep",
            "finance_manager",
        ],
        help="Authenticated role used for metadata filtering.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of chunks to retrieve.",
    )
    parser.add_argument(
        "--department",
        action="append",
        dest="departments",
        help=(
            "Optional department filter. Repeat the option "
            "to include multiple departments."
        ),
    )
    parser.add_argument(
        "--section",
        action="append",
        dest="section_ids",
        help=(
            "Optional section ID filter. Repeat the option "
            "to include multiple sections."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    response = NaiveRAG().answer(
        args.query,
        role=args.role,
        top_k=args.top_k,
        departments=args.departments,
        section_ids=args.section_ids,
    )

    print("\nAnswer:")
    print(response.answer)

    print("\nSources:")
    if not response.sources:
        print("- No authorized sources retrieved.")
    else:
        for source in response.sources:
            print(
                f"[{source.number}] "
                f"{source.doc_id} | "
                f"{source.section_id} | "
                f"score={source.score:.4f}"
            )

    print("\nResponse data:")
    print(
        json.dumps(
            asdict(response),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
