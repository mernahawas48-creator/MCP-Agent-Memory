"""Command-line interface for the Swiftrail Agentic RAG controller."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from rag.agentic_rag.controller import AgenticRAG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ask the Swiftrail Agentic RAG system."
        )
    )
    parser.add_argument(
        "--query",
        required=True,
    )
    parser.add_argument(
        "--role",
        required=True,
        choices=[
            "sales_rep",
            "finance_manager",
        ],
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    response = AgenticRAG(
        max_attempts=args.max_attempts
    ).answer(
        args.query,
        role=args.role,
        top_k=args.top_k,
    )

    print("\nAnswer:")
    print(response.answer)

    print("\nSources:")
    if not response.sources:
        print("- No sources.")
    else:
        for source in response.sources:
            print(
                f"[{source.number}] "
                f"{source.doc_id} | "
                f"{source.section_id} | "
                f"fusion={source.fused_score:.6f}"
            )

    print("\nAgent trace:")
    for step in response.trace:
        print(
            f"{step.step}. {step.action}: "
            f"{step.details}"
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
