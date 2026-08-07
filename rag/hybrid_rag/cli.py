"""CLI for the complete Hybrid RAG answer pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from rag.hybrid_rag.pipeline import HybridRAG


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--role",
        required=True,
        choices=["sales_rep", "finance_manager"],
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    response = HybridRAG().answer(
        args.query,
        role=args.role,
        top_k=args.top_k,
    )

    print("\nAnswer:")
    print(response.answer)

    print("\nSources:")
    for source in response.sources:
        print(
            f"[{source.number}] {source.doc_id} | "
            f"{source.section_id} | "
            f"fusion={source.fused_score:.6f}"
        )

    print("\nVerification:")
    print(
        json.dumps(
            asdict(response.verification)
            if response.verification is not None
            else None,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
