"""Run dense and hybrid retrieval evaluation from PowerShell."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag.evaluation.dataset import (
    DEFAULT_DATASET_PATH,
    load_evaluation_cases,
)
from rag.evaluation.evaluator import (
    RetrievalEvaluator,
    write_reports,
)
from rag.evaluation.retrievers import (
    DenseRetriever,
    HybridRetriever,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Swiftrail dense and hybrid retrieval."
        )
    )
    parser.add_argument(
        "--retriever",
        choices=["dense", "hybrid", "both"],
        default="both",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[1, 3, 5],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "results"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    cases = load_evaluation_cases(
        args.dataset
    )
    evaluator = RetrievalEvaluator(
        ks=args.k
    )

    retrievers = []

    if args.retriever in {
        "dense",
        "both",
    }:
        retrievers.append(
            DenseRetriever()
        )

    if args.retriever in {
        "hybrid",
        "both",
    }:
        retrievers.append(
            HybridRetriever()
        )

    reports = []

    for retriever in retrievers:
        print(
            f"Evaluating {retriever.name} "
            f"on {len(cases)} cases..."
        )

        report = evaluator.evaluate(
            retriever,
            cases,
        )
        reports.append(report)

        print(
            "  "
            f"Hit@1={report.metrics['hit_rate'][1]:.4f} "
            f"MRR@5={report.metrics['mrr'][5]:.4f} "
            "AccessSafety@5="
            f"{report.metrics['access_safety_rate'][5]:.4f}"
        )

    json_path, markdown_path = write_reports(
        reports,
        args.output_dir,
    )

    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
