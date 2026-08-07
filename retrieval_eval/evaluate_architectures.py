"""Run the same fixed questions through Naive, Hybrid, and Agentic RAG."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Sequence

from rag.agentic_rag.controller import AgenticRAG
from rag.hybrid_rag.pipeline import HybridRAG
from rag.naive_rag.generator import GenerationUsage
from rag.naive_rag.pipeline import NaiveRAG


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = (
    Path(__file__).resolve().parent / "questions.json"
)
DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parent / "results"
)
SAFE_PREFIX = "i could not find enough authorized"


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    category: str
    query: str
    role: str
    top_k: int
    expected_section_ids: tuple[str, ...]
    required_term_groups: tuple[tuple[str, ...], ...]
    expected_abstain: bool
    forbidden_section_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CaseResult:
    architecture: str
    case_id: str
    category: str
    correct: bool
    query: str
    role: str
    answer: str
    source_section_ids: tuple[str, ...]
    verification_passed: bool
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_seconds: float
    retrieval_attempts: int
    transient_api_retries: int
    retry_wait_seconds: float
    reason: str


@dataclass(frozen=True, slots=True)
class ArchitectureSummary:
    architecture: str
    correct: int
    total: int
    accuracy: float
    avg_input_tokens_per_query: float
    avg_output_tokens_per_query: float
    avg_total_tokens_per_query: float
    avg_latency_seconds_per_query: float
    safe_abstentions: int
    avg_retrieval_attempts: float
    total_transient_api_retries: int


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    model_name: str
    case_count: int
    summaries: tuple[ArchitectureSummary, ...]
    cases: tuple[CaseResult, ...]


def load_cases(
    path: str | Path = DEFAULT_QUESTIONS,
) -> tuple[EvaluationCase, ...]:
    payload = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    if not isinstance(payload, list):
        raise ValueError(
            "The architecture evaluation dataset must be a JSON list."
        )

    cases = tuple(
        EvaluationCase(
            case_id=str(item["case_id"]),
            category=str(item["category"]),
            query=str(item["query"]).strip(),
            role=str(item["role"]),
            top_k=int(item.get("top_k", 3)),
            expected_section_ids=tuple(
                str(value)
                for value in item.get(
                    "expected_section_ids",
                    [],
                )
            ),
            required_term_groups=tuple(
                tuple(str(value) for value in group)
                for group in item.get(
                    "required_term_groups",
                    [],
                )
            ),
            expected_abstain=bool(
                item.get("expected_abstain", False)
            ),
            forbidden_section_ids=tuple(
                str(value)
                for value in item.get(
                    "forbidden_section_ids",
                    [],
                )
            ),
        )
        for item in payload
    )

    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(
            "Architecture evaluation case IDs must be unique."
        )

    return cases


def build_architectures() -> tuple[tuple[str, Any], ...]:
    """Construct the three required complete answer pipelines."""

    return (
        ("Naive RAG", NaiveRAG()),
        ("Hybrid RAG", HybridRAG()),
        ("Agentic RAG", AgenticRAG(max_attempts=2)),
    )


def run_comparison(
    *,
    cases: Sequence[EvaluationCase],
    architectures: Sequence[tuple[str, Any]],
    max_api_retries: int = 5,
    initial_retry_delay: float = 5.0,
    inter_case_delay: float = 1.5,
) -> ComparisonReport:
    """Run every architecture against every fixed test question.

    Transient Gemini 503 errors are retried without changing the fixed
    questions or the architecture configuration. Retry waiting time is
    recorded separately and is not included in the per-query latency metric.
    """

    if max_api_retries < 0:
        raise ValueError("max_api_retries cannot be negative.")

    if initial_retry_delay < 0:
        raise ValueError("initial_retry_delay cannot be negative.")

    if inter_case_delay < 0:
        raise ValueError("inter_case_delay cannot be negative.")

    all_results: list[CaseResult] = []
    model_name = "unknown"

    for architecture_name, pipeline in architectures:
        _warm_embedding(pipeline)

        generator = getattr(
            pipeline,
            "generator",
            None,
        )
        model_name = str(
            getattr(
                generator,
                "model_name",
                model_name,
            )
        )

        for case_index, case in enumerate(cases, start=1):
            print(
                f"[{architecture_name}] "
                f"{case_index}/{len(cases)} "
                f"{case.case_id}",
                flush=True,
            )

            (
                response,
                latency,
                transient_api_retries,
                retry_wait_seconds,
            ) = _answer_with_transient_retry(
                pipeline=pipeline,
                generator=generator,
                case=case,
                max_api_retries=max_api_retries,
                initial_retry_delay=initial_retry_delay,
            )

            usage = _usage(generator)
            source_sections = tuple(
                str(source.section_id)
                for source in response.sources
            )
            verification_passed = (
                _verification_passed(response)
            )
            correct, reason = score_answer(
                case,
                response.answer,
                source_sections,
                verification_passed,
            )

            all_results.append(
                CaseResult(
                    architecture=architecture_name,
                    case_id=case.case_id,
                    category=case.category,
                    correct=correct,
                    query=case.query,
                    role=case.role,
                    answer=response.answer,
                    source_section_ids=source_sections,
                    verification_passed=(
                        verification_passed
                    ),
                    input_tokens=(
                        usage.input_tokens
                    ),
                    output_tokens=(
                        usage.output_tokens
                    ),
                    total_tokens=(
                        usage.total_tokens
                    ),
                    latency_seconds=latency,
                    retrieval_attempts=int(
                        getattr(
                            response,
                            "attempts",
                            1,
                        )
                    ),
                    transient_api_retries=(
                        transient_api_retries
                    ),
                    retry_wait_seconds=(
                        retry_wait_seconds
                    ),
                    reason=reason,
                )
            )

            if (
                inter_case_delay > 0
                and case_index < len(cases)
            ):
                time.sleep(inter_case_delay)

    summaries = tuple(
        _summarize(
            architecture_name,
            [
                item
                for item in all_results
                if item.architecture
                == architecture_name
            ],
        )
        for architecture_name, _ in architectures
    )

    return ComparisonReport(
        model_name=model_name,
        case_count=len(cases),
        summaries=summaries,
        cases=tuple(all_results),
    )



def _answer_with_transient_retry(
    *,
    pipeline: Any,
    generator: Any,
    case: EvaluationCase,
    max_api_retries: int,
    initial_retry_delay: float,
) -> tuple[Any, float, int, float]:
    """Execute one case and retry only transient Gemini 503 failures.

    The returned latency measures the successful pipeline attempt. Deliberate
    retry waiting is tracked separately so service congestion does not inflate
    the architecture latency comparison.
    """

    retry_count = 0
    total_wait = 0.0

    while True:
        _reset_usage(generator)
        started = time.perf_counter()

        try:
            response = pipeline.answer(
                case.query,
                role=case.role,
                top_k=case.top_k,
            )
            latency = time.perf_counter() - started

            return (
                response,
                latency,
                retry_count,
                total_wait,
            )
        except Exception as exc:
            if not _is_transient_503(exc):
                raise

            if retry_count >= max_api_retries:
                raise RuntimeError(
                    "Gemini remained unavailable after "
                    f"{max_api_retries} transient retries "
                    f"for case '{case.case_id}'. "
                    "The fixed evaluation dataset was not changed."
                ) from exc

            delay = initial_retry_delay * (2 ** retry_count)
            retry_count += 1
            total_wait += delay

            print(
                "  Gemini returned transient 503 UNAVAILABLE. "
                f"Retry {retry_count}/{max_api_retries} "
                f"in {delay:.1f}s...",
                flush=True,
            )

            if delay > 0:
                time.sleep(delay)


def _is_transient_503(exc: Exception) -> bool:
    """Recognize Gemini service-unavailable errors without hiding other errors."""

    status_code = getattr(
        exc,
        "status_code",
        None,
    )

    if status_code == 503:
        return True

    code = getattr(
        exc,
        "code",
        None,
    )

    if code == 503:
        return True

    message = str(exc).upper()

    return (
        "503" in message
        and "UNAVAILABLE" in message
    )


def score_answer(
    case: EvaluationCase,
    answer: str,
    source_sections: Sequence[str],
    verification_passed: bool,
) -> tuple[bool, str]:
    """Apply the fixed answer rubric for one test case."""

    normalized_answer = _normalize(answer)
    source_set = set(source_sections)

    leaked_sections = (
        set(case.forbidden_section_ids)
        .intersection(source_set)
    )
    if leaked_sections:
        return (
            False,
            "Forbidden source section retrieved: "
            + ", ".join(sorted(leaked_sections)),
        )

    abstained = normalized_answer.startswith(
        SAFE_PREFIX
    )

    if case.expected_abstain:
        if abstained:
            return True, "Safe abstention expected and returned."
        return False, "The case required a safe abstention."

    if abstained:
        return False, "The pipeline abstained on an answerable case."

    if not verification_passed:
        return False, "The final answer failed verification."

    missing_sections = (
        set(case.expected_section_ids)
        .difference(source_set)
    )
    if missing_sections:
        return (
            False,
            "Missing expected source section(s): "
            + ", ".join(sorted(missing_sections)),
        )

    for group in case.required_term_groups:
        if not any(
            _normalize(option)
            in normalized_answer
            for option in group
        ):
            return (
                False,
                "Missing required answer evidence: "
                + " OR ".join(group),
            )

    return True, "Passed the fixed answer rubric."


def write_report(
    report: ComparisonReport,
    output_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        directory
        / "architecture_comparison.json"
    )
    markdown_path = (
        directory
        / "architecture_comparison.md"
    )

    json_path.write_text(
        json.dumps(
            asdict(report),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    markdown_path.write_text(
        _markdown(report),
        encoding="utf-8",
    )

    return json_path, markdown_path


def _summarize(
    architecture_name: str,
    results: Sequence[CaseResult],
) -> ArchitectureSummary:
    total = len(results)
    correct = sum(
        1 for item in results
        if item.correct
    )

    return ArchitectureSummary(
        architecture=architecture_name,
        correct=correct,
        total=total,
        accuracy=(
            correct / total
            if total
            else 0.0
        ),
        avg_input_tokens_per_query=_mean(
            [
                item.input_tokens
                for item in results
            ]
        ),
        avg_output_tokens_per_query=_mean(
            [
                item.output_tokens
                for item in results
            ]
        ),
        avg_total_tokens_per_query=_mean(
            [
                item.total_tokens
                for item in results
            ]
        ),
        avg_latency_seconds_per_query=_mean(
            [
                item.latency_seconds
                for item in results
            ]
        ),
        safe_abstentions=sum(
            1
            for item in results
            if _normalize(item.answer).startswith(
                SAFE_PREFIX
            )
        ),
        avg_retrieval_attempts=_mean(
            [
                item.retrieval_attempts
                for item in results
            ]
        ),
        total_transient_api_retries=sum(
            item.transient_api_retries
            for item in results
        ),
    )


def _verification_passed(
    response: Any,
) -> bool:
    if hasattr(
        response,
        "verification_passed",
    ):
        return bool(
            response.verification_passed
        )

    verification = getattr(
        response,
        "verification",
        None,
    )

    if verification is None:
        return False

    return bool(
        getattr(
            verification,
            "passed",
            False,
        )
    )


def _reset_usage(generator: Any) -> None:
    reset = getattr(
        generator,
        "reset_usage",
        None,
    )
    if callable(reset):
        reset()


def _usage(generator: Any) -> GenerationUsage:
    usage = getattr(
        generator,
        "usage_totals",
        None,
    )

    if isinstance(usage, GenerationUsage):
        return usage

    return GenerationUsage()


def _warm_embedding(pipeline: Any) -> None:
    """Load the embedding model before latency measurement without calling Gemini."""

    candidates = [
        getattr(pipeline, "embedder", None),
        getattr(
            getattr(pipeline, "searcher", None),
            "embedder",
            None,
        ),
        getattr(
            getattr(pipeline, "retriever", None),
            "embedder",
            None,
        ),
    ]

    for embedder in candidates:
        if embedder is not None:
            embedder.embed_query(
                "Swiftrail retrieval evaluation warmup"
            )
            return


def _normalize(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.lower().replace("%", " percent "),
    ).strip()


def _mean(values: Sequence[int | float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _markdown(
    report: ComparisonReport,
) -> str:
    lines = [
        "# Swiftrail Retrieval Architecture Comparison",
        "",
        f"Model: `{report.model_name}`",
        f"Fixed test cases: {report.case_count}",
        "",
        "| Architecture | Correct / Total | Accuracy | Avg. input tokens/query | Avg. output tokens/query | Avg. total tokens/query | Avg. latency/query | Avg. retrieval attempts | Safe abstentions | Transient API retries |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for summary in report.summaries:
        lines.append(
            "| "
            f"{summary.architecture} | "
            f"{summary.correct}/{summary.total} | "
            f"{summary.accuracy:.1%} | "
            f"{summary.avg_input_tokens_per_query:.1f} | "
            f"{summary.avg_output_tokens_per_query:.1f} | "
            f"{summary.avg_total_tokens_per_query:.1f} | "
            f"{summary.avg_latency_seconds_per_query:.3f}s | "
            f"{summary.avg_retrieval_attempts:.2f} | "
            f"{summary.safe_abstentions} | "
            f"{summary.total_transient_api_retries} |"
        )

    lines.extend(
        [
            "",
            "## Per-case results",
            "",
            "| Architecture | Case | Category | Correct | Sources | Verification | Attempts | API retries | Latency | Reason |",
            "|---|---|---|---|---|---|---:|---:|---:|---|",
        ]
    )

    for item in report.cases:
        lines.append(
            "| "
            f"{item.architecture} | "
            f"{item.case_id} | "
            f"{item.category} | "
            f"{'yes' if item.correct else 'no'} | "
            f"{', '.join(item.source_section_ids) or '-'} | "
            f"{'pass' if item.verification_passed else 'fail'} | "
            f"{item.retrieval_attempts} | "
            f"{item.transient_api_retries} | "
            f"{item.latency_seconds:.3f}s | "
            f"{item.reason.replace('|', '/')} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the three required Swiftrail RAG architectures "
            "on the same fixed answer-level test set."
        )
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--max-api-retries",
        type=int,
        default=5,
        help=(
            "Retries for transient Gemini 503 UNAVAILABLE errors "
            "per evaluation case."
        ),
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=5.0,
        help=(
            "Initial retry delay in seconds. Each retry doubles "
            "the previous delay."
        ),
    )
    parser.add_argument(
        "--inter-case-delay",
        type=float,
        default=1.5,
        help=(
            "Pause between fixed cases to reduce burst pressure. "
            "This pause is excluded from latency metrics."
        ),
    )
    args = parser.parse_args()

    cases = load_cases(args.questions)

    print(
        f"Running {len(cases)} fixed cases "
        "through Naive, Hybrid, and Agentic RAG..."
    )
    print(
        "This command calls Gemini for answerable cases. "
        "Do not edit questions.json between architecture runs."
    )

    report = run_comparison(
        cases=cases,
        architectures=build_architectures(),
        max_api_retries=args.max_api_retries,
        initial_retry_delay=args.retry_delay,
        inter_case_delay=args.inter_case_delay,
    )
    json_path, markdown_path = write_report(
        report,
        args.output_dir,
    )

    print("\nFinal comparison:")
    for summary in report.summaries:
        print(
            f"- {summary.architecture}: "
            f"{summary.correct}/{summary.total} "
            f"({summary.accuracy:.1%}), "
            f"avg tokens={summary.avg_total_tokens_per_query:.1f}, "
            f"avg latency={summary.avg_latency_seconds_per_query:.3f}s, "
            f"avg attempts={summary.avg_retrieval_attempts:.2f}, "
            f"503 retries={summary.total_transient_api_retries}"
        )

    print(f"\nJSON: {json_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
