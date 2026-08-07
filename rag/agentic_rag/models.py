"""Data models used by the Agentic RAG controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    """One retrieval decision produced by the planner."""

    query: str
    top_k: int
    candidate_k: int
    dense_weight: float
    sparse_weight: float
    section_ids: tuple[str, ...] | None
    reason: str


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """Decision describing whether retrieved evidence is sufficient."""

    sufficient: bool
    reason: str
    matched_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentTraceStep:
    """One visible decision or action taken by the agent."""

    step: int
    action: str
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgenticRAGSource:
    """Source attached to the final answer."""

    number: int
    chunk_id: str
    doc_id: str
    title: str
    section_id: str
    section_title: str
    fused_score: float
    dense_rank: int | None
    sparse_rank: int | None


@dataclass(frozen=True, slots=True)
class AgenticRAGAnswer:
    """Final answer together with sources and the agent trace."""

    query: str
    answer: str
    sources: tuple[AgenticRAGSource, ...]
    attempts: int
    final_retrieval_query: str
    model_name: str
    trace: tuple[AgentTraceStep, ...]
