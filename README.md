# Swiftrail MCP Agent with Memory and RAG

## Problem Statement

Enterprise logistics agents operate across long, multi-step conversations involving customers, shipments, invoices, credit holds, and rate exceptions. As these interactions grow, the agent may lose important context, repeat previous work, retrieve irrelevant information, or rely on outdated facts.

A basic conversation history is not sufficient for this environment. The system must distinguish between temporary working context, important past events, and stable long-term knowledge. It must also prevent unverified, conflicting, or expired information from being stored and reused in future decisions.

This project extends the Swiftrail Logistics MCP agent with a structured Memory and Retrieval-Augmented Generation architecture. The new system introduces short-term memory, episodic and semantic memory, context-management strategies, metadata-aware vector retrieval, and evidence verification.

The objective is to build an agent that can preserve relevant context across interactions, retrieve accurate domain knowledge efficiently, and avoid using unsupported or stale information in operational and financial workflows.


## RAG Architecture Evaluation

Three RAG architectures were evaluated on the same fixed set of 10
Swiftrail domain questions using `gemini-3.5-flash-lite`.

The evaluation set included:

- semantic policy questions,
- exact policy section identifiers,
- multi-part questions requiring evidence from multiple sections,
- an authorization-sensitive query,
- and an unsupported query requiring safe abstention.

The comparison measured answer accuracy, Gemini token usage,
end-to-end latency, retrieval attempts, and safe abstention behavior.

### Retrieval-Level Evaluation

Before comparing the complete RAG architectures, dense and hybrid
retrieval were evaluated independently on 28 fixed retrieval cases.

| Retrieval Method | Hit@1 | MRR@5 | Access Safety@5 |
|---|---:|---:|---:|
| Dense Retrieval | 92.31% | 94.36% | 100% |
| Hybrid Retrieval | 100% | 100% | 100% |

Hybrid retrieval achieved perfect Hit@1 and MRR@5 on the evaluation
set while preserving full role-aware access safety.

### End-to-End Architecture Comparison

| Architecture | Correct / Total | Accuracy | Avg. Input Tokens / Query | Avg. Output Tokens / Query | Avg. Total Tokens / Query | Avg. Latency / Query | Avg. Retrieval Attempts |
|---|---:|---:|---:|---:|---:|---:|---:|
| Naive RAG | 7/10 | 70.0% | 285.8 | 37.3 | 323.1 | 0.629s | 1.00 |
| Hybrid RAG | 9/10 | 90.0% | 261.3 | 40.7 | 302.0 | 0.634s | 1.00 |
| Agentic RAG | 8/10 | 80.0% | 478.4 | 32.6 | 511.0 | 0.652s | 1.40 |

### Result Analysis

Naive RAG performed well on straightforward semantic questions but
was less reliable for exact policy identifiers and multi-section
questions. For example, it failed the `RE-2` exact-section case because
dense retrieval returned related sections instead of the requested
section.

Hybrid RAG produced the strongest overall result. It correctly answered
9 of the 10 evaluation cases and successfully handled all exact policy
identifier queries. Combining dense semantic retrieval with BM25 lexical
retrieval improved exact-section matching without introducing additional
retrieval rounds.

Agentic RAG achieved 80% accuracy. Although it could rewrite queries and
perform additional retrieval attempts, the extra reasoning did not
improve the final accuracy over Hybrid RAG on this benchmark. It also
used substantially more tokens per query and averaged 1.40 retrieval
attempts.

All architectures correctly handled the authorization-sensitive and
unsupported-information cases by returning safe abstentions instead of
fabricating unauthorized or unavailable information.

### Selected Architecture

**Hybrid RAG is selected as the final retrieval architecture for
Swiftrail.**

The decision is based on measured evaluation results rather than
architectural complexity alone. Hybrid RAG achieved:

- the highest answer accuracy: **90%**,
- the lowest average total token usage: **302 tokens/query**,
- latency comparable to Naive RAG,
- one retrieval attempt per query on average,
- perfect exact-section retrieval in the tested identifier cases,
- and correct safe-abstention behavior.

Agentic RAG remains valuable as an experimental architecture for queries
that may require iterative retrieval. However, on the current Swiftrail
evaluation set, its additional retrieval and reasoning cost did not
produce better accuracy than Hybrid RAG.

### Self-RAG Verification

The RAG pipelines include an explicit verification layer after retrieval
and after generation.

The verification process checks:

1. whether retrieved evidence is relevant to the user's query,
2. whether generated claims are supported by retrieved evidence,
3. whether citations reference valid retrieved chunks,
4. whether numeric claims are grounded in the evidence.

If verification fails, the pipeline does not expose an unsupported
answer. It returns a safe abstention instead.

This behavior was also exercised by the evaluation cases for unauthorized
and unsupported information.

### Evaluation Limitations

The architecture comparison uses a small fixed benchmark of 10
domain-specific questions. Therefore, the measured percentages should be
interpreted as results for the current Swiftrail corpus and test set,
rather than as general performance guarantees.

Latency is also affected by external model-service variability. The
comparison therefore emphasizes the combined evidence from accuracy,
retrieval behavior, token usage, and latency rather than latency alone.
