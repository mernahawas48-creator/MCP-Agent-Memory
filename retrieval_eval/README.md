# Retrieval Architecture Evaluation

This folder produces the comparison table required by the project rubric.

It is intentionally separate from `rag/evaluation/`:

- `rag/evaluation/` measures retrieval ranking quality (`Hit@K`, `Recall@K`,
  `MRR`, and access safety).
- `retrieval_eval/` runs the complete Naive RAG, Hybrid RAG, and Agentic RAG
  answer pipelines against the same fixed questions and measures answer
  accuracy, API-reported token usage, and end-to-end latency.

## Fixed test set

`questions.json` contains the final answer-level cases. It includes:

- straightforward semantic questions;
- exact policy identifiers intended to favor hybrid retrieval;
- multi-part questions intended to exercise the agentic retry path;
- an authorization case;
- an unsupported question that should safely abstain.

Do not change the file between architecture runs. Changing the test set after
seeing architecture results invalidates the comparison.

## Run

Start Qdrant and rebuild the knowledge collection:

```powershell
docker compose -f rag\vector_store\docker-compose.yml up -d
python -m rag.ingestion.pipeline --recreate
```

Make sure `.env` contains the working Gemini key and model, then run:

```powershell
python -m retrieval_eval.evaluate_architectures
```

Outputs:

```text
retrieval_eval/results/architecture_comparison.json
retrieval_eval/results/architecture_comparison.md
```

The Markdown report contains the README-ready table. The JSON report preserves
the per-question answers, sources, verification result, token usage, latency,
and retrieval-attempt count.

Token counts come from Gemini's response `usage_metadata`; they are not
estimated from character counts.
