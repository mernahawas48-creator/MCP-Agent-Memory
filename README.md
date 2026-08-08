# Swiftrail MCP Agent with Memory and RAG

## Problem Statement

Enterprise logistics agents operate across long, multi-step conversations involving customers, shipments, invoices, credit holds, and rate exceptions. As these interactions grow, the agent may lose important context, repeat previous work, retrieve irrelevant information, or rely on outdated facts.

A basic conversation history is not sufficient for this environment. The system must distinguish between temporary working context, important past events, and stable long-term knowledge. It must also prevent unverified, conflicting, or expired information from being stored and reused in future decisions.

This project extends the Swiftrail Logistics MCP agent with a structured Memory and Retrieval-Augmented Generation architecture. The new system introduces short-term memory, episodic and semantic memory, context-management strategies, metadata-aware vector retrieval, and evidence verification.

The objective is to build an agent that can preserve relevant context across interactions, retrieve accurate domain knowledge efficiently, and avoid using unsupported or stale information in operational and financial workflows.


## Memory System

### Problem

Swiftrail employees (sales reps and finance managers) work the same
customers across many separate sessions. Two things go wrong without a
memory layer:

- **Nothing persists across sessions.** A sales rep who checked on a
  customer yesterday has to re-explain the situation to the agent
  today -- the agent has no way to recall that a credit hold was
  placed, or that a rate exception was already rejected for a specific
  reason.
- **Nothing reconciles conflicting history.** A customer's standing
  changes over time (a hold gets released, then a new one gets placed
  months later). Without an explicit place these facts live and get
  updated, the agent either repeats stale information or has no
  record at all of which version is current.

Both failures are costly in this domain: approving a shipment for a
customer who is actually back on a severe credit hold, or re-approving
a rate exception that was already rejected for cause, are real
operational mistakes, not cosmetic ones.

### Architecture

```
ShortTermBuffer  --overflow-->  PromoteDropRouter --(episodic only)--> EpisodicMemory
Scratchpad (separate, survives pruning)                                     |
                                                              ConsolidationLayer (periodic)
                                                                              |
                                                                      SemanticMemory
```

| Component | Role | Swiftrail example |
|---|---|---|
| `ShortTermBuffer` | Rolling window of recent conversation turns | Last N turns of a triage call |
| `Scratchpad` | Current goal / sub-goal / working state, isolated from the buffer so pruning never destroys it | "Review open rate exceptions before approving shipment 512" survives even after the tool-call chatter that gathers each exception gets pruned |
| `PromoteDropRouter` | Decides forget vs. episodic for each turn evicted from short-term memory, with a logged reason. Never writes to semantic memory. | A credit hold placement is promoted; "good morning" is forgotten |
| `EpisodicMemory` | Durable, queryable-by-customer store of promoted events | "Credit hold placed on customer 12, severe, 90+ days overdue" |
| `ConsolidationLayer` | Separate, periodic pass over episodic memory that derives/updates semantic facts (never triggered inline by the router) | Turns repeated credit-hold episodes into a current `customer_risk_level` fact |
| `SemanticMemory` | Versioned, expiring facts with explicit conflict resolution | See conflict example below |

### A real conflict, resolved

1. Customer 12's credit hold is released → consolidation writes
   `customer_risk_level = good_standing` (version 1, active).
2. Weeks later, a new severe credit hold lands on the same customer
   (90+ days overdue) → the next consolidation pass detects that this
   contradicts the active fact.
3. Resolution: version 1 is marked `superseded` (not deleted) and
   points forward to version 2; version 2 (`high_risk`) becomes
   active, with `conflict_reason = "Superseded version 1
   ('good_standing' -> 'high_risk') based on episode 7."`
4. `fact_history()` still returns both versions, so the full timeline
   of the customer's risk status is auditable, not overwritten.

This is exercised end-to-end in `memory/demo_memory.py` and asserted in
`memory/test_consolidation.py::test_consolidation_resolves_a_real_conflict_across_two_runs`.

Facts also expire on a TTL if nothing reaffirms them (`expire_stale_facts`),
so a semantic fact that stops being reinforced by new episodes ages out
rather than staying authoritative forever.

See `memory/README.md` for exactly where each concern lives in the code.


## MCP Server

The operational layer is implemented with FastMCP and MySQL.

| MCP Concern | Implementation |
|---|---|
| Capability negotiation | The client reads the server's declared capabilities and gates protocol operations on them |
| Notifications | Authentication can change the exposed tool set and triggers `tools/list_changed` |
| Elicitation | Above-authority discounts and severe credit-hold releases require explicit human input |
| Sampling | `run_portfolio_risk_sweep` can request a narrative summary from the connected client model |
| Resources | Credit/discount authority policy is exposed as an MCP resource |
| Prompts | Parameterized rate-exception justification prompt |
| Transport | stdio for local development and Streamable HTTP for remote execution |
| Progress tracking | Portfolio risk sweep reports progress while customers are processed |
| Defensive tools | Strict Pydantic schemas, server-side validation, role/state checks, safe failures, and re-authorization before writes |

Main tools:

- `authenticate`
- `search_customer`
- `get_shipment_status`
- `list_customer_invoices`
- `approve_rate_exception`
- `release_credit_hold`
- `run_portfolio_risk_sweep`
- `list_portfolio_credit_exposure` for authorized finance-manager sessions

The MySQL schema and fixed seed data are under `db/`.

## Context Management

Four context strategies are implemented under `context_eval/strategies/`:

| Strategy | Purpose |
|---|---|
| Sliding Window | Keep the most recent messages |
| Recursive Summarization | Compress older context into a summary |
| Tool Output Masking | Mask older tool outputs while preserving the conversation |
| Zone-Based Pruning | Preserve system/important zones and recent context |

The current live `AgentLoop` uses Sliding Window. The other strategies remain independently testable.

## Agent Integration

`agent/agent_loop.py` routes requests between the selected RAG path, verified memory recall, and the existing operational path.

- policy, authority, guideline, and exact section-ID questions -> **Hybrid RAG**
- cross-session recall questions -> **verified episodic/semantic memory**
- shipment, invoice, customer, and credit operations -> **MCP operational path**
- short-term overflow -> **Promote/Drop routing into episodic memory**

The real MCP protocol lifecycle and tool execution remain in `agent/client.py`; the agent loop does not duplicate the server or database.


## RAG System

### Knowledge Corpus and Ingestion

The corpus contains six Swiftrail policy/reference documents covering credit holds, rate exceptions, portfolio risk, invoice collection, employee access, and shipment pricing. These documents contain 22 policy sections used as retrieval units.

```text
Documents
  -> validated loading
  -> section-aware chunking
  -> metadata validation
  -> embeddings
  -> Qdrant indexing
```

| Component | Configuration |
|---|---|
| Chunking | Section-aware, max 1000 characters, 120-character overlap |
| Embeddings | `BAAI/bge-small-en-v1.5` with FastEmbed |
| Vector size | 384 |
| Vector database | Qdrant |
| Similarity | Cosine |
| ANN index | HNSW |
| Metadata filtering | Role, status, document, and section metadata applied during retrieval |

### Retrieval Architectures

**Naive RAG**  
Dense query embedding -> Qdrant retrieval -> grounded generation.

**Hybrid RAG**  
Dense vector retrieval + BM25 lexical retrieval -> Reciprocal Rank Fusion (RRF). Exact identifiers such as `RE-2` are handled explicitly.

**Agentic RAG**  
Plan -> retrieve -> grade evidence -> rewrite/retrieve again when evidence is incomplete -> accumulate evidence -> grounded generation. The controller is capped at two retrieval attempts.

### Self-RAG Verification

Both RAG and memory recall use explicit verification:

1. check retrieved evidence for relevance;
2. generate only from the retrieved evidence;
3. validate citations and factual support;
4. check numeric claims against evidence or scenario values;
5. return a safe abstention when verification fails.

This prevents unsupported or unauthorized information from being returned as a confident answer.

### Retrieval-Level Evaluation

Dense and hybrid retrieval were evaluated on 28 fixed retrieval cases.

| Retrieval Method | Hit@1 | MRR@5 | Access Safety@5 |
|---|---:|---:|---:|
| Dense Retrieval | 92.31% | 94.36% | 100% |
| Hybrid Retrieval | 100% | 100% | 100% |

### End-to-End Architecture Comparison

Naive, Hybrid, and Agentic RAG were evaluated on the same fixed 10 Swiftrail questions using `gemini-3.5-flash-lite`.

| Architecture | Correct / Total | Accuracy | Avg. Input Tokens | Avg. Output Tokens | Avg. Total Tokens | Avg. Latency | Avg. Retrieval Attempts |
|---|---:|---:|---:|---:|---:|---:|---:|
| Naive RAG | 7/10 | 70.0% | 285.8 | 41.0 | 326.8 | 0.610s | 1.00 |
| Hybrid RAG | 9/10 | 90.0% | 261.3 | 40.0 | 301.3 | 0.501s | 1.00 |
| Agentic RAG | 9/10 | 90.0% | 640.9 | 41.9 | 682.8 | 0.546s | 1.40 |

The fixed set covers semantic questions, exact policy identifiers, multi-section questions, an authorization-sensitive case, and an unsupported-information case.

### Selected Architecture

**Hybrid RAG is the final retrieval architecture used by the live agent.**

Hybrid and Agentic RAG both reached **90% accuracy**, but Hybrid used fewer tokens, lower average latency, and one retrieval attempt per query. This matches Swiftrail's common mix of semantic policy questions and exact policy identifiers without paying the extra cost of iterative retrieval on every request.

Agentic RAG is retained for complex multi-part cases. In the discount-and-severe-hold case, the first retrieval missed `CH-3`; Agentic RAG detected the missing policy facet, rewrote the query, performed a second retrieval, accumulated `RE-2`, `RE-4`, and `CH-3`, and answered successfully. Hybrid RAG did not solve that case.

Detailed results:

```text
retrieval_eval/results/architecture_comparison.json
retrieval_eval/results/architecture_comparison.md
```

---

## Project Structure

```text
agent/              Agent loop, routing, MCP client, sessions
context_eval/       Context-management strategies and tests
db/                 MySQL schema, seed data, ERD
demo/               Captured MCP and RAG/Self-RAG demo evidence
mcp_server/         FastMCP server, schemas, tools, resources, prompts
memory/             Short-term, episodic, semantic memory and verified recall
rag/                Corpus, ingestion, vector store, RAG architectures, verification
retrieval_eval/     Fixed end-to-end architecture evaluation
```

## Setup

### Install dependencies

```powershell
pip install -r mcp_server\requirements.txt
pip install -r agent\requirements.txt
pip install -r rag\embeddings\requirements.txt
pip install -r rag\vector_store\requirements.txt
pip install pytest
```

### Configure MySQL

Create a local database named `swiftrail_db`, then run:

```text
db/schema.sql
db/seed.sql
```

Copy `mcp_server/.env.example` to `mcp_server/.env` and set the local database credentials.

### Configure Gemini

Create a root `.env` file:

```env
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

Do not commit real credentials.

### Start Qdrant and ingest the corpus

```powershell
docker compose -f rag\vector_store\docker-compose.yml up -d
python -m rag.ingestion.pipeline --recreate
```

## Running

MCP server with stdio:

```powershell
python mcp_server\server.py
```

Streamable HTTP:

```powershell
python mcp_server\server.py --http
```

MCP demo:

```powershell
python agent\demo.py --transport stdio
```

Memory demo:

```powershell
python -m memory.demo_memory
```

RAG examples:

```powershell
python -m rag.naive_rag.cli --query "Who can release a severe credit hold?" --role finance_manager
python -m rag.hybrid_rag.cli --query "RE-2" --role finance_manager
python -m rag.agentic_rag.cli --query "An 18 percent discount is requested for a customer with a severe credit hold. Who must approve the discount, who may release the hold, and does discount approval release the hold?" --role finance_manager --top-k 5 --max-attempts 2
```

Architecture evaluation:

```powershell
python -m retrieval_eval.evaluate_architectures
```

Do not edit `retrieval_eval/questions.json` between architecture runs.

## Tests

```powershell
python -m pytest memory -q
python -m pytest context_eval -q
python -m pytest rag\tests -q
python -m pytest retrieval_eval\test_evaluate_architectures.py -q
python -m pytest agent -q
```

Integration tests that depend on MySQL or Qdrant require those services to be running.

## Demo Evidence

The captured protocol demo and RAG/Self-RAG evidence are documented in:

```text
demo/demo_transcript.md
```

