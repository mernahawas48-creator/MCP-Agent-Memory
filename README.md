# Swiftrail MCP Agent with Memory and RAG

## Problem Statement

Enterprise logistics agents operate across long, multi-step conversations involving customers, shipments, invoices, credit holds, and rate exceptions. As these interactions grow, the agent may lose important context, repeat previous work, retrieve irrelevant information, or rely on outdated facts.

A basic conversation history is not sufficient for this environment. The system must distinguish between temporary working context, important past events, and stable long-term knowledge. It must also prevent unverified, conflicting, or expired information from being stored and reused in future decisions.

This project extends the Swiftrail Logistics MCP agent with a structured Memory and Retrieval-Augmented Generation architecture. The new system introduces short-term memory, episodic and semantic memory, context-management strategies, metadata-aware vector retrieval, and evidence verification.

The objective is to build an agent that can preserve relevant context across interactions, retrieve accurate domain knowledge efficiently, and avoid using unsupported or stale information in operational and financial workflows.
