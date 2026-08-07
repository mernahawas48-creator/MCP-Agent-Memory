# Swiftrail Retrieval Architecture Comparison

Model: `gemini-3.5-flash-lite`
Fixed test cases: 10

| Architecture | Correct / Total | Accuracy | Avg. input tokens/query | Avg. output tokens/query | Avg. total tokens/query | Avg. latency/query | Avg. retrieval attempts | Safe abstentions | Transient API retries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive RAG | 7/10 | 70.0% | 285.8 | 37.3 | 323.1 | 0.629s | 1.00 | 4 | 0 |
| Hybrid RAG | 9/10 | 90.0% | 261.3 | 40.7 | 302.0 | 0.634s | 1.00 | 3 | 0 |
| Agentic RAG | 8/10 | 80.0% | 478.4 | 32.6 | 511.0 | 0.652s | 1.40 | 3 | 0 |

## Per-case results

| Architecture | Case | Category | Correct | Sources | Verification | Attempts | API retries | Latency | Reason |
|---|---|---|---|---|---|---:|---:|---:|---|
| Naive RAG | semantic-severe-release | naive_friendly | yes | CH-3, CH-1, CH-2 | pass | 1 | 0 | 2.516s | Passed the fixed answer rubric. |
| Naive RAG | semantic-hold-thresholds | naive_friendly | yes | CH-1, CH-2, PR-2 | pass | 1 | 0 | 0.605s | Passed the fixed answer rubric. |
| Naive RAG | semantic-invoice-followup | naive_friendly | yes | IC-2, IC-1, IC-3 | pass | 1 | 0 | 0.605s | Passed the fixed answer rubric. |
| Naive RAG | exact-re2 | hybrid_friendly | no | SP-2, RE-3, RE-1 | fail | 1 | 0 | 0.010s | The pipeline abstained on an answerable case. |
| Naive RAG | exact-ac4 | hybrid_friendly | yes | AC-2, SP-4, AC-4 | pass | 1 | 0 | 0.598s | Passed the fixed answer rubric. |
| Naive RAG | exact-sp3 | hybrid_friendly | yes | SP-3, CH-3, SP-4 | pass | 1 | 0 | 0.552s | Passed the fixed answer rubric. |
| Naive RAG | multi-part-discount-and-hold | agentic_friendly | no | RE-2, RE-4 | pass | 1 | 0 | 0.821s | Missing expected source section(s): CH-3 |
| Naive RAG | multi-part-pricing-and-hold | agentic_friendly | no | SP-4, RE-4 | fail | 1 | 0 | 0.559s | The pipeline abstained on an answerable case. |
| Naive RAG | unauthorized-pr2 | authorization | yes | CH-2, SP-2, SP-3 | fail | 1 | 0 | 0.010s | Safe abstention expected and returned. |
| Naive RAG | unsupported-storage-fee | safe_abstention | yes | SP-3, SP-2, SP-4 | fail | 1 | 0 | 0.011s | Safe abstention expected and returned. |
| Hybrid RAG | semantic-severe-release | naive_friendly | yes | CH-3, CH-1, CH-2 | pass | 1 | 0 | 2.157s | Passed the fixed answer rubric. |
| Hybrid RAG | semantic-hold-thresholds | naive_friendly | yes | CH-1, CH-2, CH-4 | pass | 1 | 0 | 0.675s | Passed the fixed answer rubric. |
| Hybrid RAG | semantic-invoice-followup | naive_friendly | yes | IC-2, IC-1, IC-3 | pass | 1 | 0 | 0.604s | Passed the fixed answer rubric. |
| Hybrid RAG | exact-re2 | hybrid_friendly | yes | RE-2 | pass | 1 | 0 | 0.501s | Passed the fixed answer rubric. |
| Hybrid RAG | exact-ac4 | hybrid_friendly | yes | AC-4 | pass | 1 | 0 | 0.649s | Passed the fixed answer rubric. |
| Hybrid RAG | exact-sp3 | hybrid_friendly | yes | SP-3 | pass | 1 | 0 | 0.533s | Passed the fixed answer rubric. |
| Hybrid RAG | multi-part-discount-and-hold | agentic_friendly | no | CH-1, RE-4 | fail | 1 | 0 | 0.575s | The pipeline abstained on an answerable case. |
| Hybrid RAG | multi-part-pricing-and-hold | agentic_friendly | yes | SP-4, CH-3 | pass | 1 | 0 | 0.619s | Passed the fixed answer rubric. |
| Hybrid RAG | unauthorized-pr2 | authorization | yes | - | fail | 1 | 0 | 0.011s | Safe abstention expected and returned. |
| Hybrid RAG | unsupported-storage-fee | safe_abstention | yes | SP-1, CH-1, IC-2 | fail | 1 | 0 | 0.012s | Safe abstention expected and returned. |
| Agentic RAG | semantic-severe-release | naive_friendly | yes | CH-3, CH-1, CH-2 | pass | 1 | 0 | 1.120s | Passed the fixed answer rubric. |
| Agentic RAG | semantic-hold-thresholds | naive_friendly | yes | CH-1, CH-2, CH-4 | pass | 1 | 0 | 0.621s | Passed the fixed answer rubric. |
| Agentic RAG | semantic-invoice-followup | naive_friendly | yes | IC-2, IC-1, IC-3 | pass | 1 | 0 | 2.331s | Passed the fixed answer rubric. |
| Agentic RAG | exact-re2 | hybrid_friendly | yes | RE-2 | pass | 1 | 0 | 0.538s | Passed the fixed answer rubric. |
| Agentic RAG | exact-ac4 | hybrid_friendly | yes | AC-4 | pass | 1 | 0 | 0.537s | Passed the fixed answer rubric. |
| Agentic RAG | exact-sp3 | hybrid_friendly | yes | SP-3 | pass | 1 | 0 | 0.553s | Passed the fixed answer rubric. |
| Agentic RAG | multi-part-discount-and-hold | agentic_friendly | no | RE-4, RE-1, CH-1, RE-2, RE-3 | pass | 2 | 0 | 0.733s | Missing expected source section(s): CH-3 |
| Agentic RAG | multi-part-pricing-and-hold | agentic_friendly | no | SP-4, SP-2, CH-3, SP-1, CH-1 | fail | 2 | 0 | 0.036s | The pipeline abstained on an answerable case. |
| Agentic RAG | unauthorized-pr2 | authorization | yes | - | fail | 2 | 0 | 0.024s | Safe abstention expected and returned. |
| Agentic RAG | unsupported-storage-fee | safe_abstention | yes | SP-1, SP-4, CH-1, SP-3, SP-2 | fail | 2 | 0 | 0.028s | Safe abstention expected and returned. |
