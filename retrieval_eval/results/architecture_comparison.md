# Swiftrail Retrieval Architecture Comparison

Model: `gemini-3.5-flash-lite`
Fixed test cases: 10

| Architecture | Correct / Total | Accuracy | Avg. input tokens/query | Avg. output tokens/query | Avg. total tokens/query | Avg. latency/query | Avg. retrieval attempts | Safe abstentions | Transient API retries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive RAG | 7/10 | 70.0% | 285.8 | 41.0 | 326.8 | 0.610s | 1.00 | 4 | 0 |
| Hybrid RAG | 9/10 | 90.0% | 261.3 | 40.0 | 301.3 | 0.501s | 1.00 | 3 | 0 |
| Agentic RAG | 9/10 | 90.0% | 640.9 | 41.9 | 682.8 | 0.546s | 1.40 | 3 | 0 |

## Per-case results

| Architecture | Case | Category | Correct | Sources | Verification | Attempts | API retries | Latency | Reason |
|---|---|---|---|---|---|---:|---:|---:|---|
| Naive RAG | semantic-severe-release | naive_friendly | yes | CH-3, CH-1, CH-2 | pass | 1 | 0 | 2.335s | Passed the fixed answer rubric. |
| Naive RAG | semantic-hold-thresholds | naive_friendly | yes | CH-1, CH-2, PR-2 | pass | 1 | 0 | 0.690s | Passed the fixed answer rubric. |
| Naive RAG | semantic-invoice-followup | naive_friendly | yes | IC-2, IC-1, IC-3 | pass | 1 | 0 | 0.610s | Passed the fixed answer rubric. |
| Naive RAG | exact-re2 | hybrid_friendly | no | SP-2, RE-3, RE-1 | fail | 1 | 0 | 0.044s | The pipeline abstained on an answerable case. |
| Naive RAG | exact-ac4 | hybrid_friendly | yes | AC-2, SP-4, AC-4 | pass | 1 | 0 | 0.520s | Passed the fixed answer rubric. |
| Naive RAG | exact-sp3 | hybrid_friendly | yes | SP-3, CH-3, SP-4 | pass | 1 | 0 | 0.631s | Passed the fixed answer rubric. |
| Naive RAG | multi-part-discount-and-hold | agentic_friendly | no | RE-2, RE-4 | pass | 1 | 0 | 0.691s | Missing expected source section(s): CH-3 |
| Naive RAG | multi-part-pricing-and-hold | agentic_friendly | no | SP-4, RE-4 | fail | 1 | 0 | 0.536s | The pipeline abstained on an answerable case. |
| Naive RAG | unauthorized-pr2 | authorization | yes | CH-2, SP-2, SP-3 | fail | 1 | 0 | 0.020s | Safe abstention expected and returned. |
| Naive RAG | unsupported-storage-fee | safe_abstention | yes | SP-3, SP-2, SP-4 | fail | 1 | 0 | 0.023s | Safe abstention expected and returned. |
| Hybrid RAG | semantic-severe-release | naive_friendly | yes | CH-3, CH-1, CH-2 | pass | 1 | 0 | 1.106s | Passed the fixed answer rubric. |
| Hybrid RAG | semantic-hold-thresholds | naive_friendly | yes | CH-1, CH-2, CH-4 | pass | 1 | 0 | 0.552s | Passed the fixed answer rubric. |
| Hybrid RAG | semantic-invoice-followup | naive_friendly | yes | IC-2, IC-1, IC-3 | pass | 1 | 0 | 0.534s | Passed the fixed answer rubric. |
| Hybrid RAG | exact-re2 | hybrid_friendly | yes | RE-2 | pass | 1 | 0 | 0.501s | Passed the fixed answer rubric. |
| Hybrid RAG | exact-ac4 | hybrid_friendly | yes | AC-4 | pass | 1 | 0 | 0.540s | Passed the fixed answer rubric. |
| Hybrid RAG | exact-sp3 | hybrid_friendly | yes | SP-3 | pass | 1 | 0 | 0.554s | Passed the fixed answer rubric. |
| Hybrid RAG | multi-part-discount-and-hold | agentic_friendly | no | CH-1, RE-4 | fail | 1 | 0 | 0.590s | The pipeline abstained on an answerable case. |
| Hybrid RAG | multi-part-pricing-and-hold | agentic_friendly | yes | SP-4, CH-3 | pass | 1 | 0 | 0.608s | Passed the fixed answer rubric. |
| Hybrid RAG | unauthorized-pr2 | authorization | yes | - | fail | 1 | 0 | 0.009s | Safe abstention expected and returned. |
| Hybrid RAG | unsupported-storage-fee | safe_abstention | yes | SP-1, CH-1, IC-2 | fail | 1 | 0 | 0.014s | Safe abstention expected and returned. |
| Agentic RAG | semantic-severe-release | naive_friendly | yes | CH-3, CH-1, CH-2 | pass | 1 | 0 | 1.082s | Passed the fixed answer rubric. |
| Agentic RAG | semantic-hold-thresholds | naive_friendly | yes | CH-1, CH-2, CH-4 | pass | 1 | 0 | 0.605s | Passed the fixed answer rubric. |
| Agentic RAG | semantic-invoice-followup | naive_friendly | yes | IC-2, IC-1, IC-3 | pass | 1 | 0 | 0.612s | Passed the fixed answer rubric. |
| Agentic RAG | exact-re2 | hybrid_friendly | yes | RE-2 | pass | 1 | 0 | 0.474s | Passed the fixed answer rubric. |
| Agentic RAG | exact-ac4 | hybrid_friendly | yes | AC-4 | pass | 1 | 0 | 0.535s | Passed the fixed answer rubric. |
| Agentic RAG | exact-sp3 | hybrid_friendly | no | SP-3 | fail | 1 | 0 | 0.617s | The pipeline abstained on an answerable case. |
| Agentic RAG | multi-part-discount-and-hold | agentic_friendly | yes | CH-1, RE-4, RE-2, CH-3, AC-3, RE-1 | pass | 2 | 0 | 0.736s | Passed the fixed answer rubric. |
| Agentic RAG | multi-part-pricing-and-hold | agentic_friendly | yes | SP-4, CH-3, SP-1, SP-3, RE-4 | pass | 2 | 0 | 0.753s | Passed the fixed answer rubric. |
| Agentic RAG | unauthorized-pr2 | authorization | yes | - | fail | 2 | 0 | 0.018s | Safe abstention expected and returned. |
| Agentic RAG | unsupported-storage-fee | safe_abstention | yes | SP-1, CH-1, IC-2, SP-4, SP-3, SP-2 | fail | 2 | 0 | 0.028s | Safe abstention expected and returned. |
