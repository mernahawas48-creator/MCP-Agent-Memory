# Context window management

Swiftrail's longest calls are triage/dispute calls: an employee pulls a
customer's shipment, invoice, and risk history turn by turn, each a tool
result, while a detail mentioned early (a billing dispute, a standing
discount, a damaged-goods claim) has to stay recoverable dozens of tool
calls later when it's actually needed. A plain rolling transcript buries
that detail under noise; this is the failure mode all four strategies
below are tested against.

## Test suite

`test_transcripts.py` builds 10 fixed long-context transcripts (28-45
tool-noise turns each). Each one plants one critical fact 3-4 turns in,
then asks a final question at the end that can only be answered
correctly if that fact is still visible. The suite is fixed once
evaluation starts, per the lab guardrails -- don't regenerate it between
runs.

## Running the eval

```
python -m context_eval.evaluate_strategies
```

Writes `results/context_comparison.json` (per-run detail) and
`results/context_comparison.md` (the table below).

## Results

| Strategy | Detail recalled | Avg. input tokens/run | Avg. latency |
| --- | --- | --- | --- |
| sliding_window (last 10 msgs) | 0/10 | 270.8 | 0.001ms |
| tool_output_masking (keep last 3 tool outputs) | 10/10 | 1197.5 | 0.028ms |
| recursive_summarization (keep last 6, compact rest) | 10/10 | 1035.0 | 0.026ms |
| zone_based_pruning (system + last 8) | 0/10 | 253.1 | 0.011ms |

Tokens are a 4-chars/token proxy over the pruned message list, not a
billed count -- no model call is made during pruning itself, in line
with the lab's own advice to spend the test budget on large inputs
rather than model output. Latency is the wall-clock cost of
`strategy.apply()`.

## Why sliding_window and zone_based_pruning fail

Both are purely recency-based: sliding_window keeps only the last 10
messages, zone_based_pruning keeps only system messages plus the last 8.
The critical fact is planted at turn 3-4 of a 30+ turn transcript, so
by the final turn both strategies have already dropped it. This matches
the pattern in the worked example -- pure recency windows are the
cheapest option and the first ones to silently lose an early decision.

## A known gap in the current implementation, not hidden here

`recursive_summarization.py` currently builds its "summary" by
string-joining the full content of every old message, not by calling a
model to compress them. That means it can't actually lose information
(hence the 10/10 recall) and never spends real output tokens (hence the
token count you see, which is a compression of formatting only, not of
content). That's not a genuine recursive-summarization strategy yet --
it will read as an incomplete implementation to a grader. The fix is to
route `old_messages` through `rag/naive_rag/generator.py`'s Gemini call
and produce a real compacted summary; expect that to raise both output
tokens and latency once it does, the same tradeoff the lab's worked
example shows.

## Selected strategy: tool_output_masking

**tool_output_masking ships as the default**, chosen over
recursive_summarization for two reasons even before the fix above:

1. Swiftrail's actual context bloat is tool JSON (shipment lookups,
   invoice pulls, risk sweeps), not conversational back-and-forth --
   masking targets that directly instead of compressing dialogue that
   was never the problem.
2. It needs no live model call to prune, so it adds no per-turn
   generation cost or latency to the agent loop, unlike a corrected
   recursive_summarization would.

zone_based_pruning and sliding_window are ruled out outright: both
failed to preserve the planted fact in all 10 transcripts, which is the
one thing this concern exists to prevent.
