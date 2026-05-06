# Section 2

## Task 2.1

Prompts are in `prompts/`. Misclassification risks: non-standard verb forms, weak conjunctions, misspelled subordinators, missing punctuation, and fragments that look complete.

Embedded examples: `becaus I go Home and` -> embedded `I go Home` -> final `Simple`; `because near the old room and` -> no embedded sentence -> final `Incomplete`.

## Task 2.2

Hand-rolled orchestrator; classifier batches first, embedded only on `Incomplete`.

Includes 429 retry, missing-item fallback, deterministic routing, and `agent_path`.

## Task 2.3

Main error: Compound/Complex -> Simple, 18 cases. Add stricter Simple check: subject + finite predicate + no unresolved subordinator.

Add `embedded_recovered_label`. Production threshold: Incomplete precision >= 0.90, recall >= 0.85.
