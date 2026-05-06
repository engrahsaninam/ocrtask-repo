# Task 3.2 - Model Deprecation

Benchmark candidates on frozen labelled OCR data. Choose by Incomplete recall, no baseline regression, JSON validity, latency, cost, and 429 rate.

Rollout: shadow, then canary. Migrate Classifier first, Embedded second unless benchmarks clearly support both together.

Minimum tests: schema, all labels, non-standard text, batch order, missing fallback, 429 retry, no-embedded, embedded recovery.
