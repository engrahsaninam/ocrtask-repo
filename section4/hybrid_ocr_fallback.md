# Task 4.2 - Hybrid OCR Fallback

Threshold: low region confidence, low-confidence token ratio, blank rate, validation SFS/CER.

Merge: traditional OCR stays primary; fallback only for flagged regions; uncertainty/source-form loss goes to review.

Guardrails: no page rewrite, no spelling cleanup, preserve high-confidence non-standard tokens.

Evaluation: CER, WER, sentence F1, SFS, false-normalization rate, review burden.
