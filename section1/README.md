# Section 1

## Task 1.1

- Ruling-line suppression: removes notebook guides.
- Contrast: strengthens faint strokes.
- Median filtering: removes specks.
- Deskewing: straightens baselines.
- Adaptive thresholding: handles uneven lighting.
- Line-run removal: removes long straight borders/rules.
- Stroke normalization: light median cleanup only.

Excluded: full line segmentation; ruled handwriting can over-split.

## Task 1.2

OCR engine: Tesseract via `pytesseract`; traditional OCR, word confidence scores, easy to audit. Weak irregular-handwriting regions are flagged.

## Task 1.3

Composite = 40% CER similarity + 30% WER similarity + 30% sentence F1.

Missing piece: add non-standard-form preservation for misspellings, invented words, casing, and fragments.
