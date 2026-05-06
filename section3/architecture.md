# Task 3.1 - Architecture

OCR ends after sentence segmentation; classification starts at the Classification Queue.

Flow: Upload API -> storage/job DB -> OCR queue -> preprocessing/OCR -> sentence records -> classification queue -> two-agent classifier -> results DB -> retrieval API.

Failures: OCR degradation -> confidence/canary/SFS alerts -> alternate preprocessing or review. Agent/API failures -> retry, fallback, dead-letter replay.

Monitoring: low-confidence ratio, blank rate, normalization drift, SFS, segmentation drift.
