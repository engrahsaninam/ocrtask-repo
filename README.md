# Generative AI Engineer Assessment

Complete submission for Sections 1-4.

## Structure

- `section1/`: OCR preprocessing, extraction, segmentation, and metrics.
- `section2/`: classifier prompts, orchestrator, and error analysis.
- `section3/`: architecture, migration, and fidelity write-ups.
- `section4/`: bonus SFS metric and hybrid fallback design.
- `tests/`: runnable tests for code deliverables.

## Run

```powershell
python -m pip install -r requirements.txt
python run_all.py
```

API keys are not committed. Use environment variables and `.env.example`.

`run_all.py` runs the test suite and writes sample output JSON files under `outputs/`.

Full OCR run artifacts are included under `outputs/task_submission_ahsan_inam/`.
