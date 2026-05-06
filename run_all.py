import json
import subprocess
import sys
from pathlib import Path

from section1.eval import evaluate_samples
from section4.sfs import sfs


OUTPUT_DIR = Path("outputs")


def write_outputs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    samples = [
        {
            "sample_id": "sample_clean",
            "ground_truth": "I woke up early. The light was bright.",
            "predicted": "I woke up early. The light was bright.",
        },
        {
            "sample_id": "sample_verbatim",
            "ground_truth": "becaus I go Home and",
            "predicted": "because I go Home and",
        },
    ]
    rows, aggregate = evaluate_samples(samples)
    (OUTPUT_DIR / "evaluation_samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "evaluation_summary.json").write_text(
        json.dumps({"rows": rows, "aggregate": aggregate}, indent=2),
        encoding="utf-8",
    )

    sfs_rows = [
        {
            "sample_id": "source_correction",
            "ground_truth": "I siad I goed home.",
            "predicted": "I said I went home.",
            "sfs": sfs("I siad I goed home.", "I said I went home."),
        }
    ]
    (OUTPUT_DIR / "sfs_summary.json").write_text(json.dumps(sfs_rows, indent=2), encoding="utf-8")


def main() -> int:
    test_status = subprocess.call([sys.executable, "-m", "pytest", "-q"])
    if test_status == 0:
        write_outputs()
        print("Wrote outputs/evaluation_samples.json")
        print("Wrote outputs/evaluation_summary.json")
        print("Wrote outputs/sfs_summary.json")
    return test_status


if __name__ == "__main__":
    raise SystemExit(main())
