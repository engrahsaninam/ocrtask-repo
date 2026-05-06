import argparse
import json
import re
from statistics import mean


def _edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.findall(r"[^.!?]+[.!?]?", text) if s.strip()]


def cer(ground_truth: str, predicted: str) -> float:
    if not ground_truth:
        return 0.0 if not predicted else 1.0
    return _edit_distance(list(ground_truth), list(predicted)) / len(ground_truth)


def wer(ground_truth: str, predicted: str) -> float:
    gt = _tokens(ground_truth)
    pred = _tokens(predicted)
    if not gt:
        return 0.0 if not pred else 1.0
    return _edit_distance(gt, pred) / len(gt)


def sentence_f1(ground_truth: str, predicted: str) -> float:
    gt_sentences = _sentences(ground_truth)
    pred_sentences = _sentences(predicted)
    if not gt_sentences and not pred_sentences:
        return 1.0
    if not gt_sentences or not pred_sentences:
        return 0.0

    matched = 0
    used = set()
    for gt in gt_sentences:
        gt_tokens = set(_tokens(gt))
        for idx, pred in enumerate(pred_sentences):
            if idx in used:
                continue
            pred_tokens = set(_tokens(pred))
            denom = max(len(gt_tokens), 1)
            if len(gt_tokens & pred_tokens) / denom >= 0.5:
                matched += 1
                used.add(idx)
                break

    precision = matched / len(pred_sentences)
    recall = matched / len(gt_sentences)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def composite_score(cer_value: float, wer_value: float, sentence_f1_value: float) -> float:
    score = 0.4 * (1 - cer_value) + 0.3 * (1 - wer_value) + 0.3 * sentence_f1_value
    return round(max(0.0, min(1.0, score)), 4)


def evaluate_samples(samples: list[dict]) -> tuple[list[dict], dict]:
    rows = []
    for sample in samples:
        cer_value = cer(sample["ground_truth"], sample["predicted"])
        wer_value = wer(sample["ground_truth"], sample["predicted"])
        f1_value = sentence_f1(sample["ground_truth"], sample["predicted"])
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "cer": cer_value,
                "wer": wer_value,
                "sentence_f1": f1_value,
                "composite": composite_score(cer_value, wer_value, f1_value),
            }
        )
    aggregate = {
        "samples": len(rows),
        "cer": mean(row["cer"] for row in rows) if rows else 0.0,
        "wer": mean(row["wer"] for row in rows) if rows else 0.0,
        "sentence_f1": mean(row["sentence_f1"] for row in rows) if rows else 0.0,
        "composite": mean(row["composite"] for row in rows) if rows else 0.0,
    }
    return rows, aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    args = parser.parse_args()
    with open(args.json_file, "r", encoding="utf-8") as fh:
        samples = json.load(fh)
    rows, aggregate = evaluate_samples(samples)
    print("sample_id\tcer\twer\tsentence_f1\tcomposite")
    for row in rows:
        print(f"{row['sample_id']}\t{row['cer']:.4f}\t{row['wer']:.4f}\t{row['sentence_f1']:.4f}\t{row['composite']:.4f}")
    print(f"AGGREGATE\t{aggregate['cer']:.4f}\t{aggregate['wer']:.4f}\t{aggregate['sentence_f1']:.4f}\t{aggregate['composite']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
