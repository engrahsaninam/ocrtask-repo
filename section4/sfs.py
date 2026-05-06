import re


_COMMON_CORRECTIONS = {
    "becaus": "because",
    "siad": "said",
    "wokeup": "woke up",
    "driv": "drive",
    "brite": "bright",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text)


def _is_nonstandard(token: str) -> bool:
    if token in _COMMON_CORRECTIONS:
        return True
    if any(ch.isupper() for ch in token[1:]):
        return True
    if re.search(r"(.)\1\1", token.lower()):
        return True
    return False


def sfs(ground_truth: str, predicted: str) -> float:
    if not ground_truth:
        return 1.0 if not predicted else 0.0

    gt_tokens = _tokens(ground_truth)
    pred_tokens = _tokens(predicted)
    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0

    pred_set = set(pred_tokens)
    weights = []
    preserved = []
    for token in gt_tokens:
        weight = 3 if _is_nonstandard(token) else 1
        weights.append(weight)
        preserved.append(weight if token in pred_set else 0)

    return round(sum(preserved) / sum(weights), 4)
